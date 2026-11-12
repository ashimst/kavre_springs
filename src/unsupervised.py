"""
Unsupervised data mining of the spring inventory (no drying labels are used to build any structure).

U1  Landscape typology. The continuous GIS / remote-sensing covariates of env_hydro (LANDSCAPE_COLS: terrain,
    catchment hydrology, climate, water balance, vegetation, roads; survey-recorded fields and coordinates excluded)
    are standardised and reduced by PCA
    (components explaining 90 % of variance); Gaussian mixtures with K = 2..12 are compared by BIC and by bootstrap
    stability (mean adjusted Rand index between mixtures fitted to 80 % subsamples and the full-data mixture).
    Drying is then compared between types (rates with Wilson CIs, chi-square), and type and ward are compared with
    likelihood-ratio tests and adjusted McFadden pseudo-R^2.
U2  Regionalisation. Label-free contiguous landscape regions: spatially constrained Ward clustering of the PCA
    scores on the k-NN graph, with as many regions as there are wards (75) and as 2 km^2 DEM catchments (176).
    For each partition (and for wards, DEM catchments and municipalities) we report the share of landscape
    variance it explains (landscape homogeneity), the ICC of drying against 200 randomly shifted grids with the
    same number of units (drying homogeneity), and agreement with wards (ARI, NMI).
U3  Recalled attributes of dried sources. Ward-level structure of the reported causes and of dating to 2072 BS is
    compared with matched grids and with landscape types.
U4  Ward funnel. Observed vs landscape-expected dried sources per ward, funnel limits and overdispersion.

Outputs: results/metrics/un_*.csv and data/processed/landscape_types.csv (figures: src/figures.py)
Usage:   python -m src.unsupervised
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))  # silences a joblib core-count probe on Windows

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.spatial import cKDTree
from scipy.special import xlogy
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.catchment_analysis import grid_labels, icc1, matched_grid_null
from src.features import DYNAMIC_FEATURES, EXPLANATORY_SET, FEATURE_SETS
from src.dataset import coords_to_metric, load_master

logger = logging.getLogger("unsupervised")
METRICS = PROJECT_ROOT / "results/metrics"
FIGS = PROJECT_ROOT / "results/figures"
HG = PROJECT_ROOT / "data/processed/hydro_gis_features.csv"
SEED = 2024
K_RANGE = range(2, 13)
ENV_COLS = [c for c in FEATURE_SETS[EXPLANATORY_SET] if c not in ("latitude", "longitude")]
# Landscape covariates for the unsupervised analyses: continuous GIS / remote-sensing layers only. Survey-recorded
# fields (source type, land use, ownership) are excluded so that landscape units are independent of the survey teams,
# as are binary indicators (Gaussian mixtures assume continuous data) and the survey-derived spring density.
SURVEY_FIELDS = {"Source Type_encoded", "spring_density_500m", "road_upslope", "upa"}  # raw upa duplicated by log_upa
LANDSCAPE_COLS = [c for c in ENV_COLS if c not in SURVEY_FIELDS and c not in DYNAMIC_FEATURES
                  and not c.startswith(("landuse_", "ownership_"))]
LOG_TRANSFORM = {"dist_to_road_m", "pop_1km", "nightlights", "built_2020", "soil_soc"}  # heavy right tails


# ----------------------------------------------------------------------------- helpers
def env_matrix(df, cols=None):
    X = df[cols or LANDSCAPE_COLS].astype(float).copy()
    for c in LOG_TRANSFORM & set(X.columns):
        X[c] = np.log1p(X[c].clip(lower=0))
    return StandardScaler().fit_transform(X.fillna(X.median()).to_numpy())


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def loglik(X, y):
    """Maximised log-likelihood of an (effectively unpenalised) logistic regression; X=None -> intercept only."""
    if X is None or X.shape[1] == 0:
        p = np.full(len(y), y.mean())
    else:
        p = LogisticRegression(C=1e6, max_iter=20000).fit(X, y).predict_proba(X)[:, 1]
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.sum(xlogy(y, p) + xlogy(1 - y, 1 - p)))


def dummies(labels):
    return pd.get_dummies(pd.Series(labels).astype(str), drop_first=True, dtype=float).to_numpy()


def adj_pseudo_r2(ll, ll0, k):
    return 1 - (ll - k) / ll0


def cramers_v(a, b):
    t = pd.crosstab(a, b).to_numpy()
    chi2 = stats.chi2_contingency(t, correction=False)[0]
    n = t.sum()
    r, c = t.shape
    # bias-corrected (Bergsma 2013)
    phi2 = max(0.0, chi2 / n - (r - 1) * (c - 1) / (n - 1))
    rc, cc = r - (r - 1) ** 2 / (n - 1), c - (c - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(min(rc - 1, cc - 1), 1e-12)))


def knn_graph(coords, k=7):
    _, idx = cKDTree(coords).query(coords, k=k + 1)
    n = len(coords)
    rows = np.repeat(np.arange(n), k)
    A = sp.csr_matrix((np.ones(n * k), (rows, idx[:, 1:].ravel())), shape=(n, n))
    return ((A + A.T) > 0).astype(float).tocsr()


# ----------------------------------------------------------------------------- U1 typology
def typology(Z, n_boot=30):
    pca = PCA(random_state=SEED).fit(Z)
    n_pc = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.90) + 1)
    P = pca.transform(Z)[:, :n_pc]
    rng = np.random.default_rng(SEED)
    rows, models = [], {}
    for K in K_RANGE:
        gm = GaussianMixture(K, covariance_type="full", n_init=5, random_state=SEED).fit(P)
        lab = gm.predict(P)
        aris = []
        for b in range(n_boot):
            idx = rng.choice(len(P), int(0.8 * len(P)), replace=False)
            gb = GaussianMixture(K, covariance_type="full", n_init=2, random_state=SEED + b).fit(P[idx])
            aris.append(adjusted_rand_score(lab, gb.predict(P)))
        rows.append({"K": K, "bic": gm.bic(P), "stability_ari_mean": np.mean(aris), "stability_ari_sd": np.std(aris),
                     "min_type_share": np.bincount(lab, minlength=K).min() / len(lab)})
        models[K] = gm
    sel = pd.DataFrame(rows)
    # rule (fixed before looking at drying): among solutions with mean bootstrap ARI >= 0.75 and no type < 2 % of
    # sources, take the one with the lowest BIC
    ok = sel[(sel.stability_ari_mean >= 0.75) & (sel.min_type_share >= 0.02)]
    K_star = int((ok if len(ok) else sel).sort_values("bic").K.iloc[0])
    sel["selected"] = sel.K == K_star
    lab = models[K_star].predict(P)
    # order types by first-PC mean for readable labelling
    order = np.argsort([P[lab == k, 0].mean() for k in range(K_star)])
    remap = {old: new for new, old in enumerate(order)}
    lab = np.array([remap[v] for v in lab])
    loadings = pd.DataFrame(pca.components_[:n_pc].T, index=LANDSCAPE_COLS, columns=[f"PC{i + 1}" for i in range(n_pc)])
    evr = pca.explained_variance_ratio_[:n_pc]
    return lab, sel, loadings, evr, P


def typology_vs_drying(df, lab, Z, ward):
    y = df.target_dried.to_numpy()
    rows = []
    for k in np.unique(lab):
        m = lab == k
        lo, hi = wilson(y[m].sum(), m.sum())
        rows.append({"type": f"T{k + 1}", "n_sources": int(m.sum()), "share_ponds": float((df["Source Type_encoded"][m] == 1).mean())
                     if "Source Type_encoded" in df else np.nan,
                     "n_wards_present": int(pd.Series(ward[m]).nunique()), "drying_rate": y[m].mean(), "ci_low": lo, "ci_high": hi})
    rates = pd.DataFrame(rows)
    chi2, p_chi, dof, _ = stats.chi2_contingency(pd.crosstab(lab, y).to_numpy())
    # profiles: mean standardised covariate per type
    prof = pd.DataFrame(Z, columns=LANDSCAPE_COLS).groupby(lab).mean()
    # survey-recorded composition of each type (not used to build the types)
    comp = df.assign(type=lab).groupby("type")[["Source Type_encoded"] + [c for c in df.columns if c.startswith("landuse_")]].mean()
    prof = prof.join(comp.add_prefix("survey: ").set_axis(prof.index, axis=0)) if len(comp) == len(prof) else prof
    prof.index = [f"T{k + 1}" for k in prof.index]
    # type vs ward: likelihood-ratio tests and adjusted pseudo-R2
    T, W = dummies(lab), dummies(ward)
    ll0 = loglik(None, y)
    llT, llW, llTW = loglik(T, y), loglik(W, y), loglik(np.c_[T, W], y)
    kT, kW = T.shape[1], W.shape[1]
    lrt = pd.DataFrame([
        {"test": "type (vs null)", "lr_stat": 2 * (llT - ll0), "df": kT, "p": stats.chi2.sf(2 * (llT - ll0), kT)},
        {"test": "ward (vs null)", "lr_stat": 2 * (llW - ll0), "df": kW, "p": stats.chi2.sf(2 * (llW - ll0), kW)},
        {"test": "type | ward", "lr_stat": 2 * (llTW - llW), "df": kT, "p": stats.chi2.sf(2 * (llTW - llW), kT)},
        {"test": "ward | type", "lr_stat": 2 * (llTW - llT), "df": kW, "p": stats.chi2.sf(2 * (llTW - llT), kW)},
    ])
    lrt["adj_pseudo_r2_model"] = [adj_pseudo_r2(llT, ll0, kT), adj_pseudo_r2(llW, ll0, kW),
                                  adj_pseudo_r2(llTW, ll0, kT + kW), adj_pseudo_r2(llTW, ll0, kT + kW)]
    summary = pd.DataFrame([{"chi2": chi2, "df": dof, "p": p_chi, "cramers_v_type_drying": cramers_v(lab, y),
                             "cramers_v_ward_drying": cramers_v(ward, y), "cramers_v_type_ward": cramers_v(lab, ward)}])
    return rates, prof, lrt, summary


# ----------------------------------------------------------------------------- U2 regionalisation
def env_r2(P, groups):
    """Share of total variance of the PCA scores explained by group means (landscape homogeneity)."""
    tot = ((P - P.mean(0)) ** 2).sum()
    gm = pd.DataFrame(P).groupby(groups).transform("mean").to_numpy()
    return 1 - ((P - gm) ** 2).sum() / tot


def regionalisation(df, coords, P, ward, catch):
    y = df.target_dried.to_numpy()
    A = knn_graph(coords, k=10)
    parts = {"ward (survey unit)": ward, "DEM catchment T=2 km²": catch, "municipality": df.municipality.astype(str).to_numpy()}
    for n in (75, 176):
        parts[f"landscape region (n={n})"] = AgglomerativeClustering(n_clusters=n, linkage="ward",
                                                                           connectivity=A).fit_predict(P)
    rows = []
    rng = np.random.default_rng(SEED)
    for name, g in parts.items():
        n_units = len(np.unique(g))
        cell, null, _ = matched_grid_null(coords, y.astype(float), n_units, rng=rng)
        icc = icc1(y.astype(float), g)
        rows.append({"partition": name, "n_units": n_units, "mean_sources_per_unit": len(y) / n_units,
                     "env_r2": env_r2(P, g), "icc_drying": icc, "grid_cell_km": cell / 1000,
                     "grid_icc_mean": null.mean(), "grid_icc_q025": np.quantile(null, 0.025),
                     "grid_icc_q975": np.quantile(null, 0.975), "p_above_grid": (np.sum(null >= icc) + 1) / (len(null) + 1),
                     "ari_with_ward": adjusted_rand_score(ward, g), "nmi_with_ward": normalized_mutual_info_score(ward, g)})
    out = pd.DataFrame(rows)
    # landscape homogeneity and ward agreement of matched grids, for reference
    env_grid = []
    for name, g in parts.items():
        n_units = len(np.unique(g))
        cell = out.set_index("partition").grid_cell_km[name] * 1000
        grids = [grid_labels(coords, cell, *rng.uniform(0, cell, 2)) for _ in range(50)]
        env_grid.append((np.mean([env_r2(P, gg) for gg in grids]), np.mean([adjusted_rand_score(ward, gg) for gg in grids])))
    out["grid_env_r2_mean"] = [e for e, _ in env_grid]
    out["grid_ari_with_ward_mean"] = [a for _, a in env_grid]
    return out, parts


# ----------------------------------------------------------------------------- U3 dried-source mining
def dried_mining(df, coords, lab, ward):
    d = df.assign(type=[f"T{k + 1}" for k in lab], ward_id=ward)
    yr = pd.to_numeric(d["years_since_dried"], errors="coerce")
    yr = yr.where((yr >= 2000) & (yr <= 2085))
    d["period"] = np.select([yr < 2072, yr == 2072, yr > 2072], ["before 2072 BS", "2072 BS", "after 2072 BS"], "year unknown")
    m = (d.target_dried == 1).to_numpy()
    dd, cd = d[m].reset_index(drop=True), coords[m]
    rng = np.random.default_rng(SEED + 1)
    outcomes = {"earthquake attributed": dd.reason_of_drying.eq("earthquake").to_numpy().astype(float),
                "drought attributed": dd.reason_of_drying.eq("drought").to_numpy().astype(float),
                "infrastructure attributed": dd.reason_of_drying.eq("infrastructure_development").to_numpy().astype(float),
                "dated 2072 BS": dd.period.eq("2072 BS").to_numpy().astype(float)}
    rows = []
    for name, v in outcomes.items():
        for unit, g in [("ward", dd.ward_id.to_numpy()), ("landscape type", dd.type.to_numpy())]:
            icc = icc1(v, g)
            if unit == "ward":
                cell, null, _ = matched_grid_null(cd, v, len(np.unique(g)), rng=rng)
                gm, lo, hi, p = null.mean(), np.quantile(null, .025), np.quantile(null, .975), (np.sum(null >= icc) + 1) / (len(null) + 1)
            else:
                gm = lo = hi = p = np.nan
            rows.append({"outcome": name, "share": v.mean(), "unit": unit, "n_units": len(np.unique(g)), "icc": icc,
                         "grid_icc_mean": gm, "grid_icc_q025": lo, "grid_icc_q975": hi, "p_above_grid": p,
                         "cramers_v": cramers_v(g, v)})
    icc_tab = pd.DataFrame(rows)

    main_causes = ["earthquake", "drought", "infrastructure_development"]
    cross = pd.crosstab(dd.type, dd.reason_of_drying.where(dd.reason_of_drying.isin(main_causes), "other"), normalize="index")
    return icc_tab, cross


# ----------------------------------------------------------------------------- figures
# ----------------------------------------------------------------------------- U4-U8 extensions
def ward_funnel(df, Z, ward):
    """U7: indirectly standardised drying ratios of wards against a landscape-only model, with funnel limits and
    an overdispersion factor (mean squared z; ~1 if wards differ only by chance given landscape)."""
    y = df.target_dried.to_numpy()
    p = LogisticRegression(C=1.0, max_iter=5000).fit(Z, y).predict_proba(Z)[:, 1]
    g = pd.DataFrame({"ward": ward, "y": y, "p": p, "pq": p * (1 - p), "municipality": df.municipality.to_numpy()})
    w = g.groupby("ward").agg(n=("y", "size"), observed=("y", "sum"), expected=("p", "sum"), var=("pq", "sum"),
                              municipality=("municipality", "first")).reset_index()
    w["sdr"] = w.observed / w.expected
    w["z"] = (w.observed - w.expected) / np.sqrt(w["var"])
    w["outside_95"] = w.z.abs() > 1.96
    w["outside_998"] = w.z.abs() > 3.09
    summ = pd.DataFrame([{"n_wards": len(w), "overdispersion_phi": float(np.mean(w.z ** 2)),
                          "chi2_p": stats.chi2.sf(np.sum(w.z ** 2), len(w) - 1),
                          "n_outside_95": int(w.outside_95.sum()), "expected_outside_95": 0.05 * len(w),
                          "n_outside_998": int(w.outside_998.sum()), "expected_outside_998": 0.002 * len(w),
                          "n_high_998": int((w.z > 3.09).sum()), "n_low_998": int((w.z < -3.09).sum())}])
    return w, summ


# ----------------------------------------------------------------------------- driver
def run():
    METRICS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    df = load_master()
    if "catch_T2p0" not in df:
        df = df.merge(pd.read_csv(HG)[["id", "catch_T2p0"]], on="id", how="left")
    coords = coords_to_metric(df.latitude.to_numpy(), df.longitude.to_numpy())
    ward = (df.municipality.astype(str) + "_" + df.ward.astype(str)).to_numpy()
    Z = env_matrix(df)

    logger.info("U1 landscape typology")
    lab, sel, loadings, evr, P = typology(Z)
    sel.to_csv(METRICS / "un_typology_selection.csv", index=False)
    loadings.to_csv(METRICS / "un_pca_loadings.csv")
    pd.DataFrame({"pc": [f"PC{i + 1}" for i in range(len(evr))], "explained_variance_ratio": evr}).to_csv(
        METRICS / "un_pca_variance.csv", index=False)
    rates, prof, lrt, summ = typology_vs_drying(df, lab, Z, ward)
    rates.to_csv(METRICS / "un_typology_drying.csv", index=False)
    prof.to_csv(METRICS / "un_typology_profiles.csv")
    lrt.to_csv(METRICS / "un_typology_vs_ward_lrt.csv", index=False)
    summ.to_csv(METRICS / "un_typology_summary.csv", index=False)
    pd.DataFrame({"id": df.id, "env_type": [f"T{k + 1}" for k in lab]}).to_csv(
        PROJECT_ROOT / "data/processed/landscape_types.csv", index=False)
    logger.info("\n" + sel.round(3).to_string() + "\n" + rates.round(3).to_string() + "\n" + lrt.round(4).to_string()
                + "\n" + summ.round(4).to_string())

    logger.info("U2 regionalisation")
    reg, parts = regionalisation(df, coords, P, ward, df.catch_T2p0.to_numpy())
    reg.to_csv(METRICS / "un_regionalisation.csv", index=False)
    logger.info("\n" + reg.round(3).to_string())

    logger.info("U3 dried-source mining")
    icc_tab, cross = dried_mining(df, coords, lab, ward)
    icc_tab.to_csv(METRICS / "un_dried_icc.csv", index=False)
    cross.to_csv(METRICS / "un_type_by_reason.csv")
    logger.info("\n" + icc_tab.round(3).to_string())

    logger.info("U4 ward funnel against a landscape-only model")
    w, wsum = ward_funnel(df, Z, ward)
    w.to_csv(METRICS / "un_ward_funnel.csv", index=False)
    wsum.to_csv(METRICS / "un_ward_funnel_summary.csv", index=False)
    logger.info("\n" + wsum.round(3).to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()
