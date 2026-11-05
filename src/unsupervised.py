"""
Unsupervised data mining of the spring inventory (no drying labels are used to build any structure).

U1  Landscape typology. The continuous GIS / remote-sensing covariates of env_hydro (LANDSCAPE_COLS: terrain,
    catchment hydrology, climate, water balance, vegetation, roads; survey-recorded fields and coordinates excluded)
    are standardised and reduced by PCA
    (components explaining 90 % of variance); Gaussian mixtures with K = 2..12 are compared by BIC and by bootstrap
    stability (mean adjusted Rand index between mixtures fitted to 80 % subsamples and the full-data mixture).
    Drying is then compared between types (rates with Wilson CIs, chi-square), and type and ward are compared with
    likelihood-ratio tests and adjusted McFadden pseudo-R^2.
U2  Regionalisation. Label-free partitions of the springs are built from the environment and the spring graph:
      * spatially constrained Ward clustering of the PCA scores on the k-NN graph (contiguous landscape
        regions), with as many regions as there are wards (75) and as there are 2 km^2 DEM catchments (176);
      * Louvain communities of the k-NN graph with edges weighted by landscape similarity.
    For each partition (and for wards, DEM catchments and municipalities) we report the share of landscape
    variance it explains (landscape homogeneity), the ICC of drying against 200 randomly shifted grids with the
    same number of units (drying homogeneity), and agreement with wards (ARI, NMI).
U3  Mining of dried sources. Among dried sources, ward-level structure of the reported cause (earthquake) and of the
    reported year (2072 BS) is compared with matched grids, and with landscape types; association rules
    (one- and two-item antecedents over type, source type, land use, municipality and period) are screened for
    reported causes with Fisher's exact test and BH-FDR.

Outputs: results/metrics/un_*.csv, results/figures/un_typology.png, un_regionalisation.png, un_dried_mining.png
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

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))  # silences a joblib core-count probe on Windows

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
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
from src.graph_builder import coords_to_metric, load_master
from src.spatial_mining import bh_fdr

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


def louvain_communities(coords, Z, resolution, k=7):
    _, idx = cKDTree(coords).query(coords, k=k + 1)
    G = nx.Graph()
    G.add_nodes_from(range(len(coords)))
    d2 = np.array([[np.sum((Z[i] - Z[j]) ** 2) for j in idx[i, 1:]] for i in range(len(coords))])
    s2 = np.median(d2)
    for i in range(len(coords)):
        for jj, j in enumerate(idx[i, 1:]):
            G.add_edge(i, int(j), weight=float(np.exp(-d2[i, jj] / s2)))
    comms = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=SEED)
    lab = np.empty(len(coords), np.int64)
    for c, nodes in enumerate(comms):
        lab[list(nodes)] = c
    return lab, nx.community.modularity(G, comms, weight="weight")


def regionalisation(df, coords, P, ward, catch):
    y = df.target_dried.to_numpy()
    A = knn_graph(coords, k=10)
    parts = {"ward (survey unit)": ward, "DEM catchment T=2 km²": catch, "municipality": df.municipality.astype(str).to_numpy()}
    for n in (75, 176):
        parts[f"landscape region (n={n})"] = AgglomerativeClustering(n_clusters=n, linkage="ward",
                                                                           connectivity=A).fit_predict(P)
    # Louvain resolution chosen (label-free) to give a partition closest to the number of wards
    best = None
    for res in [0.5, 1, 2, 4, 8, 16, 32]:
        lab, Q = louvain_communities(coords, P, res)
        nc = len(np.unique(lab))
        if best is None or abs(nc - 75) < abs(best[2] - 75):
            best = (lab, Q, nc, res)
    parts["Louvain landscape-similarity graph"] = best[0]
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
    out["louvain_resolution"] = np.where(out.partition.str.startswith("Louvain"), best[3], np.nan)
    out["louvain_modularity"] = np.where(out.partition.str.startswith("Louvain"), best[1], np.nan)
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
def figures(df, coords, sel, rates, prof, reg, parts, icc_tab, lab):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), layout="constrained")
    ax = axes[0, 0]
    ax2 = ax.twinx()
    ax.plot(sel.K, sel.bic / 1e3, "o-", c="#4C72B0", label="BIC")
    ax2.errorbar(sel.K, sel.stability_ari_mean, yerr=sel.stability_ari_sd, fmt="s--", c="#C44E52", capsize=2, label="stability (ARI)")
    ks = int(sel.K[sel.selected].iloc[0])
    ax.axvline(ks, c="grey", ls=":")
    ax.set_xlabel("Number of landscape types K")
    ax.set_ylabel("BIC (×10³)", color="#4C72B0")
    ax2.set_ylabel("Bootstrap stability (mean ARI)", color="#C44E52")
    ax.set_title(f"(a) Gaussian-mixture model selection (selected K = {ks})", fontsize=10)
    ax = axes[0, 1]
    top = prof[LANDSCAPE_COLS].abs().max().sort_values(ascending=False).index[:16]
    im = ax.imshow(prof[top].to_numpy(), cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_yticks(range(len(prof)))
    ax.set_yticklabels(prof.index)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top, rotation=70, ha="right", fontsize=7)
    fig.colorbar(im, ax=ax, label="mean z-score")
    ax.set_title("(b) Landscape profile of each type (16 most distinctive covariates)", fontsize=10)
    ax = axes[1, 0]
    ax.errorbar(rates.type, rates.drying_rate, yerr=[rates.drying_rate - rates.ci_low, rates.ci_high - rates.drying_rate],
                fmt="o", capsize=3, c="k")
    ax.axhline(df.target_dried.mean(), ls=":", c="grey")
    for i, r in rates.iterrows():
        ax.text(i, r.ci_high + 0.01, f"n={r.n_sources}", ha="center", fontsize=7)
    ax.set_ylabel("Share dried (95% Wilson CI)")
    ax.set_title("(c) Drying rate by landscape type (types built without labels)", fontsize=10)
    ax = axes[1, 1]
    cmap = plt.get_cmap("tab10")
    for k in np.unique(lab):
        mm = lab == k
        ax.scatter(df.longitude[mm], df.latitude[mm], s=1.5, color=cmap(k % 10), label=f"T{k + 1}", linewidths=0)
    ax.set_aspect(1 / np.cos(np.radians(27.57)))
    ax.legend(markerscale=6, fontsize=7, loc="lower left", ncol=2)
    ax.set_title("(d) Map of landscape types", fontsize=10)
    fig.savefig(FIGS / "un_typology.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), layout="constrained")
    ax = axes[0]
    r = reg.reset_index(drop=True)
    y = np.arange(len(r))
    ax.hlines(y, r.grid_icc_q025, r.grid_icc_q975, color="#bbbbbb", lw=6, label="size-matched random grids (95%)")
    ax.plot(r.icc_drying, y, "o", c="#C44E52", label="partition")
    ax.set_yticks(y)
    ax.set_yticklabels([p if "n=" in p else f"{p} (n={n})" for p, n in zip(r.partition, r.n_units)], fontsize=8)
    ax.set_xlabel("ICC of drying")
    ax.set_title("(a) Drying homogeneity of label-free partitions", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax = axes[1]
    for i, q in r.iterrows():
        ax.scatter(q.env_r2, q.icc_drying - q.grid_icc_mean, s=40)
        ax.annotate(q.partition, (q.env_r2, q.icc_drying - q.grid_icc_mean), fontsize=7,
                    xytext=(5, 6 if i % 2 else -10), textcoords="offset points")
    ax.axvline(r.grid_env_r2_mean[r.partition.str.startswith("ward")].iloc[0], ls=":", c="grey")
    ax.text(r.grid_env_r2_mean[r.partition.str.startswith("ward")].iloc[0], ax.get_ylim()[0],
            " random squares,\n ward size", fontsize=7, color="grey", va="bottom")
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xlabel("Landscape homogeneity (share of PCA variance explained by units)")
    ax.set_ylabel("ICC of drying minus matched-grid mean")
    ax.set_title("(b) Landscape vs drying homogeneity of each partition", fontsize=10)
    fig.savefig(FIGS / "un_regionalisation.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    t = icc_tab[icc_tab.unit == "ward"].reset_index(drop=True)
    yy = np.arange(len(t))
    ax.hlines(yy, t.grid_icc_q025, t.grid_icc_q975, color="#bbbbbb", lw=6, label="size-matched random grids (95%)")
    ax.plot(t.icc, yy, "o", c="#C44E52", label="wards")
    e = icc_tab[icc_tab.unit == "landscape type"].reset_index(drop=True)
    ax.plot(e.icc, yy, "D", c="#4C72B0", label="landscape types")
    ax.set_yticks(yy)
    ax.set_yticklabels(t.outcome)
    ax.set_xlabel("ICC among dried sources")
    ax.set_title("Reported causes and years of drying: shared within wards, not within landscape types", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(FIGS / "un_dried_mining.png", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------- U4-U8 extensions
def density_structure(P, lab, y, n_pc=5):
    """U4: HDBSCAN on the leading principal components; does density-based structure agree with the mixture types?"""
    from sklearn.cluster import HDBSCAN
    hd = HDBSCAN(min_cluster_size=100, min_samples=20).fit(P[:, :n_pc])
    h = hd.labels_
    rows = []
    for k in np.unique(h):
        m = h == k
        lo, hi = wilson(y[m].sum(), m.sum())
        rows.append({"cluster": "noise" if k < 0 else f"D{k + 1}", "n_sources": int(m.sum()), "drying_rate": y[m].mean(),
                     "ci_low": lo, "ci_high": hi,
                     "dominant_landscape_type": f"T{np.bincount(lab[m]).argmax() + 1}",
                     "share_dominant_type": np.bincount(lab[m]).max() / m.sum()})
    tab = pd.DataFrame(rows)
    core = h >= 0
    summ = pd.DataFrame([{"n_clusters": int(len(np.unique(h[core]))), "noise_share": float((~core).mean()),
                          "ari_with_types_core": adjusted_rand_score(lab[core], h[core]),
                          "cramers_v_cluster_drying": cramers_v(h, y)}])
    return h, tab, summ


def embedding(P, seed=SEED):
    """U5: t-SNE of the PCA scores (visual check of landscape structure)."""
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, perplexity=50, init="pca", learning_rate="auto", random_state=seed).fit_transform(P)


def anomalies(P, y, lab, ward, contamination=0.05):
    """U6: isolation-forest landscape anomalies: are atypical settings more likely to be dried?"""
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(n_estimators=500, contamination=contamination, random_state=SEED).fit(P)
    a = iso.predict(P) == -1
    k1, n1, k0, n0 = y[a].sum(), a.sum(), y[~a].sum(), (~a).sum()
    odds, p = stats.fisher_exact([[k1, n1 - k1], [k0, n0 - k0]])
    _, cnt = np.unique(ward[a], return_counts=True)
    return a, pd.DataFrame([{"n_anomalous": int(n1), "drying_rate_anomalous": k1 / n1, "drying_rate_typical": k0 / n0,
                             "odds_ratio": odds, "fisher_p": p, "n_wards_with_anomalies": len(cnt),
                             "max_share_in_one_ward": cnt.max() / n1,
                             "anomaly_share_by_type": "; ".join(f"T{k + 1}: {a[lab == k].mean():.3f}" for k in np.unique(lab))}])


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


def mantel(D1, D2, n_perm=999, rng=None):
    rng = rng or np.random.default_rng(SEED)
    iu = np.triu_indices_from(D1, 1)
    r0 = stats.pearsonr(D1[iu], D2[iu])[0]
    null = []
    for _ in range(n_perm):
        o = rng.permutation(len(D1))
        null.append(stats.pearsonr(D1[np.ix_(o, o)][iu], D2[iu])[0])
    return r0, (np.sum(np.array(null) >= r0) + 1) / (n_perm + 1)


def partial_mantel(D1, D2, D3, n_perm=999, rng=None):
    """Mantel correlation of D1 and D2 controlling for D3 (residuals of both on D3; permutation of D1)."""
    rng = rng or np.random.default_rng(SEED)
    iu = np.triu_indices_from(D1, 1)

    def resid(a, b):
        A = np.c_[np.ones(len(b)), b]
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]

    r2 = resid(D2[iu], D3[iu])
    r0 = stats.pearsonr(resid(D1[iu], D3[iu]), r2)[0]
    null = []
    for _ in range(n_perm):
        o = rng.permutation(len(D1))
        null.append(stats.pearsonr(resid(D1[np.ix_(o, o)][iu], D3[iu]), r2)[0])
    return r0, (np.sum(np.array(null) >= r0) + 1) / (n_perm + 1)


def ward_narratives(df, coords, P, ward, min_dried=8):
    """U8: cluster wards by the profile of reported causes and periods of their dried sources (Hellinger geometry,
    Ward linkage, K by silhouette) and test whether narrative similarity follows geography or landscape (Mantel)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist, squareform
    from sklearn.metrics import silhouette_score
    d = df.assign(ward_id=ward)
    yr = pd.to_numeric(d["years_since_dried"], errors="coerce").where(lambda v: (v >= 2000) & (v <= 2085))
    d["period"] = np.select([yr < 2072, yr == 2072, yr > 2072], ["before", "2072", "after"], "unknown")
    d["cause"] = d.reason_of_drying.where(d.reason_of_drying.isin(["earthquake", "drought", "infrastructure_development"]), "other")
    dd = d[d.target_dried == 1]
    cnt = dd.groupby("ward_id").size()
    keep = cnt[cnt >= min_dried].index
    prof = pd.concat([pd.crosstab(dd.ward_id, dd.cause, normalize="index").add_prefix("cause: "),
                      pd.crosstab(dd.ward_id, dd.period, normalize="index").add_prefix("period: ")], axis=1).loc[keep].fillna(0)
    H = np.sqrt(prof.to_numpy())  # Hellinger embedding
    Lk = linkage(H, method="ward")
    sil = {k: silhouette_score(H, fcluster(Lk, k, "maxclust")) for k in range(2, 7)}
    k_best = max(sil, key=sil.get)
    regime = fcluster(Lk, k_best, "maxclust")
    cen = pd.DataFrame(coords).groupby(ward).mean().loc[keep].to_numpy()
    land = pd.DataFrame(P).groupby(ward).mean().loc[keep].to_numpy()
    Dn, Dg, Dl = squareform(pdist(H)), squareform(pdist(cen)), squareform(pdist(land))
    mt = []
    for name, (A, B, C) in {"narrative ~ geography": (Dn, Dg, None), "narrative ~ landscape": (Dn, Dl, None),
                            "narrative ~ geography | landscape": (Dn, Dg, Dl),
                            "narrative ~ landscape | geography": (Dn, Dl, Dg)}.items():
        r, p = mantel(A, B) if C is None else partial_mantel(A, B, C)
        mt.append({"test": name, "mantel_r": r, "p": p})
    prof = prof.assign(regime=[f"R{r}" for r in regime], n_dried=cnt.loc[keep].to_numpy(),
                       municipality=d.groupby("ward_id").municipality.first().loc[keep].to_numpy())
    reg_summary = prof.groupby("regime").mean(numeric_only=True).assign(n_wards=prof.groupby("regime").size())
    sel = pd.DataFrame({"k": list(sil), "silhouette": list(sil.values()), "selected": [k == k_best for k in sil]})
    return prof, reg_summary, pd.DataFrame(mt), sel, cen


def figures_ext(df, coords, lab, h, emb, a, w, wsum, prof, ward):
    y = df.target_dried.to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), layout="constrained")
    cmap = plt.get_cmap("tab10")
    ax = axes[0]
    for k in np.unique(lab):
        m = lab == k
        ax.scatter(emb[m, 0], emb[m, 1], s=1.2, color=cmap(k), label=f"T{k + 1}", linewidths=0)
    ax.legend(markerscale=6, fontsize=7)
    ax.set_title("(a) t-SNE of landscape space: landscape types", fontsize=10)
    ax = axes[1]
    o = np.argsort(y)
    ax.scatter(emb[o, 0], emb[o, 1], s=1.2, c=np.where(y[o] == 1, "#C44E52", "#c9d6e8"), linewidths=0)
    ax.scatter(emb[a, 0], emb[a, 1], s=6, facecolors="none", edgecolors="k", linewidths=0.4, label="isolation-forest anomaly")
    ax.legend(fontsize=7)
    ax.set_title("(b) t-SNE: dried (red) vs active; landscape anomalies", fontsize=10)
    ax = axes[2]
    sdr_src = pd.Series(w.set_index("ward").sdr).reindex(ward).to_numpy()
    sc = ax.scatter(emb[:, 0], emb[:, 1], s=1.2, c=np.log2(sdr_src), cmap="RdBu_r", vmin=-1.5, vmax=1.5, linewidths=0)
    fig.colorbar(sc, ax=ax, label="log2 ward drying ratio (observed / landscape-expected)")
    ax.set_title("(c) t-SNE coloured by the source's ward excess", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(FIGS / "un_embedding.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1, 1.2]})
    ax = axes[0]
    E = np.linspace(w.expected.min() * 0.8, w.expected.max() * 1.05, 200)
    vbar = (w["var"] / w.expected).mean()  # average p(1-p)/p, to draw smooth limits
    for z, ls in [(1.96, "--"), (3.09, ":")]:
        ax.plot(E, 1 + z * np.sqrt(vbar * E) / E, "k" + ls, lw=0.8)
        ax.plot(E, np.clip(1 - z * np.sqrt(vbar * E) / E, 0, None), "k" + ls, lw=0.8)
    ax.scatter(w.expected, w.sdr, s=18, c=np.where(w.outside_998, "#C44E52", "#4C72B0"))
    ax.axhline(1, c="grey", lw=0.8)
    ax.set_xlabel("Expected dried sources in ward (landscape-only model)")
    ax.set_ylabel("Observed / expected")
    ax.set_title(f"(a) Ward funnel plot: {int(wsum.n_outside_998[0])} of {int(wsum.n_wards[0])} wards outside 99.8% "
                 f"(expected {wsum.expected_outside_998[0]:.1f}); φ = {wsum.overdispersion_phi[0]:.1f}", fontsize=9)
    ax = axes[1]
    sdr_src = pd.Series(w.set_index("ward").sdr).reindex(ward).to_numpy()
    sc = ax.scatter(df.longitude, df.latitude, s=1.5, c=np.log2(sdr_src), cmap="RdBu_r", vmin=-1.5, vmax=1.5, linewidths=0)
    fig.colorbar(sc, ax=ax, label="log2 observed / expected")
    ax.set_aspect(1 / np.cos(np.radians(27.57)))
    ax.set_title("(b) Ward excess drying beyond landscape", fontsize=10)
    ax = axes[2]
    cols = [c for c in prof.columns if c.startswith(("cause:", "period:"))]
    pr = prof.sort_values("regime")
    im = ax.imshow(pr[cols].to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("infrastructure_development", "infrastructure") for c in cols], rotation=60, ha="right", fontsize=7)
    b = np.r_[0, np.cumsum(pr.groupby("regime").size().to_numpy())]
    for i in b[1:-1]:
        ax.axhline(i - 0.5, c="w", lw=1.5)
    ax.set_yticks((b[:-1] + b[1:]) / 2 - 0.5)
    ax.set_yticklabels(sorted(pr.regime.unique()))
    fig.colorbar(im, ax=ax, label="share of the ward's dried sources")
    ax.set_title("(c) Narrative regimes of wards (clustered cause and period profiles)", fontsize=10)
    fig.savefig(FIGS / "un_wards.png", dpi=300)
    plt.close(fig)


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

    figures(df, coords, sel, rates, prof, reg, parts, icc_tab, lab)

    y = df.target_dried.to_numpy()
    logger.info("U4 density-based clusters (HDBSCAN)")
    h, hd_tab, hd_sum = density_structure(P, lab, y)
    hd_tab.to_csv(METRICS / "un_hdbscan_clusters.csv", index=False)
    hd_sum.to_csv(METRICS / "un_hdbscan_summary.csv", index=False)
    logger.info("\n" + hd_tab.round(3).to_string() + "\n" + hd_sum.round(3).to_string())
    logger.info("U5 t-SNE embedding; U6 isolation-forest anomalies")
    emb = embedding(P)
    pd.DataFrame({"id": df.id, "tsne_1": emb[:, 0], "tsne_2": emb[:, 1]}).to_csv(METRICS / "un_tsne.csv", index=False)
    a, an = anomalies(P, y, lab, ward)
    an.to_csv(METRICS / "un_anomalies.csv", index=False)
    logger.info("\n" + an.round(4).to_string())
    logger.info("U7 ward funnel; U8 ward narrative regimes")
    w, wsum = ward_funnel(df, Z, ward)
    w.to_csv(METRICS / "un_ward_funnel.csv", index=False)
    wsum.to_csv(METRICS / "un_ward_funnel_summary.csv", index=False)
    logger.info("\n" + wsum.round(3).to_string())
    nprof, nreg, nmantel, nsel, _ = ward_narratives(df, coords, P, ward)
    nprof.to_csv(METRICS / "un_ward_narratives.csv")
    nreg.to_csv(METRICS / "un_narrative_regimes.csv")
    nmantel.to_csv(METRICS / "un_narrative_mantel.csv", index=False)
    nsel.to_csv(METRICS / "un_narrative_k_selection.csv", index=False)
    logger.info("\n" + nreg.round(3).to_string() + "\n" + nmantel.round(4).to_string() + "\n" + nsel.round(3).to_string())
    figures_ext(df, coords, lab, h, emb, a, w, wsum, nprof, ward)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()
