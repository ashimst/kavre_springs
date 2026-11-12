"""
Robustness analyses requested in review.

R1  Contiguous, size-matched random regions. Square grids differ from wards in shape and size distribution. We grow
    random regions on the 10-nearest-neighbour source graph with the ward size distribution; every region is
    connected and sizes are balanced to the ward targets by boundary transfers (src/regions.py; 200 replicates) and use them as a second null for the ICC of
    drying (all sources) and of recalled attributes (dried sources).
R2  Sensitivity subsets: all sources; springs only (ponds excluded); perennial only (sources recorded as seasonal
    excluded); springs and perennial. For each: drying rate; ICC of wards, DEM catchments and landscape regions
    against matched grids (and, for wards, contiguous random regions); landscape-type drying range; ward funnel
    overdispersion against a landscape-only model; Knox same-year ratios within and across wards; ward ICC of
    recalled attributes; within-ward odds ratio of the leading apriori items.
R3  Spatial-validation diagnostics: distance from every test source to the nearest training source for each seed,
    fold and buffer (0, 500, 1000, 2000 m) and for leave-one-municipality-out.
R4  Earthquake exposure with ward fixed effects and spatial (Moran eigenvector) adjustment, and the share of
    exposure variance that lies within wards.

Outputs: results/metrics/rb_*.csv, incl. rb_null_distributions.csv (figures: src/figures.py)
Usage:   python -m src.robustness
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

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.catchment_analysis import icc1, matched_grid_null
from src.features import explanatory_columns
from src.dataset import coords_to_metric, load_master
from src.pattern_mining import mantel_haenszel
from src.regions import sample_partitions
from src.spatial_analysis import mem_eigenvectors
from src.spatial_splits import grid_block_ids, leave_one_municipality_out, spatial_block_kfold
from src.unsupervised import env_matrix, knn_graph

logger = logging.getLogger("robustness")
METRICS = PROJECT_ROOT / "results/metrics"
FIGS = PROJECT_ROOT / "results/figures"
HG = PROJECT_ROOT / "data/processed/hydro_gis_features.csv"
TYPES = PROJECT_ROOT / "data/processed/landscape_types.csv"
SEED = 31
N_NULL = 200


def null_summary(obs, null):
    null = np.asarray(null)
    return {"null_mean": null.mean(), "null_q025": np.quantile(null, 0.025), "null_q975": np.quantile(null, 0.975),
            "p": (np.sum(null >= obs) + 1) / (len(null) + 1)}


# ----------------------------------------------------------------------------- helpers
def knox_within_across(coords, years, ward, d=250.0, n_perm=999, rng=None):
    """Compare same-year neighbour counts with permutation expectations; upper-tail p tests excess matching only."""
    rng = rng or np.random.default_rng(SEED)
    pairs = np.array(sorted(cKDTree(coords).query_pairs(d)))
    if len(pairs) == 0:
        return {}
    same_w = ward[pairs[:, 0]] == ward[pairs[:, 1]]
    out = {}
    for name, m in [("within", same_w), ("across", ~same_w)]:
        pp = pairs[m]
        obs = np.sum(years[pp[:, 0]] == years[pp[:, 1]])
        null = []
        for _ in range(n_perm):
            y = rng.permutation(years)
            null.append(np.sum(y[pp[:, 0]] == y[pp[:, 1]]))
        null = np.array(null)
        out[f"knox_{name}_pairs"] = len(pp)
        out[f"knox_{name}_ratio"] = obs / null.mean() if null.mean() > 0 else np.nan
        out[f"knox_{name}_p"] = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return out


def funnel_phi(Z, y, ward):
    p = LogisticRegression(C=1.0, max_iter=5000).fit(Z, y).predict_proba(Z)[:, 1]
    g = pd.DataFrame({"w": ward, "y": y, "p": p, "v": p * (1 - p)}).groupby("w").sum()
    z = (g.y - g.p) / np.sqrt(g.v)
    return float(np.mean(z ** 2)), int((z.abs() > 3.09).sum()), len(g)


# ----------------------------------------------------------------------------- R1 + R2
def inside_recorded_palika(df) -> np.ndarray:
    import geopandas as gpd

    from src.boundaries import PALIKAS, build_boundaries
    _, pal = build_boundaries()
    pts = gpd.GeoDataFrame({"recorded": df.municipality.map(PALIKAS).to_numpy()},
                           geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
    j = gpd.sjoin(pts, pal[["name", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()].reindex(pts.index)
    return (j.recorded == j["name"]).to_numpy()


def subset_analyses(df, coords):
    ward = (df.municipality.astype(str) + "_" + df.ward.astype(str)).to_numpy()
    types = df[["id"]].merge(pd.read_csv(TYPES), on="id", how="left").env_type.to_numpy()
    Z_all = env_matrix(df)
    P = PCA(random_state=SEED).fit_transform(Z_all)
    P = P[:, : int(np.searchsorted(np.cumsum(PCA().fit(Z_all).explained_variance_ratio_), 0.9) + 1)]
    land_reg = AgglomerativeClustering(n_clusters=75, linkage="ward", connectivity=knn_graph(coords, 10)).fit_predict(P)
    yr = pd.to_numeric(df["years_since_dried"], errors="coerce").where(lambda v: (v >= 2000) & (v <= 2085)).to_numpy()
    tpi_hi = (df.tpi_500m > df.tpi_500m.quantile(2 / 3)).to_numpy()
    prot = ((df.tpi_500m <= df.tpi_500m.quantile(1 / 3)) & (df.soil > df.soil.quantile(2 / 3))).to_numpy()
    subsets = {
        "all sources": np.ones(len(df), bool),
        "springs only (ponds excluded)": (df["Source Type_encoded"] == 0).to_numpy(),
        "perennial only (seasonal excluded)": (df["Perennial / Seasonal_encoded"] == 0).to_numpy(),
        "perennial springs only": ((df["Source Type_encoded"] == 0) & (df["Perennial / Seasonal_encoded"] == 0)).to_numpy(),
        # sources whose location lies inside the official polygon of their recorded palika (src/boundaries.py)
        "located in recorded palika": inside_recorded_palika(df),
    }
    rows, nulls = [], []
    for name, m in subsets.items():
        rng = np.random.default_rng(SEED)
        y = df.target_dried.to_numpy()[m].astype(float)
        c, w = coords[m], ward[m]
        r = {"subset": name, "n": int(m.sum()), "n_dried": int(y.sum()), "drying_rate": y.mean()}
        # ward ICC vs grids and vs contiguous random regions (regions grown on this subset's graph)
        icc_w = icc1(y, w)
        _, gnull, _ = matched_grid_null(c, y, len(np.unique(w)), rng=rng)
        # contiguous, size-balanced random regions grown on this subset's graph (src/regions.py)
        regs = sample_partitions(c, w, N_NULL, SEED + 1)
        rnull = [icc1(y, g) for g in regs]
        nulls += [{"subset": name, "outcome": "drying", "null": "grid", "value": v} for v in gnull]
        nulls += [{"subset": name, "outcome": "drying", "null": "region", "value": v} for v in rnull]
        r.update({"ward_icc": icc_w, "ward_grid_q025": np.quantile(gnull, .025), "ward_grid_q975": np.quantile(gnull, .975), "ward_p_grid": null_summary(icc_w, gnull)["p"],
                  "ward_region_q025": np.quantile(rnull, .025), "ward_region_q975": np.quantile(rnull, .975),
                  "ward_p_region": null_summary(icc_w, rnull)["p"]})
        for uname, lab in [("catchment_T2", df.catch_T2p0.to_numpy()[m]), ("landscape_region75", land_reg[m])]:
            icc_u = icc1(y, lab)
            _, gn, _ = matched_grid_null(c, y, len(np.unique(lab)), n_shift=100, rng=rng)
            r[f"{uname}_icc"] = icc_u
            r[f"{uname}_grid_q975"] = np.quantile(gn, .975)
            r[f"{uname}_p_grid"] = null_summary(icc_u, gn)["p"]
        tr = pd.Series(y).groupby(types[m]).mean()
        r["type_drying_min"], r["type_drying_max"] = tr.min(), tr.max()
        r["funnel_phi"], r["funnel_n_outside_998"], r["funnel_n_wards"] = funnel_phi(Z_all[m], y.astype(int), w)
        for item, mask in [("TPI high -> dried", tpi_hi), ("TPI low & soil moisture high -> dried", prot)]:
            o, lo, hi = mantel_haenszel(mask[m], y.astype(int), w)
            r[f"mh_or[{item}]"], r[f"mh_lo[{item}]"], r[f"mh_hi[{item}]"] = o, lo, hi
        # among dried sources of this subset
        dm = m & (df.target_dried == 1).to_numpy()
        ok = dm & ~np.isnan(yr)
        r.update(knox_within_across(coords[ok], yr[ok], ward[ok], rng=np.random.default_rng(SEED)))
        regs_d = [g[(df.target_dried.to_numpy()[m] == 1)] for g in regs]
        for aname, v in [("earthquake", df.reason_of_drying.eq("earthquake").to_numpy()),
                         ("dated_2072", yr == 2072)]:
            vv = v[dm].astype(float)
            icc_a = icc1(vv, ward[dm])
            _, gn, _ = matched_grid_null(coords[dm], vv, len(np.unique(ward[dm])), n_shift=100, rng=rng)
            rn = [icc1(vv, g) for g in regs_d]
            nulls += [{"subset": name, "outcome": aname, "null": "grid", "value": v} for v in gn]
            nulls += [{"subset": name, "outcome": aname, "null": "region", "value": v} for v in rn]
            r.update({f"{aname}_ward_icc": icc_a, f"{aname}_grid_q975": np.quantile(gn, .975),
                      f"{aname}_p_grid": null_summary(icc_a, gn)["p"], f"{aname}_region_q975": np.quantile(rn, .975),
                      f"{aname}_p_region": null_summary(icc_a, rn)["p"]})
        rows.append(r)
        logger.info(f"{name}: done")
    return pd.DataFrame(rows), pd.DataFrame(nulls)


# ----------------------------------------------------------------------------- R3
def validation_distances(df, coords):
    rows = []
    for buf in [0.0, 500.0, 1000.0, 2000.0]:
        for seed in [42, 123, 456, 789, 2024]:
            for s in spatial_block_kfold(coords, 5, random_state=seed, buffer_m=buf):
                d, _ = cKDTree(coords[s["train_mask"]]).query(coords[s["test_mask"]])
                rows.append({"design": "spatial block CV", "buffer_m": buf, "seed": seed, "fold": s["fold"],
                             "n_train": s["n_train"], "n_test": s["n_test"], "median_m": np.median(d),
                             "p05_m": np.quantile(d, .05), "share_within_500m": (d <= 500).mean(),
                             "share_within_1km": (d <= 1000).mean()})
    for seed in [42, 123, 456]:
        for s in leave_one_municipality_out(coords, df.municipality.to_numpy(), random_state=seed, buffer_m=500.0):
            d, _ = cKDTree(coords[s["train_mask"]]).query(coords[s["test_mask"]])
            rows.append({"design": "leave-one-municipality-out", "buffer_m": 500.0, "seed": seed, "fold": s["fold"],
                         "n_train": s["n_train"], "n_test": s["n_test"], "median_m": np.median(d),
                         "p05_m": np.quantile(d, .05), "share_within_500m": (d <= 500).mean(),
                         "share_within_1km": (d <= 1000).mean()})
    folds = pd.DataFrame(rows)
    summ = folds.groupby(["design", "buffer_m"]).agg(
        n_splits=("fold", "size"), mean_n_train=("n_train", "mean"), median_distance_m=("median_m", "median"),
        p05_distance_m=("p05_m", "median"), share_test_within_500m=("share_within_500m", "mean"),
        share_test_within_1km=("share_within_1km", "mean")).reset_index()
    return folds, summ


# ----------------------------------------------------------------------------- R4
def earthquake_adjusted(df, coords, n_boot=200, cols=None):
    y = df.target_dried.to_numpy()
    ward = (df.municipality.astype(str) + "_" + df.ward.astype(str)).to_numpy()
    seis = ["pga_main", "pga_may12", "aftershocks_20km"]
    var = []
    for c in seis:
        v = df[c].astype(float)
        within = (v - v.groupby(ward).transform("mean")).var()
        var.append({"variable": c, "share_of_variance_within_wards": within / v.var(),
                    "within_ward_sd": np.sqrt(within), "total_sd": v.std()})
    S = StandardScaler().fit_transform(df[seis].to_numpy(float))
    cols = cols or explanatory_columns(df)
    E = StandardScaler().fit_transform(df[cols].fillna(df[cols].median()).to_numpy(float))
    W = pd.get_dummies(ward, drop_first=True, dtype=float).to_numpy()
    _, V = mem_eigenvectors(coords)
    blocks = grid_block_ids(coords, 8)
    ub = np.unique(blocks)
    members = {b: np.where(blocks == b)[0] for b in ub}
    rng = np.random.default_rng(SEED)
    designs = {"seismic + environment": np.c_[S, E],
               "seismic + environment + ward fixed effects": np.c_[S, E, W],
               "seismic + environment + 100 MEMs (broad and meso spatial structure)": np.c_[S, E, V[:, :100]]}
    rows = []
    for name, X in designs.items():
        fit = lambda XX, yy: LogisticRegression(C=10.0, max_iter=10000).fit(XX, yy).coef_[0][:3]
        coef = fit(X, y)
        boots = []
        for _ in range(n_boot):
            idx = np.concatenate([members[b] for b in rng.choice(ub, len(ub), replace=True)])
            boots.append(fit(X[idx], y[idx]))
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        for k, c in enumerate(seis):
            rows.append({"model": name, "term": c, "coef_per_sd": coef[k], "ci_low": lo[k], "ci_high": hi[k],
                         "ci_excludes_zero": bool(lo[k] > 0 or hi[k] < 0)})
    return pd.DataFrame(var), pd.DataFrame(rows)


def run():
    df = load_master()
    if "catch_T2p0" not in df:
        df = df.merge(pd.read_csv(HG)[["id", "catch_T2p0"]], on="id", how="left")
    coords = coords_to_metric(df.latitude.to_numpy(), df.longitude.to_numpy())
    logger.info("R3 validation distances")
    vfolds, vsumm = validation_distances(df, coords)
    vfolds.to_csv(METRICS / "rb_validation_distances_folds.csv", index=False)
    vsumm.to_csv(METRICS / "rb_validation_distances.csv", index=False)
    logger.info("\n" + vsumm.round(3).to_string())
    logger.info("R4 earthquake adjustment")
    var, eq = earthquake_adjusted(df, coords)
    var.to_csv(METRICS / "rb_seismic_within_ward_variance.csv", index=False)
    eq.to_csv(METRICS / "rb_seismic_adjusted.csv", index=False)
    logger.info("\n" + var.round(3).to_string() + "\n" + eq.round(3).to_string())
    logger.info("R1/R2 null regions and sensitivity subsets")
    sub, nulls = subset_analyses(df, coords)
    sub.to_csv(METRICS / "rb_sensitivity_subsets.csv", index=False)
    logger.info("\n" + sub.round(3).T.to_string())
    nulls.to_csv(METRICS / "rb_null_distributions.csv", index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()
