"""
Robustness analyses requested in review.

R1  Contiguous, size-matched random regions. Square grids differ from wards in shape and size distribution. We grow
    random contiguous regions on the 10-nearest-neighbour spring graph with exactly the ward size distribution
    (round-robin region growing from random seeds; 200 replicates) and use them as a second null for the ICC of
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

Outputs: results/metrics/rb_*.csv, results/figures/rb_robustness.png
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.catchment_analysis import icc1, matched_grid_null
from src.features import explanatory_columns
from src.graph_builder import coords_to_metric, load_master
from src.pattern_mining import mantel_haenszel
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


# ----------------------------------------------------------------------------- R1 random contiguous regions
def neighbours(coords, k=10):
    A = knn_graph(coords, k=k).tocsr()
    return [A.indices[A.indptr[i]:A.indptr[i + 1]] for i in range(A.shape[0])]


def random_regions(nbrs, sizes, rng, coords):
    """Grow len(sizes) contiguous, compact regions with the given sizes from random seeds: regions take turns adding
    the unassigned graph neighbour closest to their seed until they reach their size. Sources left over (enclosed by
    other regions) are attached, nearest first, to the closest region still below its target size."""
    import heapq
    n = len(nbrs)
    lab = -np.ones(n, np.int64)
    sizes = rng.permutation(sizes)
    seeds = rng.choice(n, len(sizes), replace=False)
    heaps, count = [], np.zeros(len(sizes), np.int64)
    for r, s in enumerate(seeds):
        lab[s] = r
        count[r] = 1
        h = [(float(np.hypot(*(coords[u] - coords[s]))), int(u)) for u in nbrs[s]]
        heapq.heapify(h)
        heaps.append(h)
    active = [r for r in range(len(sizes)) if count[r] < sizes[r]]
    while active:
        nxt = []
        for r in active:
            h = heaps[r]
            while h and lab[h[0][1]] >= 0:
                heapq.heappop(h)
            if not h:
                continue
            _, v = heapq.heappop(h)
            lab[v] = r
            count[r] += 1
            s = seeds[r]
            for u in nbrs[v]:
                if lab[u] < 0:
                    heapq.heappush(h, (float(np.hypot(*(coords[u] - coords[s]))), int(u)))
            if count[r] < sizes[r]:
                nxt.append(r)
        active = nxt
    # leftovers: attach to the nearest region that is still below its target size (else to the nearest region),
    # one source at a time so that deficits are filled in turn
    left = list(rng.permutation(np.where(lab < 0)[0]))
    while left:
        deficit = np.where(count < sizes)[0]
        pool = np.where(np.isin(lab, deficit))[0] if len(deficit) else np.where(lab >= 0)[0]
        tree = cKDTree(coords[pool])
        _, j = tree.query(coords[left])
        order = np.argsort(_)  # closest leftovers first
        for k in order[: max(1, len(left) // 10)]:
            v = left[k]
            r = lab[pool[j[k]]]
            lab[v] = r
            count[r] += 1
        left = [v for v in left if lab[v] < 0]
    return lab


def region_null(nbrs, sizes, coords, n_rep=N_NULL, seed=SEED):
    rng = np.random.default_rng(seed)
    return [random_regions(nbrs, sizes, rng, coords) for _ in range(n_rep)]


def null_summary(obs, null):
    null = np.asarray(null)
    return {"null_mean": null.mean(), "null_q025": np.quantile(null, 0.025), "null_q975": np.quantile(null, 0.975),
            "p": (np.sum(null >= obs) + 1) / (len(null) + 1)}


# ----------------------------------------------------------------------------- helpers
def knox_within_across(coords, years, ward, d=250.0, n_perm=999, rng=None):
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
def subset_analyses(df, coords, nbrs_all):
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
    }
    rows, nulls = [], {}
    for name, m in subsets.items():
        rng = np.random.default_rng(SEED)
        y = df.target_dried.to_numpy()[m].astype(float)
        c, w = coords[m], ward[m]
        r = {"subset": name, "n": int(m.sum()), "n_dried": int(y.sum()), "drying_rate": y.mean()}
        # ward ICC vs grids and vs contiguous random regions (regions grown on this subset's graph)
        icc_w = icc1(y, w)
        _, gnull, _ = matched_grid_null(c, y, len(np.unique(w)), rng=rng)
        nb = nbrs_all if name == "all sources" else neighbours(c)
        sizes = pd.Series(w).value_counts().to_numpy()
        regs = region_null(nb, sizes, c, seed=SEED + 1)
        rnull = [icc1(y, g) for g in regs]
        if name == "all sources":
            nulls["drying"] = (icc_w, gnull, rnull)
        r.update({"ward_icc": icc_w, "ward_grid_q975": np.quantile(gnull, .975), "ward_p_grid": null_summary(icc_w, gnull)["p"],
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
            if name == "all sources":
                nulls[aname] = (icc_a, gn, rn)
            r.update({f"{aname}_ward_icc": icc_a, f"{aname}_grid_q975": np.quantile(gn, .975),
                      f"{aname}_p_grid": null_summary(icc_a, gn)["p"], f"{aname}_region_q975": np.quantile(rn, .975),
                      f"{aname}_p_region": null_summary(icc_a, rn)["p"]})
        rows.append(r)
        logger.info(f"{name}: done")
    return pd.DataFrame(rows), nulls


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
def earthquake_adjusted(df, coords, n_boot=200):
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
    cols = explanatory_columns(df)
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


# ----------------------------------------------------------------------------- figure
def figure(nulls, vsumm):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    ax = axes[0]
    labels = {"drying": "drying (all sources)", "earthquake": "earthquake attribution (dried)",
              "dated_2072": "dated 2072 BS (dried)"}
    for i, (k, lab) in enumerate(labels.items()):
        obs, g, r = nulls[k]
        for j, (nn, col) in enumerate([(g, "#9E9E9E"), (r, "#4C72B0")]):
            ax.hlines(i + (j - 0.5) * 0.25, np.quantile(nn, .025), np.quantile(nn, .975), color=col, lw=7,
                      label=(["matched square grids", "contiguous random regions (ward sizes)"][j] if i == 0 else None))
        ax.plot(obs, i, "o", c="#C44E52", ms=8, label="wards" if i == 0 else None)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels.values())
    ax.set_xlabel("ICC within units (95% null ranges)")
    ax.set_title("(a) Ward ICC against two null partitions", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    s = vsumm[vsumm.design == "spatial block CV"]
    ax.plot(s.buffer_m, s.median_distance_m, "o-", label="median distance test → nearest training source")
    ax.plot(s.buffer_m, s.p05_distance_m, "s--", label="5th percentile")
    l = vsumm[vsumm.design != "spatial block CV"]
    ax.scatter(l.buffer_m, l.median_distance_m, marker="*", s=120, c="k", label="leave-one-municipality-out (median)")
    ax.set_xlabel("Buffer (m)")
    ax.set_ylabel("Distance (m)")
    ax.set_title("(b) Separation of test and training sources", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(FIGS / "rb_robustness.png", dpi=300)
    plt.close(fig)


def run():
    df = load_master()
    if "catch_T2p0" not in df:
        df = df.merge(pd.read_csv(HG)[["id", "catch_T2p0"]], on="id", how="left")
    coords = coords_to_metric(df.latitude.to_numpy(), df.longitude.to_numpy())
    nbrs = neighbours(coords)
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
    sub, nulls = subset_analyses(df, coords, nbrs)
    sub.to_csv(METRICS / "rb_sensitivity_subsets.csv", index=False)
    logger.info("\n" + sub.round(3).T.to_string())
    figure(nulls, vsumm)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()
