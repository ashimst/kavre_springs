"""
Explanatory GIS, graph and data-mining analyses of spring drying (no predictive claims).

A. Seismic dose-response   drying (all sources) and, among dried sources, earthquake attribution and
                           drying in 2072 BS vs USGS ShakeMap PGA / aftershock density
                           (standardised logistic coefficients, 95 % spatial-block bootstrap CIs).
B. Directional neighbours   on the directed proximity graph (neighbours within 300 m, |dz| > 5 m):
                           does a source's status depend more on its UPSLOPE than on its DOWNSLOPE
                           neighbours?  Statistic = beta_up - beta_down from  y_i ~ f_up,i + f_down,i;
                           null = shuffle up/down labels within each source's neighbourhood (999x).
C. Knox space-time test     among dried sources: are pairs close in space also close in reported
                           drying year?  Null = permute years (999x); with and without the 2072 BS cohort.
D. Ward concordance         at matched distance, are pairs in the same ward more concordant than pairs in
                           different wards?  (Survey-team / administrative signal; ward boundaries may
                           also follow ridges and rivers.)
E. Spatial-scale partition  Moran eigenvector maps (MEMs) of the k-NN graph (graph-Laplacian spatial
                           filters) split into broad / meso / fine scales; variation partitioning of
                           drying between environmental covariates and spatial structure
                           (adjusted McFadden pseudo-R^2).
F. Spatial scan statistic   Kulldorff Bernoulli circular scan (max 10 % of sources per window) for
                           high- and low-drying clusters, and for earthquake-attributed drying among
                           dried sources; Monte-Carlo p-values (999x).

Outputs: results/metrics/sa_*.csv, results/figures/sa_*.png.
Usage:   python -m src.spatial_analysis
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
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.spatial import cKDTree
from scipy.special import xlogy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_SETS, explanatory_columns
from src.graph_builder import coords_to_metric, load_master
from src.spatial_splits import grid_block_ids

logger = logging.getLogger("spatial_analysis")
METRICS = PROJECT_ROOT / "results/metrics"
FIGS = PROJECT_ROOT / "results/figures"
SEED = 2024
N_PERM = 999


# ----------------------------------------------------------------------------- helpers
def _logit_coef(X, y):
    return LogisticRegression(C=1e6, max_iter=5000).fit(X, y).coef_[0]


def block_bootstrap_logit(X, y, blocks, names, n_boot=300, rng=None):
    rng = rng or np.random.default_rng(SEED)
    Xs = StandardScaler().fit_transform(X)
    coef = _logit_coef(Xs, y)
    ub = np.unique(blocks)
    members = {b: np.where(blocks == b)[0] for b in ub}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([members[b] for b in rng.choice(ub, len(ub), replace=True)])
        if y[idx].min() == y[idx].max():
            continue
        boots.append(_logit_coef(Xs[idx], y[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return pd.DataFrame({"term": names, "coef_per_sd": coef, "ci_low": lo, "ci_high": hi,
                         "odds_ratio_per_sd": np.exp(coef), "ci_excludes_zero": (lo > 0) | (hi < 0)})


def drying_year(df):
    y = pd.to_numeric(df["years_since_dried"], errors="coerce")
    return y.where((y >= 2000) & (y <= 2085))


# ----------------------------------------------------------------------------- A
def seismic_dose_response(df, coords, blocks):
    y = df.target_dried.to_numpy()
    rows = []
    for col in ["pga_main", "pga_may12", "aftershocks_20km"]:
        q = pd.qcut(df[col], 5, labels=False, duplicates="drop")
        g = df.assign(q=q).groupby("q").agg(lo=(col, "min"), hi=(col, "max"), n=("id", "size"), rate=("target_dried", "mean"))
        g["variable"] = col
        rows.append(g.reset_index())
    pd.concat(rows).to_csv(METRICS / "sa_seismic_quintiles.csv", index=False)

    seis = ["pga_main", "pga_may12", "aftershocks_20km"]
    hydro_env = explanatory_columns(df)
    out = []
    X = df[seis].to_numpy(float)
    t = block_bootstrap_logit(X, y, blocks, seis); t["model"] = "drying ~ seismic"; out.append(t)
    Xf = df[seis + hydro_env].fillna(df[seis + hydro_env].median()).to_numpy(float)
    t = block_bootstrap_logit(Xf, y, blocks, seis + hydro_env); t["model"] = "drying ~ seismic + environment (no lat/lon)"
    out.append(t[t.term.isin(seis)])
    dried = df.target_dried == 1
    yr = drying_year(df)
    sub = dried & yr.notna()
    for label, target in [("earthquake-attributed | dried", df.reason_of_drying.eq("earthquake")),
                          ("dried in 2072 BS | dried", yr.eq(2072))]:
        t = block_bootstrap_logit(df.loc[sub, seis].to_numpy(float), target[sub].to_numpy().astype(int),
                                  blocks[sub.to_numpy()], seis)
        t["model"] = label
        out.append(t)
    res = pd.concat(out, ignore_index=True)
    res.to_csv(METRICS / "sa_seismic_logit.csv", index=False)
    return res


# ----------------------------------------------------------------------------- B
def directional_test(df, coords, radius=300.0, dz_min=5.0, max_nb=20, n_perm=N_PERM, rng=None):
    rng = rng or np.random.default_rng(SEED)
    y = df.target_dried.to_numpy().astype(float)
    z = df.elevation_m.fillna(df.elevation_m.median()).to_numpy()
    n = len(y)
    d, idx = cKDTree(coords).query(coords, k=max_nb + 1, distance_upper_bound=radius)
    idx, d = idx[:, 1:], d[:, 1:]
    valid = np.isfinite(d)
    idx_safe = np.where(valid, idx, 0)
    dz = z[idx_safe] - z[:, None]
    up = valid & (dz > dz_min)
    down = valid & (dz < -dz_min)
    use = valid & (up | down)  # neighbours with a clear slope relation
    keep = up.any(1) & down.any(1)
    yn = y[idx_safe]

    def stat(up_mask, down_mask):
        fu = (yn * up_mask).sum(1)[keep] / up_mask.sum(1)[keep]
        fd = (yn * down_mask).sum(1)[keep] / down_mask.sum(1)[keep]
        b = _logit_coef(StandardScaler().fit_transform(np.c_[fu, fd]), y[keep])
        return b[0] - b[1], b, fu, fd

    obs, b_obs, fu, fd = stat(up, down)
    nu = up.sum(1)
    null = []
    for _ in range(n_perm):
        r = rng.random(up.shape)
        r[~use] = np.inf
        rank = np.argsort(np.argsort(r, axis=1), axis=1)
        pu = use & (rank < nu[:, None])
        pdn = use & ~pu
        null.append(stat(pu, pdn)[0])
    null = np.array(null)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    # simple conditional rates for interpretation
    yk = y[keep]
    tab = pd.DataFrame({
        "quantity": ["n sources with up- and downslope neighbours", "P(dried | all upslope nbrs dried)",
                     "P(dried | all downslope nbrs dried)", "P(dried | no upslope nbr dried)",
                     "P(dried | no downslope nbr dried)", "beta_up (std)", "beta_down (std)",
                     "beta_up - beta_down", "permutation p (two-sided)", "null 95% interval low", "null 95% interval high"],
        "value": [keep.sum(), yk[fu == 1].mean(), yk[fd == 1].mean(), yk[fu == 0].mean(), yk[fd == 0].mean(),
                  b_obs[0], b_obs[1], obs, p, *np.percentile(null, [2.5, 97.5])]})
    tab.to_csv(METRICS / "sa_directional_test.csv", index=False)
    return tab, null, obs


# ----------------------------------------------------------------------------- C
def knox_test(df, coords, rng=None, n_perm=N_PERM):
    rng = rng or np.random.default_rng(SEED)
    yr = drying_year(df)
    rows = []
    for cohort, mask in [("all dried with valid year", (df.target_dried == 1) & yr.notna()),
                         ("excluding 2072 BS cohort", (df.target_dried == 1) & yr.notna() & yr.ne(2072))]:
        m = mask.to_numpy()
        c = coords[m]
        t = yr[m].to_numpy()
        ward = (df.municipality.astype(str) + "_" + df.ward.astype(str)).to_numpy()[m]
        tree = cKDTree(c)
        for ds in [250, 500, 1000]:
            all_pairs = tree.query_pairs(ds, output_type="ndarray")
            same_w = ward[all_pairs[:, 0]] == ward[all_pairs[:, 1]]
            # pair subsets: all, within the same ward (same survey team), across ward boundaries
            for subset, pairs in [("all pairs", all_pairs), ("same ward", all_pairs[same_w]),
                                  ("different ward", all_pairs[~same_w])]:
                if len(pairs) == 0:
                    continue
                for dt in [0, 1]:
                    obs = int((np.abs(t[pairs[:, 0]] - t[pairs[:, 1]]) <= dt).sum())
                    null = np.empty(n_perm)
                    for k in range(n_perm):
                        tp = rng.permutation(t)
                        null[k] = (np.abs(tp[pairs[:, 0]] - tp[pairs[:, 1]]) <= dt).sum()
                    rows.append({"cohort": cohort, "pairs": subset, "n_sources": int(m.sum()), "space_m": ds,
                                 "time_years": dt, "n_close_pairs": len(pairs), "observed": obs,
                                 "expected": null.mean(), "ratio_obs_exp": obs / null.mean(),
                                 "p_value": (np.sum(null >= obs) + 1) / (n_perm + 1)})
    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "sa_knox_test.csv", index=False)
    return out


# ----------------------------------------------------------------------------- D
def ward_concordance(df, coords, blocks, n_boot=300, rng=None):
    rng = rng or np.random.default_rng(SEED)
    y = df.target_dried.to_numpy()
    ward = (df.municipality.astype(str) + "_" + df.ward.astype(str)).to_numpy()
    pairs = cKDTree(coords).query_pairs(400, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    dist = np.linalg.norm(coords[i] - coords[j], axis=1)
    same = ward[i] == ward[j]
    conc = (y[i] == y[j]).astype(float)
    both_dried = ((y[i] == 1) & (y[j] == 1)).astype(float)
    pb = blocks[i]
    bins = [(0, 100), (100, 200), (200, 400)]
    rows = []
    ub = np.unique(pb)
    members = {b: np.where(pb == b)[0] for b in ub}
    for lo, hi in bins:
        sel = (dist >= lo) & (dist < hi)
        def diff(ix):
            s, dsel = ix[same[ix]], ix[~same[ix]]
            return conc[s].mean() - conc[dsel].mean()
        base = np.where(sel)[0]
        obs = diff(base)
        boots = []
        for _ in range(n_boot):
            ix = np.concatenate([members[b] for b in rng.choice(ub, len(ub), replace=True)])
            ix = ix[sel[ix]]
            if same[ix].any() and (~same[ix]).any():
                boots.append(diff(ix))
        rows.append({"distance_band_m": f"{lo}-{hi}", "n_same_ward_pairs": int((sel & same).sum()),
                     "n_diff_ward_pairs": int((sel & ~same).sum()),
                     "concordance_same_ward": conc[sel & same].mean(), "concordance_diff_ward": conc[sel & ~same].mean(),
                     "difference": obs, "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
                     "mean_distance_same": dist[sel & same].mean(), "mean_distance_diff": dist[sel & ~same].mean()})
    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "sa_ward_concordance.csv", index=False)
    return out


# ----------------------------------------------------------------------------- E
def mem_eigenvectors(coords, k=7, n_vec=250):
    n = len(coords)
    _, idx = cKDTree(coords).query(coords, k=k + 1)
    rows = np.repeat(np.arange(n), k)
    W = sp.csr_matrix((np.ones(n * k), (rows, idx[:, 1:].ravel())), shape=(n, n))
    W = ((W + W.T) > 0).astype(float)

    def mv(x):
        x = np.asarray(x).ravel()
        x = x - x.mean()
        x = W @ x
        return x - x.mean()

    op = LinearOperator((n, n), matvec=mv, dtype=float)
    vals, vecs = eigsh(op, k=n_vec, which="LA", tol=1e-6, maxiter=20000)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    keep = vals > 1e-8
    return vals[keep], vecs[:, keep]


def _adj_r2(X, y):
    if X is None or X.shape[1] == 0:
        return 0.0
    m = LogisticRegression(C=1e6, max_iter=10000).fit(X, y)
    p = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    ll = np.sum(xlogy(y, p) + xlogy(1 - y, 1 - p))
    pb = y.mean()
    ll0 = len(y) * (pb * np.log(pb) + (1 - pb) * np.log(1 - pb))
    return 1 - (ll - X.shape[1]) / ll0


def variation_partitioning(df, coords):
    y = df.target_dried.to_numpy()
    env_cols = explanatory_columns(df)
    E = StandardScaler().fit_transform(df[env_cols].fillna(df[env_cols].median()).to_numpy(float))
    vals, V = mem_eigenvectors(coords)
    n_mem = V.shape[1]
    bands = {"broad (MEM 1-25)": slice(0, 25), "meso (MEM 26-100)": slice(25, 100), "fine (MEM 101-%d)" % n_mem: slice(100, n_mem)}
    S = V[:, :n_mem]
    r = {"E": _adj_r2(E, y), "S": _adj_r2(S, y), "E+S": _adj_r2(np.c_[E, S], y)}
    frac = pd.DataFrame([
        {"fraction": "[a] environment only", "adj_pseudo_r2": r["E+S"] - r["S"]},
        {"fraction": "[b] shared (spatially structured environment)", "adj_pseudo_r2": r["E"] + r["S"] - r["E+S"]},
        {"fraction": "[c] space only (MEMs)", "adj_pseudo_r2": r["E+S"] - r["E"]},
        {"fraction": "[d] unexplained", "adj_pseudo_r2": 1 - r["E+S"]},
    ])
    scale_rows = []
    for name, sl in bands.items():
        Sb = V[:, sl]
        scale_rows.append({"scale": name, "n_mem": Sb.shape[1], "r2_space_alone": _adj_r2(Sb, y),
                           "r2_unique_beyond_env": _adj_r2(np.c_[E, Sb], y) - r["E"]})
    scales = pd.DataFrame(scale_rows)
    # characteristic spatial scale of each band: mean distance at which MEMs change sign (via Moran's I of MEM)
    summary = pd.DataFrame([{"n_mem_positive": n_mem, "r2_E": r["E"], "r2_S": r["S"], "r2_E+S": r["E+S"],
                             "n_env_covariates": len(env_cols)}])
    frac.to_csv(METRICS / "sa_variation_partitioning.csv", index=False)
    scales.to_csv(METRICS / "sa_variation_by_scale.csv", index=False)
    summary.to_csv(METRICS / "sa_variation_summary.csv", index=False)
    np.save(METRICS / "sa_mem_vectors_first10.npy", V[:, :10])
    return frac, scales, summary, V


# ----------------------------------------------------------------------------- F
def kulldorff_scan(coords, cases, max_frac=0.10, n_perm=N_PERM, high=True, n_secondary=8, rng=None):
    """Bernoulli circular scan. cases: 0/1 array. Returns clusters with Monte-Carlo p-values."""
    rng = rng or np.random.default_rng(SEED)
    N, C = len(cases), cases.sum()
    K = max(2, int(max_frac * N))
    dist, idx = cKDTree(coords).query(coords, k=K)
    n = np.arange(1, K + 1)[None, :]
    ll0 = xlogy(C, C / N) + xlogy(N - C, 1 - C / N)

    def llr(cs):
        c = cs[idx].cumsum(1)
        cin, cout, nout = c, C - c, N - n
        rin, rout = cin / n, cout / nout
        l = (xlogy(cin, rin) + xlogy(n - cin, 1 - rin) + xlogy(cout, rout) + xlogy(nout - cout, 1 - rout)) - ll0
        mask = (rin > rout) if high else (rin < rout)
        return np.where(mask, l, 0.0), c

    L, c = llr(cases)
    null_max = np.array([llr(rng.permutation(cases))[0].max() for _ in range(n_perm)])
    order = np.argsort(L, axis=None)[::-1]
    used = np.zeros(N, bool)
    clusters = []
    for flat in order:
        i, k = np.unravel_index(flat, L.shape)
        if L[i, k] <= 0 or len(clusters) >= 1 + n_secondary:
            break
        members = idx[i, :k + 1]
        if used[members].any():
            continue
        used[members] = True
        nin, cin = k + 1, c[i, k]
        clusters.append({"rank": len(clusters) + 1, "center_index": int(i), "radius_m": float(dist[i, k]), "n_sources": int(nin),
                         "cases": int(cin), "observed_rate": cin / nin, "rate_outside": (C - cin) / (N - nin),
                         "relative_risk": (cin / nin) / ((C - cin) / (N - nin)), "llr": float(L[i, k]),
                         "p_value": (np.sum(null_max >= L[i, k]) + 1) / (n_perm + 1), "members": members})
    return clusters


def run_scans(df, coords):
    y = df.target_dried.to_numpy()
    out = []
    member_map = {}
    for tag, cases, mask, high in [("dried (high)", y, np.ones(len(y), bool), True),
                                   ("dried (low)", y, np.ones(len(y), bool), False),
                                   ("earthquake-attributed among dried (high)", df.reason_of_drying.eq("earthquake").to_numpy().astype(int),
                                    (df.target_dried == 1).to_numpy(), True)]:
        cl = kulldorff_scan(coords[mask], cases[mask], high=high)
        ids = df.id.to_numpy()[mask]
        for c in cl:
            member_map[(tag, c["rank"])] = ids[c.pop("members")]
            c["center_lat"] = df.latitude.to_numpy()[mask][c["center_index"]]
            c["center_lon"] = df.longitude.to_numpy()[mask][c["center_index"]]
            c["analysis"] = tag
            out.append(c)
    res = pd.DataFrame(out)
    res.to_csv(METRICS / "sa_scan_clusters.csv", index=False)
    return res, member_map


# ----------------------------------------------------------------------------- figures
def figures(df, seis_q, seis_logit, dir_null, dir_obs, knox, ward, frac, scales, V, scans, member_map):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5), layout="constrained")
    # (a) seismic dose-response
    ax = axes[0, 0]
    for var, lab in [("pga_main", "PGA 25 Apr (main)"), ("pga_may12", "PGA 12 May (M7.3)"), ("aftershocks_20km", "Aftershocks ≤20 km")]:
        s = seis_q[seis_q.variable == var]
        ax.plot(range(1, len(s) + 1), s.rate, "o-", label=lab)
    ax.axhline(df.target_dried.mean(), ls=":", c="grey")
    ax.set_xlabel("Quintile (low → high)")
    ax.set_ylabel("Share of sources dried")
    ax.set_title("(a) Drying rate by 2015 shaking / aftershock quintile", fontsize=10)
    ax.legend(fontsize=8)
    # (b) directional test
    ax = axes[0, 1]
    ax.hist(dir_null, bins=40, color="#bbbbbb")
    ax.axvline(dir_obs, c="crimson", lw=2, label=f"observed = {dir_obs:+.3f}")
    ax.set_xlabel("β(upslope neighbours) − β(downslope neighbours)")
    ax.set_title("(b) Directional dependence vs permutation null", fontsize=10)
    ax.legend(fontsize=8)
    # (c) Knox ratios
    ax = axes[0, 2]
    for (cohort, subset), mk in [(("excluding 2072 BS cohort", "same ward"), "o-"),
                                 (("excluding 2072 BS cohort", "different ward"), "s--"),
                                 (("all dried with valid year", "all pairs"), "^:")]:
        s = knox[(knox.cohort == cohort) & (knox.pairs == subset) & (knox.time_years == 0)]
        ax.plot(s.space_m, s.ratio_obs_exp, mk, label=f"{cohort}; {subset}")
    ax.axhline(1, c="k", lw=0.8)
    ax.set_xlabel("Spatial threshold (m)")
    ax.set_ylabel("Knox observed / expected (same reported year)")
    ax.set_title("(c) Knox space-time test of drying years", fontsize=10)
    ax.legend(fontsize=8)
    # (d) ward concordance
    ax = axes[1, 0]
    x = np.arange(len(ward))
    ax.bar(x - 0.2, ward.concordance_same_ward, 0.4, label="same ward")
    ax.bar(x + 0.2, ward.concordance_diff_ward, 0.4, label="different ward")
    ax.set_xticks(x)
    ax.set_xticklabels(ward.distance_band_m + " m")
    ax.set_ylim(0.5, 0.9)
    ax.set_ylabel("Pair concordance (same status)")
    ax.set_title("(d) Concordance of nearby pairs by ward", fontsize=10)
    ax.legend(fontsize=8)
    # (e) variation partitioning
    ax = axes[1, 1]
    f = frac[frac.fraction != "[d] unexplained"]
    ax.barh([t.split('] ')[-1].split(' (')[0] for t in f.fraction], f.adj_pseudo_r2, color=["#4C72B0", "#8172B3", "#55A868"])
    for _, r in scales.iterrows():
        pass
    ax.set_xlabel("Adjusted McFadden pseudo-R²")
    ax.set_title("(e) Variation partitioning: environment vs space (MEMs)", fontsize=10)
    txt = "\n".join(f"{r.scale}: unique {r.r2_unique_beyond_env:.3f}" for _, r in scales.iterrows())
    ax.text(0.98, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    # (f) scan clusters map
    ax = axes[1, 2]
    ax.scatter(df.longitude, df.latitude, s=1.5, c="#d0d0d0", linewidths=0)
    idmap = pd.Series(np.arange(len(df)), index=df.id)
    sig = scans[scans.p_value < 0.05]
    colors = {"dried (high)": "#C44E52", "dried (low)": "#4C72B0", "earthquake-attributed among dried (high)": "#DD8452"}
    for _, r in sig.iterrows():
        mem = idmap[member_map[(r.analysis, r["rank"])]].to_numpy()
        ax.scatter(df.longitude.to_numpy()[mem], df.latitude.to_numpy()[mem], s=3, c=colors[r.analysis], linewidths=0)
    for a, col in colors.items():
        ax.scatter([], [], c=col, s=20, label=f"{a} (n={int((sig.analysis == a).sum())})")
    ax.set_aspect(1 / np.cos(np.radians(27.57)))
    ax.set_title("(f) Significant scan-statistic clusters (p<0.05)", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    pass  # constrained layout
    fig.savefig(FIGS / "sa_spatial_data_mining.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for k, ax in zip([0, 4, 24, 99], axes):
        sc = ax.scatter(df.longitude, df.latitude, c=V[:, k] if k < V.shape[1] else 0, s=1.5, cmap="RdBu_r", linewidths=0)
        ax.set_title(f"MEM {k + 1}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect(1 / np.cos(np.radians(27.57)))
    fig.suptitle("Moran eigenvector maps of the spring graph: broad → fine spatial patterns", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "sa_mem_examples.png", dpi=200)
    plt.close(fig)


def scan_ward_alignment(df, coords, scans, n_null=500):
    """Are significant scan clusters concentrated in one ward more than circular windows of the same size placed at
    random? Per cluster: majority municipality, wards spanned, dominant-ward share and its percentile among random
    windows of the same number of sources."""
    rng = np.random.default_rng(SEED)
    ward = (df.municipality.astype(str) + "_" + df.ward.astype(str)).to_numpy()
    muni = df.municipality.astype(str).to_numpy()
    y = df.target_dried.to_numpy()
    masks = {"dried (high)": np.ones(len(y), bool), "dried (low)": np.ones(len(y), bool),
             "earthquake-attributed among dried (high)": y == 1}

    def dominant(idx):
        _, cnt = np.unique(ward[idx], return_counts=True)
        return cnt.max() / len(idx), len(cnt)

    rows = []
    for _, r in scans[scans.p_value < 0.05].iterrows():
        m = np.where(masks[r.analysis])[0]
        tree = cKDTree(coords[m])
        mem = m[np.atleast_1d(tree.query(coords[m][int(r.center_index)], k=int(r.n_sources))[1])]
        obs, n_w = dominant(mem)
        null = np.array([dominant(m[np.atleast_1d(tree.query(coords[m][i], k=int(r.n_sources))[1])])[0]
                         for i in rng.choice(len(m), n_null)])
        rows.append({"analysis": r.analysis, "rank": r["rank"], "n_sources": int(r.n_sources),
                     "relative_risk": r.relative_risk, "majority_municipality": pd.Series(muni[mem]).mode()[0],
                     "n_wards_spanned": n_w, "dominant_ward_share": obs, "null_mean_share": null.mean(),
                     "percentile_vs_random_windows": (null < obs).mean()})
    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "sa_scan_ward_alignment.csv", index=False)
    return out


def figures_from_saved():
    """Rebuild both figures from the saved CSVs. Scan-cluster members are recovered exactly from each cluster's
    centre and size (a circular-window cluster is the n_sources nearest sources to its centre); the cheap
    directional permutation null and the MEMs are recomputed with the same seeds."""
    df = load_master()
    coords = coords_to_metric(df.latitude.to_numpy(), df.longitude.to_numpy())
    seis_q = pd.read_csv(METRICS / "sa_seismic_quintiles.csv")
    seis_logit = pd.read_csv(METRICS / "sa_seismic_logit.csv")
    _, dir_null, dir_obs = directional_test(df, coords)
    knox = pd.read_csv(METRICS / "sa_knox_test.csv")
    ward = pd.read_csv(METRICS / "sa_ward_concordance.csv")
    frac = pd.read_csv(METRICS / "sa_variation_partitioning.csv")
    scales = pd.read_csv(METRICS / "sa_variation_by_scale.csv")
    _, V = mem_eigenvectors(coords)
    scans = pd.read_csv(METRICS / "sa_scan_clusters.csv")
    y = df.target_dried.to_numpy()
    masks = {"dried (high)": np.ones(len(y), bool), "dried (low)": np.ones(len(y), bool),
             "earthquake-attributed among dried (high)": y == 1}
    member_map = {}
    for _, r in scans.iterrows():
        m = masks[r.analysis]
        _, nb = cKDTree(coords[m]).query(coords[m][int(r.center_index)], k=int(r.n_sources))
        member_map[(r.analysis, r["rank"])] = df.id.to_numpy()[m][np.atleast_1d(nb)]
    figures(df, seis_q, seis_logit, dir_null, dir_obs, knox, ward, frac, scales, V, scans, member_map)


def run_all():
    METRICS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    df = load_master()
    coords = coords_to_metric(df.latitude.to_numpy(), df.longitude.to_numpy())
    blocks = grid_block_ids(coords, 8)
    logger.info("A. seismic dose-response")
    seis_logit = seismic_dose_response(df, coords, blocks)
    seis_q = pd.read_csv(METRICS / "sa_seismic_quintiles.csv")
    logger.info("\n" + seis_logit.round(3).to_string())
    logger.info("B. directional neighbour test")
    dir_tab, dir_null, dir_obs = directional_test(df, coords)
    logger.info("\n" + dir_tab.to_string())
    logger.info("C. Knox test")
    knox = knox_test(df, coords)
    logger.info("\n" + knox.round(3).to_string())
    logger.info("D. ward concordance")
    ward = ward_concordance(df, coords, blocks)
    logger.info("\n" + ward.round(3).to_string())
    logger.info("E. variation partitioning with MEMs")
    frac, scales, summ, V = variation_partitioning(df, coords)
    logger.info("\n" + frac.round(4).to_string() + "\n" + scales.round(4).to_string() + "\n" + summ.round(4).to_string())
    logger.info("F. spatial scan statistics")
    scans, member_map = run_scans(df, coords)
    logger.info("\n" + scans.drop(columns=["center_index"]).round(3).to_string())
    logger.info("\n" + scan_ward_alignment(df, coords, scans).round(3).to_string())
    figures(df, seis_q, seis_logit, dir_null, dir_obs, knox, ward, frac, scales, V, scans, member_map)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["knox", "figures", "scanward"],
                    help="knox: re-run the Knox test only; figures: rebuild figures from saved CSVs (no scan re-run)")
    a = ap.parse_args()
    if a.only == "figures":
        figures_from_saved()
    elif a.only == "scanward":
        d = load_master()
        print(scan_ward_alignment(d, coords_to_metric(d.latitude.to_numpy(), d.longitude.to_numpy()),
                                  pd.read_csv(METRICS / "sa_scan_clusters.csv")).round(3).to_string())
    elif a.only == "knox":
        d = load_master()
        print(knox_test(d, coords_to_metric(d.latitude.to_numpy(), d.longitude.to_numpy())).round(3).to_string())
    else:
        run_all()
