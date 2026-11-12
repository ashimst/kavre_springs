"""
Spatial analyses of spring drying (no predictive claims).

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
E. Moran eigenvector maps   (MEMs) of the k-NN graph, used by the three-way partition (src/drivers_analysis.py) and
                           the earthquake adjustment (src/robustness.py).

Outputs: results/metrics/sa_*.csv.
Usage:   python -m src.spatial_analysis
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.spatial import cKDTree
from scipy.special import xlogy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features import explanatory_columns
from src.dataset import coords_to_metric, load_master
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
    t = block_bootstrap_logit(X, y, blocks, seis)
    t["model"] = "drying ~ seismic"
    out.append(t)
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
    # fixed start vector: ARPACK's default random start depends on earlier calls in the same process
    v0 = np.random.default_rng(SEED).standard_normal(n)
    vals, vecs = eigsh(op, k=n_vec, which="LA", tol=1e-6, maxiter=20000, v0=v0)
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


# ----------------------------------------------------------------------------- F
# ----------------------------------------------------------------------------- figures
def run_all():
    METRICS.mkdir(parents=True, exist_ok=True)
    df = load_master()
    coords = coords_to_metric(df.latitude.to_numpy(), df.longitude.to_numpy())
    blocks = grid_block_ids(coords, 8)
    logger.info("A. seismic dose-response")
    logger.info("\n" + seismic_dose_response(df, coords, blocks).round(3).to_string())
    logger.info("B. directional neighbour test")
    logger.info("\n" + directional_test(df, coords)[0].to_string())
    logger.info("C. Knox test")
    logger.info("\n" + knox_test(df, coords).round(3).to_string())
    logger.info("D. ward concordance")
    logger.info("\n" + ward_concordance(df, coords, blocks).round(3).to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_all()
