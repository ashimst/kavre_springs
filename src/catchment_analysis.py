"""
Are spring drying patterns organised by hydrological units?

G1. Intraclass correlation (ICC(1), one-way ANOVA on the binary outcome) of drying within
    DEM-derived link catchments (T = 0.5, 2, 10 km^2), within wards, and within square grid
    cells. For each catchment/ward scale, the grid cell size is tuned so that the number of
    occupied cells matches the number of occupied units, and 200 randomly shifted grid origins
    give a null distribution. A unit system that organises drying better than arbitrary squares
    of the same granularity has ICC above that distribution.
G2. Pair concordance at matched distance (100-400 m), cross-classified by same/different
    catchment (T = 2 km^2) and same/different ward. Logistic model on pairs:
        concordant ~ same_catchment + same_ward + log(distance)
    with 95 % CIs from a spatial-block bootstrap (pairs resampled by block of their midpoint).
G3. Flow-path covariates (distance along the D8 path to the stream, Strahler order of the
    receiving stream, contributing area): standardised logistic coefficients for drying,
    adjusted for env_hydro covariates, with spatial-block bootstrap CIs.

Requires data/processed/hydro_gis_features.csv (python -m src.hydro_gis).
Outputs: results/metrics/ca_*.csv (figures: src/figures.py)
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
from scipy.optimize import brentq
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression

from src.features import explanatory_columns
from src.dataset import coords_to_metric, load_master
from src.spatial_analysis import block_bootstrap_logit
from src.spatial_splits import grid_block_ids

logger = logging.getLogger("catchment_analysis")
METRICS = PROJECT_ROOT / "results/metrics"
FIGS = PROJECT_ROOT / "results/figures"
HG = PROJECT_ROOT / "data/processed/hydro_gis_features.csv"
SEED = 7


def icc1(y: np.ndarray, groups: np.ndarray) -> float:
    """Return the one-way random-effects ANOVA ICC(1), including unequal group sizes.

    The estimator is (MS_B - MS_W) / (MS_B + (n0 - 1) MS_W), where
    MS_B and MS_W are the between- and within-group mean squares and
    n0 = (N - sum(n_j**2) / N) / (G - 1). It is an observed-scale
    variance decomposition, not a latent-scale logistic ICC.
    """
    s = pd.DataFrame({"y": y, "g": groups}).groupby("g").y.agg(["mean", "size"])
    N, G, ybar = len(y), len(s), y.mean()
    if G < 2:
        return np.nan
    msb = (s["size"] * (s["mean"] - ybar) ** 2).sum() / (G - 1)
    gm = pd.Series(groups).map(s["mean"]).to_numpy()
    msw = ((y - gm) ** 2).sum() / (N - G)
    n0 = (N - (s["size"] ** 2).sum() / N) / (G - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw)


def grid_labels(coords, cell, ox, oy):
    return (np.floor((coords[:, 0] - ox) / cell).astype(np.int64) * 100000
            + np.floor((coords[:, 1] - oy) / cell).astype(np.int64))


def matched_grid_null(coords, y, n_units, n_shift=200, rng=None):
    rng = rng or np.random.default_rng(SEED)
    def occupied_cells_minus_target(cell):
        return len(np.unique(grid_labels(coords, cell, 0.0, 0.0))) - n_units

    cell = brentq(occupied_cells_minus_target, 50, 40000, xtol=1.0)
    iccs, counts = [], []
    for _ in range(n_shift):
        ox, oy = rng.uniform(0, cell, 2)
        g = grid_labels(coords, cell, ox, oy)
        iccs.append(icc1(y, g))
        counts.append(len(np.unique(g)))
    return cell, np.array(iccs), np.array(counts)


def unit_icc_table(df, coords):
    y = df.target_dried.to_numpy().astype(float)
    units = {
        "catchment T=0.5 km2": df.catch_T0p5.to_numpy(),
        "catchment T=2 km2": df.catch_T2p0.to_numpy(),
        "catchment T=10 km2": df.catch_T10p0.to_numpy(),
        "ward": (df.municipality.astype(str) + "_" + df.ward.astype(str)).to_numpy(),
        "municipality": df.municipality.to_numpy(),
    }
    rows = []
    for name, g in units.items():
        n_units = len(np.unique(g))
        obs = icc1(y, g)
        cell, null, counts = matched_grid_null(coords, y, n_units)
        rows.append({"unit": name, "n_units": n_units, "mean_sources_per_unit": len(y) / n_units, "icc": obs,
                     "grid_cell_km": cell / 1000, "grid_icc_mean": null.mean(), "grid_icc_q025": np.quantile(null, 0.025),
                     "grid_icc_q975": np.quantile(null, 0.975), "p_above_grid": (np.sum(null >= obs) + 1) / (len(null) + 1),
                     "grid_units_mean": counts.mean()})
    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "ca_unit_icc.csv", index=False)
    return out


def pair_concordance(df, coords, blocks, d_lo=100, d_hi=400, n_boot=300, rng=None):
    rng = rng or np.random.default_rng(SEED)
    y = df.target_dried.to_numpy()
    ward = (df.municipality.astype(str) + "_" + df.ward.astype(str)).to_numpy()
    catch = df.catch_T2p0.to_numpy()
    pairs = cKDTree(coords).query_pairs(d_hi, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    dist = np.linalg.norm(coords[i] - coords[j], axis=1)
    keep = dist >= d_lo
    i, j, dist = i[keep], j[keep], dist[keep]
    sc = (catch[i] == catch[j]).astype(int)
    sw = (ward[i] == ward[j]).astype(int)
    conc = (y[i] == y[j]).astype(int)
    tab = (pd.DataFrame({"same_catchment": sc, "same_ward": sw, "conc": conc, "dist": dist})
           .groupby(["same_catchment", "same_ward"]).agg(n_pairs=("conc", "size"), concordance=("conc", "mean"),
                                                         mean_distance_m=("dist", "mean")).reset_index())
    tab.to_csv(METRICS / "ca_pair_concordance_table.csv", index=False)
    X = np.c_[sc, sw, np.log(dist)]
    names = ["same catchment (T=2 km2)", "same ward", "log distance"]
    pb = blocks[i]
    def fit(ix):
        return LogisticRegression(C=1e6, max_iter=2000).fit(X[ix], conc[ix]).coef_[0]

    coef = fit(np.arange(len(conc)))
    ub = np.unique(pb)
    members = {b: np.where(pb == b)[0] for b in ub}
    boots = np.array([fit(np.concatenate([members[b] for b in rng.choice(ub, len(ub), replace=True)]))
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    res = pd.DataFrame({"term": names, "log_odds": coef, "ci_low": lo, "ci_high": hi, "odds_ratio": np.exp(coef)})
    res.to_csv(METRICS / "ca_pair_concordance_logit.csv", index=False)
    # the draws are kept so that contrasts between terms can be formed from the same resamples (X12)
    pd.DataFrame(boots, columns=names).to_csv(METRICS / "ca_pair_concordance_boot.csv", index=False)
    return tab, res


def flowpath_covariates(df, blocks):
    y = df.target_dried.to_numpy()
    df = df.assign(log_flowlen_T2=np.log1p(df.flowlen_to_stream_T2p0_m), log_contrib_area=np.log10(df.contrib_area_km2))
    fp = ["log_flowlen_T2", "strahler_T2p0", "log_contrib_area"]
    adj = explanatory_columns(df)
    out = []
    t = block_bootstrap_logit(df[fp].to_numpy(float), y, blocks, fp)
    t["model"] = "drying ~ flow-path"
    out.append(t)
    X = df[fp + adj].fillna(df[fp + adj].median()).to_numpy(float)
    t = block_bootstrap_logit(X, y, blocks, fp + adj); t["model"] = "drying ~ flow-path + environment (no lat/lon)"
    out.append(t[t.term.isin(fp)])
    res = pd.concat(out, ignore_index=True)
    res.to_csv(METRICS / "ca_flowpath_logit.csv", index=False)
    return res


def run_all():
    df = load_master().merge(pd.read_csv(HG), on="id", how="left")
    coords = coords_to_metric(df.latitude.to_numpy(), df.longitude.to_numpy())
    blocks = grid_block_ids(coords, 8)
    icc = unit_icc_table(df, coords)
    logger.info("\n" + icc.round(4).to_string())
    tab, res = pair_concordance(df, coords, blocks)
    logger.info("\n" + tab.round(4).to_string() + "\n" + res.round(4).to_string())
    fp = flowpath_covariates(df, blocks)
    logger.info("\n" + fp.round(3).to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_all()
