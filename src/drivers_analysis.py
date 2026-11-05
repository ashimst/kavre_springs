"""
Drivers of drying that survive survey-team structure.

D1. Standardised logistic coefficients of env_hydro covariates (without latitude/longitude), pooled vs
    with ward fixed effects (75 wards = 75 survey teams), with 95 % spatial-block bootstrap CIs.
    Associations that persist within wards cannot be explained by between-team differences in how
    "dried" was recorded or by any other ward-level factor.
D2. Three-way variation partitioning of drying (adjusted McFadden pseudo-R^2) between
    E = environment (env_hydro without lat/lon), S = spatial structure (positive Moran eigenvector
    maps of the k-NN graph) and W = ward membership.

Outputs: results/metrics/dr_*.csv, results/figures/dr_drivers.png
Usage:   python -m src.drivers_analysis
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_SETS, explanatory_columns
from src.graph_builder import coords_to_metric, load_master
from src.spatial_analysis import _adj_r2, mem_eigenvectors
from src.spatial_splits import grid_block_ids

logger = logging.getLogger("drivers")
METRICS = PROJECT_ROOT / "results/metrics"
FIGS = PROJECT_ROOT / "results/figures"
SEED = 11


def _fit(X, y):
    # mild ridge keeps 75 ward dummies identifiable in small bootstrap resamples; same penalty for all models
    return LogisticRegression(C=10.0, max_iter=10000).fit(X, y).coef_[0]


def pooled_vs_within(df, blocks, n_boot=300):
    rng = np.random.default_rng(SEED)
    y = df.target_dried.to_numpy()
    cols = explanatory_columns(df)
    E = StandardScaler().fit_transform(df[cols].fillna(df[cols].median()).to_numpy(float))
    ward = (df.municipality.astype(str) + "_" + df.ward.astype(str))
    Wd = pd.get_dummies(ward, drop_first=True, dtype=float).to_numpy()
    ub = np.unique(blocks)
    members = {b: np.where(blocks == b)[0] for b in ub}
    res = []
    for label, X in [("pooled", E), ("within ward (ward fixed effects)", np.c_[E, Wd])]:
        coef = _fit(X, y)[: len(cols)]
        boots = []
        for _ in range(n_boot):
            idx = np.concatenate([members[b] for b in rng.choice(ub, len(ub), replace=True)])
            boots.append(_fit(X[idx], y[idx])[: len(cols)])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        res.append(pd.DataFrame({"model": label, "term": cols, "coef_per_sd": coef, "ci_low": lo, "ci_high": hi,
                                 "ci_excludes_zero": (lo > 0) | (hi < 0)}))
    out = pd.concat(res, ignore_index=True)
    out.to_csv(METRICS / "dr_pooled_vs_within_ward.csv", index=False)
    return out


def three_way_partition(df, coords):
    y = df.target_dried.to_numpy()
    cols = explanatory_columns(df)
    E = StandardScaler().fit_transform(df[cols].fillna(df[cols].median()).to_numpy(float))
    _, S = mem_eigenvectors(coords)
    W = pd.get_dummies(df.municipality.astype(str) + "_" + df.ward.astype(str), drop_first=True, dtype=float).to_numpy()
    R = {}
    for key, mats in {"E": [E], "S": [S], "W": [W], "ES": [E, S], "EW": [E, W], "SW": [S, W], "ESW": [E, S, W]}.items():
        R[key] = _adj_r2(np.hstack(mats), y)
    frac = {
        "environment only": R["ESW"] - R["SW"],
        "space only": R["ESW"] - R["EW"],
        "ward only": R["ESW"] - R["ES"],
        "environment & space": R["EW"] + R["SW"] - R["W"] - R["ESW"],
        "space & ward": R["ES"] + R["EW"] - R["E"] - R["ESW"],
        "environment & ward": R["ES"] + R["SW"] - R["S"] - R["ESW"],
        "all three": R["E"] + R["S"] + R["W"] - R["ES"] - R["EW"] - R["SW"] + R["ESW"],
        "unexplained": 1 - R["ESW"],
    }
    out = pd.DataFrame({"fraction": list(frac), "adj_pseudo_r2": list(frac.values())})
    out.to_csv(METRICS / "dr_three_way_partition.csv", index=False)
    pd.DataFrame([R]).to_csv(METRICS / "dr_three_way_r2.csv", index=False)
    return out, R


def figure(coefs, part):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    p = coefs[coefs.model == "pooled"].set_index("term")
    w = coefs[coefs.model != "pooled"].set_index("term")
    order = p.coef_per_sd.abs().sort_values().index
    yy = np.arange(len(order))
    for off, t, col, lab in [(-0.18, p, "#8C8C8C", "pooled"), (0.18, w, "#C44E52", "within ward")]:
        t = t.loc[order]
        ax.errorbar(t.coef_per_sd, yy + off, xerr=[t.coef_per_sd - t.ci_low, t.ci_high - t.coef_per_sd], fmt="o",
                    ms=3, color=col, capsize=2, label=lab)
    ax.axvline(0, c="k", lw=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels(order, fontsize=7)
    ax.set_xlabel("Standardised log-odds (95% spatial-block bootstrap CI)")
    ax.set_title("(a) Associations with drying: pooled vs within ward", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    q = part[part.fraction != "unexplained"]
    ax.barh(q.fraction, q.adj_pseudo_r2, color="#4C72B0")
    ax.axvline(0, c="k", lw=0.8)
    ax.set_xlabel("Adjusted McFadden pseudo-R²")
    ax.set_title(f"(b) Variation partitioning (unexplained {part.set_index('fraction').adj_pseudo_r2['unexplained']:.2f})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "dr_drivers.png", dpi=300)
    plt.close(fig)


def run():
    df = load_master()
    coords = coords_to_metric(df.latitude.to_numpy(), df.longitude.to_numpy())
    blocks = grid_block_ids(coords, 8)
    coefs = pooled_vs_within(df, blocks)
    logger.info("\n" + coefs.round(3).to_string())
    part, R = three_way_partition(df, coords)
    logger.info("\n" + part.round(4).to_string() + "\n" + str({k: round(v, 4) for k, v in R.items()}))
    figure(coefs, part)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()
