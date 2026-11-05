"""
Spatial structure of spring drying (label only -- no models).

1. Global Moran's I of the drying indicator for k-NN weights, k in
   {3, 5, 7, 10, 15, 20} (row-standardised binary weights), with 999-permutation
   pseudo p-values and the mean neighbour distance for each k (distance decay).
2. Local Moran's I (LISA, k = 7) with 999 conditional permutations and
   Benjamini-Hochberg FDR control across the 5,689 local tests.
3. Association rules (1- and 2-item antecedents -> dried) with Fisher's exact
   test and BH-FDR over *all* candidate rules evaluated. Survey fields subject
   to skip logic (e.g. earthquake impact) are excluded; the seasonality flag is
   kept but its rules are flagged because survey timing may confound it.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import itertools
import logging
import os

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.spatial import cKDTree

from src.graph_builder import coords_to_metric, load_master

logger = logging.getLogger("spatial_mining")
METRICS_DIR = PROJECT_ROOT / "results/metrics"
FIG_DIR = PROJECT_ROOT / "results/figures"
PAPER_FIG_DIR = PROJECT_ROOT / "paper/figures"


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def knn_weights(coords_m: np.ndarray, k: int):
    d, idx = cKDTree(coords_m).query(coords_m, k=k + 1)
    n = len(coords_m)
    rows = np.repeat(np.arange(n), k)
    W = sp.csr_matrix((np.full(n * k, 1.0 / k), (rows, idx[:, 1:].ravel())), shape=(n, n))
    return W, float(d[:, 1:].mean()), idx[:, 1:]


def morans_i(y: np.ndarray, W: sp.csr_matrix, n_perm: int = 999, rng=None):
    rng = rng or np.random.default_rng(0)
    z = y - y.mean()
    n, S0 = len(y), W.sum()
    I = lambda v: (n / S0) * (v @ (W @ v)) / (v @ v)
    obs = I(z)
    sims = np.array([I(rng.permutation(z)) for _ in range(n_perm)])
    p = (np.sum(sims >= obs) + 1) / (n_perm + 1)
    return obs, p, (obs - sims.mean()) / sims.std()


QUADRANTS = ["High-High (dried cluster)", "Low-Low (active cluster)",
             "Low-High (active among dried)", "High-Low (dried among active)"]


def quad_label(y: np.ndarray, W: sp.csr_matrix) -> np.ndarray:
    z = y - y.mean()
    lag = W @ z
    return np.select([(z > 0) & (lag > 0), (z < 0) & (lag < 0), (z < 0) & (lag > 0), (z > 0) & (lag < 0)],
                     QUADRANTS, default="Not significant").astype(object)


def lisa(y: np.ndarray, W: sp.csr_matrix, nbr_idx: np.ndarray, n_perm: int = 999, rng=None):
    rng = rng or np.random.default_rng(0)
    n, k = nbr_idx.shape
    z = (y - y.mean()) / y.std()
    lag = W @ z
    Ii = z * lag
    extreme = np.zeros(n)
    for _ in range(n_perm):
        draw = rng.integers(0, n - 1, size=(n, k))
        draw = draw + (draw >= np.arange(n)[:, None])  # conditional: never draw node i itself
        sim = z * z[draw].mean(1)
        extreme += np.where(Ii >= 0, sim >= Ii, sim <= Ii)
    p = (extreme + 1) / (n_perm + 1)
    q = bh_fdr(p)
    quad = np.where(q < 0.05, quad_label(y, W), "Not significant").astype(object)
    return Ii, p, q, quad


def association_rules(df: pd.DataFrame, min_support: float = 0.03):
    y = df["target_dried"].to_numpy().astype(bool)
    base = y.mean()
    conds = {
        "Seasonal spring [timing-confounded]": df["Perennial / Seasonal_encoded"] == 1,
        "Elevation > 1600 m": df["elevation_m"] > 1600,
        "Elevation < 1100 m": df["elevation_m"] < 1100,
        "Slope > 25 deg": df["slope_deg"] > 25,
        "Dry-season precip. < 70 mm": df["precip_dry_season_mm"] < 70,
        "NDWI < 0": df["ndwi_mean"] < 0.0,
        "Spring density > 10 per 500 m": df["spring_density_500m"] > 10,
        "Source type: pond": df["Source Type_encoded"] == 1,
        "Land use: dense forest": df["landuse_dense_forest"] == 1,
        "Land use: settlement": df["landuse_settlement"] == 1,
        "Land use: agriculture": df["landuse_agriculture"] == 1,
    }
    conds = {k: v.to_numpy() for k, v in conds.items()}
    cands = [((a,), conds[a]) for a in conds] + [((a, b), conds[a] & conds[b]) for a, b in itertools.combinations(conds, 2)]
    rows = []
    for names, m in cands:
        if m.mean() < min_support:
            continue
        a, b = int((m & y).sum()), int((m & ~y).sum())
        c, d = int((~m & y).sum()), int((~m & ~y).sum())
        odds, p = stats.fisher_exact([[a, b], [c, d]])
        conf = a / (a + b)
        rows.append({"antecedent": " AND ".join(names), "n_antecedent": a + b, "n_dried": a,
                     "support": a / len(y), "confidence": conf, "lift": conf / base, "odds_ratio": odds, "p_value": p})
    out = pd.DataFrame(rows)
    out["q_value_bh"] = bh_fdr(out.p_value.to_numpy())
    return out.sort_values("lift", ascending=False).reset_index(drop=True)


def run_spatial_mining():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_master()
    y = df["target_dried"].to_numpy(float)
    coords = coords_to_metric(df.latitude.to_numpy(), df.longitude.to_numpy())
    rng = np.random.default_rng(2024)

    decay = []
    for k in [3, 5, 7, 10, 15, 20]:
        W, mean_d, _ = knn_weights(coords, k)
        I, p, zs = morans_i(y, W, rng=rng)
        decay.append({"k_neighbors": k, "mean_neighbor_distance_m": mean_d, "morans_i": I, "z_score": zs, "p_value": p})
        logger.info(f"k={k:2d}  mean d={mean_d:6.0f} m  I={I:.3f}  z={zs:.1f}")
    decay = pd.DataFrame(decay)
    decay.to_csv(METRICS_DIR / "spatial_autocorrelation_distance_decay.csv", index=False)

    W7, _, nbr7 = knn_weights(coords, 7)
    Ii, p, q, quad = lisa(y, W7, nbr7, rng=rng)
    out = df[["id", "municipality", "latitude", "longitude", "target_dried"]].copy()
    out["local_i"], out["p_value"], out["q_value_bh"], out["lisa_cluster"] = Ii, p, q, quad
    out.to_csv(METRICS_DIR / "lisa_spatial_clusters.csv", index=False)
    summary = out.lisa_cluster.value_counts().rename_axis("cluster").reset_index(name="n_springs_fdr_q<0.05")
    raw = np.where(p < 0.05, quad_label(y, W7), "Not significant")
    summary["n_springs_uncorrected_p<0.05"] = summary.cluster.map(pd.Series(raw).value_counts()).fillna(0).astype(int)
    summary.to_csv(METRICS_DIR / "lisa_cluster_summary.csv", index=False)
    logger.info("LISA (BH-FDR q<0.05):\n" + summary.to_string())

    rules = association_rules(df)
    rules.to_csv(METRICS_DIR / "association_rules_desiccation.csv", index=False)

    # Figures
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {"Not significant": "#d9d9d9", "Low-Low (active cluster)": "#1a9850", "High-High (dried cluster)": "#d73027",
              "Low-High (active among dried)": "#fdae61", "High-Low (dried among active)": "#7b3294"}
    for c in colors:
        m = out.lisa_cluster == c
        ax.scatter(out.longitude[m], out.latitude[m], s=6 if c == "Not significant" else 10, c=colors[c],
                   label=f"{c} (n={m.sum()})", linewidths=0)
    I7 = decay.loc[decay.k_neighbors == 7, "morans_i"].item()
    ax.set_title(f"LISA clusters of spring drying (k=7, BH-FDR q<0.05)\nGlobal Moran's I = {I7:.3f}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(fontsize=8, loc="lower left", markerscale=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lisa_cluster_map.png", dpi=300)
    if PAPER_FIG_DIR.exists():
        fig.savefig(PAPER_FIG_DIR / "lisa_cluster_map.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(decay.mean_neighbor_distance_m, decay.morans_i, "o-")
    for _, r in decay.iterrows():
        axes[0].annotate(f"k={int(r.k_neighbors)}", (r.mean_neighbor_distance_m, r.morans_i), fontsize=8,
                         textcoords="offset points", xytext=(4, 4))
    axes[0].set_xlabel("Mean neighbour distance (m)")
    axes[0].set_ylabel("Global Moran's I (drying)")
    axes[0].set_title("(a) Distance decay of spatial autocorrelation")
    sig = rules[rules.q_value_bh < 0.05].head(10)[::-1]
    axes[1].barh(sig.antecedent, sig.lift, color="#4C72B0")
    axes[1].axvline(1, c="k", lw=0.8)
    axes[1].set_xlabel("Lift (P(dried | A) / P(dried))")
    axes[1].set_title("(b) Top association rules (BH q < 0.05)")
    axes[1].tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "spatial_mining_diagnostics.png", dpi=300)
    if PAPER_FIG_DIR.exists():
        fig.savefig(PAPER_FIG_DIR / "spatial_mining_diagnostics.png", dpi=300)
    plt.close(fig)
    return decay, summary, rules


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_spatial_mining()
