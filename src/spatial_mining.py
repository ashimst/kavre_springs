"""
Spatial structure of spring drying (label only -- no models).

1. Global Moran's I of the drying indicator for k-NN weights, k in
   {3, 5, 7, 10, 15, 20} (row-standardised binary weights), with 999-permutation
   pseudo p-values and the mean neighbour distance for each k (distance decay).
2. bh_fdr: Benjamini-Hochberg q-values (shared helper).
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
from scipy.spatial import cKDTree

from src.dataset import coords_to_metric, load_master

logger = logging.getLogger("spatial_mining")
METRICS_DIR = PROJECT_ROOT / "results/metrics"
FIG_DIR = PROJECT_ROOT / "results/figures"


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
    def moran(v):
        return (n / S0) * (v @ (W @ v)) / (v @ v)

    obs = moran(z)
    sims = np.array([moran(rng.permutation(z)) for _ in range(n_perm)])
    p = (np.sum(sims >= obs) + 1) / (n_perm + 1)
    return obs, p, (obs - sims.mean()) / sims.std()


QUADRANTS = ["High-High (dried cluster)", "Low-Low (active cluster)",
             "Low-High (active among dried)", "High-Low (dried among active)"]


def run_spatial_mining():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
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

    return decay
