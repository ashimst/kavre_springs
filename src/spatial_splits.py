"""
Spatial cross-validation splits.

1. Spatial block k-fold: the study area (~40 x 39 km) is cut into an
   ``n_blocks_side x n_blocks_side`` grid (default 5 x 5, ~8 km cells). Non-empty
   blocks are shuffled (seeded) and dealt into k folds. For fold i the test set
   is block-group i and the validation set (early stopping, threshold choice)
   is block-group i+1.

   ``buffer_m`` removes from train *and* validation every spring lying within
   ``buffer_m`` metres of any test spring (Roberts et al., 2017, Ecography).
   Buffered springs stay in the graph (their features are visible, as in any
   transductive setting) but contribute no labels. ``buffer_m = 0`` corresponds
   to the *interpolation* use case (surveyed neighbours are known);
   ``buffer_m = 500`` approximates *extrapolation* to unsurveyed terrain --
   500 m is the distance by which Moran's I of the label has decayed to ~0.26
   (results/metrics/spatial_autocorrelation_distance_decay.csv).

2. Leave-one-municipality-out (LOMO): one municipality is the test set; the
   remaining springs are split into train / validation by spatial blocks
   (~20 % of blocks to validation, seeded), and the same buffer is applied
   around the held-out municipality.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def grid_block_ids(coords_m: np.ndarray, n_blocks_side: int = 5) -> np.ndarray:
    x, y = coords_m[:, 0], coords_m[:, 1]
    xb = np.linspace(x.min() - 1e-6, x.max() + 1e-6, n_blocks_side + 1)
    yb = np.linspace(y.min() - 1e-6, y.max() + 1e-6, n_blocks_side + 1)
    xi = np.clip(np.digitize(x, xb) - 1, 0, n_blocks_side - 1)
    yi = np.clip(np.digitize(y, yb) - 1, 0, n_blocks_side - 1)
    return yi * n_blocks_side + xi


def within_buffer_of(coords_m: np.ndarray, target_mask: np.ndarray, buffer_m: float) -> np.ndarray:
    """Boolean mask of nodes (outside target_mask) within buffer_m of any target node."""
    if buffer_m <= 0 or not target_mask.any():
        return np.zeros(len(coords_m), dtype=bool)
    d, _ = cKDTree(coords_m[target_mask]).query(coords_m, k=1)
    return (d <= buffer_m) & ~target_mask


def _finalise(train, val, test, coords_m, buffer_m):
    buf = within_buffer_of(coords_m, test, buffer_m)
    train = train & ~buf
    val = val & ~buf
    return {
        "train_mask": train, "val_mask": val, "test_mask": test, "buffer_mask": buf,
        "n_train": int(train.sum()), "n_val": int(val.sum()), "n_test": int(test.sum()), "n_buffer": int(buf.sum()),
    }


def spatial_block_kfold(coords_m: np.ndarray, n_splits: int = 5, n_blocks_side: int = 5,
                        random_state: int = 42, buffer_m: float = 0.0) -> list[dict]:
    block = grid_block_ids(coords_m, n_blocks_side)
    blocks = np.unique(block)
    rng = np.random.RandomState(random_state)
    rng.shuffle(blocks)
    groups = np.array_split(blocks, n_splits)
    splits = []
    for i in range(n_splits):
        test = np.isin(block, groups[i])
        val = np.isin(block, groups[(i + 1) % n_splits])
        train = ~test & ~val
        s = _finalise(train, val, test, coords_m, buffer_m)
        s["fold"] = i
        splits.append(s)
    return splits


def leave_one_municipality_out(coords_m: np.ndarray, municipality: np.ndarray, random_state: int = 42,
                               buffer_m: float = 0.0, val_fraction: float = 0.2,
                               n_blocks_side: int = 8) -> list[dict]:
    block = grid_block_ids(coords_m, n_blocks_side)
    splits = []
    for m in sorted(np.unique(municipality)):
        test = municipality == m
        rest_blocks = np.unique(block[~test])
        rng = np.random.RandomState(random_state)
        rng.shuffle(rest_blocks)
        n_val = max(1, int(round(val_fraction * len(rest_blocks))))
        val = ~test & np.isin(block, rest_blocks[:n_val])
        train = ~test & ~val
        s = _finalise(train, val, test, coords_m, buffer_m)
        s["fold"] = str(m)
        splits.append(s)
    return splits
