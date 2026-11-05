"""
HydroBoost-GAT: CatBoost prior + ResGATv2 + graph residual diffusion.

Stage 1  CatBoost on the training springs. Its probabilities for *training*
         springs are out-of-fold (inner spatially-grouped 5-fold CV), so the
         residuals used in Stage 3 are honest; validation/test springs get the
         prediction of the CatBoost model fitted on all training springs.
Stage 2  ResGATv2 trained on the training springs (early stopping on validation).
Stage 3  Blend p = w p_cb + (1 - w) p_gat, then diffuse the residuals of
         *labelled* springs over the symmetric-normalised k-NN graph:
             E <- (1 - a) r + a S E   (K = 3 steps),   p* = clip(p + b E, 0, 1)
         (w, a, b) are chosen on the validation fold from a grid that includes
         w = 1 (no GAT) and b = 0 (no diffusion), so the data can reject either
         stage. Only residuals of training springs are ever diffused (validation
         labels are used for selection only), so HydroBoost sees the same labels
         as every baseline.

Note: training-node probabilities from the GAT are in-sample (early-stopped),
so their residuals are somewhat optimistic; this is stated in the SI.
"""

from __future__ import annotations

import itertools

import numpy as np
import scipy.sparse as sp
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from src.models.res_gat import ResGATv2
from src.models.training import train_node_classifier
from src.spatial_splits import grid_block_ids

FULL_GRID = {"w": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "alpha": [0.2, 0.4, 0.6], "beta": [0.0, 0.5, 1.0]}


def catboost_factory(seed: int):
    return CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05, auto_class_weights="Balanced",
                              verbose=0, random_seed=seed, thread_count=-1, allow_writing_files=False)


def normalised_adjacency(edge_index: np.ndarray, n: int) -> sp.csr_matrix:
    src, dst = edge_index
    A = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64) + sp.eye(n, format="csr")
    d = np.asarray(A.sum(1)).ravel()
    Dm = sp.diags(1.0 / np.sqrt(d))
    return (Dm @ A @ Dm).tocsr()


def diffuse(residual: np.ndarray, S: sp.csr_matrix, alpha: float, steps: int = 3) -> np.ndarray:
    E = residual.copy()
    for _ in range(steps):
        E = (1 - alpha) * residual + alpha * (S @ E)
    return E


def oof_catboost(x_np, y_np, coords_m, train_mask, seed, n_inner=5):
    """Out-of-fold CatBoost probabilities for training nodes (inner spatial groups) and
    full-train-model probabilities for every other node."""
    p = np.zeros(len(y_np))
    tr_idx = np.where(train_mask)[0]
    groups = grid_block_ids(coords_m[tr_idx], n_blocks_side=8)
    n_inner = min(n_inner, len(np.unique(groups)))
    for itr, ite in GroupKFold(n_splits=n_inner).split(tr_idx, groups=groups):
        m = catboost_factory(seed)
        m.fit(x_np[tr_idx[itr]], y_np[tr_idx[itr]])
        p[tr_idx[ite]] = m.predict_proba(x_np[tr_idx[ite]])[:, 1]
    full = catboost_factory(seed)
    full.fit(x_np[train_mask], y_np[train_mask])
    p[~train_mask] = full.predict_proba(x_np[~train_mask])[:, 1]
    return p


def select_and_predict(p_cb, p_gat, y_np, S, train_mask, val_mask, grid=FULL_GRID):
    """Choose (w, alpha, beta) on validation AUC, then predict all nodes using
    residuals of train + validation nodes. Returns (probabilities, chosen params)."""
    best = (-1.0, None)
    for w, a, b in itertools.product(grid["w"], grid["alpha"], grid["beta"]):
        base = w * p_cb + (1 - w) * p_gat
        r = np.zeros_like(base)
        r[train_mask] = y_np[train_mask] - base[train_mask]
        p = np.clip(base + b * diffuse(r, S, a), 0, 1)
        auc = roc_auc_score(y_np[val_mask], p[val_mask])
        if auc > best[0] + 1e-9:
            best = (auc, (w, a, b))
    w, a, b = best[1]
    base = w * p_cb + (1 - w) * p_gat
    # Only training-node residuals are diffused, so HydroBoost sees exactly the same
    # labels as every other model in the benchmark.
    r = np.zeros_like(base)
    r[train_mask] = y_np[train_mask] - base[train_mask]
    p = np.clip(base + b * diffuse(r, S, a), 0, 1)
    return p, {"w_catboost": w, "alpha": a, "beta": b, "val_auc": best[0]}


class HydroBoostGAT:
    def __init__(self, in_dim: int, hidden_dim: int = 64, edge_dim: int = 5, heads: int = 2,
                 num_layers: int = 1, dropout: float = 0.15, random_state: int = 42):
        self.gat_kwargs = dict(in_dim=in_dim, hidden_dim=hidden_dim, edge_dim=edge_dim, heads=heads,
                               num_layers=num_layers, dropout=dropout)
        self.random_state = random_state
        self.params_ = None

    def fit_predict(self, x, edge_index, edge_attr, y, coords_m, train_mask, val_mask,
                    p_gat=None, p_cb=None, grid=FULL_GRID):
        """Return probabilities for all nodes. ``p_gat``/``p_cb`` may be passed in to reuse
        components already computed for the same fold (the benchmark does this)."""
        x_np, y_np = x.numpy(), y.numpy()
        if p_cb is None:
            p_cb = oof_catboost(x_np, y_np, coords_m, train_mask, self.random_state)
        if p_gat is None:
            p_gat, _ = train_node_classifier(lambda: ResGATv2(**self.gat_kwargs), x, edge_index, edge_attr, y,
                                             train_mask, val_mask, seed=self.random_state)
        S = normalised_adjacency(edge_index.numpy(), len(y_np))
        p, self.params_ = select_and_predict(p_cb, p_gat, y_np, S, train_mask, val_mask, grid)
        return p
