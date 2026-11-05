"""Model smoke tests on a small random graph (fast, no data files needed)."""

import numpy as np
import torch

from src.models.baselines import MLP, SpringGCN, SpringGraphSAGE, VanillaGATv2, knn_label_predict
from src.models.hydroboost import diffuse, normalised_adjacency, select_and_predict
from src.models.res_gat import ResGATv2


def _toy(n=60, d=8, k=4, seed=0):
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.normal(size=(n, d)), dtype=torch.float32)
    src = np.repeat(np.arange(n), k)
    dst = rng.integers(0, n, size=n * k)
    ei = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    ea = torch.randn(n * k, 5)
    return x, ei, ea


def test_forward_shapes_and_finite():
    x, ei, ea = _toy()
    for m, use_e in [(MLP(8), False), (SpringGCN(8), False), (SpringGraphSAGE(8), False),
                     (VanillaGATv2(8), True), (ResGATv2(8), True), (ResGATv2(8, edge_dim=None), False),
                     (ResGATv2(8, ego_skip=False), True), (ResGATv2(8, layer_norm=True), True),
                     (ResGATv2(8, num_layers=2), True)]:
        out = m(x, ei, ea if use_e else None)
        assert out.shape == (60, 2)
        assert torch.isfinite(out).all()


def test_resgat_attention_normalised_per_receiver():
    x, ei, ea = _toy()
    m = ResGATv2(8).eval()
    _, (att_ei, alpha) = m(x, ei, ea, return_attention_weights=True)
    sums = torch.zeros(60).index_add_(0, att_ei[1], alpha.mean(1))
    assert torch.allclose(sums, torch.ones(60), atol=1e-5)


def test_diffusion_zero_residual_is_identity():
    _, ei, _ = _toy()
    S = normalised_adjacency(ei.numpy(), 60)
    assert np.allclose(diffuse(np.zeros(60), S, 0.4), 0)


def test_hydroboost_selection_uses_only_training_residuals():
    rng = np.random.default_rng(1)
    _, ei, _ = _toy()
    S = normalised_adjacency(ei.numpy(), 60)
    y = rng.integers(0, 2, 60)
    tr = np.zeros(60, bool); tr[:30] = True
    va = np.zeros(60, bool); va[30:45] = True
    p_cb, p_gat = rng.random(60), rng.random(60)
    p1, _ = select_and_predict(p_cb, p_gat, y, S, tr, va)
    y2 = y.copy(); y2[45:] = 1 - y2[45:]  # flipping TEST labels must not change predictions
    p2, _ = select_and_predict(p_cb, p_gat, y2, S, tr, va)
    assert np.allclose(p1, p2)


def test_knn_label_ignores_unlabelled():
    coords = np.random.default_rng(2).random((50, 2)) * 1000
    y = np.random.default_rng(3).integers(0, 2, 50)
    lab = np.zeros(50, bool); lab[:25] = True
    p1 = knn_label_predict(coords, y, lab, k=5)
    y2 = y.copy(); y2[25:] = 1 - y2[25:]
    assert np.allclose(p1, knn_label_predict(coords, y2, lab, k=5))
