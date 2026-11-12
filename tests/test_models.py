"""Model smoke tests (fast, no data files needed)."""

import numpy as np

from src.models import get_models, knn_label_predict


def test_models_are_seed_deterministic():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(200, 6)), rng.integers(0, 2, 200)
    for name, make in get_models(seed=7).items():
        p1 = make().fit(x, y).predict_proba(x)[:, 1]
        p2 = get_models(seed=7)[name]().fit(x, y).predict_proba(x)[:, 1]
        assert np.allclose(p1, p2), name


def test_knn_label_ignores_unlabelled():
    coords = np.random.default_rng(2).random((50, 2)) * 1000
    y = np.random.default_rng(3).integers(0, 2, 50)
    lab = np.zeros(50, bool); lab[:25] = True
    p1 = knn_label_predict(coords, y, lab, k=5)
    y2 = y.copy(); y2[25:] = 1 - y2[25:]
    assert np.allclose(p1, knn_label_predict(coords, y2, lab, k=5))
