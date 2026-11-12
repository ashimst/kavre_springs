"""Building blocks of the ward-catchment contrasts and pair-model robustness checks (X12-X15)."""

import numpy as np
import pytest

from src.catchment_analysis import icc1
from src.revision_experiments import (_block_jackknife_icc_difference, _contrast_row, _resample_blocks, _vif,
                                      dyadic_robust_cov)


def test_contrast_is_the_paired_difference_of_the_draws():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0.5, 0.1, 400), rng.normal(0.1, 0.1, 400)
    row = _contrast_row("a - b", 0.5, 0.1, a, b)
    assert row["difference"] == pytest.approx(0.4)
    assert row["ci_low"] == pytest.approx(np.percentile(a - b, 2.5))
    assert row["ci_high"] == pytest.approx(np.percentile(a - b, 97.5))
    assert row["p_one_sided"] == pytest.approx((np.sum(a - b <= 0) + 1) / 401)
    # identical draws: the difference is exactly zero in every resample, so p is 1
    assert _contrast_row("a - a", 0.5, 0.5, a, a)["p_one_sided"] == 1.0


def test_block_resample_keeps_blocks_whole():
    sizes = np.array([3, 5, 2, 4, 1, 6])
    blocks = np.repeat(np.arange(6), sizes)
    idx = _resample_blocks(blocks, np.random.default_rng(1))
    drawn, counts = np.unique(blocks[idx], return_counts=True)
    assert np.all(counts % sizes[drawn] == 0)          # each block enters zero, one or several whole times
    assert (counts // sizes[drawn]).sum() == 6         # as many draws as there are blocks


def test_jackknife_difference_separates_a_real_grouping_from_a_random_one():
    rng = np.random.default_rng(4)
    real = np.repeat(np.arange(40), 50)
    y = (rng.random(2000) < np.repeat(rng.uniform(0.05, 0.6, 40), 50)).astype(float)
    random_groups = rng.permutation(real)
    blocks = np.repeat(np.arange(20), 100)
    out = _block_jackknife_icc_difference(y, real, random_groups, blocks)
    assert out["difference"] == pytest.approx(icc1(y, real) - icc1(y, random_groups))
    assert out["ci_low"] > 0
    same = _block_jackknife_icc_difference(y, real, real, blocks)
    assert same["difference"] == 0 and same["se"] == 0


def test_dyadic_robust_covariance_matches_the_double_sum():
    rng = np.random.default_rng(2)
    n_sources, n_dyads = 12, 40
    i = rng.integers(0, n_sources, n_dyads)
    j = (i + rng.integers(1, n_sources, n_dyads)) % n_sources      # never a self-pair
    X = np.c_[rng.normal(size=(n_dyads, 2)), np.ones(n_dyads)]
    resid, weight = rng.normal(size=n_dyads), rng.uniform(0.1, 0.25, n_dyads)
    scores = X * resid[:, None]
    meat = np.zeros((3, 3))
    for a in range(n_dyads):
        for b in range(n_dyads):
            shared = len({i[a], j[a]} & {i[b], j[b]})
            if a == b:
                meat += np.outer(scores[a], scores[a])
            elif shared:
                meat += shared * np.outer(scores[a], scores[b])    # a dyad repeated in the list shares two sources
    bread = np.linalg.inv((X * weight[:, None]).T @ X)
    assert np.allclose(dyadic_robust_cov(X, resid, weight, i, j), bread @ meat @ bread)


def test_vif_is_one_for_unrelated_columns_and_large_for_near_copies():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(2000, 3))
    assert np.all(_vif(X) < 1.02)
    X[:, 2] = X[:, 0] + rng.normal(0, 0.05, 2000)
    v = _vif(X)
    assert v[0] > 100 and v[2] > 100 and v[1] < 1.02
    X[:, 2] = X[:, 0] + X[:, 1]                        # exact linear combination
    assert np.all(np.isinf(_vif(X)))
    assert np.isnan(_vif(np.c_[X[:, :2], np.ones(2000)])[2])   # constant column
