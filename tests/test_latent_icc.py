"""Latent-scale ICC: recovery of a known random-intercept variance and invariance to prevalence."""

import numpy as np
import pytest

from src.catchment_analysis import icc1
from src.latent_icc import latent_icc


def _simulate(rng, sd, b0, n_groups=150, size=60):
    g = np.repeat(np.arange(n_groups), size)
    u = rng.normal(0, sd, n_groups)[g]
    y = (rng.random(len(g)) < 1 / (1 + np.exp(-(b0 + u)))).astype(float)
    return y, g


def test_recovers_the_simulated_latent_icc():
    rng = np.random.default_rng(0)
    sd = 1.0
    target = sd ** 2 / (sd ** 2 + np.pi ** 2 / 3)
    y, g = _simulate(rng, sd, -1.0)
    assert latent_icc(y, g) == pytest.approx(target, abs=0.04)


def test_latent_icc_is_stable_across_prevalence_where_the_observed_icc_is_not():
    rng = np.random.default_rng(1)
    common = _simulate(rng, 1.0, 0.0, n_groups=600)
    rare = _simulate(rng, 1.0, -2.5, n_groups=600)
    target = 1 / (1 + np.pi ** 2 / 3)
    assert latent_icc(*common) == pytest.approx(target, abs=0.04)
    assert latent_icc(*rare) == pytest.approx(target, abs=0.04)
    assert icc1(*common) - icc1(*rare) > 0.05


def test_no_grouping_gives_near_zero_and_offset_absorbs_a_covariate():
    rng = np.random.default_rng(2)
    n = 6000
    x = rng.normal(size=n)
    g = rng.integers(0, 100, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1 + 1.5 * x)))).astype(float)
    assert latent_icc(y, g) < 0.02
    # groups sorted on x look clustered unless x is passed as an offset
    g_sorted = np.argsort(np.argsort(x)) // 60
    assert latent_icc(y, g_sorted) > 0.1
    assert latent_icc(y, g_sorted, offset=-1 + 1.5 * x) < 0.02
