"""Unit tests for the statistical building blocks of the spatial, catchment and unsupervised analyses."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.catchment_analysis import icc1, matched_grid_null
from src.features import FORBIDDEN_FEATURES
from src.spatial_analysis import kulldorff_scan
from src.unsupervised import ENV_COLS, LANDSCAPE_COLS, cramers_v, env_r2, wilson

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)


def test_icc_extremes():
    g = np.repeat(np.arange(50), 20)
    assert icc1(g % 2 == 0, g) == pytest.approx(1.0)          # all variance between groups
    y = rng.integers(0, 2, len(g)).astype(float)
    assert abs(icc1(y, g)) < 0.05                               # random labels -> ~0


def test_matched_grid_has_requested_number_of_units():
    coords = rng.uniform(0, 20000, (3000, 2))
    y = rng.integers(0, 2, 3000).astype(float)
    cell, null, counts = matched_grid_null(coords, y, 60, n_shift=20, rng=np.random.default_rng(1))
    assert abs(np.mean(counts) - 60) < 15
    assert np.all(np.abs(null) < 0.1)


def test_scan_detects_planted_cluster():
    coords = rng.uniform(0, 10000, (1500, 2))
    p = np.where(np.hypot(coords[:, 0] - 3000, coords[:, 1] - 3000) < 1200, 0.8, 0.2)
    cases = (rng.random(1500) < p).astype(int)
    cl = kulldorff_scan(coords, cases, n_perm=99, rng=np.random.default_rng(2))
    top = cl[0]
    centre = coords[top["center_index"]]
    assert top["p_value"] <= 0.02 and top["relative_risk"] > 2
    assert np.hypot(*(centre - 3000)) < 1500


def test_unsupervised_inputs_are_label_free():
    assert "latitude" not in ENV_COLS and "longitude" not in ENV_COLS
    assert not set(ENV_COLS) & set(FORBIDDEN_FEATURES)
    assert "target_dried" not in ENV_COLS
    # landscape units must not depend on survey-recorded fields
    assert set(LANDSCAPE_COLS) <= set(ENV_COLS)
    assert not any(c.startswith(("landuse_", "ownership_")) or c == "Source Type_encoded" for c in LANDSCAPE_COLS)


def test_helpers():
    lo, hi = wilson(30, 100)
    assert lo < 0.3 < hi
    a = rng.integers(0, 4, 5000)
    assert cramers_v(a, rng.integers(0, 2, 5000)) < 0.05
    assert cramers_v(a, a) == pytest.approx(1.0, abs=1e-3)
    P = rng.normal(size=(400, 3))
    assert env_r2(P, np.zeros(400)) == pytest.approx(0.0)
    assert env_r2(P, np.arange(400)) == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["dr_three_way_partition.csv"])
def test_partition_fractions_sum_to_one(name):
    p = ROOT / "results/metrics" / name
    if not p.exists():
        pytest.skip("analysis not run yet")
    assert pd.read_csv(p).adj_pseudo_r2.sum() == pytest.approx(1.0, abs=1e-9)


def test_mantel_haenszel_detects_within_stratum_effect_and_simpson_reversal():
    from src.pattern_mining import mantel_haenszel
    r = np.random.default_rng(3)
    strata = np.repeat(np.arange(20), 200)
    a = r.random(4000) < 0.5
    y = (r.random(4000) < np.where(a, 0.4, 0.2)).astype(int)
    or_mh, lo, hi = mantel_haenszel(a, y, strata)
    assert lo > 1.5 and or_mh == pytest.approx(2.67, rel=0.25)
    # Simpson: item common in high-rate strata but protective within each stratum
    base = np.where(strata < 10, 0.6, 0.1)
    a2 = r.random(4000) < np.where(strata < 10, 0.8, 0.2)
    y2 = (r.random(4000) < base * np.where(a2, 0.7, 1.0)).astype(int)
    assert y2[a2].mean() > y2[~a2].mean()             # pooled: looks like a risk factor
    assert mantel_haenszel(a2, y2, strata)[2] < 1      # within strata: protective


def test_apriori_recovers_planted_rule_and_prunes_unproductive():
    from src.pattern_mining import mine_rules
    r = np.random.default_rng(4)
    n = 3000
    I = pd.DataFrame({"A": r.random(n) < 0.3, "B": r.random(n) < 0.5, "noise": r.random(n) < 0.5})
    y = (r.random(n) < np.where(I.A & I.B, 0.8, 0.2)).astype(int)
    rules, base = mine_rules(I, y, 1, min_support=0.05, max_len=2)
    top = rules[(rules.q_bh < 0.05) & rules.productive].sort_values("lift", ascending=False).iloc[0]
    assert top.antecedent == frozenset({"A", "B"})
    noise = rules[rules.antecedent == frozenset({"noise"})].iloc[0]
    assert not noise.productive


def test_mantel_identity():
    from src.unsupervised import mantel
    pts = rng.normal(size=(30, 2))
    D = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    r0, p = mantel(D, D, n_perm=99)
    assert r0 == pytest.approx(1.0) and p <= 0.02


def test_random_regions_match_size_distribution_and_cover_all_nodes():
    from src.robustness import neighbours, random_regions
    pts = rng.uniform(0, 10000, (1200, 2))
    sizes = np.array([200, 150, 300, 250, 300])
    lab = random_regions(neighbours(pts), sizes, np.random.default_rng(1), pts)
    assert (lab >= 0).all() and len(np.unique(lab)) == len(sizes)
    got = np.sort(np.bincount(lab))
    assert np.abs(got - np.sort(sizes)).max() <= 60
