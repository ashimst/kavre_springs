"""Guards against the failure modes found in the audit: label leakage, spatial leakage,
and results that are not derived from the raw fold-level outputs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import (EARTHQUAKE_LEAKY, FEATURE_SETS, FORBIDDEN_FEATURES, PRIMARY_FEATURE_SET,
                          check_feature_set)
from src.dataset import design_matrix, load_master
from src.spatial_splits import leave_one_municipality_out, spatial_block_kfold

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def df():
    return load_master()


@pytest.fixture(scope="module")
def graph(df):
    return design_matrix(df, PRIMARY_FEATURE_SET)


def test_primary_and_sensitivity_sets_pass_audit(df):
    assert check_feature_set(df, "env") == []
    assert check_feature_set(df, "env_season") == []
    assert check_feature_set(df, "env_hydro") == []
    assert check_feature_set(df, "env_hydro_season") == []


def test_leaky_reference_is_detected(df):
    v = check_feature_set(df, "leaky_reference")
    assert any("earthquake" in s for s in v)


def test_no_forbidden_features_in_claim_sets():
    for fs in ["env", "env_season", "env_hydro", "env_hydro_season"]:
        assert not set(FEATURE_SETS[fs]) & FORBIDDEN_FEATURES
        assert not set(FEATURE_SETS[fs]) & set(EARTHQUAKE_LEAKY)


def test_hydro_and_road_features_present_and_finite(df):
    from src.features import HYDRO_CATCHMENT, ROAD_FEATURES
    for feat in HYDRO_CATCHMENT + ROAD_FEATURES:
        assert feat in df.columns, f"Feature {feat} missing from master dataset"
        assert df[feat].notna().all(), f"Feature {feat} contains NaN values"
        assert np.isfinite(df[feat].to_numpy(dtype=float)).all(), f"Feature {feat} contains non-finite values"


def test_skip_logic_multi_label_columns_detected_as_leaks(df):
    from src.features import missingness_audit
    audit = missingness_audit(df).set_index("column")
    for col in ["purpose_drinking", "purpose_household", "ethnicity_janajati"]:
        if col in audit.index:
            assert audit.loc[col, "skip_logic_leak"], f"{col} should be detected as skip-logic leak"


def test_design_matrix_shapes(graph):
    n = len(graph["y"])
    assert n == 5689
    assert graph["x"].shape == (n, len(FEATURE_SETS[PRIMARY_FEATURE_SET]))
    assert np.isfinite(graph["x"]).all()
    assert np.allclose(graph["x"].mean(0), 0, atol=1e-4)


@pytest.mark.parametrize("buffer_m", [0.0, 500.0])
def test_spatial_folds_disjoint_and_buffered(graph, buffer_m):
    from scipy.spatial import cKDTree
    splits = spatial_block_kfold(graph["coords_m"], 5, random_state=7, buffer_m=buffer_m)
    test_union = np.zeros(len(graph["y"]), bool)
    for s in splits:
        tr, va, te = s["train_mask"], s["val_mask"], s["test_mask"]
        assert not (tr & te).any() and not (va & te).any() and not (tr & va).any()
        assert not (test_union & te).any()
        test_union |= te
        if buffer_m > 0:
            d, _ = cKDTree(graph["coords_m"][te]).query(graph["coords_m"][tr | va], k=1)
            assert d.min() > buffer_m
    assert test_union.all()


def test_lomo_holds_out_whole_municipality(graph):
    for s in leave_one_municipality_out(graph["coords_m"], graph["municipality"], buffer_m=500):
        te = s["test_mask"]
        assert set(graph["municipality"][te]) == {s["fold"]}
        assert not (s["train_mask"] & te).any()


def test_reported_summary_matches_fold_results():
    summ, folds_paths = ROOT / "results/metrics/benchmark_summary.csv", list((ROOT / "results/metrics").glob("benchmark_*_folds.csv"))
    folds_paths = [p for p in folds_paths if "_quick" not in p.name]
    if not summ.exists() or not folds_paths:
        pytest.skip("benchmark not run yet")
    folds = pd.concat([pd.read_csv(p) for p in folds_paths])
    s = pd.read_csv(summ)
    recomputed = folds.groupby(["profile", "feature_set", "buffer_m", "model"]).roc_auc.mean()
    merged = s.set_index(["profile", "feature_set", "buffer_m", "model"]).roc_auc_mean
    # the summary covers the current profiles only (src.report.current_runs); every reported value must be recomputable
    assert np.allclose(merged, recomputed.loc[merged.index])


def test_no_hardcoded_result_literals_in_pipeline():
    """The old ablation table was written from literals; make sure no pipeline module does that."""
    for f in ["benchmark.py", "report.py", "spatial_mining.py", "spatial_analysis.py", "catchment_analysis.py",
              "drivers_analysis.py", "unsupervised.py", "robustness.py", "pattern_mining.py"]:
        src = (ROOT / "src" / f).read_text(encoding="utf-8")
        assert '"f1_macro": 0.' not in src and '"roc_auc": 0.' not in src
