"""Unit tests for Kavre Springs data cleaning & pipeline."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

CLEAN_CSV = Path("data/processed/clean_springs.csv")
GEOJSON_FILE = Path("data/processed/springs.geojson")


def test_clean_file_exists():
    assert CLEAN_CSV.exists(), "Clean CSV does not exist"
    assert GEOJSON_FILE.exists(), "GeoJSON does not exist"


def test_data_shape_and_ids():
    df = pd.read_csv(CLEAN_CSV)
    assert len(df) == 5689, f"Expected 5689 rows, got {len(df)}"
    assert df["id"].nunique() == 5689, "Duplicate IDs found"


def test_target_variable():
    df = pd.read_csv(CLEAN_CSV)
    assert "target_dried" in df.columns, "target_dried column missing"
    assert set(df["target_dried"].unique()) == {0, 1}, "target_dried should be strictly binary {0, 1}"
    assert (df["target_dried"] == 1).sum() == 1521, "Dried count mismatch"
    assert (df["target_dried"] == 0).sum() == 4168, "Active count mismatch"


def test_coordinates_validity():
    df = pd.read_csv(CLEAN_CSV)
    assert df["latitude"].notna().all(), "Null latitude found"
    assert df["longitude"].notna().all(), "Null longitude found"
    assert (df["latitude"] >= 27.0).all() and (df["latitude"] <= 28.5).all()
    assert (df["longitude"] >= 85.0).all() and (df["longitude"] <= 86.5).all()


def test_municipalities_count():
    df = pd.read_csv(CLEAN_CSV)
    assert df["municipality"].nunique() == 7, f"Expected 7 municipalities, got {df['municipality'].nunique()}"


def test_spatial_features():
    df = pd.read_csv(CLEAN_CSV)
    assert "spring_density_500m" in df.columns
    assert (df["spring_density_500m"] >= 0).all()


def test_label_derived_columns_never_used_as_features():
    """dist_to_nearest_dried_m / muni_drying_rate are computed from the label (descriptive only)."""
    from src.features import FEATURE_SETS
    for fs, cols in FEATURE_SETS.items():
        assert "dist_to_nearest_dried_m" not in cols
        assert "dist_to_nearest_active_m" not in cols
        assert "muni_drying_rate" not in cols


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

