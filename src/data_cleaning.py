"""
Data cleaning and standardization for the ICIMOD Kavre Springs dataset.

Source: ICIMOD Regional Database System
DOI: 10.26066/RDS.1974131
Dataset: "Status of Springs in Seven Municipalities of Kavre, Nepal"
License: CC BY 4.0

This module transforms the raw survey data (5,689 water sources) into a clean,
standardized, analysis-ready dataset suitable for machine learning and GNN modeling.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

RAW_DATA_PATH = Path("data/raw/spring_data.csv")
CLEAN_CSV_PATH = Path("data/processed/clean_springs.csv")
GEOJSON_PATH = Path("data/processed/springs.geojson")

# Columns to drop (>90% missing, free-text remarks, or redundant identifiers)
COLUMNS_TO_DROP = [
    "If not who are restricted?",                       # 98.1% missing
    "List the activities done for source revival",      # 97.8% missing
    "Remarks (source specific comment)",                # 62.0% missing, free text
    "List the events associated with sources",          # 92.0% missing
    "Women in leadership position",                     # 91.4% missing
    "Who is the chairperson?",                          # 91.2% missing
    "User group type",                                  # 91.2% missing
    "Plant Species around Water Source",                # 99.9% unique, free text
    "Source Name",                                      # Identifier, not a feature
    "Tole",                                             # Too granular (3,249 unique)
    "District",                                         # Constant (Kavrepalanchok) or near-constant
]

# Columns requiring text standardization (lowercase + strip)
TEXT_STANDARDIZE_COLS = [
    "Source Condition",
    "Source Type",
    "Ownership",
    "Surrounding Environment (Landuse)",
    "Perennial / Seasonal",
    "Reason of drying",
    "Trend of spring discharge",
    "Any impact due to 2015 earthquake?",
    "Nature of discharge",
    "Water emergence point",
    "Perception on source water quality",
    "Perception on source water quality change",
    "Is this source in use?",
    "Is this source tapped?",
    "Is water source significant for cultural/religious purpose?",
    "Is this source accessible to everyone in community? (in term of use access)",
    "Are there water user groups for management?",
    "Any spring revival / pond protection measures taken?",
    "What was done after source was dried up?",
    "After this source dried up , was there any activity for revival done in this source?",
    "Supplied to same Municipality or different?",
    "Supplied to same ward or different wards?",
    "Source Condition.1",
]

# Multi-label columns (comma-separated values)
MULTI_LABEL_COLS = [
    "Purpose of use",
    "Ethnicity of users",
    "Ethnicity of users dependent on this source only",
]

# Rename mapping for cleaner column names
RENAME_MAP = {
    "SN": "id",
    "Municipality": "municipality",
    "Ward Number": "ward",
    "Town/Village Name": "village",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Source Type": "source_type",
    "Ownership": "ownership",
    "Surrounding Environment (Landuse)": "landuse",
    "Source Condition": "source_condition",
    "Perennial / Seasonal": "seasonality",
    "Dried since how many years?": "years_since_dried",
    "Reason of drying": "reason_of_drying",
    "After this source dried up , was there any activity for revival done in this source?": "revival_activity_done",
    "What was done after source was dried up?": "post_drying_action",
    "Source Condition.1": "source_condition_detail",
    "Nature of discharge": "discharge_nature",
    "Water emergence point": "emergence_point",
    "Volume of bucket /bottle (1, 2, 5, 10 or 20 litre)": "bucket_volume_liters",
    "Time to fill (seconds)": "fill_time_seconds",
    "Approx. distance from the nearest dependent household (metre)": "distance_to_household_m",
    "One-way water fetching time (minutes)": "fetch_time_minutes",
    "How much time did it take to fill the Jar of 20 liter before 10 yr? (minutes)": "fill_20l_before_10yr_min",
    "Current time to fill 20 liter jar (minutes)": "fill_20l_current_min",
    "Trend of spring discharge": "discharge_trend",
    "Pond area (square metre)": "pond_area_m2",
    "Pond depth (metre)": "pond_depth_m",
    "Pond area 10 years earlier (square metre)": "pond_area_10yr_m2",
    "Pond depth10 years earlier (metre)": "pond_depth_10yr_m",
    "Perception on source water quality": "water_quality",
    "Perception on source water quality change": "water_quality_change",
    "Is this source in use?": "in_use",
    "Purpose of use": "purpose_of_use",
    "No. of households dependent on this source": "hh_dependent_total",
    "Ethnicity of users": "ethnicity_users",
    "No. of households dependent on this source only": "hh_dependent_exclusive",
    "Ethnicity of users dependent on this source only": "ethnicity_exclusive",
    "Any impact due to 2015 earthquake?": "earthquake_impact",
    "Is this source tapped?": "is_tapped",
    "Supplied to same Municipality or different?": "supply_municipality_scope",
    "Supplied to same ward or different wards?": "supply_ward_scope",
    "Is water source significant for cultural/religious purpose?": "cultural_significance",
    "Is this source accessible to everyone in community? (in term of use access)": "community_accessible",
    "Are there water user groups for management?": "has_user_group",
    "Any spring revival / pond protection measures taken?": "revival_measures_taken",
    "What measures have been taken?": "revival_measures_detail",
}


# =============================================================================
# Core Cleaning Functions
# =============================================================================


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw CSV data."""
    df = pd.read_csv(path)
    logger.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with excessive missing values, free-text, or redundant identifiers."""
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped {len(cols_to_drop)} columns → {df.shape[1]} remaining")
    return df


def standardize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip whitespace for all categorical text columns."""
    for col in TEXT_STANDARDIZE_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"nan": np.nan, "none": np.nan, "0": np.nan, "0.0": np.nan})
            )
    logger.info("Standardized text columns")
    return df


def fix_specific_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data quality issues found during EDA."""

    # Source Condition: 'Dried' → 'dried' (already handled by standardize_text)
    # Trend of spring discharge: 'Same' → 'same' (already handled)

    # Reason of drying: merge case variants
    if "Reason of drying" in df.columns:
        reason_map = {
            "earthquake": "earthquake",
            "drought": "drought",
            "infrastructure development": "infrastructure_development",
            "infrastructure_development": "infrastructure_development",
            "road cutting": "road_cutting",
            "road_cutting": "road_cutting",
            "land use change": "landuse_change",
            "landuse_change": "landuse_change",
        }
        df["Reason of drying"] = df["Reason of drying"].map(
            lambda x: reason_map.get(x, x) if pd.notna(x) else x
        )

    # Earthquake impact: standardize categories
    if "Any impact due to 2015 earthquake?" in df.columns:
        eq_map = {
            "no_change": "no_change",
            "no change": "no_change",
            "decreased": "decreased",
            "degraded": "degraded",
            "increased": "increased",
            "dried": "dried",
            "dried up": "dried",
            "new source": "new_source",
        }
        df["Any impact due to 2015 earthquake?"] = df[
            "Any impact due to 2015 earthquake?"
        ].map(lambda x: eq_map.get(x, x) if pd.notna(x) else x)

    # Are there water user groups: standardize yes/no
    if "Are there water user groups for management?" in df.columns:
        df["Are there water user groups for management?"] = df[
            "Are there water user groups for management?"
        ].map(lambda x: "yes" if x == "yes" else ("no" if x == "no" else x) if pd.notna(x) else x)

    logger.info("Fixed specific value inconsistencies")
    return df


def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary and multi-class target variables for ML tasks."""

    # Task A: Binary classification — Active (0) vs Dried (1)
    df["target_dried"] = (df["Source Condition"].str.lower() == "dried").astype(int)
    logger.info(
        f"Target 'target_dried': Active={int((df['target_dried'] == 0).sum())}, "
        f"Dried={int((df['target_dried'] == 1).sum())}"
    )

    # Task B: Multi-class — Discharge trend (for active springs only)
    if "Trend of spring discharge" in df.columns:
        trend_map = {"decreasing": 0, "same": 1, "increasing": 2}
        df["target_trend"] = df["Trend of spring discharge"].map(trend_map)
        valid_trend = df["target_trend"].notna().sum()
        logger.info(f"Target 'target_trend': {int(valid_trend)} valid entries")

    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features as numeric for ML models."""

    # Binary encodings
    binary_maps = {
        "Source Type": {"spring": 0, "pond": 1},
        "Perennial / Seasonal": {"perennial": 0, "seasonal": 1},
    }
    for col, mapping in binary_maps.items():
        if col in df.columns:
            df[col + "_encoded"] = df[col].map(mapping)

    # Ordinal encoding for earthquake impact
    if "Any impact due to 2015 earthquake?" in df.columns:
        eq_ordinal = {
            "no_change": 0,
            "increased": 1,  # Increased flow (positive impact)
            "new_source": 1,
            "decreased": 2,
            "degraded": 3,
            "dried": 4,
        }
        df["earthquake_impact_ordinal"] = df["Any impact due to 2015 earthquake?"].map(eq_ordinal).fillna(-1)
        df["earthquake_impact_known"] = (df["earthquake_impact_ordinal"] >= 0).astype(int)

    # Ordinal encoding for water quality
    if "Perception on source water quality" in df.columns:
        wq_map = {"excellent": 3, "good": 2, "poor": 1}
        df["water_quality_ordinal"] = df["Perception on source water quality"].map(wq_map)

    # One-hot encoding for landuse (7 categories)
    if "Surrounding Environment (Landuse)" in df.columns:
        landuse_dummies = pd.get_dummies(
            df["Surrounding Environment (Landuse)"], prefix="landuse", dtype=int
        )
        df = pd.concat([df, landuse_dummies], axis=1)
        logger.info(f"One-hot encoded landuse → {landuse_dummies.shape[1]} columns")

    # One-hot encoding for ownership (3 categories)
    if "Ownership" in df.columns:
        ownership_dummies = pd.get_dummies(df["Ownership"], prefix="ownership", dtype=int)
        df = pd.concat([df, ownership_dummies], axis=1)
        logger.info(f"One-hot encoded ownership → {ownership_dummies.shape[1]} columns")

    return df


def encode_multi_label(df: pd.DataFrame) -> pd.DataFrame:
    """Binarize multi-label columns (comma-separated strings)."""

    if "Purpose of use" in df.columns:
        # Extract core purpose categories
        core_purposes = ["drinking", "household", "livestock", "agriculture", "irrigation"]
        for purpose in core_purposes:
            df[f"purpose_{purpose}"] = (
                df["Purpose of use"]
                .fillna("")
                .str.lower()
                .str.contains(purpose)
                .astype(int)
            )
        logger.info(f"Multi-label encoded 'Purpose of use' → {len(core_purposes)} binary columns")

    if "Ethnicity of users" in df.columns:
        core_ethnicities = ["janajati", "bcts", "dalit", "mixed"]
        for eth in core_ethnicities:
            df[f"ethnicity_{eth}"] = (
                df["Ethnicity of users"]
                .fillna("")
                .str.lower()
                .str.contains(eth)
                .astype(int)
            )
        logger.info(f"Multi-label encoded 'Ethnicity of users' → {len(core_ethnicities)} binary columns")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive new features from existing columns."""

    # Flow rate (liters per minute) — only for active springs with measurement
    bucket_col = "Volume of bucket /bottle (1, 2, 5, 10 or 20 litre)"
    time_col = "Time to fill (seconds)"
    if bucket_col in df.columns and time_col in df.columns:
        mask = (df[time_col] > 0) & df[bucket_col].notna()
        df.loc[mask, "flow_rate_lpm"] = (df.loc[mask, bucket_col] / df.loc[mask, time_col]) * 60
        logger.info(f"Computed flow_rate_lpm for {int(mask.sum())} springs")

    # Temporal flow change ratio — how much longer it takes now vs. 10 years ago
    before_col = "How much time did it take to fill the Jar of 20 liter before 10 yr? (minutes)"
    current_col = "Current time to fill 20 liter jar (minutes)"
    if before_col in df.columns and current_col in df.columns:
        mask = (df[before_col] > 0) & df[current_col].notna()
        df.loc[mask, "fill_time_change_ratio"] = df.loc[mask, current_col] / df.loc[mask, before_col]
        # Ratio > 1 means it takes longer now (discharge decreased)
        logger.info(f"Computed fill_time_change_ratio for {int(mask.sum())} springs")

    # Pond area change ratio (for ponds only)
    if "Pond area (square metre)" in df.columns and "Pond area 10 years earlier (square metre)" in df.columns:
        mask = (df["Pond area 10 years earlier (square metre)"] > 0) & df["Pond area (square metre)"].notna()
        df.loc[mask, "pond_area_change_ratio"] = (
            df.loc[mask, "Pond area (square metre)"]
            / df.loc[mask, "Pond area 10 years earlier (square metre)"]
        )

    # Household dependency (convert to numeric, handle non-numeric)
    hh_excl_col = "No. of households dependent on this source only"
    if hh_excl_col in df.columns:
        df[hh_excl_col] = pd.to_numeric(df[hh_excl_col], errors="coerce")

    # Dependency ratio: proportion of exclusively dependent households
    hh_total_col = "No. of households dependent on this source"
    if hh_total_col in df.columns and hh_excl_col in df.columns:
        mask = (df[hh_total_col] > 0) & df[hh_excl_col].notna()
        df.loc[mask, "dependency_ratio"] = df.loc[mask, hh_excl_col] / df.loc[mask, hh_total_col]
        df["dependency_ratio"] = df["dependency_ratio"].clip(0, 1)

    # Boolean: yes/no columns to 0/1
    yes_no_cols = [
        "Is this source in use?",
        "Is this source tapped?",
        "Is water source significant for cultural/religious purpose?",
        "Is this source accessible to everyone in community? (in term of use access)",
        "Are there water user groups for management?",
    ]
    for col in yes_no_cols:
        if col in df.columns:
            df[col + "_binary"] = df[col].map({"yes": 1, "no": 0})

    logger.info("Feature engineering complete")
    return df


def compute_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute spatial features from coordinates using scipy KDTree."""
    from scipy.spatial import KDTree

    coords = df[["Latitude", "Longitude"]].values

    # Approximate conversion to meters (at ~27.5°N latitude)
    lat_to_m = 111_320.0  # 1 degree latitude ≈ 111.32 km
    lon_to_m = 111_320.0 * np.cos(np.radians(27.57))  # ~98.8 km at this latitude

    coords_m = np.column_stack([
        coords[:, 0] * lat_to_m,
        coords[:, 1] * lon_to_m,
    ])

    tree = KDTree(coords_m)

    # 1. Local spring density: count of springs within 500m
    counts = tree.query_ball_point(coords_m, r=500)
    df["spring_density_500m"] = [len(c) - 1 for c in counts]  # exclude self
    logger.info(f"Spring density (500m): mean={df['spring_density_500m'].mean():.1f}")

    # 2. Nearest dried spring distance (for each spring)
    dried_mask = df["target_dried"] == 1
    if dried_mask.sum() > 0:
        dried_coords = coords_m[dried_mask.values]
        dried_tree = KDTree(dried_coords)
        distances, _ = dried_tree.query(coords_m, k=1)
        df["dist_to_nearest_dried_m"] = distances
        logger.info(
            f"Distance to nearest dried spring: mean={df['dist_to_nearest_dried_m'].mean():.0f}m"
        )

    # 3. Nearest active spring distance (for each spring)
    active_mask = df["target_dried"] == 0
    if active_mask.sum() > 0:
        active_coords = coords_m[active_mask.values]
        active_tree = KDTree(active_coords)
        distances, _ = active_tree.query(coords_m, k=1)
        df["dist_to_nearest_active_m"] = distances

    # 4. Municipality-level drying rate (contextual feature)
    muni_rate = df.groupby("Municipality")["target_dried"].mean()
    df["muni_drying_rate"] = df["Municipality"].map(muni_rate)
    logger.info(f"Municipality drying rates:\n{muni_rate.sort_values(ascending=False)}")

    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to clean, snake_case names."""
    rename_final = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_final)
    logger.info(f"Renamed {len(rename_final)} columns")
    return df


def export_geojson(df: pd.DataFrame, path: Path = GEOJSON_PATH) -> None:
    """Export as GeoJSON for GIS tools and interactive mapping."""
    features = []
    for _, row in df.iterrows():
        lat_col = "latitude" if "latitude" in df.columns else "Latitude"
        lon_col = "longitude" if "longitude" in df.columns else "Longitude"

        props = {}
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                props[col] = None
            elif isinstance(val, (np.integer,)):
                props[col] = int(val)
            elif isinstance(val, (np.floating,)):
                props[col] = float(val)
            else:
                props[col] = val

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row[lon_col]), float(row[lat_col])],
            },
            "properties": props,
        }
        features.append(feature)

    geojson = {"type": "FeatureCollection", "features": features}

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)
    logger.info(f"Exported GeoJSON: {path} ({len(features)} features)")


# =============================================================================
# Main Pipeline
# =============================================================================


def run_cleaning_pipeline(
    raw_path: Path = RAW_DATA_PATH,
    output_csv: Path = CLEAN_CSV_PATH,
    output_geojson: Path = GEOJSON_PATH,
) -> pd.DataFrame:
    """Execute the full data cleaning pipeline."""
    logger.info("=" * 60)
    logger.info("Starting data cleaning pipeline")
    logger.info("=" * 60)

    # Load
    df = load_raw_data(raw_path)

    # Clean
    df = drop_unnecessary_columns(df)
    df = standardize_text(df)
    df = fix_specific_values(df)

    # Create targets
    df = create_target_variables(df)

    # Encode
    df = encode_categorical_features(df)
    df = encode_multi_label(df)

    # Engineer features
    df = engineer_features(df)
    df = compute_spatial_features(df)

    # Rename
    df = rename_columns(df)

    # Export
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved clean CSV: {output_csv} ({df.shape[0]} rows, {df.shape[1]} columns)")

    export_geojson(df, output_geojson)

    # Summary
    logger.info("=" * 60)
    logger.info("Cleaning pipeline complete!")
    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Target distribution (dried): {df['target_dried'].value_counts().to_dict()}")
    logger.info(f"Null percentage:\n{(df.isnull().sum() / len(df) * 100).describe()}")
    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    df = run_cleaning_pipeline()
    print(f"\nFinal dataset: {df.shape}")
    print(f"\nColumns:\n{list(df.columns)}")

