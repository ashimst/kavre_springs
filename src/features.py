"""
Authoritative feature-set definitions and the survey skip-logic leakage audit.

Why this module exists
----------------------
The ICIMOD Kavre survey (Thapa et al., 2025, Data in Brief 60:111466) uses a
7-section form for active sources but only 4 sections for dried sources. Every
question in the skipped sections (discharge, household dependency, water-user
groups, "Any impact due to 2015 earthquake?", ...) is therefore missing for
~100 % of dried sources and present for ~94 % of active ones. Any feature built
from those questions -- including a "was it answered?" indicator or an ordinal
code with a fill value -- encodes the label itself.

An earlier version of this project used `earthquake_impact_ordinal` and
`earthquake_impact_known` as node features. A one-line rule on the latter
reproduces the balanced accuracy of every model reported then (see
`results/metrics/leakage_audit.csv` and Table S4 of the SI).

Feature sets
------------
* ``env``          (PRIMARY) - a-priori observable covariates: location,
                    terrain, climate, vegetation indices, source type,
                    land use, ownership. Contains nothing collected
                    conditionally on the outcome.
* ``env_season``   (SENSITIVITY) - ``env`` + perennial/seasonal flag. Asked of
                    all sources, but the survey ran Oct 2023 - Jun 2024 (mostly
                    dry season), so a seasonal spring may be recorded as
                    "dried" because of survey timing. Reported separately.
* ``leaky_reference`` (DIAGNOSTIC ONLY) - ``env_season`` + the two earthquake
                    features, to quantify the leakage. Never use for claims.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

LOCATION = ["latitude", "longitude", "spring_density_500m"]
TERRAIN = ["elevation_m", "slope_deg", "aspect_sin", "aspect_cos"]
CLIMATE = ["precip_annual_mm", "precip_monsoon_mm", "precip_dry_season_mm"]
VEGETATION = ["ndvi_mean", "ndwi_mean"]
SURVEY_STRUCTURAL = [
    "Source Type_encoded",
    "landuse_agriculture",
    "landuse_dense_forest",
    "landuse_fallow",
    "landuse_grassland",
    "landuse_others",
    "landuse_settlement",
    "landuse_sparse_forest",
    "ownership_community",
    "ownership_private",
    "ownership_public",
]
HYDRO_CATCHMENT = [
    "upa",
    "log_upa",
    "hnd",
    "elv",
    "twi",
    "tpi_500m",
    "tri_90m",
    "precip_dry_deficit_mm",
    "precip_mon_deficit_mm",
    "aet",
    "def",
    "soil",
    "pdsi",
]
ROAD_FEATURES = [
    "dist_to_road_m",
    "road_density_500m",
    "road_upslope",
]
SEASONALITY = ["Perennial / Seasonal_encoded"]
EARTHQUAKE_LEAKY = ["earthquake_impact_ordinal", "earthquake_impact_known"]
# Google Earth Engine enrichment (src/extract_enrichment_gee.py): independent, label-free GIS layers.
ENRICHMENT = [
    "wc_tree", "wc_shrub", "wc_grass", "wc_crop", "wc_built", "wc_bare", "wc_water",   # ESA WorldCover fractions, 250 m
    "gfc_treecover2000", "gfc_loss_pre2015", "gfc_loss_post2015",                       # Hansen forest cover and loss
    "soil_clay", "soil_sand", "soil_soc", "soil_bdod",                                  # SoilGrids 0-30 cm
    "ndvi_trend", "ndvi_change_2015", "et_mean", "et_trend",                            # MODIS vegetation / ET dynamics
    "chirps_trend", "chirps_anom_2015",                                                 # CHIRPS 1981-2023
    "built_2020", "built_change", "pop_1km", "nightlights",                             # GHSL, WorldPop, VIIRS
    "landform_position", "chili", "mtpi", "topo_diversity",                             # ALOS landforms / morphometry
]
# USGS ShakeMap / ComCat covariates (src/seismic.py): independent of the survey, a-priori observable.
SEISMIC = ["pga_main", "pgv_main", "pga_may12", "pgv_may12", "aftershocks_20km", "dist_nearest_aftershock_km"]

FEATURE_SETS = {
    "env_hydro_seismic": LOCATION + TERRAIN + CLIMATE + VEGETATION + SURVEY_STRUCTURAL + HYDRO_CATCHMENT + ROAD_FEATURES + SEISMIC,
    "env": LOCATION + TERRAIN + CLIMATE + VEGETATION + SURVEY_STRUCTURAL,
    "env_season": LOCATION + TERRAIN + CLIMATE + VEGETATION + SURVEY_STRUCTURAL + SEASONALITY,
    "env_hydro": LOCATION + TERRAIN + CLIMATE + VEGETATION + SURVEY_STRUCTURAL + HYDRO_CATCHMENT + ROAD_FEATURES,
    "env_rich": LOCATION + TERRAIN + CLIMATE + VEGETATION + SURVEY_STRUCTURAL + HYDRO_CATCHMENT + ROAD_FEATURES + ENRICHMENT,
    "env_hydro_season": LOCATION + TERRAIN + CLIMATE + VEGETATION + SURVEY_STRUCTURAL + HYDRO_CATCHMENT + ROAD_FEATURES + SEASONALITY,
    "leaky_reference": LOCATION + TERRAIN + CLIMATE + VEGETATION + SURVEY_STRUCTURAL + SEASONALITY + EARTHQUAKE_LEAKY,
}
PRIMARY_FEATURE_SET = "env_hydro"      # prediction benchmark (incl. GNNs)
# Explanatory analyses (unsupervised, partitioning, drivers) use the enriched set when its layers are available.
EXPLANATORY_SET = "env_rich"
# Change / dynamics variables: may partly reflect drying itself (reverse causation), so they are excluded from the
# static landscape typology and interpreted with care.
DYNAMIC_FEATURES = ["ndvi_trend", "ndvi_change_2015", "et_trend", "gfc_loss_post2015", "built_change"]


def explanatory_columns(df=None, drop_coordinates=True):
    """Columns of the explanatory set (falls back to env_hydro if the enrichment layers are absent)."""
    cols = FEATURE_SETS[EXPLANATORY_SET]
    if df is not None and any(c not in df.columns for c in ENRICHMENT):
        cols = FEATURE_SETS["env_hydro"]
    return [c for c in cols if not (drop_coordinates and c in ("latitude", "longitude"))]

# Raw survey column each model feature is derived from (pre-encoding, pre-imputation),
# used by the audit so that fill values such as -1 cannot hide missingness.
FEATURE_SOURCE_COLUMN = {
    "Source Type_encoded": "source_type",
    "Perennial / Seasonal_encoded": "seasonality",
    "earthquake_impact_ordinal": "earthquake_impact",
    "earthquake_impact_known": "earthquake_impact",
    "water_quality_ordinal": "water_quality",
    "aspect_sin": "aspect_deg",
    "aspect_cos": "aspect_deg",
    **{c: "landuse" for c in SURVEY_STRUCTURAL if c.startswith("landuse_")},
    **{c: "ownership" for c in SURVEY_STRUCTURAL if c.startswith("ownership_")},
    **{f"purpose_{p}": "purpose_of_use" for p in ["drinking", "household", "livestock", "agriculture", "irrigation"]},
    **{f"ethnicity_{e}": "ethnicity_users" for e in ["janajati", "bcts", "dalit", "mixed"]},
}

# The 46 survey skip-logic columns identified during the audit that must never be predictive features
SKIP_LOGIC_SURVEY_COLUMNS = {
    "reason_of_drying",
    "years_since_dried",
    "revival_activity_done",
    "post_drying_action",
    "List the activities done for source revival",
    "flow_rate_lpm",
    "bucket_volume_liters",
    "fill_time_seconds",
    "fill_time_change_ratio",
    "fill_20l_before_10yr_min",
    "fill_20l_current_min",
    "discharge_trend",
    "discharge_nature",
    "emergence_point",
    "source_condition_detail",
    "earthquake_impact",
    "earthquake_impact_ordinal",
    "earthquake_impact_known",
    "water_quality",
    "water_quality_ordinal",
    "water_quality_change",
    "is_tapped",
    "is_tapped_binary",
    "Is this source tapped?_binary",
    "in_use",
    "in_use_binary",
    "Is this source in use?_binary",
    "community_accessible",
    "community_accessible_binary",
    "Is this source accessible to everyone in community? (in term of use access)_binary",
    "cultural_significance",
    "cultural_significance_binary",
    "Is water source significant for cultural/religious purpose?_binary",
    "has_user_group",
    "has_user_group_binary",
    "Are there water user groups for management?_binary",
    "user_group_type",
    "who_is_chairperson",
    "women_leadership",
    "revival_measures_taken",
    "revival_measures_detail",
    "purpose_of_use",
    "purpose_drinking",
    "purpose_household",
    "purpose_livestock",
    "purpose_agriculture",
    "purpose_irrigation",
    "hh_dependent_total",
    "hh_dependent_exclusive",
    "dependency_ratio",
    "distance_to_household_m",
    "fetch_time_minutes",
    "supply_municipality_scope",
    "supply_ward_scope",
    "ethnicity_users",
    "ethnicity_exclusive",
    "ethnicity_janajati",
    "ethnicity_bcts",
    "ethnicity_dalit",
    "ethnicity_mixed",
    "pond_area_m2",
    "pond_depth_m",
    "pond_area_10yr_m2",
    "pond_depth_10yr_m",
    "pond_area_change_ratio",
    "Remarks (source specific comment)",
    "List the events associated with sources",
    "If not who are restricted?",
}

LABEL_DERIVED_COLUMNS = {
    "target_dried",
    "target_trend",
    "dist_to_nearest_dried_m",
    "dist_to_nearest_active_m",
    "muni_drying_rate",
}

# Columns that are computed from the label (or only exist because of it) and must never be features.
FORBIDDEN_FEATURES = set(EARTHQUAKE_LEAKY) | SKIP_LOGIC_SURVEY_COLUMNS | LABEL_DERIVED_COLUMNS

# A model feature whose source column's missingness differs by more than this between
# dried and active sources is treated as skip-logic leakage.
MAX_MISSINGNESS_GAP = 0.05


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add circular encodings of aspect (raw degrees are not ordinal: 359 deg ~ 1 deg)."""
    df = df.copy()
    rad = np.radians(df["aspect_deg"].astype(float))
    df["aspect_sin"] = np.sin(rad)  # eastness
    df["aspect_cos"] = np.cos(rad)  # northness
    return df


def missingness_audit(df: pd.DataFrame, target: str = "target_dried") -> pd.DataFrame:
    """Missingness rate of every column among active vs dried sources."""
    y = df[target]
    rows = []
    for col in df.columns:
        if col == target:
            continue
        na = df[col].isna()
        rows.append({
            "column": col,
            "missing_active": float(na[y == 0].mean()),
            "missing_dried": float(na[y == 1].mean()),
        })
    out = pd.DataFrame(rows)
    out["missingness_gap"] = out["missing_dried"] - out["missing_active"]
    out["skip_logic_leak"] = out["missingness_gap"].abs() > MAX_MISSINGNESS_GAP
    return out.sort_values("missingness_gap", key=np.abs, ascending=False).reset_index(drop=True)


def check_feature_set(df: pd.DataFrame, feature_set: str, target: str = "target_dried") -> list[str]:
    """Return the list of features that violate the leakage rules (empty list == clean)."""
    feats = FEATURE_SETS[feature_set]
    audit = missingness_audit(df, target).set_index("column")
    violations = []
    for f in feats:
        if f in FORBIDDEN_FEATURES:
            violations.append(f"{f}: forbidden (label-derived or skip-logic leak)")
            continue
        src = FEATURE_SOURCE_COLUMN.get(f, f)
        if src in audit.index and audit.loc[src, "skip_logic_leak"]:
            violations.append(f"{f}: source column '{src}' missingness gap {audit.loc[src, 'missingness_gap']:+.3f}")
        elif f in audit.index and audit.loc[f, "skip_logic_leak"]:
            violations.append(f"{f}: column missingness gap {audit.loc[f, 'missingness_gap']:+.3f}")
    return violations


def single_feature_rule_scores(df: pd.DataFrame, target: str = "target_dried") -> pd.DataFrame:
    """ROC-AUC of each candidate feature used alone (no model) -- exposes near-label proxies."""
    from sklearn.metrics import roc_auc_score
    y = df[target].values
    rows = []
    for f in FEATURE_SETS["leaky_reference"]:
        v = df[f].astype(float).fillna(df[f].median()).values
        auc = roc_auc_score(y, v)
        rows.append({"feature": f, "single_feature_auc": max(auc, 1 - auc)})
    return pd.DataFrame(rows).sort_values("single_feature_auc", ascending=False).reset_index(drop=True)
