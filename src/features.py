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
SEASONALITY = ["Perennial / Seasonal_encoded"]
EARTHQUAKE_LEAKY = ["earthquake_impact_ordinal", "earthquake_impact_known"]

FEATURE_SETS = {
    "env": LOCATION + TERRAIN + CLIMATE + VEGETATION + SURVEY_STRUCTURAL,
    "env_season": LOCATION + TERRAIN + CLIMATE + VEGETATION + SURVEY_STRUCTURAL + SEASONALITY,
    "leaky_reference": LOCATION + TERRAIN + CLIMATE + VEGETATION + SURVEY_STRUCTURAL + SEASONALITY + EARTHQUAKE_LEAKY,
}
PRIMARY_FEATURE_SET = "env"

# Raw survey column each model feature is derived from (pre-encoding, pre-imputation),
# used by the audit so that fill values such as -1 cannot hide missingness.
FEATURE_SOURCE_COLUMN = {
    "Source Type_encoded": "source_type",
    "Perennial / Seasonal_encoded": "seasonality",
    "earthquake_impact_ordinal": "earthquake_impact",
    "earthquake_impact_known": "earthquake_impact",
    "aspect_sin": "aspect_deg",
    "aspect_cos": "aspect_deg",
    **{c: "landuse" for c in SURVEY_STRUCTURAL if c.startswith("landuse_")},
    **{c: "ownership" for c in SURVEY_STRUCTURAL if c.startswith("ownership_")},
}

# Columns that are computed from the label (or only exist because of it) and must never be features.
FORBIDDEN_FEATURES = set(EARTHQUAKE_LEAKY) | {
    "target_dried",
    "target_trend",
    "dist_to_nearest_dried_m",
    "dist_to_nearest_active_m",
    "muni_drying_rate",
}

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
            violations.append(f"{f}: forbidden (label-derived)")
            continue
        src = FEATURE_SOURCE_COLUMN.get(f, f)
        if src in audit.index and audit.loc[src, "skip_logic_leak"]:
            violations.append(f"{f}: source column '{src}' missingness gap {audit.loc[src, 'missingness_gap']:+.3f}")
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
