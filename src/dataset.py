"""
The analysis table and the design matrices built from it.

load_master       master table (src/assemble_master.py) joined with the Earth Engine enrichment layers and the USGS
                  seismic exposure; derived columns from src.features.add_derived_columns
coords_to_metric  equirectangular projection to metres (error < 0.1 % over the 40 km study area)
design_matrix     features of one named feature set (src.features.FEATURE_SETS), median-imputed and z-scored over
                  all sources (label-free), with labels, coordinates and identifiers
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_SETS, PRIMARY_FEATURE_SET, add_derived_columns

logger = logging.getLogger(__name__)

MASTER_DATA = PROJECT_ROOT / "data/processed/master_dataset.parquet"
REF_LATITUDE = 27.57


def coords_to_metric(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Equirectangular projection to metres, centred on the study-area mean."""
    lat_to_m = 111_320.0
    lon_to_m = 111_320.0 * np.cos(np.radians(REF_LATITUDE))
    return np.column_stack([(lon - lon.mean()) * lon_to_m, (lat - lat.mean()) * lat_to_m])


def load_master(path: Path = MASTER_DATA) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for name in ["gee_hydro_catchment_features.csv",   # src/extract_hydro_catchment_gee.py (already in master)
                 "gee_enrichment_features.csv",         # src/extract_enrichment_gee.py
                 "seismic_features.csv"]:               # src/seismic.py
        p = path.parent / name
        if p.exists():
            t = pd.read_csv(p)
            df = df.merge(t[[c for c in t.columns if c not in df.columns or c == "id"]], on="id", how="left")
    return add_derived_columns(df)


def design_matrix(df: pd.DataFrame, feature_set: str = PRIMARY_FEATURE_SET) -> dict:
    """Median-imputed, z-scored features of one feature set, labels and source metadata."""
    cols = FEATURE_SETS[feature_set]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Feature columns missing from master dataset: {missing}")
    X = df[cols].astype(np.float32).to_numpy()
    X_raw = X.copy()
    n_imputed = int(np.isnan(X).sum())
    med = np.nanmedian(X, axis=0)
    r, c = np.where(np.isnan(X))
    X[r, c] = med[c]
    X = StandardScaler().fit_transform(X).astype(np.float32)
    logger.info(f"[{feature_set}] {len(df)} sources, {len(cols)} features, {n_imputed} imputed values")
    return {
        "feature_set": feature_set,
        "feature_cols": list(cols),
        "x": X,
        "x_raw": X_raw,  # unimputed, unscaled: src.benchmark fits imputation and scaling on training rows only
        "y": df["target_dried"].to_numpy().astype(np.int64),
        "coords_m": coords_to_metric(df["latitude"].to_numpy(), df["longitude"].to_numpy()),
        "municipality": df["municipality"].to_numpy(),
        "id": df["id"].to_numpy(),
    }
