"""
Stage 0: rebuild the analysis table from the raw ICIMOD inventory, offline.

    data/raw/spring_data.csv  --src.data_cleaning-->  data/processed/clean_springs.csv
    + data/processed/gee_features.csv                 (src/gee_extraction.py, Earth Engine; cached)
    + data/processed/osm_road_features.csv            (src/extract_osm_roads.py, OpenStreetMap; cached)
    + data/processed/gee_hydro_catchment_features.csv (src/extract_hydro_catchment_gee.py, Earth Engine; cached)
    = data/processed/master_dataset.parquet

The Earth Engine and OpenStreetMap extractions need credentials or network access and are run once; their outputs
are versioned in data/processed so that the whole analysis is reproducible without them. The enrichment layers
(gee_enrichment_features.csv) and seismic exposure (seismic_features.csv) are joined later by
src.dataset.load_master.

Usage:  python -m src.assemble_master [--check]   (--check: compare with the versioned master and exit 1 on mismatch)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging

import numpy as np
import pandas as pd

from src.data_cleaning import run_cleaning_pipeline

logger = logging.getLogger("assemble_master")
PROC = PROJECT_ROOT / "data/processed"
MASTER = PROC / "master_dataset.parquet"
COVARIATE_TABLES = ["gee_features.csv", "osm_road_features.csv", "gee_hydro_catchment_features.csv"]


def assemble() -> pd.DataFrame:
    df = run_cleaning_pipeline()
    for name in COVARIATE_TABLES:
        t = pd.read_csv(PROC / name)
        df = df.merge(t[[c for c in t.columns if c not in df.columns or c == "id"]], on="id", how="left")
    return df


def same_table(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if set(a.columns) != set(b.columns) or len(a) != len(b):
        return False
    b = b[a.columns]
    for c in a.columns:
        if a[c].equals(b[c]):
            continue
        x, y = pd.to_numeric(a[c], errors="coerce"), pd.to_numeric(b[c], errors="coerce")
        if x.isna().all() or not np.allclose(x, y, equal_nan=True):
            return False
    return True


def run(check: bool = False) -> pd.DataFrame:
    df = assemble()
    if check:
        ok = MASTER.exists() and same_table(df, pd.read_parquet(MASTER))
        logger.info(f"Rebuilt master {'matches' if ok else 'DIFFERS FROM'} the versioned {MASTER.name}")
        if not ok:
            raise SystemExit(1)
        return df
    df.to_parquet(MASTER, index=False)
    logger.info(f"Wrote {MASTER} ({df.shape[0]} rows, {df.shape[1]} columns)")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    run(check=ap.parse_args().check)
