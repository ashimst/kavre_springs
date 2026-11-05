"""
Google Earth Engine (GEE) Covariate Extraction Pipeline for Kavre Springs.

Extracts topographic, hydro-climatic, and ecological covariates for all 5,689 points:
1. Topography: SRTM 30m (Elevation, Slope, Aspect)
2. Vegetation & Moisture: Sentinel-2 (NDVI mean, NDWI moisture)
3. Climate: CHIRPS Daily (Annual precipitation, Monsoon rainfall, Dry-season rainfall)

Clipped directly to the official OpenStreetMap Kavrepalanchok boundary polygon.
"""

import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INPUT_CSV = Path("data/processed/clean_springs.csv")
BOUNDARY_GEOJSON = Path("data/processed/kavre_district_boundary.geojson")
OUTPUT_CSV = Path("data/processed/gee_features.csv")
MASTER_PARQUET = Path("data/processed/master_dataset.parquet")


def extract_gee_covariates(
    input_csv: Path = INPUT_CSV,
    boundary_path: Path = BOUNDARY_GEOJSON,
    output_csv: Path = OUTPUT_CSV,
    batch_size: int = 1000,
):
    """Run GEE extraction for all spring points."""
    try:
        import ee
    except ImportError:
        logger.error("earthengine-api not installed. Install via: pip install earthengine-api")
        return None

    logger.info("Initializing Google Earth Engine...")
    ee.Initialize()
    logger.info("GEE successfully initialized!")

    # 1. Load spring coordinates
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} spring points from {input_csv}")

    # 2. Load official boundary geometry
    if boundary_path.exists():
        with open(boundary_path, "r", encoding="utf-8") as f:
            b_json = json.load(f)
        kavre_geom = ee.Geometry(b_json["features"][0]["geometry"])
        logger.info("Loaded official Kavre district boundary polygon.")
    else:
        kavre_geom = ee.Geometry.Rectangle([
            float(df["longitude"].min() - 0.05),
            float(df["latitude"].min() - 0.05),
            float(df["longitude"].max() + 0.05),
            float(df["latitude"].max() + 0.05),
        ])
        logger.info("Using bounding box geometry.")

    # 3. Construct multi-sensor satellite composite
    logger.info("Constructing satellite raster composite...")

    # A. Topography (SRTM 30m)
    srtm = ee.Image("USGS/SRTMGL1_003").clip(kavre_geom)
    elevation = srtm.select("elevation").rename("elevation_m")
    slope = ee.Terrain.slope(srtm).rename("slope_deg")
    aspect = ee.Terrain.aspect(srtm).rename("aspect_deg")

    # B. Precipitation (CHIRPS Daily 2018-2023, 6-year average)
    chirps = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(kavre_geom)
        .filterDate("2018-01-01", "2023-12-31")
    )
    annual_precip = chirps.sum().divide(6.0).rename("precip_annual_mm")
    monsoon_precip = (
        chirps.filter(ee.Filter.calendarRange(6, 9, "month"))
        .sum()
        .divide(6.0)
        .rename("precip_monsoon_mm")
    )
    dry_precip = (
        chirps.filter(ee.Filter.calendarRange(11, 2, "month"))
        .sum()
        .divide(6.0)
        .rename("precip_dry_season_mm")
    )

    # C. Sentinel-2 Cloud-Masked Multi-Year Composite (2022-2023)
    def mask_s2(img):
        qa = img.select("QA60")
        cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return img.updateMask(cloud_mask)

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(kavre_geom)
        .filterDate("2022-01-01", "2023-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 25))
        .map(mask_s2)
        .median()
    )

    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("ndvi_mean")
    ndwi = s2.normalizedDifference(["B3", "B8"]).rename("ndwi_mean")

    composite = ee.Image.cat([
        elevation,
        slope,
        aspect,
        annual_precip,
        monsoon_precip,
        dry_precip,
        ndvi,
        ndwi,
    ])

    band_names = composite.bandNames().getInfo()
    logger.info(f"Target raster bands ({len(band_names)}): {band_names}")

    # 4. Batch extraction across all 5,689 points
    extracted_records = []
    total_points = len(df)

    for i in range(0, total_points, batch_size):
        end_idx = min(i + batch_size, total_points)
        batch_df = df.iloc[i:end_idx]
        logger.info(f"Extracting batch [{i} to {end_idx}] ({len(batch_df)} points)...")

        features = [
            ee.Feature(
                ee.Geometry.Point([float(row["longitude"]), float(row["latitude"])]),
                {"id": int(row["id"])},
            )
            for _, row in batch_df.iterrows()
        ]

        fc = ee.FeatureCollection(features)
        sampled = composite.sampleRegions(
            collection=fc,
            scale=30,
            geometries=False,
        )

        items = sampled.getInfo()["features"]
        for item in items:
            extracted_records.append(item["properties"])

    result_df = pd.DataFrame(extracted_records)
    logger.info(f"Sampled {len(result_df)} points successfully.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)
    logger.info(f"Saved GEE covariates to {output_csv}")

    # 5. Merge with clean survey data into Master Dataset
    master_df = df.merge(result_df, on="id", how="left")
    master_df.to_parquet(MASTER_PARQUET, index=False)
    logger.info(f"Assembled master dataset: {master_df.shape[0]} rows, {master_df.shape[1]} columns")
    logger.info(f"Saved to {MASTER_PARQUET}")

    return master_df


if __name__ == "__main__":
    extract_gee_covariates()
