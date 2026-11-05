"""
Fast, robust extraction of hydro-catchment, terrain morphometry, and decadal climate
anomalies for all 5,689 springs in Kavrepalanchok District using Google Earth Engine.

Features extracted:
1. MERIT Hydro:
   - upa: Upstream Contributing Area (km^2)
   - hnd: Height Above Nearest Drainage (m)
   - elv: Hydro-enforced elevation (m)
2. SRTM 30m Topography & Morphometry:
   - twi: Topographic Wetness Index ln(upa * 1e6 / (tan(slope) + 0.001))
   - tpi_500m: Topographic Position Index (500m window)
   - tri_90m: Terrain Ruggedness Index |DEM - mean(90m)|
3. TerraClimate Decadal Climate Anomalies:
   - precip_dry_deficit_mm: Decadal dry-season precipitation anomaly (2014-2023 vs 2000-2013)
   - precip_mon_deficit_mm: Decadal monsoon precipitation anomaly (2014-2023 vs 2000-2013)
   - aet: Actual Evapotranspiration (mm, 2018-2023 mean)
   - def: Climatic Water Deficit (mm, 2018-2023 mean)
   - soil: Root-zone Soil Moisture (mm, 2018-2023 mean)
   - pdsi: Palmer Drought Severity Index (2018-2023 mean)
"""

import sys, os, time
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_CSV = PROJECT_ROOT / "data/processed/gee_hydro_catchment_features.csv"
MASTER_PARQUET = PROJECT_ROOT / "data/processed/master_dataset.parquet"

def extract_all_hydro_features(batch_size: int = 1000):
    import ee
    print("Initializing Google Earth Engine...", flush=True)
    ee.Initialize()
    print("Google Earth Engine initialized successfully.", flush=True)

    df_master = pd.read_parquet(MASTER_PARQUET)
    print(f"Loaded master dataset: {len(df_master):,} springs.", flush=True)

    # 1. Define Rasters
    # A. MERIT Hydro
    merit = ee.Image("MERIT/Hydro/v1_0_1")
    upa = merit.select("upa")
    hnd = merit.select("hnd")
    elv = merit.select("elv")

    # B. SRTM Topography & Morphometry
    srtm = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(srtm).rename("slope_deg")
    aspect = ee.Terrain.aspect(srtm).rename("aspect_deg")

    twi = ee.Image().expression(
        "log((upa * 1e6) / (tan(slope * 0.0174532925) + 0.001))",
        {"upa": upa, "slope": slope}
    ).rename("twi")

    tpi_500 = srtm.subtract(srtm.focal_mean(500, "circle", "meters")).rename("tpi_500m")
    tri_90 = srtm.subtract(srtm.focal_mean(90, "circle", "meters")).abs().rename("tri_90m")

    # C. TerraClimate Decadal Trends & Moisture Regime
    terra = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")

    base_dry = terra.filterDate("2000-01-01", "2014-01-01").filter(ee.Filter.calendarRange(11, 2, "month")).select("pr").sum().divide(14.0)
    rec_dry = terra.filterDate("2014-01-01", "2024-01-01").filter(ee.Filter.calendarRange(11, 2, "month")).select("pr").sum().divide(10.0)
    dry_deficit = rec_dry.subtract(base_dry).rename("precip_dry_deficit_mm")

    base_mon = terra.filterDate("2000-01-01", "2014-01-01").filter(ee.Filter.calendarRange(6, 9, "month")).select("pr").sum().divide(14.0)
    rec_mon = terra.filterDate("2014-01-01", "2024-01-01").filter(ee.Filter.calendarRange(6, 9, "month")).select("pr").sum().divide(10.0)
    mon_deficit = rec_mon.subtract(base_mon).rename("precip_mon_deficit_mm")

    terra_recent = terra.filterDate("2018-01-01", "2023-12-31").select(["aet", "def", "soil", "pdsi"]).mean()

    composite = ee.Image.cat([
        upa, hnd, elv, twi, tpi_500, tri_90,
        dry_deficit, mon_deficit, terra_recent
    ])

    # 2. Batch Sampling
    all_rows = []
    n_total = len(df_master)
    print(f"Extracting hydro-catchment features for {n_total:,} springs in batches of {batch_size}...", flush=True)

    t0_all = time.time()
    for start_idx in range(0, n_total, batch_size):
        end_idx = min(start_idx + batch_size, n_total)
        sub_df = df_master.iloc[start_idx:end_idx]
        t0 = time.time()

        pts = [
            ee.Feature(ee.Geometry.Point([float(row["longitude"]), float(row["latitude"])]), {"id": int(row["id"])})
            for _, row in sub_df.iterrows()
        ]
        fc = ee.FeatureCollection(pts)

        sampled = composite.sampleRegions(collection=fc, scale=30, geometries=False).getInfo()
        rows = [f["properties"] for f in sampled["features"]]
        all_rows.extend(rows)

        print(f"  Sampled springs {start_idx+1:,} to {end_idx:,} ({len(rows)} points) in {time.time()-t0:.1f}s", flush=True)

    res_df = pd.DataFrame(all_rows)
    print(f"\nExtraction complete! Total sampled: {len(res_df):,} in {time.time()-t0_all:.1f}s", flush=True)

    # Ensure consistent column ordering and merge with master IDs
    res_df = df_master[["id"]].merge(res_df, on="id", how="left")
    res_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved hydro-catchment features to {OUTPUT_CSV}", flush=True)
    return res_df

if __name__ == "__main__":
    extract_all_hydro_features(batch_size=1000)
