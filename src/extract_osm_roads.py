"""
OSM Road Network Extraction and Excavation Proximity Features for Kavre Springs.

Extracts localized OpenStreetMap road vectors for Kavrepalanchok District and computes:
1. dist_to_road_m: Euclidean distance in meters to the nearest road network segment (UTM 45N).
2. road_density_500m: Road network density (km/km^2) within a 500m radius of each spring.
3. road_upslope: Binary indicator (1 if nearest road elevation > spring elevation, 0 otherwise),
   representing potential interception/severing of recharge flow paths by upslope road cutting.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_ROADS_URL = "https://production-raw-data-api.s3.amazonaws.com/ISO3/NPL/_layers/osm/roads.parquet"
ROADS_PARQUET = PROJECT_ROOT / "data/processed/osm_roads_kavre.parquet"
ROAD_FEATURES_CSV = PROJECT_ROOT / "data/processed/osm_road_features.csv"
CLEAN_SPRINGS_CSV = PROJECT_ROOT / "data/processed/clean_springs.csv"
GEE_FEATURES_CSV = PROJECT_ROOT / "data/processed/gee_features.csv"
MASTER_PARQUET = PROJECT_ROOT / "data/processed/master_dataset.parquet"

KAVRE_ROW_GROUPS = [32, 33, 38, 39, 40, 44]


def extract_osm_roads_kavre(
    url: str = RAW_ROADS_URL,
    output_path: Path = ROADS_PARQUET,
    row_groups: list[int] = KAVRE_ROW_GROUPS,
) -> pd.DataFrame:
    """Extract all Kavrepalanchok road segments from the HDX Nepal roads parquet asset."""
    if output_path.exists():
        logger.info(f"Loading cached Kavre roads from {output_path}")
        return pd.read_parquet(output_path)

    import fsspec
    import pyarrow.parquet as pq

    logger.info(f"Extracting Kavre road segments from S3 parquet row groups: {row_groups}")
    t0 = time.time()
    dfs = []
    with fsspec.open(url, mode="rb") as f:
        pf = pq.ParquetFile(f)
        for rg in row_groups:
            tbl = pf.read_row_group(
                rg, columns=["id", "name", "highway", "surface", "adm2_name", "geom"]
            )
            df = tbl.to_pandas()
            k_df = df[df["adm2_name"] == "Kavrepalanchok"]
            dfs.append(k_df)
            logger.info(f"  Row group {rg}: {len(k_df)} Kavre road segments")

    kavre_roads = pd.concat(dfs, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kavre_roads.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(kavre_roads)} Kavre roads to {output_path} in {time.time()-t0:.1f}s")
    return kavre_roads


def compute_road_features(
    springs_df: pd.DataFrame,
    roads_df: pd.DataFrame,
    elevation_s: np.ndarray,
    batch_size: int = 2000,
) -> pd.DataFrame:
    """Compute dist_to_road_m, road_density_500m, and road_upslope for all springs."""
    logger.info("Projecting road and spring geometries to UTM Zone 45N (EPSG:32645)...")
    t0 = time.time()

    # Convert road WKB geometries to GeoSeries and project to UTM 45N (meters)
    road_geoms_wgs = gpd.GeoSeries.from_wkb(roads_df["geom"])
    roads_utm = gpd.GeoDataFrame(
        roads_df.drop(columns=["geom"]),
        geometry=road_geoms_wgs,
        crs="EPSG:4326",
    ).to_crs(epsg=32645)

    # Convert spring coordinates to UTM 45N
    springs_utm = gpd.GeoSeries.from_xy(
        springs_df["longitude"], springs_df["latitude"], crs="EPSG:4326"
    ).to_crs(epsg=32645)

    road_geoms = roads_utm.geometry.values
    spring_geoms = springs_utm.values

    # 1. Distance to nearest road
    logger.info("Computing Euclidean distance to nearest road network segment...")
    tree = shapely.STRtree(road_geoms)
    nearest_indices = tree.nearest(spring_geoms)
    nearest_road_geoms = road_geoms[nearest_indices]
    dist_to_road_m = shapely.distance(spring_geoms, nearest_road_geoms).astype(np.float32)

    # 2. Road density within 500m radius (km/km^2)
    logger.info("Computing 500m buffer road density...")
    buffers = shapely.buffer(spring_geoms, 500.0)
    b_idx, r_idx = tree.query(buffers, predicate="intersects")
    clipped_lines = shapely.intersection(buffers[b_idx], road_geoms[r_idx])
    lengths_m = shapely.length(clipped_lines)

    df_inter = pd.DataFrame({"buf_idx": b_idx, "len_m": lengths_m})
    grouped_lengths = df_inter.groupby("buf_idx")["len_m"].sum()

    total_lengths = np.zeros(len(spring_geoms), dtype=np.float64)
    total_lengths[grouped_lengths.index.values] = grouped_lengths.values
    circle_area_km2 = np.pi * (0.5**2)  # ~0.785398 km^2
    road_density_500m = ((total_lengths / 1000.0) / circle_area_km2).astype(np.float32)

    # 3. Road upslope cutting indicator
    logger.info("Computing nearest road point elevations to evaluate upslope cutting...")
    lines = shapely.shortest_line(spring_geoms, nearest_road_geoms)
    road_pts_utm = shapely.get_point(lines, -1)

    tf_to_wgs = Transformer.from_crs("EPSG:32645", "EPSG:4326", always_xy=True)
    road_lons, road_lats = tf_to_wgs.transform(
        shapely.get_x(road_pts_utm), shapely.get_y(road_pts_utm)
    )

    # Sample road elevation from GEE SRTM DEM
    import ee

    logger.info("Initializing GEE for SRTM elevation sampling at road coordinates...")
    ee.Initialize()
    srtm = ee.Image("USGS/SRTMGL1_003").select("elevation")

    road_elevations = np.zeros(len(spring_geoms), dtype=np.float32)
    n_points = len(spring_geoms)

    for start_idx in range(0, n_points, batch_size):
        end_idx = min(start_idx + batch_size, n_points)
        pts = [
            ee.Feature(
                ee.Geometry.Point([float(x), float(y)]),
                {"idx": int(i)},
            )
            for i, (x, y) in enumerate(
                zip(road_lons[start_idx:end_idx], road_lats[start_idx:end_idx]),
                start=start_idx,
            )
        ]
        fc = ee.FeatureCollection(pts)
        res = srtm.sampleRegions(collection=fc, scale=30, geometries=False).getInfo()
        for f in res["features"]:
            idx = f["properties"]["idx"]
            road_elevations[idx] = float(f["properties"]["elevation"])
        logger.info(f"  Sampled road elevations {start_idx+1} to {end_idx} ({len(res['features'])} points)")

    # road_upslope = 1 if nearest road elevation > spring elevation and road is not colocated
    road_upslope = ((road_elevations > elevation_s) & (dist_to_road_m > 1.0)).astype(int)

    out_df = pd.DataFrame(
        {
            "id": springs_df["id"].values,
            "dist_to_road_m": dist_to_road_m,
            "road_density_500m": road_density_500m,
            "road_upslope": road_upslope,
        }
    )

    logger.info(
        f"Road feature extraction complete in {time.time()-t0:.1f}s!\n"
        f"  dist_to_road_m: mean={dist_to_road_m.mean():.1f}m, median={np.median(dist_to_road_m):.1f}m\n"
        f"  road_density_500m: mean={road_density_500m.mean():.2f} km/km^2\n"
        f"  road_upslope count: {(road_upslope == 1).sum()} ({road_upslope.mean()*100:.1f}%)"
    )
    return out_df


def main():
    roads_df = extract_osm_roads_kavre()
    springs_df = pd.read_csv(CLEAN_SPRINGS_CSV)
    gee_df = pd.read_csv(GEE_FEATURES_CSV)

    merged_temp = springs_df[["id", "latitude", "longitude"]].merge(
        gee_df[["id", "elevation_m"]], on="id"
    )

    road_feats = compute_road_features(
        springs_df=merged_temp,
        roads_df=roads_df,
        elevation_s=merged_temp["elevation_m"].values,
    )

    road_feats.to_csv(ROAD_FEATURES_CSV, index=False)
    logger.info(f"Saved road features to {ROAD_FEATURES_CSV}")


if __name__ == "__main__":
    main()
