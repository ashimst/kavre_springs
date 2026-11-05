"""
Standardized 5D Physical Hydro-Topographic Graph Construction.

Constructs:
1. k-NN Spatial Proximity Graph (k=7 nearest neighbors in metric space).
2. 5-Dimensional Purely Physical Edge Tensor (Zero Administrative Boundary Leakage):
   e_ij = [log1p(d_ij), unit_dx, unit_dy, delta_z, slope_gradient]
3. Standardized Node Features (25 non-leaking hydro-environmental covariates).
4. Provides edge standardization (StandardScaler) to prevent distance from swamping unit directional vectors.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ["MPLCONFIGDIR"] = os.path.join(str(PROJECT_ROOT), ".matplotlib")

import numpy as np
import pandas as pd
import torch
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MASTER_DATA = Path("data/processed/master_dataset.parquet")
OUTPUT_GRAPH_PATH = Path("data/graph/graph_knn_k7_physical.pt")

NODE_FEATURE_COLS = [
    # Geospatial Coordinates & Density
    "latitude",
    "longitude",
    "spring_density_500m",
    # Topographic (SRTM 30m)
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    # Hydro-Climatic (CHIRPS Daily)
    "precip_annual_mm",
    "precip_monsoon_mm",
    "precip_dry_season_mm",
    # Vegetation & Moisture (Sentinel-2)
    "ndvi_mean",
    "ndwi_mean",
    # Survey Environmental & Physical
    "Source Type_encoded",
    "Perennial / Seasonal_encoded",
    "earthquake_impact_ordinal",
    "earthquake_impact_known",
    # Land Use
    "landuse_agriculture",
    "landuse_dense_forest",
    "landuse_fallow",
    "landuse_grassland",
    "landuse_others",
    "landuse_settlement",
    "landuse_sparse_forest",
    # Ownership
    "ownership_community",
    "ownership_private",
    "ownership_public",
]


def coords_to_metric(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Project WGS84 coordinates to local metric coordinates centered at Kavre (27.57 N)."""
    lat_to_m = 111_320.0
    lon_to_m = 111_320.0 * np.cos(np.radians(27.57))
    x = (lon - lon.mean()) * lon_to_m
    y = (lat - lat.mean()) * lat_to_m
    return np.column_stack([x, y])


def build_physical_knn_graph(df: pd.DataFrame, k: int = 7) -> tuple:
    """
    Construct k-NN spatial graph with strictly physical 5D edge attributes:
    e_ij = [log1p(distance_m), unit_dx, unit_dy, delta_z_m, slope_gradient]
    Zero administrative boundaries (no same_municipality leakage).
    """
    logger.info(f"Building physical hydro-topographic graph (k={k} neighbors)...")
    coords_m = coords_to_metric(df["latitude"].values, df["longitude"].values)
    elev = df["elevation_m"].fillna(df["elevation_m"].median()).values

    tree = KDTree(coords_m)
    distances, indices = tree.query(coords_m, k=k + 1)

    sources = []
    targets = []
    edge_attrs_raw = []

    n_nodes = len(df)
    for i in range(n_nodes):
        for rank in range(1, k + 1):
            j = indices[i, rank]
            d = distances[i, rank]
            if d == 0:
                d = 1.0  # avoid div by zero

            dx = coords_m[j, 0] - coords_m[i, 0]
            dy = coords_m[j, 1] - coords_m[i, 1]
            unit_dx = dx / d
            unit_dy = dy / d
            dz = elev[j] - elev[i]  # upward elevation head difference
            slope_gradient = dz / d

            sources.append(i)
            targets.append(j)
            # 5D Physical Edge Vector
            edge_attrs_raw.append([
                np.log1p(d),       # log-transformed distance to reduce scale disparity
                unit_dx,           # unit directional x vector
                unit_dy,           # unit directional y vector
                dz,                # elevation head difference in meters
                slope_gradient,    # topographic slope gradient along conduit
            ])

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_attr_raw = np.array(edge_attrs_raw, dtype=np.float32)

    # Standardize edge attributes with StandardScaler to prevent log-distance from swamping unit directional vectors
    edge_scaler = StandardScaler()
    edge_attr_norm = edge_scaler.fit_transform(edge_attr_raw)
    edge_attr = torch.tensor(edge_attr_norm, dtype=torch.float32)

    logger.info(f"Constructed graph with {n_nodes:,} nodes and {edge_index.shape[1]:,} directed edges (edge_dim=5).")
    logger.info(f"Edge features standardized: means={edge_attr.mean(dim=0).numpy().round(3)}, stds={edge_attr.std(dim=0).numpy().round(3)}")

    return edge_index, edge_attr, edge_attr_raw, edge_scaler


def prepare_node_features(df: pd.DataFrame):
    """Extract and standardize 25 hydro-environmental node features."""
    available_cols = [c for c in NODE_FEATURE_COLS if c in df.columns]
    X_raw = df[available_cols].values.astype(np.float32)

    # Median impute any missing GEE values
    for col_idx in range(X_raw.shape[1]):
        col_vals = X_raw[:, col_idx]
        if np.isnan(col_vals).any():
            med = np.nanmedian(col_vals)
            X_raw[np.isnan(col_vals), col_idx] = med

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_raw)

    x = torch.tensor(X_norm, dtype=torch.float32)
    y_dried = torch.tensor(df["target_dried"].values, dtype=torch.long)

    return x, y_dried, available_cols, scaler


def build_and_save_physical_graph():
    df = pd.read_parquet(MASTER_DATA)
    edge_index, edge_attr, edge_attr_raw, edge_scaler = build_physical_knn_graph(df, k=7)
    x, y_dried, feature_cols, node_scaler = prepare_node_features(df)

    graph_dict = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_attr_raw": torch.tensor(edge_attr_raw, dtype=torch.float32),
        "y_dried": y_dried,
        "num_nodes": len(df),
        "num_edges": edge_index.shape[1],
        "feature_cols": feature_cols,
        "latitude": torch.tensor(df["latitude"].values, dtype=torch.float32),
        "longitude": torch.tensor(df["longitude"].values, dtype=torch.float32),
        "municipality": df["municipality"].tolist(),
        "id": df["id"].tolist(),
    }

    OUTPUT_GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph_dict, OUTPUT_GRAPH_PATH)
    logger.info(f"Successfully persisted standardized 5D physical graph to {OUTPUT_GRAPH_PATH}")
    return graph_dict


if __name__ == "__main__":
    build_and_save_physical_graph()

