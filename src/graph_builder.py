"""
k-NN spring graph with physical edge attributes.

Graph
-----
Directed k-nearest-neighbour graph (k = 7) in a local metric projection.
For every spring i and each of its k nearest neighbours j we add the edge
(i -> j) stored as ``edge_index[:, e] = [i, j]``. In PyTorch Geometric's
message-passing convention (source_to_target) node j therefore *receives* a
message from i. Attention coefficients are normalised over each receiver's
incoming edges.

Edge attributes (exactly as implemented; Text S1 of the SI matches this)
------------------------------------------------------------------------
For edge (i -> j) with planar distance d_ij (metres; d := 1 m when two
springs share coordinates):

    e_ij = [ log(1 + d_ij),
             (x_j - x_i) / d_ij,
             (y_j - y_i) / d_ij,
             dz_ij = z_j - z_i,           # metres; > 0  => receiver j is upslope of sender i
             dz_ij / d_ij ]               # along-edge gradient

Each column is z-scored over all edges. Edge attributes contain no label
information, so this is not leakage; it is stated explicitly in the SI.

Node features
-------------
Chosen by name from ``src.features.FEATURE_SETS``. Missing remote-sensing
values (22 springs fall outside valid Sentinel-2 / CHIRPS pixels) are
median-imputed and all features are z-scored over all nodes (unsupervised,
label-free; standard for transductive node classification).
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
import torch
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_SETS, PRIMARY_FEATURE_SET, add_derived_columns

logger = logging.getLogger(__name__)

MASTER_DATA = PROJECT_ROOT / "data/processed/master_dataset.parquet"
GRAPH_DIR = PROJECT_ROOT / "data/graph"
K_NEIGHBOURS = 7
REF_LATITUDE = 27.57


def coords_to_metric(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Equirectangular projection to metres, centred on the study-area mean (error < 0.1 % over 40 km)."""
    lat_to_m = 111_320.0
    lon_to_m = 111_320.0 * np.cos(np.radians(REF_LATITUDE))
    x = (lon - lon.mean()) * lon_to_m
    y = (lat - lat.mean()) * lat_to_m
    return np.column_stack([x, y])


def physical_edge_attributes(coords_m: np.ndarray, elev: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Raw (unscaled) 5-D edge attributes for edges src -> dst."""
    delta = coords_m[dst] - coords_m[src]
    d = np.linalg.norm(delta, axis=1)
    d = np.where(d == 0, 1.0, d)
    dz = elev[dst] - elev[src]
    return np.column_stack([np.log1p(d), delta[:, 0] / d, delta[:, 1] / d, dz, dz / d]).astype(np.float32)


def knn_edges(coords_m: np.ndarray, k: int = K_NEIGHBOURS) -> tuple[np.ndarray, np.ndarray]:
    _, idx = cKDTree(coords_m).query(coords_m, k=k + 1)
    n = len(coords_m)
    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1:].reshape(-1)
    return src, dst


def load_master(path: Path = MASTER_DATA) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return add_derived_columns(df)


def build_graph(df: pd.DataFrame, feature_set: str = PRIMARY_FEATURE_SET, k: int = K_NEIGHBOURS) -> dict:
    """Assemble node features, labels and the physical k-NN graph for one feature set."""
    cols = FEATURE_SETS[feature_set]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Feature columns missing from master dataset: {missing}")

    X = df[cols].astype(np.float32).to_numpy()
    n_imputed = int(np.isnan(X).sum())
    col_median = np.nanmedian(X, axis=0)
    nan_r, nan_c = np.where(np.isnan(X))
    X[nan_r, nan_c] = col_median[nan_c]
    X_raw = X.copy()
    X = StandardScaler().fit_transform(X).astype(np.float32)

    coords_m = coords_to_metric(df["latitude"].to_numpy(), df["longitude"].to_numpy())
    elev = df["elevation_m"].fillna(df["elevation_m"].median()).to_numpy(dtype=np.float64)
    src, dst = knn_edges(coords_m, k)
    edge_raw = physical_edge_attributes(coords_m, elev, src, dst)
    edge_mu, edge_sd = edge_raw.mean(0), edge_raw.std(0)
    edge_attr = (edge_raw - edge_mu) / edge_sd

    logger.info(f"[{feature_set}] {len(df)} nodes, {len(src)} edges, {len(cols)} features, {n_imputed} imputed values")
    return {
        "feature_set": feature_set,
        "feature_cols": list(cols),
        "x": torch.tensor(X),
        "x_raw": X_raw,
        "y": torch.tensor(df["target_dried"].to_numpy(), dtype=torch.long),
        "edge_index": torch.tensor(np.vstack([src, dst]), dtype=torch.long),
        "edge_attr": torch.tensor(edge_attr, dtype=torch.float32),
        "edge_attr_raw": edge_raw,
        "edge_scaler": (edge_mu, edge_sd),
        "coords_m": coords_m,
        "elevation": elev,
        "latitude": df["latitude"].to_numpy(),
        "longitude": df["longitude"].to_numpy(),
        "municipality": df["municipality"].to_numpy(),
        "id": df["id"].to_numpy(),
    }


def rewired_graph(g: dict, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Null-model topology: same out-degree, neighbours drawn uniformly at random from the
    whole district; edge attributes recomputed from the true geometry of the random pairs
    and scaled with the original edge scaler."""
    n = len(g["y"])
    k = g["edge_index"].shape[1] // n
    src = np.repeat(np.arange(n), k)
    dst = rng.integers(0, n - 1, size=n * k)
    dst = dst + (dst >= src)  # no self-edges
    raw = physical_edge_attributes(g["coords_m"], g["elevation"], src, dst)
    mu, sd = g["edge_scaler"]
    return torch.tensor(np.vstack([src, dst]), dtype=torch.long), torch.tensor((raw - mu) / sd, dtype=torch.float32)


def save_graph(g: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in g.items() if k not in ("edge_scaler",)}
    keep["edge_attr_raw"] = torch.tensor(g["edge_attr_raw"])
    keep["municipality"] = list(g["municipality"])
    keep["id"] = list(g["id"])
    keep["y_dried"] = g["y"]
    torch.save(keep, path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    g = build_graph(load_master(), PRIMARY_FEATURE_SET)
    save_graph(g, GRAPH_DIR / f"graph_knn_k{K_NEIGHBOURS}_{PRIMARY_FEATURE_SET}.pt")
