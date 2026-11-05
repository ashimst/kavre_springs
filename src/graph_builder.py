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
from torch_geometric.data import Data

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


def knn_edges(
    coords_m: np.ndarray,
    k: int = K_NEIGHBOURS,
    elev: np.ndarray | None = None,
    hnd: np.ndarray | None = None,
    upa: np.ndarray | None = None,
    ridge_threshold_m: float = 250.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct catchment-aware directed k-NN edges respecting flow divides and ridgelines.

    When elev is provided, candidate edges crossing steep ridges (|dz| > ridge_threshold_m)
    or climbing excessive topographic barriers are penalized so that springs connect along
    drainage paths and valley contours. Zero self-loops are strictly enforced.
    """
    n = len(coords_m)
    if n <= 1:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    tree = cKDTree(coords_m)
    k_actual = min(k, n - 1)
    k_pool = min(max(k_actual + 25, 40), n)
    _, idx_mat = tree.query(coords_m, k=k_pool)
    if k_pool == 1 or idx_mat.ndim == 1:
        idx_mat = idx_mat.reshape(n, -1)

    src_list, dst_list = [], []
    has_elev = elev is not None and len(elev) == n

    for i in range(n):
        row = idx_mat[i]
        cands = [j for j in row if j != i]
        if not cands:
            continue
        if len(cands) <= k_actual or not has_elev:
            chosen = cands[:k_actual]
        else:
            cands_arr = np.asarray(cands)
            delta = coords_m[cands_arr] - coords_m[i]
            d_i = np.maximum(np.linalg.norm(delta, axis=1), 1.0)
            dz_i = np.abs(elev[cands_arr] - elev[i])
            s_i = dz_i / d_i
            # Effective topographic / catchment resistance cost
            cost = d_i * (1.0 + 0.8 * (dz_i / 100.0) + 1.2 * np.maximum(0.0, s_i - 0.4))
            if hnd is not None and len(hnd) == n:
                dhnd_i = np.abs(hnd[cands_arr] - hnd[i])
                cost = cost + d_i * 0.3 * (dhnd_i / 50.0)
            if upa is not None and len(upa) == n:
                d_log_upa = np.abs(np.log10(np.maximum(upa[cands_arr], 1e-4)) - np.log10(np.maximum(upa[i], 1e-4)))
                cost = cost + d_i * 0.2 * np.minimum(d_log_upa, 3.0)
            # Severe penalty for crossing mountain ridges
            mask_ridge = dz_i > ridge_threshold_m
            cost[mask_ridge] += 10000.0 * (dz_i[mask_ridge] / ridge_threshold_m)
            best_idx = np.argsort(cost)[:k_actual]
            chosen = [cands[idx] for idx in best_idx]

        src_list.extend([i] * len(chosen))
        dst_list.extend(chosen)

    return np.array(src_list, dtype=np.int64), np.array(dst_list, dtype=np.int64)


def build_catchment_graph(
    coords_m: np.ndarray,
    elevation_m: np.ndarray,
    hnd_m: np.ndarray | None = None,
    upa_km2: np.ndarray | None = None,
    k: int = K_NEIGHBOURS,
    x: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
) -> Data:
    """Interface contract: construct a catchment-aware spatial PyG Data object."""
    src, dst = knn_edges(coords_m, k=k, elev=elevation_m, hnd=hnd_m, upa=upa_km2)
    edge_raw = physical_edge_attributes(coords_m, elevation_m, src, dst)
    edge_mu, edge_sd = edge_raw.mean(0), edge_raw.std(0)
    edge_sd = np.where(edge_sd == 0, 1.0, edge_sd)
    edge_attr = (edge_raw - edge_mu) / edge_sd
    data = Data(
        edge_index=torch.tensor(np.vstack([src, dst]), dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        coords_m=coords_m,
        elevation=elevation_m,
    )
    if x is not None:
        data.x = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
    if y is not None:
        data.y = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)
    return data


def load_master(path: Path = MASTER_DATA) -> pd.DataFrame:
    df = pd.read_parquet(path)
    hydro_path = path.parent / "gee_hydro_catchment_features.csv"
    if hydro_path.exists():
        df_h = pd.read_csv(hydro_path)
        cols_to_use = [c for c in df_h.columns if c not in df.columns or c == "id"]
        df = df.merge(df_h[cols_to_use], on="id", how="left")
    enrich_path = path.parent / "gee_enrichment_features.csv"  # produced by src.extract_enrichment_gee
    if enrich_path.exists():
        df_e = pd.read_csv(enrich_path)
        df = df.merge(df_e[[c for c in df_e.columns if c not in df.columns or c == "id"]], on="id", how="left")
    seismic_path = path.parent / "seismic_features.csv"  # produced by src.seismic (USGS ShakeMap / ComCat)
    if seismic_path.exists():
        df_s = pd.read_csv(seismic_path)
        df = df.merge(df_s[[c for c in df_s.columns if c not in df.columns or c == "id"]], on="id", how="left")
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
    hnd = df["hnd"].fillna(df["hnd"].median()).to_numpy(dtype=np.float64) if "hnd" in df.columns else None
    upa = df["upa"].fillna(df["upa"].median()).to_numpy(dtype=np.float64) if "upa" in df.columns else None
    src, dst = knn_edges(coords_m, k=k, elev=elev, hnd=hnd, upa=upa)
    edge_raw = physical_edge_attributes(coords_m, elev, src, dst)
    edge_mu, edge_sd = edge_raw.mean(0), edge_raw.std(0)
    edge_sd = np.where(edge_sd == 0, 1.0, edge_sd)
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
