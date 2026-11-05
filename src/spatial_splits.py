"""
Spatial cross-validation splitting strategies for Kavre Springs dataset.

To prevent spatial data leakage, we implement:
1. Spatial Block Cross-Validation (Transductive): Divide the study area into
   spatial grid blocks; group springs by block to ensure no training springs
   are spatially adjacent to test springs.
2. Leave-One-Municipality-Out (LOMO) (Inductive): Train on 6 municipalities,
   evaluate on 1 held-out municipality. Evaluates regional transferability.
"""

import numpy as np
import pandas as pd
from typing import Generator, Tuple, List, Dict
from sklearn.model_selection import KFold


def create_spatial_grid_blocks(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    n_lat_bins: int = 5,
    n_lon_bins: int = 5,
) -> np.ndarray:
    """
    Divide bounding box into a 2D regular grid and assign each point a block ID.
    """
    lat_bins = np.linspace(latitudes.min() - 1e-5, latitudes.max() + 1e-5, n_lat_bins + 1)
    lon_bins = np.linspace(longitudes.min() - 1e-5, longitudes.max() + 1e-5, n_lon_bins + 1)

    lat_idx = np.digitize(latitudes, lat_bins) - 1
    lon_idx = np.digitize(longitudes, lon_bins) - 1

    block_ids = lat_idx * n_lon_bins + lon_idx
    return block_ids


def spatial_block_kfold(
    df: pd.DataFrame,
    n_splits: int = 5,
    n_blocks_side: int = 5,
    random_state: int = 42,
) -> List[Dict[str, np.ndarray]]:
    """
    Generate 5-fold spatial cross-validation masks (train, val, test) for transductive modeling.
    Returns list of dicts with boolean masks for each fold.
    """
    lat = df["latitude"].values
    lon = df["longitude"].values

    block_ids = create_spatial_grid_blocks(lat, lon, n_blocks_side, n_blocks_side)
    unique_blocks = np.unique(block_ids)

    # Shuffle blocks
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_blocks)

    # Group blocks into n_splits folds
    block_folds = np.array_split(unique_blocks, n_splits)

    splits = []
    n_samples = len(df)

    for i in range(n_splits):
        test_blocks = block_folds[i]
        val_blocks = block_folds[(i + 1) % n_splits]
        train_blocks = [b for b in unique_blocks if b not in test_blocks and b not in val_blocks]

        test_mask = np.isin(block_ids, test_blocks)
        val_mask = np.isin(block_ids, val_blocks)
        train_mask = np.isin(block_ids, train_blocks)

        splits.append({
            "fold": i,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_test": int(test_mask.sum()),
        })

    return splits


def leave_one_municipality_out(df: pd.DataFrame) -> List[Dict[str, any]]:
    """
    Generate Leave-One-Municipality-Out (LOMO) splits for inductive transferability.
    """
    municipalities = sorted(df["municipality"].unique())
    splits = []

    for test_muni in municipalities:
        test_mask = (df["municipality"] == test_muni).values
        # Use one remaining muni as validation, the others as train
        remaining_munis = [m for m in municipalities if m != test_muni]
        val_muni = remaining_munis[0]
        val_mask = (df["municipality"] == val_muni).values
        train_mask = (~test_mask) & (~val_mask)

        splits.append({
            "test_municipality": test_muni,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_test": int(test_mask.sum()),
        })

    return splits


if __name__ == "__main__":
    df = pd.read_csv("data/processed/clean_springs.csv")
    splits = spatial_block_kfold(df, n_splits=5)
    print("=== Spatial Block 5-Fold Splits ===")
    for s in splits:
        print(f"Fold {s['fold']}: Train={s['n_train']}, Val={s['n_val']}, Test={s['n_test']}")

    print("\n=== Leave-One-Municipality-Out Splits ===")
    lomo_splits = leave_one_municipality_out(df)
    for s in lomo_splits:
        print(f"Test on {s['test_municipality']:15s}: Train={s['n_train']}, Val={s['n_val']}, Test={s['n_test']}")

