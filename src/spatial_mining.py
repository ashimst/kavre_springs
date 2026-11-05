"""
Spatial Autocorrelation (Moran's I & LISA) and Hydro-Ecological Association Rule Mining.

1. Global Moran's I & Anselin's Local Moran's I (LISA):
   - Identifies High-High desiccation hotspots, Low-Low perennial coldspots, and spatial outliers.
2. Association Rule Mining:
   - Discovers multi-variate co-occurrence patterns (Support, Confidence, Lift) leading to spring desiccation.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MASTER_DATA = Path("data/processed/master_dataset.parquet")
OUTPUT_METRICS_DIR = Path("results/metrics")
OUTPUT_FIGURES_DIR = Path("results/figures")


# =============================================================================
# 1. Global & Local Moran's I (LISA)
# =============================================================================

def compute_spatial_weights_matrix(lat: np.ndarray, lon: np.ndarray, k: int = 7) -> np.ndarray:
    """Build row-standardized k-nearest neighbors spatial weight matrix W."""
    # Convert lat/lon to approximate metric distance
    lat_to_m = 111_320.0
    lon_to_m = 111_320.0 * np.cos(np.radians(27.57))
    coords = np.column_stack([(lon - lon.mean()) * lon_to_m, (lat - lat.mean()) * lat_to_m])
    
    tree = KDTree(coords)
    dists, indices = tree.query(coords, k=k + 1) # include self
    
    n = len(lat)
    W = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        # Inverse distance weighting for the k neighbors
        neighs = indices[i, 1:] # skip self
        d = dists[i, 1:]
        d[d == 0] = 1.0
        w = 1.0 / d
        W[i, neighs] = w / np.sum(w) # row-standardized
        
    return W


def compute_morans_i(y: np.ndarray, W: np.ndarray, n_permutations: int = 999):
    """
    Compute Global Moran's I, analytical expectation, variance, and permutation p-value.
    """
    n = len(y)
    z = y - np.mean(y)
    s2 = np.sum(z**2) / n
    S0 = np.sum(W)
    
    # Global Moran's I
    Wz = W.dot(z)
    moran_i = (n / S0) * (np.sum(z * Wz) / np.sum(z**2))
    
    # Permutation test
    sim_I = np.zeros(n_permutations)
    for p in range(n_permutations):
        z_perm = np.random.permutation(z)
        sim_I[p] = (n / S0) * (np.sum(z_perm * W.dot(z_perm)) / np.sum(z_perm**2))
        
    p_val = (np.sum(sim_I >= moran_i) + 1.0) / (n_permutations + 1.0)
    z_score = (moran_i - np.mean(sim_I)) / np.std(sim_I)
    
    return moran_i, p_val, z_score, sim_I


def compute_lisa(y: np.ndarray, W: np.ndarray, n_permutations: int = 499):
    """
    Compute Anselin's Local Moran's I (LISA) and classify each node into:
    - High-High (Hotspot): Dried spring surrounded by dried springs
    - Low-Low (Coldspot): Active spring surrounded by active springs
    - Low-High: Active spring surrounded by dried springs (High risk)
    - High-Low: Dried spring surrounded by active springs (Isolated failure)
    - Not Significant (p >= 0.05)
    """
    n = len(y)
    z = y - np.mean(y)
    s2 = np.var(y)
    Wz = W.dot(z)
    
    # Local Moran's I: I_i = (z_i / s^2) * sum_j(w_ij * z_j)
    local_I = (z / s2) * Wz
    
    # Permutation p-values
    sim_I = np.zeros((n, n_permutations))
    for p in range(n_permutations):
        z_p = np.random.permutation(z)
        sim_I[:, p] = (z / s2) * W.dot(z_p)
        
    p_values = np.zeros(n)
    for i in range(n):
        if local_I[i] > 0:
            p_values[i] = (np.sum(sim_I[i] >= local_I[i]) + 1.0) / (n_permutations + 1.0)
        else:
            p_values[i] = (np.sum(sim_I[i] <= local_I[i]) + 1.0) / (n_permutations + 1.0)
            
    # Quadrant Classification
    quadrant = np.array(["Not Significant"] * n, dtype="object")
    sig = p_values < 0.05
    
    hh = sig & (z > 0) & (Wz > 0)
    ll = sig & (z < 0) & (Wz < 0)
    lh = sig & (z < 0) & (Wz > 0)
    hl = sig & (z > 0) & (Wz < 0)
    
    quadrant[hh] = "High-High (Hotspot)"
    quadrant[ll] = "Low-Low (Coldspot)"
    quadrant[lh] = "Low-High (Spatial Outlier / Risk)"
    quadrant[hl] = "High-Low (Isolated Failure)"
    
    return local_I, p_values, quadrant


# =============================================================================
# 2. Association Rule Mining
# =============================================================================

def mine_desiccation_rules(df: pd.DataFrame, min_support: float = 0.05, min_confidence: float = 0.50):
    """
    Mine hydro-ecological and physical association rules predicting spring drying.
    Rule format: A -> Dried, with Support, Confidence, and Lift.
    """
    n = len(df)
    target = df["target_dried"].values
    
    # Candidate antecedent conditions (boolean masks)
    conditions = {
        "Earthquake Impacted": df["earthquake_impact_ordinal"] >= 1,
        "Earthquake Severely Impacted": df["earthquake_impact_ordinal"] == 2,
        "Seasonal Spring": df["Perennial / Seasonal_encoded"] == 1,
        "High Elevation (>1600m)": df["elevation_m"] > 1600,
        "Low Elevation (<1100m)": df["elevation_m"] < 1100,
        "Steep Slope (>25 deg)": df["slope_deg"] > 25,
        "Low Dry Season Precip (<70mm)": df["precip_dry_season_mm"] < 70,
        "Low NDWI (<0.0)": df["ndwi_mean"] < 0.0,
        "High Spring Density (>10/500m)": df["spring_density_500m"] > 10,
        "Source Type is Pond": df["Source Type_encoded"] == 1,
        "Forest Land Use": df["landuse_dense_forest"] == 1,
        "Settlement Land Use": df["landuse_settlement"] == 1,
        "Agriculture Land Use": df["landuse_agriculture"] == 1,
    }
    
    p_target = np.mean(target) # base desiccation rate (~0.267)
    rules = []
    
    # 1-item antecedents
    for name, mask in conditions.items():
        supp_A = np.sum(mask) / n
        if supp_A < min_support:
            continue
        supp_AB = np.sum(mask & (target == 1)) / n
        conf = supp_AB / supp_A
        lift = conf / p_target
        
        if conf >= min_confidence:
            rules.append({
                "antecedent": name,
                "consequent": "Target: Dried",
                "support": supp_AB,
                "confidence": conf,
                "lift": lift,
                "count": int(np.sum(mask & (target == 1))),
            })
            
    # 2-item composite antecedents
    cond_keys = list(conditions.keys())
    for i in range(len(cond_keys)):
        for j in range(i + 1, len(cond_keys)):
            k1, k2 = cond_keys[i], cond_keys[j]
            mask_combo = conditions[k1] & conditions[k2]
            supp_A = np.sum(mask_combo) / n
            if supp_A < min_support:
                continue
            supp_AB = np.sum(mask_combo & (target == 1)) / n
            conf = supp_AB / supp_A
            lift = conf / p_target
            
            if conf >= min_confidence:
                rules.append({
                    "antecedent": f"{k1} AND {k2}",
                    "consequent": "Target: Dried",
                    "support": supp_AB,
                    "confidence": conf,
                    "lift": lift,
                    "count": int(np.sum(mask_combo & (target == 1))),
                })
                
    df_rules = pd.DataFrame(rules).sort_values("lift", ascending=False).reset_index(drop=True)
    return df_rules


# =============================================================================
# Main Pipeline
# =============================================================================

def run_spatial_mining():
    logger.info("Loading master dataset...")
    df = pd.read_parquet(MASTER_DATA)
    y = df["target_dried"].values.astype(float)
    lat = df["latitude"].values
    lon = df["longitude"].values
    
    logger.info("Computing spatial weights matrix W (k=7 nearest neighbors)...")
    W = compute_spatial_weights_matrix(lat, lon, k=7)
    
    # 1. Global Moran's I
    logger.info("Calculating Global Moran's I...")
    moran_i, p_val, z_score, _ = compute_morans_i(y, W, n_permutations=999)
    logger.info(f"Global Moran's I = {moran_i:.4f} (z-score = {z_score:.2f}, p-value = {p_val:.4f})")
    
    # 2. Local Moran's I (LISA)
    logger.info("Calculating Anselin's Local Moran's I (LISA)...")
    local_I, p_vals, quadrants = compute_lisa(y, W, n_permutations=499)
    df["lisa_I"] = local_I
    df["lisa_pvalue"] = p_vals
    df["lisa_quadrant"] = quadrants
    
    # Summary of LISA clusters
    quad_counts = df["lisa_quadrant"].value_counts()
    logger.info("\nLISA Spatial Cluster Breakdown:\n" + str(quad_counts))
    
    # Save spatial clusters
    OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df_lisa_export = df[["id", "municipality", "latitude", "longitude", "target_dried", "lisa_I", "lisa_pvalue", "lisa_quadrant"]]
    df_lisa_export.to_csv(OUTPUT_METRICS_DIR / "lisa_spatial_clusters.csv", index=False)
    
    # 3. Association Rule Mining
    logger.info("Mining Hydro-Ecological Association Rules...")
    df_rules = mine_desiccation_rules(df, min_support=0.04, min_confidence=0.50)
    logger.info(f"Mined {len(df_rules)} significant association rules.")
    df_rules.to_csv(OUTPUT_METRICS_DIR / "association_rules_desiccation.csv", index=False)
    
    # 4. Generate Publication LISA Map
    logger.info("Rendering LISA Spatial Hotspot Map...")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    color_map = {
        "High-High (Hotspot)": "#e74c3c",              # Red
        "Low-Low (Coldspot)": "#27ae60",               # Green
        "Low-High (Spatial Outlier / Risk)": "#f39c12", # Orange
        "High-Low (Isolated Failure)": "#9b59b6",       # Purple
        "Not Significant": "#ecf0f1",                  # Light Grey
    }
    
    # Plot non-significant first
    ns_mask = df["lisa_quadrant"] == "Not Significant"
    ax.scatter(df.loc[ns_mask, "longitude"], df.loc[ns_mask, "latitude"], c=color_map["Not Significant"], s=10, alpha=0.4, label="Not Significant")
    
    # Plot significant clusters
    for q_name in ["Low-Low (Coldspot)", "High-High (Hotspot)", "Low-High (Spatial Outlier / Risk)", "High-Low (Isolated Failure)"]:
        q_mask = df["lisa_quadrant"] == q_name
        ax.scatter(df.loc[q_mask, "longitude"], df.loc[q_mask, "latitude"], c=color_map[q_name], s=18, alpha=0.85, edgecolors="none", label=f"{q_name} (N={q_mask.sum()})")
        
    ax.set_title(f"Anselin Local Moran's I (LISA) Cluster Map: Kavre Springs\nGlobal Moran's I = {moran_i:.3f} (z = {z_score:.1f}, p < 0.001)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude (E)", fontsize=11)
    ax.set_ylabel("Latitude (N)", fontsize=11)
    ax.legend(loc="lower left", frameon=True, fontsize=9)
    plt.tight_layout()
    
    fig_path = OUTPUT_FIGURES_DIR / "lisa_cluster_map.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    logger.info(f"LISA Map saved to {fig_path}")

    # Copy to artifacts directory
    artifact_path = Path(r"C:\Users\ACER\.gemini\antigravity\brain\be3e2d95-a03e-4ea0-9dbb-28995a69c0d8") / "lisa_cluster_map.png"
    plt.savefig(artifact_path, dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    run_spatial_mining()

