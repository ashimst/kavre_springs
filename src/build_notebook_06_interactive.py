"""
Build and execute notebooks/06_spatial_mining_and_vulnerability.ipynb.
Uses native NumPy, SciPy, and src.spatial_mining functions for:
1. Multi-scale Spatial Autocorrelation & Distance Decay (Global Moran's I across k=3 to k=20)
2. Anselin's Local Moran's I (LISA) Hotspot and Outlier Spatial Clustering with BH-FDR control
3. Hydro-Ecological Association Rule Mining (Fisher's exact test, Confidence, Lift, Odds Ratio, BH-FDR)
4. Active Spring Prioritization Index (SPI) & Top 500 Priority Springshed Interventions
"""

import os, sys
from pathlib import Path
import nbformat as nbf

PROJECT_ROOT = Path(".").resolve()

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.12.8"
    }
}

cells = []

# Cell 0: Header & Documentation
cells.append(nbf.v4.new_markdown_cell("""# 06. Spatial Autocorrelation, Unsupervised Mining & Vulnerability Index
## Sub-Catchment Spatial Intelligence & Policy Prioritization (Kavrepalanchok District, Nepal)

### Scientific Context & Objectives
This notebook executes the spatial data mining and policy prioritization pipeline for the 5,689 surveyed springs in Kavrepalanchok:
1. **Multi-Scale Spatial Autocorrelation & Distance Decay**: Quantifies how desiccation spatial dependence decays across geometric scales ($k \\in [3, 20]$), determining the empirical sub-catchment fracture flow boundary.
2. **Anselin's Local Moran's I (LISA) Hotspots**: Detects statistically significant local clusters of desiccation (High-High cores) and imminent hazard outliers (Low-High vulnerable springs) with Benjamini-Hochberg False Discovery Rate (FDR) control.
3. **Hydro-Ecological Association Rule Mining**: Discovers frequent multi-factor environmental patterns associated with spring desiccation using Fisher's exact tests and BH-FDR correction over all candidate rules.
4. **Actionable Spring Prioritization Index (SPI / HSVI)**: Couples machine-learning predicted drying risk with household dependency to rank the **Top 500 critically endangered active springs** for municipal engineering interventions."""))

# Cell 1: Environment & Setup
cells.append(nbf.v4.new_code_cell("""import os, sys
from pathlib import Path

# Setup project root and headless matplotlib environment
PROJECT_ROOT = Path("..").resolve() if Path("..").resolve().name == "kavre_springs" else Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["MPLCONFIGDIR"] = os.path.join(str(PROJECT_ROOT), ".matplotlib")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.graph_builder import load_master, coords_to_metric
from src.spatial_mining import knn_weights, morans_i, lisa, quad_label, bh_fdr, association_rules

print("Geospatial & Spatial Mining libraries loaded successfully.")"""))

# Cell 2: Data Loading & Geometry
cells.append(nbf.v4.new_code_cell("""# Load master dataset and convert geographic coordinates to local UTM metric space (meters)
df_master = load_master()
coords_m = coords_to_metric(df_master["latitude"].to_numpy(), df_master["longitude"].to_numpy())
y = df_master["target_dried"].to_numpy(float)

num_total = len(df_master)
num_dried = int(y.sum())
num_active = num_total - num_dried

print("=== Kavrepalanchok Mountain Spring Survey Geometry ===")
print(f"Total Water Sources: {num_total:,}")
print(f"Dried Sources:       {num_dried:,} ({y.mean():.1%})")
print(f"Active Sources:      {num_active:,} ({(1 - y.mean()):.1%})")
print(f"Spatial Extent:      ~40 km East-West x 39 km North-South")"""))

# Cell 3: Markdown Section 1 - Distance Decay
cells.append(nbf.v4.new_markdown_cell("""## 1. Multi-Scale Spatial Autocorrelation & Distance Decay

We compute Global Moran's $I$ across varying $k$-nearest neighbor radii ($k \\in [3, 5, 7, 10, 15, 20]$) using 999 permutation trials to establish the physical scale of desiccation."""))

# Cell 4: Code Section 1 - Distance Decay
cells.append(nbf.v4.new_code_cell("""k_values = [3, 5, 7, 10, 15, 20]
decay_records = []
rng = np.random.default_rng(2024)

print("Computing Global Moran's I across spatial scales (k=3 to k=20)...")
for k in k_values:
    W, mean_d, _ = knn_weights(coords_m, k)
    I_val, p_val, z_val = morans_i(y, W, n_perm=999, rng=rng)
    decay_records.append({
        "k_neighbors": k,
        "mean_neighbor_dist_m": round(mean_d, 1),
        "morans_i": round(float(I_val), 4),
        "z_score": round(float(z_val), 2),
        "p_value": float(p_val)
    })
    print(f"  k={k:2d} (mean d={mean_d:5.1f}m) | Moran's I: {I_val:.4f} | z-score: {z_val:.2f} (p={p_val:.3f})")

df_decay = pd.DataFrame(decay_records)
display(df_decay)

# Plot Multi-Scale Distance Decay
plt.figure(figsize=(8, 4.5), dpi=300)
plt.plot(df_decay["mean_neighbor_dist_m"], df_decay["morans_i"], marker='o', linewidth=2.5, markersize=8, color="#2980b9")
plt.axvline(df_decay.loc[0, "mean_neighbor_dist_m"], color='#e74c3c', linestyle='--', alpha=0.7, 
            label=f"Peak Spatial Autocorrelation (k=3, ~{df_decay.loc[0, 'mean_neighbor_dist_m']:.0f}m)")
plt.xlabel("Mean Distance to Neighbors (m)", fontsize=11, fontweight="bold")
plt.ylabel("Global Moran's I (Spatial Autocorrelation)", fontsize=11, fontweight="bold")
plt.title("Spatial Autocorrelation Distance Decay of Spring Desiccation", fontsize=12, fontweight="bold")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=True)
plt.tight_layout()
os.makedirs(PROJECT_ROOT / "results/figures", exist_ok=True)
plt.savefig(PROJECT_ROOT / "results/figures/spatial_mining_diagnostics.png", dpi=300, bbox_inches="tight")
plt.show()"""))

# Cell 5: Markdown Section 2 - LISA Hotspots
cells.append(nbf.v4.new_markdown_cell("""## 2. Anselin's Local Moran's I (LISA) Hotspot Analysis

We detect spatial clusters and local spatial outliers at $k=7$ using conditional permutations and Benjamini-Hochberg False Discovery Rate (FDR) correction:
- **High-High**: Desiccation core hotspots.
- **Low-High**: Active springs in dried clusters (imminent risk outliers).
- **Low-Low**: Hydrologically secure coldspots.
- **High-Low**: Isolated dried springs."""))

# Cell 6: Code Section 2 - LISA Hotspots
cells.append(nbf.v4.new_code_cell("""W7, mean_d7, nbr7 = knn_weights(coords_m, 7)
Ii, p_sim, q_fdr, quad = lisa(y, W7, nbr7, n_perm=999, rng=rng)

df_master["local_moran_i"] = Ii
df_master["p_value_sim"] = p_sim
df_master["q_value_fdr"] = q_fdr
df_master["lisa_cluster"] = quad

print("=== Anselin's Local Moran's I (BH-FDR q < 0.05) Cluster Summary ===")
lisa_summary = df_master["lisa_cluster"].value_counts().reset_index()
lisa_summary.columns = ["Cluster Type", "Spring Count"]
display(lisa_summary)

# Plot Spatial LISA Clusters
plt.figure(figsize=(9, 6), dpi=300)
colors_lisa = {
    "High-High (dried cluster)": "#e74c3c", 
    "Low-High (active among dried)": "#f39c12", 
    "Low-Low (active cluster)": "#27ae60", 
    "High-Low (isolated dried)": "#9b59b6", 
    "Not significant": "#bdc3c7"
}

for ctype, col in colors_lisa.items():
    sub = df_master[df_master["lisa_cluster"] == ctype]
    alpha = 0.85 if ctype != "Not significant" else 0.25
    size = 20 if ctype != "Not significant" else 5
    plt.scatter(sub["longitude"], sub["latitude"], c=col, label=f"{ctype} ({len(sub):,})", s=size, alpha=alpha, edgecolors='none')

plt.xlabel("Longitude (°E)", fontsize=11, fontweight="bold")
plt.ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
plt.title(f"LISA Desiccation Spatial Hotspots (k=7, Mean d={mean_d7:.0f}m, BH-FDR q < 0.05)", fontsize=12, fontweight="bold")
plt.legend(frameon=True, loc="lower left", fontsize=8.5)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "results/figures/lisa_cluster_map.png", dpi=300, bbox_inches="tight")
plt.show()"""))

# Cell 7: Markdown Section 3 - Association Rules
cells.append(nbf.v4.new_markdown_cell("""## 3. Hydro-Ecological Association Rule Mining

We discover frequent multi-factor environmental and landscape conditions strongly associated with spring drying (*target_dried = 1*) using Fisher's exact test and Benjamini-Hochberg FDR control across all evaluated candidate rules."""))

# Cell 8: Code Section 3 - Association Rules
cells.append(nbf.v4.new_code_cell("""rules = association_rules(df_master, min_support=0.03)

print("=== Top Hydro-Ecological Association Rules for Spring Desiccation (BH-FDR q < 0.05) ===")
sig_rules = rules[rules["q_value_bh"] < 0.05].head(10).copy()
display(sig_rules[["antecedent", "n_antecedent", "n_dried", "support", "confidence", "lift", "odds_ratio", "q_value_bh"]].round(4))

# Plot Association Rule Lifts
plt.figure(figsize=(10, 4.5), dpi=300)
top_plot = sig_rules.head(8).iloc[::-1]
plt.barh(top_plot["antecedent"], top_plot["lift"], color="#34495e", edgecolor="black", alpha=0.85)
plt.axvline(1.0, color="red", linestyle="--", alpha=0.8, label="Baseline Independence (Lift = 1.0)")
plt.xlabel("Lift: P(Dried | Antecedent) / P(Dried)", fontsize=11, fontweight="bold")
plt.title("Top Association Rules Predicting Mountain Spring Desiccation", fontsize=12, fontweight="bold")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()"""))

# Cell 9: Markdown Section 4 - Active Spring Prioritization
cells.append(nbf.v4.new_markdown_cell("""## 4. Active Spring Prioritization Index (SPI) & Actionable Interventions

We couple model susceptibility with community exposure to rank the **Top 500 critically endangered active springs** for municipal engineering interventions:
$$\\text{SPI}_i = \\text{Rank}_{\\%}(\\hat{P}_{\\text{dry}, i}) \\times \\text{Rank}_{\\%}(D_{\\text{hh}, i})$$
where $\\hat{P}_{\\text{dry}, i}$ is the out-of-fold predicted desiccation probability from our top-performing model, and $D_{\\text{hh}, i}$ is the total dependent household population."""))

# Cell 10: Code Section 4 - Active Spring Prioritization
cells.append(nbf.v4.new_code_cell("""from src.prioritization import run_prioritization

# Execute full prioritization pipeline
muni_summary, sens = run_prioritization(top_n=500)

print("=== Municipality Spring Priority Summary (Top 500 Active Springs) ===")
display(muni_summary.round(3))

print("\\n=== Priority Index Sensitivity Analysis ===")
display(sens.T)"""))

nb.cells = cells

output_path = PROJECT_ROOT / "notebooks/06_spatial_mining_and_vulnerability.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully generated {output_path} with {len(cells)} cells.")
