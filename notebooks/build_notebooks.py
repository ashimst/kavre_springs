"""
Build and execute the analysis notebooks (04-12). Each notebook explains one part of the paper, shows every table
and figure produced by the pipeline (results/metrics, results/figures) and contains cheap, re-runnable explorations.
Heavy computations are not repeated: run `python reproduce_all.py` (or the module named in each notebook) first.

Usage (from the repository root):  python notebooks/build_notebooks.py [--no-execute]
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

SETUP = """import sys, os, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown
pd.set_option("display.max_columns", 50); pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 80)
M, F = ROOT / "results/metrics", ROOT / "results/figures"
def table(name, **kw):
    t = pd.read_csv(M / name, **kw)
    return t.round(4)
def fig(name, width=1100):
    display(Image(filename=str(F / name), width=width))"""


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


NOTEBOOKS = {
    "03b_gee_enrichment.ipynb": [
        md("""# 03b · Earth Engine enrichment layers
28 label-free covariates added from Google Earth Engine (`src/extract_enrichment_gee.py`): ESA WorldCover fractions,
Hansen forest cover and loss, SoilGrids, MODIS NDVI and ET dynamics, CHIRPS trends, GHSL built-up, WorldPop,
VIIRS night lights and ALOS landforms. This notebook checks coverage, agreement with the survey's own land-use record
and univariate association with drying. (Re-extraction needs an authenticated Earth Engine account.)"""),
        code(SETUP),
        code("""from src.graph_builder import load_master
from src.features import ENRICHMENT, DYNAMIC_FEATURES, check_feature_set
from sklearn.metrics import roc_auc_score
df = load_master()
print("leakage check env_rich:", check_feature_set(df, "env_rich") or "clean")
df[ENRICHMENT].describe().T.round(3)"""),
        md("## Agreement with the survey-recorded land use"),
        code("""pairs = [("landuse_agriculture", "wc_crop"), ("landuse_dense_forest", "wc_tree"), ("landuse_sparse_forest", "wc_tree"),
         ("landuse_settlement", "wc_built"), ("landuse_grassland", "wc_grass")]
pd.DataFrame([{"survey class": a, "WorldCover fraction": b, "AUC": roc_auc_score(df[a], df[b].fillna(0))} for a, b in pairs]).round(3)"""),
        md("## Univariate association with drying (AUC; 0.5 = none)"),
        code("""y = df.target_dried
auc = pd.Series({c: roc_auc_score(y, df[c].fillna(df[c].median())) for c in ENRICHMENT}).rename("AUC").to_frame()
auc["dynamic (may reflect drying)"] = auc.index.isin(DYNAMIC_FEATURES)
auc.assign(strength=(auc.AUC - 0.5).abs()).sort_values("strength", ascending=False).round(3)"""),
        md("## Drying by ALOS slope position (1 valley … 4 ridge/peak)"),
        code("""df.groupby("landform_position").target_dried.agg(["size", "mean"]).round(3)"""),
        md("## Maps of selected layers"),
        code("""cols = ["landform_position", "soil_bdod", "wc_crop", "ndvi_trend", "gfc_loss_post2015", "built_change"]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, c in zip(axes.ravel(), cols):
    sc = ax.scatter(df.longitude, df.latitude, c=df[c], s=1.5, cmap="viridis"); ax.set_title(c); plt.colorbar(sc, ax=ax)
    ax.set_aspect(1 / np.cos(np.radians(27.57))); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()"""),
    ],
    "04_leakage_and_prediction.ipynb": [
        md("""# 04 · Skip-logic leakage and leak-free predictability
**Question.** How predictable is drying from covariates observable without knowing the outcome, and how badly does
the questionnaire's skip logic inflate naive results?

**Method.** Missingness-by-outcome audit of every survey field; single-feature ROC-AUC; benchmark of tabular, neural
and graph models under spatial block CV (5 seeds × 5 folds, 0/500 m buffer) and leave-one-municipality-out, with the
corrected resampled t-test and Holm adjustment (`src/features.py`, `src/benchmark.py`, `src/report.py`)."""),
        code(SETUP),
        md("## Leakage audit: fields whose missingness depends on the outcome"),
        code("""a = table("leakage_audit.csv")
print(f"{int(a.skip_logic_leak.sum())} of {len(a)} processed fields differ in missingness by > 5 percentage points")
a[a.skip_logic_leak].sort_values("missingness_gap", ascending=False).head(25)"""),
        md("## One unanswered question predicts drying"),
        code("""table("single_feature_auc.csv").sort_values("single_feature_auc", ascending=False).head(15)"""),
        code("""fig("leakage_feature_sets.png", 800)"""),
        md("## Benchmark (leak-free `env_hydro`)"),
        code("""s = table("benchmark_summary.csv")
view = s[s.feature_set.isin(["env_hydro"])].pivot_table(index="model", columns=["profile", "buffer_m"], values="roc_auc_mean")
view.sort_values(("main", 0.0), ascending=False)"""),
        code("""t = table("paired_tests_vs_best_tabular.csv")
t[(t.metric == "roc_auc") & (t.feature_set == "env_hydro")][["profile", "buffer_m", "model", "reference", "mean_diff", "ci95_low", "ci95_high", "p_holm"]]"""),
        code("""fig("model_benchmark_comparison.png")"""),
        md("## Feature-set sensitivity (spatial CV, no buffer)"),
        code("""s[(s.buffer_m == 0)].pivot_table(index="model", columns="feature_set", values="roc_auc_mean").round(3)"""),
        md("## Per-municipality transfer (LOMO)"),
        code("""pd.read_csv(ROOT / "results/tables/lomo_per_municipality_auc.csv")"""),
    ],
    "05_landscape_typology.ipynb": [
        md("""# 05 · Unsupervised landscape typology
**Question.** Does the landscape, summarised *without* drying labels or survey fields, structure drying?

**Method.** 23+ continuous GIS covariates (terrain, catchment hydrology, climate, water balance, vegetation, roads,
and the GEE enrichment layers) → standardisation → PCA (90 % variance) → Gaussian mixtures (K = 2–12) selected by BIC
subject to bootstrap stability (mean ARI ≥ 0.75; rule fixed before looking at drying). Density-based check with
HDBSCAN, visual check with t-SNE, landscape anomalies with an isolation forest (`src/unsupervised.py`)."""),
        code(SETUP),
        code("""from src.unsupervised import LANDSCAPE_COLS
print(len(LANDSCAPE_COLS), "landscape covariates:"); print(", ".join(LANDSCAPE_COLS))"""),
        md("## PCA"),
        code("""display(table("un_pca_variance.csv").T)
l = pd.read_csv(M / "un_pca_loadings.csv", index_col=0)
l.iloc[:, :4].round(2).sort_values("PC1")"""),
        md("## Model selection: BIC and bootstrap stability"),
        code("""table("un_typology_selection.csv")"""),
        md("## Types, their profiles and drying rates"),
        code("""fig("un_typology.png")"""),
        code("""display(table("un_typology_drying.csv"))
pd.read_csv(M / "un_typology_profiles.csv", index_col=0).T.round(2)"""),
        md("## Type vs ward (likelihood-ratio tests, Cramér's V)"),
        code("""display(table("un_typology_vs_ward_lrt.csv")); table("un_typology_summary.csv")"""),
        md("## Density-based clusters (HDBSCAN), t-SNE and anomalies"),
        code("""display(table("un_hdbscan_clusters.csv")); display(table("un_hdbscan_summary.csv")); table("un_anomalies.csv")"""),
        code("""fig("un_embedding.png")"""),
        md("## Explore: drying rate by type within each municipality"),
        code("""from src.graph_builder import load_master
df = load_master().merge(pd.read_csv(ROOT / "data/processed/landscape_types.csv"), on="id")
pd.crosstab(df.municipality, df.env_type, values=df.target_dried, aggfunc="mean").round(3)"""),
    ],
    "06_spatial_structure.ipynb": [
        md("""# 06 · Spatial structure and scales
Moran's I distance decay and LISA (`src/spatial_mining.py`); Moran eigenvector maps and variation partitioning;
Bernoulli scan statistic and its alignment with wards (`src/spatial_analysis.py`)."""),
        code(SETUP),
        code("""display(table("spatial_autocorrelation_distance_decay.csv")); table("lisa_cluster_summary.csv")"""),
        code("""fig("lisa_cluster_map.png", 700)"""),
        md("## Environment vs space (Moran eigenvector maps)"),
        code("""display(table("sa_variation_partitioning.csv")); display(table("sa_variation_by_scale.csv")); fig("sa_mem_examples.png")"""),
        md("## Scan statistic clusters"),
        code("""display(table("sa_scan_clusters.csv").drop(columns=["center_index"])); table("sa_scan_ward_alignment.csv")"""),
        code("""fig("sa_spatial_data_mining.png")"""),
    ],
    "07_wards_vs_watersheds.ipynb": [
        md("""# 07 · Wards, not watersheds
**Question.** Within which units is drying homogeneous: hydrological (DEM catchments), landscape (label-free
regions/communities) or survey (wards)?

**Method.** ICC(1) of drying for each partition against 200 randomly shifted square grids with the same number of
units; landscape homogeneity of each partition; pair concordance; upslope/downslope test; flow-path covariates;
three-way variation partitioning (`src/catchment_analysis.py`, `src/unsupervised.py`, `src/drivers_analysis.py`)."""),
        code(SETUP),
        code("""fig("hydro_gis_network.png", 600); display(table("ca_unit_icc.csv"))"""),
        code("""fig("ca_catchment_units.png")"""),
        md("## Label-free regionalisations vs wards"),
        code("""table("un_regionalisation.csv")"""),
        code("""fig("un_regionalisation.png")"""),
        md("## Pairs, directions, flow paths"),
        code("""display(table("sa_ward_concordance.csv")); display(table("ca_pair_concordance_logit.csv"))
display(table("sa_directional_test.csv")); table("ca_flowpath_logit.csv")"""),
        md("## Three-way variation partitioning"),
        code("""display(table("dr_three_way_partition.csv")); table("dr_three_way_r2.csv")"""),
        md("## Explore: ICC of any partition against matched grids"),
        code("""from src.catchment_analysis import icc1, matched_grid_null
from src.graph_builder import load_master, coords_to_metric
df = load_master(); xy = coords_to_metric(df.latitude.to_numpy(), df.longitude.to_numpy()); y = df.target_dried.to_numpy().astype(float)
part = (df.municipality.astype(str) + "_" + df.ward.astype(str)).to_numpy()   # change me
cell, null, _ = matched_grid_null(xy, y, len(np.unique(part)), n_shift=50)
print(f"ICC = {icc1(y, part):.3f}; matched grids (cell {cell/1000:.1f} km): {np.quantile(null, .025):.3f}-{np.quantile(null, .975):.3f}")"""),
    ],
    "08_apriori_pattern_mining.ipynb": [
        md("""# 08 · Apriori pattern mining of drying
**Method** (`src/pattern_mining.py`). Each source is a transaction of items: tertiles of interpretable GIS covariates,
the landscape type, and survey categorical attributes. Frequent itemsets via **apriori** (mlxtend; min support 2 %,
≤ 3 antecedent items) → rules *A → dried* / *A → active*, then:

* one-sided Fisher test + Benjamini–Hochberg FDR;
* **productive rules** only (each item must raise confidence by ≥ 0.03 over every generalisation);
* **ward robustness**: Mantel–Haenszel OR stratified by the 75 wards (rules that hold *within* wards);
* **spatial replication**: lift on held-out spatial blocks;
* ranking by **WRAcc** (subgroup-discovery quality).

The survey's seasonal-flow flag is analysed separately because it may encode dry-season fieldwork."""),
        code(SETUP),
        code("""display(table("pm_drying_rules_summary.csv")); table("pm_seasonal_flag.csv")"""),
        code("""fig("pm_rules.png")"""),
        md("## Top rules for drying (ranked by WRAcc)"),
        code("""r = table("pm_drying_rules.csv")
cols = ["antecedent", "n", "support", "confidence", "lift", "q_bh", "mh_or_ward", "mh_ci_low", "mh_ci_high", "share_wards_lift_gt1", "heldout_replication"]
r[r.consequent == "dried"].sort_values("wracc", ascending=False)[cols].head(30)"""),
        md("## Top protective rules (→ active)"),
        code("""r[r.consequent == "active"].sort_values("wracc", ascending=False)[cols].head(20)"""),
        md("## Simpson-type reversals: pooled lift > 1 but the opposite within wards"),
        code("""r[r.reverses_within_wards][["consequent"] + cols].sort_values("support", ascending=False).head(20)"""),
        md("## Item-level drying rates"),
        code("""table("pm_item_support.csv").sort_values("drying_rate", ascending=False)"""),
        md("## Explore: re-mine with your own thresholds"),
        code("""from src.graph_builder import load_master
from src.pattern_mining import tertile_items, categorical_items, mine_rules
df = load_master()
items = {**tertile_items(df), **categorical_items(df)}; items.pop("flow=seasonal (survey)", None)
I = pd.DataFrame(items).astype(bool)
rules, base = mine_rules(I, df.target_dried.to_numpy(), 1, min_support=0.05, max_len=2)   # change me
rules["antecedent"] = [" & ".join(sorted(a)) for a in rules.antecedent]
rules[(rules.q_bh < 0.05) & rules.productive].sort_values("lift", ascending=False).head(15).round(3)"""),
    ],
    "09_dried_source_narratives.ipynb": [
        md("""# 09 · Recalled causes and years of drying
Among dried sources: reported causes and years; Knox space–time test within vs across wards; ward ICC of causes and
years against matched grids; apriori narrative rules with ward-stratified tests
(`src/descriptive.py`, `src/spatial_analysis.py`, `src/unsupervised.py`, `src/pattern_mining.py`)."""),
        code(SETUP),
        code("""display(table("drying_reasons.csv")); table("drying_timing_summary.csv")"""),
        code("""fig("drying_timing_and_cause.png")"""),
        md("## Knox test: same reported year, within vs across wards"),
        code("""k = table("sa_knox_test.csv"); k[k.time_years == 0]"""),
        md("## Ward ICC of recalled attributes vs matched grids and landscape types"),
        code("""display(table("un_dried_icc.csv")); fig("un_dried_mining.png", 800)"""),
        md("## Apriori narrative rules"),
        code("""n = table("pm_narrative_rules.csv")
n[["consequent", "antecedent", "n", "support", "confidence", "lift", "q_bh", "mh_or_ward", "mh_ci_low", "ward_robust"]].head(40)"""),
        code("""pd.read_csv(M / "un_type_by_reason.csv", index_col=0).round(3)"""),
    ],
    "10_ward_effects.ipynb": [
        md("""# 10 · Ward effects: funnel plot and narrative regimes
**Funnel plot.** Observed vs expected dried sources per ward, with expectations from a landscape-only model; wards
outside the 99.8 % limits and the overdispersion factor φ (≈ 1 if wards differ only by chance given landscape).

**Narrative regimes.** Wards with ≥ 8 dried sources are clustered by the shares of reported causes and periods
(Hellinger geometry, Ward linkage, K by silhouette); Mantel and partial Mantel tests ask whether narrative similarity
follows geography or landscape (`src/unsupervised.py`)."""),
        code(SETUP),
        code("""display(table("un_ward_funnel_summary.csv")); fig("un_wards.png")"""),
        code("""table("un_ward_funnel.csv").sort_values("z")"""),
        md("## Narrative regimes"),
        code("""display(table("un_narrative_k_selection.csv")); display(pd.read_csv(M / "un_narrative_regimes.csv", index_col=0).round(3)); table("un_narrative_mantel.csv")"""),
        code("""pd.read_csv(M / "un_ward_narratives.csv", index_col=0).round(2).sort_values("regime")"""),
    ],
    "11_drivers_and_earthquake.ipynb": [
        md("""# 11 · Drivers within wards and the 2015 earthquake
Standardised logistic coefficients pooled vs with ward fixed effects (`src/drivers_analysis.py`); USGS ShakeMap
exposure, quintiles and adjusted models; cause-specific predictability (`src/spatial_analysis.py`, `src/descriptive.py`)."""),
        code(SETUP),
        code("""fig("dr_drivers.png")"""),
        code("""d = table("dr_pooled_vs_within_ward.csv")
d.pivot_table(index="term", columns="model", values=["coef_per_sd", "ci_low", "ci_high"]).round(2)"""),
        md("## Earthquake exposure"),
        code("""display(table("sa_seismic_quintiles.csv")); display(table("sa_seismic_logit.csv")); table("cause_specific_auc.csv")"""),
    ],
    "12_graph_models.ipynb": [
        md("""# 12 · Does any graph carry information?
GraphSAGE and a residual GATv2 on five topologies (kNN, same DEM catchment, ridge-aware, same ward, random) vs a
graph-free MLP under identical spatial CV (`src/graph_topology_experiment.py`)."""),
        code(SETUP),
        code("""display(table("gt_topology_graph_stats.csv")); display(table("gt_topology_summary.csv")); table("gt_topology_paired.csv")"""),
        code("""fig("gt_topology.png")"""),
    ],
    "13_robustness.ipynb": [
        md("""# 13 · Robustness: label definition, null partitions, validation distance, earthquake adjustment
Analyses added in revision (`src/robustness.py`):
* **R1** contiguous random regions with the ward size distribution as a second null for ward ICC;
* **R2** every ward result repeated without ponds, without seasonal sources, and for perennial springs only;
* **R3** how close test sources are to training sources under each validation design;
* **R4** earthquake exposure with ward fixed effects and Moran eigenvector maps.

The data show ward-level structure. They suggest, but do not prove, that collection practices produced it."""),
        code(SETUP),
        md("## R2 · Sensitivity subsets"),
        code("""s = table("rb_sensitivity_subsets.csv").set_index("subset").T
s"""),
        md("## R1 · Ward ICC against grids and contiguous random regions"),
        code("""fig("rb_robustness.png")"""),
        md("""Interactive: draw a smaller region null (50 partitions) for any outcome. Change `outcome` or `n_rep`."""),
        code("""from src.graph_builder import load_master, coords_to_metric
from src.catchment_analysis import icc1
from src.robustness import neighbours, region_null
df = load_master()
coords = coords_to_metric(df.latitude.to_numpy(), df.longitude.to_numpy())
ward = (df.municipality.astype(str) + "_" + df.ward.astype(str)).to_numpy()
sizes = pd.Series(ward).value_counts().to_numpy()
outcome, n_rep = df.target_dried.to_numpy().astype(float), 50
parts = region_null(neighbours(coords), sizes, coords, n_rep=n_rep, seed=1)
null = np.array([icc1(outcome, p) for p in parts])
print(f"ward ICC {icc1(outcome, ward):.3f}; random regions mean {null.mean():.3f}, 95% range "
      f"{np.quantile(null, .025):.3f}-{np.quantile(null, .975):.3f}; region sizes (one draw) "
      f"{np.percentile(np.bincount(parts[0]), [25, 50, 75])}")"""),
        md("## R3 · Validation distances and buffer sensitivity"),
        code("""display(table("rb_validation_distances.csv"))
b = pd.concat([pd.read_csv(M / f"benchmark_{p}_folds.csv") for p in ["main", "buffers", "lomo"]])
b = b[(b.feature_set == "env_hydro") & b.model.isin(["Logistic Regression", "Random Forest", "CatBoost"])]
b.assign(design=np.where(b.profile == "lomo", "LOMO", "buffer " + b.buffer_m.astype(int).astype(str) + " m")) \\
 .pivot_table(index="design", columns="model", values="roc_auc").round(3)"""),
        md("## R4 · Earthquake exposure: associative only"),
        code("""display(table("rb_seismic_within_ward_variance.csv")); table("rb_seismic_adjusted.csv")"""),
        md("""Only about 11% of PGA variance lies within wards, so the ward-fixed-effect model has little exposure contrast left.
Its null result does not show that shaking has no effect, and the MEM-adjusted association does not show that it has one."""),
        md("## Apriori: multiplicity summary"),
        code("""table("pm_drying_rules_summary.csv")"""),
    ],
}


def main(execute=True):
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        path = NB / name
        if execute:
            NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NB)}}).execute()
        nbf.write(nb, path)
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main(execute="--no-execute" not in sys.argv)
