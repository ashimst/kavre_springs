"""
Build and execute the analysis notebooks.

Each notebook reproduces one part of the paper: it *runs* the corresponding pipeline stage (writing the
same files to results/metrics and results/figures as `python reproduce_all.py`) and then shows the
resulting tables and figures. Run the notebooks in order on a fresh checkout; later notebooks read
predictions and intermediate tables written by earlier ones.

Two stages are slow from scratch and are checkpointed: the predictive benchmark (notebook 04, about an
hour cold, seconds when the per-fold results already exist) and the placebo-partition nulls (notebook 06,
cached under results/cache). Notebook 02 is the exception to "running reproduces everything": the Earth
Engine extractions need an authenticated account, so that notebook verifies the versioned extraction
outputs instead of re-downloading them.

Usage (from the repository root):  python notebooks/build_notebooks.py [--no-execute]
       Name prefixes limit the build to some notebooks:  python notebooks/build_notebooks.py 06 09
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

SETUP = """import logging, os, sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout, force=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)

M = ROOT / "results/metrics"
F = ROOT / "results/figures"

def table(name, **kw):
    return pd.read_csv(M / name, **kw).round(4)

def fig(name, width=1100):
    display(Image(filename=str(F / name), width=width))"""


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


NOTEBOOKS = {
    # ------------------------------------------------------------------ 01
    "01_inventory_and_labels.ipynb": [
        md("""# 01 · The inventory, the label and the survey structure
The ICIMOD (2025) community inventory of 5,689 water sources in seven municipalities of Kavrepalanchok
(doi:10.26066/RDS.1974131, CC BY 4.0), cleaned by `src/data_cleaning.py` and assembled with the versioned
covariate extractions by `src/assemble_master.py`. We took no part in the data collection; everything
here is a secondary analysis of the public release.

**The label.** `source_condition` (active / dried) is the condition recorded by the community resource
person at an undated visit between October 2023 and June 2024. The release contains no operational
definition of "dried", no discharge for dried sources, no visit date and no enumerator identifier.
Ponds and sources recorded as seasonal are treated as sensitivity cases throughout (notebook 10).

Running this notebook reproduces the boundary audit (`boundary_agreement.csv`) and the raw-field
skip-logic audit (`x0_raw_field_audit*.csv`)."""),
        code(SETUP),
        code("""from src.dataset import load_master

df = load_master()
ward = df.municipality.astype(str) + "_" + df.ward.astype(str)
print(f"{len(df):,} sources | {df.municipality.nunique()} municipalities | {ward.nunique()} wards | "
      f"{df.target_dried.mean():.1%} recorded as dried")
pd.crosstab(df.source_type, df.source_condition, margins=True)"""),
        md("## Recorded drying by source type, seasonality and municipality"),
        code("""df.groupby(["source_type", "seasonality"]).target_dried.agg(n="size", dried_share="mean").round(3)"""),
        code("""df.groupby("municipality").agg(sources=("id", "size"), wards=("ward", "nunique"),
                               dried_share=("target_dried", "mean")).sort_values("dried_share").round(3)"""),
        md("## Wards: the units in which the survey was organised"),
        code("""size = ward.value_counts()
rate = df.groupby(ward).target_dried.mean()
_f, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].hist(size, bins=25, color="grey")
axes[0].set_xlabel("sources per ward"); axes[0].set_ylabel("wards")
axes[1].hist(rate, bins=25, color="firebrick")
axes[1].set_xlabel("share recorded as dried"); axes[1].set_ylabel("wards")
plt.tight_layout()
size.describe().round(1)"""),
        md("""## Boundary audit
Recorded palika and ward identifiers are survey assignments. How many sources lie inside the official
polygon of their recorded palika?"""),
        code("""from src.boundaries import run as audit_boundaries
audit_boundaries()
table("boundary_agreement.csv")"""),
        md("""Ward polygons come from OpenStreetMap (admin_level=9 relations, ODbL), not from an official release.
How many sources lie inside the polygon of their recorded ward, and where do the others fall?"""),
        code("""from src.ward_polygons import run as audit_wards
audit_wards()
display(table("ward_boundary_agreement.csv"))
c = table("ward_label_crosstab.csv")
c[c.recorded_ward.isin(["Dhulikhel-1", "Dhulikhel-2", "Panchkhal-10", "Panchkhal-11", "Panchkhal-12"])]"""),
        md("""## Skip logic in the raw questionnaire
The form had seven modules for active sources and four for dried ones, so the *missingness* of many raw
fields encodes the outcome. This audit compares the missingness of all 57 raw fields between active and
dried sources; `src.features.check_feature_set` later rejects any outcome-dependent field from every
predictor set."""),
        code("""from src.revision_experiments import raw_field_audit
raw_field_audit()"""),
        code("""a = table("x0_raw_field_audit.csv")
print(f"{int(a.outcome_dependent.sum())} of {len(a)} raw fields are outcome-dependent")
a.head(15)"""),
        md("## Study area"),
        code("""fig("fig1_study_area.png", 900)"""),
    ],
    # ------------------------------------------------------------------ 02
    "02_covariate_extraction.ipynb": [
        md("""# 02 · Covariate extraction (versioned)
The label-free covariates come from SRTM, MERIT Hydro, CHIRPS, Sentinel-2, TerraClimate, OpenStreetMap,
the Copernicus GLO-30 DEM and USGS ShakeMap (`src/gee_extraction.py`, `src/extract_hydro_catchment_gee.py`,
`src/extract_osm_roads.py`, `src/hydro_gis.py`, `src/seismic.py`). Re-extracting the Earth Engine layers
needs an authenticated account, so the extraction outputs are versioned under `data/processed/` and this
notebook verifies them instead of re-downloading: coverage, ranges and the tier definitions used
everywhere else. Every other notebook runs from these files alone."""),
        code(SETUP),
        code("""from src.dataset import load_master
from src.features import FEATURE_SETS

df = load_master()
for name, cols in FEATURE_SETS.items():
    missing = [c for c in cols if c not in df.columns]
    print(f"{name:18s} {len(cols):3d} covariates   missing from master: {missing or 'none'}")"""),
        md("## Coverage and ranges of the extracted covariates"),
        code("""cols = FEATURE_SETS["env_rich"]
cov = df[cols].describe().T[["count", "mean", "std", "min", "max"]]
cov["share_missing"] = 1 - cov["count"] / len(df)
cov.round(3).sort_values("share_missing", ascending=False).head(20)"""),
        md("## The DEM-routing and seismic layers used only in association analyses"),
        code("""hg = pd.read_csv(ROOT / "data/processed/hydro_gis_features.csv")
print(f"{len(hg):,} sources with DEM-routing features; "
      f"{hg.catch_T2p0.nunique()} occupied 2 km^2 link catchments")
hg.describe().T[["mean", "min", "max"]].round(3)"""),
    ],
    # ------------------------------------------------------------------ 03
    "03_gee_enrichment.ipynb": [
        md("""# 03 · Do the Earth Engine enrichment layers behave sensibly?
28 label-free covariates from Google Earth Engine (`src/extract_enrichment_gee.py`): ESA WorldCover
fractions, Hansen forest cover and loss, SoilGrids, MODIS NDVI and ET dynamics, GHSL built-up, WorldPop,
VIIRS night lights and ALOS landforms. Before giving them to any model, this notebook checks the leakage
rule, their agreement with the survey's own land-use record and their univariate association with
drying. Whether they add predictive skill is answered in notebook 04."""),
        code(SETUP),
        code("""from src.dataset import load_master
from src.features import DYNAMIC_FEATURES, ENRICHMENT, check_feature_set
from sklearn.metrics import roc_auc_score

df = load_master()
print("leakage check env_rich:", check_feature_set(df, "env_rich") or "clean")
df[ENRICHMENT].describe().T.round(3)"""),
        md("## Agreement with the survey-recorded land use"),
        code("""pairs = [("landuse_agriculture", "wc_crop"), ("landuse_dense_forest", "wc_tree"),
         ("landuse_sparse_forest", "wc_tree"), ("landuse_settlement", "wc_built"),
         ("landuse_grassland", "wc_grass")]
pd.DataFrame([{"survey class": a, "WorldCover fraction": b,
               "AUC": roc_auc_score(df[a], df[b].fillna(0))} for a, b in pairs]).round(3)"""),
        md("## Univariate association with drying (AUC; 0.5 = none)"),
        code("""y = df.target_dried
auc = pd.Series({c: roc_auc_score(y, df[c].fillna(df[c].median())) for c in ENRICHMENT}).rename("AUC").to_frame()
auc["dynamic (may reflect drying)"] = auc.index.isin(DYNAMIC_FEATURES)
auc.assign(strength=(auc.AUC - 0.5).abs()).sort_values("strength", ascending=False).round(3)"""),
        md("## Maps of selected layers"),
        code("""cols = ["landform_position", "soil_bdod", "wc_crop", "ndvi_trend", "gfc_loss_post2015", "built_change"]
_f, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, c in zip(axes.ravel(), cols):
    sc = ax.scatter(df.longitude, df.latitude, c=df[c], s=1.5, cmap="viridis")
    ax.set_title(c); plt.colorbar(sc, ax=ax)
    ax.set_aspect(1 / np.cos(np.radians(27.57))); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()"""),
    ],
    # ------------------------------------------------------------------ 04
    "04_leakage_and_prediction.ipynb": [
        md("""# 04 · Skip-logic leakage and leak-free predictability (Fig. 2)
**Question.** How much does each data layer add, and how badly does the questionnaire's skip logic
inflate naive results?

**Method.** Logistic regression, random forest, CatBoost and kNN label interpolation under spatial block
CV (5 seeds x 5 folds; buffers 0-2 km) and leave-one-municipality-out, with imputation and scaling
fitted inside each fold (`src/benchmark.py`); fold-matched tier differences with the corrected resampled
t-test and Holm adjustment; within-ward ranking skill from the out-of-fold predictions.

Running this notebook reproduces the full benchmark. The per-fold results are checkpointed: from scratch
this takes about an hour on a laptop CPU, with existing checkpoints it finishes in seconds and recomputes
only the summary tables."""),
        code(SETUP),
        code("""from src.benchmark import PROFILES, run_profile

for profile in PROFILES:
    run_profile(profile)
print("benchmark folds up to date")"""),
        md("## Summary tables (also written to results/metrics)"),
        code("""from src.report import CORE_ORDER, current_runs, load_folds, lomo_table, paired_tests, summarise
from src.features import missingness_audit, single_feature_rule_scores
from src.dataset import load_master

df = load_master()
folds = current_runs(load_folds())
summary = summarise(folds)
summary.to_csv(M / "benchmark_summary.csv", index=False)
tests = pd.concat([paired_tests(folds, None, CORE_ORDER, m) for m in ["roc_auc", "pr_auc"]])
tests.to_csv(M / "paired_tests_vs_best_tabular.csv", index=False)
missingness_audit(df).to_csv(M / "leakage_audit.csv", index=False)
single_feature_rule_scores(df).to_csv(M / "single_feature_auc.csv", index=False)
lomo = lomo_table(folds, df)
summary.pivot_table(index=["profile", "feature_set", "buffer_m"], columns="model", values="roc_auc_mean").round(3)"""),
        md("## Leakage: outcome-dependent fields and the one-question model"),
        code("""a = table("leakage_audit.csv")
print(f"{int(a.skip_logic_leak.sum())} of {len(a)} processed fields differ in missingness by > 5 percentage points")
a[a.skip_logic_leak].sort_values("missingness_gap", ascending=False).head(12)"""),
        code("""table("single_feature_auc.csv").head(10)"""),
        md("## Tier increments, model comparisons and fold spread"),
        code("""from src.revision_experiments import run as run_experiments
run_experiments(only=["X1", "X2", "X11"])"""),
        code("""x1 = table("x1_tier_increments.csv")
x1[["design", "model", "from", "to", "auc_from", "auc_to", "mean_diff", "ci95_low", "ci95_high", "p_holm"]]"""),
        code("""display(table("x2_within_ward_auc.csv"))
display(table("x11_lomo_by_municipality.csv").query("feature_set == 'env_hydro'"))
table("x11_springs_only_auc.csv")"""),
        md("## Distance between test and training sources"),
        code("""from src.dataset import coords_to_metric
from src.robustness import validation_distances

coords = coords_to_metric(df.latitude.to_numpy(), df.longitude.to_numpy())
vfolds, vsumm = validation_distances(df, coords)
vfolds.to_csv(M / "rb_validation_distances_folds.csv", index=False)
vsumm.to_csv(M / "rb_validation_distances.csv", index=False)
vsumm.round(3)"""),
        code("""fig("fig2_information.png")"""),
    ],
    # ------------------------------------------------------------------ 05
    "05_landscape_typology.ipynb": [
        md("""# 05 · Label-free landscape typology (Fig. 3)
**Question.** Does the landscape, summarised *without* drying labels or survey fields, structure
recorded drying?

**Method.** 46 continuous GIS covariates -> standardisation -> PCA (90 % of variance) -> Gaussian
mixtures (K = 2-12) selected by BIC subject to bootstrap stability (mean ARI >= 0.75; the rule was fixed
before drying was examined). The same module builds the contiguous landscape regions, the ward funnel
plot and the ward ICC of recalled attributes (`src/unsupervised.py`).

Running this notebook reproduces every `un_*` table and `data/processed/landscape_types.csv`
(a few minutes)."""),
        code(SETUP),
        code("""from src.unsupervised import run as run_unsupervised
run_unsupervised()"""),
        code("""fig("fig3_landscape_types.png")"""),
        md("## PCA, model selection and the eight types"),
        code("""display(table("un_pca_variance.csv").T)
display(table("un_typology_selection.csv"))
table("un_typology_drying.csv")"""),
        md("## Type against ward"),
        code("""display(table("un_typology_vs_ward_lrt.csv"))
table("un_typology_summary.csv")"""),
        md("## Explore: drying rate by type within each municipality"),
        code("""from src.dataset import load_master
df = load_master().merge(pd.read_csv(ROOT / "data/processed/landscape_types.csv"), on="id")
pd.crosstab(df.municipality, df.env_type, values=df.target_dried, aggfunc="mean").round(3)"""),
    ],
    # ------------------------------------------------------------------ 06
    "06_wards_vs_catchments.ipynb": [
        md("""# 06 · Wards against catchments and spatial nulls (Fig. 5)
**Question.** Within which units is recorded drying homogeneous: hydrological (DEM link catchments),
landscape (label-free regions) or survey (wards)? The 2 km^2 catchments are the interpretable
hydrological counterpart to wards; wards are additionally tested against three families of 1,000
placebo partitions (rotated grids, contiguous size-matched regions, and regions grown inside each
municipality), every placebo region verified connected (`src/regions.py`).

Running this notebook reproduces the Moran distance decay, the unit ICC table, the pair-concordance
and flow-path models, the directional test, the variation partitioning and all placebo-null
experiments. The partition draws are cached under `results/cache`; the first run builds them
(roughly half an hour), later runs reuse them."""),
        code(SETUP),
        code("""from src.spatial_mining import run_spatial_mining
run_spatial_mining()
table("spatial_autocorrelation_distance_decay.csv")"""),
        md("## ICC of drying within each unit type, against matched grids"),
        code("""from src.catchment_analysis import run_all as run_catchments
run_catchments()
table("ca_unit_icc.csv")"""),
        md("## Directional (upslope/downslope) dependence and nearby-pair concordance"),
        code("""from src.dataset import load_master, coords_to_metric
from src.spatial_splits import grid_block_ids
from src.spatial_analysis import directional_test, ward_concordance

df = load_master()
coords = coords_to_metric(df.latitude.to_numpy(), df.longitude.to_numpy())
blocks = grid_block_ids(coords, 8)
tab = directional_test(df, coords)[0]
display(tab)
ward_concordance(df, coords, blocks).round(3)"""),
        md("## Variation partitioning: environment, space, ward"),
        code("""from src.drivers_analysis import three_way_partition
part, R = three_way_partition(df, coords)
part.round(4)"""),
        md("## Ward funnel against a landscape-only model"),
        code("""from src.unsupervised import env_matrix, ward_funnel
ward = (df.municipality.astype(str) + "_" + df.ward.astype(str)).to_numpy()
w, wsum = ward_funnel(df, env_matrix(df), ward)
wsum.round(3)"""),
        md("""## Placebo-partition experiments
* X10: ward ICC against 1,000 rotated grids, 1,000 contiguous size-matched regions and 1,000
  municipality-nested regions, for drying, covariate residuals and recalled attributes, all sources and
  springs only, with partition diagnostics.
* X3: the same-ward pair term against placebo units.
* X4: residual ward ICC against 200-draw nulls.
* X5: variation partitioning with placebo units in place of wards."""),
        code("""from src.revision_experiments import run as run_experiments
run_experiments(only=["X10", "X3", "X4", "X5"])"""),
        code("""x = table("x10_strengthened_nulls.csv")
x[["subset", "outcome", "null", "obs", "null_q975", "null_max", "z", "n_exceed", "p_one_sided"]]"""),
        code("""display(table("x10_partition_diagnostics.csv"))
display(table("x10_ward_connectivity.csv"))
display(table("x3_pair_placebo.csv"))
display(table("x4_residual_ward_icc.csv"))
table("x5_placebo_partition.csv").groupby("unit")[["unit_total", "unit_only", "space_only", "space_and_unit"]].agg(["min", "max"]).round(3)"""),
        md("""## Wards from polygons
X26 assigns each source to the OpenStreetMap ward polygon that contains it and repeats the ward ICC tests
with those wards, with recorded wards on the same sources, and among sources inside their recorded ward."""),
        md("""## Latent-scale ICC (X28)
The ANOVA ICC of a 0/1 outcome depends on prevalence. X28 repeats every ward and unit ICC comparison with
the latent-scale ICC of a random-intercept logit (`src/latent_icc.py`) against the same partitions."""),
        code("""run_experiments(only=["X28"])
x = table("x28_latent_icc.csv")
display(x[["block", "subset", "outcome", "unit", "null", "observed_obs", "observed_n_exceed", "latent_obs",
           "latent_null_max", "latent_n_exceed", "latent_p_one_sided"]])
table("x28_latent_icc_contrasts.csv")"""),
        code("""run_experiments(only=["X26"])
x = table("x26_ward_polygon_icc.csv")
display(x[["unit", "sample", "n_sources", "outcome", "null", "obs", "null_q975", "null_max", "p_one_sided"]])
display(table("x26_ward_polygon_contrast.csv"))
table("x26_ward_connectivity.csv")"""),
        md("""## Direct ward-catchment contrasts
The comparisons above set each unit type against its own nulls. These three steps compare wards with
catchments directly.

* X12: same-ward minus same-catchment log-odds in the pair model, from the 300 bootstrap draws of the
  pair model fitted above (both terms come from one regression, so each draw gives a paired difference).
* X13: catchments re-delineated at the area threshold that gives as many occupied catchments as there
  are wards, so that the two unit systems have the same granularity. Reports their ICC against rotated
  grids, a block-jackknife interval for ICC(ward) - ICC(catchment), and the pair model with the
  count-matched catchment. The first run routes flow on the DEM (a few minutes) and saves the labels
  to `data/processed/hydro_gis_matched_catchments.csv`.
* X14: the pair model for springs only, with dyadic-robust standard errors and leave-one-block-out
  estimates (which spatial blocks move the ward term)."""),
        code("""run_experiments(only=["X12", "X13", "X14"])"""),
        code("""display(table("x12_pair_contrast.csv"))
display(table("x13_matched_count_icc.csv")[["unit", "n_units", "obs", "null_q975", "null_max", "z", "p_one_sided"]])
display(table("x13_matched_count_contrast.csv"))
table("x14_pair_robustness.csv").set_index(["population", "term"]).T"""),
        code("""from src.figures import figS1_pair_bootstrap
from src.plotstyle import apply
apply()
figS1_pair_bootstrap()
fig("figS1_pair_bootstrap.png")"""),
        code("""fig("fig5_wards_watersheds.png")"""),
    ],
    # ------------------------------------------------------------------ 07
    "07_apriori_pattern_mining.ipynb": [
        md("""# 07 · Apriori pattern mining of drying (Fig. 4)
**Method** (`src/pattern_mining.py`). Each source is a transaction of items: tertiles of interpretable
GIS covariates, the landscape type and survey categorical attributes. Frequent itemsets via apriori
(min support 2 %, <= 3 antecedent items) -> rules *A -> dried* / *A -> active*; one-sided Fisher tests
with BH-FDR; productive rules only; ward-stratified Mantel-Haenszel odds ratios with a second BH screen;
replication on held-out spatial blocks; ranking by WRAcc. X6 adds FDR control to the candidate
Simpson-type reversals. Running this notebook re-mines everything (a few minutes)."""),
        code(SETUP),
        code("""from src.pattern_mining import run as run_mining
run_mining()"""),
        code("""from src.revision_experiments import run as run_experiments
run_experiments(only=["X6"])"""),
        code("""fig("fig4_rules.png")"""),
        code("""display(table("pm_drying_rules_summary.csv"))
display(table("x6_reversals_fdr.csv"))
table("pm_seasonal_flag.csv")"""),
        md("## Top rules for drying and for activity (ranked by WRAcc)"),
        code("""r = table("pm_drying_rules.csv")
cols = ["antecedent", "n", "support", "confidence", "lift", "q_bh", "mh_or_ward", "mh_ci_low",
        "mh_ci_high", "heldout_replication"]
display(r[r.consequent == "dried"].sort_values("wracc", ascending=False)[cols].head(15))
r[r.consequent == "active"].sort_values("wracc", ascending=False)[cols].head(10)"""),
        md("## Candidate Simpson-type reversals (none survives FDR control)"),
        code("""r[r.reverses_within_wards][["consequent"] + cols].sort_values("support", ascending=False)"""),
    ],
    # ------------------------------------------------------------------ 08
    "08_recalled_causes.ipynb": [
        md("""# 08 · Recalled causes and years of drying (Fig. 6)
Among the 1,521 dried sources: reported causes and years (first summarised by Pandit et al. 2026),
cause-specific predictability, and the Knox space-time test of same-year reporting within against
across wards. Recalled causes and years are survey responses reconstructed from memory, not verified
events. The ward ICC of these attributes is computed with the typology in notebook 05 (`un_dried_icc.csv`)
and against the strengthened nulls in notebook 06 (`x10_strengthened_nulls.csv`); both are shown here.

Running this notebook reproduces the descriptive tables, the cause-specific AUC and the Knox test."""),
        code(SETUP),
        code("""from src.descriptive import run_descriptive
run_descriptive()"""),
        code("""from src.dataset import load_master, coords_to_metric
from src.spatial_analysis import knox_test

df = load_master()
coords = coords_to_metric(df.latitude.to_numpy(), df.longitude.to_numpy())
k = knox_test(df, coords)
k[k.time_years == 0].round(3)"""),
        code("""fig("fig6_recalled.png")"""),
        code("""display(table("drying_reasons.csv"))
table("drying_timing_summary.csv")"""),
        md("## Ward clustering of recalled attributes (computed in notebooks 05 and 06)"),
        code("""display(table("un_dried_icc.csv"))
x = table("x10_strengthened_nulls.csv")
x[x.outcome.str.contains("attribution|2072")][["subset", "outcome", "null", "obs", "null_max", "p_one_sided"]]"""),
    ],
    # ------------------------------------------------------------------ 09
    "09_drivers_and_earthquake.ipynb": [
        md("""# 09 · Covariates within wards and the 2015 earthquake (Fig. 7)
Standardised logistic coefficients pooled against ward fixed effects (spatial-block bootstrap);
X9 repeats the comparison with matched-grid indicators; seismic exposure (USGS ShakeMap) under
environmental, Moran-eigenvector and ward adjustment; X7 compares the predictability of
earthquake-attributed and other dried sources; X8 looks for ward-level correlates of the ward excess.
All results are associations; 89-93 % of the exposure variance lies between wards, so shaking cannot be
separated from ward differences in these data.

Running this notebook reproduces all of the above (about ten minutes)."""),
        code(SETUP),
        code("""from src.dataset import load_master, coords_to_metric
from src.spatial_splits import grid_block_ids
from src.drivers_analysis import pooled_vs_within
from src.spatial_analysis import seismic_dose_response

df = load_master()
coords = coords_to_metric(df.latitude.to_numpy(), df.longitude.to_numpy())
blocks = grid_block_ids(coords, 8)
coefs = pooled_vs_within(df, blocks)
seis = seismic_dose_response(df, coords, blocks)"""),
        code("""from src.robustness import earthquake_adjusted
var, eq = earthquake_adjusted(df, coords)
var.to_csv(M / "rb_seismic_within_ward_variance.csv", index=False)
eq.to_csv(M / "rb_seismic_adjusted.csv", index=False)"""),
        code("""from src.revision_experiments import run as run_experiments
run_experiments(only=["X7", "X8", "X9", "X15"])"""),
        code("""fig("fig7_drivers.png")"""),
        code("""d = table("dr_pooled_vs_within_ward.csv")
d.pivot_table(index="term", columns="model", values=["coef_per_sd", "ci_low", "ci_high"]).round(2).head(20)"""),
        code("""display(table("sa_seismic_quintiles.csv"))
display(table("rb_seismic_within_ward_variance.csv"))
display(table("rb_seismic_adjusted.csv"))
display(table("x7_cause_auc_difference.csv"))
table("x8_ward_correlates.csv")"""),
        md("""## Collinearity in the joint within-ward model (X15)
The within-ward coefficients come from one model with all 65 covariates and ward indicators. For the
covariates whose interval excludes zero, this compares the joint coefficient with the coefficient when
the covariate enters alone (with ward indicators), and reports its strongest correlate and variance
inflation factor."""),
        code("""table("x15_joint_model_collinearity.csv")"""),
        md("""## Full-rank covariate set (X27)
The 65-covariate set has rank 62: shrub cover is constant and the WorldCover fractions and ownership
classes each sum to one. X27 drops one column of each and recomputes the residual ward ICC, the
three-way variation partition and the earthquake-exposure models with the 62-covariate full-rank set."""),
        code("""run_experiments(only=["X27"])
display(table("x27_full_rank_residual_icc.csv"))
display(table("x27_full_rank_partition.csv"))
table("x27_full_rank_earthquake.csv")"""),
    ],
    # ------------------------------------------------------------------ 10
    "10_robustness.ipynb": [
        md("""# 10 · Robustness (Fig. 8)
Every ward diagnostic repeated in five populations: all sources, springs only (ponds excluded),
perennial sources only, perennial springs only, and sources inside the official polygon of their
recorded palika. The nulls are regrown on each population's own source graph. Running this notebook
reproduces `rb_sensitivity_subsets.csv`, `rb_null_distributions.csv`, the validation-distance tables
and the seismic adjustment (about ten minutes)."""),
        code(SETUP),
        code("""from src.robustness import run as run_robustness
run_robustness()"""),
        code("""fig("fig8_robustness.png")"""),
        code("""table("rb_sensitivity_subsets.csv").set_index("subset").T"""),
        md("## Explore: a small fresh set of contiguous placebo partitions (no cache)"),
        code("""from src.dataset import load_master, coords_to_metric
from src.catchment_analysis import icc1
from src.regions import sample_partitions

df = load_master()
coords = coords_to_metric(df.latitude.to_numpy(), df.longitude.to_numpy())
ward = (df.municipality.astype(str) + "_" + df.ward.astype(str)).to_numpy()
y = df.target_dried.to_numpy().astype(float)
parts = sample_partitions(coords, ward, 50, seed=1, cache=False)
null = np.array([icc1(y, p) for p in parts])
print(f"ward ICC {icc1(y, ward):.3f}; 50 contiguous placebo partitions: "
      f"{np.quantile(null, .025):.3f}-{np.quantile(null, .975):.3f} (max {null.max():.3f})")"""),
    ],
    # ------------------------------------------------------------------ 11
    "11_figures_and_tables.ipynb": [
        md("""# 11 · Figures, supplementary tables and the reproducibility check
The eight main figures and one supplementary figure are drawn from `results/metrics` by `src/figures.py`; the 41 supplementary
tables (and their LaTeX version `paper/si_tables.tex`) by `src/report.py`. Running this notebook last
regenerates all of them from whatever the earlier notebooks computed, which is the same end state as
`python reproduce_all.py`."""),
        code(SETUP),
        code("""from src.figures import run as run_figures
run_figures()"""),
        code("""from src.report import run_report
run_report()
print((ROOT / "paper/si_tables.tex").stat().st_size, "bytes of SI tables")"""),
        code("""for name in sorted(os.listdir(F)):
    if name.endswith(".png"):
        print(name)"""),
        code("""fig("fig5_wards_watersheds.png")"""),
    ],
}


def main(execute=True, only=None):
    for name, cells in NOTEBOOKS.items():
        if only and not any(name.startswith(prefix) for prefix in only):
            continue
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        path = NB / name
        if execute:
            NotebookClient(nb, timeout=7200, kernel_name="python3",
                           resources={"metadata": {"path": str(NB)}}).execute()
        nbf.write(nb, path)
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main(execute="--no-execute" not in sys.argv, only=[a for a in sys.argv[1:] if not a.startswith("--")])
