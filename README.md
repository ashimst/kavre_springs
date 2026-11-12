# Kavre springs: landscape covariates versus survey structure in recorded spring drying

A statistical audit of the open citizen-science spring inventory of Kavrepalanchok, Nepal: 5,689 springs and ponds
mapped ward by ward in 75 wards of seven municipalities (26.7 % recorded as dried).

The repository turns the published inventory into an **analysis-ready, leakage-audited dataset with 67 label-free
remote-sensing and terrain covariates**, and asks one data-science question:

> When municipalities rely on large, ward-enumerated citizen-science inventories, how much of the recorded drying
> signal reflects questionnaire design and administrative survey structure, and how much reflects remote-sensing
> and terrain covariates?

**One command rebuilds every number, table and figure from the raw inventory, on a CPU, bit for bit.**

The work is a methodological follow-up to the inventory's data paper (Thapa et al., 2025, *Data in Brief*) and its
descriptive analysis (Pandit et al., 2026, *Environmental Challenges*). It does not repeat their descriptive results.
Terrain and remote-sensing variables are treated as empirical covariates: the analyses report associations, not
physical mechanisms.

---

## Contents

- [Key findings](#key-findings)
- [Quick start](#quick-start)
- [Pipeline](#pipeline)
- [Data](#data)
- [Methods at a glance](#methods-at-a-glance)
- [Notebooks](#notebooks)
- [Repository layout](#repository-layout)
- [Reproducibility](#reproducibility)
- [Experiments that were tried and removed](#experiments-that-were-tried-and-removed)
- [Limitations](#limitations)
- [Citation and license](#citation-and-license)

---

## Key findings

| Question | Result |
|:---|:---|
| Does questionnaire skip logic leak the outcome? | Yes. 41 of 56 non-outcome raw survey fields exceed the 5-percentage-point missingness-gap screen; one unanswered question predicts drying with ROC-AUC 0.958 |
| What does each data layer add? | Random forest ROC-AUC 0.638 (base) → 0.683 (+ terrain and catchment indices, water balance, roads) → 0.693 (+ 28 Earth Engine layers; no detectable gain) |
| How far does prediction transfer? | 0.683 → 0.652 with 2 km buffers → 0.635 for an unseen municipality; kNN label interpolation 0.598 |
| Does the seasonal-flow record shift the label? | Yes. 63 % of seasonal vs 20 % of perennial sources recorded as dried; the flag lifts ROC-AUC to 0.75 |
| Do terrain types differ in recorded drying? | Rates vary descriptively (13–40 %), but the association is not significant after shuffling outcomes within 8 km blocks (p = 0.187); ward effects remain the stronger result |
| Do delineated catchments organise drying? | At 2 km², catchments show no detectable excess over their matched-grid null (ICC 0.122 in 0.116–0.135); wards exceed the tested null families. Separate null comparisons do not establish that wards outperform catchments |
| Do survey wards? | Yes. Ward ICC 0.132 exceeds all 1,000 draws of rotated grids (max 0.116), contiguous size-matched regions (max 0.106) and regions nested within municipalities (max 0.117), also after covariate adjustment and for springs only; funnel φ = 5.5, 13 of 75 wards outside 99.8 % limits |
| Pairs of nearby sources | Same ward OR 1.56; same catchment OR 1.11 (n.s.); no upslope/downslope asymmetry |
| Where is the spatial structure? | Three-way partition: space only −0.001, space ∩ ward 0.035 |
| Association rules | 3,321 significant, productive rules for drying; 79 % hold within wards; 15 reverse (Simpson-type) |
| Recalled causes and years | Ward ICC 0.23–0.28 vs 0.03–0.05 within terrain types; Knox 1.61 within wards, no excess across |
| What survives ward fixed effects? | TPI, ALOS slope position, CHILI and land cover; 2015 PGA does not (only 11 % of its variance lies within wards) |
| Robustness | Ward signal holds for springs only, perennial sources, perennial springs and correctly located sources |
| Do complex models help? | No. MLP, Kolmogorov–Arnold and graph neural networks never beat the random forest |

The data show ward-level structure. They suggest, but do not prove, that collection practice produced it: each ward
was mapped by its own pair of resource persons, and the published data contain no enumerator ID or visit date.

**Predictive benchmark** (ROC-AUC, mean ± SD over 5 seeds × 5 spatial folds; `env_hydro` = 39 covariates,
`env_rich` = 67):

| Model | terrain tier, no buffer | terrain tier, 2 km buffer | terrain tier, unseen municipality | enriched, no buffer |
|:---|:---:|:---:|:---:|:---:|
| Logistic regression | 0.669 ± 0.052 | 0.650 ± 0.056 | 0.630 ± 0.035 | 0.664 ± 0.065 |
| Random forest | 0.683 ± 0.061 | 0.652 ± 0.058 | 0.635 ± 0.045 | 0.693 ± 0.060 |
| CatBoost | 0.678 ± 0.058 | 0.644 ± 0.049 | 0.638 ± 0.038 | 0.684 ± 0.051 |
| kNN interpolation of labels | 0.598 ± 0.043 | – | 0.513 ± 0.071 | – |

After Holm adjustment no covariate model differs significantly from the best one in any design.

---

## Quick start

```bash
git clone https://github.com/ashimst/kavre_springs && cd kavre_springs
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # Python 3.12, CPU only

python reproduce_all.py --skip-benchmark                # all analyses, tables and figures (~11 min)
python reproduce_all.py                                 # also re-run the predictive benchmark (about 1 h more)
python reproduce_all.py --skip-benchmark --notebooks    # also rebuild and execute the notebooks
pytest -q                                               # 28 unit tests
```

Individual stages can be run on their own, for example `python -m src.figures --only fig5_wards_watersheds` or
`python -m src.assemble_master --check`.

---

## Pipeline

`reproduce_all.py` runs, in order:

| Stage | What happens | Code |
|:---|:---|:---|
| 0 | Raw inventory → cleaned table → master table, checked against the versioned file | `src/data_cleaning.py`, `src/assemble_master.py` |
| 1 | USGS ShakeMap and Copernicus DEM downloaded once into `data/external/`; seismic exposure and DEM routing | `src/seismic.py`, `src/hydro_gis.py` |
| 2 | Official palika and district boundaries, OpenStreetMap ward polygons; location audits | `src/boundaries.py`, `src/ward_polygons.py` |
| 3 | Leakage guard on every feature set (aborts if a predictor leaks) | `src/features.py` |
| 4 | Predictive benchmark (checkpointed per fold; skipped with `--skip-benchmark`) | `src/benchmark.py` |
| 5 | Leakage audit tables, unsupervised typology, association rules, spatial and unit-level tests, drivers, robustness | `src/descriptive.py`, `src/unsupervised.py`, `src/pattern_mining.py`, `src/spatial_*.py`, `src/catchment_analysis.py`, `src/drivers_analysis.py`, `src/robustness.py` |
| 6 | The eight figures, drawn from the result tables | `src/figures.py`, `src/plotstyle.py` |
| 7 | Benchmark summaries, paired tests and Markdown tables for the analysis record | `src/report.py` |
| 8 | Manuscript build (only if a local `paper/` folder exists) | `src/build_paper.py` |
| 9 | Notebooks (only with `--notebooks`) | `notebooks/build_notebooks.py` |
| 10 | Unit tests | `tests/` |

All outputs land in `results/`: `metrics/` (49 result tables), `figures/` (fig1–fig8), `predictions/` (out-of-fold
predictions) and `tables/` (`si_tables.md`, per-municipality scores).

---

## Data

| File | Content | Source |
|:---|:---|:---|
| `data/raw/spring_data.csv` | ICIMOD inventory, 5,689 sources | ICIMOD (2025), [doi:10.26066/RDS.1974131](https://doi.org/10.26066/RDS.1974131), CC BY 4.0 |
| `data/processed/clean_springs.csv` | cleaned, encoded inventory | `src/data_cleaning.py` |
| `data/processed/gee_features.csv` | SRTM terrain, CHIRPS precipitation, Sentinel-2 NDVI/NDWI | `src/gee_extraction.py` (Earth Engine) |
| `data/processed/gee_hydro_catchment_features.csv` | MERIT Hydro upstream area and HAND, TerraClimate water balance | `src/extract_hydro_catchment_gee.py` (Earth Engine) |
| `data/processed/osm_road_features.csv` | road distance, density, upslope road | `src/extract_osm_roads.py` (OpenStreetMap) |
| `data/processed/gee_enrichment_features.csv` | 28 layers: WorldCover, Hansen, SoilGrids, MODIS, CHIRPS trends, GHSL, WorldPop, VIIRS, ALOS (slope position, CHILI, multi-scale TPI) | `src/extract_enrichment_gee.py` (Earth Engine) |
| `data/processed/seismic_features.csv` | 2015 PGA and aftershock exposure | `src/seismic.py` (USGS ShakeMap, ComCat) |
| `data/processed/hydro_gis_features.csv` | Copernicus GLO-30 D8 link catchments, flow length, Strahler order | `src/hydro_gis.py` |
| `data/processed/hydro_gis_matched_catchments.csv` | link catchments at the area threshold (6.25 km²) that gives 75 occupied catchments, as many as wards | `src/hydro_gis.py` |
| `data/processed/boundaries/` | official palika and district boundaries; ward polygons | `src/boundaries.py` (MoFAGA via geoBoundaries, CC BY-SA 4.0); `src/ward_polygons.py` (OpenStreetMap, ODbL) |
| `data/processed/master_dataset.parquet` | the analysis table | `src/assemble_master.py` |
| `data/processed/landscape_types.csv` | label-free terrain types | `src/unsupervised.py` |

The Earth Engine and OpenStreetMap extractions are versioned, so reproducing the results needs no credentials.

**Covariate tiers.** `env` (base: terrain, precipitation, Sentinel-2, structural survey fields) → `env_hydro`
(39 covariates: + MERIT Hydro, TWI, TPI, TerraClimate, roads) → `env_rich` (67 covariates: + 28 Earth Engine
layers). The unsupervised analyses use a separate *landscape set* of 46 continuous, static covariates that excludes
survey fields and change variables.

**What "dried" means.** The label is the source condition (active or dried) recorded by the resource person at an
undated visit between October 2023 and June 2024. There is no field criterion and no discharge measurement for dried
sources. Ponds and seasonal sources are analysed as sensitivity cases.

**Leakage guard.** No predictor may come from a field whose missingness differs by more than 5 percentage points
between dried and active sources, and no predictor may be computed from labels (`src/features.check_feature_set`,
unit-tested).

**Boundaries and locations.** 93.3 % of sources lie inside the official polygon of their recorded palika. Ward
polygons are not openly available, so the analyses use recorded ward assignments and no ward map is drawn.

---

## Methods at a glance

| Analysis | Method | Main outputs |
|:---|:---|:---|
| Leakage audit | missingness indicators by outcome, ROC-AUC | `leakage_audit.csv`, `single_feature_auc.csv` |
| Prediction | logistic regression, random forest, CatBoost, kNN label interpolation; imputation and scaling fitted within folds; spatial block CV, 0.5–2 km buffers, leave-one-municipality-out; corrected resampled t-test with Holm adjustment; fold spread, per-municipality and springs-only scores | `benchmark_*`, `paired_tests_vs_best_tabular.csv`, `x1_*`, `x2_*`, `x11_*` |
| Terrain typology | PCA + Gaussian mixtures; BIC/bootstrap-stability selection rule fixed before drying was examined | `un_typology_*`, `un_pca_*` |
| Association rules | apriori; Fisher tests with Benjamini–Hochberg FDR; productivity pruning; WRAcc; ward-stratified Mantel–Haenszel OR; held-out block replication | `pm_*` |
| Unit homogeneity | ICC of wards, catchments and landscape regions vs 200 size-matched grids; wards also vs 1,000 rotated grids, 1,000 contiguous size-matched regions and 1,000 regions nested in municipalities (every region verified connected; `src/regions.py`) | `ca_unit_icc.csv`, `un_regionalisation.csv`, `x10_*.csv`, `rb_null_distributions.csv` |
| Pairs and direction | pair-concordance logit (same ward, same catchment, distance); upslope/downslope permutation test; flow paths | `ca_pair_*`, `sa_directional_test.csv`, `ca_flowpath_logit.csv` |
| Variation partitioning | environment, 250 Moran eigenvector maps, ward (adjusted McFadden pseudo-R²) | `dr_three_way_*` |
| Ward funnel | observed vs landscape-expected drying per ward; overdispersion φ | `un_ward_funnel*` |
| Recalled attributes | ICC of causes and 2072 BS dating; Knox test within and across wards | `un_dried_icc.csv`, `sa_knox_test.csv`, `drying_*` |
| Within-ward covariates | pooled vs ward fixed-effects logistic regression; spatial-block bootstrap | `dr_pooled_vs_within_ward.csv` |
| Earthquake exposure | PGA under environmental, MEM and ward adjustment; within-ward variance share | `rb_seismic_*`, `sa_seismic_*` |
| Sensitivity | springs only, perennial only, perennial springs, sources inside their recorded palika | `rb_sensitivity_subsets.csv` |

---

## Notebooks

The notebooks are the analysis itself: each one runs its part of the pipeline (writing the same files
as `reproduce_all.py`) and then shows the resulting tables and figures. Run them in order on a fresh
checkout; the executed copies are committed, and `python notebooks/build_notebooks.py` rebuilds and
re-executes all of them. The predictive benchmark and the placebo-partition draws are checkpointed, so
only the first run is slow.

| Notebook | Runs | Shows |
|:---|:---|:---|
| `01_inventory_and_labels` | boundary audit, raw-field skip-logic audit | the inventory, the label, wards |
| `02_covariate_extraction` | verification of the versioned extractions (re-extraction needs Earth Engine credentials) | coverage, tiers |
| `03_gee_enrichment` | leakage check, univariate associations | enrichment-layer behaviour |
| `04_leakage_and_prediction` | the full benchmark, tier tests, within-ward AUC, fold spread | Fig. 2, leakage tables |
| `05_landscape_typology` | PCA, Gaussian-mixture typology, regions, funnel, recalled ICC | Fig. 3 |
| `06_wards_vs_catchments` | unit ICCs, pair and flow-path models, partitioning, placebo nulls (X3-X5, X10), direct ward-catchment contrasts and pair-model robustness (X12-X14) | Fig. 5, Fig. S1 |
| `07_apriori_pattern_mining` | rule mining, ward-stratified tests, reversal FDR | Fig. 4 |
| `08_recalled_causes` | causes, years, cause-specific AUC, Knox test | Fig. 6 |
| `09_drivers_and_earthquake` | within-ward covariates, grid fixed effects, collinearity of the joint model (X15), seismic adjustment | Fig. 7 |
| `10_robustness` | all sensitivity populations, validation distances | Fig. 8 |
| `11_figures_and_tables` | the nine figures, Fig. S1 and the 62 SI tables | the full set |

---

## Repository layout

```
data/
  raw/                     ICIMOD inventory (CSV, XLSX)
  processed/               cleaned table, covariate extractions, master table, boundaries
  external/                downloaded DEM and ShakeMap grids (git-ignored, fetched automatically)
src/
  data_cleaning.py, assemble_master.py        raw inventory -> analysis table
  gee_extraction.py, extract_hydro_catchment_gee.py, extract_enrichment_gee.py, extract_osm_roads.py
                                              covariate extraction (credentials / network; outputs versioned)
  seismic.py, hydro_gis.py, boundaries.py     ShakeMap exposure; DEM routing; official boundaries
  ward_polygons.py                            OpenStreetMap ward polygons and recorded-ward audit
  features.py, dataset.py, spatial_splits.py  feature sets and leakage guard; design matrices; spatial CV and LOMO
  models.py, benchmark.py                     the four models; checkpointed benchmark
  unsupervised.py, pattern_mining.py          terrain typology, regions, funnel, recalled ICC; association rules
  spatial_mining.py, spatial_analysis.py      Moran's I; directional test, Knox, ward concordance, MEMs, shaking
  catchment_analysis.py, drivers_analysis.py  catchments vs wards vs grids; within-ward covariates, partitioning
  robustness.py, descriptive.py               nulls, sensitivity subsets, validation distances; causes and timing
  regions.py, revision_experiments.py         contiguous size-matched placebo partitions; experiments X0-X28
  figures.py, plotstyle.py                    the nine figures and their shared visual style
  report.py, build_paper.py                   summaries, paired tests; single-file manuscript build helper
notebooks/                 executed notebooks and build_notebooks.py
results/                   metrics/, figures/, predictions/, tables/
tests/                     leakage guards, splits, models, statistical building blocks
docs/experiment_log.md     every experiment, kept or removed, with its result
reproduce_all.py           one command for everything
requirements.txt, LICENSE, CITATION.cff, pytest.ini
```

The manuscript sources (`paper/`) are kept locally and are not part of the repository.

---

## Reproducibility

- **Bit-for-bit on a CPU.** Every model is seeded independently, so results do not depend on which other models are
  run. Eigenvector computations use a fixed starting vector. A clean clone reproduces all 49 result tables exactly
  (largest difference 0).
- **Offline from versioned inputs.** The raw inventory and every covariate extraction are versioned. The DEM,
  ShakeMap grids and official boundaries are public downloads, fetched once and cached.
- **Pinned environment.** Exact package versions are in `requirements.txt` (Python 3.12.8). No GPU or deep-learning
  stack is needed.
- **Checks.** `python -m src.assemble_master --check` verifies that the master table rebuilds identically from the
  raw inventory, and `pytest` runs 28 unit tests covering leakage guards, spatial splits, model determinism and the
  statistical building blocks.

---

## Experiments that were tried and removed

Only experiments that change a conclusion, add a significant result or test the research question are kept. The
rest are documented with their numbers in [`docs/experiment_log.md`](docs/experiment_log.md), among them:

- **Graph neural networks** (GCN, GraphSAGE, GATv2 on five spring graphs): none beat a graph-free network.
- **Multilayer perceptron**: equal to logistic regression.
- **Kolmogorov–Arnold networks**: 0.668–0.676, never better than the random forest.
- **XGBoost**: duplicated CatBoost.
- **HDBSCAN, t-SNE, isolation forest, Louvain communities, LISA and scan-statistic maps, Mantel tests**: redundant,
  descriptive or underpowered.

---

## Limitations

- One district and one inventory; the diagnostics transfer, but effect sizes may not.
- A single cross-section; dates and causes of drying are recalled.
- "Dried" is recorded at an undated visit without an operational criterion or discharge measurement.
- Ward and survey team are fully confounded by design; ward effects cannot be attributed to enumerators without
  enumerator metadata.
- Earthquake exposure is an interpolated model product and varies mostly between wards.

---

## Citation and license

If you use this code or the enriched dataset, please cite this work (`CITATION.cff`) and the original inventory:

- ICIMOD (2025). *Status of Springs in Seven Municipalities of Kavre, Nepal* [Data set].
  [doi:10.26066/RDS.1974131](https://doi.org/10.26066/RDS.1974131)
- Thapa, S., Pandit, A., Bhuchar, S., Dhakal, M. (2025). Citizen science approach for springshed management.
  *Data in Brief* 60, 111466. [doi:10.1016/j.dib.2025.111466](https://doi.org/10.1016/j.dib.2025.111466)
- Pandit, A., Thapa, S., Bhuchar, S., Dhakal, M., Kafle, S. (2026). Community-driven assessment of springs and
  ponds. *Environmental Challenges* 22, 101394. [doi:10.1016/j.envc.2025.101394](https://doi.org/10.1016/j.envc.2025.101394)

Code is MIT licensed (`LICENSE`). Data derivatives are CC BY 4.0, inherited from the ICIMOD dataset; official
boundaries are CC BY-SA 4.0 (geoBoundaries); ward polygons are (c) OpenStreetMap contributors, ODbL 1.0.
