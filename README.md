# Wards, not watersheds

Code, derived data, notebooks and results for *"Wards, not watersheds: unsupervised and spatial data mining of spring
drying in a community inventory of 5,689 water sources, Nepal Himalaya"*. arXiv preprint package:
`paper/arxiv_submission.zip`. Target journal: *Hydrology and Earth System Sciences*.

Data: ICIMOD (2025) *Status of Springs in Seven Municipalities of Kavre, Nepal*,
doi:[10.26066/RDS.1974131](https://doi.org/10.26066/RDS.1974131), CC BY 4.0; described in Thapa et al. (2025)
*Data in Brief* 60:111466, doi:[10.1016/j.dib.2025.111466](https://doi.org/10.1016/j.dib.2025.111466).
The inventory is enriched with 67 label-free covariates:
- Google Earth Engine layers: MERIT Hydro, TerraClimate, CHIRPS, Sentinel-2, ESA WorldCover, Hansen forest change,
  SoilGrids, MODIS NDVI/ET, GHSL, WorldPop, VIIRS and ALOS landforms;
- OpenStreetMap roads;
- the Copernicus GLO-30 DEM and USGS ShakeMap/ComCat, which are downloaded and cached automatically.

## Key findings

All numbers are produced by `python reproduce_all.py` (`paper/si_generated_tables.md`, Tables S2–S55).

1. **Landscape structures drying.** Eight stable landscape types, built from 46 GIS covariates without labels or
   survey fields, dry at 13–40 %. Peri-urban built-up settings and steep, water-limited slopes high above drainage
   dry most. Label-free landscape regions are more drying-homogeneous than size-matched grids in the full data, but
   not once seasonal sources are excluded.
2. **Apriori patterns:** 3,321 significant, productive rules for drying and 2,337 for activity.
   - 79 % (drying) and 81 % (activity) stay significant within wards (ward-stratified Mantel–Haenszel test, BH
     q < 0.05). This summarises robustness after selection; it is not independent confirmation. 78–86 % replicate
     on held-out spatial blocks.
   - The most robust patterns involve topographic position, height above drainage and water deficit; "low position &
     moist soil" is strongly protective (confidence 0.93, within-ward OR 4.0).
   - 15 rules reverse within wards (Simpson-type).
3. **Watersheds do not structure drying.** DEM catchments are no more homogeneous than random squares. No upslope→downslope
   propagation is detected, and flow-path covariates are uninformative after adjustment.
4. **Wards do, without being landscape units.** Wards are as landscape-mixed as random squares, yet they are the
   most drying-homogeneous units: ICC 0.132, against 0.093–0.106 for matched grids and 0.081–0.111 for contiguous
   random regions of ward size. This holds without ponds and seasonal sources.
   - Against a landscape-only model, 13 of 75 wards fall outside 99.8 % funnel limits (0.15 expected); φ = 5.5.
   - Spatial structure in drying is almost entirely shared with wards.
5. **Recalled causes and years are ward-shared.**
   - Same-year clustering appears only within wards (Knox).
   - Cause and year ICCs within wards are 0.23–0.28, against 0.11–0.22 for matched grids and ≈0.03–0.05 within
     landscape types.
   - Wards form six narrative regimes. No association of narrative similarity with geography or landscape was
     detected (Mantel p ≥ 0.25; low power with 56 wards).
   - **Interpretation.** The data demonstrate ward-level structure. They suggest, but do not prove, that collection
     practices produced it. The inventory has no enumerator ID or visit date, so this is ward-associated
     heterogeneity consistent with survey-process effects.
6. **Drivers within wards:** topographic position, ALOS slope position and heat load.
   - Earthquake results are associative only. 12 May PGA is associated with drying after environmental and MEM
     adjustment but not with ward fixed effects.
   - 89 % of shaking variance lies between wards, so the two cannot be separated.
7. **Prediction:** skip logic leaks the label (ROC-AUC 0.958 from one unanswered question).
   - Leak-free ROC-AUC is at most 0.69, including the enrichment layers. It falls to 0.65 with 2 km buffers and
     0.64 for an unseen municipality.
   - No graph topology improves GNNs over a graph-free MLP.

## Notebooks (executed; open to see every table and figure)

| Notebook | Content |
|:---|:---|
| `01–03` | exploratory analysis and covariate extraction |
| `03b_gee_enrichment` | Earth Engine layers: coverage, agreement with survey, univariate association |
| `04_leakage_and_prediction` | skip-logic audit, benchmark, feature sets, LOMO |
| `05_landscape_typology` | PCA, Gaussian-mixture typology, HDBSCAN, t-SNE, anomalies |
| `06_spatial_structure` | Moran's I, LISA, Moran eigenvector maps, scan statistic |
| `07_wards_vs_watersheds` | ICC vs matched grids, regionalisation, pairs, flow paths, partitioning (interactive ICC) |
| `08_apriori_pattern_mining` | apriori rules, ward robustness, reversals, seasonal flag (interactive re-mining) |
| `09_dried_source_narratives` | causes, years, Knox, ward ICC, narrative rules |
| `10_ward_effects` | funnel plot, narrative regimes, Mantel tests |
| `11_drivers_and_earthquake` | within-ward drivers, ShakeMap exposure |
| `12_graph_models` | GNNs on five graph topologies |
| `13_robustness` | ponds/seasonal sensitivity, contiguous random-region null, validation distances, earthquake adjustment |

Rebuild with `python notebooks/build_notebooks.py`. The notebooks read cached results; run the pipeline first.

## Repository layout

```
src/
  data_cleaning.py, gee_extraction.py, extract_hydro_catchment_gee.py, extract_osm_roads.py,
  extract_enrichment_gee.py   data preparation and covariate extraction (Earth Engine / OpenStreetMap)
  seismic.py              USGS ShakeMap PGA/PGV + ComCat aftershock exposure per source
  hydro_gis.py            Copernicus DEM D8 routing (pysheds): link catchments, flow length, Strahler order
  features.py             feature sets, skip-logic missingness audit, leakage guard
  graph_builder.py        master-table loader, k-NN spring graph, edge attributes
  spatial_splits.py       spatial block k-fold with buffer; leave-one-municipality-out
  models/                 tabular baselines, MLP, GNNs, ResGATv2, HydroBoost-GAT, training loop
  benchmark.py            predictive benchmark and placebo graphs (checkpointed per fold)
  unsupervised.py         landscape typology, HDBSCAN, t-SNE, isolation forest, regionalisation, ward funnel,
                          ward narrative regimes with Mantel tests
  pattern_mining.py       apriori rules with FDR, productivity, ward-stratified MH odds ratios, spatial replication
  spatial_mining.py       Moran's I decay, LISA with BH-FDR
  spatial_analysis.py     seismic dose-response, directional test, Knox, ward concordance, MEM partitioning, scan
  catchment_analysis.py   ICC of catchments/wards vs size-matched grids, pair concordance, flow paths
  drivers_analysis.py     pooled vs ward-fixed-effect drivers; three-way partition
  graph_topology_experiment.py  GNNs on kNN / catchment / ridge / ward / random graphs
  descriptive.py          study-area map; reported drying year/reason; cause-specific discrimination
  robustness.py           review analyses: random-region null, label sensitivity, validation distances, seismic FE
  report.py               benchmark summaries, paired tests, SI tables (markdown + LaTeX)
reproduce_all.py          one command for every number, table and figure
notebooks/                analysis notebooks (see above) and build_notebooks.py
results/                  metrics/ (all tables), figures/, predictions/, tables/
paper/                    manuscript.tex, supplement_text.tex, si_tables.tex, references.bib, manuscript.bbl,
                          figures/, make_bbl.py, build_arxiv.py, arxiv_submission.zip
tests/                    leakage guards, splits, model invariants, statistical building blocks
```

## Reproduce

```bash
pip install -r requirements.txt           # Python 3.12; install torch for your platform first
python reproduce_all.py --quick           # benchmark smoke test (writes *_quick files only)
python reproduce_all.py                   # everything (several hours on one GPU; benchmarks are checkpointed)
python reproduce_all.py --skip-benchmark --notebooks   # all analyses, arXiv package and notebooks from cached folds
pytest -q tests
```

For arXiv, run `python paper/build_arxiv.py` and upload `paper/arxiv_submission.zip`. It contains `manuscript.tex`,
the generated `manuscript.bbl`, the supplement and the figures. The same zip compiles on Overleaf.

## Design principles

* No unit, typology, region, community, rule item or graph is built from drying labels. Landscape units use only
  continuous GIS layers, not survey-recorded fields.
* Every homogeneity statistic is compared with randomly shifted grids that have the same number of units. Ward
  statistics are also compared with contiguous random regions that have the ward size distribution.
* Every mined rule is tested with FDR control and productivity pruning, then within wards and on held-out blocks.
* Predictors exclude every skip-logic and label-derived field (`src/features.check_feature_set`, unit-tested).
* Model comparisons are paired over seed × fold and use the corrected resampled t-test with Holm adjustment.

## License

Code: MIT. Data derivatives: CC BY 4.0, inherited from the ICIMOD dataset. Cite ICIMOD (2025) and Thapa et al. (2025).
