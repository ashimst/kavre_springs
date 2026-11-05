# Kavre Springs: a leakage-audited spatial benchmark for predicting spring drying

Code, data derivatives and results for the manuscript *"What predicts spring drying in the Nepal Mid-Hills? A
leakage-audited spatial benchmark of tabular and graph models on 5,689 community-mapped springs"* (`paper/`).

Data: ICIMOD (2025) *Status of Springs in Seven Municipalities of Kavre, Nepal*, doi:[10.26066/RDS.1974131](https://doi.org/10.26066/RDS.1974131), CC BY 4.0;
described in Thapa et al. (2025) *Data in Brief* 60:111466, doi:[10.1016/j.dib.2025.111466](https://doi.org/10.1016/j.dib.2025.111466).

---

## Key findings

All numbers are produced by `python reproduce_all.py` (see `paper/si_generated_tables.md`).

1. **Survey skip logic leaks the label.** Dried sources skip three of the seven questionnaire sections, so 46
   processed fields are outcome-dependent. "Was the 2015-earthquake question answered?" alone predicts drying
   with ROC-AUC 0.958; adding it lifts every model to ROC-AUC ≈ 0.99. An earlier version of this repository
   reported those inflated numbers. They are withdrawn (see *Changes* below).
2. **With leak-free covariates, drying is modestly predictable.** On the primary 23-covariate set (terrain, climate,
   vegetation, land use, source type, ownership):

   | Setting | ROC-AUC range across 11 models | Best model |
   |:---|:---:|:---|
   | Spatial block 5-fold CV, no buffer | 0.598–0.641 | HydroBoost-GAT 0.641 ± 0.057 (n.s. vs Random Forest 0.638) |
   | Spatial block 5-fold CV, 500 m buffer | 0.595–0.641 | HydroBoost-GAT 0.641 ± 0.059 (n.s. vs Logistic Regression 0.634) |
   | Leave-one-municipality-out, 500 m buffer | 0.513–0.588 | Logistic Regression 0.588 ± 0.030 |

   No model differs significantly from the best graph-free model (corrected resampled t-test, Holm-adjusted p = 1).
3. **Graph structure adds no measurable value.** GNNs do not beat an MLP trained with the same loop. ResGATv2's
   accuracy is unchanged when its terrain edge attributes are removed or shuffled, or when the graph is replaced
   by random links.
4. **Seasonality is informative but possibly confounded.** The perennial/seasonal flag raises ROC-AUC to
   0.71–0.73. Because the survey ran mostly in the dry season, it is reported only as a sensitivity analysis.
5. **Drying is spatially clustered.** Moran's I is 0.28 at ~160 m and 0.18 at ~480 m; 95 springs are in
   FDR-significant dried clusters. Drying is associated with lower NDVI/NDWI, non-agricultural and non-forest
   surroundings, and pond sources.
6. **The 2015 earthquake shows up in the timing, not as a predictor.** 35% of dried sources are reported to have
   dried in 2072 BS (2015/16), 84% of them attributed to the earthquake by respondents. These earthquake-attributed
   dryings are less predictable from environmental covariates (pooled AUC 0.63–0.64) than other dryings (0.67–0.69).

---

## Repository layout

```
src/
  features.py          feature sets, skip-logic missingness audit, leakage guard (single source of truth)
  graph_builder.py     k-NN graph (k=7), 5-D edge attributes, node features for any feature set
  spatial_splits.py    spatial block k-fold with distance buffer; leave-one-municipality-out
  models/              baselines (LR, RF, XGB, CatBoost, MLP, kNN-label, GCN, SAGE, GATv2),
                       ResGATv2 (with ablation flags), HydroBoost-GAT, shared training loop
  benchmark.py         all experiments, ablations and placebo controls; per-fold CSVs, checkpointed
  spatial_mining.py    Moran's I decay, LISA with BH-FDR, association rules with Fisher tests
  explainability.py    TreeSHAP, logistic coefficients (spatial-block bootstrap), attention vs placebo
  prioritization.py    Spring Priority Index for active springs
  descriptive.py       study-area map; reported drying year/reason; cause-specific discrimination
  report.py            summaries, corrected paired tests, figures, paper/si_generated_tables.md
  data_cleaning.py, gee_extraction.py, visualization/maps.py   upstream data preparation
reproduce_all.py       one command for every number, table and figure
results/metrics/       raw per-fold results (benchmark_*_folds.csv) and all derived metrics
results/tables/        priority index tables, LOMO per-municipality table
results/figures/       all figures
results/predictions/   out-of-fold test predictions
paper/                 manuscript.tex, supplementary_information.md, si_generated_tables.md, references.bib
notebooks/             01–03 EDA and covariate extraction; 04–06 display pipeline outputs
tests/                 leakage guards, split/buffer properties, model invariants, result consistency
```

## Reproduce

```bash
pip install -r requirements.txt          # Python 3.12; install torch for your platform first
python reproduce_all.py --quick          # smoke test, ~1 min (writes *_quick files only)
python reproduce_all.py                  # everything, ~35 min on one GPU
python reproduce_all.py --skip-benchmark # regenerate analyses/figures from existing per-fold CSVs
pytest -q tests
cd paper && latexmk -pdf manuscript.tex  # figures are read from ../results/figures
```

The pipeline starts from `data/processed/master_dataset.parquet`. Re-extracting the remote-sensing covariates
(`src/gee_extraction.py`, notebook 03) requires a Google Earth Engine account.

## Evaluation protocol (summary)

* **Primary features** exclude every field affected by survey skip logic and every label-derived field. This is
  enforced by `src/features.check_feature_set` and unit-tested.
* **Spatial block CV**: ~8 km grid cells assigned to 5 folds. Test fold *i*, validation fold *i+1* (early stopping,
  threshold, HydroBoost blend). Five seeds re-shuffle blocks and re-initialise models. Evaluated with 0 m
  (interpolation) and 500 m (extrapolation) buffers.
* **LOMO**: each municipality held out, spatial validation blocks, 500 m buffer, 3 seeds.
* **Statistics**: paired over seed × fold, corrected resampled t-test (Nadeau & Bengio 2003) with Holm adjustment.
* Hyperparameters are fixed a priori (SI Table S12). No test-fold tuning.

## Changes from the previous version (Nov 2026 audit)

The earlier README, manuscript and SI reported ROC-AUC ≈ 0.986 and "state-of-the-art" results for HydroBoost-GAT.
These were withdrawn because:

- the earthquake skip-logic fields leaked the label;
- the progressive ablation table was hard-coded rather than computed;
- the placebo ("Frog Trap") results had no generating code, and their numbers did not support the stated
  conclusion;
- the per-municipality LOMO table listed municipalities that are not in the data;
- the hyperparameter search was never run;
- the attention significance test defined its groups using labels and trained on all labels;
- the HSVI table had no code and scored dried springs on fields that are missing for them;
- two references could not be verified.

All of these are replaced by computed, tested equivalents (see SI Text S6).

## License

Code: MIT. Data derivatives: CC BY 4.0, inherited from the ICIMOD dataset. Cite ICIMOD (2025) and Thapa et al. (2025).
