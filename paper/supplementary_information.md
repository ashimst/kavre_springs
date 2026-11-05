# Supplementary Information

## What predicts spring drying in the Nepal Mid-Hills? A leakage-audited spatial benchmark of tabular and graph models on 5,689 community-mapped springs

All result tables (S2–S11) are generated automatically by `python reproduce_all.py` into
[`si_generated_tables.md`](si_generated_tables.md); they must not be edited by hand. This file
contains the static methodological material.

---

### Text S1. Survey skip logic and the leakage audit

The ICIMOD Kavre survey (ICIMOD, 2025; Thapa et al., 2025) records 57 fields per water source. Its
digital form has seven sections for active sources but only four for dried sources (location, source
information plus dried-only questions, photographs). Every field in the skipped sections — discharge
measurements, perceived discharge trend, household dependency, water-user groups, tapping, revival
measures and *"Any impact due to 2015 earthquake?"* — is therefore missing for ≈100 % of dried sources
and present for ≈94 % of active sources (Table S4).

An earlier version of this analysis encoded the earthquake question as an ordinal variable (missing
→ −1) plus an "answer known" indicator. The indicator alone gives ROC-AUC 0.958 and balanced accuracy
0.958 — as good as every model reported in that version. This is target leakage in the sense of
Kaufman et al. (2012): the feature is a by-product of the outcome. It also made the earthquake appear
as the dominant "driver" of drying in SHAP analyses, which is not supported.

The pipeline now enforces two rules (`src/features.py`, tested in `tests/test_leakage_and_splits.py`):

1. No model feature may be derived from a column whose missingness differs by more than 5
   percentage points between dried and active sources (checked on the raw, pre-imputation column).
2. Columns computed from the label (`dist_to_nearest_dried_m`, `dist_to_nearest_active_m`,
   `muni_drying_rate`) are forbidden.

**Seasonality.** The perennial/seasonal flag is asked of all sources and passes rule 1. However,
fieldwork ran from October 2023 to June 2024 (mostly dry season; Thapa et al., 2025), so a seasonal
spring visited while not flowing may have been recorded as dried. We therefore exclude it from the
primary feature set and report it as a sensitivity analysis (Table S5).

---

### Table S1. Primary feature set `env` (23 covariates)

| Feature | Group | Source | Native resolution | Notes |
|:---|:---|:---|:---:|:---|
| `latitude`, `longitude` | Location | Survey GPS (WGS84) | point | Capture unmodelled regional gradients |
| `spring_density_500m` | Location | Derived from survey | 500 m radius | Count of other mapped sources (active **or** dried) within 500 m; label-free |
| `elevation_m`, `slope_deg` | Terrain | SRTM (Farr et al., 2007) | 30 m | |
| `aspect_sin`, `aspect_cos` | Terrain | SRTM | 30 m | Eastness / northness (circular encoding of aspect) |
| `precip_annual_mm`, `precip_monsoon_mm`, `precip_dry_season_mm` | Climate | CHIRPS daily (Funk et al., 2015), 2018–2023 | 0.05° (~5 km) | Means of annual, Jun–Sep and Oct–May totals; strongly collinear |
| `ndvi_mean`, `ndwi_mean` | Vegetation / moisture | Sentinel-2 SR (Drusch et al., 2012), 2022–2023 | 10–20 m | Contemporaneous with the survey: may partly reflect, not only cause, drying |
| `Source Type_encoded` | Survey (structural) | Field survey | point | spring = 0, pond = 1 |
| `landuse_*` (7 one-hot) | Survey (structural) | Field survey | point | agriculture, dense forest, fallow, grassland, other, settlement, sparse forest |
| `ownership_*` (3 one-hot) | Survey (structural) | Field survey | point | community, private, public |

Sensitivity set `env_season` adds `Perennial / Seasonal_encoded`. The diagnostic set
`leaky_reference` further adds `earthquake_impact_ordinal` and `earthquake_impact_known` and is used
**only** to quantify leakage. 207 missing remote-sensing values (22 springs outside valid pixels) are
median-imputed; all features are z-scored over all springs (label-free, transductive).

---

### Text S2. Graph and edge attributes (as implemented in `src/graph_builder.py`)

Coordinates are projected to a local equirectangular plane (reference latitude 27.57° N). For each
spring *i* and each of its *k* = 7 nearest neighbours *j* we add the directed edge (*i* → *j*); in
PyTorch Geometric's message-passing convention *j* receives the message from *i*, and GATv2 attention
is softmax-normalised over each receiver's incoming edges (plus a self-loop). With
*d<sub>ij</sub>* the planar distance (set to 1 m for the 96 pairs of co-located sources):

$$
\mathbf{e}_{ij} = \Big[\log(1+d_{ij}),\;\frac{x_j-x_i}{d_{ij}},\;\frac{y_j-y_i}{d_{ij}},\;\Delta z_{ij}=z_j-z_i,\;\frac{\Delta z_{ij}}{d_{ij}}\Big]
$$

so Δ*z* > 0 means the receiver is upslope of the sender. Each column is z-scored over all edges (edge
attributes contain no label information). The mean edge length is ≈262 m (Table S9, k = 7).

---

### Text S3. Evaluation protocol

* **Spatial block CV.** The ~40 × 39 km survey extent is divided into a 5 × 5 grid (~8 km cells);
  non-empty cells are shuffled and dealt into five folds. Fold *i* is the test set, fold *i*+1 the
  validation set, the rest training. Each of five seeds (42, 123, 456, 789, 2024) re-shuffles the
  block assignment and re-initialises all models.
* **Buffer.** With a 500 m buffer, springs within 500 m of any test spring are removed from training
  and validation (Roberts et al., 2017). They remain in the graph without labels. The 0 m setting
  corresponds to *interpolation* (filling gaps among surveyed springs); 500 m approximates
  *extrapolation* to unsurveyed terrain.
* **LOMO.** Each municipality is held out in turn; ~20 % of the remaining 8 × 8 grid cells form the
  validation set; a 500 m buffer is applied around the held-out municipality; three seeds.
* **Label use.** Every model sees training labels only. Validation labels are used for early
  stopping (GNNs, MLP: max 300 epochs, patience 40, AdamW, lr 3·10⁻³, weight decay 10⁻⁴,
  class-balanced cross-entropy), for HydroBoost blend selection and for choosing the decision
  threshold (maximising validation macro-F1). Test labels are never used.
* **Metrics.** ROC-AUC and PR-AUC (threshold-free; primary), macro-F1 and balanced accuracy at the
  validation threshold, Brier score. Note that PR-AUC depends on test-fold prevalence, which varies
  across blocks (9–40 % across municipalities).
* **Statistics.** Paired differences over (seed, fold) with the corrected resampled *t*-test
  (Nadeau & Bengio, 2003; Bouckaert & Frank, 2004) and Holm adjustment within each setting. The
  reference is the best graph-free tabular model in that setting.
* **Hyperparameters** were fixed a priori (Table S12) and not tuned on test folds; the only
  data-driven choices are early stopping, thresholds and the HydroBoost blend, all on validation.

### Table S12. Fixed hyperparameters

| Model | Settings |
|:---|:---|
| Logistic regression | L2, C = 1, class-balanced |
| Random forest | 300 trees, max depth 12, class-balanced |
| XGBoost | 300 trees, depth 6, lr 0.05, `scale_pos_weight` = neg/pos in training fold |
| CatBoost | 400 iterations, depth 6, lr 0.05, balanced class weights |
| kNN label interpolation | inverse-distance-weighted mean label of 7 nearest training springs |
| MLP / GCN / GraphSAGE | 2 layers, 64 hidden, dropout 0.25 |
| GATv2 | 2 layers, 64 hidden, 2 heads, edge attributes, dropout 0.25 |
| ResGATv2 | 1 layer, 64 hidden, 2 heads, ego skip, no normalisation, dropout 0.15 |
| HydroBoost-GAT | CatBoost (out-of-fold on training springs, inner 5-fold spatial groups) + ResGATv2; blend weight *w* ∈ {0, 0.2, …, 1}, diffusion α ∈ {0.2, 0.4, 0.6}, gain β ∈ {0, 0.5, 1}, 3 diffusion steps; selected on validation ROC-AUC |

---

### Text S4. Ablations and negative controls

All controls are run in the same folds and seeds as the main benchmark (`src/benchmark.py`):

| Control | What changes | Question answered |
|:---|:---|:---|
| ResGATv2 − edge attributes | GATv2 attention without **e**<sub>ij</sub> | Do physical edge attributes help? |
| ResGATv2 − ego skip | no W<sub>ego</sub>x<sub>i</sub> term | Does the skip connection help? |
| ResGATv2 + LayerNorm | LayerNorm after each layer | Is omitting normalisation justified? |
| ResGATv2 (2 layers) | two message-passing layers | Is a 1-hop receptive field preferable? |
| Placebo: shuffled edge attributes | rows of the edge-attribute matrix permuted | Is the *specific* geometry used? |
| Placebo: random topology | each spring linked to 7 random springs district-wide (attributes recomputed from true geometry) | Does local neighbourhood structure matter? |
| MLP vs GNNs | same training loop, no message passing | Does message passing add anything? |
| HydroBoost: CatBoost + diffusion only | *w* fixed to 1 | Contribution of the GAT stage |
| HydroBoost: no diffusion | β fixed to 0 | Contribution of residual diffusion |

The attention analysis (Table S7) compares attention statistics of the trained ResGATv2 with those
of the shuffled-edge placebo, using only test receivers and geometry-defined (never label-defined)
edge groups. Attention weights are not explanations in general (Jain & Wallace, 2019); we use them
only to ask whether the model's use of geometry differs from chance.

---

### Text S5. Spring Priority Index

For active springs with household data (household fields are skipped for dried sources):
SPI = pct(susceptibility) × pct(households dependent), where susceptibility is the out-of-fold
predicted probability of the best model on the spatial-CV (0 m) setting, averaged over seeds, and
pct is the percentile rank among ranked springs. Equal weighting is a transparent default; Table S11
reports how strongly the ranking depends on each component. "Susceptibility" means environmental
similarity to springs that have dried — given the modest discrimination of all models, it should be
used to *triage field verification*, not as a forecast.

---

### Supplementary figures

| Figure | File | Content |
|:---|:---|:---|
| S1 | `results/figures/shap_summary_beeswarm.png` | TreeSHAP for CatBoost on the primary feature set |
| S2 | `results/figures/spatial_mining_diagnostics.png` | Moran's I distance decay; FDR-significant association rules |
| S3 | `results/figures/attention_analysis.png` | ResGATv2 attention on test springs vs edge length; within-receiver statistics vs shuffled-edge placebo |
| S4 | `results/figures/shap_feature_importance_bar.png` | Mean absolute SHAP values |

Tables S13–S15 (timing and reported causes of drying; cause-specific discrimination) are produced by
`src/descriptive.py`. The reported drying year and reason are dried-only fields and are used descriptively,
never as predictors.

---

### Text S6. Changes from the previous version of this analysis

For transparency to reviewers and co-authors, the following elements of an earlier draft were
withdrawn: (i) all performance figures (ROC-AUC ≈ 0.99) — driven by the earthquake skip-logic
feature; (ii) a hard-coded "progressive ablation" table not produced by any experiment; (iii)
placebo/"Frog Trap" results for which no code existed and whose numbers did not support the stated
conclusion; (iv) a per-municipality LOMO table listing municipalities absent from the data; (v) a
hyperparameter "search" that was never run; (vi) a label-defined attention-regime test trained on
all labels; (vii) an HSVI table with no generating code that scored dried springs on fields that are
missing for them; (viii) two references that could not be verified.
