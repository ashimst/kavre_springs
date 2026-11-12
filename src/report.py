"""
Turn raw per-fold results into every summary table, statistical test, figure
and the auto-generated Markdown tables (results/tables/si_tables.md).
Nothing in this file contains a result value; everything is read from
results/metrics/benchmark_*_folds.csv and the other metric files.

Statistics
----------
Model comparisons are paired over (seed, fold). Because repeated CV folds share
training data, the naive paired t-test is anti-conservative; we use the
corrected resampled t-test (Nadeau & Bengio, 2003; Bouckaert & Frank, 2004):

    t = mean(d) / sqrt( (1/J + n_test/n_train) * var(d) ),  df = J - 1,

with Holm correction across the models compared within one setting. The
reference is the best graph-free tabular model in that setting (chosen from the
results). 95 % CIs on means use the same corrected variance.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import os

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import stats

from src.features import FEATURE_SETS, PRIMARY_FEATURE_SET, missingness_audit, single_feature_rule_scores
from src.dataset import load_master

logger = logging.getLogger("report")
METRICS_DIR = PROJECT_ROOT / "results/metrics"
TABLE_DIR = PROJECT_ROOT / "results/tables"
FIG_DIR = PROJECT_ROOT / "results/figures"
SI_PATH = PROJECT_ROOT / "results/tables/si_tables.md"
SI_TEX = PROJECT_ROOT / "paper/si_tables.tex"

METRICS = ["roc_auc", "pr_auc", "f1_macro", "bal_acc", "brier"]
TABULAR = ["Logistic Regression", "Random Forest", "CatBoost"]
CORE_ORDER = ["Logistic Regression", "Random Forest", "CatBoost", "kNN label interpolation"]
GROUP_COLS = ["profile", "scheme", "feature_set", "buffer_m"]


def load_folds() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(METRICS_DIR.glob("benchmark_*_folds.csv")) if "_quick" not in p.name]
    return pd.concat(frames, ignore_index=True)


def corrected_t(d: np.ndarray, ratio: float):
    J = len(d)
    var = d.var(ddof=1)
    se = np.sqrt((1 / J + ratio) * var) if var > 0 else np.nan
    t = d.mean() / se if se and se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), J - 1) if np.isfinite(t) else np.nan
    half = stats.t.ppf(0.975, J - 1) * se if np.isfinite(se) else np.nan
    return t, p, half


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1)
    return out


def summarise(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, grp in folds.groupby(GROUP_COLS + ["model"]):
        ratio = (grp.n_test / grp.n_train).mean()
        r = dict(zip(GROUP_COLS + ["model"], key))
        r["n_folds"] = len(grp)
        for m in METRICS:
            v = grp[m].to_numpy()
            r[f"{m}_mean"] = v.mean()
            r[f"{m}_sd"] = v.std(ddof=1)
            r[f"{m}_ci95"] = stats.t.ppf(0.975, len(v) - 1) * np.sqrt((1 / len(v) + ratio) * v.var(ddof=1))
        rows.append(r)
    return pd.DataFrame(rows)


def paired_tests(folds: pd.DataFrame, reference: str | None = None, models=None, metric="roc_auc") -> pd.DataFrame:
    rows = []
    for key, grp in folds.groupby(GROUP_COLS):
        piv = grp.pivot_table(index=["seed", "fold"], columns="model", values=metric)
        ratio = (grp.n_test / grp.n_train).mean()
        ref = reference
        if ref is None:
            tab = [m for m in TABULAR if m in piv.columns]
            ref = piv[tab].mean().idxmax()
        if ref not in piv.columns:
            continue
        cand = [m for m in (models or piv.columns) if m in piv.columns and m != ref]
        sub = []
        for m in cand:
            d = (piv[m] - piv[ref]).dropna().to_numpy()
            t, p, half = corrected_t(d, ratio)
            sub.append({**dict(zip(GROUP_COLS, key)), "metric": metric, "reference": ref, "model": m,
                        "mean_diff": d.mean(), "ci95_low": d.mean() - half, "ci95_high": d.mean() + half,
                        "wins": int((d > 0).sum()), "n_pairs": len(d), "t_corrected": t, "p_corrected": p})
        if sub:
            s = pd.DataFrame(sub)
            s["p_holm"] = holm(s.p_corrected.fillna(1).to_numpy())
            rows.append(s)
    return pd.concat(rows, ignore_index=True)


def fmt(m, s):
    return f"{m:.3f} ± {s:.3f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join([":---"] + ["---:"] * (len(cols) - 1)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "\\|") for c in cols) + " |")
    return "\n".join(lines)


def benchmark_table(summary: pd.DataFrame, profile: str, buffer_m: float, feature_set=PRIMARY_FEATURE_SET, models=None):
    s = summary[(summary.profile == profile) & (summary.buffer_m == buffer_m) & (summary.feature_set == feature_set)]
    models = [m for m in (models or CORE_ORDER) if m in set(s.model)]
    s = s.set_index("model").loc[models]
    return pd.DataFrame({
        "Model": models,
        "ROC-AUC": [fmt(r.roc_auc_mean, r.roc_auc_sd) for _, r in s.iterrows()],
        "PR-AUC": [fmt(r.pr_auc_mean, r.pr_auc_sd) for _, r in s.iterrows()],
        "Macro-F1": [fmt(r.f1_macro_mean, r.f1_macro_sd) for _, r in s.iterrows()],
        "Bal. acc.": [fmt(r.bal_acc_mean, r.bal_acc_sd) for _, r in s.iterrows()],
        "Brier": [f"{r.brier_mean:.3f}" for _, r in s.iterrows()],
    })


def tests_table(tests: pd.DataFrame, profile, buffer_m, feature_set=PRIMARY_FEATURE_SET, models=None):
    t = tests[(tests.profile == profile) & (tests.buffer_m == buffer_m) & (tests.feature_set == feature_set)]
    if models:
        t = t[t.model.isin(models)].set_index("model").loc[[m for m in models if m in set(t.model)]].reset_index()
    return pd.DataFrame({
        "Model": t.model, "Reference": t.reference,
        "ΔROC-AUC [95% CI]": [f"{r.mean_diff:+.3f} [{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]" for _, r in t.iterrows()],
        "Wins": [f"{r.wins}/{r.n_pairs}" for _, r in t.iterrows()],
        "p (corrected)": [f"{r.p_corrected:.3g}" for _, r in t.iterrows()],
        "p (Holm)": [f"{r.p_holm:.3g}" for _, r in t.iterrows()],
    })


def lomo_table(folds: pd.DataFrame, df: pd.DataFrame):
    l = folds[(folds.profile == "lomo") & (folds.feature_set == PRIMARY_FEATURE_SET)]
    if l.empty:
        return None
    piv = l.groupby(["fold", "model"]).roc_auc.mean().unstack()
    piv = piv[[m for m in CORE_ORDER if m in piv.columns]]
    info = df.groupby("municipality").agg(N=("target_dried", "size"), dried=("target_dried", "sum"))
    info["Drying rate"] = (info.dried / info.N).map(lambda v: f"{100 * v:.1f}%")
    out = info.join(piv.round(3)).reset_index().rename(columns={"municipality": "Held-out municipality"})
    out.to_csv(TABLE_DIR / "lomo_per_municipality_auc.csv", index=False)
    return out


def current_runs(folds: pd.DataFrame) -> pd.DataFrame:
    """Keep only (profile, feature set, model) combinations that the current benchmark profiles produce, so that
    rows from earlier, since-removed experiments in the checkpointed CSVs cannot leak into tables."""
    from src.benchmark import PROFILES
    keep = np.zeros(len(folds), bool)
    for name, prof in PROFILES.items():
        keep |= ((folds.profile == name) & folds.feature_set.isin(prof["feature_sets"])
                 & folds.model.isin(prof["models"])).to_numpy()
    return folds[keep].reset_index(drop=True)


# Supplementary tables, in the order in which the manuscript first cites them; numbered S1, S2, ... by run_report.
# Each entry is (key, title). Keys are either built in run_report ("leakage", "single", "featuresets", "paired")
# or name a file in results/metrics via SI_FILES. Tables that only repeat a main-text table, a figure or a number
# stated in the text are not reprinted; every result table remains in results/metrics.
SI_ORDER = [
    ("boundary_agreement.csv", "Recorded palika vs. official palika polygon containing each source"),
    ("x0_raw_field_audit.csv", "Skip-logic audit of the 56 non-outcome raw survey fields: missingness, gap sensitivity and missingness-only AUC"),
    ("single", "ROC-AUC of single features used alone (no model)"),
    ("featuresets", "Feature-set sensitivity: the enrichment ladder (spatial CV, no buffer)"),
    ("x1_tier_increments.csv", "Fold-matched ROC-AUC gains between feature tiers (corrected resampled t-test, Holm within design)"),
    ("rb_validation_distances.csv", "Distance from test sources to the nearest training source by validation design"),
    ("paired", "Paired comparisons against the best tabular model (ROC-AUC; corrected resampled t-test, Holm)"),
    ("x2_within_ward_auc.csv", "Pooled, within-ward and ward-mean ROC-AUC of out-of-fold predictions (mean over 5 seeds)"),
    ("x11_fold_variation.csv", "Spread of fold ROC-AUC by design, tier and model (SD over seed x fold; SD of seed means)"),
    ("x11_lomo_by_municipality.csv", "Leave-one-municipality-out ROC-AUC of each held-out municipality (mean and SD over 3 seeds)"),
    ("x11_springs_only_auc.csv", "Pooled and within-ward ROC-AUC of out-of-fold predictions, all sources vs springs only"),
    ("ca_unit_icc.csv", "ICC of drying: DEM catchments, wards, municipalities vs size-matched grids"),
    ("un_regionalisation.csv", "Label-free landscape regions vs wards: landscape and drying homogeneity"),
    ("x10_partition_diagnostics.csv", "Placebo partitions: draws, seeds, connectivity and size match against the ward sizes"),
    ("x10_strengthened_nulls.csv", "Ward and seasonal-status ICCs against sampled spatial partitions; nominal tail probabilities"),
    ("x4_residual_ward_icc.csv", "Ward ICC of recorded drying and of covariate-model residuals against two spatial nulls"),
    ("ca_pair_concordance_logit.csv", "Pair concordance ~ same catchment + same ward + log distance (spatial-block bootstrap CI)"),
    ("x3_pair_placebo.csv", "Same-unit pair term for wards vs placebo units (matched grids, contiguous random regions)"),
    ("x10_pair_placebo.csv", "Same-ward pair term against 200 contiguous size-matched and 200 municipality-nested placebo partitions"),
    ("sa_directional_test.csv", "Upslope vs downslope neighbour dependence (permutation test)"),
    ("ca_flowpath_logit.csv", "Flow-path covariates with and without environmental adjustment"),
    ("x5_placebo_partition.csv", "Variation partitioning with ward indicators vs placebo-unit indicators (adjusted McFadden pseudo-R2)"),
    ("x8_ward_correlates.csv", "Ward-level correlates of the ward funnel z-score (Spearman, BH-FDR)"),
    ("sa_knox_test.csv", "Knox space-time test of reported drying year, within vs across wards"),
    ("un_typology_selection.csv", "Gaussian-mixture model selection: BIC and bootstrap stability"),
    ("un_typology_profiles.csv", "Mean standardised covariates (and survey composition) by terrain type"),
    ("un_typology_vs_ward_lrt.csv", "Terrain type vs ward: likelihood-ratio tests"),
    ("pm_drying_rules.csv", "Top 40 significant productive apriori rules per consequent (BH-FDR q < 0.05), ranked by WRAcc"),
    ("x6_reversals_fdr.csv", "Rule counts, ward robustness, held-out replication and Simpson-type reversals with and without FDR control"),
    ("dr_pooled_vs_within_ward.csv", "Covariates of drying: pooled vs ward fixed effects (per SD, 95 % block-bootstrap CI)"),
    ("x9_grid_fe_survival.csv", "Covariate coefficients (per SD): pooled, ward fixed effects and matched-grid fixed effects (10 shifts)"),
    ("x7_cause_auc_difference.csv", "Out-of-fold ROC-AUC for earthquake-attributed vs other dried sources (bootstrap CI of the difference)"),
    ("sa_seismic_quintiles.csv", "Drying by quintile of 2015 shaking and aftershock exposure"),
    ("sa_seismic_logit.csv", "Seismic exposure models, including attribution and dating among dried sources"),
    ("rb_seismic_adjusted.csv", "Earthquake exposure with ward fixed effects or Moran-eigenvector adjustment"),
    ("rb_seismic_within_ward_variance.csv", "Share of earthquake-exposure variance within wards"),
    # appended, so that the numbers of the tables above stay as cited in the manuscript
    ("x12_pair_contrast.csv", "Pair model: same-ward minus same-catchment log-odds (paired spatial-block bootstrap, 300 draws)"),
    ("x13_matched_count_icc.csv", "ICC of drying for wards, 2 km2 catchments and count-matched catchments against 1,000 rotated grids"),
    ("x13_matched_count_contrast.csv", "Ward minus catchment contrasts at 2 km2 and at matched unit count (block jackknife for ICC, block bootstrap for the pair term)"),
    ("x14_pair_robustness.csv", "Pair model for all sources and springs only: block-bootstrap and dyadic-robust intervals, leave-one-block-out range"),
    ("x15_joint_model_collinearity.csv", "Covariates with a within-ward signal: joint vs single-covariate coefficients, strongest correlate and VIF"),
    ("x16_partition_geometry.csv", "Spatial extent diagnostics for wards and the three reference partition families"),
    ("x17_top_decile_triage.csv", "PR-AUC and out-of-fold top-decile precision, recall and lift by model and feature tier"),
    ("x18_terrain_type_block_permutation.csv", "Terrain-type and recorded-drying association tested by within-block outcome permutation"),
    ("x19_within_unit_auc.csv", "Within-known-unit out-of-fold AUC in wards and DEM catchments at three thresholds"),
    ("x20_coordinate_ablation.csv", "Paired spatial-CV model performance after dropping coordinates and source density"),
    ("x21_ward_null_calibration.csv", "Simulated rejection rates of partition-based ward-ICC tests under spatial fields with and without ward effects"),
    ("x22_reduced_covariates.csv", "Post hoc reduced full-rank continuous-covariate models, pooled and with ward indicators"),
    ("x23_feature_family_ablation.csv", "Drop-one-family ablation of the enriched model on identical spatial folds"),
    ("x24_within_ward_auc_block_bootstrap.csv", "Within-ward AUC with spatial-block bootstrap intervals"),
    ("x25_ward_reported_year.csv", "Reported-year distribution by ward among dried sources with numeric year-like values"),
    ("x25_recent_year_nulls.csv", "Ward ICC of reported values coded 2071-2085 among valid-year dried sources against spatial reference partitions"),
    ("x25_worldcover_by_ward.csv", "Ward-level disagreement between recorded land use and dominant 2021 WorldCover fractions"),
    ("x25_spatially_isolated_sources.csv", "Assigned-ward vs nearest-source majority agreement for spatially isolated records"),
    ("x25_ascertainment_checks.csv", "Descriptive checks of serial order, reported years, cover disagreement and spatially isolated sources"),
    ("ward_boundary_agreement.csv", "Recorded ward vs. OpenStreetMap ward polygon containing each source"),
    ("x26_ward_polygon_icc.csv", "Ward ICC with wards assigned from OpenStreetMap polygons and from recorded labels, against 1,000 rotated grids and 1,000 contiguous size-matched regions"),
    ("x26_ward_connectivity.csv", "Connectivity of recorded and polygon wards on the 10-nearest-neighbour source graph"),
    ("x27_full_rank_residual_icc.csv", "Residual ward ICC after the 65-covariate set and after the 62-covariate full-rank set, against the three 1,000-draw reference families"),
    ("x27_full_rank_earthquake.csv", "Earthquake-exposure coefficients with the 65-covariate set and with the 62-covariate full-rank set"),
    ("x28_latent_icc.csv", "Observed-scale (ANOVA) and latent-scale (random-intercept logit) ICC against the same reference partitions"),
    ("x28_latent_icc_contrasts.csv", "Ward minus catchment ICC differences on the observed and latent scales (delete-one-block jackknife)"),
]
SI_COLUMNS = {
    "x10_strengthened_nulls.csv": ["subset", "outcome", "null", "obs", "null_q975", "null_max", "z", "n_exceed",
                                   "p_one_sided"],
    "x1_tier_increments.csv": ["design", "model", "from", "to", "auc_from", "auc_to", "mean_diff", "ci95_low",
                               "ci95_high", "p_holm"],
    "x0_raw_field_audit.csv": ["field", "missing_active", "missing_dried", "gap", "outcome_dependent",
                               "direction", "missingness_only_auc", "gap_gt_1pp", "gap_gt_5pp", "gap_gt_10pp"],
    "x16_partition_geometry.csv": ["partition", "n_draws", "n_units_mean", "n_units_min", "n_units_max",
                                   "share_units_disconnected_mean", "mean_within_unit_pair_distance_m_mean",
                                   "mean_within_unit_pair_distance_m_min", "mean_within_unit_pair_distance_m_max",
                                   "median_unit_median_pair_distance_m_mean",
                                   "mean_within_unit_nearest_neighbor_m_mean",
                                   "share_100_400m_pairs_same_unit_mean"],
    "x17_top_decile_triage.csv": ["model", "feature_set", "buffer_m", "n_test_folds", "roc_auc_mean", "pr_auc_mean",
                                  "prevalence_mean",
                                  "precision_at_10pct_mean", "precision_at_10pct_std", "recall_at_10pct_mean",
                                  "recall_at_10pct_std", "lift_at_10pct_mean", "lift_at_10pct_std"],
    "x18_terrain_type_block_permutation.csv": ["observed_chi2", "naive_chi2_p", "block_permutation_p",
                                               "n_permutations", "block_size_km", "null_q95", "null_max"],
    "x19_within_unit_auc.csv": ["model", "unit", "n_seeds", "pooled_auc", "within_unit_auc",
                                "within_unit_auc_sd", "eligible_units", "within_unit_pairs",
                                "share_case_control_pairs_within_units"],
    "x20_coordinate_ablation.csv": ["model", "variant", "n_folds", "roc_auc_mean", "delta_roc_auc_mean",
                                    "delta_roc_auc_sd", "pr_auc_mean", "delta_pr_auc_mean"],
    "x21_ward_null_calibration.csv": ["spatial_length_m", "ward_effect_sd_logit", "null_family", "n_simulations",
                                      "n_null_partitions", "rejection_rate", "rejection_ci_low", "rejection_ci_high",
                                      "median_ward_icc", "mean_null_q95"],
    "x22_reduced_covariates.csv": ["model", "term", "coef_per_sd", "ci_low", "ci_high", "ci_excludes_zero",
                                   "vif", "n_covariates", "design_rank"],
    "x23_feature_family_ablation.csv": ["model", "dropped_family", "n_folds", "delta_roc_auc_mean", "ci95_low",
                                        "ci95_high", "p_uncorrected", "p_holm_within_model"],
    "x24_within_ward_auc_block_bootstrap.csv": ["model", "within_ward_auc", "ci_low", "ci_high", "n_blocks",
                                                "n_bootstrap", "bootstrap_unit"],
    "x25_ward_reported_year.csv": ["ward", "n_valid_reported_years", "mean_reported_value",
                                   "median_reported_value", "q25_reported_value", "q75_reported_value",
                                   "share_coded_2071_or_later", "share_coded_2072"],
    "x25_recent_year_nulls.csv": ["outcome", "family", "n_sources", "observed_ward_icc", "null_q025",
                                  "null_q975", "null_max", "p_nominal", "n_null"],
    "x25_worldcover_by_ward.csv": ["ward", "n_comparable_sources",
                                   "share_recorded_cover_not_largest_worldcover_class", "recorded_drying_rate"],
    "x25_spatially_isolated_sources.csv": ["subset", "n_records", "n_non_tied_comparisons",
                                           "share_matches_recorded_ward_majority",
                                           "share_matches_10_nearest_source_majority",
                                           "ward_minus_local_match", "block_boot_ci_low", "block_boot_ci_high",
                                           "bootstrap_blocks"],
    "x25_ascertainment_checks.csv": ["check", "n", "estimate", "detail"],
    "x27_full_rank_residual_icc.csv": ["covariate_set", "n_covariates", "design_rank", "null", "obs", "null_q975",
                                       "null_max", "n_exceed", "p_one_sided"],
    "x27_full_rank_earthquake.csv": ["covariate_set", "model", "term", "coef_per_sd", "ci_low", "ci_high",
                                     "ci_excludes_zero"],
    "x28_latent_icc.csv": ["block", "subset", "outcome", "unit", "null", "observed_obs", "observed_null_max",
                           "observed_p_one_sided", "latent_obs", "latent_null_q975", "latent_null_max",
                           "latent_n_exceed", "latent_p_one_sided"],
    "x28_latent_icc_contrasts.csv": ["contrast", "scale", "difference", "se", "ci_low", "ci_high", "p_one_sided",
                                     "n_blocks"],
    "x26_ward_polygon_icc.csv": ["unit", "sample", "n_sources", "n_units", "outcome", "null", "obs", "null_q975",
                                 "null_max", "n_exceed", "p_one_sided"],
    "x13_matched_count_icc.csv": ["unit", "n_units", "obs", "null_mean", "null_q975", "null_max", "z", "n_null",
                                  "n_exceed", "p_one_sided"],
    "x14_pair_robustness.csv": ["population", "term", "n_pairs", "log_odds", "boot_ci_low", "boot_ci_high",
                                "boot_p_one_sided", "boot_skewness", "se_dyadic", "dyadic_ci_low", "dyadic_ci_high",
                                "dyadic_p_two_sided", "loo_block_min", "loo_block_max"],
    "pm_drying_rules.csv": ["consequent", "antecedent", "n", "support", "confidence", "lift", "wracc", "q_bh",
                            "mh_or_ward", "mh_ci_low", "mh_ci_high", "share_wards_lift_gt1", "heldout_replication",
                            "reverses_within_wards"],
}
SI_NUMBERS = {key: k + 1 for k, (key, _) in enumerate(SI_ORDER)}

# Former SI list, kept for reference to the metric files (no longer printed in the supplement)
_SI_ALL = [
    # landscape typology
    ("un_pca_variance.csv", "Principal components of the static landscape covariates", None),
    ("un_typology_selection.csv", "Gaussian-mixture model selection: BIC and bootstrap stability", None),
    ("un_typology_drying.csv", "Drying rate by landscape type (95 % Wilson CI)", None),
    ("un_typology_profiles.csv", "Mean standardised covariates (and survey composition) by landscape type", None),
    ("un_typology_vs_ward_lrt.csv", "Landscape type vs ward: likelihood-ratio tests", None),
    ("un_typology_summary.csv", "Association strengths (bias-corrected Cramér's V)", None),
    # spatial structure
    ("spatial_autocorrelation_distance_decay.csv", "Global Moran's I distance decay", None),
    # wards vs watersheds
    ("ca_unit_icc.csv", "ICC of drying: DEM catchments, wards, municipalities vs size-matched grids", None),
    ("un_regionalisation.csv", "Label-free landscape regions vs wards: landscape and drying homogeneity", None),
    ("sa_ward_concordance.csv", "Status concordance of spring pairs by distance band, within vs across wards", None),
    ("ca_pair_concordance_logit.csv", "Pair concordance ~ same catchment + same ward + log distance", None),
    ("sa_directional_test.csv", "Upslope vs downslope neighbour dependence (permutation test)", None),
    ("ca_flowpath_logit.csv", "Flow-path covariates with and without environmental adjustment", None),
    ("dr_three_way_partition.csv", "Three-way variation partitioning: environment, space, ward", None),
    ("un_ward_funnel_summary.csv", "Ward funnel plot against a landscape-only model: overdispersion", None),
    ("un_ward_funnel.csv", "Observed and landscape-expected dried sources per ward", None),
    # apriori pattern mining
    ("pm_drying_rules_summary.csv", "Apriori drying rules: counts, ward robustness, replication", None),
    ("pm_seasonal_flag.csv", "The survey seasonal-flow flag (analysed separately)", None),
    ("pm_drying_rules.csv", "Significant productive apriori rules (BH-FDR q < 0.05), ranked by WRAcc",
     ["consequent", "antecedent", "n", "support", "confidence", "lift", "wracc", "q_bh", "mh_or_ward", "mh_ci_low",
      "mh_ci_high", "share_wards_lift_gt1", "heldout_replication", "reverses_within_wards"]),
    # recalled causes and years
    ("drying_reasons.csv", "Reported reason of drying", None),
    ("drying_timing_summary.csv", "Reported timing of drying", None),
    ("sa_knox_test.csv", "Knox space-time test of reported drying year, within vs across wards", None),
    ("un_dried_icc.csv", "Ward and landscape-type structure of reported causes and years among dried sources", None),
    ("un_type_by_reason.csv", "Reported cause by landscape type (row shares)", None),
    # drivers and earthquake
    ("dr_pooled_vs_within_ward.csv", "Drivers of drying: pooled vs ward fixed effects (per SD, 95 % block-bootstrap CI)", None),
    ("sa_seismic_quintiles.csv", "Drying by quintile of 2015 shaking and aftershock exposure", None),
    ("sa_seismic_logit.csv", "Seismic exposure models, including attribution and dating among dried sources", None),
    ("cause_specific_auc.csv", "Pooled out-of-fold ROC-AUC by reported cause (spatial CV, no buffer)", None),
    # robustness
    ("rb_sensitivity_subsets.csv", "Sensitivity of the ward analyses to the population (ponds, seasonal sources, location)", None),
    ("rb_validation_distances.csv", "Distance from test sources to the nearest training source by validation design", None),
    ("rb_seismic_within_ward_variance.csv", "Share of earthquake-exposure variance within wards", None),
    ("rb_seismic_adjusted.csv", "Earthquake exposure with ward fixed effects or Moran-eigenvector adjustment", None),
    # data quality
    ("boundary_agreement.csv", "Recorded palika vs. official palika polygon containing each source", None),
]


def run_report():
    for d in (TABLE_DIR, FIG_DIR, SI_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)
    df = load_master()
    folds = current_runs(load_folds())
    summary = summarise(folds)
    summary.to_csv(METRICS_DIR / "benchmark_summary.csv", index=False)
    core_tests = pd.concat([paired_tests(folds, None, CORE_ORDER, m) for m in ["roc_auc", "pr_auc"]])
    core_tests.to_csv(METRICS_DIR / "paired_tests_vs_best_tabular.csv", index=False)

    audit = missingness_audit(df)
    audit.to_csv(METRICS_DIR / "leakage_audit.csv", index=False)
    rules = single_feature_rule_scores(df)
    rules.to_csv(METRICS_DIR / "single_feature_auc.csv", index=False)

    lomo_table(folds, df)   # writes results/tables/lomo_per_municipality_auc.csv

    # ---------------- supplementary tables for the analysis record ----------------
    built = {}  # key -> (note, DataFrame) for the tables assembled here rather than read from a metric file
    built["leakage"] = ("Missingness of survey fields among active vs dried sources (fields with |gap| > 0.05).",
                        audit[audit.skip_logic_leak][["column", "missing_active", "missing_dried",
                                                      "missingness_gap"]].round(3))
    built["single"] = ("", rules.round(3))
    fs_rows = []
    for fs in ["env", "env_hydro", "env_rich", "env_hydro_season", "leaky_reference"]:
        s = summary[(summary.feature_set == fs) & (summary.buffer_m == 0)]
        for _, r in s.iterrows():
            fs_rows.append({"Feature set": fs, "n features": len(FEATURE_SETS.get(fs, [])), "Model": r.model,
                            "ROC-AUC": fmt(r.roc_auc_mean, r.roc_auc_sd), "PR-AUC": fmt(r.pr_auc_mean, r.pr_auc_sd)})
    built["featuresets"] = ("Mean ± SD over 5 seeds x 5 spatial folds.", pd.DataFrame(fs_rows))
    roc_tests = core_tests[core_tests.metric == "roc_auc"]
    paired = [tests_table(roc_tests, prof, buf, models=CORE_ORDER).assign(Design=design)
              for prof, buf, design in [("main", 0.0, "spatial CV, no buffer"), ("main", 500.0, "spatial CV, 500 m buffer"),
                                        ("lomo", 500.0, "unseen municipality (LOMO)")]
              if not roc_tests[(roc_tests.profile == prof) & (roc_tests.buffer_m == buf)].empty]
    if paired:
        p = pd.concat(paired, ignore_index=True)
        built["paired"] = ("env_hydro covariates. Positive differences favour the model over the reference.",
                           p[["Design"] + [c for c in p.columns if c != "Design"]])

    tabs = []  # (title, note, DataFrame)
    for key, title in SI_ORDER:
        label = f"Table S{SI_NUMBERS[key]}. {title}"
        if key in built:
            tabs.append((label, *built[key]))
            continue
        path = METRICS_DIR / key
        if not (path.exists() and path.stat().st_size > 10):
            logger.warning(f"SI table source missing: {key}")
            continue
        t = pd.read_csv(path)
        if key in SI_COLUMNS:
            t = t[[c for c in SI_COLUMNS[key] if c in t.columns]]
        if key == "pm_drying_rules.csv":
            t = pd.concat([t[t.consequent == c].head(40) for c in ["dried", "active"]])
        tabs.append((label, "", t.round(4)))

    parts = ["# Auto-generated SI tables", "",
             "_Generated by `python -m src.report` from `results/metrics/`. Do not edit by hand; re-run the pipeline. "
             "Benchmark values are mean ± SD over seeds × folds; tests use the corrected resampled t-test with Holm "
             "adjustment. Every other result table is in `results/metrics/`._", ""]
    for title, note, t in tabs:
        parts += [f"## {title}", ""] + ([note, ""] if note else []) + [md_table(t), ""]
    SI_PATH.write_text("\n".join(parts), encoding="utf-8")
    logger.info(f"Wrote {SI_PATH} ({len(tabs)} tables)")
    if SI_TEX.parent.exists():   # LaTeX version for the supplement (paper/ is kept outside version control)
        tex = ["% Auto-generated by `python -m src.report` from results/metrics/. Do not edit by hand.",
               r"\section*{Supplementary tables}"]
        tex += [latex_table(title, note, t) for title, note, t in tabs]
        SI_TEX.write_text("\n".join(tex), encoding="utf-8")
        logger.info(f"Wrote {SI_TEX} ({len(tabs)} tables)")
    return summary, core_tests


_TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
            "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "±": r"$\pm$", "→": r"$\rightarrow$",
            "≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$", "²": r"$^2$", "é": r"\'e", "∩": r"$\cap$", "Δ": r"$\Delta$", "φ": r"$\phi$", "ρ": r"$\rho$", "|": r"$|$",
            "−": "-", "–": "--", "—": "---", "\u00a0": " "}


def tex_escape(v) -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return "--"
        if np.isinf(v):
            return "inf" if v > 0 else "-inf"
        if v == int(v) or abs(v) >= 100:
            v = f"{v:.0f}"
        else:
            v = f"{v:.3g}"
    s = str(v)
    return "".join(_TEX_ESC.get(ch, ch if ord(ch) < 128 else "?") for ch in s)


def _stack(text: str, width: int) -> str:
    """Break a header or long text cell into short lines inside a one-column tabular."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    if len(lines) == 1:
        return lines[0]
    # "{}" after each break stops a following "[" from being read as the break's optional argument
    return "\\begin{tabular}[t]{@{}l@{}}" + " \\\\{} ".join(lines) + "\\end{tabular}"


def _header(c) -> str:
    return "\\textbf{" + _stack(tex_escape(str(c).replace("_", " ")), 12) + "}"


def _cell(v) -> str:
    s = tex_escape(v)
    return _stack(s, 22) if isinstance(v, str) and len(s) > 24 else s


TRANSPOSE_MIN_COLUMNS = 8   # short tables wider than this are transposed
TRANSPOSE_MAX_ROWS = 8


def latex_table(title, note, t: pd.DataFrame, max_rows=40) -> str:
    """One supplementary table as a non-floating block, so tables follow each other on the page.

    Short, wide tables (more than TRANSPOSE_MIN_COLUMNS columns and at most TRANSPOSE_MAX_ROWS rows) are transposed
    so that their many metrics become rows; long tables are split into chunks of max_rows rows."""
    t = t.reset_index(drop=True) if t.index.name is None else t.reset_index()
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if len(t.columns) > TRANSPOSE_MIN_COLUMNS and 1 <= len(t) <= TRANSPOSE_MAX_ROWS:
        first = t.columns[0]
        labels = [str(v) for v in t[first]]
        if len(set(labels)) < len(labels):          # keep columns distinct when the first column repeats
            labels = [f"{v} ({i + 1})" for i, v in enumerate(labels)]
        t = t.drop(columns=first).T
        t.columns = labels
        t = t.reset_index().rename(columns={"index": str(first).replace("_", " ")})
        t.iloc[:, 0] = [str(v).replace("_", " ") for v in t.iloc[:, 0]]
    head = " & ".join(_header(c) for c in t.columns)
    parts = []
    for start in range(0, max(len(t), 1), max_rows):
        chunk = t.iloc[start:start + max_rows]
        body = "\n".join(" & ".join(_cell(v) for v in r) + r" \\" for r in chunk.itertuples(index=False))
        caption = title if start == 0 else f"{title} (continued)"
        note_tex = (f"\\par\\smallskip{{\\scriptsize {tex_escape(note)}\\par}}"
                    if note and start + max_rows >= len(t) else "")
        parts.append(
            "\\begin{center}\\footnotesize\n"
            f"\\captionof*{{table}}{{\\textbf{{{tex_escape(caption)}}}}}\n"
            "\\begin{adjustbox}{max width=\\linewidth, max totalheight=0.9\\textheight}\n"
            f"\\begin{{tabular}}{{{'l' * len(t.columns)}}}\n\\toprule\n{head} \\\\\n\\midrule\n{body}\n\\bottomrule\n"
            f"\\end{{tabular}}\n\\end{{adjustbox}}{note_tex}\n\\end{{center}}\n\\medskip\n"
        )
    return "".join(parts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_report()
