"""
Turn raw per-fold results into every summary table, statistical test, figure
and the auto-generated SI tables (paper/si_generated_tables.md).
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.features import FEATURE_SETS, PRIMARY_FEATURE_SET, missingness_audit, single_feature_rule_scores
from src.graph_builder import load_master

logger = logging.getLogger("report")
METRICS_DIR = PROJECT_ROOT / "results/metrics"
TABLE_DIR = PROJECT_ROOT / "results/tables"
FIG_DIR = PROJECT_ROOT / "results/figures"
SI_PATH = PROJECT_ROOT / "paper/si_generated_tables.md"

METRICS = ["roc_auc", "pr_auc", "f1_macro", "bal_acc", "brier"]
TABULAR = ["Logistic Regression", "Random Forest", "XGBoost", "CatBoost"]
CORE_ORDER = ["Logistic Regression", "Random Forest", "XGBoost", "CatBoost", "kNN label interpolation", "MLP",
              "GCN", "GraphSAGE", "GATv2", "ResGATv2", "HydroBoost-GAT"]
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


def fig_benchmark(summary: pd.DataFrame):
    panels = [("main", 0.0, "Spatial 5-fold CV, no buffer\n(interpolation)"),
              ("main", 500.0, "Spatial 5-fold CV, 500 m buffer\n(extrapolation)"),
              ("lomo", 500.0, "Leave-one-municipality-out\n(500 m buffer)")]
    panels = [p for p in panels if ((summary.profile == p[0]) & (summary.buffer_m == p[1])).any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (prof, buf, title) in zip(axes, panels):
        cand_fs = ["env_hydro", PRIMARY_FEATURE_SET, "env"]
        fs = next((f for f in cand_fs if ((summary.profile == prof) & (summary.buffer_m == buf) & (summary.feature_set == f)).any()), "env")
        s = summary[(summary.profile == prof) & (summary.buffer_m == buf) & (summary.feature_set == fs)]
        models_to_plot = [m for m in CORE_ORDER if m in set(s.model)]
        if not models_to_plot:
            continue
        s = s.set_index("model").loc[models_to_plot]
        colors = ["#4C72B0" if m in TABULAR else "#DD8452" if m == "kNN label interpolation" else
                  "#8172B3" if m == "MLP" else "#55A868" for m in s.index]
        y = np.arange(len(s))[::-1]
        for yi, (_, r), c in zip(y, s.iterrows(), colors):
            ax.errorbar(r.roc_auc_mean, yi, xerr=r.roc_auc_ci95, fmt="o", color=c, capsize=3)
        ax.axvline(0.5, ls=":", c="grey")
        ax.set_yticks(y)
        ax.set_yticklabels(s.index, fontsize=9)
        ax.set_xlabel("ROC-AUC (mean, corrected 95 % CI)")
        ax.set_title(title, fontsize=10)
    fig.suptitle("Spring drying prediction from a-priori covariates (primary feature set)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_benchmark_comparison.png", dpi=300)
    plt.close(fig)


def fig_leakage(summary: pd.DataFrame, rule_auc: float):
    names = {
        "env": "Basic covariates (env)",
        "env_hydro": "+ catchment hydrology & roads (env_hydro)",
        "env_hydro_season": "+ seasonal-flow flag",
        "leaky_reference": "basic + skip-logic earthquake fields (leaky)",
    }
    order = [fs for fs in names if (summary.feature_set == fs).any()]
    s = summary[(summary.buffer_m == 0) & summary.feature_set.isin(order) & summary.profile.isin(["main", "features"])]
    models = [m for m in ["Logistic Regression", "CatBoost", "MLP", "ResGATv2", "HydroBoost-GAT"] if m in set(s.model)]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    w = 0.8 / len(order)
    for i, fs in enumerate(order):
        v = s[s.feature_set == fs].set_index("model").reindex(models)
        ax.bar(np.arange(len(models)) + (i - (len(order) - 1) / 2) * w, v.roc_auc_mean, w, yerr=v.roc_auc_ci95,
               capsize=2, label=names[fs])
    ax.axhline(rule_auc, ls="--", c="crimson", label=f"one-line rule: earthquake question unanswered (AUC {rule_auc:.3f})")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("ROC-AUC (spatial CV, no buffer)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "leakage_feature_sets.png", dpi=300)
    plt.close(fig)


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


# SI tables in the order in which the manuscript cites them. (file, title, optional column subset)
SI_TABLES = [
    # landscape typology and unsupervised structure
    ("un_pca_variance.csv", "Table S8. Principal components of the static landscape covariates", None),
    ("un_typology_selection.csv", "Table S9. Gaussian-mixture model selection: BIC and bootstrap stability", None),
    ("un_typology_drying.csv", "Table S10. Drying rate by landscape type (95 % Wilson CI)", None),
    ("un_typology_profiles.csv", "Table S11. Mean standardised covariates (and survey composition) by landscape type", None),
    ("un_typology_vs_ward_lrt.csv", "Table S12. Landscape type vs ward: likelihood-ratio tests", None),
    ("un_typology_summary.csv", "Table S13. Association strengths (bias-corrected Cramér's V)", None),
    ("un_hdbscan_clusters.csv", "Table S14. HDBSCAN density clusters of landscape space", None),
    ("un_hdbscan_summary.csv", "Table S15. HDBSCAN summary and agreement with landscape types", None),
    ("un_anomalies.csv", "Table S16. Isolation-forest landscape anomalies and drying", None),
    # spatial structure
    ("spatial_autocorrelation_distance_decay.csv", "Table S17. Global Moran's I distance decay", None),
    ("lisa_cluster_summary.csv", "Table S18. LISA clusters (k = 7; BH-FDR)", None),
    ("sa_variation_partitioning.csv", "Table S19. Environment vs Moran-eigenvector space (adjusted pseudo-R²)", None),
    ("sa_variation_by_scale.csv", "Table S20. Spatial fraction by MEM scale band", None),
    ("sa_scan_clusters.csv", "Table S21. Bernoulli spatial scan clusters (999 Monte Carlo replicates)",
     ["analysis", "rank", "radius_m", "n_sources", "cases", "observed_rate", "rate_outside", "relative_risk", "llr", "p_value",
      "center_lat", "center_lon"]),
    ("sa_scan_ward_alignment.csv", "Table S22. Ward concentration of significant scan clusters vs random windows", None),
    # wards vs watersheds
    ("ca_unit_icc.csv", "Table S23. ICC of drying: DEM catchments, wards, municipalities vs size-matched grids", None),
    ("un_regionalisation.csv", "Table S24. Label-free regionalisations vs wards: landscape and drying homogeneity", None),
    ("sa_ward_concordance.csv", "Table S25. Status concordance of spring pairs by distance band, within vs across wards", None),
    ("ca_pair_concordance_logit.csv", "Table S26. Pair concordance ~ same catchment + same ward + log distance", None),
    ("sa_directional_test.csv", "Table S27. Upslope vs downslope neighbour dependence (permutation test)", None),
    ("ca_flowpath_logit.csv", "Table S28. Flow-path covariates with and without environmental adjustment", None),
    ("dr_three_way_partition.csv", "Table S29. Three-way variation partitioning: environment, space, ward", None),
    ("un_ward_funnel_summary.csv", "Table S30. Ward funnel plot against a landscape-only model: overdispersion", None),
    ("un_ward_funnel.csv", "Table S31. Observed and landscape-expected dried sources per ward", None),
    # apriori pattern mining
    ("pm_drying_rules_summary.csv", "Table S32. Apriori drying rules: counts, ward robustness, replication", None),
    ("pm_seasonal_flag.csv", "Table S33. The survey seasonal-flow flag (analysed separately)", None),
    ("pm_drying_rules.csv", "Table S34. Significant productive apriori rules (BH-FDR q < 0.05), ranked by WRAcc",
     ["consequent", "antecedent", "n", "support", "confidence", "lift", "wracc", "q_bh", "mh_or_ward", "mh_ci_low",
      "mh_ci_high", "share_wards_lift_gt1", "heldout_replication", "reverses_within_wards"]),
    # recalled causes and years
    ("drying_reasons.csv", "Table S35. Reported reason of drying", None),
    ("drying_timing_summary.csv", "Table S36. Reported timing of drying", None),
    ("sa_knox_test.csv", "Table S37. Knox space-time test of reported drying year, within vs across wards", None),
    ("un_dried_icc.csv", "Table S38. Ward and landscape-type structure of reported causes and years among dried sources", None),
    ("pm_narrative_rules.csv", "Table S39. Apriori narrative rules among dried sources (ward-stratified)",
     ["consequent", "antecedent", "n", "support", "confidence", "lift", "q_bh", "mh_or_ward", "mh_ci_low", "mh_ci_high",
      "ward_robust"]),
    ("un_type_by_reason.csv", "Table S40. Reported cause by landscape type (row shares)", None),
    ("un_narrative_k_selection.csv", "Table S41. Ward narrative regimes: silhouette by number of clusters", None),
    ("un_narrative_regimes.csv", "Table S42. Ward narrative regimes: mean profiles", None),
    ("un_narrative_mantel.csv", "Table S43. Mantel tests: narrative similarity vs geography and landscape", None),
    # drivers and earthquake
    ("dr_pooled_vs_within_ward.csv", "Table S44. Drivers of drying: pooled vs ward fixed effects (per SD, 95 % block-bootstrap CI)", None),
    ("sa_seismic_quintiles.csv", "Table S45. Drying by quintile of 2015 shaking and aftershock exposure", None),
    ("sa_seismic_logit.csv", "Table S46. Seismic exposure models (per SD, 95 % block-bootstrap CI)", None),
    ("cause_specific_auc.csv", "Table S47. Pooled out-of-fold ROC-AUC by reported cause (spatial CV, no buffer)", None),
    # graphs
    ("gt_topology_graph_stats.csv", "Table S48. Graph topologies: edge length and label assortativity", None),
    ("gt_topology_summary.csv", "Table S49. GNN performance by graph topology", None),
    ("gt_topology_paired.csv", "Table S50. GNN topology comparisons (corrected resampled t-test)", None),
    ("hydroboost_selection_summary.csv", "Table S51. HydroBoost-GAT blend parameters selected on validation folds", None),
    # robustness (review)
    ("rb_sensitivity_subsets.csv", "Table S52. Sensitivity of the main results to excluding ponds and seasonal sources", None),
    ("rb_validation_distances.csv", "Table S53. Distance from test sources to the nearest training source by validation design", None),
    ("rb_seismic_within_ward_variance.csv", "Table S54. Share of earthquake-exposure variance within wards", None),
    ("rb_seismic_adjusted.csv", "Table S55. Earthquake exposure with ward fixed effects or Moran-eigenvector adjustment", None),
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
    rule_auc = float(rules.loc[rules.feature == "earthquake_impact_known", "single_feature_auc"].iloc[0])

    fig_benchmark(summary)
    fig_leakage(summary, rule_auc)
    lomo = lomo_table(folds, df)

    hb = pd.concat([pd.read_csv(p) for p in METRICS_DIR.glob("hydroboost_selected_params_*.csv")
                    if "_quick" not in p.name and p.stat().st_size > 10])  # profiles without HydroBoost write an empty file
    hb = hb[hb.feature_set.isin(["env_hydro", "env_hydro_season", "leaky_reference"])]
    hb_sel = (hb[hb.variant == "HydroBoost-GAT"].groupby(["profile", "feature_set", "buffer_m"])
              .agg(folds=("w_catboost", "size"), mean_w_catboost=("w_catboost", "mean"),
                   share_w0=("w_catboost", lambda v: (v == 0).mean()), share_w1=("w_catboost", lambda v: (v == 1).mean()),
                   share_no_diffusion=("beta", lambda v: (v == 0).mean())).reset_index())
    hb_sel.to_csv(METRICS_DIR / "hydroboost_selection_summary.csv", index=False)

    # ---------------- SI tables (markdown for the repository, LaTeX for the paper/arXiv) ----------------
    n_seeds = folds[folds.profile == "main"].seed.nunique()
    tabs = []  # (title, note, DataFrame)
    tabs.append(("Table S2. Survey skip-logic leakage audit",
                 "Missingness of survey fields among active vs dried sources (fields with |gap| > 0.05).",
                 audit[audit.skip_logic_leak][["column", "missing_active", "missing_dried", "missingness_gap"]].round(3)))
    tabs.append(("Table S3. ROC-AUC of single features used alone (no model)", "", rules.round(3)))
    for buf, tag, label in [(0.0, "a", "no buffer (interpolation)"), (500.0, "b", "500 m buffer (extrapolation)")]:
        tabs.append((f"Table S4{tag}. Spatial block 5-fold CV, {label}, env_hydro ({n_seeds} seeds x 5 folds)", "",
                     benchmark_table(summary, "main", buf)))
        tabs.append((f"Table S4{tag} (continued). Paired comparison against the best tabular model (ROC-AUC)", "",
                     tests_table(core_tests[core_tests.metric == "roc_auc"], "main", buf, models=CORE_ORDER)))
    if lomo is not None:
        tabs.append(("Table S5. Leave-one-municipality-out (500 m buffer), env_hydro", "",
                     benchmark_table(summary, "lomo", 500.0)))
        tabs.append(("Table S5 (continued). Paired comparison (LOMO)", "",
                     tests_table(core_tests[core_tests.metric == "roc_auc"], "lomo", 500.0, models=CORE_ORDER)))
        tabs.append(("Table S5 (continued). ROC-AUC per held-out municipality (mean over seeds)", "", lomo))
    fs_rows = []
    for fs in ["env", "env_hydro", "env_rich", "env_hydro_season", "env_hydro_seismic", "leaky_reference"]:
        s = summary[(summary.feature_set == fs) & (summary.buffer_m == 0)]
        for _, r in s.iterrows():
            fs_rows.append({"Feature set": fs, "n features": len(FEATURE_SETS.get(fs, [])), "Model": r.model,
                            "ROC-AUC": fmt(r.roc_auc_mean, r.roc_auc_sd), "PR-AUC": fmt(r.pr_auc_mean, r.pr_auc_sd)})
    tabs.append(("Table S6. Feature-set sensitivity (spatial CV, no buffer)", "", pd.DataFrame(fs_rows)))
    # buffer-distance sensitivity (review): env_hydro, spatial CV with 0-2 km buffers, plus LOMO
    bf = folds[(folds.feature_set == "env_hydro") & folds.profile.isin(["main", "buffers", "lomo"])
               & folds.model.isin(["Logistic Regression", "Random Forest", "CatBoost"])].copy()
    if len(bf):
        bf["design"] = np.where(bf.profile == "lomo", "LOMO (500 m buffer)",
                                "spatial CV, buffer " + bf.buffer_m.astype(int).astype(str) + " m")
        bt = (bf.groupby(["design", "model"]).roc_auc.agg(["mean", "std", "size"]).reset_index()
              .assign(**{"ROC-AUC": lambda d: [fmt(m, s) for m, s in zip(d["mean"], d["std"])]})
              .pivot(index="design", columns="model", values="ROC-AUC"))
        order = [d for d in ["spatial CV, buffer 0 m", "spatial CV, buffer 500 m", "spatial CV, buffer 1000 m",
                             "spatial CV, buffer 2000 m", "LOMO (500 m buffer)"] if d in bt.index]
        tabs.append(("Table S6 (continued). ROC-AUC by distance between test and training sources, env_hydro",
                     "Spatial CV 5 seeds x 5 folds; LOMO 3 seeds x 7 municipalities. Distances in Table S53.",
                     bt.loc[order].reset_index()))
    pt = paired_tests(folds[folds.profile == "main"], "ResGATv2", [m for m in folds.model.unique() if "placebo" in m], "roc_auc")
    tabs.append(("Table S7. ResGATv2 placebo graphs (paired vs ResGATv2, ROC-AUC)", "",
                 pd.DataFrame({"Feature set": pt.feature_set, "Buffer (m)": pt.buffer_m.astype(int), "Model": pt.model,
                               "dROC-AUC [95% CI]": [f"{r.mean_diff:+.3f} [{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]"
                                                     for _, r in pt.iterrows()],
                               "p (Holm)": [f"{r.p_holm:.3g}" for _, r in pt.iterrows()]})))
    for name, title, cols in SI_TABLES:
        p = METRICS_DIR / name
        if p.exists() and p.stat().st_size > 10:
            t = pd.read_csv(p)
            if cols:
                t = t[[c for c in cols if c in t.columns]]
            if name == "pm_drying_rules.csv":
                t = pd.concat([t[t.consequent == c].head(40) for c in ["dried", "active"]])
            tabs.append((title, "", t.round(4)))
        else:
            logger.warning(f"SI table source missing: {name}")

    parts = ["# Auto-generated SI tables", "",
             "_Generated by `python -m src.report` from `results/metrics/`. Do not edit by hand; re-run the pipeline. "
             "Benchmark values are mean ± SD over seeds × folds; tests use the corrected resampled t-test with Holm "
             "adjustment. Table S1 (covariate definitions) is in the Supplement text._", ""]
    for title, note, t in tabs:
        parts += [f"## {title}", ""] + ([note, ""] if note else []) + [md_table(t), ""]
    SI_PATH.write_text("\n".join(parts), encoding="utf-8")
    (SI_PATH.parent / "si_tables.tex").write_text("\n".join(latex_table(*x) for x in tabs), encoding="utf-8")
    logger.info(f"Wrote {SI_PATH} and si_tables.tex ({len(tabs)} tables)")
    return summary, core_tests


_TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
            "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "±": r"$\pm$", "→": r"$\rightarrow$",
            "≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$", "²": r"$^2$", "é": r"\'e", "∩": r"$\cap$", "Δ": r"$\Delta$", "φ": r"$\phi$", "ρ": r"$\rho$", "|": r"$|$",
            "−": "-", "–": "--", "—": "---", "\u00a0": " "}


def tex_escape(v) -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return "--"
        v = f"{v:.4g}" if abs(v) < 1e4 else f"{v:.0f}"
    s = str(v)
    return "".join(_TEX_ESC.get(ch, ch if ord(ch) < 128 else "?") for ch in s)


def latex_table(title, note, t: pd.DataFrame, max_rows=45) -> str:
    t = t.reset_index(drop=True) if t.index.name is None else t.reset_index()
    more = len(t) - max_rows
    t = t.head(max_rows)
    head = " & ".join(r"\textbf{" + tex_escape(c) + "}" for c in t.columns)
    body = "\n".join(" & ".join(tex_escape(v) for v in r) + r" \\" for r in t.itertuples(index=False))
    tail = (f"\n\\par\\footnotesize {more} further rows in \\texttt{{paper/si\\_generated\\_tables.md}} and "
            f"\\texttt{{results/metrics}}." if more > 0 else "")
    note_tex = f"\\par\\footnotesize {tex_escape(note)}" if note else ""
    return (f"\\begin{{table}}[!htbp]\\centering\\scriptsize\n\\caption*{{\\textbf{{{tex_escape(title)}}}}}\n"
            f"\\begin{{adjustbox}}{{max width=\\linewidth, max totalheight=0.85\\textheight}}\n"
            f"\\begin{{tabular}}{{{'l' * len(t.columns)}}}\n\\toprule\n{head} \\\\\n\\midrule\n{body}\n\\bottomrule\n"
            f"\\end{{tabular}}\n\\end{{adjustbox}}{note_tex}{tail}\n\\end{{table}}\n\\clearpage\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_report()
