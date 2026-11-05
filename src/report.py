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
        s = summary[(summary.profile == prof) & (summary.buffer_m == buf) & (summary.feature_set == PRIMARY_FEATURE_SET)]
        s = s.set_index("model").loc[[m for m in CORE_ORDER if m in set(s.model)]]
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


def fig_controls(tests_ctrl: pd.DataFrame):
    t = tests_ctrl[(tests_ctrl.profile == "main")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True, sharey=True)
    for ax, buf in zip(axes, [0.0, 500.0]):
        s = t[t.buffer_m == buf].reset_index(drop=True)
        y = np.arange(len(s))[::-1]
        ax.errorbar(s.mean_diff, y, xerr=[s.mean_diff - s.ci95_low, s.ci95_high - s.mean_diff], fmt="o", capsize=3)
        ax.axvline(0, c="k", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{m}  (vs {r})" for m, r in zip(s.model, s.reference)], fontsize=8)
        ax.set_xlabel("ΔROC-AUC vs reference\n(corrected 95 % CI)")
        ax.set_title(f"{'No buffer' if buf == 0 else f'{buf:.0f} m buffer'}", fontsize=10)
    fig.suptitle("Ablations and placebo controls (spatial 5-fold CV, 5 seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_controls.png", dpi=300)
    plt.close(fig)


def fig_leakage(summary: pd.DataFrame, rule_auc: float):
    order = [PRIMARY_FEATURE_SET, "env_season", "leaky_reference"]
    names = {"env": "Primary (env)", "env_season": "+ seasonality", "leaky_reference": "+ earthquake fields (leaky)"}
    s = summary[(summary.buffer_m == 0) & summary.feature_set.isin(order) & summary.profile.isin(["main", "features"])]
    models = [m for m in ["Logistic Regression", "CatBoost", "MLP", "ResGATv2", "HydroBoost-GAT"] if m in set(s.model)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.8 / len(order)
    for i, fs in enumerate(order):
        v = s[s.feature_set == fs].set_index("model").reindex(models)
        ax.bar(np.arange(len(models)) + (i - 1) * w, v.roc_auc_mean, w, yerr=v.roc_auc_ci95, capsize=2, label=names[fs])
    ax.axhline(rule_auc, ls="--", c="crimson", label=f"1-line rule: earthquake answer missing (AUC {rule_auc:.3f})")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("ROC-AUC (spatial CV, no buffer)")
    ax.set_title("Survey skip-logic leakage inflates performance")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "leakage_feature_sets.png", dpi=300)
    plt.close(fig)


def lomo_table(folds: pd.DataFrame, df: pd.DataFrame):
    l = folds[folds.profile == "lomo"]
    if l.empty:
        return None
    piv = l.groupby(["fold", "model"]).roc_auc.mean().unstack()
    piv = piv[[m for m in CORE_ORDER if m in piv.columns]]
    info = df.groupby("municipality").agg(N=("target_dried", "size"), dried=("target_dried", "sum"))
    info["Drying rate"] = (info.dried / info.N).map(lambda v: f"{100 * v:.1f}%")
    out = info.join(piv.round(3)).reset_index().rename(columns={"municipality": "Held-out municipality"})
    out.to_csv(TABLE_DIR / "lomo_per_municipality_auc.csv", index=False)
    return out


def run_report():
    for d in (TABLE_DIR, FIG_DIR, SI_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)
    df = load_master()
    folds = load_folds()
    summary = summarise(folds)
    summary.to_csv(METRICS_DIR / "benchmark_summary.csv", index=False)

    core_tests = pd.concat([paired_tests(folds, None, CORE_ORDER, m) for m in ["roc_auc", "pr_auc"]])
    core_tests.to_csv(METRICS_DIR / "paired_tests_vs_best_tabular.csv", index=False)
    cb_tests = paired_tests(folds, "CatBoost", CORE_ORDER, "roc_auc")
    cb_tests.to_csv(METRICS_DIR / "paired_tests_vs_catboost.csv", index=False)

    main = folds[folds.profile == "main"]
    ctrl = pd.concat([
        paired_tests(main, "ResGATv2", [m for m in main.model.unique() if m.startswith("ResGATv2 ")], "roc_auc"),
        paired_tests(main, "HydroBoost-GAT", [m for m in main.model.unique() if m.startswith("HydroBoost |")], "roc_auc"),
        paired_tests(main, "MLP", ["ResGATv2", "GraphSAGE", "GCN", "GATv2"], "roc_auc"),
    ], ignore_index=True)
    ctrl.to_csv(METRICS_DIR / "ablation_and_placebo_tests.csv", index=False)

    audit = missingness_audit(df)
    audit.to_csv(METRICS_DIR / "leakage_audit.csv", index=False)
    rules = single_feature_rule_scores(df)
    rules.to_csv(METRICS_DIR / "single_feature_auc.csv", index=False)
    rule_auc = float(rules.loc[rules.feature == "earthquake_impact_known", "single_feature_auc"].iloc[0])

    fig_benchmark(summary)
    fig_controls(ctrl[ctrl.reference.isin(["ResGATv2", "HydroBoost-GAT", "MLP"])])
    if (summary.feature_set != PRIMARY_FEATURE_SET).any():
        fig_leakage(summary, rule_auc)
    lomo = lomo_table(folds, df)

    hb = pd.concat([pd.read_csv(p) for p in METRICS_DIR.glob("hydroboost_selected_params_*.csv") if "_quick" not in p.name])
    hb_sel = (hb[hb.variant == "HydroBoost-GAT"].groupby(["profile", "feature_set", "buffer_m"])
              .agg(folds=("w_catboost", "size"), mean_w_catboost=("w_catboost", "mean"),
                   share_w0=("w_catboost", lambda v: (v == 0).mean()), share_w1=("w_catboost", lambda v: (v == 1).mean()),
                   share_no_diffusion=("beta", lambda v: (v == 0).mean())).reset_index())
    hb_sel.to_csv(METRICS_DIR / "hydroboost_selection_summary.csv", index=False)

    # ---------------- SI markdown ----------------
    parts = ["# Auto-generated SI tables",
             "",
             "_Generated by `python -m src.report` from `results/metrics/`. Do not edit by hand; "
             "re-run the pipeline instead. Values are mean ± SD over seeds × folds; "
             "statistical tests use the corrected resampled t-test with Holm adjustment._", ""]
    n_seeds = main.seed.nunique()
    for buf, label in [(0.0, "no buffer (interpolation)"), (500.0, "500 m buffer (extrapolation)")]:
        parts += [f"## Table S2{'a' if buf == 0 else 'b'}. Spatial block 5-fold CV, {label}, primary feature set "
                  f"({n_seeds} seeds × 5 folds)", "", md_table(benchmark_table(summary, "main", buf)), "",
                  f"Paired comparison against the best graph-free tabular model (ROC-AUC):", "",
                  md_table(tests_table(core_tests[core_tests.metric == "roc_auc"], "main", buf, models=CORE_ORDER)), ""]
    if lomo is not None:
        parts += ["## Table S3. Leave-one-municipality-out (500 m buffer), primary feature set", "",
                  md_table(benchmark_table(summary, "lomo", 500.0)), "",
                  "Per held-out municipality (ROC-AUC, mean over seeds):", "", md_table(lomo), ""]
    parts += ["## Table S4. Survey skip-logic leakage audit", "",
              "Missingness of survey fields among active vs dried sources (fields with |gap| > 0.05 shown).", "",
              md_table(audit[audit.skip_logic_leak][["column", "missing_active", "missing_dried", "missingness_gap"]]
                       .round(3).head(40)), "",
              "ROC-AUC of single features used alone (no model):", "", md_table(rules.round(3)), ""]
    fs_rows = []
    for fs in [PRIMARY_FEATURE_SET, "env_season", "leaky_reference"]:
        s = summary[(summary.feature_set == fs) & (summary.buffer_m == 0) & summary.profile.isin(["main", "features"])]
        for _, r in s.iterrows():
            if r.model in ["Logistic Regression", "CatBoost", "MLP", "ResGATv2", "HydroBoost-GAT"]:
                fs_rows.append({"Feature set": fs, "n features": len(FEATURE_SETS[fs]), "Model": r.model,
                                "ROC-AUC": fmt(r.roc_auc_mean, r.roc_auc_sd), "PR-AUC": fmt(r.pr_auc_mean, r.pr_auc_sd),
                                "Bal. acc.": fmt(r.bal_acc_mean, r.bal_acc_sd)})
    if fs_rows:
        parts += ["## Table S5. Feature-set sensitivity (spatial CV, no buffer)", "", md_table(pd.DataFrame(fs_rows)), ""]
    c = ctrl.copy()
    parts += ["## Table S6. Ablations and negative controls (ROC-AUC, paired vs reference)", "",
              md_table(pd.DataFrame({
                  "Buffer (m)": c.buffer_m.astype(int), "Model": c.model, "Reference": c.reference,
                  "ΔROC-AUC [95% CI]": [f"{r.mean_diff:+.3f} [{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]" for _, r in c.iterrows()],
                  "Wins": [f"{r.wins}/{r.n_pairs}" for _, r in c.iterrows()],
                  "p (Holm)": [f"{r.p_holm:.3g}" for _, r in c.iterrows()]})), "",
              "HydroBoost-GAT blend parameters selected on validation folds:", "", md_table(hb_sel.round(3)), ""]
    for name, title in [("attention_analysis.csv", "Table S7. ResGATv2 attention vs geometry (test receivers), real vs placebo"),
                        ("logistic_coefficients_block_bootstrap.csv", "Table S8. Logistic regression coefficients (95 % spatial-block bootstrap CI)"),
                        ("spatial_autocorrelation_distance_decay.csv", "Table S9. Global Moran's I distance decay"),
                        ("lisa_cluster_summary.csv", "Table S10. LISA clusters (k = 7)"),
                        ("priority_index_sensitivity.csv", "Table S11. Spring Priority Index sensitivity"),
                        ("drying_timing_summary.csv", "Table S13. Reported timing of drying (dried sources only)"),
                        ("drying_reasons.csv", "Table S14. Respondent-reported reason of drying"),
                        ("cause_specific_auc.csv", "Table S15. Pooled out-of-fold ROC-AUC by reported cause of drying (spatial CV, no buffer)")]:
        p = METRICS_DIR / name
        if p.exists():
            t = pd.read_csv(p)
            if name.startswith("logistic"):
                t = t[["label", "coef_per_sd", "ci_low", "ci_high", "odds_ratio_per_sd"]]
            parts += [f"## {title}", "", md_table(t.round(4)), ""]
    SI_PATH.write_text("\n".join(parts), encoding="utf-8")
    logger.info(f"Wrote {SI_PATH}")
    return summary, core_tests, ctrl


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_report()
