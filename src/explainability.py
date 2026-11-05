"""
Explainability on the primary (leak-free) feature set.

1. TreeSHAP for CatBoost fitted on all springs (descriptive attribution).
2. Logistic-regression standardized coefficients with 95 % spatial-block
   bootstrap CIs (8 x 8 grid blocks resampled with replacement, 500 reps).
   Logistic regression is the best-performing model on the primary feature
   set, so its coefficients are the most defensible driver summary.
3. ResGATv2 attention analysis -- designed to avoid the circularity of the
   previous version (which trained on all labels and defined edge "regimes"
   by the labels themselves):
     * models are trained on the training folds only (spatial 5-fold, seed 42);
     * attention is analysed only for edges whose *receiver* is a test spring;
     * edge groups are defined by geometry only (never by labels);
     * because GATv2 attention is softmax-normalised per receiver, the unit of
       analysis is the receiver: for each receiver we compute (a) the Spearman
       correlation between attention and edge length and (b) the mean attention
       on upslope senders minus downslope senders; we report the mean over
       receivers with a bootstrap 95 % CI;
     * the same statistics are computed for the placebo model trained with
       shuffled edge attributes; a physical interpretation is only supported if
       the real model differs from the placebo.
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
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression

from src.features import PRIMARY_FEATURE_SET
from src.graph_builder import build_graph, load_master
from src.models.res_gat import ResGATv2
from src.models.training import DEVICE, train_node_classifier
from src.spatial_splits import grid_block_ids, spatial_block_kfold

logger = logging.getLogger("explainability")
METRICS_DIR = PROJECT_ROOT / "results/metrics"
FIG_DIR = PROJECT_ROOT / "results/figures"

PRETTY = {
    "latitude": "Latitude", "longitude": "Longitude", "spring_density_500m": "Spring density (500 m)",
    "elevation_m": "Elevation", "slope_deg": "Slope", "aspect_sin": "Aspect (eastness)",
    "aspect_cos": "Aspect (northness)", "precip_annual_mm": "Annual precip.", "precip_monsoon_mm": "Monsoon precip.",
    "precip_dry_season_mm": "Dry-season precip.", "ndvi_mean": "NDVI", "ndwi_mean": "NDWI",
    "Source Type_encoded": "Source type (pond=1)", "landuse_agriculture": "LU: agriculture",
    "landuse_dense_forest": "LU: dense forest", "landuse_fallow": "LU: fallow", "landuse_grassland": "LU: grassland",
    "landuse_others": "LU: other", "landuse_settlement": "LU: settlement", "landuse_sparse_forest": "LU: sparse forest",
    "ownership_community": "Own.: community", "ownership_private": "Own.: private", "ownership_public": "Own.: public",
    "Perennial / Seasonal_encoded": "Seasonal (=1)",
}


def shap_analysis(g):
    import shap
    from catboost import CatBoostClassifier
    X = pd.DataFrame(g["x_raw"], columns=[PRETTY.get(c, c) for c in g["feature_cols"]])
    y = g["y"].numpy()
    m = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05, auto_class_weights="Balanced",
                           verbose=0, random_seed=42, allow_writing_files=False).fit(X, y)
    sv = shap.TreeExplainer(m).shap_values(X)
    imp = pd.DataFrame({"feature": g["feature_cols"], "label": X.columns, "mean_abs_shap": np.abs(sv).mean(0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)
    imp.to_csv(METRICS_DIR / "feature_importance_shap.csv", index=False)

    plt.figure()
    shap.summary_plot(sv, X, show=False, max_display=15)
    plt.title("TreeSHAP (CatBoost, primary feature set)", fontsize=11)
    plt.savefig(FIG_DIR / "shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close("all")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    top = imp.head(15)[::-1]
    ax.barh(top["label"], top["mean_abs_shap"], color="#4C72B0")
    ax.set_xlabel("mean |SHAP| (log-odds)")
    ax.set_title("Global feature attribution (CatBoost, primary set)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "shap_feature_importance_bar.png", dpi=300)
    plt.close(fig)

    top_feat = imp["label"].iloc[0]
    plt.figure()
    shap.dependence_plot(top_feat, sv, X, show=False, interaction_index=None)
    plt.savefig(FIG_DIR / "shap_dependence_top_feature.png", dpi=300, bbox_inches="tight")
    plt.close("all")
    return imp


def logistic_coefficients(g, n_boot=500, seed=0):
    X, y = g["x"].numpy(), g["y"].numpy()
    blocks = grid_block_ids(g["coords_m"], 8)
    ub = np.unique(blocks)
    members = {b: np.where(blocks == b)[0] for b in ub}
    fit = lambda Xs, ys: LogisticRegression(class_weight="balanced", max_iter=3000).fit(Xs, ys).coef_[0]
    coef = fit(X, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([members[b] for b in rng.choice(ub, size=len(ub), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(fit(X[idx], y[idx]))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    out = pd.DataFrame({"feature": g["feature_cols"], "label": [PRETTY.get(c, c) for c in g["feature_cols"]],
                        "coef_per_sd": coef, "ci_low": lo, "ci_high": hi, "odds_ratio_per_sd": np.exp(coef)})
    out["ci_excludes_zero"] = (out.ci_low > 0) | (out.ci_high < 0)
    out = out.reindex(out.coef_per_sd.abs().sort_values(ascending=False).index).reset_index(drop=True)
    out.to_csv(METRICS_DIR / "logistic_coefficients_block_bootstrap.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    d = out[::-1]
    col = np.where(d.ci_excludes_zero, "#C44E52", "#8C8C8C")
    ax.errorbar(d.coef_per_sd, range(len(d)), xerr=[d.coef_per_sd - d.ci_low, d.ci_high - d.coef_per_sd],
                fmt="none", ecolor=col, elinewidth=1.5)
    ax.scatter(d.coef_per_sd, range(len(d)), c=col, zorder=3, s=18)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.label, fontsize=8)
    ax.set_xlabel("Standardized log-odds coefficient (95 % spatial-block bootstrap CI)")
    ax.set_title("Logistic regression: associations with drying")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "logistic_coefficients.png", dpi=300)
    plt.close(fig)
    return out


def _receiver_stats(att_ei, alpha, g, test_mask):
    src, dst = att_ei
    keep = (src != dst) & test_mask[dst]
    src, dst, a = src[keep], dst[keep], alpha[keep].mean(1)
    d = np.linalg.norm(g["coords_m"][dst] - g["coords_m"][src], axis=1)
    upslope_sender = g["elevation"][src] > g["elevation"][dst]
    df = pd.DataFrame({"dst": dst, "a": a, "d": d, "up": upslope_sender})
    rho, diff = [], []
    for _, grp in df.groupby("dst"):
        if grp.d.nunique() > 2:
            rho.append(stats.spearmanr(grp.a, grp.d).statistic)
        if grp.up.any() and (~grp.up).any():
            diff.append(grp.a[grp.up].mean() - grp.a[~grp.up].mean())
    return np.array([r for r in rho if np.isfinite(r)]), np.array(diff), df


def _boot_ci(v, rng, n=2000):
    means = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return float(np.mean(v)), *np.percentile(means, [2.5, 97.5])


def attention_analysis(g, seed=42):
    x, y, ei, ea = g["x"], g["y"], g["edge_index"], g["edge_attr"]
    d_in = x.shape[1]
    rng = np.random.default_rng(seed)
    perm = torch.as_tensor(rng.permutation(ea.shape[0]))
    acc = {"real": ([], []), "placebo_shuffled_edges": ([], [])}
    edge_frames = []
    for split in spatial_block_kfold(g["coords_m"], 5, random_state=seed):
        for name, e_attr in [("real", ea), ("placebo_shuffled_edges", ea[perm])]:
            _, model = train_node_classifier(lambda: ResGATv2(d_in, edge_dim=5), x, ei, e_attr, y,
                                             split["train_mask"], split["val_mask"], seed=seed)
            with torch.no_grad():
                _, (att_ei, alpha) = model(x.to(DEVICE), ei.to(DEVICE), e_attr.to(DEVICE),
                                           return_attention_weights=True)
            rho, diff, df = _receiver_stats(att_ei.cpu().numpy(), alpha.cpu().numpy(), g, split["test_mask"])
            acc[name][0].append(rho)
            acc[name][1].append(diff)
            if name == "real":
                df["fold"] = split["fold"]
                edge_frames.append(df)
    rows = []
    for name, (rhos, diffs) in acc.items():
        r, dfs = np.concatenate(rhos), np.concatenate(diffs)
        m, lo, hi = _boot_ci(r, rng)
        rows.append({"model": name, "statistic": "within-receiver Spearman(attention, edge length)",
                     "n_receivers": len(r), "mean": m, "ci_low": lo, "ci_high": hi})
        m, lo, hi = _boot_ci(dfs, rng)
        rows.append({"model": name, "statistic": "attention(upslope senders) - attention(downslope senders)",
                     "n_receivers": len(dfs), "mean": m, "ci_low": lo, "ci_high": hi})
    res = pd.DataFrame(rows)
    res.to_csv(METRICS_DIR / "attention_analysis.csv", index=False)

    edges = pd.concat(edge_frames)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = [0, 100, 250, 500, 1000, 1e9]
    labels = ["<100 m", "100-250", "250-500", "500-1000", ">1 km"]
    edges["bin"] = pd.cut(edges.d, bins, labels=labels)
    s = edges.groupby("bin", observed=True).a.agg(["mean", "sem"])
    axes[0].errorbar(range(len(s)), s["mean"], yerr=1.96 * s["sem"], marker="o", capsize=3)
    axes[0].axhline(1 / 8, ls="--", c="grey", lw=0.8, label="uniform (1/8)")
    axes[0].set_xticks(range(len(s)))
    axes[0].set_xticklabels(s.index, rotation=15)
    axes[0].set_ylabel("mean attention (test receivers)")
    axes[0].set_title("(a) Attention vs edge length")
    axes[0].legend(fontsize=8)
    ax = axes[1]
    for i, name in enumerate(["real", "placebo_shuffled_edges"]):
        sub = res[res.model == name].reset_index(drop=True)
        ax.errorbar(np.arange(2) + (i - 0.5) * 0.2, sub["mean"],
                    yerr=[sub["mean"] - sub.ci_low, sub.ci_high - sub["mean"]], fmt="o", capsize=3,
                    label="ResGATv2" if name == "real" else "placebo (shuffled edge attrs)")
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Spearman(att, distance)", "upslope - downslope"], fontsize=9)
    ax.set_title("(b) Within-receiver attention statistics (95 % CI)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "attention_analysis.png", dpi=300)
    plt.close(fig)
    return res


def run_all():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    g = build_graph(load_master(), PRIMARY_FEATURE_SET)
    logger.info("TreeSHAP ...")
    shap_analysis(g)
    logger.info("Logistic coefficients with spatial-block bootstrap ...")
    logistic_coefficients(g)
    logger.info("Attention analysis ...")
    print(attention_analysis(g).to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_all()
