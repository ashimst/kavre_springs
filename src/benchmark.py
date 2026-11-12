"""
Predictive benchmark -- every prediction number in the paper is computed here.

Standard models only: logistic regression, random forest, CatBoost and distance-weighted kNN label interpolation.
Neural networks (MLP, Kolmogorov-Arnold networks, graph neural networks) were tried and removed: none differed from
logistic regression or improved on the random forest (docs/experiment_log.md).

Profiles
--------
main      Spatial block 5-fold CV on ``env`` (basic covariates) and ``env_hydro`` (primary: + catchment
          hydrology and roads), buffers 0 m and 500 m, 5 seeds (each seed reshuffles the block-to-fold
          assignment AND re-initialises models).
features  Same CV (buffer 0 m) for the seasonal-flow sensitivity set ``env_hydro_season`` and the diagnostic
          ``leaky_reference`` set (skip-logic earthquake fields).
enriched  Do the Earth Engine enrichment layers (``env_rich``) add skill?
buffers   Wider 1 and 2 km buffers around test sources (env_hydro and env_rich).
lomo      Leave-one-municipality-out with a 500 m buffer, 3 seeds (env, env_hydro and env_rich).

Protocol (identical for all models)
-----------------------------------
* Models see labels of training sources only. Median imputation and z-scoring are fitted on the training rows of
  each fold (fold_preprocess); hyperparameters are fixed and there is no feature selection. Validation sources set the decision threshold (chosen to maximise
  macro-F1 on validation). Test sources are never used.
* Every model is seeded independently, so results do not depend on which other models are run.
* Metrics on the test fold: ROC-AUC, PR-AUC (average precision), macro-F1 and balanced accuracy at the
  validation-chosen threshold and Brier score.

Outputs: results/metrics/benchmark_<profile>_folds.csv (one row per model x fold x seed x setting) and
results/predictions/test_predictions_<profile>.parquet. Runs are checkpointed per fold and resume automatically.

Usage:  python -m src.benchmark --profile main [--seeds 42 123] [--quick]
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             roc_auc_score)

from src.features import check_feature_set
from src.dataset import design_matrix, load_master
from src.models import get_models, knn_label_predict
from src.spatial_splits import leave_one_municipality_out, spatial_block_kfold

logger = logging.getLogger("benchmark")

METRICS_DIR = PROJECT_ROOT / "results/metrics"
PRED_DIR = PROJECT_ROOT / "results/predictions"
SEEDS = [42, 123, 456, 789, 2024]

TABULAR = ["Logistic Regression", "Random Forest", "CatBoost"]
BENCHMARK_MODELS = TABULAR + ["kNN label interpolation"]

PROFILES = {
    "main": dict(scheme="spatial", feature_sets=["env", "env_hydro"], buffers=[0.0, 500.0], seeds=SEEDS,
                 models=BENCHMARK_MODELS),
    "features": dict(scheme="spatial", feature_sets=["leaky_reference", "env_hydro_season"], buffers=[0.0], seeds=SEEDS,
                     models=["Logistic Regression", "CatBoost"]),
    # Do the Earth Engine enrichment layers (land cover, forest loss, soils, vegetation dynamics, built-up) add skill?
    "enriched": dict(scheme="spatial", feature_sets=["env_rich"], buffers=[0.0, 500.0], seeds=SEEDS,
                     models=TABULAR),
    # Wider 1 and 2 km buffers between test and training sources
    "buffers": dict(scheme="spatial", feature_sets=["env_hydro", "env_rich"], buffers=[1000.0, 2000.0], seeds=SEEDS,
                    models=["Logistic Regression", "Random Forest", "CatBoost"]),
    "lomo": dict(scheme="lomo", feature_sets=["env", "env_hydro", "env_rich"], buffers=[500.0], seeds=SEEDS[:3],
                 models=BENCHMARK_MODELS),
}
SAVE_PRED_MODELS = {"CatBoost", "Logistic Regression", "Random Forest", "kNN label interpolation"}


def evaluate(y, p, val_mask, test_mask):
    thresholds = np.linspace(0.02, 0.98, 97)
    yv, pv = y[val_mask], p[val_mask]
    f1s = [f1_score(yv, pv >= t, average="macro", zero_division=0) for t in thresholds]
    thr = float(thresholds[int(np.argmax(f1s))])
    yt, pt = y[test_mask], p[test_mask]
    return {
        "roc_auc": roc_auc_score(yt, pt),
        "pr_auc": average_precision_score(yt, pt),
        "f1_macro": f1_score(yt, pt >= thr, average="macro", zero_division=0),
        "bal_acc": balanced_accuracy_score(yt, pt >= thr),
        "brier": brier_score_loss(yt, pt),
        "threshold": thr,
        "test_prevalence": float(yt.mean()),
    }


def fold_preprocess(x_raw, tr):
    """Median imputation and z-scoring fitted on the training rows of this fold only, applied to all rows."""
    x = x_raw.astype(np.float64).copy()
    med = np.nanmedian(x[tr], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    r, c = np.where(np.isnan(x))
    x[r, c] = med[c]
    mu, sd = x[tr].mean(axis=0), x[tr].std(axis=0)
    sd[sd == 0] = 1.0
    return ((x - mu) / sd).astype(np.float32)


def run_fold(g, split, seed, models):
    """Return {model_name: probabilities for all nodes}. Preprocessing is fitted inside the fold; hyperparameters
    are fixed (src.models) and no feature selection is performed, so nothing is tuned on validation or test rows
    except the decision threshold, which uses validation rows only."""
    y_np = g["y"]
    tr = split["train_mask"]
    x_np = fold_preprocess(g["x_raw"], tr)
    out = {}
    pos_weight = float((y_np[tr] == 0).sum() / max((y_np[tr] == 1).sum(), 1))
    tab = get_models(seed, pos_weight)
    for name in TABULAR:
        if name in models:
            m = tab[name]()
            m.fit(x_np[tr], y_np[tr])
            out[name] = m.predict_proba(x_np)[:, 1]
    if "kNN label interpolation" in models:
        out["kNN label interpolation"] = knn_label_predict(g["coords_m"], y_np, tr)
    return out


def make_splits(g, scheme, seed, buffer_m):
    if scheme == "spatial":
        return spatial_block_kfold(g["coords_m"], n_splits=5, random_state=seed, buffer_m=buffer_m)
    return leave_one_municipality_out(g["coords_m"], g["municipality"], random_state=seed, buffer_m=buffer_m)


def run_profile(profile: str, seeds=None, quick=False):
    cfg = PROFILES[profile]
    seeds = seeds or cfg["seeds"]
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    fold_path = METRICS_DIR / f"benchmark_{profile}_folds.csv"
    pred_path = PRED_DIR / f"test_predictions_{profile}.parquet"
    if quick:
        fold_path = fold_path.with_name(fold_path.stem + "_quick.csv")
        pred_path = pred_path.with_name(pred_path.stem + "_quick.parquet")

    done = set()
    rows, preds = [], []
    if fold_path.exists():
        prev = pd.read_csv(fold_path)
        rows = prev.to_dict("records")
        done = set(zip(prev.feature_set, prev.buffer_m, prev.seed, prev.fold.astype(str)))
        if pred_path.exists():
            preds = [pd.read_parquet(pred_path)]
        logger.info(f"Resuming {profile}: {len(done)} folds already done")

    df = load_master()
    for fs in cfg["feature_sets"]:
        violations = check_feature_set(df, fs)
        if violations and fs != "leaky_reference":
            raise RuntimeError(f"Feature set '{fs}' fails the leakage audit: {violations}")
        g = design_matrix(df, fs)
        for buffer_m in cfg["buffers"]:
            for seed in seeds:
                splits = make_splits(g, cfg["scheme"], seed, buffer_m)
                if quick:
                    splits = splits[:1]
                for split in splits:
                    key = (fs, buffer_m, seed, str(split["fold"]))
                    if key in done:
                        continue
                    t0 = time.time()
                    probs = run_fold(g, split, seed, cfg["models"])
                    y_np = g["y"]
                    meta = dict(profile=profile, scheme=cfg["scheme"], feature_set=fs, buffer_m=buffer_m, seed=seed,
                                fold=str(split["fold"]), n_train=split["n_train"], n_val=split["n_val"],
                                n_test=split["n_test"], n_buffer=split["n_buffer"])
                    for name, p in probs.items():
                        rows.append({**meta, "model": name, **evaluate(y_np, p, split["val_mask"], split["test_mask"])})
                        if name in SAVE_PRED_MODELS:
                            te = split["test_mask"]
                            preds.append(pd.DataFrame({"id": g["id"][te], "model": name, "feature_set": fs,
                                                       "buffer_m": buffer_m, "seed": seed, "fold": str(split["fold"]),
                                                       "y": y_np[te], "p": p[te]}))
                    pd.DataFrame(rows).to_csv(fold_path, index=False)
                    pd.concat(preds).to_parquet(pred_path, index=False)
                    cb = [r for r in rows if r["model"] == "CatBoost" and r["seed"] == seed and r["fold"] == str(split["fold"])
                          and r["buffer_m"] == buffer_m and r["feature_set"] == fs]
                    logger.info(f"[{profile}|{fs}|buf={buffer_m:.0f}|seed={seed}|fold={split['fold']}] "
                                f"{time.time() - t0:.0f}s  CatBoost AUC={cb[-1]['roc_auc']:.3f}" if cb else
                                f"[{profile}|{fs}|buf={buffer_m:.0f}|seed={seed}|fold={split['fold']}] {time.time() - t0:.0f}s")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(PROFILES) + ["all"], default="all")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--quick", action="store_true", help="1 fold per seed, separate *_quick outputs")
    a = ap.parse_args()
    for prof in (PROFILES if a.profile == "all" else [a.profile]):
        run_profile(prof, a.seeds, a.quick)
