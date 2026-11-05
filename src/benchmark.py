"""
Benchmark, ablations and negative controls -- every number is computed here.

Profiles
--------
main      Spatial block 5-fold CV, primary feature set (``env``), buffers 0 m
          (interpolation) and 500 m (extrapolation), 5 seeds (each seed
          reshuffles the block-to-fold assignment AND re-initialises models).
          All models, ResGATv2 ablations/placebos and HydroBoost ablations.
features  Same CV (buffer 0 m) for the sensitivity feature set ``env_season``
          and the diagnostic ``leaky_reference`` set, on a subset of models.
lomo      Leave-one-municipality-out with a 500 m buffer, 3 seeds.

Protocol (identical for all models)
-----------------------------------
* Models see labels of training springs only. Validation springs are used for
  early stopping, HydroBoost blend selection and the decision threshold
  (chosen to maximise macro-F1 on validation). Test springs are never used.
* Metrics on the test fold: ROC-AUC, PR-AUC (average precision), macro-F1 and
  balanced accuracy at the validation-chosen threshold, Brier score.

Outputs (results/metrics/): benchmark_<profile>_folds.csv (one row per
model x fold x seed x setting), hydroboost_selected_params.csv,
results/predictions/test_predictions_<profile>.parquet.
Runs are checkpointed per fold and resume automatically.

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
import zlib

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             roc_auc_score)

from src.features import PRIMARY_FEATURE_SET, check_feature_set
from src.graph_builder import build_graph, load_master, rewired_graph
from src.models.baselines import (MLP, SpringGCN, SpringGraphSAGE, VanillaGATv2, get_tabular_models,
                                  knn_label_predict)
from src.models.hydroboost import FULL_GRID, normalised_adjacency, oof_catboost, select_and_predict
from src.models.res_gat import ResGATv2
from src.models.training import train_node_classifier
from src.spatial_splits import leave_one_municipality_out, spatial_block_kfold

logger = logging.getLogger("benchmark")

METRICS_DIR = PROJECT_ROOT / "results/metrics"
PRED_DIR = PROJECT_ROOT / "results/predictions"
SEEDS = [42, 123, 456, 789, 2024]

CORE_MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "CatBoost", "kNN label interpolation",
               "MLP", "GCN", "GraphSAGE", "GATv2", "ResGATv2", "HydroBoost-GAT"]
CONTROL_MODELS = ["ResGATv2 - no edge attributes", "ResGATv2 - no ego skip", "ResGATv2 + LayerNorm",
                  "ResGATv2 (2 layers)", "ResGATv2 | placebo: shuffled edge attributes",
                  "ResGATv2 | placebo: random topology",
                  "HydroBoost | CatBoost + diffusion only", "HydroBoost | CatBoost + GAT, no diffusion"]

PROFILES = {
    "main": dict(scheme="spatial", feature_sets=[PRIMARY_FEATURE_SET], buffers=[0.0, 500.0], seeds=SEEDS,
                 models=CORE_MODELS + CONTROL_MODELS),
    "features": dict(scheme="spatial", feature_sets=["env_season", "leaky_reference"], buffers=[0.0], seeds=SEEDS,
                     models=["Logistic Regression", "CatBoost", "MLP", "ResGATv2", "HydroBoost-GAT"]),
    "lomo": dict(scheme="lomo", feature_sets=[PRIMARY_FEATURE_SET], buffers=[500.0], seeds=SEEDS[:3],
                 models=["Logistic Regression", "Random Forest", "CatBoost", "kNN label interpolation", "MLP",
                         "GraphSAGE", "ResGATv2", "HydroBoost-GAT"]),
}
SAVE_PRED_MODELS = {"CatBoost", "ResGATv2", "HydroBoost-GAT", "Logistic Regression", "kNN label interpolation"}


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


def run_fold(g, split, seed, models, rng):
    """Return {model_name: probabilities for all nodes} and HydroBoost params."""
    x, y, ei, ea = g["x"], g["y"], g["edge_index"], g["edge_attr"]
    x_np, y_np = x.numpy(), y.numpy()
    tr, va = split["train_mask"], split["val_mask"]
    d_in = x.shape[1]
    out, hb_params = {}, []
    pos_weight = float((y_np[tr] == 0).sum() / max((y_np[tr] == 1).sum(), 1))
    tab = get_tabular_models(seed, pos_weight)

    need_cb = any(m in models for m in ["CatBoost", "HydroBoost-GAT", "HydroBoost | CatBoost + diffusion only",
                                         "HydroBoost | CatBoost + GAT, no diffusion"])
    p_cb_oof = oof_catboost(x_np, y_np, g["coords_m"], tr, seed) if need_cb else None
    if "CatBoost" in models:
        m = tab["CatBoost"]()
        m.fit(x_np[tr], y_np[tr])
        out["CatBoost"] = m.predict_proba(x_np)[:, 1]
    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        if name in models:
            m = tab[name]()
            m.fit(x_np[tr], y_np[tr])
            out[name] = m.predict_proba(x_np)[:, 1]
    if "kNN label interpolation" in models:
        out["kNN label interpolation"] = knn_label_predict(g["coords_m"], y_np, tr)

    def nn_run(factory, edge_index=ei, edge_attr=ea):
        p, _ = train_node_classifier(factory, x, edge_index, edge_attr, y, tr, va, seed=seed)
        return p

    neural = {
        "MLP": (lambda: MLP(d_in), None, None),
        "GCN": (lambda: SpringGCN(d_in), ei, None),
        "GraphSAGE": (lambda: SpringGraphSAGE(d_in), ei, None),
        "GATv2": (lambda: VanillaGATv2(d_in, edge_dim=5), ei, ea),
        "ResGATv2": (lambda: ResGATv2(d_in, edge_dim=5), ei, ea),
        "ResGATv2 - no edge attributes": (lambda: ResGATv2(d_in, edge_dim=None), ei, None),
        "ResGATv2 - no ego skip": (lambda: ResGATv2(d_in, edge_dim=5, ego_skip=False), ei, ea),
        "ResGATv2 + LayerNorm": (lambda: ResGATv2(d_in, edge_dim=5, layer_norm=True), ei, ea),
        "ResGATv2 (2 layers)": (lambda: ResGATv2(d_in, edge_dim=5, num_layers=2), ei, ea),
    }
    needs_resgat = any(m.startswith("HydroBoost") for m in models)
    for name, (factory, e_i, e_a) in neural.items():
        if name in models or (name == "ResGATv2" and needs_resgat):
            out[name] = nn_run(factory, e_i, e_a)
    if "ResGATv2 | placebo: shuffled edge attributes" in models:
        perm = torch.as_tensor(rng.permutation(ea.shape[0]))
        out["ResGATv2 | placebo: shuffled edge attributes"] = nn_run(lambda: ResGATv2(d_in, edge_dim=5), ei, ea[perm])
    if "ResGATv2 | placebo: random topology" in models:
        ei_r, ea_r = rewired_graph(g, rng)
        out["ResGATv2 | placebo: random topology"] = nn_run(lambda: ResGATv2(d_in, edge_dim=5), ei_r, ea_r)

    if needs_resgat:
        S = normalised_adjacency(ei.numpy(), len(y_np))
        variants = {
            "HydroBoost-GAT": FULL_GRID,
            "HydroBoost | CatBoost + diffusion only": {**FULL_GRID, "w": [1.0]},
            "HydroBoost | CatBoost + GAT, no diffusion": {**FULL_GRID, "beta": [0.0], "alpha": [0.4]},
        }
        for name, grid in variants.items():
            if name in models:
                p, params = select_and_predict(p_cb_oof, out["ResGATv2"], y_np, S, tr, va, grid)
                out[name] = p
                hb_params.append({"variant": name, **params})
    if "ResGATv2" not in models:
        out.pop("ResGATv2", None)
    return out, hb_params


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
    hb_path = METRICS_DIR / f"hydroboost_selected_params_{profile}.csv"
    pred_path = PRED_DIR / f"test_predictions_{profile}.parquet"
    if quick:
        fold_path = fold_path.with_name(fold_path.stem + "_quick.csv")
        hb_path = hb_path.with_name(hb_path.stem + "_quick.csv")
        pred_path = pred_path.with_name(pred_path.stem + "_quick.parquet")

    done = set()
    rows, hb_rows, preds = [], [], []
    if fold_path.exists():
        prev = pd.read_csv(fold_path)
        rows = prev.to_dict("records")
        done = set(zip(prev.feature_set, prev.buffer_m, prev.seed, prev.fold.astype(str)))
        if hb_path.exists():
            hb_rows = pd.read_csv(hb_path).to_dict("records")
        if pred_path.exists():
            preds = [pd.read_parquet(pred_path)]
        logger.info(f"Resuming {profile}: {len(done)} folds already done")

    df = load_master()
    for fs in cfg["feature_sets"]:
        violations = check_feature_set(df, fs)
        if violations and fs != "leaky_reference":
            raise RuntimeError(f"Feature set '{fs}' fails the leakage audit: {violations}")
        g = build_graph(df, fs)
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
                    rng = np.random.default_rng([seed, zlib.crc32(str(split["fold"]).encode()), int(buffer_m)])
                    probs, hb = run_fold(g, split, seed, cfg["models"], rng)
                    y_np = g["y"].numpy()
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
                    hb_rows += [{**meta, **h} for h in hb]
                    pd.DataFrame(rows).to_csv(fold_path, index=False)
                    pd.DataFrame(hb_rows).to_csv(hb_path, index=False)
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
