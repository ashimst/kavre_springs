"""
Which graph topology carries information about spring drying?

The same GNNs (GraphSAGE, ResGATv2) and a graph-free MLP are trained on the env_hydro features under
identical spatial block CV (5 seeds x 5 folds, no buffer); only the edge set changes:

  knn        7 nearest springs (geographic proximity)                         [reference]
  catchment  7 nearest springs draining to the same DEM link catchment (T = 2 km^2)
  ridge      ridge/catchment-penalised k-NN (src.graph_builder.knn_edges with elevation, HAND, UPA)
  ward       7 nearest springs in the same ward (= same community survey team): a diagnostic for
             survey-team structure, not a hydrological hypothesis
  random     7 random springs anywhere in the district (placebo)

Edge attributes are recomputed from true geometry for every topology. For each topology we also
report label assortativity (share of edges joining same-status springs, relative to the
expectation under random labels) and mean edge length.

Output: results/metrics/gt_topology_folds.csv, gt_topology_summary.csv, gt_topology_paired.csv,
        results/figures/gt_topology.png
Usage:  python -m src.graph_topology_experiment
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
from scipy.spatial import cKDTree

from src.benchmark import evaluate
from src.graph_builder import build_graph, knn_edges, load_master, physical_edge_attributes
from src.models.baselines import MLP, SpringGraphSAGE
from src.models.res_gat import ResGATv2
from src.models.training import train_node_classifier
from src.spatial_splits import spatial_block_kfold

logger = logging.getLogger("graph_topology")
METRICS = PROJECT_ROOT / "results/metrics"
FIGS = PROJECT_ROOT / "results/figures"
HG = PROJECT_ROOT / "data/processed/hydro_gis_features.csv"
SEEDS = [42, 123, 456, 789, 2024]
K = 7


def group_knn(coords, groups, k=K):
    """k nearest neighbours restricted to the same group (fewer if the group is small)."""
    src, dst = [], []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 2:
            continue
        kk = min(k, len(idx) - 1)
        _, nb = cKDTree(coords[idx]).query(coords[idx], k=kk + 1)
        nb = nb.reshape(len(idx), -1)[:, 1:]
        src.append(np.repeat(idx, kk))
        dst.append(idx[nb.ravel()])
    return np.concatenate(src), np.concatenate(dst)


def topologies(df, g, rng):
    coords, elev = g["coords_m"], g["elevation"]
    n = len(coords)
    out = {}
    out["knn"] = knn_edges(coords, K)
    out["catchment"] = group_knn(coords, df.catch_T2p0.to_numpy())
    hnd = df["hnd"].fillna(df["hnd"].median()).to_numpy() if "hnd" in df else None
    upa = df["upa"].fillna(df["upa"].median()).to_numpy() if "upa" in df else None
    out["ridge"] = knn_edges(coords, K, elev=elev, hnd=hnd, upa=upa)
    out["ward"] = group_knn(coords, (df.municipality.astype(str) + "_" + df.ward.astype(str)).to_numpy())
    s = np.repeat(np.arange(n), K)
    d = rng.integers(0, n - 1, n * K)
    out["random"] = (s, d + (d >= s))
    res = {}
    for name, (src, dst) in out.items():
        raw = physical_edge_attributes(coords, elev, src, dst)
        mu, sd = raw.mean(0), raw.std(0)
        sd[sd == 0] = 1
        res[name] = (torch.tensor(np.vstack([src, dst]), dtype=torch.long),
                     torch.tensor((raw - mu) / sd, dtype=torch.float32), raw)
    return res


def assortativity(y, src, dst):
    same = (y[src] == y[dst]).mean()
    p = y.mean()
    exp = p ** 2 + (1 - p) ** 2
    return same, (same - exp) / (1 - exp)


def run():
    METRICS.mkdir(parents=True, exist_ok=True)
    df = load_master().merge(pd.read_csv(HG), on="id", how="left")
    g = build_graph(df, "env_hydro")
    x, y = g["x"], g["y"]
    y_np = y.numpy()
    d_in = x.shape[1]
    topo = topologies(df, g, np.random.default_rng(0))

    stats_rows = []
    for name, (ei, ea, raw) in topo.items():
        same, kappa = assortativity(y_np, ei[0].numpy(), ei[1].numpy())
        stats_rows.append({"topology": name, "n_edges": ei.shape[1], "mean_edge_length_m": float(np.expm1(raw[:, 0]).mean()),
                           "median_edge_length_m": float(np.median(np.expm1(raw[:, 0]))),
                           "share_same_status": same, "assortativity_kappa": kappa})
    gstats = pd.DataFrame(stats_rows)
    gstats.to_csv(METRICS / "gt_topology_graph_stats.csv", index=False)
    logger.info("\n" + gstats.round(3).to_string())

    rows = []
    for seed in SEEDS:
        for split in spatial_block_kfold(g["coords_m"], 5, random_state=seed):
            tr, va, te = split["train_mask"], split["val_mask"], split["test_mask"]
            p, _ = train_node_classifier(lambda: MLP(d_in), x, None, None, y, tr, va, seed=seed)
            rows.append({"seed": seed, "fold": split["fold"], "topology": "none", "model": "MLP",
                         "n_test": split["n_test"], "n_train": split["n_train"], **evaluate(y_np, p, va, te)})
            for name, (ei, ea, _) in topo.items():
                for mname, fac, use_e in [("GraphSAGE", lambda: SpringGraphSAGE(d_in), False),
                                          ("ResGATv2", lambda: ResGATv2(d_in, edge_dim=5), True)]:
                    p, _ = train_node_classifier(fac, x, ei, ea if use_e else None, y, tr, va, seed=seed)
                    rows.append({"seed": seed, "fold": split["fold"], "topology": name, "model": mname,
                                 "n_test": split["n_test"], "n_train": split["n_train"], **evaluate(y_np, p, va, te)})
            logger.info(f"seed {seed} fold {split['fold']} done")
    folds = pd.DataFrame(rows)
    folds.to_csv(METRICS / "gt_topology_folds.csv", index=False)
    summ = folds.groupby(["model", "topology"]).agg(roc_auc=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
                                                     pr_auc=("pr_auc", "mean")).reset_index()
    summ.to_csv(METRICS / "gt_topology_summary.csv", index=False)

    # paired corrected t-tests vs the kNN topology (same model) and vs the MLP
    ratio = (folds.n_test / folds.n_train).mean()
    prow = []
    for mname in ["GraphSAGE", "ResGATv2"]:
        piv = folds[folds.model == mname].pivot_table(index=["seed", "fold"], columns="topology", values="roc_auc")
        mlp = folds[folds.model == "MLP"].set_index(["seed", "fold"]).roc_auc
        for name in topo:
            for ref_name, ref in [("knn", piv["knn"]), ("MLP", mlp)]:
                if name == ref_name:
                    continue
                d = (piv[name] - ref.reindex(piv.index)).dropna().to_numpy()
                se = np.sqrt((1 / len(d) + ratio) * d.var(ddof=1))
                t = d.mean() / se
                prow.append({"model": mname, "topology": name, "reference": ref_name, "mean_diff": d.mean(),
                             "ci_low": d.mean() - stats.t.ppf(0.975, len(d) - 1) * se,
                             "ci_high": d.mean() + stats.t.ppf(0.975, len(d) - 1) * se,
                             "p_corrected": 2 * stats.t.sf(abs(t), len(d) - 1)})
    paired = pd.DataFrame(prow)
    paired.to_csv(METRICS / "gt_topology_paired.csv", index=False)
    logger.info("\n" + summ.round(3).to_string() + "\n" + paired.round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    ax.bar(gstats.topology, gstats.assortativity_kappa, color="#4C72B0")
    ax.set_ylabel("Label assortativity (kappa over random labels)")
    ax.set_title("(a) How strongly does each graph join same-status springs?", fontsize=10)
    for i, r in gstats.iterrows():
        ax.text(i, r.assortativity_kappa, f"{r.median_edge_length_m:.0f} m", ha="center", va="bottom", fontsize=8)
    ax = axes[1]
    sub = paired[paired.reference == "MLP"]
    for k, mname in enumerate(["GraphSAGE", "ResGATv2"]):
        s = sub[sub.model == mname].reset_index(drop=True)
        xs = np.arange(len(s)) + (k - 0.5) * 0.25
        ax.errorbar(xs, s.mean_diff, yerr=[s.mean_diff - s.ci_low, s.ci_high - s.mean_diff], fmt="o", capsize=3, label=mname)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(s.topology)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_ylabel("ΔROC-AUC vs graph-free MLP (corrected 95% CI)")
    ax.set_title("(b) GNN gain over MLP by graph topology (spatial CV)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "gt_topology.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()
