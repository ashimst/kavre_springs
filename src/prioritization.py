"""
Spring Priority Index (SPI) for currently ACTIVE springs.

Household-dependency fields exist only for active springs (the survey skips
them for dried sources), so prioritisation is restricted to active springs.

    susceptibility_i = out-of-fold predicted drying probability, averaged over
                       the 5 seeds of the main benchmark (spatial CV, 0 m buffer),
                       from the best model on that setting (chosen by mean ROC-AUC
                       from the saved results -- not hard-coded);
    exposure_i       = number of households dependent on spring i;
    SPI_i            = pct_rank(susceptibility_i) * pct_rank(exposure_i),

with percentile ranks computed among active springs that have both values.
Equal weighting is a transparent default, not an optimum; the Spearman
correlation of SPI with susceptibility-only and exposure-only rankings is
reported so readers can judge sensitivity. "Susceptibility" means "environmental
similarity to springs that have dried", not a calibrated probability of future
drying.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.features import PRIMARY_FEATURE_SET
from src.graph_builder import load_master

logger = logging.getLogger("prioritization")
METRICS_DIR = PROJECT_ROOT / "results/metrics"
TABLE_DIR = PROJECT_ROOT / "results/tables"
PRED_PATH = PROJECT_ROOT / "results/predictions/test_predictions_main.parquet"


def run_prioritization(top_n: int = 500):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(METRICS_DIR / "benchmark_main_folds.csv")
    preds = pd.read_parquet(PRED_PATH)
    cand = preds[(preds.buffer_m == 0) & (preds.feature_set == PRIMARY_FEATURE_SET)].model.unique()
    perf = (folds[(folds.buffer_m == 0) & (folds.feature_set == PRIMARY_FEATURE_SET) & folds.model.isin(cand)]
            .groupby("model").roc_auc.mean().sort_values(ascending=False))
    model = perf.index[0]
    logger.info(f"Susceptibility model: {model} (mean ROC-AUC {perf.iloc[0]:.3f})")

    p = preds[(preds.model == model) & (preds.buffer_m == 0) & (preds.feature_set == PRIMARY_FEATURE_SET)]
    n_seeds = p.seed.nunique()
    sus = p.groupby("id").p.mean().rename("susceptibility")

    df = load_master().set_index("id")
    df = df.join(sus)
    act = df[(df.target_dried == 0) & df.susceptibility.notna() & df.hh_dependent_total.notna()].copy()
    act["susceptibility_pct"] = act.susceptibility.rank(pct=True)
    act["exposure_pct"] = act.hh_dependent_total.rank(pct=True)
    act["SPI"] = act.susceptibility_pct * act.exposure_pct
    act = act.sort_values("SPI", ascending=False)
    act["SPI_rank"] = np.arange(1, len(act) + 1)

    cols = ["SPI_rank", "municipality", "ward", "latitude", "longitude", "susceptibility", "hh_dependent_total",
            "hh_dependent_exclusive", "fetch_time_minutes", "susceptibility_pct", "exposure_pct", "SPI"]
    act.reset_index()[["id"] + cols].head(top_n).to_csv(TABLE_DIR / "top_500_priority_active_springs.csv", index=False)

    top = act.head(top_n)
    all_m = df.groupby("municipality").agg(total_sources=("target_dried", "size"), dried=("target_dried", "sum"))
    all_m["drying_rate"] = all_m.dried / all_m.total_sources
    m = act.groupby("municipality").agg(active_ranked=("SPI", "size"), mean_susceptibility=("susceptibility", "mean"),
                                        households_dependent=("hh_dependent_total", "sum"))
    m["in_top_500"] = top.groupby("municipality").size()
    out = all_m.join(m).fillna({"in_top_500": 0}).reset_index()
    out["in_top_500"] = out.in_top_500.astype(int)
    out.to_csv(TABLE_DIR / "municipality_priority_summary.csv", index=False)

    sens = pd.DataFrame([{
        "susceptibility_model": model, "model_mean_roc_auc": perf.iloc[0], "n_seeds_averaged": n_seeds,
        "n_active_springs_ranked": len(act),
        "n_active_springs_missing_households": int(((df.target_dried == 0) & df.hh_dependent_total.isna()).sum()),
        "spearman_SPI_vs_susceptibility": stats.spearmanr(act.SPI, act.susceptibility).statistic,
        "spearman_SPI_vs_exposure": stats.spearmanr(act.SPI, act.hh_dependent_total).statistic,
        "top500_overlap_with_susceptibility_only_top500":
            len(set(act.index[:top_n]) & set(act.sort_values("susceptibility", ascending=False).index[:top_n])) / top_n,
    }])
    sens.to_csv(METRICS_DIR / "priority_index_sensitivity.csv", index=False)
    logger.info("\n" + sens.T.to_string())
    return out, sens


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_prioritization()
