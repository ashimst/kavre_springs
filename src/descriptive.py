"""
Descriptive analyses of the dried-only survey fields (never used as model inputs).

1. The ambiguously worded reported-year field is operationally treated as a
   Bikram Sambat (BS) calendar year when values fall in the 2000--2085 range;
   this interpretation is not verified by questionnaire documentation or
   visit dates. 2072 BS runs from mid-April 2015 to mid-April 2016.
2. Discrimination by cause: out-of-fold predicted probabilities from the main
   benchmark (spatial CV, 0 m buffer, averaged over seeds) are used to compute
   ROC-AUC of dried-vs-active separately for dried sources attributed to the
   earthquake and to other causes. If earthquake-related drying is less
   predictable from environmental covariates, its AUC should be lower.
   Confidence intervals: 2,000 bootstrap resamples of springs.
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
from sklearn.metrics import roc_auc_score

from src.features import PRIMARY_FEATURE_SET
from src.dataset import load_master

logger = logging.getLogger("descriptive")
METRICS_DIR = PROJECT_ROOT / "results/metrics"
FIG_DIR = PROJECT_ROOT / "results/figures"
PRED_PATH = PROJECT_ROOT / "results/predictions/test_predictions_main.parquet"
GORKHA_BS = 2072


def drying_years(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.target_dried == 1].copy()
    d["year_bs"] = pd.to_numeric(d["years_since_dried"], errors="coerce")
    # Treat plausible calendar-year values as BS years; this operational coding is unverified.
    d.loc[(d.year_bs < 2000) | (d.year_bs > 2085), "year_bs"] = np.nan
    d["earthquake_attributed"] = d["reason_of_drying"].eq("earthquake")
    return d


def cause_specific_auc(df: pd.DataFrame, models=("Logistic Regression", "CatBoost"), n_boot=2000):
    preds = pd.read_parquet(PRED_PATH)
    # leak-free, season-free env_hydro predictions (main profile) when available (the seasonal-flow flag may be
    # confounded by dry-season fieldwork); otherwise fall back to the configured primary set
    preds = preds[preds.buffer_m == 0]
    fs = next((f for f in ["env_hydro", PRIMARY_FEATURE_SET, "env"] if (preds.feature_set == f).any()), PRIMARY_FEATURE_SET)
    preds = preds[preds.feature_set == fs]
    rng = np.random.default_rng(0)
    rows = []
    reason = df.set_index("id")["reason_of_drying"]
    for m in models:
        p = preds[preds.model == m].groupby("id").agg(p=("p", "mean"), y=("y", "first"))
        p["reason"] = reason.reindex(p.index)
        active = p[p.y == 0]
        for label, mask in [("all dried", p.y == 1),
                            ("earthquake-attributed", (p.y == 1) & p.reason.eq("earthquake")),
                            ("other causes", (p.y == 1) & ~p.reason.eq("earthquake"))]:
            pos = p[mask]
            yy = np.r_[np.ones(len(pos)), np.zeros(len(active))]
            pp = np.r_[pos.p.to_numpy(), active.p.to_numpy()]
            auc = roc_auc_score(yy, pp)
            boots = []
            for _ in range(n_boot):
                ip = rng.integers(0, len(pos), len(pos))
                ia = rng.integers(0, len(active), len(active))
                boots.append(roc_auc_score(yy, np.r_[pos.p.to_numpy()[ip], active.p.to_numpy()[ia]]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"model": m, "dried_subset": label, "n_dried": len(pos), "n_active": len(active),
                         "roc_auc": auc, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def run_descriptive():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_master()
    d = drying_years(df)
    yrs = (d.groupby("year_bs").agg(n_dried=("id", "size"), n_earthquake_attributed=("earthquake_attributed", "sum"))
           .reset_index())
    yrs["approx_year_ad"] = yrs.year_bs - 57
    yrs.to_csv(METRICS_DIR / "drying_year_distribution.csv", index=False)
    reasons = d.reason_of_drying.value_counts().rename_axis("reported_reason").reset_index(name="n_dried")
    reasons["share"] = reasons.n_dried / len(d)
    reasons.to_csv(METRICS_DIR / "drying_reasons.csv", index=False)
    in_gorkha = d.year_bs == GORKHA_BS
    summary = pd.DataFrame([{
        "n_dried": len(d), "n_valid_year": int(d.year_bs.notna().sum()),
        "n_dried_in_2072BS": int(in_gorkha.sum()), "share_dried_in_2072BS": float(in_gorkha.mean()),
        "share_earthquake_attributed_among_2072BS": float(d.earthquake_attributed[in_gorkha].mean()),
        "n_earthquake_attributed": int(d.earthquake_attributed.sum()),
        "share_of_earthquake_attributed_in_2072BS": float(in_gorkha[d.earthquake_attributed].mean()),
        "share_dried_before_2072BS": float((d.year_bs < GORKHA_BS).mean()),
        "share_dried_after_2072BS": float((d.year_bs > GORKHA_BS).mean()),
    }])
    summary.to_csv(METRICS_DIR / "drying_timing_summary.csv", index=False)
    logger.info("\n" + summary.T.to_string())

    cs = cause_specific_auc(df)
    cs.to_csv(METRICS_DIR / "cause_specific_auc.csv", index=False)
    logger.info("\n" + cs.round(3).to_string())

    return summary, cs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_descriptive()
