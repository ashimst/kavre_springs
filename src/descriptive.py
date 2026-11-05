"""
Descriptive analyses of the dried-only survey fields (never used as model inputs).

1. Reported year of drying (Bikram Sambat, BS; BS - 57 ~ AD for mid-April to
   December) and respondent-attributed reason of drying. 2072 BS runs from
   mid-April 2015 to mid-April 2016 and contains the 25 April 2015 Gorkha
   earthquake (12 Baisakh 2072 BS).
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.features import PRIMARY_FEATURE_SET
from src.graph_builder import load_master

logger = logging.getLogger("descriptive")
METRICS_DIR = PROJECT_ROOT / "results/metrics"
FIG_DIR = PROJECT_ROOT / "results/figures"
PRED_PATH = PROJECT_ROOT / "results/predictions/test_predictions_main.parquet"
GORKHA_BS = 2072


def drying_years(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.target_dried == 1].copy()
    d["year_bs"] = pd.to_numeric(d["years_since_dried"], errors="coerce")
    d.loc[(d.year_bs < 2000) | (d.year_bs > 2085), "year_bs"] = np.nan  # e.g. typos such as 20720
    d["earthquake_attributed"] = d["reason_of_drying"].eq("earthquake")
    return d


def cause_specific_auc(df: pd.DataFrame, models=("Logistic Regression", "HydroBoost-GAT", "CatBoost"), n_boot=2000):
    preds = pd.read_parquet(PRED_PATH)
    preds = preds[(preds.buffer_m == 0) & (preds.feature_set == PRIMARY_FEATURE_SET)]
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


def study_area_map(df: pd.DataFrame):
    """Springs, municipality boundaries (OpenStreetMap, ODbL) and the 5 x 5 spatial-CV grid. No web tiles."""
    import geopandas as gpd
    from src.graph_builder import coords_to_metric
    district = gpd.read_file(PROJECT_ROOT / "data/processed/kavre_district_boundary.geojson")
    munis = gpd.read_file(PROJECT_ROOT / "data/processed/kavre_municipalities.geojson")
    fig, ax = plt.subplots(figsize=(8, 7.5))
    district.boundary.plot(ax=ax, color="k", lw=1.2, ls="--")
    munis = munis[munis.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]  # OSM stores Bethanchowk as a point
    munis.plot(ax=ax, color="#f2f2f2", edgecolor="#7f7f7f", lw=0.8)
    clean = {"Bethanchownk": "Bethanchowk", "Namobuudha": "Namobuddha", "Roshi RM": "Roshi"}
    for m, g in df.groupby("municipality"):
        ax.annotate(clean.get(m, m), (g.longitude.median(), g.latitude.median()), fontsize=8, ha="center",
                    color="#222222", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
    act, dry = df[df.target_dried == 0], df[df.target_dried == 1]
    ax.scatter(act.longitude, act.latitude, s=3, c="#4C72B0", alpha=0.6, linewidths=0, label=f"Active (n={len(act):,})")
    ax.scatter(dry.longitude, dry.latitude, s=4, c="#C44E52", alpha=0.8, linewidths=0, label=f"Dried (n={len(dry):,})")
    # 5 x 5 spatial-CV grid (same construction as src.spatial_splits, drawn in lon/lat)
    lat, lon = df.latitude.to_numpy(), df.longitude.to_numpy()
    for v in np.linspace(lon.min(), lon.max(), 6):
        ax.plot([v, v], [lat.min(), lat.max()], c="#999999", lw=0.5, ls=":")
    for v in np.linspace(lat.min(), lat.max(), 6):
        ax.plot([lon.min(), lon.max()], [v, v], c="#999999", lw=0.5, ls=":")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect(1 / np.cos(np.radians(27.57)))
    ax.legend(loc="lower left", fontsize=8, markerscale=3, title="Water sources", title_fontsize=8)
    ax.set_title("Kavre spring inventory, municipalities and spatial-CV grid (~8 km cells)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "study_area_map.png", dpi=300)
    plt.close(fig)


def run_descriptive():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_master()
    study_area_map(df)
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2, 1]})
    y = yrs[yrs.year_bs >= 2050]
    axes[0].bar(y.year_bs, y.n_dried - y.n_earthquake_attributed, color="#8C8C8C", label="other / unspecified cause")
    axes[0].bar(y.year_bs, y.n_earthquake_attributed, bottom=y.n_dried - y.n_earthquake_attributed, color="#C44E52",
                label="respondent-attributed to earthquake")
    axes[0].axvline(GORKHA_BS, ls=":", c="k", lw=0.8)
    axes[0].annotate("2072 BS (2015/16):\nGorkha earthquake", (GORKHA_BS, y.n_dried.max() * 0.9), xytext=(8, 0),
                     textcoords="offset points", fontsize=8)
    axes[0].set_xlabel("Reported year of drying (Bikram Sambat)")
    axes[0].set_ylabel("Dried sources")
    axes[0].set_title("(a) Reported drying year (heaping at 2060/2065/2070 suggests recall rounding)", fontsize=9)
    axes[0].legend(fontsize=8)
    c = cs[cs.model == "Logistic Regression"].reset_index(drop=True)
    axes[1].errorbar(c.roc_auc, range(len(c))[::-1], xerr=[c.roc_auc - c.ci_low, c.ci_high - c.roc_auc], fmt="o", capsize=3)
    axes[1].set_yticks(range(len(c))[::-1])
    axes[1].set_yticklabels([f"{s}\n(n={n})" for s, n in zip(c.dried_subset, c.n_dried)], fontsize=8)
    axes[1].axvline(0.5, ls=":", c="grey")
    axes[1].set_xlabel("Out-of-fold ROC-AUC vs active springs")
    axes[1].set_title("(b) Discrimination by reported cause\n(logistic regression, spatial CV)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "drying_timing_and_cause.png", dpi=300)
    plt.close(fig)
    return summary, cs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_descriptive()
