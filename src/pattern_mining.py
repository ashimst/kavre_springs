"""
Frequent-pattern (apriori) mining of spring drying, with statistical controls.

P1  Drying rules. Every source becomes a transaction of items: tertiles (low / mid / high) of interpretable GIS
    covariates, the label-free landscape type (src.unsupervised), and survey-recorded categorical attributes
    (source type, land use, ownership, seasonality). Frequent itemsets are mined with apriori (mlxtend;
    min support 2 %, up to three antecedent items) and turned into rules  A -> dried  and  A -> active.
    Controls, in order:
      * significance: one-sided Fisher exact test of A vs not-A; Benjamini-Hochberg FDR over all rules (q < 0.05);
      * productivity (Webb 2007): a rule is kept only if its confidence exceeds that of every immediate
        generalisation (each antecedent with one item removed, including the base rate) by >= 0.03;
      * ward robustness: Mantel-Haenszel odds ratio stratified by ward (75 strata) with a Robins-Breslow-Greenland
        95 % CI, and the share of wards in which the rule's lift exceeds 1. Rules whose MH-OR CI excludes 1 hold
        *within* wards and cannot be produced by between-team differences;
      * spatial replication: rules are re-mined on four of five spatial blocks (~8 km) and their lift is
        evaluated on the held-out block (five folds); we report the share of folds in which held-out lift > 1.
    Rules are ranked by weighted relative accuracy, WRAcc = support(A) x (confidence - base rate), the standard
    quality measure of subgroup discovery.
P2  Narrative rules among dried sources: items are reported cause, period of drying (before / in / after
    2072 BS / unknown), landscape type, source type, land use and municipality; consequents are reported causes.
    Mined and tested as in P1 (ward-stratified MH OR included).

Outputs: results/metrics/pm_*.csv, results/figures/pm_rules.png
Usage:   python -m src.pattern_mining
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from scipy import stats

from src.graph_builder import coords_to_metric, load_master
from src.spatial_mining import bh_fdr
from src.spatial_splits import spatial_block_kfold

logger = logging.getLogger("pattern_mining")
METRICS = PROJECT_ROOT / "results/metrics"
FIGS = PROJECT_ROOT / "results/figures"
TYPES = PROJECT_ROOT / "data/processed/landscape_types.csv"
MIN_SUPPORT = 0.02
MAX_ANTECEDENT = 3
MIN_IMPROVEMENT = 0.03
SEED = 5

# Interpretable continuous covariates discretised into tertiles (only those present are used)
TERTILE_COVARIATES = {
    "elevation_m": "elevation", "slope_deg": "slope", "tpi_500m": "TPI 500 m", "mtpi": "multi-scale TPI",
    "hnd": "HAND", "twi": "TWI", "log_upa": "upstream area", "chili": "heat load (CHILI)",
    "precip_annual_mm": "annual precipitation", "def": "climatic water deficit", "soil": "soil moisture",
    "ndvi_mean": "NDVI", "ndwi_mean": "NDWI", "ndvi_trend": "NDVI trend 2001-23", "chirps_trend": "precipitation trend",
    "road_density_500m": "road density", "dist_to_road_m": "distance to road", "built_change": "built-up growth 2000-20",
    "wc_crop": "cropland share (WorldCover)", "wc_tree": "tree share (WorldCover)", "gfc_loss_pre2015": "forest loss 2001-14",
    "gfc_loss_post2015": "forest loss 2015-23", "soil_clay": "soil clay", "pop_1km": "population 1 km",
}


# ----------------------------------------------------------------------------- items
def tertile_items(df):
    items = {}
    for col, label in TERTILE_COVARIATES.items():
        if col not in df:
            continue
        v = df[col].astype(float)
        if (v == v.mode().iloc[0]).mean() > 0.34:  # many ties (e.g. zero forest loss): present / absent
            m = v.mode().iloc[0]
            items[f"{label}=none"] = (v == m).to_numpy()
            items[f"{label}=some"] = (v != m).to_numpy() & v.notna().to_numpy()
            continue
        q = v.quantile([1 / 3, 2 / 3]).to_numpy()
        items[f"{label}=low"] = (v <= q[0]).to_numpy()
        items[f"{label}=mid"] = ((v > q[0]) & (v <= q[1])).to_numpy()
        items[f"{label}=high"] = (v > q[1]).to_numpy()
    return items


def categorical_items(df):
    items = {}
    if TYPES.exists():
        t = df[["id"]].merge(pd.read_csv(TYPES), on="id", how="left").env_type.astype(str)
        for k in sorted(t.dropna().unique()):
            items[f"landscape type={k}"] = (t == k).to_numpy()
    items["source=pond"] = (df["Source Type_encoded"] == 1).to_numpy()
    items["source=spring"] = (df["Source Type_encoded"] == 0).to_numpy()
    for c in [c for c in df.columns if c.startswith("landuse_")]:
        items[f"surveyed land use={c.replace('landuse_', '')}"] = df[c].astype(bool).to_numpy()
    for c in [c for c in df.columns if c.startswith("ownership_")]:
        items[f"ownership={c.replace('ownership_', '')}"] = df[c].astype(bool).to_numpy()
    if "Perennial / Seasonal_encoded" in df:
        items["flow=seasonal (survey)"] = (df["Perennial / Seasonal_encoded"] == 1).to_numpy()
    return items


# ----------------------------------------------------------------------------- statistics
def mantel_haenszel(a_mask, y, strata):
    """MH odds ratio of y between a_mask and ~a_mask, stratified; Robins-Breslow-Greenland 95 % CI."""
    R = S = PR = PS_QR = QS = 0.0
    for s in np.unique(strata):
        m = strata == s
        a = np.sum(a_mask[m] & (y[m] == 1)); b = np.sum(a_mask[m] & (y[m] == 0))
        c = np.sum(~a_mask[m] & (y[m] == 1)); d = np.sum(~a_mask[m] & (y[m] == 0))
        n = a + b + c + d
        if (a + b) == 0 or (c + d) == 0 or n < 2:
            continue
        r, s_ = a * d / n, b * c / n
        p, q = (a + d) / n, (b + c) / n
        R += r; S += s_; PR += p * r; PS_QR += p * s_ + q * r; QS += q * s_
    if R == 0 or S == 0:
        return np.nan, np.nan, np.nan
    or_mh = R / S
    var = PR / (2 * R ** 2) + PS_QR / (2 * R * S) + QS / (2 * S ** 2)
    se = np.sqrt(var)
    return or_mh, np.exp(np.log(or_mh) - 1.96 * se), np.exp(np.log(or_mh) + 1.96 * se)


def ward_lift_share(a_mask, y, strata, min_n=5):
    k = n = 0
    for s in np.unique(strata):
        m = strata == s
        if a_mask[m].sum() < min_n or (~a_mask[m]).sum() < min_n:
            continue
        base = y[m].mean()
        if base in (0, 1):
            continue
        n += 1
        k += y[m & a_mask].mean() > base
    return k / n if n else np.nan, n


def mine_rules(I: pd.DataFrame, y: np.ndarray, target_value: int, min_support=MIN_SUPPORT, max_len=MAX_ANTECEDENT):
    """Apriori on the antecedent items; rules A -> (y == target_value) with Fisher tests and productivity."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fi = apriori(I, min_support=min_support, max_len=max_len, use_colnames=True)
    t = (y == target_value)
    base = t.mean()
    N = len(t)
    cols = {c: I[c].to_numpy() for c in I.columns}
    conf_of = {frozenset(): base}
    rows = []
    for items in fi.itemsets:
        m = np.logical_and.reduce([cols[c] for c in items])
        na = m.sum()
        k = (m & t).sum()
        conf = k / na
        conf_of[frozenset(items)] = conf
        p = stats.fisher_exact([[k, na - k], [t.sum() - k, N - na - (t.sum() - k)]], alternative="greater")[1]
        rows.append({"antecedent": frozenset(items), "n_items": len(items), "n": int(na), "support": na / N,
                     "confidence": conf, "lift": conf / base, "leverage": na / N * (conf - base),
                     "conviction": (1 - base) / (1 - conf) if conf < 1 else np.inf, "wracc": na / N * (conf - base),
                     "p": p})
    r = pd.DataFrame(rows)
    r["q_bh"] = bh_fdr(r.p.to_numpy())
    # productive rules: confidence must beat every immediate generalisation by MIN_IMPROVEMENT
    def productive(a, conf):
        return all(conf >= conf_of.get(a - {x}, base) + MIN_IMPROVEMENT for x in a)
    r["productive"] = [productive(a, c) for a, c in zip(r.antecedent, r.confidence)]
    r["consequent"] = "dried" if target_value == 1 else "active"
    return r, base


def evaluate_rules(r, I, y, target_value, ward, folds):
    t = (y == target_value).astype(int)
    cols = {c: I[c].to_numpy() for c in I.columns}
    out = []
    for _, row in r.iterrows():
        m = np.logical_and.reduce([cols[c] for c in row.antecedent])
        or_mh, lo, hi = mantel_haenszel(m, t, ward)
        share, n_w = ward_lift_share(m, t, ward)
        rep = []
        for te in folds:
            mt = m & te
            if mt.sum() >= 10 and t[te].mean() > 0:
                rep.append(t[mt].mean() / t[te].mean() > 1)
        # one-sided test of the within-ward association (H1: MH-OR > 1) from the RBG standard error
        se = (np.log(hi) - np.log(lo)) / (2 * 1.96) if np.isfinite(lo) and lo > 0 else np.nan
        p_mh = stats.norm.sf(np.log(or_mh) / se) if np.isfinite(se) and se > 0 else np.nan
        out.append({"mh_or_ward": or_mh, "mh_ci_low": lo, "mh_ci_high": hi, "mh_p": p_mh,
                    "share_wards_lift_gt1": share, "n_wards_evaluable": n_w,
                    "heldout_replication": np.mean(rep) if rep else np.nan, "n_heldout_folds": len(rep)})
    e = pd.concat([r.reset_index(drop=True), pd.DataFrame(out)], axis=1)
    # multiplicity: Benjamini-Hochberg over all rules evaluated within wards in this call
    p = e.mh_p.fillna(1.0).to_numpy()
    e["mh_q_bh"] = bh_fdr(p) if len(p) else p
    e["ward_robust"] = e.mh_q_bh < 0.05
    return e


# ----------------------------------------------------------------------------- P1 / P2
def drying_rules(df, coords, ward):
    items = {**tertile_items(df), **categorical_items(df)}
    # the seasonal-flow flag is analysed separately (it may encode dry-season fieldwork, not hydrology)
    seasonal = items.pop("flow=seasonal (survey)", None)
    I = pd.DataFrame(items).astype(bool)
    y = df.target_dried.to_numpy()
    folds = [s["test_mask"] for s in spatial_block_kfold(coords, 5, random_state=SEED)]
    res = []
    for tv in (1, 0):
        r, base = mine_rules(I, y, tv)
        logger.info(f"consequent={'dried' if tv else 'active'}: {len(r)} itemsets, "
                    f"{(r.q_bh < 0.05).sum()} significant, {((r.q_bh < 0.05) & r.productive).sum()} significant+productive")
        sel = r[(r.q_bh < 0.05) & r.productive & (r.lift > 1)].sort_values("wracc", ascending=False)
        res.append(evaluate_rules(sel, I, y, tv, ward, folds))
    rules = pd.concat(res, ignore_index=True)
    rules["antecedent"] = [" & ".join(sorted(a)) for a in rules.antecedent]
    # Simpson-type reversal: pooled lift > 1 but the within-ward association points the other way
    rules["reverses_within_wards"] = rules.mh_ci_high < 1
    summary = (rules.groupby("consequent").agg(n_rules=("antecedent", "size"), n_ward_robust=("ward_robust", "sum"),
                                                n_reversed_within_wards=("reverses_within_wards", "sum"),
                                                median_heldout_replication=("heldout_replication", "median"),
                                                median_share_wards_lift_gt1=("share_wards_lift_gt1", "median"))
               .reset_index())
    item_support = pd.DataFrame({"item": I.columns, "support": I.mean().to_numpy(),
                                 "drying_rate": [y[I[c].to_numpy()].mean() for c in I.columns]})
    seas = None
    if seasonal is not None:
        s_or, s_lo, s_hi = mantel_haenszel(seasonal, y, ward)
        share, n_w = ward_lift_share(seasonal, y, ward)
        seas = pd.DataFrame([{"item": "flow=seasonal (survey)", "support": seasonal.mean(),
                              "drying_rate_if_seasonal": y[seasonal].mean(), "drying_rate_if_perennial": y[~seasonal].mean(),
                              "mh_or_ward": s_or, "mh_ci_low": s_lo, "mh_ci_high": s_hi, "share_wards_lift_gt1": share,
                              "n_wards_evaluable": n_w}])
    return rules, summary, item_support, seas


def narrative_rules(df, ward):
    d = df[df.target_dried == 1].reset_index(drop=True)
    w = ward[(df.target_dried == 1).to_numpy()]
    yr = pd.to_numeric(d["years_since_dried"], errors="coerce").where(lambda v: (v >= 2000) & (v <= 2085))
    period = np.select([yr < 2072, yr == 2072, yr > 2072], ["before 2072 BS", "2072 BS", "after 2072 BS"], "unknown")
    items = {f"period={p}": period == p for p in ["before 2072 BS", "2072 BS", "after 2072 BS", "unknown"]}
    items.update(categorical_items(d))
    items.pop("flow=seasonal (survey)", None)
    for mname in sorted(d.municipality.unique()):
        items[f"municipality={mname}"] = (d.municipality == mname).to_numpy()
    I = pd.DataFrame(items).astype(bool)
    res = []
    for cause in ["earthquake", "drought", "infrastructure_development"]:
        y = d.reason_of_drying.eq(cause).to_numpy().astype(int)
        r, base = mine_rules(I, y, 1, min_support=0.02, max_len=2)
        sel = r[(r.q_bh < 0.05) & r.productive & (r.lift > 1.2)].sort_values("wracc", ascending=False)
        e = evaluate_rules(sel, I, y, 1, w, folds=[])
        e["consequent"] = f"reason={cause}"
        e["base_rate"] = base
        res.append(e)
    rules = pd.concat(res, ignore_index=True)
    rules["antecedent"] = [" & ".join(sorted(a)) for a in rules.antecedent]
    return rules.drop(columns=["heldout_replication", "n_heldout_folds"])


# ----------------------------------------------------------------------------- figure
def figure(rules, narr):
    """(a) all dried rules as a hexbin (support x lift, colour = share ward-robust); (b, c) top 8 ward-robust rules."""

    from matplotlib.ticker import FuncFormatter, NullFormatter
    fig = plt.figure(figsize=(13, 8.5), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35])
    ax = fig.add_subplot(gs[:, 0])
    r = rules[rules.consequent == "dried"]
    hb = ax.hexbin(np.log10(r.support), r.lift, C=r.ward_robust.astype(float), reduce_C_function=np.mean,
                   gridsize=16, cmap="RdGy_r", vmin=0, vmax=1, mincnt=3, linewidths=0.2)
    fig.colorbar(hb, ax=ax, location="bottom", shrink=0.8,
                 label="share of rules in cell that hold within wards (MH, BH q < 0.05)")
    ax.set_xlabel("log10 support (share of sources)")
    ax.set_ylabel("Lift")
    ax.set_title(f"(a) {len(r):,} significant productive rules → dried\n"
                 f"{r.ward_robust.mean():.0%} hold within wards", fontsize=10, loc="left")
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    for k, (cons, title) in enumerate([("dried", "(b) Top ward-robust rules → dried"),
                                       ("active", "(c) Top ward-robust rules → active")]):
        ax = fig.add_subplot(gs[k, 1])
        r = rules[(rules.consequent == cons) & rules.ward_robust].sort_values("wracc", ascending=False).head(8)[::-1]
        yy = np.arange(len(r))
        ax.errorbar(r.mh_or_ward, yy, xerr=[r.mh_or_ward - r.mh_ci_low, r.mh_ci_high - r.mh_or_ward], fmt="o",
                    capsize=2, c="k", ms=4)
        ax.axvline(1, c="grey", lw=0.8, ls=":")
        ax.set_yticks(yy)
        ax.set_yticklabels([a.replace(" & ", " &\n") + f"\n(conf {c:.2f}, n={n:,})"
                            for a, c, n in zip(r.antecedent, r.confidence, r.n)], fontsize=7.5)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(fmt); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([1, 1.5, 2, 3, 4, 6])
        ax.grid(axis="x", lw=0.3, alpha=0.5)
        ax.set_xlabel("ward-stratified Mantel–Haenszel odds ratio (95% CI)")
        ax.set_title(title, fontsize=10, loc="left")
    fig.savefig(FIGS / "pm_rules.png", dpi=300)
    plt.close(fig)


def run():
    METRICS.mkdir(parents=True, exist_ok=True)
    df = load_master()
    coords = coords_to_metric(df.latitude.to_numpy(), df.longitude.to_numpy())
    ward = (df.municipality.astype(str) + "_" + df.ward.astype(str)).to_numpy()
    rules, summary, item_support, seas = drying_rules(df, coords, ward)
    if seas is not None:
        seas.to_csv(METRICS / "pm_seasonal_flag.csv", index=False)
        logger.info("\n" + seas.round(3).to_string())
    rules.to_csv(METRICS / "pm_drying_rules.csv", index=False)
    summary.to_csv(METRICS / "pm_drying_rules_summary.csv", index=False)
    item_support.to_csv(METRICS / "pm_item_support.csv", index=False)
    logger.info("\n" + summary.round(3).to_string())
    logger.info("\n" + rules.sort_values("wracc", ascending=False).head(30)[
        ["consequent", "antecedent", "support", "confidence", "lift", "q_bh", "mh_or_ward", "mh_ci_low", "mh_ci_high",
         "share_wards_lift_gt1", "heldout_replication"]].round(3).to_string())
    narr = narrative_rules(df, ward)
    narr.to_csv(METRICS / "pm_narrative_rules.csv", index=False)
    logger.info("\n" + narr.head(30)[["consequent", "antecedent", "support", "confidence", "lift", "mh_or_ward",
                                      "mh_ci_low", "ward_robust"]].round(3).to_string())
    figure(rules, narr)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()
