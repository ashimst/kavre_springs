"""
Every main figure of the paper, drawn from the versioned result tables (results/metrics) and the analysis table.
No analysis is run here, so the figures can be restyled without recomputing anything.

  fig1_study_area        official palikas and the 5,689 sources
  fig2_information       enrichment ladder and distance decay of leak-free skill
  fig3_landscape_types   label-free landscape types: profiles, drying, map
  fig4_rules             apriori rules: robustness within wards, top ward-robust rules
  fig5_wards_watersheds  homogeneity of drying by unit vs size-matched nulls; pairs; funnel
  fig6_recalled          reported years and causes; ward vs landscape ICC; Knox within vs across wards
  fig7_drivers           within-ward drivers, three-way partition, earthquake exposure under adjustment
  fig8_robustness        ward signal across populations against grid and random-region nulls
  figS1_pair_bootstrap   bootstrap distribution of the same-ward pair term, all sources and springs only
  fig9_ward_rates        recorded drying rate by recorded ward; agreement of recorded wards with OSM ward polygons

Usage:  python -m src.figures            (writes results/figures/fig*.png)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.dataset import load_master
from src.features import FEATURE_SETS
from src.plotstyle import C, DIV, SEQ, apply, map_axes, panel_label, save

M = PROJECT_ROOT / "results/metrics"
F = PROJECT_ROOT / "results/figures"
ASPECT_LAT = 27.57

# short names of the landscape types (profiles in Table S8; T-numbers are fixed by the seeded mixture)
TYPE_NAMES = {"T1": "Peri-urban, built-up", "T2": "Low cultivated flats", "T3": "Dry low-rainfall slopes",
              "T4": "Valley-bottom corridors", "T5": "Steep, water-limited, high above drainage",
              "T6": "Moist cultivated slopes", "T7": "Forested steep upper slopes", "T8": "High, wet forest"}
NICE = {"elevation_m": "elevation", "slope_deg": "slope", "precip_annual_mm": "annual rainfall", "ndvi_mean": "NDVI",
        "aspect_sin": "aspect (sine)", "aspect_cos": "aspect (cosine)",
        "log_upa": "upstream area (log)", "hnd": "height above drainage", "twi": "wetness index", "tpi_500m": "TPI 500 m",
        "def": "climatic water deficit", "soil": "soil moisture", "aet": "actual ET", "dist_to_road_m": "distance to road",
        "road_density_500m": "road density", "wc_tree": "tree cover", "wc_crop": "cropland", "wc_built": "built-up",
        "soil_soc": "soil organic carbon", "soil_bdod": "soil bulk density", "pop_1km": "population",
        "nightlights": "night lights", "landform_position": "slope position (ALOS)", "chili": "heat load (CHILI)",
        "mtpi": "multi-scale TPI", "chirps_trend": "rainfall trend", "gfc_treecover2000": "tree cover 2000",
        "built_2020": "built-up 2020", "et_mean": "ET (MODIS)", "pdsi": "PDSI", "topo_diversity": "topographic diversity",
        "ndwi_mean": "NDWI", "built_change": "built-up change", "ndvi_trend": "NDVI trend",
        "wc_grass": "grassland", "wc_water": "open water (WorldCover)", "wc_shrub": "shrubland", "soil_clay": "soil clay", "soil_sand": "soil sand",
        "landuse_agriculture": "agriculture (survey)", "landuse_dense_forest": "dense forest (survey)",
        "precip_dry_season_mm": "dry-season rainfall", "Source Type_encoded": "pond (survey)",
        "spring_density_500m": "source density", "upa": "upstream area", "elv": "hydro-conditioned elevation"}


def nice(c):
    return NICE.get(c, c.replace("_", " "))


def palika_layers():
    import geopandas as gpd
    pal = gpd.read_file(PROJECT_ROOT / "data/processed/boundaries/kavre_palikas.geojson")
    dist = gpd.read_file(PROJECT_ROOT / "data/processed/boundaries/kavre_district.geojson")
    return dist, pal


# ----------------------------------------------------------------------------- fig 1
def fig1_study_area(df):
    dist, pal = palika_layers()
    sv = pal[pal.surveyed]
    fig, (ax0, ax) = plt.subplots(1, 2, figsize=(7.2, 3.9), gridspec_kw={"width_ratios": [0.82, 1.45]})
    pal[~pal.surveyed].plot(ax=ax0, color="#ffffff", edgecolor="#c9c8c1", lw=0.4)
    sv.plot(ax=ax0, color="#e9e8e3", edgecolor="#8f8e88", lw=0.5)
    dist.boundary.plot(ax=ax0, color=C["ink2"], lw=0.8)
    for _, r in pal.iterrows():
        pt = r.geometry.representative_point()
        ax0.annotate(r["name"], (pt.x, pt.y), fontsize=5, ha="center", va="center",
                     color=C["ink"] if r.surveyed else C["muted"])
    map_axes(ax0)
    panel_label(ax0, "a", "Kavrepalanchok: 13 palikas, 7 surveyed")

    pal[~pal.surveyed].plot(ax=ax, color="#ffffff", edgecolor="#d9d8d2", lw=0.4)
    sv.plot(ax=ax, color="#f5f5f2", edgecolor="#8f8e88", lw=0.6)
    act, dry = df[df.target_dried == 0], df[df.target_dried == 1]
    ax.scatter(act.longitude, act.latitude, s=1.1, c=C["active"], alpha=0.55, linewidths=0, rasterized=True)
    ax.scatter(dry.longitude, dry.latitude, s=1.6, c=C["dried"], alpha=0.85, linewidths=0, rasterized=True)
    for _, r in sv.iterrows():
        pt = r.geometry.representative_point()
        ax.annotate(r["name"], (pt.x, pt.y), fontsize=6.5, ha="center", color=C["ink"],
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    xmin, ymin, xmax, ymax = sv.total_bounds
    ax.set_xlim(xmin - 0.01, xmax + 0.01); ax.set_ylim(ymin - 0.01, ymax + 0.01)
    map_axes(ax)
    # scale bar (5 km)
    km = 5 / (111.32 * np.cos(np.radians(ASPECT_LAT)))
    x0, y0 = xmin + 0.012, ymin + 0.008
    ax.plot([x0, x0 + km], [y0, y0], c=C["ink"], lw=1.5, solid_capstyle="butt")
    ax.text(x0 + km / 2, y0 + 0.006, "5 km", ha="center", fontsize=6.5, color=C["ink2"])
    ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=3.5, color=C["active"], label=f"active  {len(act):,}"),
                       Line2D([], [], ls="", marker="o", ms=3.5, color=C["dried"], label=f"dried  {len(dry):,}")],
              loc="upper left", fontsize=7, handletextpad=0.2)
    panel_label(ax, "b", "5,689 mapped water sources")
    fig.text(0.0, -0.01, "Boundaries: MoFAGA via geoBoundaries (CC BY-SA 4.0)", fontsize=5.5, color=C["muted"])
    fig.tight_layout(w_pad=0.6)
    save(fig, F / "fig1_study_area.png")


# ----------------------------------------------------------------------------- fig 2
def fig2_information():
    folds = pd.concat([pd.read_csv(p) for p in sorted(M.glob("benchmark_*_folds.csv")) if "_quick" not in p.name])
    models = [("Random Forest", C["blue"]), ("CatBoost", C["orange"]), ("Logistic Regression", C["aqua"])]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})

    ladder = [("env", "main", "base survey and\nsurface covariates"),
              ("env_hydro", "main", "+ hydrology, water\nbalance, roads"),
              ("env_rich", "enriched", "+ 28 Earth Engine\nenrichment layers")]
    xs = np.arange(len(ladder))
    for k, (m, col) in enumerate(models):
        vals = []
        for fs, prof, _ in ladder:
            d = folds[(folds.profile == prof) & (folds.feature_set == fs) & (folds.buffer_m == 0) & (folds.model == m)]
            vals.append((d.roc_auc.mean(), d.roc_auc.std() / np.sqrt(len(d))))
        v = np.array(vals)
        ax.errorbar(xs + (k - 1) * 0.09, v[:, 0], yerr=v[:, 1], fmt="-o", color=col, ms=4, lw=1.4, capsize=0,
                    elinewidth=1.0, label=m.replace("Logistic Regression", "logistic regression").replace("Random Forest", "random forest"))
    knn = folds[(folds.profile == "main") & (folds.feature_set == "env_hydro") & (folds.buffer_m == 0)
                & (folds.model == "kNN label interpolation")].roc_auc.mean()
    ax.axhline(knn, color=C["muted"], ls=(0, (3, 2)), lw=0.9)
    ax.text(-0.3, knn + 0.003, "kNN interpolation of labels only", fontsize=6.5, color=C["ink2"], ha="left", va="bottom")
    # two reference models that lie above the plotted range; three decimals, as in the prediction table
    catboost = folds[folds.model == "CatBoost"]
    seasonal = catboost[catboost.feature_set == "env_hydro_season"].roc_auc.mean()
    ax.annotate(f"+ seasonal-flow flag: {seasonal:.3f}", xy=(2.3, 0.735), fontsize=6.5, color=C["ink2"], ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{lab}\n({len(FEATURE_SETS[fs])} covariates)" for fs, _, lab in ladder], fontsize=6.3)
    ax.set_ylabel("ROC-AUC, grid-cell-grouped CV")
    ax.set_ylim(0.58, 0.765)
    ax.set_xlim(-0.35, 2.35)
    ax.legend(loc="upper left", fontsize=6.5)
    panel_label(ax, "a", "Information added by each data layer")

    dist = pd.read_csv(M / "rb_validation_distances.csv")
    for m, col in models:
        pts = []
        for buf in [0.0, 500.0, 1000.0, 2000.0]:
            d = folds[(folds.feature_set == "env_hydro") & (folds.buffer_m == buf) & (folds.model == m)
                      & folds.profile.isin(["main", "buffers"])]
            dd = dist[(dist.design == "spatial block CV") & (dist.buffer_m == buf)].median_distance_m.iloc[0]
            pts.append((dd / 1000, d.roc_auc.mean(), d.roc_auc.std() / np.sqrt(len(d))))
        p = np.array(pts)
        bx.errorbar(p[:, 0], p[:, 1], yerr=p[:, 2], fmt="-o", color=col, ms=4, lw=1.4, elinewidth=1.0)
        lomo = folds[(folds.profile == "lomo") & (folds.feature_set == "env_hydro") & (folds.model == m)].roc_auc.mean()
        dl = dist[dist.design == "leave-one-municipality-out"].median_distance_m.iloc[0] / 1000
        bx.plot(dl, lomo, marker="D", ms=4.5, color=col, mec="white", mew=0.6, ls="")
    for buf, x in zip(["0", "0.5", "1", "2 km"], dist[dist.design == "spatial block CV"].sort_values("buffer_m").median_distance_m / 1000):
        bx.text(x, 0.700, f"buffer {buf}" if buf == "0" else buf, fontsize=6, color=C["muted"], ha="center")
    bx.text(dl, 0.618, "unseen\nmunicipality", fontsize=6, color=C["ink2"], ha="center", va="top")
    bx.set_xlabel("test source to nearest training source, median (km)")
    bx.set_ylabel("ROC-AUC")
    bx.set_ylim(0.60, 0.705)
    panel_label(bx, "b", "Skill decays with distance to training data")
    fig.tight_layout(w_pad=1.5)
    save(fig, F / "fig2_information.png")


# ----------------------------------------------------------------------------- fig 3
def fig3_landscape_types(df):
    rates = pd.read_csv(M / "un_typology_drying.csv").sort_values("drying_rate", ascending=False)
    prof = pd.read_csv(M / "un_typology_profiles.csv", index_col=0)
    order = rates.type.tolist()
    cols = ["wc_built", "nightlights", "road_density_500m", "wc_crop", "wc_tree", "slope_deg", "hnd", "tpi_500m",
            "landform_position", "twi", "log_upa", "def", "soil", "precip_annual_mm", "soil_soc", "elevation_m"]
    cols = [c for c in cols if c in prof.columns]
    norm = plt.Normalize(rates.drying_rate.min() - 0.05, rates.drying_rate.max())
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1, 0.95], hspace=0.45, wspace=0.08)

    # (a) drying by type
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(rates))[::-1]
    base = df.target_dried.mean()
    ax.axvline(base, color=C["muted"], lw=0.8, ls=(0, (3, 2)))
    ax.text(base + 0.004, -0.45, f"all sources {base:.0%}", fontsize=6, color=C["muted"], ha="left", va="center")
    for yi, (_, r) in zip(y, rates.iterrows()):
        col = SEQ(norm(r.drying_rate))
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=col, lw=2.4, solid_capstyle="round")
        ax.plot(r.drying_rate, yi, "o", color=col, ms=5.5, mec="white", mew=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t}  {TYPE_NAMES[t]}\n      n = {int(n):,}" for t, n in zip(rates.type, rates.n_sources)],
                       fontsize=6.1, linespacing=1.0)
    ax.set_xlim(0.08, 0.48)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("recorded as dried (95% CI)")
    ax.grid(axis="y", visible=False)
    panel_label(ax, "a", "Drying differs threefold between types")

    # (b) map
    bx = fig.add_subplot(gs[0, 1])
    t = df[["id", "longitude", "latitude"]].merge(pd.read_csv(PROJECT_ROOT / "data/processed/landscape_types.csv"), on="id")
    t["rate"] = t.env_type.map(rates.set_index("type").drying_rate)
    dist, pal = palika_layers()
    sv = pal[pal.surveyed]
    sv.plot(ax=bx, color="#f5f5f2", edgecolor="#b9b8b0", lw=0.5)
    o = np.argsort(t.rate.to_numpy())
    sc = bx.scatter(t.longitude.to_numpy()[o], t.latitude.to_numpy()[o], c=t.rate.to_numpy()[o], cmap=SEQ, norm=norm,
                    s=1.3, linewidths=0, rasterized=True)
    xmin, ymin, xmax, ymax = sv.total_bounds
    bx.set_xlim(xmin, xmax); bx.set_ylim(ymin, ymax)
    map_axes(bx)
    cax = bx.inset_axes([0.04, 0.06, 0.3, 0.035])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.5, length=1.5)
    cb.ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    cb.set_label("drying rate of the source's type", fontsize=5.8, labelpad=1)
    panel_label(bx, "b", "Where the types are")

    # (c) profiles: one row per type, same order as (a)
    cx = fig.add_subplot(gs[1, :])
    Z = prof.loc[order, cols].to_numpy()
    im = cx.imshow(Z, cmap=DIV, vmin=-1.6, vmax=1.6, aspect="auto")
    cx.set_yticks(range(len(order)))
    cx.set_yticklabels([f"{k}  {TYPE_NAMES[k]}" for k in order], fontsize=6.1)
    cx.set_xticks(range(len(cols)))
    cx.set_xticklabels([nice(c) for c in cols], fontsize=6.1, rotation=40, ha="right", rotation_mode="anchor")
    cx.grid(False)
    cx.tick_params(length=0)
    for sp_ in cx.spines.values():
        sp_.set_visible(False)
    cx.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
    cx.set_yticks(np.arange(len(order) + 1) - 0.5, minor=True)
    cx.grid(which="minor", color="white", lw=1.2)
    cx.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=cx, fraction=0.025, pad=0.01)
    cb.set_label("mean z-score", fontsize=6.3); cb.ax.tick_params(labelsize=5.8); cb.outline.set_visible(False)
    panel_label(cx, "c", "What distinguishes the types (built from 46 GIS layers; no labels or survey fields)")
    save(fig, F / "fig3_landscape_types.png")


# ----------------------------------------------------------------------------- fig 4
def fig4_rules():
    rules = pd.read_csv(M / "pm_drying_rules.csv")
    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.78, 1.3], hspace=0.6, wspace=0.95)
    ax = fig.add_subplot(gs[:, 0])
    r = rules[rules.consequent == "dried"].copy()
    r["lift_bin"] = pd.cut(r.lift, [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 3.0])
    g = r.groupby("lift_bin", observed=True).agg(n=("ward_robust", "size"), robust=("ward_robust", "mean"),
                                                 rep=("heldout_replication", lambda v: (v >= 0.8).mean()))
    yy = np.arange(len(g))
    ax.barh(yy, g.robust, height=0.62, color=C["orange"], label="significant within wards")
    ax.plot(g.rep, yy, "o", color=C["ink"], ms=3.5, label="replicates on ≥4 of 5 held-out blocks")
    for yi, (iv, row) in zip(yy, g.iterrows()):
        ax.text(0.03, yi, f"{int(row.n):,} rules", va="center", fontsize=5.8, color="white")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{iv.left:.1f}–{iv.right:.1f}" for iv in g.index], fontsize=6.5)
    ax.set_ylabel("pooled lift of the rule")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("share of rules")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), fontsize=6, handlelength=1.2)
    panel_label(ax, "a", f"{len(r):,} rules → dried")
    for k, (cons, title, col) in enumerate([("dried", "Top drying rules", C["dried"]),
                                             ("active", "Top activity rules", C["active"])]):
        bx = fig.add_subplot(gs[k, 1])
        t = rules[(rules.consequent == cons) & rules.ward_robust].sort_values("wracc", ascending=False).head(6)[::-1]
        yy = np.arange(len(t))
        bx.hlines(yy, t.mh_ci_low, t.mh_ci_high, color=col, lw=2, alpha=0.45)
        bx.plot(t.mh_or_ward, yy, "o", color=col, ms=4.5, mec="white", mew=0.6)
        bx.axvline(1, color=C["muted"], lw=0.8)
        bx.set_xscale("log")
        bx.set_xticks([1, 1.5, 2, 3, 4, 6]); bx.set_xticklabels(["1", "1.5", "2", "3", "4", "6"])
        bx.minorticks_off()
        bx.set_yticks(yy)
        bx.set_yticklabels([a.replace(" & ", " &\n") for a in t.antecedent], fontsize=6, linespacing=0.95)
        for yi, (_, row) in zip(yy, t.iterrows()):
            bx.text(6.6, yi, f"{row.confidence:.2f}", va="center", fontsize=6, color=C["ink2"])
        bx.text(6.6, len(t) - 0.3, "conf.", fontsize=6, color=C["muted"])
        bx.set_xlim(0.9, 6.3)
        bx.grid(axis="y", visible=False)
        if k == 1:
            bx.set_xlabel("ward-stratified odds ratio (95% CI)")
        panel_label(bx, "bc"[k], title)
    save(fig, F / "fig4_rules.png")


# ----------------------------------------------------------------------------- fig 5
def fig5_wards_watersheds():
    ca = pd.read_csv(M / "ca_unit_icc.csv")
    reg = pd.read_csv(M / "un_regionalisation.csv")
    nulls = pd.read_csv(M / "rb_null_distributions.csv")
    rn = nulls[(nulls.subset == "all sources") & (nulls.outcome == "drying") & (nulls.null == "region")].value
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.42, width_ratios=[1.2, 1])

    ax = fig.add_subplot(gs[0, 0])
    rows = []
    for unit, lab, fam in [("catchment T=0.5 km2", "DEM catchments, 0.5 km² (619)", "catchment"),
                           ("catchment T=2 km2", "DEM catchments, 2 km² (176)", "catchment"),
                           ("catchment T=10 km2", "DEM catchments, 10 km² (52)", "catchment")]:
        r = ca[ca.unit == unit].iloc[0]
        rows.append((lab, fam, r.icc, r.grid_icc_q025, r.grid_icc_q975))
    for part, lab in [("landscape region (n=176)", "landscape regions (176)"), ("landscape region (n=75)", "landscape regions (75)")]:
        r = reg[reg.partition == part].iloc[0]
        rows.append((lab, "landscape", r.icc_drying, r.grid_icc_q025, r.grid_icc_q975))
    r = ca[ca.unit == "ward"].iloc[0]
    rows.append(("survey wards (75)", "ward", r.icc, r.grid_icc_q025, r.grid_icc_q975))
    yy = np.arange(len(rows))[::-1]
    for yi, (lab, fam, icc, lo, hi) in zip(yy, rows):
        ax.barh(yi, hi - lo, left=lo, height=0.5, color=C["null"], zorder=1)
        ax.plot(icc, yi, "o", color=C[fam], ms=6, mec="white", mew=0.8, zorder=3)
    ax.barh(yy[-1] - 0.42, np.quantile(rn, .975) - np.quantile(rn, .025), left=np.quantile(rn, .025), height=0.22,
            color="#bdbcb5", zorder=1)
    ax.text(np.quantile(rn, .975) + 0.0015, yy[-1] - 0.42, "contiguous regions of ward size", fontsize=5.6,
            color=C["ink2"], ha="left", va="center")
    ax.set_yticks(yy)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    ax.set_xlabel("ICC of drying within units")
    ax.grid(axis="y", visible=False)
    top = rows[0]
    ax.text(top[4], yy[0] + 0.42, "size-matched grids (95%)", fontsize=5.6, color=C["ink2"],
            ha="right", va="bottom")
    ax.set_ylim(yy[-1] - 0.75, yy[0] + 0.85)
    panel_label(ax, "a", "Wards exceed both nulls; catchments do not")

    bx = fig.add_subplot(gs[0, 1])
    pts = [("ward (survey unit)", "ward", "wards"), ("DEM catchment T=2 km²", "catchment", "catchments 2 km²"),
           ("landscape region (n=75)", "landscape", "landscape regions 75"),
           ("landscape region (n=176)", "landscape", "landscape regions 176")]
    for part, fam, lab in pts:
        r = reg[reg.partition.str.startswith(part[:18])].iloc[0] if part.startswith("DEM") else reg[reg.partition == part].iloc[0]
        dx, dy = r.env_r2 - r.grid_env_r2_mean, r.icc_drying - r.grid_icc_mean
        bx.plot(dx, dy, "o", color=C[fam], ms=6.5, mec="white", mew=0.8)
        bx.annotate(lab, (dx, dy), xytext=(5, 3), textcoords="offset points", fontsize=6.3, color=C["ink2"])
    bx.axhline(0, color=C["axis"], lw=0.8); bx.axvline(0, color=C["axis"], lw=0.8)
    bx.set_xlabel("landscape homogeneity\nminus matched grids")
    bx.set_ylabel("drying ICC minus matched grids")
    bx.set_xlim(-0.06, 0.19)
    panel_label(bx, "b", "Drying vs landscape homogeneity")

    cx = fig.add_subplot(gs[1, 0])
    wc = pd.read_csv(M / "sa_ward_concordance.csv")
    x = [50, 150, 300]
    cx.plot(x, wc.concordance_same_ward, "-o", color=C["ward"], ms=4.5, label="same ward")
    cx.plot(x, wc.concordance_diff_ward, "-o", color=C["muted"], ms=4.5, label="different ward")
    for xi, a, b in zip(x, wc.concordance_same_ward, wc.concordance_diff_ward):
        cx.annotate("", xy=(xi, b), xytext=(xi, a), arrowprops=dict(arrowstyle="-", color=C["grid"], lw=4), zorder=0)
    cx.set_xticks(x); cx.set_xticklabels(["0–100", "100–200", "200–400"])
    cx.set_xlabel("distance between the two sources (m)")
    cx.set_ylabel("pairs with the same status")
    cx.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    cx.legend(loc="upper right", fontsize=6.5)
    lg = pd.read_csv(M / "ca_pair_concordance_logit.csv").set_index("term")
    cx.text(0.02, 0.04, f"odds ratio, same ward {lg.loc['same ward', 'odds_ratio']:.2f}; "
                        f"same catchment {lg.loc['same catchment (T=2 km2)', 'odds_ratio']:.2f} (n.s.)",
            transform=cx.transAxes, fontsize=6, color=C["ink2"])
    panel_label(cx, "c", "Neighbours agree more within a ward")

    dx_ = fig.add_subplot(gs[1, 1])
    w = pd.read_csv(M / "un_ward_funnel.csv")
    ws = pd.read_csv(M / "un_ward_funnel_summary.csv").iloc[0]
    E = np.linspace(w.expected.min() * 0.85, w.expected.max() * 1.05, 200)
    vbar = (w["var"] / w.expected).mean()
    for z, ls, lab in [(1.96, (0, (3, 2)), "95%"), (3.09, ":", "99.8%")]:
        dx_.plot(E, 1 + z * np.sqrt(vbar * E) / E, color=C["muted"], ls=ls, lw=0.9)
        dx_.plot(E, np.clip(1 - z * np.sqrt(vbar * E) / E, 0, None), color=C["muted"], ls=ls, lw=0.9)
        dx_.text(E[-1], 1 + z * np.sqrt(vbar * E[-1]) / E[-1], f" {lab}", fontsize=5.8, color=C["muted"], va="center")
    out = w.outside_998
    dx_.scatter(w.expected[~out], w.sdr[~out], s=12, color="#b9b8b0", linewidths=0)
    dx_.scatter(w.expected[out], w.sdr[out], s=16, color=C["ward"], linewidths=0)
    dx_.axhline(1, color=C["axis"], lw=0.8)
    dx_.set_xlabel("expected dried sources in ward\n(landscape-only model)")
    dx_.set_ylabel("observed / expected")
    dx_.text(0.98, 0.97, f"{int(ws.n_outside_998)} of {int(ws.n_wards)} wards outside 99.8%\n"
                          f"(expected {ws.expected_outside_998:.2f});  φ = {ws.overdispersion_phi:.1f}",
             transform=dx_.transAxes, ha="right", va="top", fontsize=6, color=C["ink2"])
    panel_label(dx_, "d", "Ward excess over a landscape-only model")
    save(fig, F / "fig5_wards_watersheds.png")


# ----------------------------------------------------------------------------- fig 6
def fig6_recalled(df):
    d = df[df.target_dried == 1].copy()
    yr = pd.to_numeric(d.years_since_dried, errors="coerce")
    d["year"] = yr.where((yr >= 2040) & (yr <= 2081))
    d["cause"] = d.reason_of_drying.map({"earthquake": "earthquake", "drought": "drought",
                                         "infrastructure_development": "infrastructure"}).fillna("other")
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.6, wspace=0.38)
    ax = fig.add_subplot(gs[0, :])
    years = np.arange(2050, 2082)
    bottom = np.zeros(len(years))
    for cause, col in [("earthquake", C["red"]), ("drought", C["orange"]), ("infrastructure", C["blue"]),
                       ("other", "#c9c8c1")]:
        cnt = d[d.cause == cause].year.value_counts().reindex(years, fill_value=0).to_numpy()
        ax.bar(years, cnt, bottom=bottom, width=0.82, color=col, label=cause, linewidth=0)
        bottom += cnt
    ax.annotate("Value coded 2072 BS (tentative):\n35% of dried sources, 84% of them\nattributed to the earthquake",
                xy=(2072, bottom[years == 2072][0]), xytext=(2061.2, bottom.max() * 0.62), fontsize=6.3, color=C["ink2"],
                arrowprops=dict(arrowstyle="-", color=C["muted"], lw=0.7))
    for yv in [2060, 2070, 2075, 2080]:
        if yv in years and bottom[years == yv][0] > 0:
            pass
    ax.set_xlim(2049.3, 2081.7)
    ax.set_xticks([2050, 2055, 2060, 2065, 2070, 2072, 2075, 2080])
    ax.set_xticklabels(["2050", "2055", "2060", "2065", "2070", "2072", "2075", "2080 BS"])
    ax.set_ylabel("dried sources")
    ax.legend(ncol=4, loc="upper left", fontsize=6.5, handlelength=1, columnspacing=1)
    ax.grid(axis="x", visible=False)
    panel_label(ax, "a", "Reported year values and causes (year treated as BS for analysis)")

    bx = fig.add_subplot(gs[1, 0])
    icc = pd.read_csv(M / "un_dried_icc.csv")
    outs = ["earthquake attributed", "drought attributed", "infrastructure attributed", "dated 2072 BS"]
    yy = np.arange(len(outs))[::-1]
    for yi, o in zip(yy, outs):
        w = icc[(icc.outcome == o) & (icc.unit == "ward")].iloc[0]
        bx.barh(yi, w.grid_icc_q975 - w.grid_icc_q025, left=w.grid_icc_q025, height=0.45, color=C["null"])
        bx.plot(w.icc, yi, "o", color=C["ward"], ms=5.5, mec="white", mew=0.7)
    bx.text(icc[(icc.outcome == outs[0]) & (icc.unit == "ward")].icc.iloc[0], yy[0] + 0.38, "wards", fontsize=6,
            color=C["ward"], ha="center")
    bx.set_yticks(yy)
    bx.set_yticklabels([o.replace(" attributed", "").replace("dated 2072 BS", "value coded 2072 BS") for o in outs], fontsize=6.5)
    bx.set_xlabel("ICC among dried sources")
    bx.set_xlim(-0.01, 0.31)
    bx.grid(axis="y", visible=False)
    w0 = icc[(icc.outcome == outs[0]) & (icc.unit == "ward")].iloc[0]
    bx.text((w0.grid_icc_q025 + w0.grid_icc_q975) / 2, yy[0] + 0.38, "grids", fontsize=6, color=C["muted"], ha="center")
    panel_label(bx, "b", "Recalled causes are shared within wards")

    cx = fig.add_subplot(gs[1, 1])
    k = pd.read_csv(M / "sa_knox_test.csv")
    k = k[(k.time_years == 0) & (k.pairs.isin(["same ward", "different ward"]))]
    for (cohort, ls, lab) in [("all dried with valid year", "-", "all values treated as years"),
                              ("excluding 2072 BS cohort", (0, (3, 2)), "without value coded 2072 BS")]:
        for pairs, col in [("same ward", C["ward"]), ("different ward", C["muted"])]:
            s = k[(k.cohort == cohort) & (k.pairs == pairs)].sort_values("space_m")
            cx.plot(s.space_m, s.ratio_obs_exp, ls=ls, marker="o", ms=3.5, color=col, lw=1.3)
    cx.axhline(1, color=C["axis"], lw=0.8)
    cx.set_xscale("log")
    cx.set_xticks([250, 500, 1000]); cx.set_xticklabels(["250", "500", "1000"]); cx.minorticks_off()
    cx.set_xlabel("pair distance up to (m)")
    cx.set_ylabel("same-value pairs, obs. / exp.")
    cx.legend(handles=[Line2D([], [], color=C["ward"], label="pairs in the same ward"),
                       Line2D([], [], color=C["muted"], label="pairs across a ward boundary"),
                       Line2D([], [], color=C["ink2"], ls="-", label="all dated sources"),
                       Line2D([], [], color=C["ink2"], ls=(0, (3, 2)), label="without 2072 BS")],
              fontsize=5.8, loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.6)
    cx.set_ylim(0.6, 2.95)
    panel_label(cx, "c", "Reported-year matching by ward status")
    save(fig, F / "fig6_recalled.png")


# ----------------------------------------------------------------------------- fig 7
def fig7_drivers():
    co = pd.read_csv(M / "x22_reduced_covariates.csv")
    within_name = "within ward (ward fixed effects)"
    pooled = co[co.model == "pooled"].set_index("term")
    within = co[co.model == within_name].set_index("term")
    terms = within.index.tolist()
    wv = within.loc[terms]
    pv = pooled.loc[terms]
    order = wv.coef_per_sd.sort_values().index
    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1.1, 1], hspace=0.55, wspace=0.75)
    ax = fig.add_subplot(gs[:, 0])
    yy = np.arange(len(order))
    for yi, t in zip(yy, order):
        ax.plot([pv.loc[t, "coef_per_sd"], wv.loc[t, "coef_per_sd"]], [yi, yi], color=C["grid"], lw=2.5, zorder=0)
        ax.hlines(yi, wv.loc[t, "ci_low"], wv.loc[t, "ci_high"], color=C["ward"], lw=1.2, alpha=0.6)
        ax.plot(pv.loc[t, "coef_per_sd"], yi, "o", color="#b9b8b0", ms=4.5)
        ax.plot(wv.loc[t, "coef_per_sd"], yi, "o", color=C["ward"] if wv.loc[t, "ci_excludes_zero"] else "white",
                mec=C["ward"], mew=1.0, ms=5)
    ax.axvline(0, color=C["axis"], lw=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels([nice(t) for t in order], fontsize=6.3)
    ax.set_xlabel("change in log-odds of drying per SD")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", color="#b9b8b0", label="pooled"),
                       Line2D([], [], ls="", marker="o", color=C["ward"], label="within wards (filled: 95% CI excludes 0)")],
              loc="upper left", fontsize=6)
    panel_label(ax, "a", "Reduced full-rank model, pooled and within wards")

    bx = fig.add_subplot(gs[0, 1])
    part = pd.read_csv(M / "dr_three_way_partition.csv").set_index("fraction").adj_pseudo_r2
    items = [("environment only", "environment only", C["landscape"]), ("ward only", "ward only", C["ward"]),
             ("space only", "space only", C["blue"]),
             ("space & ward", "space ∩ ward", "#f4b491"),
             ("all three", "all three", "#b9b8b0"),
             ("environment & space", "environment ∩ space", "#9fd8c2"),
             ("environment & ward", "environment ∩ ward", "#9fd8c2")]
    items = [t for t in items if t[0] in part.index]
    yy = np.arange(len(items))[::-1]
    for yi, (key, lab, col) in zip(yy, items):
        v = part[key]
        bx.barh(yi, max(v, 0), height=0.6, color=col)
        bx.text(max(v, 0) + 0.001, yi, f"{v:.3f}", va="center", fontsize=6, color=C["ink2"])
    bx.set_yticks(yy)
    bx.set_yticklabels([t[1] for t in items], fontsize=6.3)
    bx.set_xlim(0, part.drop("unexplained", errors="ignore").max() * 1.3)
    bx.set_xlabel("adjusted pseudo-R² of drying")
    bx.grid(axis="y", visible=False)
    panel_label(bx, "b", "Variation: space is shared with ward")

    cx = fig.add_subplot(gs[1, 1])
    eq = pd.read_csv(M / "rb_seismic_adjusted.csv")
    eq = eq[eq.term == "pga_may12"]
    labels = {"seismic + environment": "environment", "seismic + environment + ward fixed effects": "+ ward fixed effects"}
    rows = []
    for _, r in eq.iterrows():
        lab = labels.get(r.model, "+ 100 Moran eigenvectors" if "MEM" in r.model else r.model)
        rows.append((lab, r.coef_per_sd, r.ci_low, r.ci_high))
    rows = sorted(rows, key=lambda t: ["environment", "+ 100 Moran eigenvectors", "+ ward fixed effects"].index(t[0]))
    yy = np.arange(len(rows))[::-1]
    for yi, (lab, c, lo, hi) in zip(yy, rows):
        col = C["ward"] if "ward" in lab else C["ink2"]
        cx.hlines(yi, lo, hi, color=col, lw=1.6, alpha=0.6)
        cx.plot(c, yi, "o", color=col, ms=5)
    cx.axvline(0, color=C["axis"], lw=0.8)
    cx.set_yticks(yy); cx.set_yticklabels([r[0] for r in rows], fontsize=6.3)
    cx.set_xlabel("12 May 2015 PGA, log-odds per SD")
    var = pd.read_csv(M / "rb_seismic_within_ward_variance.csv").set_index("variable")
    cx.text(0.98, 0.58, f"only {var.loc['pga_may12', 'share_of_variance_within_wards']:.0%} of PGA variance\nlies within wards",
            transform=cx.transAxes, ha="right", fontsize=5.8, color=C["ink2"])
    cx.grid(axis="y", visible=False)
    panel_label(cx, "c", "Shaking: associated, but confounded with ward")
    save(fig, F / "fig7_drivers.png")


# ----------------------------------------------------------------------------- fig 8
def fig8_robustness():
    sub = pd.read_csv(M / "rb_sensitivity_subsets.csv")
    nulls = pd.read_csv(M / "rb_null_distributions.csv")
    names = {"all sources": "all sources", "springs only (ponds excluded)": "springs only",
             "perennial only (seasonal excluded)": "perennial only", "perennial springs only": "perennial springs",
             "located in recorded palika": "inside recorded palika"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, (outcome, col_obs, title, letter) in zip(axes, [
            ("drying", "ward_icc", "Drying", "a"),
            ("dated_2072", "dated_2072_ward_icc", "Value coded 2072 BS (dried sources)", "b")]):
        yy = np.arange(len(sub))[::-1]
        for yi, (_, r) in zip(yy, sub.iterrows()):
            for k, (nl, col) in enumerate([("grid", "#cfcec8"), ("region", "#a9a8a1")]):
                v = nulls[(nulls.subset == r.subset) & (nulls.outcome == outcome) & (nulls.null == nl)].value.to_numpy()
                if len(v):
                    jit = (np.random.default_rng(k).random(len(v)) - 0.5) * 0.22
                    ax.scatter(v, np.full(len(v), yi + (0.17 if k == 0 else -0.17)) + jit, s=2.2, color=col, linewidths=0,
                               rasterized=True)
            ax.plot(r[col_obs], yi, "D", color=C["ward"], ms=5.5, mec="white", mew=0.7)
        ax.set_yticks(yy)
        ax.set_yticklabels([f"{names.get(s, s)}  (n={n:,})" for s, n in zip(sub.subset, sub.n)], fontsize=6.5)
        ax.set_xlabel("ICC within wards")
        ax.grid(axis="y", visible=False)
        panel_label(ax, letter, title)
    fig.legend(handles=[Line2D([], [], ls="", marker="D", color=C["ward"], label="wards"),
                        Line2D([], [], ls="", marker="o", color="#cfcec8", label="200 size-matched grids"),
                        Line2D([], [], ls="", marker="o", color="#a9a8a1", label="200 contiguous regions of ward size")],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=6.5)
    fig.tight_layout(w_pad=1.2, rect=(0, 0.07, 1, 1))
    save(fig, F / "fig8_robustness.png")


def fig9_ward_rate_caterpillar(df):
    """(a) caterpillar plot of the recorded drying rate by recorded ward; (b) share of each recorded ward's sources
    that lie inside the OpenStreetMap ward polygon of the same name (Table S56)."""
    import geopandas as gpd
    from matplotlib.colors import Normalize
    from src.ward_polygons import WARD_FILE, polygon_ward, recorded_key
    grouped = (df.assign(ward_id=df.municipality.astype(str) + "_" + df.ward.astype(str))
               .groupby(["ward_id", "municipality", "ward"], as_index=False)
               .agg(n=("target_dried", "size"), dried=("target_dried", "sum")))
    grouped["rate"] = grouped.dried / grouped.n
    z = 1.96
    denom = 1 + z**2 / grouped.n
    center = (grouped.rate + z**2 / (2 * grouped.n)) / denom
    half = z * np.sqrt(grouped.rate * (1 - grouped.rate) / grouped.n + z**2 / (4 * grouped.n**2)) / denom
    grouped["low"] = center - half
    grouped["high"] = center + half
    grouped = grouped.sort_values(["rate", "municipality", "ward"]).reset_index(drop=True)

    wards = gpd.read_file(WARD_FILE)
    key = wards.palika + "-" + wards.ward.astype(str)
    rec, pw = recorded_key(df), polygon_ward(df, wards)
    wards["inside"] = key.map(pd.Series(rec == pw).groupby(rec).mean()).values
    _, pal = palika_layers()
    sv = pal[pal.surveyed]

    fig, (ax, axa) = plt.subplots(1, 2, figsize=(7.2, 5.6), gridspec_kw={"width_ratios": [0.8, 1.35]})
    y = np.arange(len(grouped))
    ax.hlines(y, grouped.low, grouped.high, color=C["grid"], lw=0.8, zorder=1)
    ax.scatter(grouped.rate, y, s=10, color=C["ward"], zorder=2)
    overall = df.target_dried.mean()
    ax.axvline(overall, color=C["ink2"], ls=(0, (3, 2)), lw=0.9)
    ax.text(overall + 0.008, len(grouped) - 0.4, f"overall {overall:.1%}", fontsize=6, color=C["ink2"])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.ward_id} (n={r.n})" for r in grouped.itertuples()], fontsize=3.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, len(grouped) - 0.2)
    ax.set_xlabel("Share recorded as dried\n(Wilson 95% interval)")
    ax.grid(axis="x", color=C["grid"], lw=0.45, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    panel_label(ax, "a", "by recorded ward")

    ABBR = {"Bethanchok": "Bt", "Dhulikhel": "Dh", "Namobuddha": "Nb", "Paunauti": "Pn", "Panchkhal": "Pk",
            "Roshi": "Ro", "Temal": "Te"}
    norm, cmap = Normalize(0, 1), plt.get_cmap("Blues")
    wards.plot(ax=axa, column="inside", cmap=cmap, norm=norm, edgecolor="white", lw=0.35)
    sv.boundary.plot(ax=axa, color=C["ink2"], lw=0.7)
    for _, r in sv.iterrows():
        pt = r.geometry.representative_point()
        axa.annotate(r["name"], (pt.x, pt.y), fontsize=6, ha="center", va="center", color=C["ink"],
                     bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))
    for _, r in wards[wards.inside < 0.5].iterrows():
        pt = r.geometry.representative_point()
        axa.annotate(f"{ABBR[r.palika]}{r.ward}", (pt.x, pt.y), fontsize=5, ha="center", va="center",
                     color=C["ink"], bbox=dict(boxstyle="round,pad=0.08", fc="#fff3c4", ec="none", alpha=0.9))
    map_axes(axa)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axa, orientation="horizontal",
                      fraction=0.05, pad=0.03, shrink=0.85, anchor=(0.5, 1.0))
    cb.set_label("share of the recorded ward's sources inside its polygon", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    panel_label(axa, "b", "recorded ward vs its polygon")
    fig.text(0.99, 0.01, "Ward polygons: OpenStreetMap contributors (ODbL); palikas: MoFAGA via geoBoundaries",
             fontsize=5.2, color=C["muted"], ha="right")
    fig.tight_layout(w_pad=0.8)
    save(fig, F / "fig9_ward_rates.png")


# ----------------------------------------------------------------------------- fig S1
def figS1_pair_bootstrap():
    """Spatial-block bootstrap distribution of the same-ward pair term, all sources and springs only (X14)."""
    draws = pd.read_csv(M / "x14_pair_bootstrap_draws.csv")
    est = pd.read_csv(M / "x14_pair_robustness.csv").query("term == 'same ward'").set_index("population")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharex=True, sharey=True)
    bins = np.linspace(draws.same_ward.min(), draws.same_ward.max(), 31)
    for ax, letter, pop in zip(axes, "ab", ["all sources", "springs only"]):
        v = draws[draws.population == pop].same_ward
        r = est.loc[pop]
        ax.hist(v, bins=bins, color="#cfcec8", edgecolor="white", linewidth=0.4)
        ax.axvline(r.log_odds, color=C["ward"], lw=1.4, label="estimate")
        ax.axvline(r.boot_ci_low, color=C["ink2"], lw=0.9, ls=(0, (3, 2)), label="95% bootstrap interval")
        ax.axvline(r.boot_ci_high, color=C["ink2"], lw=0.9, ls=(0, (3, 2)))
        ax.axvline(0, color=C["muted"], lw=0.8)
        ax.set_xlabel("same-ward log-odds of concordance")
        panel_label(ax, letter, f"{pop} ({int(r.n_pairs):,} pairs, {len(v)} draws)")
    axes[0].set_ylabel("bootstrap draws")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=6.5)
    fig.tight_layout(w_pad=1.2, rect=(0, 0.07, 1, 1))
    save(fig, F / "figS1_pair_bootstrap.png")


def run():
    apply()
    F.mkdir(parents=True, exist_ok=True)
    df = load_master()
    fig1_study_area(df)
    fig2_information()
    fig3_landscape_types(df)
    fig4_rules()
    fig5_wards_watersheds()
    fig6_recalled(df)
    fig7_drivers()
    fig8_robustness()
    figS1_pair_bootstrap()
    fig9_ward_rate_caterpillar(df)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="figure function names, e.g. fig2_information")
    a = ap.parse_args()
    if a.only:
        apply()
        F.mkdir(parents=True, exist_ok=True)
        df = load_master()
        for name in a.only:
            fn = globals()[name]
            fn(df) if "df" in fn.__code__.co_varnames[:fn.__code__.co_argcount] else fn()
    else:
        run()
