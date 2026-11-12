"""
Additional analyses that test claims the manuscript makes about wards, catchments, covariates and prediction.

X0  Raw-field skip-logic audit: missingness of each of the 57 raw survey fields among active vs dried sources.
X1  Feature-tier increments. Fold-matched paired differences in ROC-AUC between feature tiers for each model
    (env -> env_hydro, env_hydro -> env_rich), under spatial CV (0 and 500 m buffers), 2 km buffers and leave-one-
    municipality-out. Corrected resampled t-test (Nadeau-Bengio) with Holm adjustment within each design.
X2  Within-ward ranking skill. Out-of-fold predictions (spatial CV, env_hydro, no buffer) are scored by the pooled
    ROC-AUC and by the within-ward AUC: the share of (dried, active) pairs in the SAME ward that the model ranks
    correctly (per-ward AUC weighted by n_dried x n_active). The second is the skill a model offers for ranking
    sources inside one ward.
X3  Placebo units for pair concordance. The pair model  concordant ~ same catchment + same unit + log distance  is
    refitted with "same unit" defined by 200 randomly shifted grids with as many occupied cells as wards and by
    100 contiguous random regions with the ward size distribution. A one-sided spatial-block bootstrap p-value is
    added for the observed same-ward term.
X4  Ward clustering beyond the covariates. Residuals of a logistic model of drying on the 65 environmental
    covariates (and, separately, on the 46-covariate landscape set) are tested for ward ICC against the same two
    spatial nulls. This asks whether wards remain more homogeneous than arbitrary compact partitions after the
    covariates are accounted for, which the funnel plot alone cannot show (it has no spatial null).
X5  Placebo variation partitioning. The three-way partition E / S / W is repeated with W replaced by matched-grid
    and contiguous-random-region indicators, giving a reference distribution for the "unit only" and "space & unit"
    fractions that any compact 75-unit partition would capture.
X6  Simpson-type reversals with multiplicity control. Lower-tail within-ward Mantel-Haenszel p-values of every
    retained rule, Benjamini-Hochberg over rules per consequent; a reversal is counted only if q < 0.05.
X7  Cause-specific predictability. Bootstrap CI of the AUC difference between earthquake-attributed and other dried
    sources (each against all active sources), for the logistic and CatBoost out-of-fold predictions.
X8  Ward-level correlates of the ward excess. Spearman correlations of the funnel z-score of each ward with
    ward means of label-free infrastructure and settlement covariates, the number of mapped sources, and the share of
    variance in z explained by municipality.
X9  Covariate survival under grid fixed effects. The pooled vs within-unit coefficients are recomputed with
    matched-grid indicators in place of ward indicators (10 shifts) to see whether the covariates that survive
    ward conditioning also survive conditioning on arbitrary compact units of the same size.
X10 Strengthened spatial nulls for the ward ICC. Three reference families of 1,000 draws each: (G) square grids with
    random rotation and offset and as many occupied cells as wards; (C) contiguous random regions grown on the
    source graph with the ward size distribution, every region verified connected and sizes balanced to the ward
    targets; (M) the same, grown separately inside each municipality with that municipality's ward sizes, which also
    holds the municipality level fixed. Outcomes: recorded drying, seasonal/perennial status, its residuals after
    the 65 environmental and the 46 landscape covariates, and four recalled attributes among dried sources.
    Repeated for springs only (ponds excluded; partitions regrown on the spring graph). Reports null quantiles,
    the standardised excess, the
    exceedance count with a Clopper-Pearson interval, and partition diagnostics (connectivity, size mismatch). The
    same-ward pair term is also refitted against 200 C and 200 M partitions.
X11 Fold variation: spread of ROC-AUC across spatial folds and seeds, LOMO ROC-AUC per held-out municipality, and
    pooled and within-ward ROC-AUC of the out-of-fold predictions restricted to springs.
X12 Direct ward-catchment contrast in the pair model: same-ward minus same-catchment log-odds, with a percentile
    interval from the 300 spatial-block draws behind the pair-model table (both terms come from one regression).
X13 Count-matched catchments. DEM catchments are re-delineated at the area threshold that gives as many occupied
    catchments as there are wards; their ICC is compared with rotated grids, ICC(ward) - ICC(catchment) gets a
    delete-one-block jackknife interval, and the pair model is refitted with the count-matched catchment.
X14 Pair-model robustness: springs only, dyadic-robust (Aronow-Samii-Assenova) standard errors, and leave-one-
    block-out estimates that show which spatial blocks drive the skew of the bootstrap distribution.
X15 Collinearity in the joint 65-covariate within-ward model: strongest correlate and variance inflation factor of
    the covariates whose interval excludes zero, and their coefficients when entered alone with ward indicators.
X19 Within-unit ranking. Compare out-of-fold source ranking within wards and DEM catchments at three thresholds.
X20 Coordinate ablation. Refit the main spatial-CV models without coordinates and without coordinates plus source density.
X21 Null calibration. Simulate spatially autocorrelated outcomes with and without a ward random effect and compare
    ward ICC tests with the three reference-partition families.
X22 Reduced covariate model. Refit a full-rank set of continuous terrain and climate proxies without survey-recorded
    land use or exhaustive categorical indicators.
X23 Feature-family ablation. Remove one pre-specified family at a time from the enriched model on identical spatial
    folds to distinguish hydrology, climate, roads and Earth Engine terrain contributions.
X24 Within-ward uncertainty. Spatial-block bootstrap intervals for within-ward OOF AUC.
X25 Ascertainment checks. Describe reported-year distributions by ward, recent-coded years, serial-order patterns,
    WorldCover/recorded-land-use disagreement, and agreement for spatially isolated records.
X26 Ward polygons. Sources are assigned to the OpenStreetMap ward polygon that contains them (src/ward_polygons.py).
    The ward ICC of recorded drying and of its covariate residuals is recomputed with polygon wards and with recorded
    wards on the same sources, and among sources that lie inside their recorded ward, each against 1,000 rotated
    grids and 1,000 contiguous size-matched regions; ICC(recorded) - ICC(polygon) gets a delete-one-block jackknife
    interval. Ward connectivity on the 10-nearest-neighbour source graph is compared for recorded and polygon wards.
X27 Full-rank covariate set. The residual ward ICC (against the three 1,000-draw families of X10), the three-way
    variation partition and the earthquake-exposure models are recomputed with the 62-covariate full-rank set
    (src.features.FULL_RANK_DROP removed) next to the original 65-covariate set.
X28 Latent-scale ICC. Every ward and unit ICC test is repeated with the latent-scale ICC of a random-intercept
    logit (src/latent_icc.py), which does not depend on prevalence: ward ICC of drying, the seasonality flag and the
    covariate-adjusted outcome (pooled linear predictor as offset) against the three 1,000-draw families, all
    sources and springs; recalled attributes among dried sources; wards against DEM catchments at three thresholds
    and the count-matched system (rotated grids; jackknife differences); and the five sensitivity populations
    against 200 grids and 200 contiguous regions.

Outputs: results/metrics/x*_*.csv
Usage:   python -m src.revision_experiments [--only X1 X2 ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
import os

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.catchment_analysis import grid_labels, icc1, matched_grid_null
from src.dataset import coords_to_metric, design_matrix, load_master
from src.features import (
    CLIMATE,
    ENRICHMENT,
    FEATURE_SETS,
    HYDRO_CATCHMENT,
    LOCATION,
    ROAD_FEATURES,
    SURVEY_STRUCTURAL,
    TERRAIN,
    VEGETATION,
    explanatory_columns,
)
from src.spatial_mining import bh_fdr
from src.spatial_splits import grid_block_ids

logger = logging.getLogger("revision_experiments")
METRICS = PROJECT_ROOT / "results/metrics"
PRED = PROJECT_ROOT / "results/predictions"
HG = PROJECT_ROOT / "data/processed/hydro_gis_features.csv"
SEED = 2026
TABULAR = ["Logistic Regression", "Random Forest", "CatBoost"]


def _ward(df):
    return (df.municipality.astype(str) + "_" + df.ward.astype(str)).to_numpy()


def _grid_cell_for(coords, n_units):
    from scipy.optimize import brentq
    def occupied_cells_minus_target(cell):
        return len(np.unique(grid_labels(coords, cell, 0.0, 0.0))) - n_units

    return brentq(occupied_cells_minus_target, 50, 40000, xtol=1.0)


def _region_partitions(coords, ward, n_rep, seed):
    """Contiguous, size-balanced random regions with the ward size distribution (src/regions.py)."""
    from src.regions import sample_partitions
    return sample_partitions(coords, ward, n_rep, seed)


def _null_row(obs, null):
    null = np.asarray(null, float)
    return {"null_mean": null.mean(), "null_q025": np.quantile(null, 0.025), "null_q975": np.quantile(null, 0.975),
            "null_max": null.max(), "p_one_sided": (np.sum(null >= obs) + 1) / (len(null) + 1), "n_null": len(null)}


# ----------------------------------------------------------------------------- X0
def raw_field_audit():
    """Audit candidate raw survey fields, excluding the outcome column, for missingness leakage."""
    r = pd.read_csv(PROJECT_ROOT / "data/raw/spring_data.csv", low_memory=False)
    outcome = "Source Condition"
    y = r[outcome].astype(str).str.strip().str.lower().eq("dried")
    na = r.drop(columns=outcome).isna()
    out = pd.DataFrame({"field": na.columns, "missing_active": na.loc[~y].mean().to_numpy(),
                        "missing_dried": na.loc[y].mean().to_numpy()})
    out["gap"] = out.missing_dried - out.missing_active
    out["outcome_dependent"] = out.gap.abs() > 0.05
    out["direction"] = np.where(~out.outcome_dependent, "", np.where(out.gap > 0, "skipped for dried", "dried only"))
    from sklearn.metrics import roc_auc_score
    missingness_auc = []
    for field in out.field:
        indicator = na[field].to_numpy(dtype=int)
        missingness_auc.append(
            max(roc_auc_score(y, indicator), 1 - roc_auc_score(y, indicator))
            if np.unique(indicator).size == 2 else np.nan
        )
    out["missingness_only_auc"] = missingness_auc
    for threshold, suffix in [(0.01, "1pp"), (0.05, "5pp"), (0.10, "10pp")]:
        out[f"gap_gt_{suffix}"] = out.gap.abs() > threshold
    eq = "Any impact due to 2015 earthquake?"
    out.attrs["eq_answered_dried"] = int((~na.loc[y, eq]).sum()) if eq in r else -1
    out = out.sort_values("gap", key=np.abs, ascending=False).reset_index(drop=True)
    summ = pd.DataFrame([{"n_fields": len(out), "n_outcome_dependent": int(out.outcome_dependent.sum()),
                          "n_gap_gt_1pp": int(out.gap_gt_1pp.sum()),
                          "n_gap_gt_10pp": int(out.gap_gt_10pp.sum()),
                          "n_skipped_for_dried": int((out.direction == "skipped for dried").sum()),
                          "n_dried_only": int((out.direction == "dried only").sum()),
                          "n_dried": int(y.sum()), "earthquake_question_answered_dried": out.attrs["eq_answered_dried"]}])
    out.to_csv(METRICS / "x0_raw_field_audit.csv", index=False)
    summ.to_csv(METRICS / "x0_raw_field_audit_summary.csv", index=False)
    return summ


# ----------------------------------------------------------------------------- X1
def tier_increments():
    from src.report import corrected_t, holm
    folds = pd.concat([pd.read_csv(METRICS / f"benchmark_{p}_folds.csv")
                       for p in ["main", "enriched", "buffers", "lomo"]], ignore_index=True)
    designs = [("spatial CV, no buffer", "spatial", 0.0), ("spatial CV, 500 m buffer", "spatial", 500.0),
               ("spatial CV, 1 km buffer", "spatial", 1000.0), ("spatial CV, 2 km buffer", "spatial", 2000.0),
               ("leave one municipality out (500 m buffer)", "lomo", 500.0)]
    rows = []
    for design, scheme, buf in designs:
        f = folds[(folds.scheme == scheme) & (folds.buffer_m == buf)]
        for model in TABULAR:
            g = f[f.model == model]
            piv = g.pivot_table(index=["seed", "fold"], columns="feature_set", values="roc_auc")
            ratio = (g.n_test / g.n_train).mean()
            for a, b in [("env", "env_hydro"), ("env_hydro", "env_rich"), ("env", "env_rich")]:
                if a not in piv or b not in piv:
                    continue
                d = (piv[b] - piv[a]).dropna().to_numpy()
                if len(d) < 3:
                    continue
                t, p, half = corrected_t(d, ratio)
                rows.append({"design": design, "model": model, "from": a, "to": b, "auc_from": piv[a].mean(),
                             "auc_to": piv[b].mean(), "mean_diff": d.mean(), "ci95_low": d.mean() - half,
                             "ci95_high": d.mean() + half, "wins": int((d > 0).sum()), "n_pairs": len(d),
                             "p_corrected": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for design, idx in out.groupby("design").groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p_corrected"].fillna(1).to_numpy())
    out.to_csv(METRICS / "x1_tier_increments.csv", index=False)
    return out


# ----------------------------------------------------------------------------- X2
def within_ward_auc(df):
    preds = pd.read_parquet(PRED / "test_predictions_main.parquet")
    ward = pd.Series(_ward(df), index=df.id)
    rows = []
    for (model, fs, buf), g in preds.groupby(["model", "feature_set", "buffer_m"]):
        for seed, s in g.groupby("seed"):
            s = s.assign(ward=ward.reindex(s.id).to_numpy())
            pooled = roc_auc_score(s.y, s.p)
            num = den = 0.0
            for _, w in s.groupby("ward"):
                npos, nneg = int(w.y.sum()), int((1 - w.y).sum())
                if npos == 0 or nneg == 0:
                    continue
                num += roc_auc_score(w.y, w.p) * npos * nneg
                den += npos * nneg
            # between-ward skill: AUC of the ward-mean prediction
            wm = s.groupby("ward").p.transform("mean")
            rows.append({"model": model, "feature_set": fs, "buffer_m": buf, "seed": seed, "pooled_auc": pooled,
                         "within_ward_auc": num / den, "ward_mean_prediction_auc": roc_auc_score(s.y, wm)})
    per_seed = pd.DataFrame(rows)
    out = (per_seed.groupby(["model", "feature_set", "buffer_m"])
           .agg(pooled_auc=("pooled_auc", "mean"), pooled_sd=("pooled_auc", "std"),
                within_ward_auc=("within_ward_auc", "mean"), within_ward_sd=("within_ward_auc", "std"),
                ward_mean_prediction_auc=("ward_mean_prediction_auc", "mean"), n_seeds=("seed", "size"))
           .reset_index())
    out.to_csv(METRICS / "x2_within_ward_auc.csv", index=False)
    return out


# ----------------------------------------------------------------------------- X3
def _pair_design(df, coords, d_lo=100, d_hi=400):
    y = df.target_dried.to_numpy()
    pairs = cKDTree(coords).query_pairs(d_hi, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    dist = np.linalg.norm(coords[i] - coords[j], axis=1)
    keep = dist >= d_lo
    return i[keep], j[keep], dist[keep], (y[i[keep]] == y[j[keep]]).astype(int)


def _pair_coef(sc, su, ld, conc):
    X = np.c_[sc, su, ld]
    return LogisticRegression(C=1e6, max_iter=2000).fit(X, conc).coef_[0]


def pair_placebo(df, coords, n_grid=200, n_region=100, n_boot=1000):
    rng = np.random.default_rng(SEED)
    ward = _ward(df)
    catch = df.catch_T2p0.to_numpy()
    i, j, dist, conc = _pair_design(df, coords)
    sc = (catch[i] == catch[j]).astype(int)
    ld = np.log(dist)
    sw = (ward[i] == ward[j]).astype(int)
    obs = _pair_coef(sc, sw, ld, conc)
    # one-sided bootstrap p for the same-ward term (pairs resampled by spatial block of their first source)
    blocks = grid_block_ids(coords, 8)[i]
    ub = np.unique(blocks)
    members = {b: np.where(blocks == b)[0] for b in ub}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([members[b] for b in rng.choice(ub, len(ub), replace=True)])
        boots.append(_pair_coef(sc[idx], sw[idx], ld[idx], conc[idx]))
    boots = np.array(boots)
    n_units = len(np.unique(ward))
    cell = _grid_cell_for(coords, n_units)
    grid_coefs, grid_same, catch_in_grid = [], [], []
    for _ in range(n_grid):
        g = grid_labels(coords, cell, *rng.uniform(0, cell, 2))
        su = (g[i] == g[j]).astype(int)
        c = _pair_coef(sc, su, ld, conc)
        grid_coefs.append(c[1])
        catch_in_grid.append(c[0])
        grid_same.append(su.mean())
    reg_coefs, reg_same = [], []
    for part in _region_partitions(coords, ward, n_region, SEED + 1):
        su = (part[i] == part[j]).astype(int)
        c = _pair_coef(sc, su, ld, conc)
        reg_coefs.append(c[1])
        reg_same.append(su.mean())
    rows = [{"unit": "ward (observed)", "same_unit_log_odds": obs[1], "same_unit_or": np.exp(obs[1]),
             "same_catchment_log_odds": obs[0], "share_pairs_same_unit": sw.mean(),
             "boot_ci_low": np.percentile(boots[:, 1], 2.5), "boot_ci_high": np.percentile(boots[:, 1], 97.5),
             "boot_p_one_sided": (np.sum(boots[:, 1] <= 0) + 1) / (n_boot + 1), "n_pairs": len(conc)}]
    rows.append({"unit": "matched grids (placebo)", "same_unit_log_odds": np.mean(grid_coefs),
                 "same_unit_or": np.exp(np.mean(grid_coefs)), "same_catchment_log_odds": np.mean(catch_in_grid),
                 "share_pairs_same_unit": np.mean(grid_same), **_null_row(obs[1], grid_coefs)})
    rows.append({"unit": "contiguous random regions (placebo)", "same_unit_log_odds": np.mean(reg_coefs),
                 "same_unit_or": np.exp(np.mean(reg_coefs)), "share_pairs_same_unit": np.mean(reg_same),
                 **_null_row(obs[1], reg_coefs)})
    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "x3_pair_placebo.csv", index=False)
    return out


# ----------------------------------------------------------------------------- X4
def residual_ward_icc(df, coords, n_grid=200, n_region=200):
    from src.unsupervised import env_matrix
    y = df.target_dried.to_numpy().astype(float)
    ward = _ward(df)
    cols = explanatory_columns(df)
    E = StandardScaler().fit_transform(df[cols].fillna(df[cols].median()).to_numpy(float))
    Z = env_matrix(df)
    regions = _region_partitions(coords, ward, n_region, SEED + 2)
    rows = []
    for label, X in [("raw recorded drying", None), ("residual after 65 environmental covariates", E),
                     ("residual after 46 landscape covariates", Z)]:
        r = y if X is None else y - LogisticRegression(C=1.0, max_iter=10000).fit(X, y).predict_proba(X)[:, 1]
        obs = icc1(r, ward)
        _, gnull, _ = matched_grid_null(coords, r, len(np.unique(ward)), n_shift=n_grid,
                                        rng=np.random.default_rng(SEED))
        rnull = [icc1(r, p) for p in regions]
        g, rr = _null_row(obs, gnull), _null_row(obs, rnull)
        rows.append({"outcome": label, "ward_icc": obs,
                     **{f"grid_{k}": v for k, v in g.items()}, **{f"region_{k}": v for k, v in rr.items()}})
    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "x4_residual_ward_icc.csv", index=False)
    return out


# ----------------------------------------------------------------------------- X5
def placebo_partition(df, coords, n_grid=100, n_region=100):
    from src.spatial_analysis import _adj_r2, mem_eigenvectors
    y = df.target_dried.to_numpy()
    ward = _ward(df)
    cols = explanatory_columns(df)
    E = StandardScaler().fit_transform(df[cols].fillna(df[cols].median()).to_numpy(float))
    _, S = mem_eigenvectors(coords)
    R0 = pd.read_csv(METRICS / "dr_three_way_r2.csv").iloc[0].to_dict()   # E, S, ES do not depend on the unit
    rng = np.random.default_rng(SEED + 3)
    cell = _grid_cell_for(coords, len(np.unique(ward)))
    units = [("ward (observed)", ward)]
    units += [("matched grid", grid_labels(coords, cell, *rng.uniform(0, cell, 2))) for _ in range(n_grid)]
    units += [("contiguous random region", p) for p in _region_partitions(coords, ward, n_region, SEED + 4)]

    def dummies(g):
        return pd.get_dummies(pd.Series(g).astype(str), drop_first=True, dtype=float).to_numpy()

    rows = []
    for k, (label, g) in enumerate(units):
        W = dummies(g)
        R = {"E": R0["E"], "S": R0["S"], "ES": R0["ES"]}
        R["W"] = _adj_r2(W, y)
        R["EW"] = _adj_r2(np.hstack([E, W]), y)
        R["SW"] = _adj_r2(np.hstack([S, W]), y)
        R["ESW"] = _adj_r2(np.hstack([E, S, W]), y)
        rows.append({"unit": label, "rep": k, "n_units": len(np.unique(g)), "unit_total": R["W"],
                     "unit_only": R["ESW"] - R["ES"], "space_only": R["ESW"] - R["EW"],
                     "environment_only": R["ESW"] - R["SW"],
                     "space_and_unit": R["ES"] + R["EW"] - R["E"] - R["ESW"],
                     "environment_and_unit": R["ES"] + R["SW"] - R["S"] - R["ESW"], "full_model": R["ESW"]})
        logger.info(f"X5 {label} {k}: {rows[-1]}")
    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "x5_placebo_partition.csv", index=False)
    return out


# ----------------------------------------------------------------------------- X16
def partition_geometry(df, coords, n=100):
    """Compare within-unit spatial extent of wards with the three reference families."""
    from scipy.spatial.distance import pdist
    from src.regions import contiguity_graph, sample_partitions, verify

    ward = _ward(df)
    municipality = df.municipality.to_numpy()
    graph = contiguity_graph(coords)
    nearby_pairs = cKDTree(coords).query_pairs(400, output_type="ndarray")
    pair_distance = np.linalg.norm(coords[nearby_pairs[:, 0]] - coords[nearby_pairs[:, 1]], axis=1)
    nearby_pairs = nearby_pairs[(pair_distance >= 100) & (pair_distance <= 400)]
    families = [
        ("observed wards", [ward]),
        ("rotated matched grids", rotated_grids(coords, len(np.unique(ward)), n, X10_SEEDS["grid"])),
        ("contiguous size-matched regions", sample_partitions(coords, ward, n, X10_SEEDS["district"])),
        ("municipality-nested regions", sample_partitions(
            coords, ward, n, X10_SEEDS["nested"], kind="nested", groups=municipality)),
    ]
    rows = []
    for family, partitions in families:
        for rep, labels in enumerate(partitions):
            labels = np.asarray(labels)
            unit_pair_distance = []
            pair_weights = []
            unit_median_distance = []
            nearest_neighbor_distances = []
            for unit in np.unique(labels):
                points = coords[labels == unit]
                if len(points) < 2:
                    continue
                distances = pdist(points)
                unit_pair_distance.append(float(distances.mean()))
                pair_weights.append(len(distances))
                unit_median_distance.append(float(np.median(distances)))
                nearest_neighbor_distances.extend(cKDTree(points).query(points, k=2)[0][:, 1].tolist())
            rows.append({
                "partition": family,
                "replicate": rep,
                "n_units": len(np.unique(labels)),
                "share_units_disconnected": verify(labels, graph)["n_disconnected"] / len(np.unique(labels)),
                "mean_within_unit_pair_distance_m": float(np.average(unit_pair_distance, weights=pair_weights)),
                "median_unit_median_pair_distance_m": float(np.median(unit_median_distance)),
                "mean_within_unit_nearest_neighbor_m": float(np.mean(nearest_neighbor_distances)),
                "share_100_400m_pairs_same_unit": float(
                    np.mean(labels[nearby_pairs[:, 0]] == labels[nearby_pairs[:, 1]])
                ),
            })
    draws = pd.DataFrame(rows)
    draws.to_csv(METRICS / "x16_partition_geometry_draws.csv", index=False)
    metrics = ["n_units", "share_units_disconnected", "mean_within_unit_pair_distance_m",
               "median_unit_median_pair_distance_m", "mean_within_unit_nearest_neighbor_m",
               "share_100_400m_pairs_same_unit"]
    summary = draws.groupby("partition")[metrics].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "n_draws", draws.groupby("partition").size())
    summary = summary.reset_index()
    summary.to_csv(METRICS / "x16_partition_geometry.csv", index=False)
    return summary


# ----------------------------------------------------------------------------- X17
def _expected_top_k_hits(y, scores, k):
    """Expected positives in a size-k ranking when ties at the cutoff share selection equally."""
    y = np.asarray(y, dtype=float)
    scores = np.asarray(scores, dtype=float)
    cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = scores > cutoff
    tied = scores == cutoff
    tied_slots = k - int(above.sum())
    return float(y[above].sum() + tied_slots * y[tied].mean())


def top_decile_triage_metrics():
    """Evaluate how many positives each held-out fold ranks in its top decile."""
    preds = pd.read_parquet(PRED / "test_predictions_main.parquet")
    if "Random Forest" not in set(preds.model):
        forest_path = METRICS / "x17_random_forest_oof.parquet"
        if forest_path.exists():
            forest = pd.read_parquet(forest_path)
        else:
            from src.benchmark import make_splits, run_fold
            from src.dataset import design_matrix

            df = load_master()
            forest_rows = []
            for feature_set in ["env", "env_hydro"]:
                g = design_matrix(df, feature_set)
                for buffer_m in [0.0, 500.0]:
                    for seed in [42, 123, 456, 789, 2024]:
                        for split in make_splits(g, "spatial", seed, buffer_m):
                            probabilities = run_fold(g, split, seed, ["Random Forest"])["Random Forest"]
                            test = split["test_mask"]
                            forest_rows.append(pd.DataFrame({
                                "id": g["id"][test], "model": "Random Forest", "feature_set": feature_set,
                                "buffer_m": buffer_m, "seed": seed, "fold": str(split["fold"]),
                                "y": g["y"][test], "p": probabilities[test],
                            }))
            forest = pd.concat(forest_rows, ignore_index=True)
            forest.to_parquet(forest_path, index=False)
        preds = pd.concat([preds, forest], ignore_index=True)
    rows = []
    keys = ["model", "feature_set", "buffer_m", "seed", "fold"]
    for key, fold in preds.groupby(keys):
        y = fold.y.to_numpy(dtype=int)
        p = fold.p.to_numpy(dtype=float)
        k = max(1, int(np.ceil(0.10 * len(y))))
        hits = _expected_top_k_hits(y, p, k)
        precision = hits / k
        prevalence = float(y.mean())
        rows.append({
            **dict(zip(keys, key)),
            "n_test": len(y),
            "top_k": k,
            "prevalence": prevalence,
            "precision_at_10pct": precision,
            "recall_at_10pct": hits / max(float(y.sum()), 1.0),
            "lift_at_10pct": precision / prevalence if prevalence > 0 else np.nan,
        })
    folds = pd.DataFrame(rows)
    folds.to_csv(METRICS / "x17_top_decile_triage_folds.csv", index=False)
    keys = ["model", "feature_set", "buffer_m"]
    metrics = ["prevalence", "precision_at_10pct", "recall_at_10pct", "lift_at_10pct"]
    out = folds.groupby(keys)[metrics].agg(["mean", "std"]).reset_index()
    out.columns = [f"{metric}_{stat}" if stat else metric for metric, stat in out.columns]
    counts = folds.groupby(keys).size().rename("n_test_folds").reset_index()
    out = out.merge(counts, on=keys, validate="one_to_one")
    benchmark = pd.read_csv(METRICS / "benchmark_main_folds.csv")
    benchmark = (benchmark.groupby(keys)[["roc_auc", "pr_auc"]].mean().rename(
        columns={"roc_auc": "roc_auc_mean", "pr_auc": "pr_auc_mean"}).reset_index())
    out = out.merge(benchmark, on=keys, how="left", validate="one_to_one")
    out.to_csv(METRICS / "x17_top_decile_triage.csv", index=False)
    return out


# ----------------------------------------------------------------------------- X18
def terrain_type_block_permutation(n_perm=999):
    """Test terrain-type drying differences against label shuffles within spatial blocks."""
    from scipy.stats import chi2_contingency

    df = load_master()
    types = pd.read_csv(PROJECT_ROOT / "data/processed/landscape_types.csv")
    df = df.merge(types[["id", "env_type"]], on="id", how="inner", validate="one_to_one")
    y = df.target_dried.to_numpy(dtype=int)
    labels = df.env_type.to_numpy()
    coords = coords_to_metric(df.latitude.to_numpy(), df.longitude.to_numpy())
    blocks = grid_block_ids(coords, 8)
    observed = chi2_contingency(pd.crosstab(labels, y).to_numpy(), correction=False)[0]
    rng = np.random.default_rng(SEED + 18)
    null = np.empty(n_perm)
    permuted = y.copy()
    members = [np.flatnonzero(blocks == block) for block in np.unique(blocks)]
    for i in range(n_perm):
        for idx in members:
            permuted[idx] = rng.permutation(y[idx])
        null[i] = chi2_contingency(pd.crosstab(labels, permuted).to_numpy(), correction=False)[0]
    out = pd.DataFrame([{
        "observed_chi2": observed,
        "naive_chi2_p": chi2_contingency(pd.crosstab(labels, y).to_numpy(), correction=False)[1],
        "block_permutation_p": (1 + int((null >= observed).sum())) / (n_perm + 1),
        "n_permutations": n_perm,
        "block_size_km": 8,
        "null_q95": float(np.quantile(null, 0.95)),
        "null_max": float(null.max()),
    }])
    out.to_csv(METRICS / "x18_terrain_type_block_permutation.csv", index=False)
    return out


# ----------------------------------------------------------------------------- X19
def _weighted_within_unit_auc(y, score, units):
    """Pool within-unit case-control comparisons, weighting each AUC by its number of pairs."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    units = np.asarray(units)
    keep = pd.notna(units)
    y, score, units = y[keep], score[keep], units[keep]
    numerator = total_pairs = eligible_units = 0.0
    for unit in pd.unique(units):
        m = units == unit
        positives = int(y[m].sum())
        negatives = int(m.sum() - positives)
        if positives and negatives:
            pairs = positives * negatives
            numerator += roc_auc_score(y[m], score[m]) * pairs
            total_pairs += pairs
            eligible_units += 1
    all_pairs = int(y.sum()) * int(len(y) - y.sum())
    return {"within_unit_auc": numerator / total_pairs if total_pairs else np.nan,
            "within_unit_pairs": int(total_pairs),
            "share_case_control_pairs_within_units": total_pairs / all_pairs if all_pairs else np.nan,
            "eligible_units": int(eligible_units)}


def within_unit_auc_by_family(df):
    """Score paired out-of-fold predictions within wards and three DEM catchment thresholds."""
    pred_paths = [PRED / "test_predictions_main.parquet", METRICS / "x17_random_forest_oof.parquet"]
    frames = [pd.read_parquet(path) for path in pred_paths if path.exists()]
    if not frames:
        raise FileNotFoundError("No main or random-forest out-of-fold predictions were found.")
    preds = pd.concat(frames, ignore_index=True)
    preds = preds[(preds.feature_set == "env_hydro") & (preds.buffer_m == 0)]
    preds = preds.drop_duplicates(["model", "seed", "id", "feature_set", "buffer_m"])
    labels = df.set_index("id")
    units = {
        "ward": pd.Series(_ward(df), index=df.id),
        "catchment_0.5_km2": labels.catch_T0p5,
        "catchment_2_km2": labels.catch_T2p0,
        "catchment_10_km2": labels.catch_T10p0,
    }
    rows = []
    for (model, seed), pred in preds.groupby(["model", "seed"]):
        for unit_name, unit_labels in units.items():
            groups = unit_labels.reindex(pred.id).to_numpy()
            result = _weighted_within_unit_auc(pred.y.to_numpy(), pred.p.to_numpy(), groups)
            rows.append({"model": model, "seed": seed, "unit": unit_name,
                         "pooled_auc": roc_auc_score(pred.y, pred.p), **result,
                         "n_sources": int(pd.notna(groups).sum())})
    per_seed = pd.DataFrame(rows)
    out = (per_seed.groupby(["model", "unit"])
           .agg(within_unit_auc=("within_unit_auc", "mean"),
                within_unit_auc_sd=("within_unit_auc", "std"),
                pooled_auc=("pooled_auc", "mean"),
                within_unit_pairs=("within_unit_pairs", "mean"),
                share_case_control_pairs_within_units=("share_case_control_pairs_within_units", "mean"),
                eligible_units=("eligible_units", "mean"), n_sources=("n_sources", "mean"),
                n_seeds=("seed", "nunique"))
           .reset_index())
    out.to_csv(METRICS / "x19_within_unit_auc.csv", index=False)
    return out


# ----------------------------------------------------------------------------- X20
def coordinate_ablation(df):
    """Compare the main model with coordinate and source-density ablations on identical spatial folds."""
    from src.benchmark import make_splits, run_fold

    g = design_matrix(df, "env_hydro")
    cols = g["feature_cols"]
    variants = {
        "without coordinates": {"latitude", "longitude"},
        "without coordinates or source density": {"latitude", "longitude", "spring_density_500m"},
    }
    reference = pd.read_csv(METRICS / "benchmark_main_folds.csv")
    reference = reference[(reference.profile == "main") & (reference.feature_set == "env_hydro")
                          & (reference.buffer_m == 0)]
    reference = reference[["model", "seed", "fold", "roc_auc", "pr_auc"]].rename(
        columns={"roc_auc": "baseline_roc_auc", "pr_auc": "baseline_pr_auc"})
    models = ["Logistic Regression", "Random Forest", "CatBoost"]
    rows = []
    for seed in [42, 123, 456, 789, 2024]:
        for fold, split in enumerate(make_splits(g, "spatial", seed, 0.0)):
            for variant, drop in variants.items():
                keep = [i for i, col in enumerate(cols) if col not in drop]
                ablated = dict(g, x_raw=g["x_raw"][:, keep], feature_cols=[cols[i] for i in keep])
                probabilities = run_fold(ablated, split, seed, models)
                test = split["test_mask"]
                y_test = g["y"][test]
                for model, probability in probabilities.items():
                    rows.append({"model": model, "variant": variant, "seed": seed, "fold": fold,
                                 "roc_auc": roc_auc_score(y_test, probability[test]),
                                 "pr_auc": average_precision_score(y_test, probability[test])})
    folds = pd.DataFrame(rows).merge(reference, on=["model", "seed", "fold"], validate="many_to_one")
    folds["delta_roc_auc"] = folds.roc_auc - folds.baseline_roc_auc
    folds["delta_pr_auc"] = folds.pr_auc - folds.baseline_pr_auc
    folds.to_csv(METRICS / "x20_coordinate_ablation_folds.csv", index=False)
    out = (folds.groupby(["model", "variant"])
           .agg(n_folds=("fold", "size"), roc_auc_mean=("roc_auc", "mean"),
                roc_auc_sd=("roc_auc", "std"), delta_roc_auc_mean=("delta_roc_auc", "mean"),
                delta_roc_auc_sd=("delta_roc_auc", "std"), pr_auc_mean=("pr_auc", "mean"),
                delta_pr_auc_mean=("delta_pr_auc", "mean"))
           .reset_index())
    out.to_csv(METRICS / "x20_coordinate_ablation.csv", index=False)
    return out


# ----------------------------------------------------------------------------- X21
def calibrate_ward_nulls(df, coords, n_sim=100, n_null=100):
    """Check one-sided ward-ICC null rejection under spatial fields with/without ward-level effects."""
    from scipy.optimize import brentq
    from scipy.special import expit
    from src.regions import sample_partitions

    ward = _ward(df)
    ward_codes, ward_labels = pd.factorize(ward)
    null_families = {
        "rotated matched grids": rotated_grids(coords, len(ward_labels), n_null, SEED + 31),
        "contiguous size-matched regions": sample_partitions(coords, ward, 1000, X10_SEEDS["district"])[:n_null],
        "municipality-nested regions": sample_partitions(
            coords, ward, 1000, X10_SEEDS["nested"], kind="nested", groups=df.municipality.to_numpy())[:n_null],
    }
    null_codes = {name: [np.asarray(part) for part in parts] for name, parts in null_families.items()}
    rng = np.random.default_rng(SEED + 21)
    rows = []
    prevalence = float(df.target_dried.mean())
    n_features = 128
    for length_m in [500.0, 2000.0, 8000.0]:
        for ward_sd in [0.0, 0.75]:
            simulations = []
            for _ in range(n_sim):
                frequencies = rng.normal(size=(2, n_features)) / length_m
                phase = rng.uniform(0, 2 * np.pi, n_features)
                spatial = np.sqrt(2 / n_features) * np.cos(coords @ frequencies + phase).sum(axis=1)
                ward_effect = rng.normal(0, ward_sd, len(ward_labels))[ward_codes]
                latent = spatial + ward_effect
                intercept = brentq(lambda a: expit(a + latent).mean() - prevalence, -20, 20)
                y = rng.binomial(1, expit(intercept + latent))
                observed = icc1(y, ward_codes)
                for family, parts in null_codes.items():
                    null = np.asarray([icc1(y, part) for part in parts])
                    p = (int((null >= observed).sum()) + 1) / (len(null) + 1)
                    simulations.append({"family": family, "ward_icc": observed,
                                        "null_q95": float(np.quantile(null, .95)), "p": p})
            simulations = pd.DataFrame(simulations)
            for family, sim in simulations.groupby("family"):
                rejected = int((sim.p <= .05).sum())
                lo = 0.0 if rejected == 0 else stats.beta.ppf(.025, rejected, n_sim - rejected + 1)
                hi = 1.0 if rejected == n_sim else stats.beta.ppf(.975, rejected + 1, n_sim - rejected)
                rows.append({"spatial_length_m": length_m, "ward_effect_sd_logit": ward_sd,
                             "null_family": family, "n_simulations": n_sim, "n_null_partitions": n_null,
                             "rejection_rate": rejected / n_sim, "rejection_ci_low": lo, "rejection_ci_high": hi,
                             "median_ward_icc": sim.ward_icc.median(), "mean_null_q95": sim.null_q95.mean()})
    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "x21_ward_null_calibration.csv", index=False)
    return out


# ----------------------------------------------------------------------------- X22
def reduced_covariate_model(df, n_boot=300):
    """Refit a low-dimensional, full-rank model without survey-reported land use or redundant classes."""
    from src.drivers_analysis import _fit

    cols = ["elevation_m", "slope_deg", "aspect_sin", "aspect_cos", "tpi_500m", "chili",
            "hnd", "soil", "precip_annual_mm", "dist_to_road_m"]
    missing = [col for col in cols if col not in df]
    if missing:
        raise KeyError(f"Reduced covariate model is missing columns: {missing}")
    raw = df[cols].fillna(df[cols].median()).to_numpy(dtype=float)
    E = StandardScaler().fit_transform(raw)
    ward_dummies = pd.get_dummies(pd.Series(_ward(df)), drop_first=True, dtype=float).to_numpy()
    y = df.target_dried.to_numpy()
    blocks = grid_block_ids(coords_to_metric(df.latitude.to_numpy(), df.longitude.to_numpy()), 8)
    rng = np.random.default_rng(SEED + 22)
    resamples = [_resample_blocks(blocks, rng) for _ in range(n_boot)]
    vif = _vif(E)
    rows = []
    for model, X in [("pooled", E), ("within ward (ward fixed effects)", np.c_[E, ward_dummies])]:
        coef = _fit(X, y)[:len(cols)]
        boots = np.asarray([_fit(X[idx], y[idx])[:len(cols)] for idx in resamples])
        ci = np.percentile(boots, [2.5, 97.5], axis=0)
        for i, term in enumerate(cols):
            rows.append({"model": model, "term": term, "coef_per_sd": coef[i],
                         "ci_low": ci[0, i], "ci_high": ci[1, i],
                         "ci_excludes_zero": bool(ci[0, i] > 0 or ci[1, i] < 0),
                         "vif": vif[i], "n_covariates": len(cols),
                         "design_rank": int(np.linalg.matrix_rank(X))})
    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "x22_reduced_covariates.csv", index=False)
    return out


# ----------------------------------------------------------------------------- X23
def feature_family_ablation(df):
    """Drop one pre-specified predictor family from the enriched model on identical spatial folds."""
    from src.benchmark import make_splits, run_fold

    groups = {
        "coordinates": ["latitude", "longitude"],
        "source density": ["spring_density_500m"],
        "survey structure": SURVEY_STRUCTURAL,
        "SRTM terrain": TERRAIN,
        "CHIRPS and Sentinel-2": CLIMATE + VEGETATION,
        "hydrology and water balance": HYDRO_CATCHMENT,
        "roads": ROAD_FEATURES,
        "WorldCover": ["wc_tree", "wc_shrub", "wc_grass", "wc_crop", "wc_built", "wc_bare", "wc_water"],
        "Hansen forest": ["gfc_treecover2000", "gfc_loss_pre2015", "gfc_loss_post2015"],
        "SoilGrids": ["soil_clay", "soil_sand", "soil_soc", "soil_bdod"],
        "MODIS vegetation and ET": ["ndvi_trend", "ndvi_change_2015", "et_mean", "et_trend"],
        "CHIRPS trends": ["chirps_trend", "chirps_anom_2015"],
        "human footprint": ["built_2020", "built_change", "pop_1km", "nightlights"],
        "ALOS terrain indices": ["landform_position", "chili", "mtpi", "topo_diversity"],
    }
    all_cols = list(FEATURE_SETS["env_rich"])
    assigned = [column for family in groups.values() for column in family]
    if set(assigned) != set(all_cols) or len(assigned) != len(set(assigned)):
        missing = sorted(set(all_cols) - set(assigned))
        extra = sorted(set(assigned) - set(all_cols))
        raise ValueError(f"Feature-family map must partition env_rich; missing={missing}, extra={extra}")

    matrix = design_matrix(df, "env_rich")
    reference = pd.read_csv(METRICS / "benchmark_enriched_folds.csv")
    reference = reference[(reference.feature_set == "env_rich") & (reference.buffer_m == 0)]
    reference = reference[["model", "seed", "fold", "roc_auc", "n_test", "n_train"]].rename(
        columns={"roc_auc": "baseline_roc_auc"})
    models = ["Logistic Regression", "Random Forest", "CatBoost"]
    seeds = [42, 123, 456, 789, 2024]
    rows = []
    for seed in seeds:
        for fold, split in enumerate(make_splits(matrix, "spatial", seed, 0.0)):
            test = split["test_mask"]
            y_test = matrix["y"][test]
            for family, dropped in groups.items():
                keep = [i for i, col in enumerate(all_cols) if col not in set(dropped)]
                ablated = dict(matrix, x_raw=matrix["x_raw"][:, keep],
                               feature_cols=[all_cols[i] for i in keep])
                probabilities = run_fold(ablated, split, seed, models)
                for model, probability in probabilities.items():
                    rows.append({"model": model, "dropped_family": family, "seed": seed, "fold": fold,
                                 "n_features": len(keep), "roc_auc": roc_auc_score(y_test, probability[test])})
            logger.info(f"X23 enriched family ablations: seed={seed}, fold={fold}")
    folds = pd.DataFrame(rows).merge(reference, on=["model", "seed", "fold"], validate="many_to_one")
    folds["delta_roc_auc"] = folds.roc_auc - folds.baseline_roc_auc
    folds.to_csv(METRICS / "x23_feature_family_ablation_folds.csv", index=False)

    summaries = []
    for (model, family), group in folds.groupby(["model", "dropped_family"]):
        diff = group.delta_roc_auc.to_numpy()
        ratio = (group.n_test / group.n_train).mean()
        se = np.sqrt((1 / len(diff) + ratio) * diff.var(ddof=1))
        t_value = diff.mean() / se if se > 0 else np.nan
        summaries.append({"model": model, "dropped_family": family, "n_folds": len(diff),
                          "delta_roc_auc_mean": diff.mean(),
                          "ci95_low": diff.mean() - stats.t.ppf(.975, len(diff) - 1) * se,
                          "ci95_high": diff.mean() + stats.t.ppf(.975, len(diff) - 1) * se,
                          "p_uncorrected": 2 * stats.t.sf(abs(t_value), len(diff) - 1) if np.isfinite(t_value) else np.nan})
    summary = pd.DataFrame(summaries)
    summary["p_holm_within_model"] = summary.groupby("model").p_uncorrected.transform(
        lambda p: _holm(p.to_numpy()))
    summary.to_csv(METRICS / "x23_feature_family_ablation.csv", index=False)
    return summary


def _holm(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.maximum.accumulate((len(p_values) - np.arange(len(p_values))) * p_values[order])
    output = np.empty(len(p_values), dtype=float)
    output[order] = np.minimum(adjusted, 1.0)
    return output


# ----------------------------------------------------------------------------- X24
def within_ward_auc_block_bootstrap(df, n_boot=1000):
    """Block-bootstrap CIs for the within-ward AUC of spatial-CV out-of-fold predictions."""
    preds = pd.read_parquet(PRED / "test_predictions_main.parquet")
    preds = preds[(preds.feature_set == "env_hydro") & (preds.buffer_m == 0)]
    ward_by_id = pd.Series(_ward(df), index=df.id)
    block_by_id = pd.Series(
        grid_block_ids(coords_to_metric(df.latitude.to_numpy(), df.longitude.to_numpy()), 8), index=df.id)
    rng = np.random.default_rng(SEED + 24)
    rows = []
    for model, group in preds.groupby("model"):
        averaged = group.groupby("id").agg(y=("y", "first"), p=("p", "mean")).reset_index()
        units = ward_by_id.reindex(averaged.id).to_numpy()
        blocks = block_by_id.reindex(averaged.id).to_numpy()
        observed = _weighted_within_unit_auc(averaged.y, averaged.p, units)["within_unit_auc"]
        unique_blocks = pd.unique(blocks)
        members = {block: np.flatnonzero(blocks == block) for block in unique_blocks}
        draws = []
        for _ in range(n_boot):
            sampled_blocks = rng.choice(unique_blocks, len(unique_blocks), replace=True)
            sample_idx = np.concatenate([members[block] for block in sampled_blocks])
            value = _weighted_within_unit_auc(
                averaged.y.to_numpy()[sample_idx], averaged.p.to_numpy()[sample_idx], units[sample_idx]
            )["within_unit_auc"]
            if np.isfinite(value):
                draws.append(value)
        rows.append({"model": model, "within_ward_auc": observed, "ci_low": np.percentile(draws, 2.5),
                     "ci_high": np.percentile(draws, 97.5), "n_blocks": len(unique_blocks),
                     "n_bootstrap": len(draws), "bootstrap_unit": "8-km spatial blocks",
                     "prediction_summary": "OOF probabilities averaged over 5 seeds"})
    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "x24_within_ward_auc_block_bootstrap.csv", index=False)
    return out


# ----------------------------------------------------------------------------- X25
def ascertainment_checks(df, coords, n_null=1000, n_boot=1000):
    """Run descriptive checks of reported year, record sequence, cover disagreement and spatial enclaves."""
    import geopandas as gpd
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from src.regions import contiguity_graph, sample_partitions

    checks = []
    ward = _ward(df)
    y = df.target_dried.to_numpy().astype(int)
    raw = pd.read_csv(PROJECT_ROOT / "data/raw/spring_data.csv")
    if raw.SN.duplicated().any() or set(raw.SN) != set(df.id):
        raise ValueError("Raw serial numbers do not provide a one-to-one key to the processed source IDs.")
    raw = raw.set_index("SN").reindex(df.id)
    if not np.array_equal(raw.index.to_numpy(), df.id.to_numpy()):
        raise ValueError("Could not align raw serial numbers to the processed source records.")

    # Serial number is only an identifier in the release; this records association, not visit chronology.
    serial_rows = []
    for ward_id, idx in pd.Series(np.arange(len(df))).groupby(pd.Series(ward)):
        source_idx = idx.to_numpy()
        if len(source_idx) > 2 and np.unique(y[source_idx]).size > 1:
            rho = stats.spearmanr(raw.index.to_numpy()[source_idx], y[source_idx]).statistic
            serial_rows.append({"ward": ward_id, "n_sources": len(source_idx), "spearman_rho": rho})
    serial = pd.DataFrame(serial_rows)
    checks.append({"check": "within-ward SN/status association", "n": int(len(df)),
                   "estimate": float(serial.spearman_rho.median()),
                   "detail": f"median within-ward Spearman rho; {int((serial.spearman_rho > 0).sum())} positive of {len(serial)} wards; SN chronology is undocumented"})

    # Distribution of reported values by ward; recent is explicitly a reported-year category, not event dating.
    year = pd.to_numeric(df.years_since_dried, errors="coerce").to_numpy()
    valid_year = (y == 1) & (year >= 2000) & (year <= 2085)
    recent = year >= 2071
    year_rows = []
    for ward_id in pd.unique(ward):
        mask = (ward == ward_id) & valid_year
        values = year[mask]
        year_rows.append({"ward": ward_id, "n_valid_reported_years": int(mask.sum()),
                          "mean_reported_value": float(np.mean(values)) if len(values) else np.nan,
                          "median_reported_value": float(np.median(values)) if len(values) else np.nan,
                          "q25_reported_value": float(np.quantile(values, .25)) if len(values) else np.nan,
                          "q75_reported_value": float(np.quantile(values, .75)) if len(values) else np.nan,
                          "share_coded_2071_or_later": float(np.mean(values >= 2071)) if len(values) else np.nan,
                          "share_coded_2072": float(np.mean(values == 2072)) if len(values) else np.nan})
    pd.DataFrame(year_rows).to_csv(METRICS / "x25_ward_reported_year.csv", index=False)

    sel = np.flatnonzero(valid_year)
    recent_dried = recent[sel].astype(float)
    observed = icc1(recent_dried, ward[sel])
    all_ward_codes = ward
    null_families = {
        "rotated matched grids": rotated_grids(coords, len(np.unique(ward)), n_null, X10_SEEDS["grid"]),
        "contiguous size-matched regions": sample_partitions(coords, ward, n_null, X10_SEEDS["district"]),
        "municipality-nested regions": sample_partitions(
            coords, ward, n_null, X10_SEEDS["nested"], kind="nested", groups=df.municipality.to_numpy()),
    }
    recent_nulls = []
    for family, partitions in null_families.items():
        null_values = np.array([icc1(recent_dried, np.asarray(part)[sel]) for part in partitions])
        recent_nulls.append({"outcome": "reported value coded 2071-2085 among valid-year dried sources",
                             "family": family, "n_sources": len(sel), "observed_ward_icc": observed,
                             "null_q025": np.quantile(null_values, .025), "null_q975": np.quantile(null_values, .975),
                             "null_max": null_values.max(),
                             "p_nominal": (int((null_values >= observed).sum()) + 1) / (len(null_values) + 1),
                             "n_null": len(null_values)})
    pd.DataFrame(recent_nulls).to_csv(METRICS / "x25_recent_year_nulls.csv", index=False)
    checks.append({"check": "reported value coded 2071-2085 among valid-year dried records", "n": int(len(sel)),
                   "estimate": float(observed),
                   "detail": "ward observed-scale ICC; period is operational, questionnaire wording and visit dates are unavailable"})
    checks.append({"check": "valid reported-year coverage among dried records", "n": int((y == 1).sum()),
                   "estimate": float(valid_year.sum() / max((y == 1).sum(), 1)),
                   "detail": f"{int(valid_year.sum())} of {int((y == 1).sum())} dried records had numeric values 2000-2085"})

    # Survey land-use agreement is only a coarse comparison with 2021 cover fractions in a 250 m buffer.
    wc_columns = ["wc_tree", "wc_shrub", "wc_grass", "wc_crop", "wc_built", "wc_bare", "wc_water"]
    missing_wc = [column for column in wc_columns if column not in df]
    if missing_wc:
        raise KeyError(f"WorldCover features required for land-use comparison are unavailable: {missing_wc}")
    expected_cover = {"agriculture": "wc_crop", "dense_forest": "wc_tree", "sparse_forest": "wc_tree",
                      "grassland": "wc_grass", "settlement": "wc_built"}
    landuse = df.landuse.astype(str).to_numpy()
    cover = df[wc_columns].to_numpy(dtype=float)
    max_cover = np.nanmax(cover, axis=1)
    expected = np.array([expected_cover.get(value) for value in landuse], dtype=object)
    eligible = np.array([column is not None for column in expected])
    expected_value = np.array([df.iloc[i][column] if column else np.nan
                               for i, column in enumerate(expected)], dtype=float)
    mismatch = eligible & np.isfinite(expected_value) & (expected_value < max_cover)
    ward_cover_rows = []
    for ward_id in pd.unique(ward):
        mask = (ward == ward_id) & mismatch
        denom = int(((ward == ward_id) & eligible & np.isfinite(expected_value)).sum())
        ward_cover_rows.append({"ward": ward_id, "n_comparable_sources": denom,
                                "share_recorded_cover_not_largest_worldcover_class":
                                    float(mismatch[ward == ward_id].sum() / denom) if denom else np.nan,
                                "recorded_drying_rate": float(y[ward == ward_id].mean())})
    ward_cover = pd.DataFrame(ward_cover_rows)
    ward_cover.to_csv(METRICS / "x25_worldcover_by_ward.csv", index=False)
    cover_rho = stats.spearmanr(ward_cover.share_recorded_cover_not_largest_worldcover_class,
                                ward_cover.recorded_drying_rate, nan_policy="omit")
    checks.append({"check": "ward WorldCover/recorded-land-use disagreement vs drying rate",
                   "n": int(ward_cover.n_comparable_sources.sum()), "estimate": float(cover_rho.statistic),
                   "detail": "Spearman rho across wards; coarse 2021 250-m-buffer proxy, not a validation or style measure"})

    # Compare majority status in the assigned ward and among ten nearest sources for spatially isolated records.
    points = gpd.GeoDataFrame(df[["municipality"]].copy(),
                              geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    palikas = gpd.read_file(PROJECT_ROOT / "data/processed/boundaries/kavre_palikas.geojson").to_crs(points.crs)
    joined = gpd.sjoin(points, palikas[["name", "geometry"]], how="left", predicate="within")
    containing = joined.groupby(level=0).name.first().reindex(np.arange(len(df))).to_numpy()
    normalize = lambda value: "".join(ch for ch in str(value).lower() if ch.isalnum())
    municipality_aliases = {
        "bethanchownk": "bethanchowk",
        "namobuudha": "namobuddha",
        "roshirm": "roshi",
    }
    recorded_palika = np.array([
        municipality_aliases.get(normalize(value), normalize(value)) for value in df.municipality
    ])
    containing_palika = np.array([
        municipality_aliases.get(normalize(value), normalize(value)) for value in containing
    ])
    outside_palika = containing_palika != recorded_palika

    nbrs = contiguity_graph(coords)
    outside_main_component = np.zeros(len(df), dtype=bool)
    for ward_id in pd.unique(ward):
        members = np.flatnonzero(ward == ward_id)
        local_index = {int(source): local for local, source in enumerate(members)}
        rows, cols = [], []
        for source in members:
            local = local_index[int(source)]
            for neighbor in nbrs[source]:
                if int(neighbor) in local_index:
                    rows.append(local)
                    cols.append(local_index[int(neighbor)])
        graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(members), len(members)))
        n_components, labels = connected_components(graph, directed=False)
        component_sizes = np.bincount(labels, minlength=n_components)
        main_component = int(np.argmax(component_sizes))
        outside_main_component[members] = labels != main_component

    nearest = cKDTree(coords).query(coords, k=min(11, len(coords)))[1]
    ward_sum = pd.Series(y).groupby(ward).transform("sum").to_numpy()
    ward_n = pd.Series(y).groupby(ward).transform("size").to_numpy()
    ward_rate_loo = (ward_sum - y) / np.maximum(ward_n - 1, 1)
    local_rate = np.empty(len(df), dtype=float)
    for i, candidates in enumerate(nearest):
        other = candidates[candidates != i][:10]
        local_rate[i] = float(y[other].mean()) if len(other) else np.nan
    block_labels = grid_block_ids(coords, 8)
    rng = np.random.default_rng(SEED + 25)
    enclave_rows = []
    for name, mask in [("outside recorded palika", outside_palika),
                       ("outside largest within-ward 10-NN component", outside_main_component)]:
        eligible_rows = np.flatnonzero(mask & (ward_rate_loo != .5) & (local_rate != .5))
        if not len(eligible_rows):
            continue
        ward_match = (y[eligible_rows] == (ward_rate_loo[eligible_rows] > .5)).astype(float)
        local_match = (y[eligible_rows] == (local_rate[eligible_rows] > .5)).astype(float)
        difference = ward_match - local_match
        block_values = block_labels[eligible_rows]
        unique_blocks = pd.unique(block_values)
        block_members = {block: np.flatnonzero(block_values == block) for block in unique_blocks}
        boot = []
        for _ in range(n_boot):
            chosen = rng.choice(unique_blocks, len(unique_blocks), replace=True)
            sample_idx = np.concatenate([block_members[block] for block in chosen])
            boot.append(float(difference[sample_idx].mean()))
        enclave_rows.append({"subset": name, "n_records": int(mask.sum()), "n_non_tied_comparisons": len(eligible_rows),
                             "share_matches_recorded_ward_majority": float(ward_match.mean()),
                             "share_matches_10_nearest_source_majority": float(local_match.mean()),
                             "ward_minus_local_match": float(difference.mean()),
                             "block_boot_ci_low": float(np.percentile(boot, 2.5)),
                             "block_boot_ci_high": float(np.percentile(boot, 97.5)),
                             "bootstrap_blocks": len(unique_blocks)})
    pd.DataFrame(enclave_rows).to_csv(METRICS / "x25_spatially_isolated_sources.csv", index=False)
    checks.append({"check": "spatially isolated source comparison", "n": int(outside_palika.sum()),
                   "estimate": float(outside_palika.mean()),
                   "detail": f"{int(outside_main_component.sum())} sources are outside the largest within-ward component; majority matches are descriptive"})

    out = pd.DataFrame(checks)
    out.to_csv(METRICS / "x25_ascertainment_checks.csv", index=False)
    return out


# ----------------------------------------------------------------------------- X6
def reversals_fdr():
    r = pd.read_csv(METRICS / "pm_drying_rules.csv")
    out = []
    for cons, g in r.groupby("consequent"):
        g = g.copy()
        se = (np.log(g.mh_ci_high) - np.log(g.mh_ci_low)) / (2 * 1.96)
        p_low = stats.norm.cdf(np.log(g.mh_or_ward) / se)
        g["mh_p_lower"] = p_low
        g["mh_q_lower"] = bh_fdr(np.nan_to_num(p_low, nan=1.0))
        out.append(g)
    r = pd.concat(out)
    summ = (r.groupby("consequent")
            .agg(n_rules=("antecedent", "size"), n_ward_robust=("ward_robust", "sum"),
                 n_reversal_ci=("reverses_within_wards", "sum"),
                 n_reversal_fdr=("mh_q_lower", lambda q: int((q < 0.05).sum())),
                 median_heldout_replication=("heldout_replication", "median"),
                 share_heldout_all_folds=("heldout_replication", lambda v: float((v >= 0.999).mean())),
                 share_heldout_4of5=("heldout_replication", lambda v: float((v >= 0.79).mean())))
            .reset_index())
    summ["share_ward_robust"] = summ.n_ward_robust / summ.n_rules
    summ.to_csv(METRICS / "x6_reversals_fdr.csv", index=False)
    r[r.reverses_within_wards][["consequent", "antecedent", "support", "lift", "mh_or_ward", "mh_ci_low",
                                "mh_ci_high", "mh_p_lower", "mh_q_lower"]].to_csv(
        METRICS / "x6_reversal_rules.csv", index=False)
    return summ


# ----------------------------------------------------------------------------- X7
def cause_auc_difference(df, n_boot=2000):
    preds = pd.read_parquet(PRED / "test_predictions_main.parquet")
    preds = preds[(preds.buffer_m == 0) & (preds.feature_set == "env_hydro")]
    reason = df.set_index("id")["reason_of_drying"]
    rng = np.random.default_rng(SEED)
    rows = []
    for m in ["Logistic Regression", "CatBoost"]:
        p = preds[preds.model == m].groupby("id").agg(p=("p", "mean"), y=("y", "first"))
        eq = ((p.y == 1) & reason.reindex(p.index).eq("earthquake")).to_numpy()
        ot = ((p.y == 1) & ~reason.reindex(p.index).eq("earthquake")).to_numpy()
        ac = (p.y == 0).to_numpy()
        pv = p.p.to_numpy()

        def auc(pos, neg):
            return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
        a_eq, a_ot = auc(pv[eq], pv[ac]), auc(pv[ot], pv[ac])
        diffs = []
        pe, po, pa = pv[eq], pv[ot], pv[ac]
        for _ in range(n_boot):
            be, bo, ba = (rng.choice(pe, len(pe)), rng.choice(po, len(po)), rng.choice(pa, len(pa)))
            diffs.append(auc(bo, ba) - auc(be, ba))
        diffs = np.array(diffs)
        rows.append({"model": m, "auc_earthquake": a_eq, "auc_other": a_ot, "diff_other_minus_eq": a_ot - a_eq,
                     "ci_low": np.percentile(diffs, 2.5), "ci_high": np.percentile(diffs, 97.5),
                     "p_two_sided": min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()) + 1 / n_boot),
                     "n_earthquake": int(eq.sum()), "n_other": int(ot.sum()), "n_active": int(ac.sum())})
    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "x7_cause_auc_difference.csv", index=False)
    return out


# ----------------------------------------------------------------------------- X8
def ward_correlates(df):
    w = pd.read_csv(METRICS / "un_ward_funnel.csv")
    d = df.assign(ward_id=_ward(df))
    cand = {"road_density_500m": "road density (500 m)", "dist_to_road_m": "distance to road",
            "built_change": "built-up growth 2000-20", "built_2020": "built-up fraction 2020", "pop_1km": "population (1 km)",
            "nightlights": "night-time lights", "elevation_m": "elevation", "precip_annual_mm": "annual precipitation",
            "pga_may12": "12 May 2015 PGA", "pga_main": "25 April 2015 PGA"}
    sm = PROJECT_ROOT / "data/processed/seismic_features.csv"
    if sm.exists() and "pga_main" not in d:
        d = d.merge(pd.read_csv(sm), on="id", how="left")
    agg = {c: (c, "mean") for c in cand if c in d}
    wm = d.groupby("ward_id").agg(n_sources=("id", "size"), **agg).reset_index()
    wm = wm.merge(w[["ward", "z", "sdr", "municipality"]], left_on="ward_id", right_on="ward")
    rows = []
    for c in ["n_sources"] + [c for c in cand if c in wm]:
        rho, p = stats.spearmanr(wm[c], wm.z)
        rows.append({"ward_covariate": cand.get(c, "mapped sources per ward"), "spearman_rho": rho, "p": p})
    out = pd.DataFrame(rows)
    out["q_bh"] = bh_fdr(out.p.to_numpy())
    # share of the variance in ward z explained by municipality (one-way ANOVA eta^2)
    gm = wm.groupby("municipality").z.transform("mean")
    eta2 = 1 - ((wm.z - gm) ** 2).sum() / ((wm.z - wm.z.mean()) ** 2).sum()
    f, p = stats.f_oneway(*[g.z.to_numpy() for _, g in wm.groupby("municipality")])
    out = pd.concat([out, pd.DataFrame([{"ward_covariate": "municipality (eta^2 of ward z)", "spearman_rho": eta2,
                                         "p": p, "q_bh": np.nan}])], ignore_index=True)
    out.to_csv(METRICS / "x8_ward_correlates.csv", index=False)
    return out


# ----------------------------------------------------------------------------- X9
def grid_fe_survival(df, coords, n_shift=10):
    y = df.target_dried.to_numpy()
    cols = explanatory_columns(df)
    E = StandardScaler().fit_transform(df[cols].fillna(df[cols].median()).to_numpy(float))
    def covariate_coefs(X):
        """Standardised covariate coefficients, leaving out any appended unit indicators."""
        return LogisticRegression(C=10.0, max_iter=10000).fit(X, y).coef_[0][: len(cols)]

    def dmy(g):
        return pd.get_dummies(pd.Series(g).astype(str), drop_first=True, dtype=float).to_numpy()

    ward = _ward(df)
    res = {"pooled": covariate_coefs(E), "ward FE": covariate_coefs(np.c_[E, dmy(ward)])}
    rng = np.random.default_rng(SEED + 5)
    cell = _grid_cell_for(coords, len(np.unique(ward)))
    grid = np.array([covariate_coefs(np.c_[E, dmy(grid_labels(coords, cell, *rng.uniform(0, cell, 2)))])
                     for _ in range(n_shift)])
    out = pd.DataFrame({"term": cols, "pooled": res["pooled"], "ward_fe": res["ward FE"],
                        "grid_fe_mean": grid.mean(0), "grid_fe_min": grid.min(0), "grid_fe_max": grid.max(0)})
    out.to_csv(METRICS / "x9_grid_fe_survival.csv", index=False)
    return out


ALL = ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15",
       "X16", "X17", "X18", "X19", "X20", "X21", "X22", "X23", "X24", "X25", "X26", "X27", "X28"]


# ----------------------------------------------------------------------------- X10
X10_SEEDS = {"grid": SEED + 10, "district": SEED + 11, "nested": SEED + 12}
X10_N = 1000


def rotated_grids(coords, n_units, n_rep, seed):
    """Square grids at a random angle and offset, cell size chosen so that the grid has n_units occupied cells."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_rep):
        th = rng.uniform(0, np.pi / 2)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        c = (coords - coords.mean(0)) @ R.T
        cell = _grid_cell_for(c, n_units)
        out.append(grid_labels(c, cell, *rng.uniform(0, cell, 2)))
    return out


def _null_stats(obs, null):
    from scipy.stats import beta
    null = np.asarray(null, float)
    n, k = len(null), int(np.sum(null >= obs))
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return {"obs": obs, "null_mean": null.mean(), "null_sd": null.std(ddof=1), "null_q975": np.quantile(null, .975),
            "null_q99": np.quantile(null, .99), "null_max": null.max(), "z": (obs - null.mean()) / null.std(ddof=1),
            "n_null": n, "n_exceed": k, "p_one_sided": (k + 1) / (n + 1), "exceed_prob_ci_low": lo,
            "exceed_prob_ci_high": hi}


def strengthened_nulls(df, coords, n=X10_N):
    from src.regions import contiguity_graph, partition_diagnostics, sample_partitions, verify
    from src.unsupervised import env_matrix
    rows, diag = [], []
    subsets = {"all sources": np.ones(len(df), bool), "springs only": (df["Source Type_encoded"] == 0).to_numpy()}
    yr = pd.to_numeric(df["years_since_dried"], errors="coerce").where(lambda v: (v >= 2000) & (v <= 2085)).to_numpy()
    reason = df.reason_of_drying.to_numpy()
    cols = explanatory_columns(df)
    for sname, m in subsets.items():
        d = df[m].reset_index(drop=True)
        c, w, mun = coords[m], _ward(d), d.municipality.to_numpy()
        y = d.target_dried.to_numpy().astype(float)
        E = StandardScaler().fit_transform(d[cols].fillna(d[cols].median()).to_numpy(float))
        Z = env_matrix(d)
        allrows = np.ones(len(d), bool)
        outcomes = {"recorded drying": (y, allrows)}
        outcomes["seasonality flag"] = (
            (d["Perennial / Seasonal_encoded"].to_numpy() == 0).astype(float), allrows)
        for lab, X in [("residual after 65 environmental covariates", E), ("residual after 46 landscape covariates", Z)]:
            res = y - LogisticRegression(C=1.0, max_iter=10000).fit(X, y).predict_proba(X)[:, 1]
            outcomes[lab] = (res, allrows)
        dm = y == 1
        for lab, v in [("earthquake attribution", reason[m] == "earthquake"),
                       ("drought attribution", reason[m] == "drought"),
                       ("infrastructure attribution", reason[m] == "infrastructure_development"),
                       ("reported year value coded as 2072 BS", yr[m] == 2072)]:
            outcomes[lab] = (v[dm].astype(float), dm)
        fams = {"rotated matched grids": rotated_grids(c, len(np.unique(w)), n, X10_SEEDS["grid"]),
                "contiguous size-matched regions": sample_partitions(c, w, n, X10_SEEDS["district"]),
                "contiguous size-matched regions within municipality": sample_partitions(
                    c, w, n, X10_SEEDS["nested"], kind="nested", groups=mun)}
        for fam, parts in fams.items():
            if fam == "rotated matched grids":
                cnt = np.array([len(np.unique(g)) for g in parts])
                diag.append({"subset": sname, "null": fam, "n_draws": len(parts), "occupied_cells_mean": cnt.mean(),
                             "occupied_cells_min": cnt.min(), "occupied_cells_max": cnt.max(),
                             "seed": X10_SEEDS["grid"]})
            else:
                kind = "nested" if "within" in fam else "district"
                dg = partition_diagnostics(parts, c, w, kind=kind, groups=mun if kind == "nested" else None)
                diag.append({"subset": sname, "null": fam, "n_draws": len(parts),
                             "share_draws_all_regions_connected": float((dg.n_disconnected == 0).mean()),
                             "mean_size_l1_share": dg.size_l1_share.mean(), "max_size_l1_share": dg.size_l1_share.max(),
                             "median_max_abs_size_dev": dg.size_max_abs_dev.median(),
                             "max_abs_size_dev": dg.size_max_abs_dev.max(), "seed": X10_SEEDS[kind]})
            for oname, (v, sel) in outcomes.items():
                obs = icc1(v, w[sel])
                null = [icc1(v, np.asarray(g)[sel]) for g in parts]
                rows.append({"subset": sname, "outcome": oname, "null": fam, **_null_stats(obs, null)})
            logger.info(f"X10 {sname} | {fam}: done")
    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "x10_strengthened_nulls.csv", index=False)
    pd.DataFrame(diag).to_csv(METRICS / "x10_partition_diagnostics.csv", index=False)
    # connectivity of the wards themselves on the same graph, for comparison
    ward = _ward(df)
    wv = verify(pd.factorize(ward)[0], contiguity_graph(coords))
    pd.DataFrame([{"n_wards": wv["n_regions"], "wards_not_connected_on_graph": wv["n_disconnected"]}]).to_csv(
        METRICS / "x10_ward_connectivity.csv", index=False)
    # same-ward pair term against verified contiguous partitions (first 200 draws of each family)
    catch = df.catch_T2p0.to_numpy()
    i, j, dist, conc = _pair_design(df, coords)
    sc, ld = (catch[i] == catch[j]).astype(int), np.log(dist)
    obs = _pair_coef(sc, (ward[i] == ward[j]).astype(int), ld, conc)[1]
    prow = []
    for fam, kind, seed, groups in [("contiguous size-matched regions", "district", X10_SEEDS["district"], None),
                                    ("contiguous size-matched regions within municipality", "nested",
                                     X10_SEEDS["nested"], df.municipality.to_numpy())]:
        parts = sample_partitions(coords, ward, n, seed, kind=kind, groups=groups)[:200]
        null = [_pair_coef(sc, (np.asarray(g)[i] == np.asarray(g)[j]).astype(int), ld, conc)[1] for g in parts]
        prow.append({"null": fam, **_null_stats(obs, null)})
    pd.DataFrame(prow).to_csv(METRICS / "x10_pair_placebo.csv", index=False)
    return out


# ----------------------------------------------------------------------------- X11
def _within_ward(s):
    num = den = 0.0
    for _, w in s.groupby("ward"):
        npos, nneg = int(w.y.sum()), int((1 - w.y).sum())
        if npos and nneg:
            num += roc_auc_score(w.y, w.p) * npos * nneg
            den += npos * nneg
    return num / den if den else np.nan


def fold_variation(df):
    """Spread of ROC-AUC across spatial folds and held-out municipalities, and springs-only scores.

    (a) For every design, model and tier: mean, SD, min and max of fold ROC-AUC over seed x fold, and the SD of the
        seed-level mean (how much the estimate moves with the block-to-fold assignment).
    (b) Leave-one-municipality-out ROC-AUC of each held-out municipality (mean over seeds).
    (c) Pooled and within-ward ROC-AUC of the out-of-fold predictions restricted to springs (ponds excluded)."""
    folds = pd.concat([pd.read_csv(METRICS / f"benchmark_{p}_folds.csv")
                       for p in ["main", "enriched", "buffers", "lomo"]], ignore_index=True)
    a = (folds.groupby(["scheme", "buffer_m", "feature_set", "model"])
         .agg(auc_mean=("roc_auc", "mean"), auc_sd=("roc_auc", "std"), auc_min=("roc_auc", "min"),
              auc_max=("roc_auc", "max"), n_folds=("roc_auc", "size")).reset_index())
    seed_sd = (folds.groupby(["scheme", "buffer_m", "feature_set", "model", "seed"]).roc_auc.mean()
               .groupby(["scheme", "buffer_m", "feature_set", "model"]).std().rename("seed_mean_sd").reset_index())
    a = a.merge(seed_sd, on=["scheme", "buffer_m", "feature_set", "model"])
    a.to_csv(METRICS / "x11_fold_variation.csv", index=False)
    lomo = folds[folds.scheme == "lomo"]
    b = (lomo.groupby(["fold", "feature_set", "model"])
         .agg(auc_mean=("roc_auc", "mean"), auc_sd_over_seeds=("roc_auc", "std"), n_test=("n_test", "mean"),
              test_prevalence=("test_prevalence", "mean")).reset_index().rename(columns={"fold": "municipality"}))
    b.to_csv(METRICS / "x11_lomo_by_municipality.csv", index=False)
    preds = pd.read_parquet(PRED / "test_predictions_main.parquet")
    meta = df.set_index("id")
    preds = preds.assign(ward=pd.Series(_ward(df), index=df.id).reindex(preds.id).to_numpy(),
                         spring=(meta["Source Type_encoded"].reindex(preds.id) == 0).to_numpy())
    rows = []
    for (model, fs, buf), g in preds[preds.buffer_m == 0].groupby(["model", "feature_set", "buffer_m"]):
        for subset, s0 in [("all sources", g), ("springs only", g[g.spring])]:
            per = [(roc_auc_score(s.y, s.p), _within_ward(s)) for _, s in s0.groupby("seed")]
            per = np.array(per)
            rows.append({"model": model, "feature_set": fs, "subset": subset, "pooled_auc": per[:, 0].mean(),
                         "pooled_sd": per[:, 0].std(ddof=1), "within_ward_auc": per[:, 1].mean(),
                         "within_ward_sd": per[:, 1].std(ddof=1), "n_seeds": len(per)})
    c = pd.DataFrame(rows)
    c.to_csv(METRICS / "x11_springs_only_auc.csv", index=False)
    return c


# ----------------------------------------------------------------------------- X12
WARD_TERM, CATCH_TERM = "same ward", "same catchment (T=2 km2)"


def _contrast_row(label, obs_a, obs_b, draws_a, draws_b):
    """Difference a - b of two estimates, with a percentile interval from paired bootstrap draws."""
    diff = np.asarray(draws_a) - np.asarray(draws_b)
    return {"contrast": label, "estimate_a": obs_a, "estimate_b": obs_b, "difference": obs_a - obs_b,
            "ci_low": np.percentile(diff, 2.5), "ci_high": np.percentile(diff, 97.5),
            "p_one_sided": (np.sum(diff <= 0) + 1) / (len(diff) + 1), "n_boot": len(diff)}


def pair_contrast():
    """Same-ward minus same-catchment log-odds in the pair model, from the draws behind the pair-model table.

    Both terms are estimated in one regression, so each bootstrap resample gives a paired difference. This reuses
    the 300 spatial-block draws saved by src.catchment_analysis.pair_concordance; no new resampling is done."""
    coef = pd.read_csv(METRICS / "ca_pair_concordance_logit.csv").set_index("term").log_odds
    draws = pd.read_csv(METRICS / "ca_pair_concordance_boot.csv")
    row = _contrast_row("same ward - same 2 km2 catchment (log-odds)", coef[WARD_TERM], coef[CATCH_TERM],
                        draws[WARD_TERM], draws[CATCH_TERM])
    row["ratio_of_odds_ratios"] = np.exp(row["difference"])
    out = pd.DataFrame([row])
    out.to_csv(METRICS / "x12_pair_contrast.csv", index=False)
    return out


# ----------------------------------------------------------------------------- X13
def _resample_blocks(blocks, rng):
    """Row indices of one spatial-block bootstrap resample (whole blocks drawn with replacement)."""
    ub = np.unique(blocks)
    members = {b: np.where(blocks == b)[0] for b in ub}
    return np.concatenate([members[b] for b in rng.choice(ub, len(ub), replace=True)])


def _block_jackknife_icc_difference(y, units_a, units_b, blocks, icc=icc1):
    """ICC(a) - ICC(b) with a delete-one-block jackknife standard error.

    A bootstrap is not used here: rows repeated by resampling agree with their own copies, which inflates the ICC,
    and more so for finer units, so bootstrap differences between unit systems of different size are biased.
    Leaving out one spatial block at a time repeats nothing."""
    estimate = icc(y, units_a) - icc(y, units_b)
    left_out = []
    for b in np.unique(blocks):
        keep = blocks != b
        left_out.append(icc(y[keep], units_a[keep]) - icc(y[keep], units_b[keep]))
    left_out = np.array(left_out)
    n = len(left_out)
    se = np.sqrt((n - 1) / n * ((left_out - left_out.mean()) ** 2).sum())
    return {"difference": estimate, "se": se, "ci_low": estimate - 1.96 * se, "ci_high": estimate + 1.96 * se,
            "p_one_sided": float(stats.norm.sf(estimate / se)), "n_blocks": n}


def matched_count_contrast(df, coords, n_null=1000, n_boot=1000):
    """Wards against DEM catchments with the same number of occupied units.

    (a) ICC of recorded drying for wards, the 2 km^2 catchments and the count-matched catchments, each against
        1,000 rotated grids with as many occupied cells as the unit system has units.
    (b) ICC(ward) - ICC(catchment) for both catchment systems, with a delete-one-block jackknife interval. Raw
        ICCs rise as units get smaller, so only the count-matched difference compares like with like.
    (c) The pair model refitted with the count-matched catchment, and the same-ward minus same-catchment term
        (paired spatial-block bootstrap)."""
    from src.hydro_gis import MATCHED, build_matched_count_catchments
    if not MATCHED.exists():
        _, scan = build_matched_count_catchments(target_units=len(np.unique(_ward(df))))
        scan.to_csv(METRICS / "x13_threshold_scan.csv", index=False)
    matched = pd.read_csv(MATCHED).set_index("id").reindex(df.id)
    threshold = float(matched.matched_threshold_km2.iloc[0])
    matched_name = f"catchment T={threshold:g} km2 (count-matched)"
    y = df.target_dried.to_numpy().astype(float)
    units = {"ward": _ward(df), "catchment T=2 km2": df.catch_T2p0.to_numpy(),
             matched_name: matched.catch_matched.to_numpy()}

    rows = []
    for k, (name, g) in enumerate(units.items()):
        n_units = len(np.unique(g))
        null = [icc1(y, grid) for grid in rotated_grids(coords, n_units, n_null, SEED + 20 + k)]
        rows.append({"unit": name, "n_units": n_units, **_null_stats(icc1(y, g), null)})
    icc_table = pd.DataFrame(rows)
    icc_table.to_csv(METRICS / "x13_matched_count_icc.csv", index=False)

    blocks = grid_block_ids(coords, 8)
    obs = icc_table.set_index("unit").obs
    contrasts = []
    for name in list(units)[1:]:
        jack = _block_jackknife_icc_difference(y, units["ward"], units[name], blocks)
        contrasts.append({"contrast": f"ICC: ward - {name}", "estimate_a": obs["ward"], "estimate_b": obs[name],
                          "interval": "delete-one-block jackknife", **jack})

    # pair model with the count-matched catchment in place of the 2 km^2 catchment
    i, j, dist, conc = _pair_design(df, coords)
    ward, catch = units["ward"], units[matched_name]
    sc, sw, ld = (catch[i] == catch[j]).astype(int), (ward[i] == ward[j]).astype(int), np.log(dist)
    coef = _pair_coef(sc, sw, ld, conc)
    rng = np.random.default_rng(SEED + 24)
    draws = []
    for _ in range(n_boot):
        idx = _resample_blocks(blocks[i], rng)
        draws.append(_pair_coef(sc[idx], sw[idx], ld[idx], conc[idx]))
    draws = np.array(draws)
    contrasts.append({**_contrast_row(f"pair log-odds: same ward - same {matched_name}", coef[1], coef[0],
                                      draws[:, 1], draws[:, 0]), "interval": "spatial-block bootstrap"})
    out = pd.DataFrame(contrasts)
    out.to_csv(METRICS / "x13_matched_count_contrast.csv", index=False)
    return out


# ----------------------------------------------------------------------------- X14
def dyadic_robust_cov(X, resid, weight, i, j):
    """Sandwich covariance of a logit fitted on dyads, robust to dyads that share a source.

    Aronow, Samii and Assenova (2015): the scores of two dyads may covary whenever the dyads have a source in
    common. With S_k the summed score of all dyads containing source k, the meat is
        sum_k S_k S_k'  -  sum_d s_d s_d'
    (the second term removes each dyad's own product, which the first counts once per member).
    X includes the intercept column; resid = y - p; weight = p (1 - p)."""
    scores = X * resid[:, None]
    bread = np.linalg.inv((X * weight[:, None]).T @ X)
    n_sources = int(max(i.max(), j.max())) + 1
    S = np.zeros((n_sources, X.shape[1]))
    np.add.at(S, i, scores)
    np.add.at(S, j, scores)
    meat = S.T @ S - scores.T @ scores
    return bread @ meat @ bread


def _pair_fit_with_se(sc, sw, ld, conc, i, j):
    """Pair-model coefficients (catchment, ward, log distance) with naive and dyadic-robust covariance."""
    X = np.c_[sc, sw, ld]
    m = LogisticRegression(C=1e6, max_iter=2000).fit(X, conc)
    p = m.predict_proba(X)[:, 1]
    X1 = np.c_[X, np.ones(len(X))]
    w = p * (1 - p)
    naive = np.sqrt(np.diag(np.linalg.inv((X1 * w[:, None]).T @ X1)))[:3]
    return m.coef_[0], naive, dyadic_robust_cov(X1, conc - p, w, i, j)


def _pair_robustness_row(population, term, estimate, draws, se_dyadic, n_sources, n_pairs):
    return {"population": population, "term": term, "n_sources": n_sources, "n_pairs": n_pairs,
            "log_odds": estimate, "odds_ratio": np.exp(estimate),
            "boot_ci_low": np.percentile(draws, 2.5), "boot_ci_high": np.percentile(draws, 97.5),
            "boot_p_one_sided": (np.sum(draws <= 0) + 1) / (len(draws) + 1), "boot_skewness": stats.skew(draws),
            "n_boot": len(draws), "se_dyadic": se_dyadic, "dyadic_ci_low": estimate - 1.96 * se_dyadic,
            "dyadic_ci_high": estimate + 1.96 * se_dyadic,
            "dyadic_p_two_sided": 2 * stats.norm.sf(abs(estimate / se_dyadic))}


def pair_robustness(df, coords, n_boot=300):
    """Is the same-ward pair term stable when ponds are dropped, and under dyadic-robust inference?

    For all sources and for springs only, the pair model is reported with
    (a) the spatial-block bootstrap interval used in the paper (all sources: the saved draws of the pair-model
        table; springs: 300 new draws by the same scheme),
    (b) Wald intervals from dyadic-robust standard errors, and
    (c) leave-one-block-out estimates, which show how far single spatial blocks move the ward term and so where
        the skew of the bootstrap distribution comes from.
    The bootstrap draws are written out for a supplementary figure."""
    rows, draw_tables = [], []
    populations = {"all sources": np.ones(len(df), bool), "springs only": (df["Source Type_encoded"] == 0).to_numpy()}
    for k, (pop, mask) in enumerate(populations.items()):
        d, c = df[mask].reset_index(drop=True), coords[mask]
        ward, catch = _ward(d), d.catch_T2p0.to_numpy()
        i, j, dist, conc = _pair_design(d, c)
        sc, sw, ld = (catch[i] == catch[j]).astype(int), (ward[i] == ward[j]).astype(int), np.log(dist)
        coef, se_naive, cov = _pair_fit_with_se(sc, sw, ld, conc, i, j)
        blocks = grid_block_ids(c, 8)[i]
        if pop == "all sources":
            draws = pd.read_csv(METRICS / "ca_pair_concordance_boot.csv")[[CATCH_TERM, WARD_TERM]].to_numpy()
        else:
            rng = np.random.default_rng(SEED + 30 + k)
            draws = []
            for _ in range(n_boot):
                idx = _resample_blocks(blocks, rng)
                draws.append(_pair_coef(sc[idx], sw[idx], ld[idx], conc[idx])[:2])
            draws = np.array(draws)
        size = (int(mask.sum()), len(conc))
        ward_row = _pair_robustness_row(pop, "same ward", coef[1], draws[:, 1], np.sqrt(cov[1, 1]), *size)
        ward_row["se_naive"] = se_naive[1]
        catch_row = _pair_robustness_row(pop, "same 2 km2 catchment", coef[0], draws[:, 0], np.sqrt(cov[0, 0]), *size)
        catch_row["se_naive"] = se_naive[0]
        se_diff = np.sqrt(cov[1, 1] + cov[0, 0] - 2 * cov[0, 1])
        diff_row = _pair_robustness_row(pop, "same ward - same 2 km2 catchment", coef[1] - coef[0],
                                        draws[:, 1] - draws[:, 0], se_diff, *size)
        # leave-one-block-out: the ward term refitted without each spatial block in turn
        loo = {}
        for b in np.unique(blocks):
            keep = blocks != b
            loo[b] = _pair_coef(sc[keep], sw[keep], ld[keep], conc[keep])[1]
        most_influential = max(loo, key=lambda b: abs(loo[b] - coef[1]))
        ward_row.update({"loo_block_min": min(loo.values()), "loo_block_max": max(loo.values()),
                         "n_blocks": len(loo),
                         "most_influential_block_pair_share": float((blocks == most_influential).mean()),
                         "loo_without_most_influential_block": loo[most_influential]})
        rows += [ward_row, catch_row, diff_row]
        draw_tables.append(pd.DataFrame({"population": pop, "same_ward": draws[:, 1], "same_catchment": draws[:, 0]}))
    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "x14_pair_robustness.csv", index=False)
    pd.concat(draw_tables, ignore_index=True).to_csv(METRICS / "x14_pair_bootstrap_draws.csv", index=False)
    return out


# ----------------------------------------------------------------------------- X15
EXACT_VIF = 1e6   # above this a covariate is treated as an exact linear combination of the others


def _vif(X, extra=None):
    """Variance inflation factor of each column of X given the other columns (and, optionally, extra regressors).

    A constant column gets NaN. A column that is an exact linear combination of the others gets inf."""
    out = []
    for k in range(X.shape[1]):
        total = ((X[:, k] - X[:, k].mean()) ** 2).sum()
        if total == 0:
            out.append(np.nan)
            continue
        others = np.delete(X, k, axis=1)
        Z = np.c_[np.ones(len(X)), others] if extra is None else np.c_[np.ones(len(X)), others, extra]
        fitted = Z @ np.linalg.lstsq(Z, X[:, k], rcond=None)[0]
        unexplained = ((X[:, k] - fitted) ** 2).sum() / total
        out.append(np.inf if unexplained < 1 / EXACT_VIF else 1 / unexplained)
    return np.array(out)


def joint_model_collinearity(df, n_boot=300):
    """How far does collinearity among the 65 covariates affect those that keep a within-ward signal?

    The within-ward model is ONE ridge-penalised logit with all 65 standardised covariates and ward indicators.
    For each covariate whose within-ward interval excludes zero this reports its strongest correlate, its variance
    inflation factor (among the covariates, and with ward indicators added), and its coefficient when it enters
    alone with ward indicators (same penalty, spatial-block bootstrap), so that the joint estimate can be compared
    with a one-covariate-at-a-time estimate. A design summary records the rank of the covariate matrix: the
    land-cover fractions sum to one and the land-use indicators are exhaustive, so the joint model is identified
    only through its ridge penalty."""
    from src.drivers_analysis import SEED as DRIVERS_SEED, _fit
    joint = pd.read_csv(METRICS / "dr_pooled_vs_within_ward.csv")
    joint = joint[joint.model.str.startswith("within ward")].set_index("term")
    cols = explanatory_columns(df)
    y = df.target_dried.to_numpy()
    E = StandardScaler().fit_transform(df[cols].fillna(df[cols].median()).to_numpy(float))
    ward_dummies = pd.get_dummies(pd.Series(_ward(df)), drop_first=True, dtype=float).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.nan_to_num(np.corrcoef(E, rowvar=False))      # a constant covariate correlates with nothing
    np.fill_diagonal(corr, 0)
    vif, vif_ward = _vif(E), _vif(E, ward_dummies)
    pd.DataFrame({"term": cols, "vif_covariates": vif, "vif_with_ward_indicators": vif_ward}).to_csv(
        METRICS / "x15_vif_all_covariates.csv", index=False)
    pd.DataFrame([{"n_covariates": len(cols), "rank_of_covariate_matrix": int(np.linalg.matrix_rank(E)),
                   "constant_covariates": ", ".join(c for c, s in zip(cols, E.std(0)) if s == 0),
                   "n_exactly_collinear": int(np.isinf(vif).sum()),
                   "n_vif_above_10": int((vif > 10).sum()), "median_vif": float(np.nanmedian(vif[np.isfinite(vif)]))}]
                 ).to_csv(METRICS / "x15_design_summary.csv", index=False)

    blocks = grid_block_ids(coords_to_metric(df.latitude.to_numpy(), df.longitude.to_numpy()), 8)
    rng = np.random.default_rng(DRIVERS_SEED + 1)
    resamples = [_resample_blocks(blocks, rng) for _ in range(n_boot)]
    rows = []
    for term in joint.index[joint.ci_excludes_zero]:
        k = cols.index(term)
        partner = int(np.abs(corr[k]).argmax())
        X = np.c_[E[:, k], ward_dummies]
        alone = np.array([_fit(X[idx], y[idx])[0] for idx in resamples])
        rows.append({"term": term, "joint_coef_per_sd": joint.coef_per_sd[term], "joint_ci_low": joint.ci_low[term],
                     "joint_ci_high": joint.ci_high[term], "alone_coef_per_sd": _fit(X, y)[0],
                     "alone_ci_low": np.percentile(alone, 2.5), "alone_ci_high": np.percentile(alone, 97.5),
                     "strongest_correlate": cols[partner], "max_abs_r": abs(corr[k, partner]),
                     "vif_covariates": vif[k], "vif_with_ward_indicators": vif_ward[k]})
    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "x15_joint_model_collinearity.csv", index=False)
    return out


# ----------------------------------------------------------------------------- X26
def ward_polygon_icc(df, coords, n_null=1000):
    """Ward ICC with wards taken from the OpenStreetMap polygons instead of the recorded labels."""
    from src.regions import contiguity_graph, sample_partitions, verify
    from src.unsupervised import env_matrix
    from src.ward_polygons import build_wards, polygon_ward, recorded_key
    y = df.target_dried.to_numpy().astype(float)
    wards = build_wards()
    rec, poly = recorded_key(df), polygon_ward(df, wards)
    located = np.array([p is not None for p in poly])
    inside = located & (rec == poly)
    cols = explanatory_columns(df)
    E = StandardScaler().fit_transform(df[cols].fillna(df[cols].median()).to_numpy(float))
    outcomes = {"recorded drying": y}
    for lab, X in [("residual after 65 environmental covariates", E),
                   ("residual after 46 landscape covariates", env_matrix(df))]:
        outcomes[lab] = y - LogisticRegression(C=1.0, max_iter=10000).fit(X, y).predict_proba(X)[:, 1]
    systems = [("polygon ward", "sources inside a surveyed ward polygon", located, poly),
               ("recorded ward", "sources inside a surveyed ward polygon", located, rec),
               ("recorded ward", "sources inside their recorded ward polygon", inside, rec)]
    rows, conn = [], []
    for k, (unit, sample, m, lab) in enumerate(systems):
        c, g = coords[m], np.asarray(lab[m]).astype(str)
        grids = rotated_grids(c, len(np.unique(g)), n_null, SEED + 60 + k)
        regions = sample_partitions(c, g, n_null, SEED + 70 + k)
        for oname, v in outcomes.items():
            obs = icc1(v[m], g)
            for fam, parts in [("rotated matched grids", grids), ("contiguous size-matched regions", regions)]:
                rows.append({"unit": unit, "sample": sample, "n_sources": int(m.sum()), "n_units": len(np.unique(g)),
                             "outcome": oname, "null": fam, **_null_stats(obs, [icc1(v[m], p) for p in parts])})
        vv = verify(pd.factorize(g)[0], contiguity_graph(c))
        conn.append({"unit": unit, "sample": sample, "n_units": vv["n_regions"],
                     "units_not_connected_on_source_graph": vv["n_disconnected"]})
        logger.info(f"X26 {unit} | {sample}: done")
    out = pd.DataFrame(rows)
    blocks = grid_block_ids(coords, 8)[located]
    jack = _block_jackknife_icc_difference(y[located], rec[located].astype(str), poly[located].astype(str), blocks)
    diff = pd.DataFrame([{"contrast": "ICC: recorded ward - polygon ward (same sources)",
                          "interval": "delete-one-block jackknife", **jack}])
    # polygon adjacency: how many wards each ward touches
    w = wards.to_crs(32645)
    touch = [int(w.geometry.buffer(1).intersects(gm).sum() - 1) for gm in w.geometry]
    conn.append({"unit": "polygon ward", "sample": "ward polygons (all contiguous)", "n_units": len(w),
                 "units_not_connected_on_source_graph": int(sum(gm.geom_type != "Polygon" for gm in w.geometry)),
                 "median_polygon_neighbours": float(np.median(touch))})
    out.to_csv(METRICS / "x26_ward_polygon_icc.csv", index=False)
    diff.to_csv(METRICS / "x26_ward_polygon_contrast.csv", index=False)
    pd.DataFrame(conn).to_csv(METRICS / "x26_ward_connectivity.csv", index=False)
    return out


# ----------------------------------------------------------------------------- X27
def full_rank_sensitivity(df, coords, n_null=X10_N):
    """65-covariate set vs the 62-covariate full-rank set in the analyses that adjust for the environment."""
    from src.drivers_analysis import three_way_partition
    from src.features import full_rank_columns
    from src.regions import sample_partitions
    from src.robustness import earthquake_adjusted
    y = df.target_dried.to_numpy().astype(float)
    ward, mun = _ward(df), df.municipality.to_numpy()
    sets = {"65 covariates (rank 62)": explanatory_columns(df), "62 covariates (full rank)": full_rank_columns(df)}
    fams = {"rotated matched grids": rotated_grids(coords, len(np.unique(ward)), n_null, X10_SEEDS["grid"]),
            "contiguous size-matched regions": sample_partitions(coords, ward, n_null, X10_SEEDS["district"]),
            "contiguous size-matched regions within municipality": sample_partitions(
                coords, ward, n_null, X10_SEEDS["nested"], kind="nested", groups=mun)}
    icc_rows, part_rows, eq_rows = [], [], []
    for name, cols in sets.items():
        E = StandardScaler().fit_transform(df[cols].fillna(df[cols].median()).to_numpy(float))
        r = y - LogisticRegression(C=1.0, max_iter=10000).fit(E, y).predict_proba(E)[:, 1]
        obs = icc1(r, ward)
        for fam, parts in fams.items():
            icc_rows.append({"covariate_set": name, "n_covariates": len(cols),
                             "design_rank": int(np.linalg.matrix_rank(np.c_[np.ones(len(E)), E])) - 1,
                             "null": fam, **_null_stats(obs, [icc1(r, p) for p in parts])})
        part, _ = three_way_partition(df, coords, cols=cols, write=False)
        part_rows.append(part.set_index("fraction").adj_pseudo_r2.rename(name))
        _, eq = earthquake_adjusted(df, coords, cols=cols)
        eq_rows.append(eq.assign(covariate_set=name))
        logger.info(f"X27 {name}: done")
    icc = pd.DataFrame(icc_rows)
    part = pd.concat(part_rows, axis=1).reset_index()
    eq = pd.concat(eq_rows, ignore_index=True)
    icc.to_csv(METRICS / "x27_full_rank_residual_icc.csv", index=False)
    part.to_csv(METRICS / "x27_full_rank_partition.csv", index=False)
    eq.to_csv(METRICS / "x27_full_rank_earthquake.csv", index=False)
    return icc


# ----------------------------------------------------------------------------- X28
def _latent_null(v, parts, offset=None, n_jobs=12):
    from joblib import Parallel, delayed
    from src.latent_icc import latent_icc
    if offset is None:
        return [latent_icc(v, p) for p in parts]
    return Parallel(n_jobs=n_jobs)(delayed(latent_icc)(v, p, offset) for p in parts)


def latent_scale_icc(df, coords, n_null=X10_N, n_pop=200):
    """The observed-scale ICC tests of X10, X13 and the sensitivity table, repeated on the latent scale.

    Covariate adjustment: on the observed scale the ICC of residuals y - p (as in X10); on the latent scale the
    pooled logistic linear predictor enters as an offset of the random-intercept model."""
    from src.hydro_gis import MATCHED
    from src.latent_icc import latent_icc
    from src.regions import sample_partitions
    from src.robustness import inside_recorded_palika
    from src.unsupervised import env_matrix
    rows = []

    def add(block, subset, outcome, unit, fam, obs_obs, obs_lat, null_obs, null_lat, **extra):
        rows.append({"block": block, "subset": subset, "outcome": outcome, "unit": unit, "null": fam, **extra,
                     **{f"observed_{k}": v for k, v in _null_stats(obs_obs, null_obs).items()},
                     **{f"latent_{k}": v for k, v in _null_stats(obs_lat, null_lat).items()}})

    # (1) ward ICC against the three 1,000-draw families (X10), all sources and springs only
    cols = explanatory_columns(df)
    reason = df.reason_of_drying.to_numpy()
    yr = pd.to_numeric(df["years_since_dried"], errors="coerce").where(lambda v: (v >= 2000) & (v <= 2085)).to_numpy()
    for sname, m in {"all sources": np.ones(len(df), bool),
                     "springs only": (df["Source Type_encoded"] == 0).to_numpy()}.items():
        d = df[m].reset_index(drop=True)
        c, w, mun = coords[m], _ward(d), d.municipality.to_numpy()
        y = d.target_dried.to_numpy().astype(float)
        dm = y == 1
        allrows = np.ones(len(d), bool)
        E = StandardScaler().fit_transform(d[cols].fillna(d[cols].median()).to_numpy(float))
        fams = {"rotated matched grids": rotated_grids(c, len(np.unique(w)), n_null, X10_SEEDS["grid"]),
                "contiguous size-matched regions": sample_partitions(c, w, n_null, X10_SEEDS["district"]),
                "contiguous size-matched regions within municipality": sample_partitions(
                    c, w, n_null, X10_SEEDS["nested"], kind="nested", groups=mun)}
        # outcome: (latent-scale y, offset or None, observed-scale values, rows used)
        outcomes = {"recorded drying": (y, None, y, allrows)}
        season = (d["Perennial / Seasonal_encoded"].to_numpy() == 0).astype(float)
        outcomes["seasonality flag"] = (season, None, season, allrows)
        for lab, X in [("adjusted for 65 environmental covariates", E),
                       ("adjusted for 46 landscape covariates", env_matrix(d))]:
            fit = LogisticRegression(C=1.0, max_iter=10000).fit(X, y)
            outcomes[lab] = (y, fit.decision_function(X), y - fit.predict_proba(X)[:, 1], allrows)
        for lab, v in [("earthquake attribution", reason[m] == "earthquake"),
                       ("drought attribution", reason[m] == "drought"),
                       ("infrastructure attribution", reason[m] == "infrastructure_development"),
                       ("reported year value coded as 2072 BS", yr[m] == 2072)]:
            vv = v[dm].astype(float)
            outcomes[lab] = (vv, None, vv, dm)
        for fam, parts in fams.items():
            for oname, (v, offset, ov, sel) in outcomes.items():
                pp = [np.asarray(p)[sel] for p in parts]
                add("ward vs reference families", sname, oname, "ward", fam,
                    icc1(ov, w[sel]), latent_icc(v, w[sel], offset),
                    [icc1(ov, p) for p in pp], _latent_null(v, pp, offset))
            logger.info(f"X28 (1) {sname} | {fam}: done")

    # (2) wards and DEM catchments against rotated grids with as many occupied cells (X13), and jackknife contrasts
    y = df.target_dried.to_numpy().astype(float)
    matched = pd.read_csv(MATCHED).set_index("id").reindex(df.id)
    units = {"ward": _ward(df), "catchment T=0.5 km2": df.catch_T0p5.to_numpy(),
             "catchment T=2 km2": df.catch_T2p0.to_numpy(), "catchment T=10 km2": df.catch_T10p0.to_numpy(),
             "catchment count-matched": matched.catch_matched.to_numpy()}
    for k, (name, g) in enumerate(units.items()):
        grids = rotated_grids(coords, len(np.unique(g)), n_null, SEED + 20 + k)
        add("units vs rotated grids", "all sources", "recorded drying", name, "rotated matched grids",
            icc1(y, g), latent_icc(y, g), [icc1(y, p) for p in grids], _latent_null(y, grids),
            n_units=len(np.unique(g)))
    logger.info("X28 (2) units: done")
    blocks = grid_block_ids(coords, 8)
    contrasts = []
    for name in ["catchment T=2 km2", "catchment count-matched"]:
        for scale, fn in [("observed", icc1), ("latent", latent_icc)]:
            contrasts.append({"contrast": f"ward - {name}", "scale": scale,
                              **_block_jackknife_icc_difference(y, units["ward"], units[name], blocks, icc=fn)})

    # (3) the five sensitivity populations against 200 grids and 200 contiguous regions
    for name, m in {"all sources": np.ones(len(df), bool),
                    "springs only": (df["Source Type_encoded"] == 0).to_numpy(),
                    "perennial only": (df["Perennial / Seasonal_encoded"] == 0).to_numpy(),
                    "perennial springs": ((df["Source Type_encoded"] == 0)
                                          & (df["Perennial / Seasonal_encoded"] == 0)).to_numpy(),
                    "in recorded palika": inside_recorded_palika(df)}.items():
        c, w, yy = coords[m], _ward(df)[m], y[m]
        for fam, parts in [("rotated matched grids", rotated_grids(c, len(np.unique(w)), n_pop, SEED + 80)),
                           ("contiguous size-matched regions", sample_partitions(c, w, n_pop, SEED + 1))]:
            add("sensitivity populations", name, "recorded drying", "ward", fam,
                icc1(yy, w), latent_icc(yy, w), [icc1(yy, p) for p in parts], _latent_null(yy, parts),
                prevalence=yy.mean())
    logger.info("X28 (3) populations: done")
    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "x28_latent_icc.csv", index=False)
    pd.DataFrame(contrasts).to_csv(METRICS / "x28_latent_icc_contrasts.csv", index=False)
    return out


def run(only=None):
    only = only or ALL
    df = load_master()
    missing_catchments = [col for col in ["catch_T0p5", "catch_T2p0", "catch_T10p0"] if col not in df]
    if missing_catchments:
        hydro = pd.read_csv(HG)
        df = df.merge(hydro[["id", *missing_catchments]], on="id", how="left", validate="one_to_one")
    coords = coords_to_metric(df.latitude.to_numpy(), df.longitude.to_numpy())
    steps = {"X0": lambda: raw_field_audit(), "X1": lambda: tier_increments(), "X2": lambda: within_ward_auc(df),
             "X3": lambda: pair_placebo(df, coords), "X4": lambda: residual_ward_icc(df, coords),
             "X5": lambda: placebo_partition(df, coords), "X6": lambda: reversals_fdr(),
             "X7": lambda: cause_auc_difference(df), "X8": lambda: ward_correlates(df),
             "X9": lambda: grid_fe_survival(df, coords), "X10": lambda: strengthened_nulls(df, coords),
             "X11": lambda: fold_variation(df), "X12": lambda: pair_contrast(),
             "X13": lambda: matched_count_contrast(df, coords), "X14": lambda: pair_robustness(df, coords),
             "X15": lambda: joint_model_collinearity(df),
             "X16": lambda: partition_geometry(df, coords),
             "X17": lambda: top_decile_triage_metrics(),
             "X18": lambda: terrain_type_block_permutation(),
             "X19": lambda: within_unit_auc_by_family(df),
             "X20": lambda: coordinate_ablation(df),
             "X21": lambda: calibrate_ward_nulls(df, coords),
             "X22": lambda: reduced_covariate_model(df),
             "X23": lambda: feature_family_ablation(df),
             "X24": lambda: within_ward_auc_block_bootstrap(df),
             "X25": lambda: ascertainment_checks(df, coords),
             "X26": lambda: ward_polygon_icc(df, coords),
             "X27": lambda: full_rank_sensitivity(df, coords),
             "X28": lambda: latent_scale_icc(df, coords)}
    for k in only:
        logger.info(f"{k} ...")
        out = steps[k]()
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            logger.info("\n" + out.round(4).to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=ALL)
    run(ap.parse_args().only)
