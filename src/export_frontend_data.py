"""
Export the data behind the interactive map (frontend/src/data.json).

Every number shown by the frontend is computed here from the master table or read from results/metrics, so the
site cannot drift from the paper. Nothing is typed in by hand except labels and short descriptions.

Usage:  python -m src.export_frontend_data
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping, shape

from src.boundaries import DISTRICT_FILE, PALIKA_FILE, PALIKAS
from src.ward_polygons import OSM_PALIKA, WARD_FILE, polygon_ward, recorded_key

M = PROJECT_ROOT / "results/metrics"
PARQUET_PATH = PROJECT_ROOT / "data/processed/master_dataset.parquet"
OUTPUT_JSON = PROJECT_ROOT / "frontend/src/data.json"
SIMPLIFY = 0.0005


def _r(v, d=3):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), d)


def _simplified(gj):
    for feat in gj["features"]:
        feat["geometry"] = mapping(shape(feat["geometry"]).simplify(SIMPLIFY, preserve_topology=True))
    return gj


def _rate_table(df, col, order=None):
    g = df.groupby(col, observed=False).target_dried.agg(["count", "sum"]).reset_index()
    if order is not None:
        g[col] = pd.Categorical(g[col], order, ordered=True)
        g = g.sort_values(col)
    return [{"label": str(r[col]), "total": int(r["count"]), "dried": int(r["sum"]),
             "rate": _r(100 * r["sum"] / r["count"], 1) if r["count"] else 0.0}
            for _, r in g.iterrows() if r["count"] > 0]


def build():
    df = pd.read_parquet(PARQUET_PATH)
    y = df.target_dried.astype(int)

    # --- boundaries --------------------------------------------------------------------------------------------
    district = _simplified(json.loads(Path(DISTRICT_FILE).read_text(encoding="utf-8")))
    palikas = _simplified(json.loads(Path(PALIKA_FILE).read_text(encoding="utf-8")))
    official = df.municipality.map(PALIKAS)
    pstats = df.assign(official=official).groupby("official").agg(
        total=("target_dried", "size"), dried=("target_dried", "sum"), elev=("elevation_m", "mean"),
        road=("dist_to_road_m", "mean"), springs=("source_type", lambda s: int((s == "spring").sum())))
    for feat in palikas["features"]:
        p = feat["properties"]
        if p["name"] in pstats.index:
            s = pstats.loc[p["name"]]
            p.update(surveyed=True, total_sources=int(s.total), dried_sources=int(s.dried),
                     drying_rate=_r(100 * s.dried / s.total, 1), mean_elev=int(round(s.elev)),
                     mean_road=int(round(s.road)))
        else:
            p.update(surveyed=False, total_sources=0, dried_sources=0, drying_rate=None)

    # --- ward polygons (OpenStreetMap) with the agreement of recorded labels ----------------------------------
    wards = gpd.read_file(WARD_FILE)
    rec, poly = recorded_key(df), polygon_ward(df, wards)
    inside = pd.Series(rec == poly).groupby(rec).mean()
    n_rec = pd.Series(rec).value_counts()
    rate_rec = pd.Series(y.to_numpy()).groupby(rec).mean()
    back = {v: k for k, v in OSM_PALIKA.items()}
    ward_features = []
    for _, w in wards.iterrows():
        key = f"{w.palika}-{w.ward}"
        ward_features.append({"type": "Feature", "geometry": mapping(w.geometry.simplify(SIMPLIFY / 2, preserve_topology=True)),
                              "properties": {"palika": PALIKAS.get(back.get(w.palika, ""), w.palika), "ward": int(w.ward),
                                             "n_recorded": int(n_rec.get(key, 0)),
                                             "share_inside": _r(inside.get(key, np.nan), 3),
                                             "recorded_rate": _r(100 * rate_rec.get(key, np.nan), 1)}})
    ward_gj = {"type": "FeatureCollection", "features": ward_features}

    # --- sources: [lat, lon, dried, elevation, official palika, type, recorded ward] -------------------------
    sources = [[_r(r.latitude, 4), _r(r.longitude, 4), int(r.target_dried), int(round(r.elevation_m)),
                PALIKAS.get(r.municipality, r.municipality), str(r.source_type), int(r.ward)]
               for r in df.itertuples()]

    # --- descriptive charts -----------------------------------------------------------------------------------
    elev_labels = ["< 1,000 m", "1,000-1,300 m", "1,300-1,600 m", "1,600-1,900 m", "> 1,900 m"]
    road_labels = ["< 50 m", "50-100 m", "100-250 m", "250-500 m", "> 500 m"]
    df["elev_bin"] = pd.cut(df.elevation_m, [-1, 1000, 1300, 1600, 1900, 9000], labels=elev_labels)
    df["road_bin"] = pd.cut(df.dist_to_road_m, [-1, 50, 100, 250, 500, 1e6], labels=road_labels)
    quad = np.select([(df.aspect_deg >= 315) | (df.aspect_deg < 45), df.aspect_deg < 135, df.aspect_deg < 225],
                     ["North", "East", "South"], "West")
    df["aspect_quad"] = quad
    lu_names = {"dense_forest": "Dense forest", "sparse_forest": "Sparse forest", "agriculture": "Agriculture",
                "grassland": "Grassland", "settlement": "Settlement", "fallow": "Fallow", "others": "Other"}
    df["landuse_label"] = df.landuse.map(lu_names).fillna("Other")
    landuse = sorted(_rate_table(df, "landuse_label"), key=lambda r: r["rate"])
    cause_names = {"earthquake": "2015 earthquake", "drought": "Drought", "infrastructure_development": "Infrastructure",
                   "natural_disaster": "Other natural disaster", "source_neglected": "Source neglected",
                   "source_disruption": "Source disruption", "deforestation": "Deforestation"}
    dried = df[df.target_dried == 1]
    cause = dried.reason_of_drying.map(cause_names).fillna("Other or not stated")
    causes = [{"label": k, "count": int(v), "pct": _r(100 * v / len(dried), 1)} for k, v in cause.value_counts().items()]
    yr = pd.to_numeric(dried.years_since_dried, errors="coerce")
    yr = yr[(yr >= 2050) & (yr <= 2081)].astype(int)
    years = [{"year": int(k), "count": int(v)} for k, v in yr.value_counts().sort_index().items()]

    # --- findings, read from the result tables -----------------------------------------------------------------
    audit = pd.read_csv(M / "x0_raw_field_audit_summary.csv").iloc[0]
    single = pd.read_csv(M / "single_feature_auc.csv").set_index("feature").single_feature_auc
    bench = pd.read_csv(M / "benchmark_summary.csv")
    tab = bench[(bench.scheme == "spatial") & (bench.buffer_m == 0) & bench.profile.isin(["main", "enriched"])
                & bench.model.isin(["Logistic Regression", "Random Forest", "CatBoost"])]
    within = pd.read_csv(M / "x2_within_ward_auc.csv").query("feature_set == 'env_hydro' and buffer_m == 0")
    within = within[within.model.isin(["Logistic Regression", "Random Forest", "CatBoost"])]
    pal_agree = pd.read_csv(M / "boundary_agreement.csv").set_index("palika").loc["all"]
    ward_agree = pd.read_csv(M / "ward_boundary_agreement.csv").set_index("palika").loc["all"]
    lat = pd.read_csv(M / "x28_latent_icc.csv")
    con = pd.read_csv(M / "x28_latent_icc_contrasts.csv")
    reduced = pd.read_csv(M / "x22_reduced_covariates.csv")

    units = lat[lat.block == "units vs rotated grids"].set_index("unit")
    unit_rows = []
    for key, label in [("ward", "Wards (75)"), ("catchment T=0.5 km2", "Catchments, 0.5 km²"),
                       ("catchment T=2 km2", "Catchments, 2 km²"), ("catchment count-matched", "Catchments, matched to 75"),
                       ("catchment T=10 km2", "Catchments, 10 km²")]:
        u = units.loc[key]
        unit_rows.append({"unit": label, "observed": _r(u.observed_obs), "observed_beats_grids": bool(u.observed_n_exceed == 0),
                          "latent": _r(u.latent_obs), "latent_beats_grids": bool(u.latent_n_exceed == 0),
                          "latent_grid_exceed": int(u.latent_n_exceed)})
    fam = lat[(lat.block == "ward vs reference families") & (lat.subset == "all sources")]
    fam_rows = []
    names = {"rotated matched grids": "Rotated grids", "contiguous size-matched regions": "Contiguous random regions",
             "contiguous size-matched regions within municipality": "Regions within municipality"}
    for outcome, olabel in [("recorded drying", "Recorded drying"),
                            ("adjusted for 65 environmental covariates", "After 65 covariates")]:
        for null, nlabel in names.items():
            r = fam[(fam.outcome == outcome) & (fam["null"] == null)].iloc[0]
            fam_rows.append({"outcome": olabel, "reference": nlabel,
                             "observed_exceed": int(r.observed_n_exceed), "latent_exceed": int(r.latent_n_exceed),
                             "n": int(r.latent_n_null)})
    contrasts = [{"contrast": c.contrast.replace("ward - catchment count-matched", "Wards minus catchments matched to 75")
                  .replace("ward - catchment T=2 km2", "Wards minus 2 km² catchments"),
                  "scale": c.scale, "difference": _r(c.difference), "low": _r(c.ci_low), "high": _r(c.ci_high)}
                 for c in con.itertuples()]
    within_ward = reduced[(reduced.model.str.startswith("within")) & reduced.ci_excludes_zero]
    drivers = [{"term": {"tpi_500m": "Topographic position (TPI, 500 m)", "chili": "Heat load (CHILI)"}.get(t.term, t.term),
                "beta": _r(t.coef_per_sd, 2), "low": _r(t.ci_low, 2), "high": _r(t.ci_high, 2)}
               for t in within_ward.itertuples()]

    findings = {
        "skipLogic": {"fields": int(audit.n_fields), "outcomeDependent": int(audit.n_outcome_dependent),
                      "completionIndicatorAuc": _r(single.get("earthquake_impact_known"))},
        "prediction": {"aucLow": _r(tab.roc_auc_mean.min(), 2), "aucHigh": _r(tab.roc_auc_mean.max(), 2),
                       "withinWardLow": _r(within.within_ward_auc.min(), 2),
                       "withinWardHigh": _r(within.within_ward_auc.max(), 2)},
        "location": {"insidePalika": _r(100 * pal_agree.inside_recorded_palika, 1),
                     "insideWard": _r(100 * ward_agree.inside_recorded_ward, 1),
                     "within100m": _r(100 * ward_agree.within_100m_of_recorded_ward, 1),
                     "adjacentWard": _r(100 * ward_agree.outside_in_adjacent_ward, 1),
                     "wardsUnderHalf": int(ward_agree.wards_under_50pct_inside)},
        "icc": {"units": unit_rows, "references": fam_rows, "contrasts": contrasts},
        "withinWardDrivers": drivers,
    }

    return {
        "totalSources": int(len(df)), "totalDried": int(y.sum()), "dryingRate": _r(100 * y.mean(), 1),
        "springs": int((df.source_type == "spring").sum()), "ponds": int((df.source_type == "pond").sum()),
        "district": district, "palikas": palikas, "wards": ward_gj, "sources": sources,
        "charts": {"elevation": _rate_table(df, "elev_bin", elev_labels), "road": _rate_table(df, "road_bin", road_labels),
                   "landuse": landuse, "aspect": _rate_table(df, "aspect_quad", ["North", "East", "South", "West"]),
                   "causes": causes, "years": years},
        "findings": findings,
        "credits": {
            "dataProvider": "International Centre for Integrated Mountain Development (ICIMOD)",
            "dataTitle": "Status of Springs in Seven Municipalities of Kavre, Nepal",
            "dataDoi": "https://doi.org/10.26066/RDS.1974131", "dataLicence": "CC BY 4.0",
            "authors": ["Ashim Shrestha", "Kushal Kunwar"],
            "organization": "Noesis Frameworks", "orgUrl": "https://noesisframeworks.com",
            "githubUrl": "https://github.com/ashimst/kavre_springs", "userGithub": "https://github.com/ashimst",
        },
    }


if __name__ == "__main__":
    payload = build()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Exported {OUTPUT_JSON} ({OUTPUT_JSON.stat().st_size / 1024:.0f} KB)")
