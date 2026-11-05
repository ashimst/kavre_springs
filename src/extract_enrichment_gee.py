"""
Enrichment covariates from Google Earth Engine (all label-free, independent of the survey).

Layer (Earth Engine asset)                                  -> columns
ESA WorldCover v200, 2021, 10 m (ESA/WorldCover/v200)       -> wc_tree, wc_shrub, wc_grass, wc_crop, wc_built, wc_bare,
                                                               wc_water: class fractions within 250 m
Hansen Global Forest Change v1.13 (UMD/hansen/...2025_v1_13) -> gfc_treecover2000 (mean %, 250 m), gfc_loss_pre2015,
                                                               gfc_loss_post2015 (fraction of 250 m area lost 2001-14 / 2015-23)
SoilGrids 250 m (projects/soilgrids-isric/*_mean)           -> soil_clay, soil_sand, soil_soc, soil_bdod (0-30 cm mean)
MODIS MOD13Q1 v6.1 NDVI, 250 m                              -> ndvi_trend (dry season Nov-Apr, per year, 2001-2023),
                                                               ndvi_change_2015 (2016-2020 minus 2010-2014 dry-season mean)
MODIS MOD16A2GF v6.1 ET, 500 m                              -> et_mean (mm/yr, 2001-2023), et_trend (mm/yr per yr)
CHIRPS pentad, 0.05 deg                                     -> chirps_trend (annual mm per yr, 1981-2023),
                                                               chirps_anom_2015 (2015-2016 mean / 1981-2013 mean - 1)
GHSL GHS-BUILT-S R2023A, 100 m                              -> built_2020 (built-up fraction within 500 m),
                                                               built_change (2020 minus 2000 fraction)
WorldPop 100 m, Nepal 2020                                  -> pop_1km (persons within 1 km)
VIIRS DNB annual v2.2, 2022                                 -> nightlights
ALOS landforms / CHILI / mTPI / topographic diversity       -> landform (class code), landform_position (1 valley ...
                                                               4 ridge/peak), chili, mtpi, topo_diversity

Output: data/processed/gee_enrichment_features.csv (one row per source id)
Usage:  python -m src.extract_enrichment_gee     (requires an authenticated Earth Engine account)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

OUTPUT_CSV = PROJECT_ROOT / "data/processed/gee_enrichment_features.csv"
MASTER_PARQUET = PROJECT_ROOT / "data/processed/master_dataset.parquet"
WC_CLASSES = {"wc_tree": 10, "wc_shrub": 20, "wc_grass": 30, "wc_crop": 40, "wc_built": 50, "wc_bare": 60, "wc_water": 80}
# ALOS landform classes -> slope position (Theobald et al. 2015)
LANDFORM_POSITION = {11: 4, 12: 4, 13: 4, 14: 4, 15: 3, 21: 3, 22: 3, 23: 3, 24: 3, 31: 2, 32: 2, 33: 2, 34: 2, 41: 1, 42: 1}


def build_layers(region):
    """Return {name: (image, scale_m, buffer_m, reducer)}; each image has the output column names as bands.
    buffer_m = None: value at the point; otherwise the reducer ('mean' or 'sum') over a circular buffer. Buffered
    statistics are computed with reduceRegions over the buffers only (a moving-window reduceNeighborhood at 10 m is
    prohibitively slow in Earth Engine)."""
    import ee
    L = {}

    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    L["worldcover"] = (ee.Image.cat([wc.eq(v).rename(k) for k, v in WC_CLASSES.items()]), 10, 250, "mean")

    gfc = ee.Image("UMD/hansen/global_forest_change_2025_v1_13")
    ly = gfc.select("lossyear")
    L["hansen"] = (ee.Image.cat([gfc.select("treecover2000").rename("gfc_treecover2000"),
                                 ly.gte(1).And(ly.lte(14)).rename("gfc_loss_pre2015"),
                                 ly.gte(15).And(ly.lte(23)).rename("gfc_loss_post2015")]).unmask(0), 30, 250, "mean")

    soil = []
    for prop, name in [("clay", "soil_clay"), ("sand", "soil_sand"), ("soc", "soil_soc"), ("bdod", "soil_bdod")]:
        img = ee.Image(f"projects/soilgrids-isric/{prop}_mean")
        soil.append(img.select([f"{prop}_0-5cm_mean", f"{prop}_5-15cm_mean", f"{prop}_15-30cm_mean"])
                    .reduce(ee.Reducer.mean()).rename(name))
    L["soilgrids"] = (ee.Image.cat(soil), 250, None, None)

    def dry_season_ndvi(y):
        y = ee.Number(y)
        start = ee.Date.fromYMD(y.subtract(1), 11, 1)
        img = (ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start, start.advance(6, "month"))
               .select("NDVI").mean().multiply(0.0001))
        return img.rename("ndvi").addBands(ee.Image.constant(y).toFloat().rename("year")).set("year", y)

    nd = ee.ImageCollection(ee.List.sequence(2001, 2023).map(dry_season_ndvi))
    trend = nd.select(["year", "ndvi"]).reduce(ee.Reducer.linearFit()).select("scale").rename("ndvi_trend")
    pre = nd.filter(ee.Filter.rangeContains("year", 2010, 2014)).select("ndvi").mean()
    post = nd.filter(ee.Filter.rangeContains("year", 2016, 2020)).select("ndvi").mean()
    L["modis_ndvi"] = (ee.Image.cat([trend, post.subtract(pre).rename("ndvi_change_2015")]), 250, None, None)

    def annual_et(y):
        y = ee.Number(y)
        s = ee.Date.fromYMD(y, 1, 1)
        img = ee.ImageCollection("MODIS/061/MOD16A2GF").filterDate(s, s.advance(1, "year")).select("ET").sum().multiply(0.1)
        return img.rename("et").addBands(ee.Image.constant(y).toFloat().rename("year"))

    et = ee.ImageCollection(ee.List.sequence(2001, 2023).map(annual_et))
    L["modis_et"] = (ee.Image.cat([et.select("et").mean().rename("et_mean"),
                                   et.select(["year", "et"]).reduce(ee.Reducer.linearFit()).select("scale").rename("et_trend")]),
                       500, None, None)

    def annual_p(y):
        y = ee.Number(y)
        s = ee.Date.fromYMD(y, 1, 1)
        img = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD").filterDate(s, s.advance(1, "year")).select("precipitation").sum()
        return img.rename("p").addBands(ee.Image.constant(y).toFloat().rename("year")).set("year", y)

    pc = ee.ImageCollection(ee.List.sequence(1981, 2023).map(annual_p))
    base = pc.filter(ee.Filter.rangeContains("year", 1981, 2013)).select("p").mean()
    anom = pc.filter(ee.Filter.rangeContains("year", 2015, 2016)).select("p").mean().divide(base).subtract(1)
    L["chirps"] = (ee.Image.cat([pc.select(["year", "p"]).reduce(ee.Reducer.linearFit()).select("scale").rename("chirps_trend"),
                                 anom.rename("chirps_anom_2015")]), 5566, None, None)

    gh = ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S")
    b2020 = gh.filter(ee.Filter.eq("system:index", "2020")).first().select("built_surface").divide(10000)
    b2000 = gh.filter(ee.Filter.eq("system:index", "2000")).first().select("built_surface").divide(10000)
    L["ghsl"] = (ee.Image.cat([b2020.rename("built_2020"), b2020.subtract(b2000).rename("built_change")]), 100, 500, "mean")

    pop = (ee.ImageCollection("WorldPop/GP/100m/pop").filter(ee.Filter.eq("country", "NPL"))
           .filter(ee.Filter.eq("year", 2020)).first().select("population").unmask(0))
    L["worldpop"] = (pop.rename("pop_1km"), 100, 1000, "sum")

    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/ANNUAL_V22").filterDate("2022-01-01", "2023-01-01").first().select("average")
    L["viirs"] = (viirs.rename("nightlights"), 500, None, None)

    lf = ee.Image("CSP/ERGo/1_0/Global/ALOS_landforms").select(0).rename("landform")
    pos = lf.remap(list(LANDFORM_POSITION), list(LANDFORM_POSITION.values())).rename("landform_position")
    L["alos"] = (ee.Image.cat([lf, pos,
                               ee.Image("CSP/ERGo/1_0/Global/ALOS_CHILI").select(0).rename("chili"),
                               ee.Image("CSP/ERGo/1_0/Global/ALOS_mTPI").select(0).rename("mtpi"),
                               ee.Image("CSP/ERGo/1_0/Global/ALOS_topoDiversity").select(0).rename("topo_diversity")]),
                   90, None, None)
    return L


def extract(batch_size: int = 1000) -> pd.DataFrame:
    import ee
    ee.Initialize()
    df = pd.read_parquet(MASTER_PARQUET)[["id", "latitude", "longitude"]]
    region = ee.Geometry.Rectangle([df.longitude.min() - 0.05, df.latitude.min() - 0.05,
                                    df.longitude.max() + 0.05, df.latitude.max() + 0.05])
    layers = build_layers(region)
    out = df[["id"]].copy()
    for name, (img, scale, buf, red) in layers.items():
        t0, rows = time.time(), []
        try:
            for s in range(0, len(df), batch_size):
                sub = df.iloc[s:s + batch_size]
                fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.longitude), float(r.latitude)]),
                                                      {"id": int(r.id)}) for r in sub.itertuples()])
                if buf is None:
                    res = img.sampleRegions(collection=fc, scale=scale, geometries=False).getInfo()
                else:
                    reducer = ee.Reducer.mean() if red == "mean" else ee.Reducer.sum()
                    if len(img.bandNames().getInfo()) == 1:
                        reducer = reducer.setOutputs(img.bandNames())
                    res = img.reduceRegions(collection=fc.map(lambda f: f.buffer(buf)), reducer=reducer, scale=scale,
                                            tileScale=4).getInfo()
                rows += [{k: v for k, v in f["properties"].items()} for f in res["features"]]
            got = pd.DataFrame(rows)
            out = out.merge(got, on="id", how="left")
            print(f"  {name:12s} {len(got):5d} points, columns {[c for c in got.columns if c != 'id']} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:  # keep the other layers if one dataset is unavailable
            print(f"  {name:12s} FAILED: {str(e)[:300]}", flush=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV} ({out.shape[1] - 1} columns)", flush=True)
    return out


if __name__ == "__main__":
    extract()
