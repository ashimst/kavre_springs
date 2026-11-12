"""
Official administrative boundaries for the study area.

Source: geoBoundaries gbOpen NPL ADM3 (palikas: municipalities and rural municipalities),
release commit 9469f09, built from Open Data Nepal / Ministry of Federal Affairs and General Administration
(MoFAGA) boundaries (2019 vintage), licence CC BY-SA 4.0 (Runfola et al., 2020, geoBoundaries, PLoS ONE 15(4):
e0231866). The seven surveyed palikas cover 743.8 km^2, matching the study-area size reported by the data providers.

Outputs (versioned, CC BY-SA 4.0):
  data/processed/boundaries/kavre_district.geojson     Kabhrepalanchok district (union of its 13 palikas)
  data/processed/boundaries/kavre_palikas.geojson      the 13 palikas, with a flag for the 7 surveyed ones
  results/metrics/boundary_agreement.csv               recorded palika vs. palika polygon containing the source

The source files are downloaded once into data/external/geoboundaries/ (ignored by git). Official ward polygons
(Level 4) are not part of the Survey Department's free data; src/ward_polygons.py uses OpenStreetMap wards instead.

Usage:  python -m src.boundaries
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import urllib.request

import geopandas as gpd
import pandas as pd

logger = logging.getLogger("boundaries")
RELEASE = "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/NPL/{lvl}/geoBoundaries-NPL-{lvl}.geojson"
CACHE = PROJECT_ROOT / "data/external/geoboundaries"
OUT = PROJECT_ROOT / "data/processed/boundaries"
DISTRICT_FILE = OUT / "kavre_district.geojson"
PALIKA_FILE = OUT / "kavre_palikas.geojson"
METRICS = PROJECT_ROOT / "results/metrics"
# survey spelling -> official name
PALIKAS = {"Bethanchownk": "Bethanchowk", "Dhulikhel": "Dhulikhel", "Namobuudha": "Namobuddha",
           "Panauti": "Panauti", "Panchkhal": "Panchkhal", "Roshi RM": "Roshi", "Temal": "Temal"}
# the 13 palikas of Kabhrepalanchok district (names as in the source layer)
KAVRE_PALIKAS = {"Banepa": "municipality", "Dhulikhel": "municipality", "Mandandeupur": "municipality",
                 "Namobuddha": "municipality", "Panauti": "municipality", "Panchkhal": "municipality",
                 "Bethanchowk": "rural municipality", "Bhumlu": "rural municipality",
                 "Chaurideurali": "rural municipality", "Khanikhola": "rural municipality",
                 "Mahabharat": "rural municipality", "Roshi": "rural municipality", "Temal": "rural municipality"}
UA = {"User-Agent": "kavre-springs-reproduction/1.0"}


def _download(level: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"geoBoundaries-NPL-{level}.geojson"
    if not path.exists():
        url = RELEASE.format(lvl=level)
        logger.info(f"Downloading {url}")
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
            path.write_bytes(r.read())
    return path


def build_boundaries(force: bool = False) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Write the versioned district and palika layers (downloads the source layer if needed)."""
    if not force and DISTRICT_FILE.exists() and PALIKA_FILE.exists():
        return gpd.read_file(DISTRICT_FILE), gpd.read_file(PALIKA_FILE)
    adm3 = gpd.read_file(_download("ADM3"))
    pal = adm3[adm3.shapeName.isin(KAVRE_PALIKAS)][["shapeName", "shapeID", "geometry"]].rename(
        columns={"shapeName": "name"})
    assert len(pal) == len(KAVRE_PALIKAS), sorted(pal.name)
    pal["type"] = pal.name.map(KAVRE_PALIKAS)
    pal["surveyed"] = pal.name.isin(PALIKAS.values())
    pal["area_km2"] = pal.to_crs(32645).area / 1e6
    # the geoBoundaries ADM2 layer is heavily generalised; the district is the union of its 13 palikas
    district = gpd.GeoDataFrame({"name": ["Kabhrepalanchok"], "n_palikas": [len(pal)],
                                 "area_km2": [pal.area_km2.sum()]},
                                geometry=[pal.geometry.union_all()], crs=adm3.crs)
    OUT.mkdir(parents=True, exist_ok=True)
    attribution = "geoBoundaries gbOpen NPL ADM3 (commit 9469f09); Open Data Nepal / MoFAGA; CC BY-SA 4.0"
    for g, f in [(district, DISTRICT_FILE), (pal, PALIKA_FILE)]:
        g = g.assign(source=attribution).sort_values("name").reset_index(drop=True)
        g.to_file(f, driver="GeoJSON")
    logger.info(f"Wrote {DISTRICT_FILE.name} ({district.area_km2[0]:.0f} km^2) and {PALIKA_FILE.name} "
                f"(surveyed palikas {pal[pal.surveyed].area_km2.sum():.1f} km^2)")
    return district, pal


def boundary_agreement(df: pd.DataFrame, pal: gpd.GeoDataFrame) -> pd.DataFrame:
    """Share of sources lying inside the polygon of their recorded palika, and where the others lie."""
    pts = gpd.GeoDataFrame(df[["id", "municipality"]].copy(),
                           geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
    pts["recorded"] = pts.municipality.map(PALIKAS)
    j = gpd.sjoin(pts, pal[["name", "geometry"]], how="left", predicate="within").rename(columns={"name": "polygon"})
    j = j[~j.index.duplicated()]
    pm = pal.to_crs(32645).set_index("name")
    jm = j.to_crs(32645)
    j["dist_to_recorded_m"] = [0.0 if r == c else float(pm.loc[r].geometry.boundary.distance(g))
                               for r, c, g in zip(jm.recorded, jm.polygon, jm.geometry)]
    j["inside_recorded"] = j.recorded == j.polygon
    surveyed = set(PALIKAS.values())
    rows = []
    for name, g in list(j.groupby("recorded")) + [("all", j)]:
        out = g[~g.inside_recorded]
        rows.append({"palika": name, "n_sources": len(g), "inside_recorded_palika": g.inside_recorded.mean(),
                     "n_outside": len(out),
                     "in_other_surveyed_palika": int(out.polygon.isin(surveyed).sum()),
                     "in_non_surveyed_kavre_palika": int((out.polygon.notna() & ~out.polygon.isin(surveyed)).sum()),
                     "outside_district": int(out.polygon.isna().sum()),
                     "median_dist_outside_m": out.dist_to_recorded_m.median() if len(out) else 0.0,
                     "share_outside_within_500m": (out.dist_to_recorded_m <= 500).mean() if len(out) else 1.0})
    return pd.DataFrame(rows)


def run():
    from src.dataset import load_master
    _, pal = build_boundaries()
    agree = boundary_agreement(load_master(), pal)
    METRICS.mkdir(parents=True, exist_ok=True)
    agree.to_csv(METRICS / "boundary_agreement.csv", index=False)
    logger.info("\n" + agree.round(3).to_string())
    return agree


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()
