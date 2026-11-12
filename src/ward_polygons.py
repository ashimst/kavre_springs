"""
Ward boundary polygons for the seven surveyed palikas, from OpenStreetMap.

Source: OpenStreetMap boundary relations tagged boundary=administrative, admin_level=9, named "<Palika>-<ward>"
(Nepal's convention for wards), downloaded through the Overpass API. OpenStreetMap data are (c) OpenStreetMap
contributors, ODbL 1.0. The layer is community-mapped, not an official release: the Survey Department's ward level
(Level 4) is not part of its free data. The 75 relations cover the 75 wards of the seven palikas; their union matches
the official palika areas (geoBoundaries / MoFAGA, src/boundaries.py) to within 0.5% per palika.

Outputs:
  data/processed/boundaries/kavre_wards_osm.geojson   75 ward polygons (palika, ward, area_km2, osm_id)
  results/metrics/ward_boundary_agreement.csv          recorded ward vs ward polygon containing each source
  results/metrics/ward_label_crosstab.csv              recorded ward x polygon ward counts (non-zero cells)

The Overpass response is cached in data/external/osm_wards/ (ignored by git); the processed layer is versioned so
results do not depend on later OSM edits.

Usage:  python -m src.ward_polygons
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import re
import urllib.parse
import urllib.request

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union

logger = logging.getLogger("ward_polygons")
CACHE = PROJECT_ROOT / "data/external/osm_wards/osm_kavre_wards_overpass.json"
WARD_FILE = PROJECT_ROOT / "data/processed/boundaries/kavre_wards_osm.geojson"
METRICS = PROJECT_ROOT / "results/metrics"
OVERPASS = "https://overpass-api.de/api/interpreter"
UA = {"User-Agent": "kavre-springs-reproduction/1.0"}
# survey spelling -> OSM relation-name prefix
OSM_PALIKA = {"Bethanchownk": "Bethanchok", "Dhulikhel": "Dhulikhel", "Namobuudha": "Namobuddha",
              "Panauti": "Paunauti", "Panchkhal": "Panchkhal", "Roshi RM": "Roshi", "Temal": "Temal"}
QUERY = ('[out:json][timeout:180];rel["boundary"="administrative"]["admin_level"="9"]'
         '["name"~"^(' + "|".join(OSM_PALIKA.values()) + ')-"](27.35,85.35,27.85,85.95);out body;>;out skel qt;')
ATTRIBUTION = "OpenStreetMap contributors, ODbL 1.0 (admin_level=9 ward relations via Overpass API)"


def _download() -> dict:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(OVERPASS, data=urllib.parse.urlencode({"data": QUERY}).encode(), headers=UA)
        logger.info("Downloading ward relations from the Overpass API")
        with urllib.request.urlopen(req, timeout=300) as r:
            CACHE.write_bytes(r.read())
    return json.loads(CACHE.read_text(encoding="utf-8"))


def build_wards(force: bool = False) -> gpd.GeoDataFrame:
    """Assemble the 75 ward polygons from the relation member ways and write the versioned layer."""
    if not force and WARD_FILE.exists():
        return gpd.read_file(WARD_FILE)
    el = _download()["elements"]
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in el if e["type"] == "node"}
    ways = {e["id"]: [nodes[n] for n in e["nodes"] if n in nodes] for e in el if e["type"] == "way"}
    rows = []
    for r in (e for e in el if e["type"] == "relation"):
        lines = [LineString(ways[m["ref"]]) for m in r["members"]
                 if m["type"] == "way" and m.get("role", "outer") in ("outer", "") and len(ways.get(m["ref"], [])) > 1]
        palika, ward = re.match(r"(.+)-(\d+)$", r["tags"]["name"]).groups()
        rows.append({"palika": palika, "ward": int(ward), "osm_id": r["id"],
                     "geometry": unary_union(list(polygonize(unary_union(lines))))})
    g = gpd.GeoDataFrame(rows, crs=4326).sort_values(["palika", "ward"]).reset_index(drop=True)
    assert len(g) == 75 and not g.geometry.is_empty.any(), len(g)
    g["area_km2"] = g.to_crs(32645).area / 1e6
    g["source"] = ATTRIBUTION
    WARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(WARD_FILE, driver="GeoJSON")
    logger.info(f"Wrote {WARD_FILE.name}: {len(g)} wards, {g.area_km2.sum():.1f} km^2")
    return g


def recorded_key(df: pd.DataFrame) -> np.ndarray:
    return (df.municipality.map(OSM_PALIKA) + "-" + df.ward.astype(int).astype(str)).to_numpy()


def polygon_ward(df: pd.DataFrame, wards: gpd.GeoDataFrame) -> np.ndarray:
    """'<palika>-<ward>' of the polygon containing each source; None outside the 75 surveyed wards."""
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
    j = gpd.sjoin(pts, wards[["palika", "ward", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()].reindex(pts.index)
    key = j.palika + "-" + j.ward.astype("Int64").astype(str)
    return key.where(j.palika.notna(), None).to_numpy(dtype=object)


def ward_agreement(df: pd.DataFrame, wards: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rec, poly = recorded_key(df), polygon_ward(df, wards)
    geom = dict(zip(wards.palika + "-" + wards.ward.astype(str), wards.to_crs(32645).geometry))
    pts = gpd.GeoSeries(gpd.points_from_xy(df.longitude, df.latitude), crs=4326).to_crs(32645)
    dist = np.array([geom[k].distance(p) for k, p in zip(rec, pts)])
    inside = rec == poly
    adjacent = np.array([p is not None and not i and geom[r].buffer(1).intersects(geom[p])
                         for r, p, i in zip(rec, poly, inside)])
    ct = pd.crosstab(rec, pd.Series(poly).fillna("outside surveyed wards").to_numpy())
    modal = ct.drop(columns="outside surveyed wards", errors="ignore").max(axis=1)
    share_inside_by_ward = pd.Series(inside).groupby(rec).mean()
    pal = df.municipality.map(OSM_PALIKA).to_numpy()
    rows = []
    for name, m in [(p, pal == p) for p in sorted(set(pal))] + [("all", np.ones(len(df), bool))]:
        out = m & ~inside
        wards_here = [w for w in share_inside_by_ward.index if name == "all" or w.startswith(name + "-")]
        rows.append({"palika": name, "n_sources": int(m.sum()), "inside_recorded_ward": inside[m].mean(),
                     "inside_recorded_palika": np.mean([p is not None and p.rsplit("-", 1)[0] == q
                                                        for p, q in zip(poly[m], pal[m])]),
                     "outside_surveyed_wards": np.mean([p is None for p in poly[m]]),
                     "within_100m_of_recorded_ward": (dist[m] <= 100).mean(),
                     "within_500m_of_recorded_ward": (dist[m] <= 500).mean(),
                     "outside_in_adjacent_ward": adjacent[out].mean() if out.any() else np.nan,
                     "median_dist_outside_m": np.median(dist[out]) if out.any() else 0.0,
                     "in_modal_polygon_ward": modal[wards_here].sum() / m.sum(),
                     "wards_under_50pct_inside": int((share_inside_by_ward[wards_here] < 0.5).sum())})
    cells = ct.stack().rename("n").reset_index()
    cells.columns = ["recorded_ward", "polygon_ward", "n"]
    cells = cells[cells.n > 0].sort_values(["recorded_ward", "n"], ascending=[True, False])
    return pd.DataFrame(rows), cells


def run():
    from src.dataset import load_master
    wards = build_wards()
    agree, cells = ward_agreement(load_master(), wards)
    METRICS.mkdir(parents=True, exist_ok=True)
    agree.to_csv(METRICS / "ward_boundary_agreement.csv", index=False)
    cells.to_csv(METRICS / "ward_label_crosstab.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        logger.info("\n" + agree.round(3).to_string())
    return agree


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()
