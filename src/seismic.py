"""
Seismic covariates for every water source from public USGS products (no labels involved).

* USGS ShakeMap grids (grid.xml) for the 25 April 2015 Mw 7.8 Gorkha mainshock (us20002926)
  and the 12 May 2015 Mw 7.3 Dolakha aftershock (us20002ejl): peak ground acceleration (PGA, %g),
  peak ground velocity (PGV, cm/s) and macroseismic intensity (MMI), bilinearly interpolated
  at each source.
* USGS ComCat aftershocks M >= 4.0, 25 Apr - 31 Dec 2015, within 150 km: count within 20 km
  and distance to the nearest event; distance to both epicentres.

These are a-priori observable (independent of the survey) and may legitimately be used as
predictors. Downloads are cached in data/external/usgs/ and are parsed as data only.

Usage: python -m src.seismic
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger("seismic")
CACHE = PROJECT_ROOT / "data/external/usgs"
OUT = PROJECT_ROOT / "data/processed/seismic_features.csv"
MASTER = PROJECT_ROOT / "data/processed/master_dataset.parquet"
UA = {"User-Agent": "kavre-springs-research/1.0"}
EVENTS = {"main": "us20002926", "may12": "us20002ejl"}
SEISMIC_FEATURES = ["pga_main", "pgv_main", "mmi_main", "pga_may12", "pgv_may12", "mmi_may12",
                    "pga_max", "dist_epi_main_km", "dist_epi_may12_km", "aftershocks_20km", "dist_nearest_aftershock_km"]


def _get(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading {url}")
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
            path.write_bytes(r.read())
    return path


def event_detail(eid: str) -> dict:
    p = _get(f"https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={eid}&format=geojson", CACHE / f"{eid}.json")
    return json.loads(p.read_text(encoding="utf-8"))


def shakemap_grid(eid: str):
    det = event_detail(eid)
    url = det["properties"]["products"]["shakemap"][0]["contents"]["download/grid.xml"]["url"]
    p = _get(url, CACHE / f"{eid}_grid.xml")
    root = ET.parse(p).getroot()
    ns = {"sm": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
    q = (lambda t: f"sm:{t}") if ns else (lambda t: t)
    spec = root.find(q("grid_specification"), ns).attrib
    fields = {f.attrib["name"].lower(): int(f.attrib["index"]) - 1 for f in root.findall(q("grid_field"), ns)}
    data = np.array(root.find(q("grid_data"), ns).text.split(), dtype=float)
    nlon, nlat = int(spec["nlon"]), int(spec["nlat"])
    data = data.reshape(nlat * nlon, len(fields))
    lons = data[:, fields["lon"]].reshape(nlat, nlon)[0]
    lats = data[:, fields["lat"]].reshape(nlat, nlon)[:, 0]
    out = {}
    for name in ("pga", "pgv", "mmi"):
        grid = data[:, fields[name]].reshape(nlat, nlon)
        order = np.argsort(lats)
        out[name] = RegularGridInterpolator((lats[order], lons), grid[order], bounds_error=False, fill_value=np.nan)
    coords = det["geometry"]["coordinates"]
    return out, (coords[1], coords[0])


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def aftershocks(center_lat: float, center_lon: float) -> pd.DataFrame:
    url = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2015-04-25&endtime=2015-12-31"
           f"&minmagnitude=4&latitude={center_lat:.3f}&longitude={center_lon:.3f}&maxradiuskm=150")
    feats = json.loads(_get(url, CACHE / "aftershocks_2015_M4_150km.json").read_text(encoding="utf-8"))["features"]
    return pd.DataFrame([{"lon": f["geometry"]["coordinates"][0], "lat": f["geometry"]["coordinates"][1],
                          "mag": f["properties"]["mag"], "time": f["properties"]["time"]} for f in feats])


def build_seismic_features() -> pd.DataFrame:
    df = pd.read_parquet(MASTER)[["id", "latitude", "longitude"]]
    pts = df[["latitude", "longitude"]].to_numpy()
    out = df[["id"]].copy()
    epis = {}
    for tag, eid in EVENTS.items():
        grids, epi = shakemap_grid(eid)
        epis[tag] = epi
        for name, f in grids.items():
            out[f"{name}_{tag}"] = f(pts)
        out[f"dist_epi_{tag}_km"] = haversine_km(df.latitude, df.longitude, *epi)
    out["pga_max"] = out[["pga_main", "pga_may12"]].max(axis=1)
    aft = aftershocks(df.latitude.mean(), df.longitude.mean())
    d = haversine_km(df.latitude.to_numpy()[:, None], df.longitude.to_numpy()[:, None],
                     aft.lat.to_numpy()[None, :], aft.lon.to_numpy()[None, :])
    out["aftershocks_20km"] = (d <= 20).sum(1)
    out["dist_nearest_aftershock_km"] = d.min(1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    logger.info(f"{len(aft)} aftershocks M>=4 within 150 km; wrote {OUT}")
    logger.info("\n" + out[SEISMIC_FEATURES].describe().T.round(2).to_string())
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_seismic_features()
