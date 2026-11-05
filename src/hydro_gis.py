"""
DEM-based hydrological GIS for the Kavre springs (label-free).

1. Copernicus GLO-30 DEM tile N27E085 (public AWS bucket; cached in data/external/dem/), clipped
   to the district with a 0.05 degree margin.
2. pysheds: pit filling, depression filling, flat resolution, D8 flow direction, flow accumulation.
3. Stream networks at contributing-area thresholds T = 0.5, 2 and 10 km^2. Each network is split
   into links (segments between sources/confluences); Strahler order is computed per link.
4. For every spring: follow the D8 path downslope until it enters the network at scale T. The
   receiving link defines the spring's sub-catchment ("link catchment") at that scale. We record
   the link id, the flow-path length to the stream, the Strahler order of the receiving link and
   the contributing area at the spring's own cell.

Output: data/processed/hydro_gis_features.csv and results/figures/hydro_gis_network.png.
Usage:  python -m src.hydro_gis
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import os
import urllib.request

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

logger = logging.getLogger("hydro_gis")
DEM_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N27_00_E085_00_DEM/"
           "Copernicus_DSM_COG_10_N27_00_E085_00_DEM.tif")
DEM_DIR = PROJECT_ROOT / "data/external/dem"
OUT = PROJECT_ROOT / "data/processed/hydro_gis_features.csv"
MASTER = PROJECT_ROOT / "data/processed/master_dataset.parquet"
THRESHOLDS_KM2 = [0.5, 2.0, 10.0]
# pysheds default D8 encoding: N, NE, E, SE, S, SW, W, NW
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)
OFFSETS = {64: (-1, 0), 128: (-1, 1), 1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1)}


def get_dem(lat_min, lat_max, lon_min, lon_max, margin=0.05) -> Path:
    DEM_DIR.mkdir(parents=True, exist_ok=True)
    full = DEM_DIR / "Copernicus_DSM_COG_10_N27_00_E085_00_DEM.tif"
    if not full.exists():
        logger.info(f"Downloading {DEM_URL}")
        urllib.request.urlretrieve(DEM_URL, full)
    clip = DEM_DIR / "kavre_dem_clip.tif"
    if not clip.exists():
        with rasterio.open(full) as src:
            win = from_bounds(lon_min - margin, lat_min - margin, lon_max + margin, lat_max + margin, src.transform)
            win = win.round_offsets().round_lengths()
            arr = src.read(1, window=win)
            prof = src.profile.copy()
            prof.update(height=arr.shape[0], width=arr.shape[1], transform=src.window_transform(win),
                        driver="GTiff", tiled=False, compress="deflate")
            prof.pop("blockxsize", None); prof.pop("blockysize", None)
            with rasterio.open(clip, "w", **prof) as dst:
                dst.write(arr, 1)
    return clip


def condition_and_route(dem_path: Path):
    from pysheds.grid import Grid
    grid = Grid.from_raster(str(dem_path))
    dem = grid.read_raster(str(dem_path))
    dem = grid.fill_pits(dem)
    dem = grid.fill_depressions(dem)
    dem = grid.resolve_flats(dem)
    fdir = grid.flowdir(dem, dirmap=DIRMAP)
    acc = grid.accumulation(fdir, dirmap=DIRMAP)
    return grid, np.asarray(dem, dtype=float), np.asarray(fdir), np.asarray(acc, dtype=float)


def cell_area_km2(transform, lat):
    dx_deg, dy_deg = transform.a, -transform.e
    return (dx_deg * 111.32 * np.cos(np.radians(lat))) * (dy_deg * 111.32)


def downstream(r, c, fdir):
    d = fdir[r, c]
    if d not in OFFSETS:
        return None
    dr, dc = OFFSETS[d]
    r2, c2 = r + dr, c + dc
    if 0 <= r2 < fdir.shape[0] and 0 <= c2 < fdir.shape[1]:
        return r2, c2
    return None


def stream_links(fdir, stream):
    """Label stream links and compute Strahler order. Returns link-id raster (-1 off-network) and order dict."""
    rows, cols = np.where(stream)
    H, W = fdir.shape
    down = {}
    n_up = np.zeros(fdir.shape, np.int16)
    for r, c in zip(rows, cols):
        nxt = downstream(r, c, fdir)
        if nxt is not None and stream[nxt]:
            down[(r, c)] = nxt
            n_up[nxt] += 1
    link = -np.ones(fdir.shape, np.int64)
    starts = [(r, c) for r, c in zip(rows, cols) if n_up[r, c] != 1]  # sources (0) and confluences (>=2)
    link_down = {}
    for lid, s in enumerate(starts):
        cur = s
        while True:
            link[cur] = lid
            nxt = down.get(cur)
            if nxt is None:
                link_down[lid] = None
                break
            if n_up[nxt] != 1:  # next cell starts another link
                link_down[lid] = nxt
                break
            cur = nxt
    # resolve downstream link ids and Strahler order
    ds_link = {lid: (int(link[c]) if c is not None else None) for lid, c in link_down.items()}
    ups = {}
    for lid, d in ds_link.items():
        if d is not None:
            ups.setdefault(d, []).append(lid)
    order = {}

    def strahler(lid):
        if lid in order:
            return order[lid]
        u = [strahler(x) for x in ups.get(lid, [])]
        if not u:
            o = 1
        else:
            m = max(u)
            o = m + 1 if u.count(m) >= 2 else m
        order[lid] = o
        return o

    sys.setrecursionlimit(1_000_000)
    for lid in ds_link:
        strahler(lid)
    return link, order, ds_link


def trace_to_stream(r, c, fdir, link, cell_m, max_steps=20000):
    length = 0.0
    for _ in range(max_steps):
        if link[r, c] >= 0:
            return int(link[r, c]), length
        nxt = downstream(r, c, fdir)
        if nxt is None:
            return -1, length
        diag = abs(nxt[0] - r) + abs(nxt[1] - c) == 2
        length += cell_m * (np.sqrt(2) if diag else 1.0)
        r, c = nxt
    return -1, length


def build_hydro_gis_features():
    df = pd.read_parquet(MASTER)[["id", "latitude", "longitude"]]
    dem_path = get_dem(df.latitude.min(), df.latitude.max(), df.longitude.min(), df.longitude.max())
    logger.info("Conditioning DEM and routing flow (pysheds) ...")
    grid, dem, fdir, acc = condition_and_route(dem_path)
    with rasterio.open(dem_path) as src:
        transform = src.transform
    a_km2 = cell_area_km2(transform, df.latitude.mean())
    cell_m = np.sqrt(a_km2) * 1000
    rows, cols = rasterio.transform.rowcol(transform, df.longitude.to_numpy(), df.latitude.to_numpy())
    rows, cols = np.asarray(rows), np.asarray(cols)
    out = df[["id"]].copy()
    out["dem_elev_m"] = dem[rows, cols]
    out["contrib_area_km2"] = (acc[rows, cols] + 1) * a_km2
    nets = {}
    for T in THRESHOLDS_KM2:
        stream = (acc + 1) * a_km2 >= T
        link, order, ds = stream_links(fdir, stream)
        nets[T] = (stream, link, order)
        lids, lens = zip(*[trace_to_stream(r, c, fdir, link, cell_m) for r, c in zip(rows, cols)])
        tag = str(T).replace(".", "p")
        out[f"catch_T{tag}"] = lids
        out[f"flowlen_to_stream_T{tag}_m"] = lens
        out[f"strahler_T{tag}"] = [order.get(l, np.nan) if l >= 0 else np.nan for l in lids]
        logger.info(f"T={T} km2: {stream.sum()} stream cells, {len(order)} links, "
                    f"{pd.Series(lids).nunique()} catchments occupied by springs, unresolved={int((np.array(lids) < 0).sum())}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    fig, ax = plt.subplots(figsize=(8, 7.5))
    ext = [transform.c, transform.c + transform.a * dem.shape[1], transform.f + transform.e * dem.shape[0], transform.f]
    ax.imshow(np.where(dem > 0, dem, np.nan), extent=ext, cmap="terrain", alpha=0.75)
    s2 = nets[2.0][0]
    rr, cc = np.where(s2)
    xs, ys = rasterio.transform.xy(transform, rr, cc)
    ax.scatter(xs, ys, s=0.15, c="#1f4e9c", linewidths=0, label="streams (T = 2 km²)")
    ax.scatter(df.longitude, df.latitude, s=1.5, c="k", linewidths=0, label="water sources")
    ax.set_xlim(df.longitude.min() - 0.02, df.longitude.max() + 0.02)
    ax.set_ylim(df.latitude.min() - 0.02, df.latitude.max() + 0.02)
    ax.set_aspect(1 / np.cos(np.radians(27.57)))
    ax.set_title("Copernicus 30 m DEM, D8 stream network (2 km²) and mapped sources", fontsize=10)
    ax.legend(fontsize=8, loc="lower left", markerscale=6)
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "results/figures/hydro_gis_network.png", dpi=250)
    plt.close(fig)
    logger.info("\n" + out.describe().T.round(2).to_string())
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_hydro_gis_features()
