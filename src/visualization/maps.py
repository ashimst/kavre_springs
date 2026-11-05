"""
Geospatial visualization module for Kavre Springs research.

Produces:
1. Static publication maps with OSM tiles and official administrative boundaries.
2. Interactive Folium web maps with layer controls, clusters, and tooltips.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
import folium
from folium.plugins import MarkerCluster
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DISTRICT_GEOJSON = Path("data/processed/kavre_district_boundary.geojson")
MUNIS_GEOJSON = Path("data/processed/kavre_municipalities.geojson")
SPRINGS_GEOJSON = Path("data/processed/springs.geojson")


def generate_static_publication_map(
    output_path: Path = Path("results/figures/kavre_springs_official_map.png"),
    dpi: int = 300,
):
    """Generate high-DPI publication map with OpenStreetMap Mapnik basemap tiles."""
    district_3857 = gpd.read_file(DISTRICT_GEOJSON).to_crs(epsg=3857)
    munis_3857 = gpd.read_file(MUNIS_GEOJSON).to_crs(epsg=3857)
    springs_3857 = gpd.read_file(SPRINGS_GEOJSON).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(13, 10))

    # District boundary
    district_3857.boundary.plot(
        ax=ax, color="#111111", linewidth=2.5, linestyle="--", label="Kavre District Boundary (OSM)"
    )

    # Municipality boundaries
    munis_3857.plot(ax=ax, facecolor="#2b5c8f", alpha=0.06, edgecolor="#222222", linewidth=1.4)

    # Municipality labels
    for _, row in munis_3857.iterrows():
        c = row.geometry.centroid
        ax.annotate(
            row["query_name"],
            (c.x, c.y),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="gray"),
        )

    # Springs
    active = springs_3857[springs_3857["target_dried"] == 0]
    dried = springs_3857[springs_3857["target_dried"] == 1]

    active.plot(ax=ax, color="#1f77b4", markersize=12, alpha=0.55, label=f"Active Source (n={len(active):,})")
    dried.plot(ax=ax, color="#d62728", marker="x", markersize=18, alpha=0.85, label=f"Dried Source (n={len(dried):,})")

    # Add OSM tiles
    try:
        cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, attribution=False)
        logger.info("Baked OpenStreetMap Mapnik tiles into map.")
    except Exception as e:
        logger.warning(f"Tile error: {e}")

    plt.title(
        "Geospatial Distribution of Water Springs in Kavrepalanchok, Nepal (N=5,689)\n"
        "Official OpenStreetMap Administrative Boundaries & ICIMOD Hydro-Census",
        fontsize=12,
        fontweight="bold",
    )
    plt.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95)
    ax.set_axis_off()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info(f"Saved publication map to {output_path}")


def generate_interactive_web_map(
    output_html: Path = Path("results/maps/kavre_springs_interactive.html"),
):
    """Generate interactive Folium web map with layers, clusters, and tooltips."""
    district_gdf = gpd.read_file(DISTRICT_GEOJSON)
    munis_gdf = gpd.read_file(MUNIS_GEOJSON)
    springs_gdf = gpd.read_file(SPRINGS_GEOJSON)

    m = folium.Map(location=[27.57, 85.61], zoom_start=11, tiles="OpenStreetMap")

    # Municipality boundaries
    folium.GeoJson(
        munis_gdf,
        name="Municipality Boundaries (OSM)",
        style_function=lambda x: {"fillColor": "#3186cc", "color": "#222222", "weight": 2, "fillOpacity": 0.1},
        tooltip=folium.GeoJsonTooltip(fields=["query_name"], aliases=["Municipality:"]),
    ).add_to(m)

    # District boundary
    folium.GeoJson(
        district_gdf,
        name="Kavre District Border (OSM)",
        style_function=lambda x: {"color": "#e41a1c", "weight": 3, "dashArray": "6, 6", "fillOpacity": 0.0},
    ).add_to(m)

    # Dried springs
    fg_dried = folium.FeatureGroup(name="Dried Springs (Red)", show=True)
    for _, row in springs_gdf[springs_gdf["target_dried"] == 1].iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color="#d62728",
            fill=True,
            fill_color="#d62728",
            fill_opacity=0.85,
            popup=f"ID: {row['id']} | Muni: {row['municipality']} | Type: {row['source_type']} | DRIED",
        ).add_to(fg_dried)
    fg_dried.add_to(m)

    # Active springs
    fg_active = folium.FeatureGroup(name="Active Springs (Blue Cluster)", show=True)
    cluster = MarkerCluster().add_to(fg_active)
    for _, row in springs_gdf[springs_gdf["target_dried"] == 0].iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2,
            color="#1f77b4",
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=0.6,
            popup=f"ID: {row['id']} | Muni: {row['municipality']} | ACTIVE",
        ).add_to(cluster)
    fg_active.add_to(m)

    folium.LayerControl().add_to(m)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(output_html)
    logger.info(f"Saved interactive HTML map to {output_html}")


if __name__ == "__main__":
    generate_static_publication_map()
    generate_interactive_web_map()

