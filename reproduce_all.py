"""
One-command reproduction of every number, table and figure in the paper and SI.

Starts from data/processed/master_dataset.parquet and the cached Earth Engine covariates in data/processed
(re-extracting them needs GEE credentials: src/gee_extraction.py, src/extract_hydro_catchment_gee.py,
src/extract_enrichment_gee.py). Open GIS layers are downloaded once and cached:
Copernicus GLO-30 DEM (src/hydro_gis.py) and USGS ShakeMap/ComCat (src/seismic.py).

  1. GIS layers: seismic exposure, DEM flow routing and catchments.
  2. Leakage audit of all feature sets (aborts if a non-diagnostic set leaks).
  3. Predictive benchmark: spatial block CV (0/500 m buffers), feature-set sensitivity, LOMO.
  4. Spatial statistics: Moran's I decay, LISA (spatial_mining); descriptive timing/causes and study-area map.
  5. Unsupervised mining: landscape typology, HDBSCAN, t-SNE, anomalies, label-free regionalisation, ward funnel,
     ward narrative regimes; apriori pattern mining with ward-stratified tests.
  6. Spatial analysis: seismic dose-response, directional test, Knox, ward concordance, MEM partitioning, scan.
  7. Catchment vs ward analysis; within-ward drivers; GNN topology experiment.
  8. Report (SI tables as markdown and LaTeX), arXiv package (paper/arxiv_submission.zip) and unit tests.
  9. Optional (--notebooks): build and execute the analysis notebooks 03b-12.

Usage:
    python reproduce_all.py                   # full run (several hours on one GPU; benchmarks are checkpointed)
    python reproduce_all.py --quick           # 1 seed smoke test of the benchmark (writes *_quick files only)
    python reproduce_all.py --skip-benchmark  # reuse existing benchmark CSVs
    python reproduce_all.py --skip-benchmark --notebooks   # also rebuild and execute the notebooks
"""

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("reproduce")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--notebooks", action="store_true", help="build and execute notebooks/03b-12")
    a = ap.parse_args()
    t0 = time.time()

    from src.features import FEATURE_SETS, check_feature_set
    from src.graph_builder import GRAPH_DIR, K_NEIGHBOURS, build_graph, load_master, save_graph

    # External, label-free GIS layers (cached downloads): USGS ShakeMap/ComCat and Copernicus DEM hydrology
    from src.seismic import build_seismic_features
    from src.hydro_gis import build_hydro_gis_features
    build_seismic_features()
    build_hydro_gis_features()

    df = load_master()
    for fs in FEATURE_SETS:
        v = check_feature_set(df, fs)
        log.info(f"Leakage audit [{fs}]: {'CLEAN' if not v else v}")
        if v and fs != "leaky_reference":
            raise SystemExit("Primary/sensitivity feature set leaks -- aborting.")

    g = build_graph(df, "env_hydro")
    save_graph(g, GRAPH_DIR / f"graph_knn_k{K_NEIGHBOURS}_env_hydro.pt")

    if not a.skip_benchmark:
        from src.benchmark import PROFILES, run_profile
        for prof in PROFILES:
            run_profile(prof, seeds=[42] if a.quick else None, quick=a.quick)
    if a.quick:
        log.info(f"Quick smoke test finished in {time.time() - t0:.0f}s (outputs: *_quick files).")
        return

    from src.spatial_mining import run_spatial_mining
    from src.descriptive import run_descriptive
    from src.unsupervised import run as run_unsupervised
    from src.pattern_mining import run as run_pattern_mining
    from src.robustness import run as run_robustness
    from src.report import run_report

    from src.spatial_analysis import run_all as run_spatial_analysis
    from src.catchment_analysis import run_all as run_catchment_analysis
    from src.drivers_analysis import run as run_drivers
    from src.graph_topology_experiment import run as run_graph_topology

    run_spatial_mining()
    run_descriptive()
    run_unsupervised()          # landscape typology, regionalisation, ward funnel and narrative regimes
    run_pattern_mining()        # apriori rules with FDR, productivity, ward-stratified MH odds ratios, replication
    run_spatial_analysis()      # seismic dose-response, directional graph test, Knox, ward concordance, MEMs, scan
    run_catchment_analysis()    # DEM catchments vs wards vs grids (ICC), pair concordance, flow-path covariates
    run_drivers()               # pooled vs within-ward drivers; three-way variation partitioning
    run_graph_topology()        # GNNs on kNN / catchment / ridge / ward / random graphs
    run_robustness()            # contiguous random-region null, sensitivity subsets, validation distances, earthquake FE
    run_report()
    import subprocess
    subprocess.run([sys.executable, str(PROJECT_ROOT / "paper/build_arxiv.py")], check=True)
    if a.notebooks:
        subprocess.run([sys.executable, str(PROJECT_ROOT / "notebooks/build_notebooks.py")], check=True)

    import pytest
    rc = pytest.main(["-q", "tests"])
    log.info(f"Finished in {(time.time() - t0) / 60:.1f} min; tests exit code {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
