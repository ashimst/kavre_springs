"""
One-command reproduction of every number, table and figure in the paper and supplement.

  0. Data: raw ICIMOD inventory (data/raw/spring_data.csv) -> cleaned table -> master table, joined with the
     versioned covariate extractions in data/processed (Earth Engine, OpenStreetMap). Re-extracting those needs
     credentials (src/gee_extraction.py, src/extract_hydro_catchment_gee.py, src/extract_enrichment_gee.py,
     src/extract_osm_roads.py) and is not part of this script.
  1. Open GIS layers, downloaded once and cached in data/external: USGS ShakeMap/ComCat (src/seismic.py), the
     Copernicus GLO-30 DEM for flow routing and catchments (src/hydro_gis.py) and official palika boundaries
     (src/boundaries.py; geoBoundaries / MoFAGA, versioned in data/processed/boundaries), and OpenStreetMap ward
     polygons (src/ward_polygons.py; ODbL, versioned in data/processed/boundaries).
  2. Leakage audit of all feature sets (aborts if a non-diagnostic set leaks).
  3. Predictive benchmark with standard models (skipped with --skip-benchmark; results are versioned).
  4. Spatial statistics and descriptive analyses.
  5. Unsupervised mining: landscape typology, regionalisation, ward funnel; apriori pattern mining with
     ward-stratified tests.
  6. Spatial analysis, catchment vs ward analysis, within-ward drivers, robustness.
  7. Figures (src/figures.py), report (SI tables as markdown and LaTeX), paper build (if paper/ exists) and unit tests.
  8. Optional (--notebooks): build and execute the analysis notebooks.

Usage:
    python reproduce_all.py                   # everything, including the benchmark (~1 h on a laptop CPU)
    python reproduce_all.py --skip-benchmark  # all analyses from the versioned benchmark results (~1 h)
    python reproduce_all.py --quick           # 1-seed smoke test of the benchmark (writes *_quick files only)
    python reproduce_all.py --skip-benchmark --notebooks
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
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))  # silences joblib's core-count probe on Windows
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("reproduce")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--notebooks", action="store_true", help="build and execute the analysis notebooks")
    a = ap.parse_args()
    t0 = time.time()

    from src.assemble_master import run as assemble_master
    assemble_master()           # raw inventory -> master table (offline, from versioned covariate extractions)

    from src.features import FEATURE_SETS, check_feature_set
    from src.dataset import load_master

    # External, label-free GIS layers (cached downloads): USGS ShakeMap/ComCat and Copernicus DEM hydrology
    from src.seismic import build_seismic_features
    from src.hydro_gis import build_hydro_gis_features
    build_seismic_features()
    build_hydro_gis_features()
    from src.boundaries import run as run_boundaries
    run_boundaries()            # official palika/district boundaries (versioned) and source-location agreement
    from src.ward_polygons import run as run_ward_polygons
    run_ward_polygons()         # OpenStreetMap ward polygons (versioned) and recorded-ward agreement

    df = load_master()
    for fs in FEATURE_SETS:
        v = check_feature_set(df, fs)
        log.info(f"Leakage audit [{fs}]: {'CLEAN' if not v else v}")
        if v and fs != "leaky_reference":
            raise SystemExit("Primary/sensitivity feature set leaks -- aborting.")

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

    run_spatial_mining()
    run_descriptive()
    run_unsupervised()          # landscape typology, regionalisation, ward funnel
    run_pattern_mining()        # apriori rules with FDR, productivity, ward-stratified MH odds ratios, replication
    run_spatial_analysis()      # seismic dose-response, directional test, Knox, ward concordance
    run_catchment_analysis()    # DEM catchments vs wards vs grids (ICC), pair concordance, flow-path covariates
    run_drivers()               # pooled vs within-ward drivers; three-way variation partitioning
    run_robustness()            # contiguous random-region null, sensitivity subsets, validation distances, earthquake FE
    from src.revision_experiments import run as run_revision
    run_revision()              # tier tests, within-ward AUC, placebo units, residual ward ICC, reversal FDR, ...
    from src.figures import run as run_figures
    run_figures()               # the eight main figures, drawn from the result tables
    run_report()
    import subprocess
    if (PROJECT_ROOT / "paper/manuscript.tex").exists():   # the manuscript is kept locally, outside version control
        from src.build_paper import main as build_paper
        build_paper()           # refresh paper/figures and paper/manuscript.bbl
    if a.notebooks:
        subprocess.run([sys.executable, str(PROJECT_ROOT / "notebooks/build_notebooks.py")], check=True)

    import pytest
    rc = pytest.main(["-q", "tests"])
    log.info(f"Finished in {(time.time() - t0) / 60:.1f} min; tests exit code {rc}")
    sys.exit(rc)

if __name__ == "__main__":
    main()
