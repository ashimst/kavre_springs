"""
One-command reproduction of every number, table and figure in the paper/SI.

Steps (starting from data/processed/master_dataset.parquet; re-extracting the
Earth Engine covariates requires GEE credentials -- see src/gee_extraction.py):

  1. Leakage audit of the feature sets (fails loudly if a feature leaks).
  2. Build and save the primary graph.
  3. Benchmark profiles: main (spatial CV, 0/500 m buffers), features
     (feature-set sensitivity) and lomo (leave-one-municipality-out).
  4. Spatial statistics (Moran's I decay, LISA with FDR, association rules).
  5. Explainability (TreeSHAP, logistic coefficients, attention vs placebo).
  6. Spring Priority Index for active springs.
  7. Report: summary tables, paired tests, figures, paper/si_generated_tables.md.
  8. Unit tests.

Usage:
    python reproduce_all.py               # full run (~45 min on a single GPU)
    python reproduce_all.py --quick       # 1 seed x 1 fold smoke test (writes *_quick files only)
    python reproduce_all.py --skip-benchmark   # reuse existing benchmark CSVs
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
    a = ap.parse_args()
    t0 = time.time()

    from src.features import FEATURE_SETS, check_feature_set
    from src.graph_builder import GRAPH_DIR, K_NEIGHBOURS, build_graph, load_master, save_graph

    df = load_master()
    for fs in FEATURE_SETS:
        v = check_feature_set(df, fs)
        log.info(f"Leakage audit [{fs}]: {'CLEAN' if not v else v}")
        if v and fs != "leaky_reference":
            raise SystemExit("Primary/sensitivity feature set leaks -- aborting.")

    g = build_graph(df, "env")
    save_graph(g, GRAPH_DIR / f"graph_knn_k{K_NEIGHBOURS}_env.pt")

    if not a.skip_benchmark:
        from src.benchmark import PROFILES, run_profile
        for prof in PROFILES:
            run_profile(prof, seeds=[42] if a.quick else None, quick=a.quick)
    if a.quick:
        log.info(f"Quick smoke test finished in {time.time() - t0:.0f}s (outputs: *_quick files).")
        return

    from src.spatial_mining import run_spatial_mining
    from src.explainability import run_all as run_explainability
    from src.prioritization import run_prioritization
    from src.descriptive import run_descriptive
    from src.report import run_report

    run_spatial_mining()
    run_explainability()
    run_prioritization()
    run_descriptive()
    run_report()

    import pytest
    rc = pytest.main(["-q", "tests"])
    log.info(f"Finished in {(time.time() - t0) / 60:.1f} min; tests exit code {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
