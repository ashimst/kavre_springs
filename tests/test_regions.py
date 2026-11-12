"""Placebo partitions (src/regions.py): every region connected, sizes close to targets, reproducible seeds."""

import numpy as np
import pandas as pd

from src.regions import contiguity_graph, grow_partition, nested_partition, sample_partitions, verify


def _cloud(n=900, seed=0):
    rng = np.random.default_rng(seed)
    # two separated clusters, so the kNN graph has two components that must be joined
    a = rng.normal(0, 1000, (n // 2, 2))
    b = rng.normal(0, 1000, (n - n // 2, 2)) + [20000, 0]
    return np.vstack([a, b])


def test_graph_is_connected():
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    c = _cloud()
    nb = contiguity_graph(c)
    r = np.repeat(np.arange(len(nb)), [len(x) for x in nb])
    A = coo_matrix((np.ones(len(r)), (r, np.concatenate(nb))), shape=(len(c), len(c)))
    assert connected_components(A, directed=False)[0] == 1


def test_regions_connected_and_size_matched():
    c = _cloud()
    nb = contiguity_graph(c)
    sizes = np.array([30, 45, 60, 75, 90] * 3)
    assert sizes.sum() == len(c)
    for s in range(3):
        lab = grow_partition(nb, c, sizes, np.random.default_rng(s))
        v = verify(lab, nb, sizes)
        assert v["n_disconnected"] == 0
        assert v["n_regions"] == len(sizes)
        assert v["size_l1_share"] < 0.05


def test_nested_regions_stay_inside_groups():
    c = _cloud()
    groups = np.where(c[:, 0] > 10000, "east", "west")
    sbg = {"east": np.array([150, 150, 150]), "west": np.array([200, 250])}
    lab = nested_partition(groups, c, sbg, np.random.default_rng(1))
    for r in np.unique(lab):
        assert len(np.unique(groups[lab == r])) == 1


def test_sampling_reproducible_regardless_of_workers():
    c = _cloud(400)
    units = np.repeat(np.arange(8), 50)
    a = sample_partitions(c, units, 3, seed=5, n_jobs=1, cache=False)
    b = sample_partitions(c, units, 3, seed=5, n_jobs=2, cache=False)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert pd.Series(a[0]).nunique() == 8
