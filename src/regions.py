"""
Contiguous, size-matched random partitions of the source cloud (placebo units for the ward analyses).

Graph.  Contiguity is defined on the symmetric 10-nearest-neighbour graph of the sources. Where that graph has more
than one connected component, the components are joined by the shortest Euclidean edge between them (repeated
until one component remains), so every region can be grown and checked on one connected graph.

Growth.  Given target sizes (e.g. the 75 ward sizes), regions are seeded at random sources and take turns adding the
unassigned graph neighbour closest to their seed until they reach their target or run out of unassigned neighbours.
Sources left unassigned are then attached, frontier by frontier, to the ADJACENT region with the largest remaining
deficit. Attaching a vertex to a region it is adjacent to keeps that region connected.

Balancing.  Region sizes are then moved toward their targets by boundary transfers: a source u in region A that is
adjacent to region B moves to B when A's surplus (count - target) exceeds B's by at least 2 and removing u leaves A
connected (checked by breadth-first search). Surplus diffuses through chains of regions until no improving move is
left or every region matches its target.

Verification.  ``verify`` returns, for a partition, the number of regions that are not connected on the graph and the
size mismatch against the targets (sum |count - target| / N, and the largest single deviation).
"""

from __future__ import annotations

import heapq
import random
from collections import deque

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def contiguity_graph(coords: np.ndarray, k: int = 10) -> list[np.ndarray]:
    """Adjacency lists of the symmetric kNN graph, with components joined by their shortest connecting edges."""
    from scipy.sparse import coo_matrix
    n = len(coords)
    k = min(k, n - 1)
    _, idx = cKDTree(coords).query(coords, k + 1)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    edges = set(zip(rows.tolist(), cols.tolist()))
    edges |= {(b, a) for a, b in edges}
    while True:
        r, c = zip(*edges)
        A = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
        n_comp, lab = connected_components(A, directed=False)
        if n_comp == 1:
            break
        # join the smallest component to its nearest outside source
        comp = np.bincount(lab).argmin()
        inside = np.where(lab == comp)[0]
        outside = np.where(lab != comp)[0]
        d, j = cKDTree(coords[outside]).query(coords[inside])
        a = int(inside[np.argmin(d)])
        b = int(outside[j[np.argmin(d)]])
        edges |= {(a, b), (b, a)}
    A = A.tocsr()
    return [A.indices[A.indptr[i]:A.indptr[i + 1]] for i in range(n)]


def grow_partition(nbrs, coords, sizes, rng, balance=True, max_passes=400):
    n = len(nbrs)
    sizes = np.asarray(rng.permutation(sizes), np.int64)
    R = len(sizes)
    lab = -np.ones(n, np.int64)
    seeds = rng.choice(n, R, replace=False)
    count = np.zeros(R, np.int64)
    heaps = []
    for r, s in enumerate(seeds):
        lab[s] = r
        count[r] = 1
        h = [(float(np.hypot(*(coords[u] - coords[s]))), int(u)) for u in nbrs[s]]
        heapq.heapify(h)
        heaps.append(h)
    active = [r for r in range(R) if count[r] < sizes[r]]
    while active:
        nxt = []
        for r in active:
            h = heaps[r]
            while h and lab[h[0][1]] >= 0:
                heapq.heappop(h)
            if not h:
                continue
            _, v = heapq.heappop(h)
            lab[v] = r
            count[r] += 1
            s = seeds[r]
            for u in nbrs[v]:
                if lab[u] < 0:
                    heapq.heappush(h, (float(np.hypot(*(coords[u] - coords[s]))), int(u)))
            if count[r] < sizes[r]:
                nxt.append(r)
        active = nxt
    # leftovers: attach to the adjacent region with the largest deficit, frontier by frontier
    left = np.where(lab < 0)[0]
    while len(left):
        progress = False
        for v in rng.permutation(left):
            adj = [lab[u] for u in nbrs[v] if lab[u] >= 0]
            if not adj:
                continue
            adj = np.unique(adj)
            r = adj[np.argmax(sizes[adj] - count[adj])]
            lab[v] = r
            count[r] += 1
            progress = True
        left = np.where(lab < 0)[0]
        if not progress:  # cannot happen on a connected graph
            raise RuntimeError("unassigned sources not adjacent to any region")
    if balance:
        _balance(lab, count, sizes, nbrs, rng, max_passes)
    return lab


def _balance(lab, count, sizes, nbrs, rng, max_passes, phase2_passes=60):
    """Boundary transfers toward the target sizes (pure Python for speed). Phase 1 makes only strictly improving
    moves (surplus gap >= 2); phase 2 also allows neutral moves (gap == 1) so that a surplus can travel through
    regions that are already on target until it reaches a deficit."""
    n = len(lab)
    L = [int(x) for x in lab]
    nb = [list(map(int, a)) for a in nbrs]
    cnt = [int(x) for x in count]
    tgt = [int(x) for x in sizes]
    members = [set() for _ in tgt]
    for v, r in enumerate(L):
        members[r].add(v)

    def still_connected(u, A):
        size_after = cnt[A] - 1
        mem = members[A]
        start = next((v for v in nb[u] if v in mem), None)
        if start is None:
            return size_after == 0
        seen = {start}
        q = [start]
        while q:
            v = q.pop()
            for w in nb[v]:
                if w != u and w in mem and w not in seen:
                    seen.add(w)
                    q.append(w)
        return len(seen) == size_after

    pr = random.Random(int(rng.integers(1 << 31)))
    for phase, min_gap, passes in [(1, 2, max_passes), (2, 1, phase2_passes)]:
        for _ in range(passes):
            if all(c == s for c, s in zip(cnt, tgt)):
                break
            surplus = [v for v in range(n) if cnt[L[v]] > tgt[L[v]]]
            pr.shuffle(surplus)
            moved = False
            for u in surplus:
                A = L[u]
                dA = cnt[A] - tgt[A]
                if dA <= 0:
                    continue
                best, bd = -1, None
                for w in nb[u]:
                    B = L[w]
                    if B != A:
                        d = cnt[B] - tgt[B]
                        if bd is None or d < bd:
                            best, bd = B, d
                if best < 0 or dA - bd < min_gap:
                    continue
                if phase == 2 and bd >= 0 and pr.random() < 0.5:
                    continue  # damp neutral moves so surplus random-walks instead of oscillating
                if not still_connected(u, A):
                    continue
                members[A].discard(u)
                members[best].add(u)
                L[u] = best
                cnt[A] -= 1
                cnt[best] += 1
                moved = True
            if not moved:
                break
    lab[:] = L
    count[:] = cnt


def verify(lab, nbrs, sizes=None):
    """Regions not connected on the graph, and size mismatch against the (sorted) target sizes."""
    n_bad = 0
    for r in np.unique(lab):
        m = lab == r
        idx = np.where(m)[0]
        if not _is_connected(idx, m, nbrs):
            n_bad += 1
    out = {"n_regions": len(np.unique(lab)), "n_disconnected": n_bad}
    if sizes is not None:
        got = np.sort(np.bincount(lab))
        tgt = np.sort(np.asarray(sizes))
        out["size_l1_share"] = float(np.abs(got - tgt).sum() / len(lab))
        out["size_max_abs_dev"] = int(np.abs(got - tgt).max())
    return out


def _is_connected(idx, mask, nbrs):
    start = idx[0]
    seen = {start}
    q = deque([start])
    while q:
        v = q.popleft()
        for w in nbrs[v]:
            if mask[w] and w not in seen:
                seen.add(w)
                q.append(w)
    return len(seen) == len(idx)


def nested_partition(groups, coords, sizes_by_group, rng, graphs=None, balance=True):
    """Grow regions separately inside each group (e.g. municipality) with that group's target sizes."""
    lab = -np.ones(len(coords), np.int64)
    offset = 0
    for gname, sizes in sizes_by_group.items():
        idx = np.where(groups == gname)[0]
        nb = graphs[gname] if graphs is not None else contiguity_graph(coords[idx])
        sub = grow_partition(nb, coords[idx], sizes, rng, balance=balance)
        lab[idx] = sub + offset
        offset += len(sizes)
    return lab


# ----------------------------------------------------------------------------- parallel, reproducible sampling
def _one_draw(args):
    kind, coords, sizes, groups, sizes_by_group, child_seed, graph = args
    rng = np.random.default_rng(child_seed)
    if kind == "district":
        return grow_partition(graph, coords, sizes, rng)
    return nested_partition(groups, coords, sizes_by_group, rng, graphs=graph)


def sample_partitions(coords, units, n_rep, seed, kind="district", groups=None, n_jobs=12, cache=True):
    """n_rep contiguous random partitions with the size distribution of ``units`` (e.g. ward labels).

    kind="district": regions grown on the whole source graph with the unit size distribution.
    kind="nested":   regions grown inside each group (municipality) with that group's unit sizes.
    Each draw has its own child seed (SeedSequence(seed).spawn), so results do not depend on n_jobs.
    Partitions are cached in results/cache/ keyed by kind, seed, n and a hash of the inputs."""
    import hashlib
    from pathlib import Path

    import pandas as pd
    from joblib import Parallel, delayed
    units = np.asarray(units)
    h = hashlib.sha1(np.ascontiguousarray(coords).tobytes() + units.astype(str).tobytes()
                     + (b"" if groups is None else np.asarray(groups).astype(str).tobytes())).hexdigest()[:12]
    path = Path(__file__).resolve().parents[1] / "results/cache" / f"partitions_{kind}_{seed}_{n_rep}_{h}.npy"
    if cache and path.exists():
        return list(np.load(path))
    sizes = pd.Series(units).value_counts().to_numpy()
    sizes_by_group, graph = None, None
    if kind == "district":
        graph = contiguity_graph(coords)
    else:
        groups = np.asarray(groups)
        sizes_by_group = {g: pd.Series(units[groups == g]).value_counts().to_numpy() for g in np.unique(groups)}
        graph = {g: contiguity_graph(coords[groups == g]) for g in sizes_by_group}
    children = np.random.SeedSequence(seed).spawn(n_rep)
    parts = Parallel(n_jobs=n_jobs)(delayed(_one_draw)((kind, coords, sizes, groups, sizes_by_group, c, graph))
                                    for c in children)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.stack(parts))
    return parts


def partition_diagnostics(parts, coords, units, kind="district", groups=None):
    """Connectivity and size match of every partition (on the graph it was grown on)."""
    import pandas as pd
    units = np.asarray(units)
    rows = []
    if kind == "district":
        g = contiguity_graph(coords)
        sizes = pd.Series(units).value_counts().to_numpy()
        for p in parts:
            rows.append(verify(np.asarray(p), g, sizes))
    else:
        groups = np.asarray(groups)
        gs = {m: contiguity_graph(coords[groups == m]) for m in np.unique(groups)}
        for p in parts:
            tot = {"n_regions": 0, "n_disconnected": 0, "size_abs_dev_sum": 0, "size_max_abs_dev": 0}
            for m, gr in gs.items():
                idx = groups == m
                v = verify(pd.factorize(np.asarray(p)[idx])[0], gr, pd.Series(units[idx]).value_counts().to_numpy())
                tot["n_regions"] += v["n_regions"]
                tot["n_disconnected"] += v["n_disconnected"]
                tot["size_abs_dev_sum"] += v["size_l1_share"] * idx.sum()
                tot["size_max_abs_dev"] = max(tot["size_max_abs_dev"], v["size_max_abs_dev"])
            tot["size_l1_share"] = tot.pop("size_abs_dev_sum") / len(units)
            rows.append(tot)
    return pd.DataFrame(rows)
