"""NSGA-II building blocks (Deb et al. 2002), all objectives minimised.

* vectorised (epsilon-)dominance matrix and fast non-dominated sorting
* crowding distance
* binary/k-way tournament on (rank, crowding distance)
* exact hypervolume by recursive slicing (fine for the <= 40-point,
  <= 4-objective fronts used here)
"""

from __future__ import annotations

import numpy as np


def dominance_matrix(F: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """``D[i, j]`` is True iff solution i dominates solution j.

    epsilon-dominance (additive): i dominates j iff i is no worse in every
    objective and better by more than ``eps`` in at least one. ``eps = 0``
    recovers standard Pareto dominance.
    """
    F = np.asarray(F, dtype=float)
    if not np.isfinite(F).all():          # a failed evaluation counts as the worst value
        F = np.where(np.isfinite(F), F, np.inf)
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :] - eps, axis=2)
    D = le & lt
    np.fill_diagonal(D, False)
    return D


def nondominated_sort(F: np.ndarray, eps: float = 0.0) -> list[np.ndarray]:
    """Return fronts as index arrays, best first."""
    D = dominance_matrix(F, eps)
    remaining = np.ones(len(F), dtype=bool)
    fronts = []
    while remaining.any():
        n_dom = D[remaining][:, remaining].sum(axis=0)
        idx = np.flatnonzero(remaining)
        front = idx[n_dom == 0]
        if len(front) == 0:  # cannot happen for a strict partial order; safety net
            front = idx
        fronts.append(front)
        remaining[front] = False
    return fronts


def ranks_from_fronts(fronts: list[np.ndarray], n: int) -> np.ndarray:
    rank = np.empty(n, dtype=int)
    for r, f in enumerate(fronts):
        rank[f] = r
    return rank


def crowding_distance(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    n, m = F.shape
    if n <= 2:
        return np.full(n, np.inf)
    cd = np.zeros(n)
    for k in range(m):
        order = np.argsort(F[:, k], kind="stable")
        lo, hi = F[order[0], k], F[order[-1], k]
        cd[order[0]] = cd[order[-1]] = np.inf
        if hi - lo <= 0:
            continue
        cd[order[1:-1]] += (F[order[2:], k] - F[order[:-2], k]) / (hi - lo)
    return cd


def assign_rank_and_crowding(F: np.ndarray, eps: float = 0.0):
    fronts = nondominated_sort(F, eps)
    rank = ranks_from_fronts(fronts, len(F))
    crowd = np.zeros(len(F))
    for f in fronts:
        crowd[f] = crowding_distance(F[f])
    return fronts, rank, crowd


def tournament(rank: np.ndarray, crowd: np.ndarray, k: int, rng: np.random.Generator) -> int:
    """k-way tournament: lowest rank wins, ties broken by larger crowding."""
    cand = rng.choice(len(rank), size=min(k, len(rank)), replace=False)
    best = min(cand, key=lambda i: (rank[i], -crowd[i]))
    return int(best)


def environmental_selection(F: np.ndarray, size: int, eps: float = 0.0) -> np.ndarray:
    """Elitist (mu + lambda) truncation by front, then by crowding distance."""
    fronts = nondominated_sort(F, eps)
    chosen: list[int] = []
    for f in fronts:
        if len(chosen) + len(f) <= size:
            chosen.extend(f.tolist())
        else:
            cd = crowding_distance(F[f])
            order = np.argsort(-cd, kind="stable")
            chosen.extend(f[order[: size - len(chosen)]].tolist())
            break
    return np.array(chosen, dtype=int)


# --------------------------------------------------------------------------- #
# Hypervolume
# --------------------------------------------------------------------------- #

def _nondominated_points(P: np.ndarray) -> np.ndarray:
    if len(P) <= 1:
        return P
    D = dominance_matrix(P)
    keep = ~D.any(axis=0)
    P = P[keep]
    # drop exact duplicates
    return np.unique(P, axis=0)


def _hv_rec(P: np.ndarray, ref: np.ndarray) -> float:
    d = P.shape[1]
    if len(P) == 0:
        return 0.0
    if d == 1:
        return float(ref[0] - P[:, 0].min())
    if d == 2:
        P = P[np.argsort(P[:, 0], kind="stable")]
        hv, y_best = 0.0, ref[1]
        for i in range(len(P)):
            y_best = min(y_best, P[i, 1])
            nx = P[i + 1, 0] if i + 1 < len(P) else ref[0]
            hv += (nx - P[i, 0]) * (ref[1] - y_best)
        return float(hv)
    P = P[np.argsort(P[:, -1], kind="stable")]
    hv = 0.0
    for i in range(len(P)):
        z = P[i, -1]
        nz = P[i + 1, -1] if i + 1 < len(P) else ref[-1]
        if nz > z:
            hv += (nz - z) * _hv_rec(_nondominated_points(P[: i + 1, :-1]), ref[:-1])
    return float(hv)


def hypervolume(F: np.ndarray, ref: np.ndarray) -> float:
    """Exact hypervolume dominated by ``F`` w.r.t. reference point ``ref``."""
    F = np.asarray(F, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if F.size == 0:
        return 0.0
    P = F[np.all(F < ref, axis=1)]
    if len(P) == 0:
        return 0.0
    return _hv_rec(_nondominated_points(P), ref)
