import itertools

import numpy as np

from mogapvqnn.nsga2 import (
    crowding_distance,
    dominance_matrix,
    environmental_selection,
    hypervolume,
    nondominated_sort,
)


def brute_dominates(a, b, eps=0.0):
    return np.all(a <= b) and np.any(a < b - eps)


def test_dominance_matrix_matches_brute_force():
    rng = np.random.default_rng(0)
    F = rng.integers(0, 4, size=(25, 3)).astype(float)
    for eps in (0.0, 0.5):
        D = dominance_matrix(F, eps)
        for i, j in itertools.product(range(len(F)), repeat=2):
            assert D[i, j] == (i != j and brute_dominates(F[i], F[j], eps))


def test_fronts_are_valid_partition():
    rng = np.random.default_rng(1)
    F = rng.random((40, 4))
    fronts = nondominated_sort(F)
    allidx = np.sort(np.concatenate(fronts))
    assert np.array_equal(allidx, np.arange(40))
    D = dominance_matrix(F)
    for r, f in enumerate(fronts):
        assert not D[np.ix_(f, f)].any()                    # mutually non-dominated
        for later in fronts[r + 1:]:
            for j in later:                                  # every later point dominated by something earlier
                earlier = np.concatenate(fronts[: r + 1])
                assert D[earlier, j].any()


def test_crowding_boundaries_infinite():
    F = np.array([[0, 3], [1, 2], [2, 1], [3, 0]], float)
    cd = crowding_distance(F)
    assert np.isinf(cd[0]) and np.isinf(cd[3])
    # each objective contributes (neighbour gap) / (range) = 2/3
    assert np.allclose(cd[1:3], [4 / 3, 4 / 3])


def test_environmental_selection_keeps_first_front():
    F = np.array([[0, 1], [1, 0], [2, 2], [3, 3], [0.5, 0.5]])
    sel = environmental_selection(F, 3)
    assert set(sel) == {0, 1, 4}


def test_hypervolume_known_values():
    ref = np.array([1.0, 1.0])
    assert np.isclose(hypervolume(np.array([[0.5, 0.5]]), ref), 0.25)
    assert np.isclose(hypervolume(np.array([[0.0, 0.5], [0.5, 0.0]]), ref), 0.75)
    # dominated and out-of-box points do not change the volume
    assert np.isclose(hypervolume(np.array([[0.0, 0.5], [0.5, 0.0], [0.6, 0.6], [2, 0]]), ref), 0.75)
    assert np.isclose(hypervolume(np.array([[0.5, 0.5, 0.5]]), np.ones(3)), 0.125)


def test_hypervolume_matches_monte_carlo_4d():
    rng = np.random.default_rng(2)
    F = rng.random((12, 4))
    ref = np.full(4, 1.1)
    exact = hypervolume(F, ref)
    S = rng.random((400000, 4)) * 1.1
    dominated = np.zeros(len(S), bool)
    for p in F:
        dominated |= np.all(S >= p, axis=1)
    mc = dominated.mean() * 1.1**4
    assert abs(exact - mc) < 0.01
