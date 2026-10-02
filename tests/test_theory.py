"""Numerical check of Proposition 1 (sections/expressivity.tex): every
post-variational feature with amplitude encoding is a quadratic form
x^T S x with S = Re Phi^dagger(P), also under gate noise."""

import numpy as np
import pytest

from mogapvqnn.circuits import random_chromosome
from mogapvqnn.shadows import exact_features, make_observables
from mogapvqnn.simulator import PAULIS, density_to_matrix, run_density


def heisenberg_matrix(chrom, n, term, noise, model):
    """A = Phi^dagger(P) via A_ji = tr(P Phi(|i><j|)) (Phi is linear, so it
    can be applied to the non-Hermitian matrix units)."""
    D = 2**n
    P = np.array([[1.0]])
    ops = {w: p for w, p in term}
    for w in range(n):
        P = np.kron(P, PAULIS[ops.get(w, 0)])
    units = np.zeros((D * D, D, D), complex)
    for i in range(D):
        for j in range(D):
            units[i * D + j, i, j] = 1.0
    out = density_to_matrix(run_density(chrom, units.reshape((D * D,) + (2,) * (2 * n)), n, noise, model), n)
    A = np.empty((D, D), complex)
    for i in range(D):
        for j in range(D):
            A[j, i] = np.trace(P @ out[i * D + j])
    return A


@pytest.mark.parametrize("noise,model", [(0.0, "local"), (0.07, "local"), (0.1, "global")])
def test_features_are_quadratic_forms(noise, model):
    rng = np.random.default_rng(0)
    n = 3
    chrom = random_chromosome(n, 10, rng)
    obs = make_observables(n)
    X = np.abs(rng.normal(size=(6, 2**n)))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    F = exact_features(chrom, X, obs, noise, model)
    for k, term in enumerate(obs.terms):
        A = heisenberg_matrix(chrom, n, term, noise, model)
        assert np.allclose(A, A.conj().T)                     # Hermitian
        S = A.real
        assert np.allclose(S, S.T)                            # real symmetric
        assert np.linalg.norm(S, 2) <= 1 + 1e-9               # contraction
        assert np.allclose(np.einsum("bi,ij,bj->b", X, S, X), F[:, k])
