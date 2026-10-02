import numpy as np
import pytest

from mogapvqnn.circuits import Gene, random_chromosome
from mogapvqnn.shadows import exact_features, make_observables, shadow_features
from mogapvqnn.simulator import (
    run_density,
    run_statevector,
    states_to_density,
    states_to_tensor,
    tensor_to_states,
)


def rand_states(n, k, rng, real=False):
    s = rng.normal(size=(k, 2**n))
    if not real:
        s = s + 1j * rng.normal(size=(k, 2**n))
    return s / np.linalg.norm(s, axis=1, keepdims=True)


def test_cnot_convention_wire0_is_msb():
    # |10> --CNOT(0->1)--> |11>  (index 2 -> index 3)
    psi = np.zeros((1, 4), complex)
    psi[0, 2] = 1
    out = tensor_to_states(run_statevector((Gene("CNOT", (0, 1)),), states_to_tensor(psi, 2)))
    assert np.allclose(out[0], [0, 0, 0, 1])


def test_unitarity_preserves_norm():
    rng = np.random.default_rng(1)
    c = random_chromosome(5, 20, rng)
    psi = rand_states(5, 7, rng)
    out = tensor_to_states(run_statevector(c, states_to_tensor(psi, 5)))
    assert np.allclose(np.linalg.norm(out, axis=1), 1)


def test_density_matches_statevector_without_noise():
    rng = np.random.default_rng(2)
    n = 3
    c = random_chromosome(n, 12, rng)
    psi = rand_states(n, 4, rng)
    out = tensor_to_states(run_statevector(c, states_to_tensor(psi, n)))
    rho = run_density(c, states_to_density(psi, n), n).reshape(4, 8, 8)
    assert np.allclose(rho, np.einsum("bi,bj->bij", out, out.conj()))


def test_full_depolarizing_on_single_qubit_gives_mixed_marginal():
    # DepolarizingChannel(p=3/4) is the completely depolarizing channel
    n = 1
    psi = np.array([[1, 0]], complex)
    rho = run_density((Gene("H", (0,)),), states_to_density(psi, n), n, noise=0.75).reshape(2, 2)
    assert np.allclose(rho, np.eye(2) / 2)


@pytest.mark.parametrize("model", ["local", "global"])
def test_noisy_density_is_valid_state(model):
    rng = np.random.default_rng(3)
    n = 3
    c = random_chromosome(n, 10, rng)
    rho = run_density(c, states_to_density(rand_states(n, 2, rng), n), n, 0.1, model).reshape(2, 8, 8)
    assert np.allclose(np.trace(rho, axis1=1, axis2=2), 1)
    assert np.allclose(rho, rho.conj().transpose(0, 2, 1))
    assert np.linalg.eigvalsh(rho).min() > -1e-12


def test_shadow_estimator_is_unbiased_clean():
    rng = np.random.default_rng(4)
    n = 3
    c = random_chromosome(n, 10, rng)
    psi = rand_states(n, 2, rng)
    obs = make_observables(n)
    ex = exact_features(c, psi, obs)
    T = 60000
    sh = shadow_features(c, psi, obs, T, rng)
    sd = np.array([3.0 ** (len(t) / 2) for t in obs.terms]) / np.sqrt(T)
    assert np.all(np.abs(sh - ex) < 5 * sd)


@pytest.mark.parametrize("model", ["local", "global"])
def test_noisy_trajectory_shadows_match_density_matrix(model):
    rng = np.random.default_rng(5)
    n = 3
    c = random_chromosome(n, 10, rng)
    psi = rand_states(n, 2, rng)
    obs = make_observables(n)
    ex = exact_features(c, psi, obs, 0.08, model)
    T = 60000
    sh = shadow_features(c, psi, obs, T, rng, 0.08, model)
    sd = np.array([3.0 ** (len(t) / 2) for t in obs.terms]) / np.sqrt(T)
    assert np.all(np.abs(sh - ex) < 5 * sd)
    # and noise actually changes the expectations
    assert np.abs(ex - exact_features(c, psi, obs)).max() > 0.01


def test_shadow_chunking_does_not_change_results():
    rng = np.random.default_rng(6)
    n = 3
    c = random_chromosome(n, 8, rng)
    psi = rand_states(n, 5, rng)
    obs = make_observables(n)
    a = shadow_features(c, psi, obs, 64, np.random.default_rng(9), max_elems=1 << 22)
    b = shadow_features(c, psi, obs, 64, np.random.default_rng(9), max_elems=1 << 22)
    assert np.array_equal(a, b)


def test_observable_counts():
    assert make_observables(4, "local1").size == 12
    assert make_observables(4, "local2").size == 12 + 6
