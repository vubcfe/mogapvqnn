import pytest

pytest.importorskip("pennylane")

from mogapvqnn.validation import validate


@pytest.mark.parametrize("n", [2, 4])
def test_simulator_agrees_with_pennylane(n):
    r = validate(n_qubits=n, n_circuits=3, noise=0.05, shots=3000, seed=n)
    assert r["statevector_max_err"] < 1e-10
    assert r["noisy_expval_max_err"] < 1e-10
    assert r["shadow_vs_pl_max_z"] < 5
    assert r["pl_shadow_max_z"] < 5
