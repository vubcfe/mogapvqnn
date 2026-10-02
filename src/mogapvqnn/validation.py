"""Cross-validation of the NumPy simulator against PennyLane 0.38."""

from __future__ import annotations

import numpy as np

from .circuits import random_chromosome, to_pennylane_ops
from .shadows import exact_features, make_observables, shadow_features
from .simulator import run_statevector, states_to_tensor


def _random_states(n: int, k: int, rng) -> np.ndarray:
    s = np.abs(rng.normal(size=(k, 2**n)))
    return s / np.linalg.norm(s, axis=1, keepdims=True)


def _pl_observable(term):
    import pennylane as qml

    P = {1: qml.PauliX, 2: qml.PauliY, 3: qml.PauliZ}
    op = P[term[0][1]](term[0][0])
    for w, p in term[1:]:
        op = op @ P[p](w)
    return op


def validate(n_qubits: int = 4, n_circuits: int = 5, noise: float = 0.05, shots: int = 4000, seed: int = 0) -> dict:
    import pennylane as qml

    rng = np.random.default_rng(seed)
    n = n_qubits
    obs = make_observables(n, "local2")
    pl_obs = [_pl_observable(t) for t in obs.terms]
    dev_sv = qml.device("default.qubit", wires=n)
    dev_dm = qml.device("default.mixed", wires=n)
    report = {"statevector_max_err": 0.0, "noisy_expval_max_err": 0.0, "shadow_vs_pl_max_z": 0.0}

    for _ in range(n_circuits):
        chrom = random_chromosome(n, 10, rng, ("RX", "RZ", "H", "CNOT"))
        state = _random_states(n, 1, rng)
        ops = to_pennylane_ops(chrom)  # built outside the QNodes so they are not queued twice

        @qml.qnode(dev_sv)
        def sv():
            qml.StatePrep(state[0], wires=range(n))
            for op in ops:
                qml.apply(op)
            return qml.state()

        ours = run_statevector(chrom, states_to_tensor(state, n)).reshape(-1)
        report["statevector_max_err"] = max(report["statevector_max_err"], float(np.abs(ours - sv()).max()))

        @qml.qnode(dev_dm)
        def noisy():
            qml.StatePrep(state[0], wires=range(n))
            for op in ops:
                qml.apply(op)
                for w in op.wires:
                    qml.DepolarizingChannel(noise, wires=w)
            return [qml.expval(o) for o in pl_obs]

        ours_noisy = exact_features(chrom, state, obs, noise, "local")[0]
        report["noisy_expval_max_err"] = max(report["noisy_expval_max_err"],
                                             float(np.abs(ours_noisy - np.array(noisy())).max()))

        @qml.qnode(qml.device("default.qubit", wires=n, shots=shots, seed=int(rng.integers(1 << 31))))
        def pl_shadow():
            qml.StatePrep(state[0], wires=range(n))
            for op in ops:
                qml.apply(op)
            return qml.shadow_expval(pl_obs)

        exact = exact_features(chrom, state, obs)[0]
        ours_sh = shadow_features(chrom, state, obs, shots, rng)[0]
        theirs = np.array(pl_shadow())
        # both are unbiased estimators of `exact`; compare each against it in
        # units of the shadow std bound sqrt(3^k / T)
        weights = np.array([len(t) for t in obs.terms])
        sd = 3.0 ** (weights / 2) / np.sqrt(shots)  # std bound sqrt(3^k / T) for a weight-k Pauli
        report["shadow_vs_pl_max_z"] = max(report["shadow_vs_pl_max_z"], float(np.max(np.abs(ours_sh - exact) / sd)))
        report["pl_shadow_max_z"] = max(report.get("pl_shadow_max_z", 0.0), float(np.max(np.abs(theirs - exact) / sd)))
    report["passed"] = (report["statevector_max_err"] < 1e-10 and report["noisy_expval_max_err"] < 1e-10
                        and report["shadow_vs_pl_max_z"] < 5.0)
    return report
