"""Classical-shadow feature extraction (Huang, Kueng & Preskill 2020).

Random Pauli-basis measurement: in every round each qubit is measured in a
uniformly random basis X, Y or Z (rotation H, H S^dagger or I followed by a
computational-basis measurement). For a Pauli observable
P = P_{w1} (x) ... (x) P_{wk} the single-snapshot unbiased estimator is

    prod_{j=1..k} 3 * s_{wj} * [basis_{wj} == P_{wj}],      s = (-1)^bit,

i.e. the classical shadow  rho_hat = (x)_i (3 U_i^dag |b_i><b_i| U_i - I)
evaluated on P. Features are the mean over T rounds. The basis is drawn
independently for every sample and every round.

Readout noise: every measured bit is flipped independently with probability
q. For the Pauli protocol this multiplies the expectation of each
single-qubit factor by f = 1 - 2q, so a weight-k estimate is biased by f^k.
With ``mitigate_readout`` the snapshot factors are divided by f, which is the
calibrated "robust shadow" estimator for this noise (unbiased again, at the
cost of a variance increase f^-2k); cf. Chen et al. PRX Quantum 2 (2021),
Hu et al. Nat. Commun. 16 (2025).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circuits import Chromosome
from .simulator import (
    H,
    I2,
    PAULIS,
    SDG,
    apply_matrix,
    run_density,
    run_statevector,
    states_to_density,
    states_to_tensor,
)

# Pauli codes follow simulator.PAULIS: 1 = X, 2 = Y, 3 = Z.
PAULI_NAMES = {1: "X", 2: "Y", 3: "Z"}
# Measurement-basis rotation U with U^dag Z U = P for basis 0 = X, 1 = Y, 2 = Z.
BASIS_ROT = np.stack([H, H @ SDG, I2])


@dataclass(frozen=True)
class ObservableSet:
    """Pauli observables of weight 1 or 2, stored as index arrays."""

    n_qubits: int
    terms: tuple[tuple[tuple[int, int], ...], ...]  # ((wire, pauli), ...)

    @property
    def size(self) -> int:
        return len(self.terms)

    def names(self) -> list[str]:
        return ["".join(f"{PAULI_NAMES[p]}{w}" for w, p in t) for t in self.terms]

    def arrays(self):
        wa = np.array([t[0][0] for t in self.terms])
        pa = np.array([t[0][1] for t in self.terms])
        has_b = np.array([len(t) == 2 for t in self.terms])
        wb = np.array([t[1][0] if len(t) == 2 else 0 for t in self.terms])
        pb = np.array([t[1][1] if len(t) == 2 else 3 for t in self.terms])
        return wa, pa, has_b, wb, pb


def make_observables(n_qubits: int, kind: str = "local2") -> ObservableSet:
    """``local1``: X_i, Y_i, Z_i for all i (3n terms).
    ``local2``: local1 plus nearest-neighbour X_iX_{i+1} and Z_iZ_{i+1}
    (3n + 2(n-1) terms)."""
    terms: list[tuple[tuple[int, int], ...]] = []
    for w in range(n_qubits):
        for p in (1, 2, 3):
            terms.append(((w, p),))
    if kind == "local2":
        for w in range(n_qubits - 1):
            terms.append(((w, 1), (w + 1, 1)))
            terms.append(((w, 3), (w + 1, 3)))
    elif kind != "local1":
        raise ValueError(f"unknown observable set {kind!r}")
    return ObservableSet(n_qubits, tuple(terms))


def sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cum = np.cumsum(probs, axis=1)
    u = rng.random(probs.shape[0]) * cum[:, -1]
    idx = (cum < u[:, None]).sum(axis=1)
    return np.minimum(idx, probs.shape[1] - 1)


def shadow_features(
    chrom: Chromosome,
    states: np.ndarray,
    obs: ObservableSet,
    shots: int,
    rng: np.random.Generator,
    noise: float = 0.0,
    noise_model: str = "local",
    max_elems: int = 1 << 22,
    readout_error: float = 0.0,
    mitigate_readout: bool = False,
) -> np.ndarray:
    """Classical-shadow estimates of ``obs`` on ``chrom |psi_b>`` for every
    input state. Returns an array of shape ``(B, obs.size)``.

    With local noise every round re-simulates the circuit along a fresh
    random Pauli trajectory per sample; without noise the circuit output is
    computed once and re-measured.
    """
    n = obs.n_qubits
    D = 2**n
    psi0 = states_to_tensor(states, n)
    B = psi0.shape[0]
    traj = noise > 0 and noise_model == "local"
    final = None if traj else run_statevector(chrom, psi0)

    if not 0.0 <= readout_error < 0.5:
        raise ValueError("readout_error must be in [0, 0.5)")
    gain = 3.0 / (1.0 - 2.0 * readout_error) if mitigate_readout else 3.0
    wa, pa, has_b, wb, pb = obs.arrays()
    shifts = np.arange(n - 1, -1, -1)
    acc = np.zeros((obs.size, B))
    chunk = max(1, min(shots, max_elems // (B * D)))
    done = 0
    while done < shots:
        r = min(chunk, shots - done)
        reps = (r,) + (1,) * n
        if traj:
            phi = run_statevector(chrom, np.tile(psi0, reps), noise, "local", rng)
        else:
            phi = np.tile(final, reps)
        bases = rng.integers(0, 3, size=(r * B, n))
        for w in range(n):
            phi = apply_matrix(phi, BASIS_ROT[bases[:, w]], w + 1)
        probs = np.abs(phi.reshape(r * B, D)) ** 2
        outcome = sample_categorical(probs, rng)
        bits = (outcome[:, None] >> shifts) & 1
        if noise > 0 and noise_model == "global":
            flip = rng.random(r * B) < noise
            bits[flip] = rng.integers(0, 2, size=(int(flip.sum()), n))
        if readout_error > 0:
            bits = bits ^ (rng.random(bits.shape) < readout_error)
        s = 1 - 2 * bits
        # single[k, w, i]: g s_w [basis_w == k] for basis k in {X, Y, Z}, g = 3 (or 3/f mitigated)
        single = gain * (bases.T[None, :, :] == np.arange(3)[:, None, None]) * s.T[None, :, :]
        est = single[pa - 1, wa]
        est = est * np.where(has_b[:, None], single[pb - 1, wb], 1.0)
        acc += est.reshape(obs.size, r, B).sum(axis=1)
        done += r
    return (acc / shots).T


def _pauli_term_on(t: np.ndarray, term, offset: int = 1) -> np.ndarray:
    for w, p in term:
        t = apply_matrix(t, PAULIS[p], offset + w)
    return t


def exact_features(
    chrom: Chromosome,
    states: np.ndarray,
    obs: ObservableSet,
    noise: float = 0.0,
    noise_model: str = "local",
    max_elems: int = 1 << 22,
    readout_error: float = 0.0,
    mitigate_readout: bool = False,
) -> np.ndarray:
    """Exact expectation values tr(P rho) (no shot noise). Noisy circuits use
    the density-matrix simulator, processed in sample chunks so that at most
    ``max_elems`` complex entries are held per density-matrix batch.
    Unmitigated readout error scales a weight-k term by (1 - 2q)^k."""
    out = _exact_expectations(chrom, states, obs, noise, noise_model, max_elems)
    if readout_error > 0 and not mitigate_readout:
        weights = np.array([len(t) for t in obs.terms])
        out = out * (1.0 - 2.0 * readout_error) ** weights[None, :]
    return out


def _exact_expectations(chrom, states, obs, noise, noise_model, max_elems):
    n = obs.n_qubits
    B = states.shape[0]
    out = np.zeros((B, obs.size))
    if noise == 0:
        psi = run_statevector(chrom, states_to_tensor(states, n))
        flat = psi.reshape(B, -1)
        for k, term in enumerate(obs.terms):
            phi = _pauli_term_on(psi, term).reshape(B, -1)
            out[:, k] = np.real(np.sum(flat.conj() * phi, axis=1))
        return out
    step = max(1, max_elems // 4**n)
    supports = sorted({tuple(w for w, _ in t) for t in obs.terms})
    for lo in range(0, B, step):
        rho = run_density(chrom, states_to_density(states[lo : lo + step], n), n, noise, noise_model)
        # reduced density matrices of every 1-/2-qubit support, computed once
        rdms = {S: _reduced_density(rho, n, S) for S in supports}
        for k, term in enumerate(obs.terms):
            S = tuple(w for w, _ in term)
            P = PAULIS[term[0][1]]
            for _, p in term[1:]:
                P = np.kron(P, PAULIS[p])
            out[lo : lo + step, k] = np.real(np.einsum("bij,ji->b", rdms[S], P))
    return out


def _reduced_density(rho: np.ndarray, n: int, keep: tuple[int, ...]) -> np.ndarray:
    """Partial trace of a density tensor (B, 2^n row axes, 2^n column axes)
    onto the wires ``keep`` (in that order); returns (B, 2^k, 2^k)."""
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    row = list(letters[:n])
    col = [row[w] if w not in keep else letters[n + w] for w in range(n)]
    out = "Z" + "".join(row[w] for w in keep) + "".join(col[w] for w in keep)
    red = np.einsum("Z" + "".join(row) + "".join(col) + "->" + out, rho)
    d = 2 ** len(keep)
    return red.reshape(rho.shape[0], d, d)
