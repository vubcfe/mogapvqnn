"""Batched NumPy simulation of chromosome circuits.

States are stored as tensors of shape ``(B, 2, ..., 2)`` (one axis per
wire, wire 0 = most significant bit, matching PennyLane). Density matrices
are stored as ``(B, 2,...,2, 2,...,2)`` with the ``n`` row axes followed by
the ``n`` column axes.

Noise model (Phase 2)
---------------------
``local``  : a single-qubit depolarizing channel
             rho -> (1-p) rho + p/3 (X rho X + Y rho Y + Z rho Z)
             (PennyLane ``DepolarizingChannel(p)``) on every wire a gate acts
             on, after every non-identity gate (``I`` genes are absent from
             the circuit, not idle time steps).
``global`` : rho -> (1-p) rho + p I/2^n applied once at the end of the circuit
             (the channel written in the original manuscript).

For sampling, noise is simulated with quantum trajectories (random Pauli
insertions), which reproduces the density-matrix measurement statistics
exactly while keeping memory at O(2^n) per sample. The density-matrix
simulator is kept for validation.
"""

from __future__ import annotations

import numpy as np

from .circuits import Chromosome, Gene, active

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
SDG = np.diag([1, -1j]).astype(complex)
PAULIS = np.stack([I2, X, Y, Z])  # index 0..3


def rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])


def single_qubit_matrix(g: Gene) -> np.ndarray:
    if g.gate == "H":
        return H
    if g.gate == "RX":
        return rx(g.angle)
    if g.gate == "RY":
        return ry(g.angle)
    if g.gate == "RZ":
        return rz(g.angle)
    if g.gate == "I":
        return I2
    raise ValueError(g.gate)


# --------------------------------------------------------------------------- #
# Tensor primitives
# --------------------------------------------------------------------------- #

def apply_matrix(t: np.ndarray, U: np.ndarray, axis: int) -> np.ndarray:
    """Contract a 2x2 matrix (or a per-sample stack ``(B,2,2)``) into ``axis``.

    The tensor is viewed as (B, L, 2, R) around ``axis`` and contracted with
    a broadcast ``matmul``, which avoids moveaxis copies and is ~2x faster
    than the equivalent einsum (this is the hot loop of shadow sampling)."""
    sh = t.shape
    L = int(np.prod(sh[1:axis], dtype=np.int64))
    R = int(np.prod(sh[axis + 1:], dtype=np.int64))
    v = t.reshape(sh[0], L, 2, R)
    if U.ndim == 2:
        out = np.matmul(U, v)
    else:
        out = np.matmul(U[:, None], v)
    return out.reshape(sh)


def apply_cnot_axes(t: np.ndarray, c_axis: int, t_axis: int) -> np.ndarray:
    """Flip ``t_axis`` on the slice where ``c_axis`` == 1."""
    out = t.copy()
    idx = [slice(None)] * t.ndim
    idx[c_axis] = 1
    idx = tuple(idx)
    sub_axis = t_axis - 1 if t_axis > c_axis else t_axis
    out[idx] = np.flip(t[idx], axis=sub_axis)
    return out


def states_to_tensor(states: np.ndarray, n_qubits: int) -> np.ndarray:
    states = np.asarray(states, dtype=complex)
    if states.ndim != 2 or states.shape[1] != 2**n_qubits:
        raise ValueError(f"expected states of shape (B, {2**n_qubits}), got {states.shape}")
    return states.reshape((states.shape[0],) + (2,) * n_qubits)


def tensor_to_states(psi: np.ndarray) -> np.ndarray:
    return psi.reshape(psi.shape[0], -1)


# --------------------------------------------------------------------------- #
# Statevector simulation (with optional trajectory noise)
# --------------------------------------------------------------------------- #

def _apply_gene(psi: np.ndarray, g: Gene) -> np.ndarray:
    if g.gate == "CNOT":
        c, t = g.wires
        return apply_cnot_axes(psi, c + 1, t + 1)
    return apply_matrix(psi, single_qubit_matrix(g), g.wires[0] + 1)


def _random_paulis(psi: np.ndarray, wire: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Trajectory unravelling of DepolarizingChannel(p) on one wire: with
    probability p apply X, Y or Z (uniformly), otherwise nothing.

    Modifies ``psi`` in place; callers only pass arrays they own (every gate
    application returns a fresh array)."""
    hit = rng.random(psi.shape[0]) < p
    if not hit.any():
        return psi
    err = 1 + rng.integers(0, 3, size=int(hit.sum()))
    psi[hit] = apply_matrix(psi[hit], PAULIS[err], wire + 1)
    return psi


def run_statevector(
    chrom: Chromosome,
    psi: np.ndarray,
    noise: float = 0.0,
    noise_model: str = "local",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply ``chrom`` to a batch of states ``psi`` (tensor form).

    With ``noise > 0`` and ``noise_model == 'local'`` each call samples one
    noise trajectory per batch element (requires ``rng``). Global
    depolarizing is not a unitary mixture per gate and is handled at
    measurement time (see ``shadows.py``).
    """
    if noise > 0 and noise_model == "global":
        raise NotImplementedError("global depolarizing acts at measurement time; "
                                  "use shadows.shadow_features or run_density")
    use_traj = noise > 0 and noise_model == "local"
    if use_traj and rng is None:
        raise ValueError("rng required for noisy trajectories")
    for g in active(chrom):
        psi = _apply_gene(psi, g)
        if use_traj:
            for w in g.wires:
                psi = _random_paulis(psi, w, noise, rng)
    return psi


# --------------------------------------------------------------------------- #
# Density-matrix simulation (exact; used for validation and small n)
# --------------------------------------------------------------------------- #

def states_to_density(states: np.ndarray, n_qubits: int) -> np.ndarray:
    states = np.asarray(states, dtype=complex)
    rho = np.einsum("bi,bj->bij", states, states.conj())
    return rho.reshape((states.shape[0],) + (2,) * (2 * n_qubits))


def _dm_apply_1q(rho: np.ndarray, U: np.ndarray, w: int, n: int) -> np.ndarray:
    rho = apply_matrix(rho, U, 1 + w)
    return apply_matrix(rho, U.conj(), 1 + n + w)


def _dm_depolarize(rho: np.ndarray, p: float, w: int, n: int) -> np.ndarray:
    """DepolarizingChannel(p) on wire w, using X.X + Y.Y + Z.Z = 2 Tr_w(.) (x) I - (.):
    rho -> (1 - 4p/3) rho + (2p/3) Tr_w(rho) (x) I_w   (one pass instead of six)."""
    r, c = 1 + w, 1 + n + w
    traced = np.trace(rho, axis1=r, axis2=c)               # Tr_w rho, wire axes removed
    traced = np.expand_dims(np.expand_dims(traced, r), c)   # back to rho's rank, size-1 axes
    out = (1 - 4 * p / 3) * rho
    eye = np.eye(2).reshape([2 if a in (r, c) else 1 for a in range(rho.ndim)])
    out += (2 * p / 3) * traced * eye
    return out


def run_density(
    chrom: Chromosome, rho: np.ndarray, n_qubits: int, noise: float = 0.0, noise_model: str = "local"
) -> np.ndarray:
    n = n_qubits
    for g in active(chrom):
        if g.gate == "CNOT":
            c, t = g.wires
            rho = apply_cnot_axes(rho, 1 + c, 1 + t)
            rho = apply_cnot_axes(rho, 1 + n + c, 1 + n + t)
        else:
            rho = _dm_apply_1q(rho, single_qubit_matrix(g), g.wires[0], n)
        if noise > 0 and noise_model == "local":
            for w in g.wires:
                rho = _dm_depolarize(rho, noise, w, n)
    if noise > 0 and noise_model == "global":
        B = rho.shape[0]
        d = 2**n
        mixed = np.broadcast_to(np.eye(d, dtype=complex) / d, (B, d, d)).reshape(rho.shape)
        rho = (1 - noise) * rho + noise * mixed
    return rho


def density_to_matrix(rho: np.ndarray, n_qubits: int) -> np.ndarray:
    d = 2**n_qubits
    return rho.reshape(rho.shape[0], d, d)
