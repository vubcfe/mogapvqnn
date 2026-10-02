"""Chromosome representation of quantum circuits.

A chromosome is a fixed-length tuple of genes. Each gene is one gate from
the search gate set G = {RX, RZ, H, CNOT}; the no-op gene ``I`` lets the
search express circuits with fewer than ``n_genes`` gates ("up to d gates",
paper Sec. 3.1). Rotation genes carry an angle in [-pi, pi].

Wire 0 is the most significant qubit (PennyLane convention).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

SEARCH_GATES: tuple[str, ...] = ("RX", "RZ", "H", "CNOT")
ROTATION_GATES = frozenset({"RX", "RY", "RZ"})
ALL_GATES = frozenset({"I", "RX", "RY", "RZ", "H", "CNOT"})


@dataclass(frozen=True)
class Gene:
    gate: str
    wires: tuple[int, ...]
    angle: float | None = None

    def __post_init__(self):
        if self.gate not in ALL_GATES:
            raise ValueError(f"unknown gate {self.gate!r}")
        expected = 2 if self.gate == "CNOT" else 1
        if len(self.wires) != expected:
            raise ValueError(f"{self.gate} needs {expected} wire(s), got {self.wires}")
        if self.gate == "CNOT" and self.wires[0] == self.wires[1]:
            raise ValueError("CNOT control and target must differ")
        if (self.gate in ROTATION_GATES) != (self.angle is not None):
            raise ValueError(f"{self.gate}: angle must be set iff the gate is a rotation")

    def to_dict(self) -> dict:
        return {"gate": self.gate, "wires": list(self.wires), "angle": self.angle}

    @staticmethod
    def from_dict(d: dict) -> "Gene":
        return Gene(d["gate"], tuple(d["wires"]), d.get("angle"))


Chromosome = tuple[Gene, ...]


def wrap_angle(theta: float) -> float:
    """Map an angle into [-pi, pi)."""
    return float((theta + math.pi) % (2 * math.pi) - math.pi)


def random_gene(n_qubits: int, rng: np.random.Generator, gates: Sequence[str]) -> Gene:
    gate = gates[rng.integers(len(gates))]
    if gate == "CNOT":
        if n_qubits < 2:
            raise ValueError("CNOT needs at least 2 qubits")
        c, t = rng.choice(n_qubits, size=2, replace=False)
        return Gene("CNOT", (int(c), int(t)))
    w = int(rng.integers(n_qubits))
    if gate in ROTATION_GATES:
        return Gene(gate, (w,), float(rng.uniform(-math.pi, math.pi)))
    return Gene(gate, (w,))


def random_chromosome(
    n_qubits: int, n_genes: int, rng: np.random.Generator, gates: Sequence[str] = SEARCH_GATES
) -> Chromosome:
    return tuple(random_gene(n_qubits, rng, gates) for _ in range(n_genes))


def active(chrom: Iterable[Gene]) -> list[Gene]:
    return [g for g in chrom if g.gate != "I"]


def depth(chrom: Iterable[Gene], n_qubits: int) -> int:
    """Circuit depth under as-soon-as-possible layer scheduling."""
    level = [0] * n_qubits
    for g in active(chrom):
        if len(g.wires) == 1:
            level[g.wires[0]] += 1
        else:
            d = max(level[w] for w in g.wires) + 1
            for w in g.wires:
                level[w] = d
    return max(level) if level else 0


def cnot_count(chrom: Iterable[Gene]) -> int:
    return sum(1 for g in chrom if g.gate == "CNOT")


def gate_count(chrom: Iterable[Gene]) -> int:
    return len(active(chrom))


def noise_proxy(chrom: Chromosome, n_qubits: int, lam: float = 0.05) -> float:
    """Complexity-based noise-stability proxy S_hat (paper Eq. 2)."""
    return math.exp(-lam * (depth(chrom, n_qubits) + 2 * cnot_count(chrom)))


def chrom_key(chrom: Chromosome) -> str:
    """Canonical string key: equal keys <=> identical circuits. Angles are
    rounded to 1e-9 (stable across JSON round-trips, -0.0 == 0.0) and the
    wire of a no-op ``I`` gene is irrelevant, so it is dropped."""
    parts = []
    for g in chrom:
        if g.gate == "I":
            parts.append("I")
            continue
        a = "" if g.angle is None else f"{g.angle + 0.0:.9f}"
        parts.append(f"{g.gate}:{','.join(map(str, g.wires))}:{a}")
    return "|".join(parts)


def stable_hash32(text: str) -> int:
    """Process-independent 32-bit hash (Python's hash() is salted)."""
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=4).digest(), "little")


def describe(chrom: Chromosome) -> str:
    out = []
    for g in active(chrom):
        w = "->".join(f"q{x}" for x in g.wires)
        out.append(f"{g.gate}({w}{'' if g.angle is None else f', {g.angle:+.3f}'})")
    return " ".join(out) if out else "(identity)"


def to_json(chrom: Chromosome) -> list[dict]:
    return [g.to_dict() for g in chrom]


def from_json(data: list[dict]) -> Chromosome:
    return tuple(Gene.from_dict(d) for d in data)


def to_pennylane_ops(chrom: Chromosome):
    """Return the chromosome as a list of PennyLane operations (for validation)."""
    import pennylane as qml

    ops = []
    for g in active(chrom):
        if g.gate == "CNOT":
            ops.append(qml.CNOT(wires=list(g.wires)))
        elif g.gate == "H":
            ops.append(qml.Hadamard(wires=g.wires[0]))
        else:
            ops.append(getattr(qml, g.gate)(g.angle, wires=g.wires[0]))
    return ops


# --------------------------------------------------------------------------- #
# Fixed-structure circuits used by the baselines
# --------------------------------------------------------------------------- #

def manual_ansatz(n_qubits: int, layers: int, rng: np.random.Generator) -> Chromosome:
    """Alternating RY rotation layer + linear CNOT chain (baseline ii)."""
    genes: list[Gene] = []
    for _ in range(layers):
        genes += [Gene("RY", (w,), float(rng.uniform(-math.pi, math.pi))) for w in range(n_qubits)]
        genes += [Gene("CNOT", (w, w + 1)) for w in range(n_qubits - 1)]
    return tuple(genes)


def hardware_efficient_ansatz(n_qubits: int, layers: int, rng: np.random.Generator) -> Chromosome:
    """RY rotation layer + all-to-all CNOT entangler (baseline iii)."""
    genes: list[Gene] = []
    for _ in range(layers):
        genes += [Gene("RY", (w,), float(rng.uniform(-math.pi, math.pi))) for w in range(n_qubits)]
        genes += [Gene("CNOT", (i, j)) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    return tuple(genes)


def ansatz_expansion_circuits(n_qubits: int) -> list[Chromosome]:
    """Post-variational 'ansatz expansion' set (Huang & Rebentrost): the
    identity circuit plus first-order +/- pi/2 RY shifts on each qubit
    (baseline iv)."""
    circuits: list[Chromosome] = [tuple()]
    for w in range(n_qubits):
        for s in (+1, -1):
            circuits.append((Gene("RY", (w,), s * math.pi / 2),))
    return circuits
