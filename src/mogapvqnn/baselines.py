"""Baselines (paper Sec. 4.2).

Quantum baselines use the same data splits, observables, shot budget
(``shadows.max_shots``) and linear read-out as the final MO-GA-PVQNN
ensemble, so differences come from the circuits only:

* ``random``        K random circuits (no search)
* ``random_search`` equal-budget random search: as many random circuits as
                    the GA evaluated, then the *same* Pareto filtering and
                    greedy ensemble selection as MO-GA-PVQNN. Isolates the
                    contribution of the evolutionary operators (random search
                    is a strong NAS baseline, Deligkaris 2024).
* ``manual``, ``hea``, ``pvqnn``, ``grid``  fixed / swept structures

Classical references (no quantum circuit), on the same amplitude-encoded,
l2-normalised inputs. With amplitude encoding every Pauli feature
<psi|U^dag P U|psi> is a quadratic form of the input vector, so a degree-2
polynomial kernel spans the whole function class reachable by a linear
read-out on such features (cf. Bermejo et al. 2024; Jerbi et al. 2024):

* ``linear``  StandardScaler + LinearSVC on the input vector
* ``poly2``   SVC with kernel (x.x' + 1)^2 on the input vector
"""

from __future__ import annotations

import time
from concurrent.futures import Executor

import numpy as np
from sklearn.svm import SVC

from .circuits import (
    SEARCH_GATES,
    Chromosome,
    ansatz_expansion_circuits,
    chrom_key,
    hardware_efficient_ansatz,
    manual_ansatz,
    random_chromosome,
    random_gene,
    stable_hash32,
    to_json,
)
from .classifier import accuracy, fit_linear
from .config import RunConfig
from .nsga2 import assign_rank_and_crowding, environmental_selection
from .search import CircuitEvaluator, greedy_ensemble, needs_measured_robustness, noise_levels, objective_vector


def _rng(cfg: RunConfig, name: str) -> np.random.Generator:
    return np.random.default_rng([cfg.seed, 101, stable_hash32(name)])


def _ensemble_report(chroms: list[Chromosome], ev: CircuitEvaluator, cfg: RunConfig,
                     test_p2: str, noise_eval: bool, executor: Executor | None = None) -> dict:
    shots = cfg.shadows.max_shots
    levels = [0.0] + noise_levels(cfg)
    ev.prefetch(chroms, ["val", "test"], shots, [0.0], executor)
    if noise_eval:
        ev.prefetch(chroms, [test_p2], shots, levels, executor)
    out = {
        "val_acc": ev.ensemble_acc(chroms, "val", shots),
        "test_acc": ev.ensemble_acc(chroms, "test", shots),
        "circuits": [to_json(c) for c in chroms],
    }
    if noise_eval:
        out["noisy_test"] = {str(e): ev.ensemble_acc(chroms, test_p2, shots, e) for e in levels}
    return out


def random_ensemble(cfg, ev, test_p2, noise_eval, executor=None):
    rng = _rng(cfg, "random")
    K = cfg.ensemble.size
    chroms = [random_chromosome(cfg.n_qubits, cfg.search.n_genes, rng, SEARCH_GATES) for _ in range(K)]
    return _ensemble_report(chroms, ev, cfg, test_p2, noise_eval, executor)


def manual_baseline(cfg, ev, test_p2, noise_eval, executor=None):
    rng = _rng(cfg, "manual")
    chroms = [manual_ansatz(cfg.n_qubits, cfg.baselines.manual_layers, rng) for _ in range(cfg.ensemble.size)]
    return _ensemble_report(chroms, ev, cfg, test_p2, noise_eval, executor)


def hea_baseline(cfg, ev, test_p2, noise_eval, executor=None):
    rng = _rng(cfg, "hea")
    chroms = [hardware_efficient_ansatz(cfg.n_qubits, cfg.baselines.hea_layers, rng)
              for _ in range(cfg.ensemble.size)]
    return _ensemble_report(chroms, ev, cfg, test_p2, noise_eval, executor)


def pvqnn_baseline(cfg, ev, test_p2, noise_eval, executor=None):
    """Post-variational QNN without architecture search: features of the
    fixed ansatz-expansion circuits are concatenated into one linear model."""
    shots = cfg.shadows.max_shots
    circuits = ansatz_expansion_circuits(cfg.n_qubits)
    levels = [0.0] + noise_levels(cfg)
    if executor is not None:   # compute all feature matrices in parallel first
        jobs = [(c, sp, 0.0) for c in circuits for sp in ("fit", "val", "test")]
        if noise_eval:
            jobs += [(c, test_p2, e) for c in circuits for e in levels]
        list(executor.map(lambda j: ev.fx.features(j[0], j[1], shots, j[2]), jobs))

    def feats(split, noise=0.0):
        return np.hstack([ev.fx.features(c, split, shots, noise) for c in circuits])

    model = fit_linear(feats("fit"), ev.labels["fit"], cfg.classifier.C, cfg.classifier.max_iter)

    def acc(split, noise=0.0):
        return accuracy(model.predict(feats(split, noise)), ev.labels[split])

    out = {"val_acc": acc("val"), "test_acc": acc("test"), "circuits": [to_json(c) for c in circuits]}
    if noise_eval:
        out["noisy_test"] = {str(e): acc(test_p2, e) for e in levels}
    return out


def _grid_chromosome(n_qubits: int, n_genes: int, n_cnot: int, rng) -> Chromosome:
    genes = [random_gene(n_qubits, rng, ("RX", "RZ", "H")) for _ in range(n_genes - n_cnot)]
    genes += [random_gene(n_qubits, rng, ("CNOT",)) for _ in range(n_cnot)]
    order = rng.permutation(len(genes))
    return tuple(genes[i] for i in order)


def grid_baseline(cfg, ev, test_p2, noise_eval, executor=None):
    """Sweep circuit length x CNOT ratio; for each cell draw an ensemble of
    K random circuits with that structure and keep the best cell on the
    validation split."""
    rng = _rng(cfg, "grid")
    shots = cfg.shadows.max_shots
    cells = []
    for d in cfg.baselines.grid_depths:              # circuits drawn serially: RNG order fixed
        for r in cfg.baselines.grid_cnot_ratios:
            n_cnot = int(round(r * d)) if cfg.n_qubits > 1 else 0
            chroms = [_grid_chromosome(cfg.n_qubits, d, n_cnot, rng) for _ in range(cfg.ensemble.size)]
            cells.append({"genes": d, "cnot_ratio": r, "chroms": chroms})
    ev.prefetch([c for cell in cells for c in cell["chroms"]], ["val"], shots, [0.0], executor)
    for cell in cells:
        cell["val_acc"] = ev.ensemble_acc(cell["chroms"], "val", shots)
    best = max(cells, key=lambda c: (c["val_acc"], -c["genes"], -c["cnot_ratio"]))
    out = _ensemble_report(best["chroms"], ev, cfg, test_p2, noise_eval, executor)
    out["selected_cell"] = {"genes": best["genes"], "cnot_ratio": best["cnot_ratio"]}
    out["grid"] = [{"genes": c["genes"], "cnot_ratio": c["cnot_ratio"], "val_acc": c["val_acc"]} for c in cells]
    return out


def random_search_baseline(cfg, ev, test_p2, noise_eval, budget: int = 0, executor: Executor | None = None):
    """Equal-budget random search followed by the MO-GA-PVQNN selection stage
    (Pareto filter on the same objectives, max-shot re-evaluation, greedy
    clean + robust ensemble)."""
    rng = _rng(cfg, "random_search")
    n_eval = cfg.baselines.random_search_budget or budget or cfg.search.population * (cfg.search.generations + 1)
    base, hi = cfg.shadows.base_shots, cfg.shadows.max_shots
    measured = needs_measured_robustness(cfg)
    pool, seen = [], set()
    attempts = 0
    while len(pool) < n_eval and attempts < 50 * n_eval:
        c = random_chromosome(cfg.n_qubits, cfg.search.n_genes, rng, SEARCH_GATES)
        attempts += 1
        if chrom_key(c) not in seen:
            seen.add(chrom_key(c))
            pool.append(c)
    mapper = executor.map if executor is not None else map

    def fitness(c):
        a = ev.acc(c, "val", base)
        r = ev.acc(c, "val", base, cfg.search.loop_noise) if measured else None
        return objective_vector(c, a, cfg, r)

    F = np.stack(list(mapper(fitness, pool)))
    # Same hand-off as NSGA-II: keep a "final population" of P survivors by
    # (rank, crowding) and use its first front, so random search does not get
    # more selection candidates than the GA (whose front is <= P circuits).
    survivors = environmental_selection(F, min(cfg.search.population, len(pool)), cfg.search.epsilon_dominance)
    fronts, _, _ = assign_rank_and_crowding(F[survivors], cfg.search.epsilon_dominance)
    front = [pool[survivors[i]] for i in fronts[0]]
    val_hi = list(mapper(lambda c: ev.acc(c, "val", hi), front))
    order = sorted(range(len(front)), key=lambda i: -val_hi[i])
    cap = cfg.phase2.max_circuits or len(front)
    cands = [front[i] for i in order[:cap]]
    ev.prefetch(cands, ["val_p2"], hi, noise_levels(cfg), executor)
    chosen, _ = greedy_ensemble(cands, ev, cfg, cfg.ensemble.robust_weight, "val_p2")
    out = _ensemble_report([cands[i] for i in chosen], ev, cfg, test_p2, noise_eval, executor)
    out["budget"] = len(pool)
    out["front_size"] = len(front)
    return out


def _input_states(ev: CircuitEvaluator, split: str) -> np.ndarray:
    return np.real(np.asarray(ev.fx.states[split]))


def linear_baseline(cfg, ev, test_p2, noise_eval, executor=None):
    model = fit_linear(_input_states(ev, "fit"), ev.labels["fit"], cfg.classifier.C, cfg.classifier.max_iter)
    acc = lambda split: accuracy(model.predict(_input_states(ev, split)), ev.labels[split])  # noqa: E731
    return {"val_acc": acc("val"), "test_acc": acc("test"), "classical": True}


def poly2_baseline(cfg, ev, test_p2, noise_eval, executor=None):
    model = SVC(kernel="poly", degree=2, gamma=1.0, coef0=1.0, C=cfg.classifier.C)
    model.fit(_input_states(ev, "fit"), ev.labels["fit"])
    acc = lambda split: accuracy(model.predict(_input_states(ev, split)), ev.labels[split])  # noqa: E731
    return {"val_acc": acc("val"), "test_acc": acc("test"), "classical": True}


BASELINES = {
    "random": random_ensemble,
    "random_search": random_search_baseline,
    "manual": manual_baseline,
    "hea": hea_baseline,
    "pvqnn": pvqnn_baseline,
    "grid": grid_baseline,
    "linear": linear_baseline,
    "poly2": poly2_baseline,
}
CLASSICAL_BASELINES = frozenset({"linear", "poly2"})


def run_baselines(cfg: RunConfig, ev: CircuitEvaluator, test_p2: str, noise_eval: bool = True, log=None,
                  budget: int = 0, executor: Executor | None = None) -> dict:
    enabled = list(dict.fromkeys(cfg.baselines.enabled))   # de-duplicate, keep order
    unknown = set(enabled) - set(BASELINES)
    if unknown:
        raise ValueError(f"unknown baseline(s) {sorted(unknown)}; known: {sorted(BASELINES)}")
    results = {}
    for name in enabled:
        t0 = time.perf_counter()
        if name == "random_search":
            r = random_search_baseline(cfg, ev, test_p2, noise_eval, budget, executor)
        else:
            r = BASELINES[name](cfg, ev, test_p2, noise_eval, executor)
        r["seconds"] = time.perf_counter() - t0
        results[name] = r
        if log:
            log(f"  baseline {name:7s} val={r['val_acc']:.3f} test={r['test_acc']:.3f} ({r['seconds']:.1f}s)")
    return results
