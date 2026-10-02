"""MO-GA-PVQNN search: Phase 1 (noiseless NSGA-II), Phase 2 (noise
evaluation of the Pareto front) and greedy ensemble selection."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from .circuits import (
    ROTATION_GATES,
    SEARCH_GATES,
    Chromosome,
    Gene,
    chrom_key,
    cnot_count,
    depth,
    noise_proxy,
    random_chromosome,
    random_gene,
    wrap_angle,
)
from .classifier import accuracy, fit_linear, majority_vote
from .config import RunConfig
from .data import BinaryTask
from .features import FeatureExtractor
from .nsga2 import assign_rank_and_crowding, environmental_selection, hypervolume, tournament

# --------------------------------------------------------------------------- #
# Execution helpers
# --------------------------------------------------------------------------- #


class SerialExecutor(Executor):
    """In-thread executor so that ``make_executor`` always returns an Executor."""

    def submit(self, fn, /, *args, **kwargs):
        f: Future = Future()
        try:
            f.set_result(fn(*args, **kwargs))
        except BaseException as e:  # propagate like a real executor
            f.set_exception(e)
        return f

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return map(fn, *iterables)


def make_executor(workers: int) -> Executor:
    """Thread pool for ``workers > 1``, otherwise a serial executor. Use as a
    context manager (``with make_executor(n) as pool``)."""
    return ThreadPoolExecutor(max_workers=workers) if workers and workers > 1 else SerialExecutor()


def noise_levels(cfg: RunConfig) -> list[float]:
    """Sorted, de-duplicated strictly positive Phase-2 noise levels."""
    return sorted({float(e) for e in cfg.phase2.noise_levels if float(e) > 0})


# --------------------------------------------------------------------------- #
# Circuit evaluation
# --------------------------------------------------------------------------- #


class CircuitEvaluator:
    """Fits one linear read-out per (circuit, shot budget) on the fit split
    and returns decision values on any split / noise level."""

    def __init__(self, task: BinaryTask, extractor: FeatureExtractor, cfg: RunConfig,
                 labels: dict[str, np.ndarray] | None = None):
        self.task = task
        self.fx = extractor
        self.cfg = cfg
        self.labels = {"fit": task.y_fit, "val": task.y_val, "test": task.y_test}
        self.labels.update(labels or {})
        self._models: dict[tuple, object] = {}
        self._lock = threading.Lock()
        self.n_model_fits = 0

    def model(self, chrom: Chromosome, shots: int):
        mk = (chrom_key(chrom), shots)
        with self._lock:
            m = self._models.get(mk)
        if m is not None:
            return m
        F = self.fx.features(chrom, "fit", shots)
        m = fit_linear(F, self.task.y_fit, self.cfg.classifier.C, self.cfg.classifier.max_iter)
        with self._lock:
            kept = self._models.setdefault(mk, m)
            if kept is m:
                self.n_model_fits += 1
        return kept

    def decision(self, chrom: Chromosome, split: str, shots: int, noise: float = 0.0,
                 readout: tuple[float, bool] | None = None) -> np.ndarray:
        """Decision values of the clean-trained read-out on (possibly noisy) features."""
        m = self.model(chrom, shots)
        return m.decision_function(self.fx.features(chrom, split, shots, noise, readout))

    def acc(self, chrom: Chromosome, split: str, shots: int, noise: float = 0.0,
            readout: tuple[float, bool] | None = None) -> float:
        return accuracy((self.decision(chrom, split, shots, noise, readout) > 0).astype(int), self.labels[split])

    def prefetch(self, chroms: list[Chromosome], splits: list[str], shots: int, noises: list[float],
                 executor: Executor | None, readout: tuple[float, bool] | None = None) -> None:
        """Compute (in parallel) every feature matrix that the given
        combinations will need, so later serial ensemble scoring only hits
        the cache. Results are identical with or without prefetching."""
        if executor is None:
            return
        jobs = list(product(chroms, splits, noises))
        list(executor.map(lambda j: self.decision(j[0], j[1], shots, j[2], readout), jobs))

    def ensemble_acc(self, chroms: list[Chromosome], split: str, shots: int, noise: float = 0.0,
                     readout: tuple[float, bool] | None = None) -> float:
        if not chroms:
            return 0.0
        pred = majority_vote([self.decision(c, split, shots, noise, readout) for c in chroms])
        return accuracy(pred, self.labels[split])


def objective_vector(chrom: Chromosome, val_acc: float, cfg: RunConfig,
                     robust_acc: float | None = None) -> np.ndarray:
    """Normalised minimisation objectives (paper Eq. 1):
    [1 - Acc, Depth / L, CNOT / L, f_4] with f_4 (only if ``use_proxy``)
    either 1 - S_hat (``robust_objective = 'proxy'``) or 1 - Acc under
    depolarizing noise ``loop_noise`` (``'measured'``)."""
    s = cfg.search
    L = s.n_genes
    f = [1.0 - val_acc, depth(chrom, cfg.n_qubits) / L, cnot_count(chrom) / L]
    if s.use_proxy:
        if s.robust_objective == "proxy":
            f.append(1.0 - noise_proxy(chrom, cfg.n_qubits, s.proxy_lambda))
        elif s.robust_objective == "measured":
            if robust_acc is None:
                raise ValueError("measured robust objective needs robust_acc")
            f.append(1.0 - robust_acc)
        else:
            raise ValueError(f"unknown robust_objective {s.robust_objective!r}")
    return np.array(f)


def needs_measured_robustness(cfg: RunConfig) -> bool:
    return cfg.search.use_proxy and cfg.search.robust_objective == "measured"


def normalised_hv(F: np.ndarray, ref: float) -> float:
    m = F.shape[1]
    return hypervolume(F, np.full(m, ref)) / ref**m


# --------------------------------------------------------------------------- #
# Genetic operators
# --------------------------------------------------------------------------- #


def crossover(a: Chromosome, b: Chromosome, rng: np.random.Generator) -> tuple[Chromosome, Chromosome]:
    """Single-point crossover with the cut in [1, L-1]."""
    L = len(a)
    if L < 2:
        return a, b
    cut = int(rng.integers(1, L))
    return a[:cut] + b[cut:], b[:cut] + a[cut:]


def mutate(chrom: Chromosome, n_qubits: int, rng: np.random.Generator, rate: float,
           sigma: float, gates: tuple[str, ...]) -> Chromosome:
    """Per gene, with probability ``rate``: replace the gate type (re-sampling
    wires and angle). Otherwise, for rotation genes, with probability
    ``rate`` perturb the angle by N(0, sigma)."""
    out = []
    for g in chrom:
        if rng.random() < rate:
            out.append(random_gene(n_qubits, rng, gates))
        elif g.gate in ROTATION_GATES and rng.random() < rate:
            out.append(Gene(g.gate, g.wires, wrap_angle(g.angle + rng.normal(0.0, sigma))))
        else:
            out.append(g)
    return tuple(out)


# --------------------------------------------------------------------------- #
# Phase 1: NSGA-II
# --------------------------------------------------------------------------- #


@dataclass
class Phase1Result:
    population: list[Chromosome]
    objectives: np.ndarray
    front: list[int]                 # indices into population (unique circuits)
    history: list[dict] = field(default_factory=list)
    generations_run: int = 0
    stopped_early: bool = False
    n_unique_evaluations: int = 0
    seconds: float = 0.0


def _dedupe(pop: list[Chromosome]):
    seen, keep, dup = set(), [], []
    for i, c in enumerate(pop):
        k = chrom_key(c)
        (dup if k in seen else keep).append(i)
        seen.add(k)
    return keep, dup


def run_phase1(cfg: RunConfig, ev: CircuitEvaluator, rng: np.random.Generator,
               executor: Executor | None = None, log=None) -> Phase1Result:
    t0 = time.perf_counter()
    s = cfg.search
    n = cfg.n_qubits
    shots = cfg.shadows.base_shots
    mut_gates = SEARCH_GATES + (("I",) if s.allow_identity else ())
    evaluated: set[str] = set()

    measured = needs_measured_robustness(cfg)

    def fitness(c: Chromosome):
        clean = ev.acc(c, "val", shots)
        robust = ev.acc(c, "val", shots, s.loop_noise) if measured else None
        return clean, robust

    def evaluate(chroms: list[Chromosome]) -> np.ndarray:
        mapper = executor.map if executor is not None else map
        res = list(mapper(fitness, chroms))
        evaluated.update(chrom_key(c) for c in chroms)
        return np.stack([objective_vector(c, a, cfg, r) for c, (a, r) in zip(chroms, res)])

    def record(gen: int, pop, F) -> dict:
        fronts, _, _ = assign_rank_and_crowding(F, s.epsilon_dominance)
        f0 = F[fronts[0]]
        rec = {
            "generation": gen,
            "hv": normalised_hv(f0, s.hv_reference),
            "front_size": int(len({chrom_key(pop[i]) for i in fronts[0]})),
            "best_val_acc": float(1 - F[:, 0].min()),
            "mean_val_acc": float(1 - F[:, 0].mean()),
            "unique_evaluations": len(evaluated),
            "cache_hit_rate": ev.fx.hit_rate,
            "elapsed_s": time.perf_counter() - t0,
        }
        if log:
            log(f"  gen {gen:2d}  HV={rec['hv']:.4f}  |F0|={rec['front_size']:2d}  "
                f"best={rec['best_val_acc']:.3f}  evals={rec['unique_evaluations']}")
        return rec

    pop = [random_chromosome(n, s.n_genes, rng, SEARCH_GATES) for _ in range(s.population)]
    F = evaluate(pop)
    history = [record(0, pop, F)]
    best_hv, stall, stopped, gen = history[0]["hv"], 0, False, 0

    for gen in range(1, s.generations + 1):
        _, rank, crowd = assign_rank_and_crowding(F, s.epsilon_dominance)
        offspring: list[Chromosome] = []
        while len(offspring) < s.population:
            a = pop[tournament(rank, crowd, s.tournament_k, rng)]
            b = pop[tournament(rank, crowd, s.tournament_k, rng)]
            if rng.random() < s.crossover_rate:
                a, b = crossover(a, b, rng)
            offspring.append(mutate(a, n, rng, s.mutation_rate, s.angle_sigma, mut_gates))
            if len(offspring) < s.population:
                offspring.append(mutate(b, n, rng, s.mutation_rate, s.angle_sigma, mut_gates))
        F_off = evaluate(offspring)
        union = pop + offspring
        F_union = np.vstack([F, F_off])
        keep, dup = _dedupe(union)
        sel_keep = environmental_selection(F_union[keep], min(s.population, len(keep)), s.epsilon_dominance)
        chosen = [keep[i] for i in sel_keep] + dup[: max(0, s.population - len(keep))]
        assert len(chosen) == s.population, (len(chosen), s.population)
        pop = [union[i] for i in chosen]
        F = F_union[chosen]
        rec = record(gen, pop, F)
        history.append(rec)
        if rec["hv"] > best_hv + s.early_stop_tol:
            best_hv, stall = rec["hv"], 0
        else:
            stall += 1
            if stall >= s.early_stop_patience:
                stopped = True
                break

    fronts, _, _ = assign_rank_and_crowding(F, s.epsilon_dominance)
    seen, front = set(), []
    for i in fronts[0]:
        k = chrom_key(pop[i])
        if k not in seen:
            seen.add(k)
            front.append(int(i))
    return Phase1Result(pop, F, front, history, gen, stopped, len(evaluated), time.perf_counter() - t0)


# --------------------------------------------------------------------------- #
# Phase 2: noise evaluation and ensemble selection
# --------------------------------------------------------------------------- #


def phase2_evaluate(cands: list[Chromosome], ev: CircuitEvaluator, cfg: RunConfig,
                    val_split: str, test_split: str, executor: Executor | None = None) -> list[dict]:
    """For every candidate and noise level (including 0) compute accuracy of
    its clean-trained read-out on noisy features of the *same* samples."""
    shots = cfg.shadows.max_shots
    nz = noise_levels(cfg)
    levels = [0.0] + nz

    def one(c):
        row = {"val": {}, "test": {}}
        for eps in levels:
            row["val"][eps] = ev.acc(c, val_split, shots, eps)
            row["test"][eps] = ev.acc(c, test_split, shots, eps)
        row["robust_val"] = float(np.mean([row["val"][e] for e in nz])) if nz else row["val"][0.0]
        row["robust_test"] = float(np.mean([row["test"][e] for e in nz])) if nz else row["test"][0.0]
        return row

    mapper = executor.map if executor is not None else map
    return list(mapper(one, cands))


def _complexity(c: Chromosome, n: int) -> int:
    return depth(c, n) + 2 * cnot_count(c)


def greedy_ensemble(cands: list[Chromosome], ev: CircuitEvaluator, cfg: RunConfig, beta: float,
                    val_p2: str) -> tuple[list[int], list[float]]:
    """Greedy forward selection (paper Eq. 7) maximising
    (1 - beta) * Acc_val(S) + beta * mean_eps Acc_val_p2(S, eps).

    The clean term uses the full validation split; the robust term uses the
    Phase-2 validation subset ``val_p2`` (identical to the full split unless
    ``phase2.val_samples`` sub-samples it to bound noisy-simulation cost).
    All decision values are cached, so repeated scoring is cheap."""
    shots = cfg.shadows.max_shots
    levels = noise_levels(cfg)
    K = min(cfg.ensemble.size, len(cands))
    n = cfg.n_qubits

    def score(members: list[int]) -> float:
        chroms = [cands[i] for i in members]
        clean = ev.ensemble_acc(chroms, "val", shots)
        if beta == 0 or not levels:
            return clean
        robust = np.mean([ev.ensemble_acc(chroms, val_p2, shots, e) for e in levels])
        return (1 - beta) * clean + beta * robust

    chosen: list[int] = []
    trace: list[float] = []
    remaining = list(range(len(cands)))
    while len(chosen) < K and remaining:
        best = max(remaining, key=lambda i: (score(chosen + [i]), -_complexity(cands[i], n), -i))
        chosen.append(best)
        remaining.remove(best)
        trace.append(score(chosen))
    return chosen, trace


def topk_ensemble(cands: list[Chromosome], val_accs: list[float], K: int) -> list[int]:
    order = sorted(range(len(cands)), key=lambda i: -val_accs[i])
    return order[: min(K, len(cands))]


