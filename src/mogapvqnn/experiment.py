"""One complete run: data -> Phase 1 -> front re-evaluation -> Phase 2 ->
ensembles -> baselines, saved as a single JSON record."""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from . import __version__
from .baselines import run_baselines
from .circuits import chrom_key, cnot_count, depth, describe, gate_count, noise_proxy, to_json
from .config import RunConfig, config_name
from .data import BinaryTask, make_task
from .features import FeatureExtractor
from .nsga2 import hypervolume
from .search import (
    CircuitEvaluator,
    greedy_ensemble,
    make_executor,
    noise_levels,
    normalised_hv,
    phase2_evaluate,
    run_phase1,
    topk_ensemble,
)
from .shadows import make_observables


def _balanced_subset(y: np.ndarray, k: int, seed: int) -> np.ndarray:
    if k <= 0 or k >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng([seed, 202])
    idx = []
    for label in (0, 1):
        pool = np.flatnonzero(y == label)
        want = k // 2 + (label and k % 2)
        if len(pool) < want:
            raise ValueError(f"Phase-2 subset: class {label} has {len(pool)} samples, {want} requested")
        idx.append(rng.choice(pool, size=want, replace=False))
    return np.sort(np.concatenate(idx))


def _versions() -> dict:
    import scipy
    import sklearn

    v = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
         "scikit-learn": sklearn.__version__, "mogapvqnn": __version__, "machine": platform.machine(),
         "processor": platform.processor() or platform.machine(), "node": platform.node()}
    try:
        import pennylane

        v["pennylane"] = pennylane.__version__
    except Exception:
        pass
    return v


def run_experiment(cfg: RunConfig, task: BinaryTask | None = None, log=print,
                   run_baseline_methods: bool = True) -> dict:
    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    if task is None:
        task = make_task(cfg.dataset, cfg.classes, cfg.n_qubits, cfg.n_train, cfg.n_test, cfg.seed,
                         cfg.val_fraction, cfg.resize)
    timings["data_s"] = time.perf_counter() - t0

    # Phase 2 sub-samples (same samples for clean and noisy evaluation)
    iv = _balanced_subset(task.y_val, cfg.phase2.val_samples, cfg.seed)
    it = _balanced_subset(task.y_test, cfg.phase2.test_samples, cfg.seed + 1)
    extra = {"val_p2": task.X_val[iv], "test_p2": task.X_test[it]}
    labels = {"val_p2": task.y_val[iv], "test_p2": task.y_test[it]}

    obs = make_observables(cfg.n_qubits, cfg.shadows.observables)
    fx = FeatureExtractor(task, obs, cfg.seed, cfg.shadows.estimator, cfg.phase2.noise_model, extra,
                          cfg.phase2.readout_error, cfg.phase2.readout_mitigation)
    ev = CircuitEvaluator(task, fx, cfg, labels)
    rng = np.random.default_rng([cfg.seed, 303])
    n = cfg.n_qubits
    log(f"[{config_name(cfg)} seed={cfg.seed}] fit/val/test = {len(task.y_fit)}/{len(task.y_val)}/"
        f"{len(task.y_test)}, {obs.size} observables, estimator={cfg.shadows.estimator}, "
        f"4th objective={cfg.search.robust_objective if cfg.search.use_proxy else 'off'}, "
        f"readout={cfg.phase2.readout_error}{' (mitigated)' if cfg.phase2.readout_mitigation else ''}")

    with make_executor(cfg.workers) as pool:
        # ---------------- Phase 1 ----------------
        p1 = run_phase1(cfg, ev, rng, pool, log)
        timings["phase1_s"] = p1.seconds
        front = [p1.population[i] for i in p1.front]
        F_front = p1.objectives[p1.front]

        # ---------------- front re-evaluation at max shots ----------------
        t0 = time.perf_counter()
        shots = cfg.shadows.max_shots
        mapper = pool.map if pool is not None else map
        val_hi = list(mapper(lambda c: ev.acc(c, "val", shots), front))
        test_hi = list(mapper(lambda c: ev.acc(c, "test", shots), front))
        timings["reeval_s"] = time.perf_counter() - t0

        order = sorted(range(len(front)), key=lambda i: -val_hi[i])
        cap = cfg.phase2.max_circuits or len(front)
        cand_idx = order[:cap]
        cands = [front[i] for i in cand_idx]

        # ---------------- Phase 2 ----------------
        t0 = time.perf_counter()
        p2 = phase2_evaluate(cands, ev, cfg, "val_p2", "test_p2", pool)
        timings["phase2_s"] = time.perf_counter() - t0

        # ---------------- ensembles ----------------
        # (all features they need were computed in Phase 2 / re-evaluation)
        t0 = time.perf_counter()
        levels = [0.0] + noise_levels(cfg)
        ev.prefetch(cands, ["val", "test"], shots, [0.0], pool)

    def ensemble_report(members: list[int], trace=None) -> dict:
        chroms = [cands[i] for i in members]
        return {
            "members": [int(cand_idx[i]) for i in members],
            "val_acc": ev.ensemble_acc(chroms, "val", shots),
            "test_acc": ev.ensemble_acc(chroms, "test", shots),
            "noisy_test": {str(e): ev.ensemble_acc(chroms, "test_p2", shots, e) for e in levels},
            "noisy_val": {str(e): ev.ensemble_acc(chroms, "val_p2", shots, e) for e in levels},
            "greedy_trace": trace,
        }

    beta = cfg.ensemble.robust_weight
    g_idx, g_trace = greedy_ensemble(cands, ev, cfg, beta, "val_p2")
    ensembles = {"greedy": ensemble_report(g_idx, g_trace)}
    if beta != 0:
        c_idx, c_trace = greedy_ensemble(cands, ev, cfg, 0.0, "val_p2")
        ensembles["greedy_clean"] = ensemble_report(c_idx, c_trace)
    else:
        ensembles["greedy_clean"] = ensembles["greedy"]
    ensembles["topk"] = ensemble_report(topk_ensemble(cands, [val_hi[i] for i in cand_idx], cfg.ensemble.size))
    ensembles["best_single"] = ensemble_report([0])
    timings["ensemble_s"] = time.perf_counter() - t0

    # ---------------- post-hoc readout study (same ensemble, paired) ----------------
    if cfg.phase2.readout_levels:
        t0 = time.perf_counter()
        members = [cands[i] for i in g_idx]
        study: dict[str, dict] = {}
        with make_executor(cfg.workers) as pool:
            for q in sorted({float(x) for x in cfg.phase2.readout_levels}):
                entry = {}
                for mit in (False, True):
                    ro = (q, mit)
                    ev.prefetch(members, ["test_p2"], shots, levels, pool, readout=ro)
                    entry["mitigated" if mit else "raw"] = {
                        str(e): ev.ensemble_acc(members, "test_p2", shots, e, readout=ro) for e in levels}
                study[str(q)] = entry
        ensembles["greedy"]["readout"] = study
        timings["readout_s"] = time.perf_counter() - t0

    # ---------------- Pareto-front quality ----------------
    ref = cfg.search.hv_reference
    pareto = {
        "hv": normalised_hv(F_front, ref),
        "hv_3obj": hypervolume(F_front[:, :3], np.full(3, ref)) / ref**3,
        "size": len(front),
        "val_acc_spread_base_shots": float(F_front[:, 0].max() - F_front[:, 0].min()),
        "val_acc_spread_max_shots": float(max(val_hi) - min(val_hi)),
    }

    # ---------------- baselines ----------------
    baselines = {}
    if run_baseline_methods and cfg.baselines.enabled:
        t0 = time.perf_counter()
        with make_executor(cfg.workers) as pool:
            baselines = run_baselines(cfg, ev, "test_p2", noise_eval=True, log=log,
                                      budget=p1.n_unique_evaluations, executor=pool)
        timings["baselines_s"] = time.perf_counter() - t0

    timings["total_s"] = time.perf_counter() - t_start
    front_rows = []
    p2_by_front = {cand_idx[j]: p2[j] for j in range(len(cands))}
    for i, c in enumerate(front):
        row = {
            "key": chrom_key(c), "circuit": to_json(c), "text": describe(c),
            "depth": depth(c, n), "cnot": cnot_count(c), "gates": gate_count(c),
            "proxy": noise_proxy(c, n, cfg.search.proxy_lambda),
            "objectives": F_front[i].tolist(),
            "val_acc_base": float(1 - F_front[i, 0]), "val_acc": val_hi[i], "test_acc": test_hi[i],
        }
        if i in p2_by_front:
            r = p2_by_front[i]
            row["phase2"] = {
                "val": {str(k): v for k, v in r["val"].items()},
                "test": {str(k): v for k, v in r["test"].items()},
                "robust_val": r["robust_val"], "robust_test": r["robust_test"],
            }
        front_rows.append(row)

    ens = ensembles["greedy"]
    log(f"  MO-GA greedy ensemble: val={ens['val_acc']:.3f} test={ens['test_acc']:.3f} "
        f"noisy(test_p2)={ {k: round(v, 3) for k, v in ens['noisy_test'].items()} }  "
        f"total {timings['total_s']:.1f}s")
    return {
        "name": config_name(cfg),
        "config": cfg.to_dict(),
        "task": task.meta,
        "phase2_samples": {"val": len(iv), "test": len(it)},
        "observables": obs.names(),
        "phase1": {
            "history": p1.history, "generations_run": p1.generations_run,
            "stopped_early": p1.stopped_early, "unique_evaluations": p1.n_unique_evaluations,
        },
        "pareto": pareto,
        "front": front_rows,
        "ensembles": ensembles,
        "baselines": baselines,
        "cache": {"hits": fx.hits, "misses": fx.misses, "hit_rate": fx.hit_rate},
        "timings": timings,
        "environment": _versions(),
    }


def save_result(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.part")
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
