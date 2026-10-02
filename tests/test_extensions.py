"""Tests for the literature-driven extensions: readout noise and its
robust-shadow mitigation, measured noise-aware objective, equal-budget random
search, classical references and Holm correction."""

import numpy as np
import pytest

from mogapvqnn.baselines import linear_baseline, poly2_baseline, random_search_baseline
from mogapvqnn.circuits import random_chromosome
from mogapvqnn.config import RunConfig
from mogapvqnn.data import synthetic_task
from mogapvqnn.experiment import run_experiment
from mogapvqnn.features import FeatureExtractor
from mogapvqnn.report import holm
from mogapvqnn.search import CircuitEvaluator, objective_vector
from mogapvqnn.shadows import exact_features, make_observables, shadow_features


def _states(n, k, rng):
    s = np.abs(rng.normal(size=(k, 2**n)))
    return s / np.linalg.norm(s, axis=1, keepdims=True)


@pytest.mark.parametrize("mitigate", [False, True])
def test_readout_error_shadows_match_exact(mitigate):
    rng = np.random.default_rng(10)
    n, q, T = 3, 0.05, 60000
    c = random_chromosome(n, 8, rng)
    psi = _states(n, 2, rng)
    obs = make_observables(n)
    ex = exact_features(c, psi, obs, 0.02, "local", readout_error=q, mitigate_readout=mitigate)
    sh = shadow_features(c, psi, obs, T, rng, 0.02, "local", readout_error=q, mitigate_readout=mitigate)
    f = (1 - 2 * q) if mitigate else 1.0
    sd = np.array([3.0 ** (len(t) / 2) / f ** len(t) for t in obs.terms]) / np.sqrt(T)
    assert np.all(np.abs(sh - ex) < 5 * sd)


def test_readout_mitigation_removes_bias():
    rng = np.random.default_rng(11)
    n = 3
    c = random_chromosome(n, 8, rng)
    psi = _states(n, 2, rng)
    obs = make_observables(n)
    true = exact_features(c, psi, obs, 0.02, "local")
    biased = exact_features(c, psi, obs, 0.02, "local", readout_error=0.1)
    fixed = exact_features(c, psi, obs, 0.02, "local", readout_error=0.1, mitigate_readout=True)
    assert np.allclose(fixed, true)
    w = np.array([len(t) for t in obs.terms])
    assert np.allclose(biased, true * 0.8 ** w)


def test_readout_only_affects_noisy_evaluations():
    task = synthetic_task(3, 40, 20)
    obs = make_observables(3)
    c = random_chromosome(3, 5, np.random.default_rng(0))
    a = FeatureExtractor(task, obs, 0).features(c, "val", 40)
    b = FeatureExtractor(task, obs, 0, readout_error=0.2).features(c, "val", 40)
    assert np.array_equal(a, b)


def test_measured_objective_requires_and_uses_robust_acc():
    cfg = RunConfig(n_qubits=4)
    cfg.search.robust_objective = "measured"
    c = random_chromosome(4, cfg.search.n_genes, np.random.default_rng(1))
    with pytest.raises(ValueError):
        objective_vector(c, 0.8, cfg)
    f = objective_vector(c, 0.8, cfg, robust_acc=0.7)
    assert f.shape == (4,) and np.isclose(f[3], 0.3)


def _small_cfg():
    cfg = RunConfig(n_qubits=4, workers=2)
    cfg.search.population, cfg.search.generations = 6, 2
    cfg.shadows.base_shots, cfg.shadows.max_shots = 30, 60
    cfg.phase2.noise_levels = [0.02]
    return cfg


def _evaluator(cfg, task):
    obs = make_observables(cfg.n_qubits)
    extra = {"val_p2": task.X_val, "test_p2": task.X_test}
    fx = FeatureExtractor(task, obs, cfg.seed, extra_splits=extra)
    return CircuitEvaluator(task, fx, cfg, {"val_p2": task.y_val, "test_p2": task.y_test})


def test_random_search_uses_requested_budget():
    cfg = _small_cfg()
    task = synthetic_task(4, 60, 30)
    r = random_search_baseline(cfg, _evaluator(cfg, task), "test_p2", True, budget=17)
    assert r["budget"] == 17 and 1 <= r["front_size"] <= 17
    assert 0 < len(r["circuits"]) <= cfg.ensemble.size


def test_classical_references_fit_synthetic_task():
    cfg = _small_cfg()
    task = synthetic_task(4, 120, 60)
    ev = _evaluator(cfg, task)
    for fn in (linear_baseline, poly2_baseline):
        r = fn(cfg, ev, "test_p2", False)
        assert r["classical"] and r["test_acc"] > 0.7   # the synthetic task is linearly structured


def test_end_to_end_with_all_extensions():
    cfg = _small_cfg()
    cfg.search.robust_objective = "measured"
    cfg.phase2.readout_error, cfg.phase2.readout_mitigation = 0.02, True
    r = run_experiment(cfg, synthetic_task(4, 80, 40), log=lambda *_: None)
    assert {"random_search", "linear", "poly2"} <= set(r["baselines"])
    assert r["baselines"]["random_search"]["budget"] == r["phase1"]["unique_evaluations"]
    assert len(r["front"][0]["objectives"]) == 4


def test_holm_adjustment():
    adj = holm([0.01, 0.04, 0.03, float("nan")])
    assert np.allclose(adj[:3], [0.03, 0.06, 0.06]) and np.isnan(adj[3])
