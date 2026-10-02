"""Regression tests for issues found in the code review."""

import threading
from pathlib import Path

import numpy as np
import pytest

from mogapvqnn.circuits import Gene, chrom_key, random_chromosome
from mogapvqnn.config import RunConfig, load_suite, validate
from mogapvqnn.data import _stratified_holdout, synthetic_task
from mogapvqnn.features import FeatureExtractor
from mogapvqnn.nsga2 import nondominated_sort
from mogapvqnn.search import SerialExecutor, make_executor, noise_levels
from mogapvqnn.shadows import make_observables
from mogapvqnn.simulator import X, Y, Z, _dm_apply_1q, _dm_depolarize, run_statevector, states_to_density

PAPER = Path(__file__).parents[1] / "configs" / "paper.yaml"


def test_identity_wire_does_not_change_key():
    a = (Gene("H", (0,)), Gene("I", (0,)))
    b = (Gene("H", (0,)), Gene("I", (3,)))
    assert chrom_key(a) == chrom_key(b)
    assert chrom_key((Gene("RX", (0,), -0.0),)) == chrom_key((Gene("RX", (0,), 0.0),))


def test_partial_trace_depolarizing_equals_kraus_form():
    rng = np.random.default_rng(0)
    n = 3
    s = rng.normal(size=(2, 8)) + 1j * rng.normal(size=(2, 8))
    s /= np.linalg.norm(s, axis=1, keepdims=True)
    rho = states_to_density(s, n)
    for w in range(n):
        kraus = 0.7 * rho + 0.1 * sum(_dm_apply_1q(rho, P, w, n) for P in (X, Y, Z))
        assert np.allclose(_dm_depolarize(rho, 0.3, w, n), kraus)


def test_statevector_refuses_global_noise():
    psi = np.zeros((1, 2, 2), complex)
    psi[0, 0, 0] = 1
    with pytest.raises(NotImplementedError):
        run_statevector((Gene("H", (0,)),), psi, noise=0.1, noise_model="global")


def test_concurrent_requests_compute_once_and_return_read_only():
    task = synthetic_task(4, 60, 30)
    fx = FeatureExtractor(task, make_observables(4), seed=0)
    c = random_chromosome(4, 6, np.random.default_rng(1))
    out = []
    threads = [threading.Thread(target=lambda: out.append(fx.features(c, "fit", 200))) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert fx.misses == 1 and fx.hits == 7
    assert all(o is out[0] for o in out)
    with pytest.raises(ValueError):
        out[0][0, 0] = 1.0


def test_feature_extractor_input_validation():
    task = synthetic_task(4, 40, 20)
    obs = make_observables(4)
    with pytest.raises(ValueError):
        FeatureExtractor(task, obs, 0, estimator="shadows")
    with pytest.raises(ValueError):
        FeatureExtractor(task, obs, 0, extra_splits={"val": task.X_val})
    fx = FeatureExtractor(task, obs, 0)
    with pytest.raises(ValueError):
        fx.features(random_chromosome(4, 3, np.random.default_rng(0)), "val", 10, noise=-1e-9)


def test_config_rejects_typos_and_unknown_variants():
    suite = load_suite(PAPER)
    with pytest.raises(KeyError):
        suite.run_config(suite.configurations[0], 0, variant="no-proxy")
    cfg = RunConfig()
    cfg.shadows.estimator = "Shadow"
    with pytest.raises(ValueError):
        validate(cfg)
    cfg = RunConfig()
    cfg.search.robust_objective = "measurd"
    with pytest.raises(ValueError):
        validate(cfg)


def test_noise_levels_are_deduplicated_and_positive():
    cfg = RunConfig()
    cfg.phase2.noise_levels = [0.0, 0.05, 0.01, 0.05]
    assert noise_levels(cfg) == [0.01, 0.05]


def test_holdout_keeps_both_parts_non_empty():
    y = np.array([0, 0, 1, 1])
    fit, val = _stratified_holdout(y, 0.2, np.random.default_rng(0))
    assert len(fit) == 2 and len(val) == 2
    with pytest.raises(ValueError):
        _stratified_holdout(np.array([0, 1, 1]), 0.2, np.random.default_rng(0))


def test_nan_objectives_rank_last():
    F = np.array([[0.1, 0.1], [np.nan, 0.2], [0.2, 0.2]])   # NaN -> +inf: dominated by row 0
    fronts = nondominated_sort(F)
    assert 0 in fronts[0] and 1 not in fronts[0]


def test_serial_executor_is_an_executor():
    with make_executor(1) as ex:
        assert isinstance(ex, SerialExecutor)
        assert list(ex.map(lambda x: x * 2, [1, 2])) == [2, 4]
        assert ex.submit(lambda: 3).result() == 3


def test_mitigated_features_are_the_raw_record_rescaled():
    task = synthetic_task(4, 40, 20)
    obs = make_observables(4)
    fx = FeatureExtractor(task, obs, seed=0)
    c = random_chromosome(4, 6, np.random.default_rng(2))
    q = 0.05
    raw = fx.features(c, "val", 300, 0.02, readout=(q, False))
    mit = fx.features(c, "val", 300, 0.02, readout=(q, True))
    w = np.array([len(t) for t in obs.terms])
    assert np.allclose(mit, raw / (1 - 2 * q) ** w)
    # an explicit readout setting differs from the readout-free features
    assert not np.allclose(raw, fx.features(c, "val", 300, 0.02))
    # and is applied even at zero gate noise
    assert not np.allclose(fx.features(c, "val", 300, 0.0, readout=(q, False)), fx.features(c, "val", 300))


def test_full_run_contains_paired_readout_study():
    from mogapvqnn.experiment import run_experiment

    cfg = RunConfig(n_qubits=4, workers=2)
    cfg.search.population, cfg.search.generations = 6, 2
    cfg.shadows.base_shots, cfg.shadows.max_shots = 30, 60
    cfg.phase2.noise_levels = [0.02]
    cfg.phase2.readout_levels = [0.05]
    cfg.baselines.enabled = ["random"]
    r = run_experiment(cfg, synthetic_task(4, 80, 40), log=lambda *_: None)
    ro = r["ensembles"]["greedy"]["readout"]["0.05"]
    assert set(ro) == {"raw", "mitigated"}
    assert set(ro["raw"]) == set(r["ensembles"]["greedy"]["noisy_test"]) == {"0.0", "0.02"}
