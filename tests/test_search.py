import json
import math

import numpy as np
import pytest

from mogapvqnn.circuits import (
    Gene,
    chrom_key,
    cnot_count,
    depth,
    from_json,
    noise_proxy,
    random_chromosome,
    to_json,
)
from mogapvqnn.classifier import majority_vote
from mogapvqnn.config import RunConfig, load_suite
from mogapvqnn.data import amplitude_encode, downsample, synthetic_task
from mogapvqnn.experiment import run_experiment
from mogapvqnn.features import FeatureExtractor
from mogapvqnn.search import crossover, mutate, objective_vector
from mogapvqnn.shadows import make_observables


def test_depth_and_cnot_count():
    c = (Gene("H", (0,)), Gene("H", (1,)), Gene("CNOT", (0, 1)), Gene("RX", (2,), 0.1), Gene("I", (3,)))
    assert depth(c, 4) == 2
    assert cnot_count(c) == 1
    assert math.isclose(noise_proxy(c, 4, 0.05), math.exp(-0.05 * (2 + 2)))


def test_gene_validation():
    with pytest.raises(ValueError):
        Gene("CNOT", (1, 1))
    with pytest.raises(ValueError):
        Gene("RX", (0,))
    with pytest.raises(ValueError):
        Gene("H", (0,), 0.3)


def test_json_roundtrip_preserves_key():
    rng = np.random.default_rng(0)
    c = random_chromosome(4, 8, rng)
    c2 = from_json(json.loads(json.dumps(to_json(c))))
    assert chrom_key(c) == chrom_key(c2)


def test_crossover_and_mutation_keep_length_and_validity():
    rng = np.random.default_rng(1)
    a, b = random_chromosome(4, 8, rng), random_chromosome(4, 8, rng)
    for _ in range(50):
        x, y = crossover(a, b, rng)
        assert len(x) == len(y) == 8
        m = mutate(x, 4, rng, 0.5, 0.3, ("RX", "RZ", "H", "CNOT", "I"))
        assert len(m) == 8
        for g in m:
            assert all(0 <= w < 4 for w in g.wires)
            if g.angle is not None:
                assert -math.pi <= g.angle < math.pi


def test_objectives_normalised_and_proxy_switch():
    cfg = RunConfig(n_qubits=4)
    c = random_chromosome(4, cfg.search.n_genes, np.random.default_rng(2))
    f = objective_vector(c, 0.8, cfg)
    assert f.shape == (4,) and np.all((f >= 0) & (f <= 1))
    cfg.search.use_proxy = False
    assert objective_vector(c, 0.8, cfg).shape == (3,)


def test_majority_vote_tie_break():
    d = [np.array([1.0, -1.0]), np.array([-0.5, -1.0])]
    # sample 0 is a 1-1 tie; per-model standardised decisions are +1 and -2,
    # so the tie is broken towards class 0
    assert majority_vote(d).tolist() == [0, 0]
    assert majority_vote([np.array([1.0]), np.array([2.0]), np.array([-3.0])]).tolist() == [1]


def test_amplitude_encoding_unit_norm_and_zero_image():
    x = amplitude_encode(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(np.linalg.norm(x, axis=1), 1)
    assert np.allclose(x[1], [1 / math.sqrt(2)] * 2)


def test_downsample_rgb_to_gray_shape():
    imgs = np.random.default_rng(0).integers(0, 255, size=(3, 64, 64, 3), dtype=np.uint8)
    assert downsample(imgs, 8).shape == (3, 64)


def test_feature_cache_is_split_aware_and_deterministic():
    task = synthetic_task(4, 60, 30)
    obs = make_observables(4)
    fx = FeatureExtractor(task, obs, seed=0)
    c = random_chromosome(4, 6, np.random.default_rng(3))
    Ff, Ft = fx.features(c, "fit", 50), fx.features(c, "test", 50)
    assert Ff.shape[0] == len(task.y_fit) and Ft.shape[0] == len(task.y_test)
    fx2 = FeatureExtractor(task, obs, seed=0)
    assert np.array_equal(fx2.features(c, "test", 50), Ft)
    assert fx.features(c, "test", 50) is Ft and fx.hits == 1


def test_suite_config_loads_paper_yaml():
    from pathlib import Path

    suite = load_suite(Path(__file__).parents[1] / "configs" / "paper.yaml")
    assert len(suite.configurations) == 7
    conf = next(c for c in suite.configurations if c["n_qubits"] == 8)
    cfg = suite.run_config(conf, seed=1, variant="no_proxy")
    assert cfg.search.population == 15 and cfg.search.generations == 25
    assert cfg.search.use_proxy is False and cfg.seed == 1


def test_end_to_end_smoke_is_reproducible():
    def run():
        cfg = RunConfig(n_qubits=4, workers=3)
        cfg.search.population, cfg.search.generations = 6, 3
        cfg.shadows.base_shots, cfg.shadows.max_shots = 30, 60
        cfg.baselines.enabled = ["random", "pvqnn"]
        return run_experiment(cfg, synthetic_task(4, 80, 40), log=lambda *_: None)

    a, b = run(), run()
    assert a["ensembles"]["greedy"]["test_acc"] == b["ensembles"]["greedy"]["test_acc"]
    assert [r["key"] for r in a["front"]] == [r["key"] for r in b["front"]]
    assert set(a["baselines"]) == {"random", "pvqnn"}
    assert 0 < len(a["ensembles"]["greedy"]["members"]) <= 8
    json.dumps(a)  # serialisable
