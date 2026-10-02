"""Tables and figures build from (tiny, synthetic) run records."""

import importlib.util
from pathlib import Path

import pytest

from mogapvqnn.config import RunConfig
from mogapvqnn.data import synthetic_task
from mogapvqnn.experiment import run_experiment, save_result
from mogapvqnn.report import aggregate


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    for variant, use_proxy in (("full", True), ("no_proxy", False)):
        for seed in (0, 1):
            cfg = RunConfig(dataset="fashion_mnist", n_qubits=4, workers=2, seed=seed)
            cfg.search.population, cfg.search.generations = 6, 2
            cfg.search.use_proxy = use_proxy
            cfg.shadows.base_shots, cfg.shadows.max_shots = 30, 60
            cfg.phase2.noise_levels = [0.02]
            cfg.phase2.readout_levels = [0.05]
            r = run_experiment(cfg, synthetic_task(4, 80, 40, seed), log=lambda *_: None,
                               run_baseline_methods=(variant == "full"))
            save_result(r, root / variant / r["name"] / f"seed{seed}.json")
    return root


def test_aggregate_writes_all_tables(runs_dir, tmp_path):
    files = aggregate(runs_dir, tmp_path)
    for t in ("table_main", "table_stats", "table_pareto", "table_noise", "table_readout", "table_ensemble",
              "table_ablation", "table_runtime"):
        assert f"{t}.tex" in files and f"{t}.md" in files
    assert "SUMMARY.md" in files and "convergence_fashion_mnist_4q.csv" in files
    assert "Poly-2 SVM" in (tmp_path / "latex" / "table_main.tex").read_text()


def test_make_figures(runs_dir, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    path = Path(__file__).parents[1] / "scripts" / "make_figures.py"
    spec = importlib.util.spec_from_file_location("make_figures", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr("sys.argv", ["make_figures", "--runs", str(runs_dir), "--out", str(tmp_path),
                                     "--pareto-run", "fashion_mnist_4q"])
    mod.main()
    for f in ("fig_accuracy", "fig_scaling", "fig_convergence", "fig_pareto", "fig_noise"):
        assert (tmp_path / f"{f}.pdf").exists()
