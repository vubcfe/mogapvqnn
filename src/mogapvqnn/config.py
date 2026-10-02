"""Experiment configuration (YAML-backed dataclasses)."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class SearchConfig:
    population: int = 20
    generations: int = 30
    mutation_rate: float = 0.20
    angle_sigma: float = 0.3
    crossover_rate: float = 1.0
    tournament_k: int = 3
    n_genes: int = 8
    allow_identity: bool = True       # no-op gene lets circuits use < n_genes gates
    epsilon_dominance: float = 0.005
    proxy_lambda: float = 0.05
    use_proxy: bool = True            # ablation switch (4 vs 3 objectives)
    # 4th objective: "proxy" = 1 - S_hat (structural, free) or "measured" =
    # 1 - validation accuracy under depolarizing noise `loop_noise`, simulated
    # inside the NSGA-II loop (noise-aware search in the sense of NA-QAS)
    robust_objective: str = "proxy"
    loop_noise: float = 0.01
    early_stop_patience: int = 5
    early_stop_tol: float = 1e-4
    hv_reference: float = 1.1         # per normalised objective


@dataclass
class ShadowConfig:
    estimator: str = "shadow"         # "shadow" | "exact" (ablation)
    observables: str = "local2"       # "local1" | "local2"
    base_shots: int = 200             # Phase 1 fitness
    max_shots: int = 1000             # Pareto front re-evaluation, baselines, Phase 2


@dataclass
class ClassifierConfig:
    C: float = 1.0
    max_iter: int = 20000


@dataclass
class Phase2Config:
    noise_levels: list = field(default_factory=lambda: [0.01, 0.03, 0.05])
    noise_model: str = "local"        # "local" (per-gate) | "global" (end of circuit)
    max_circuits: int = 0             # 0 = whole Pareto front
    val_samples: int = 0              # 0 = full split
    test_samples: int = 0
    readout_error: float = 0.0        # bit-flip probability added to every noisy (eps > 0) evaluation
    readout_mitigation: bool = False  # robust-shadow correction with the known readout error
    # post-hoc readout study of the selected ensemble: for every q listed, the
    # ensemble is re-evaluated at every noise level (incl. 0) with readout
    # error q, unmitigated and robust-shadow mitigated (same circuits, paired)
    readout_levels: list = field(default_factory=list)


@dataclass
class EnsembleConfig:
    size: int = 8
    robust_weight: float = 0.5        # beta in score = (1-beta) clean + beta robust


@dataclass
class BaselineConfig:
    enabled: list = field(default_factory=lambda: [
        "random", "random_search", "manual", "hea", "pvqnn", "grid", "linear", "poly2"])
    random_search_budget: int = 0     # 0 = same number of unique circuit evaluations as the GA run
    manual_layers: int = 1
    hea_layers: int = 1
    grid_depths: list = field(default_factory=lambda: [2, 4, 6, 8])
    grid_cnot_ratios: list = field(default_factory=lambda: [0.0, 0.25, 0.5])


@dataclass
class RunConfig:
    dataset: str = "fashion_mnist"
    classes: list = field(default_factory=lambda: [0, 4])
    n_qubits: int = 4
    n_train: int = 800
    n_test: int = 200
    val_fraction: float = 0.2
    resize: str = "nearest"
    seed: int = 0
    workers: int = 8
    search: SearchConfig = field(default_factory=SearchConfig)
    shadows: ShadowConfig = field(default_factory=ShadowConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)

    def to_dict(self) -> dict:
        return asdict(self)


_SECTIONS = {
    "search": SearchConfig, "shadows": ShadowConfig, "classifier": ClassifierConfig,
    "phase2": Phase2Config, "ensemble": EnsembleConfig, "baselines": BaselineConfig,
}


def _merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


_CHOICES = {
    ("search", "robust_objective"): ("proxy", "measured"),
    ("shadows", "estimator"): ("shadow", "exact"),
    ("shadows", "observables"): ("local1", "local2"),
    ("phase2", "noise_model"): ("local", "global"),
}


def validate(cfg: RunConfig) -> RunConfig:
    """Reject values that would silently select a different method."""
    for (section, key), allowed in _CHOICES.items():
        v = getattr(getattr(cfg, section), key)
        if v not in allowed:
            raise ValueError(f"{section}.{key}={v!r}; expected one of {allowed}")
    if cfg.resize not in ("nearest", "box", "bilinear"):
        raise ValueError(f"resize={cfg.resize!r}")
    if cfg.n_qubits % 2:
        raise ValueError("amplitude encoding of square images needs an even n_qubits")
    if not 0 < cfg.val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    if not 0.0 <= cfg.ensemble.robust_weight <= 1.0:
        raise ValueError("ensemble.robust_weight must be in [0, 1]")
    if not 0.0 <= cfg.phase2.readout_error < 0.5:
        raise ValueError("phase2.readout_error must be in [0, 0.5)")
    if any(not 0.0 < float(q) < 0.5 for q in cfg.phase2.readout_levels):
        raise ValueError("phase2.readout_levels must lie in (0, 0.5)")
    if any(float(e) < 0 for e in cfg.phase2.noise_levels):
        raise ValueError("phase2.noise_levels must be non-negative")
    if cfg.shadows.base_shots < 1 or cfg.shadows.max_shots < 1:
        raise ValueError("shot budgets must be positive")
    if cfg.search.population < 2 or cfg.search.n_genes < 1:
        raise ValueError("search.population >= 2 and search.n_genes >= 1 required")
    return cfg


def _from_dict(d: dict) -> RunConfig:
    known = {f.name for f in fields(RunConfig)}
    unknown = set(d) - known
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    kwargs = {}
    for k, v in d.items():
        if k in _SECTIONS:
            cls = _SECTIONS[k]
            if not isinstance(v, dict):
                raise KeyError(f"config section {k!r} must be a mapping, got {type(v).__name__}")
            bad = set(v) - {f.name for f in fields(cls)}
            if bad:
                raise KeyError(f"unknown keys in {k}: {sorted(bad)}")
            kwargs[k] = cls(**v)
        else:
            kwargs[k] = v
    return validate(RunConfig(**kwargs))


@dataclass
class ExperimentSuite:
    """The full grid of (dataset, qubits) configurations from a YAML file."""

    seeds: list
    defaults: dict
    by_qubits: dict
    configurations: list   # list of dicts: dataset, classes, n_qubits, n_train, n_test
    variants: dict = field(default_factory=lambda: {"full": {}})

    def run_config(self, conf: dict, seed: int, variant: str = "full",
                   overrides: dict | None = None) -> RunConfig:
        if variant not in self.variants:
            raise KeyError(f"unknown variant {variant!r}; known: {sorted(self.variants)}")
        if overrides and "seed" in overrides:
            raise ValueError("pass the seed via the seed argument, not overrides")
        d = RunConfig().to_dict()
        _merge(d, self.defaults)
        _merge(d, self.by_qubits.get(int(conf["n_qubits"])) or {})
        _merge(d, {k: v for k, v in conf.items() if k != "name"})
        _merge(d, self.variants[variant] or {})
        _merge(d, overrides or {})
        d["seed"] = seed
        return _from_dict(d)


def load_suite(path: str | Path) -> ExperimentSuite:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    seeds = raw.get("seeds", [0, 1, 2])
    seeds = [int(seeds)] if isinstance(seeds, (int, str)) else [int(x) for x in (seeds or [])]
    if not seeds:
        raise ValueError(f"{path}: 'seeds' is empty")
    if not raw.get("configurations"):
        raise ValueError(f"{path}: no configurations")
    return ExperimentSuite(
        seeds=list(dict.fromkeys(seeds)),
        defaults=raw.get("defaults") or {},
        by_qubits={int(k): v for k, v in (raw.get("by_qubits") or {}).items()},
        configurations=list(raw["configurations"]),
        variants=raw.get("variants") or {"full": {}},
    )


def config_name(cfg: RunConfig) -> str:
    return f"{cfg.dataset}_{cfg.n_qubits}q"
