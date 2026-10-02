"""Thread-safe, deterministic feature extraction with caching.

Cache key = (chromosome key, split, shots, estimator, noise level). The split
is part of the key so features of one split can never be served for another
(the original implementation keyed on the chromosome only, which returned
training features for test data).

Readout error: the extractor-level default (``phase2.readout_error``) is
applied to every *noisy* evaluation (noise > 0) on top of the gate noise;
an explicit ``readout=(q, mitigate)`` argument applies to that call only
(used for the post-hoc readout study of the selected ensemble).

Every evaluation draws its randomness from a generator seeded by
(run seed, chromosome hash, split, shots, noise), so results do not depend
on thread scheduling or on cache hits. Concurrent requests for the same key
are computed once (the other threads wait), and cached arrays are read-only
so no consumer can corrupt them.
"""

from __future__ import annotations

import threading

import numpy as np

from .circuits import Chromosome, chrom_key, stable_hash32
from .data import BinaryTask
from .shadows import ObservableSet, exact_features, shadow_features

_SPLIT_CODE = {"fit": 1, "val": 2, "test": 3, "val_p2": 4, "test_p2": 5}
ESTIMATORS = ("shadow", "exact")


class FeatureExtractor:
    def __init__(
        self,
        task: BinaryTask,
        obs: ObservableSet,
        seed: int,
        estimator: str = "shadow",
        noise_model: str = "local",
        extra_splits: dict[str, np.ndarray] | None = None,
        readout_error: float = 0.0,
        readout_mitigation: bool = False,
    ):
        if estimator not in ESTIMATORS:
            raise ValueError(f"unknown estimator {estimator!r}; expected one of {ESTIMATORS}")
        if not 0.0 <= readout_error < 0.5:
            raise ValueError("readout_error must be in [0, 0.5)")
        self.obs = obs
        self._weights = np.array([len(t) for t in obs.terms], dtype=float)
        self.seed = seed
        self.estimator = estimator
        self.noise_model = noise_model
        self.readout_error = readout_error
        self.readout_mitigation = readout_mitigation
        self.states = {"fit": task.X_fit, "val": task.X_val, "test": task.X_test}
        extra = dict(extra_splits or {})
        clash = set(extra) & set(self.states)
        if clash:
            raise ValueError(f"extra_splits would override {sorted(clash)}")
        self.states.update(extra)
        self._cache: dict[tuple, np.ndarray] = {}
        self._inflight: dict[tuple, threading.Event] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _rng(self, key: str, split: str, shots: int, noise_q: int, readout_q: int) -> np.random.Generator:
        seed = [self.seed, stable_hash32(key), _SPLIT_CODE.get(split, stable_hash32(split)), shots, noise_q]
        if readout_q:      # readout-free evaluations keep their historical streams
            seed.append(readout_q)
        return np.random.default_rng(seed)

    def _readout(self, noise: float, readout: tuple[float, bool] | None) -> tuple[float, bool]:
        """Effective (readout error, mitigation) of one evaluation. ``None``
        uses the extractor default, which only applies to noisy (noise > 0)
        evaluations; an explicit pair applies as given (also at noise 0)."""
        if readout is None:
            q, mit = (self.readout_error if noise > 0 else 0.0), self.readout_mitigation
        else:
            q, mit = float(readout[0]), bool(readout[1])
        if not 0.0 <= q < 0.5:
            raise ValueError("readout error must be in [0, 0.5)")
        return q, (mit if q > 0 else False)

    def features(self, chrom: Chromosome, split: str, shots: int, noise: float = 0.0,
                 readout: tuple[float, bool] | None = None) -> np.ndarray:
        if noise < 0:
            raise ValueError(f"noise must be non-negative, got {noise}")
        if split not in self.states:
            raise KeyError(f"unknown split {split!r}")
        key = chrom_key(chrom)
        noise_q = int(round(noise * 1e6))   # one representation for cache key and RNG seed
        noise = noise_q / 1e6
        q, mit = self._readout(noise, readout)
        readout_q = int(round(q * 1e6))
        ck = (key, split, shots if self.estimator == "shadow" else 0, self.estimator, noise_q, readout_q, mit)
        while True:
            with self._lock:
                hit = self._cache.get(ck)
                if hit is not None:
                    self.hits += 1
                    return hit
                pending = self._inflight.get(ck)
                if pending is None:
                    pending = self._inflight[ck] = threading.Event()
                    self.misses += 1
                    break
            pending.wait()          # another thread is computing this key
        try:
            F = self._compute(chrom, key, split, shots, noise, noise_q, readout_q / 1e6, readout_q, mit)
            F.setflags(write=False)
            with self._lock:
                self._cache[ck] = F
        finally:
            with self._lock:
                self._inflight.pop(ck, None)
            pending.set()
        return F

    def _compute(self, chrom, key, split, shots, noise, noise_q, q, readout_q, mit) -> np.ndarray:
        if mit:
            # Readout mitigation is post-processing of the *same* measurement
            # record: every single-qubit snapshot factor is divided by
            # f = 1 - 2q, i.e. a weight-k feature by f^k. Deriving it from the
            # cached raw features makes raw vs mitigated an exactly paired
            # comparison and halves the simulation cost.
            raw = self.features(chrom, split, shots, noise, (q, False))
            return raw / (1.0 - 2.0 * q) ** self._weights[None, :]
        states = self.states[split]
        if self.estimator == "exact":
            return exact_features(chrom, states, self.obs, noise, self.noise_model, readout_error=q)
        return shadow_features(
            chrom, states, self.obs, shots, self._rng(key, split, shots, noise_q, readout_q),
            noise=noise, noise_model=self.noise_model, readout_error=q,
        )

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
