"""Command-line interface.

    mogapvqnn run       --config configs/paper.yaml --dataset fashion_mnist --qubits 6 --seed 0
    mogapvqnn suite     configs/paper.yaml --jobs 6 [--variants full no_proxy] [--only fashion_mnist_6q]
    mogapvqnn aggregate [--results results/runs] [--out results/tables]
    mogapvqnn validate  [--qubits 4]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def _result_path(results: Path, variant: str, name: str, seed: int) -> Path:
    return results / "runs" / variant / name / f"seed{seed}.json"


def estimated_cost(cfg, with_baselines: bool = True) -> float:
    """Relative run cost for load balancing (not seconds): shadow sampling is
    ~ samples x shots x n x 2^n per circuit evaluation."""
    n, s = cfg.n_qubits, cfg.search
    per = n * 2**n
    n_tr, n_te = cfg.n_train, cfg.n_test
    phase1 = s.population * (s.generations + 1) * n_tr * cfg.shadows.base_shots * per
    if cfg.search.use_proxy and cfg.search.robust_objective == "measured":
        phase1 *= 1.0 + 1.5 * cfg.val_fraction          # noisy validation evaluation per individual
    front = s.population * (n_tr + n_te) * cfg.shadows.max_shots * per
    phase2 = s.population * len(cfg.phase2.noise_levels) * 1.6 * (cfg.val_fraction * n_tr + n_te) \
        * cfg.shadows.max_shots * per
    if cfg.shadows.estimator == "exact":                 # density matrices for the noisy part
        phase1, front = phase1 * 0.05, front * 0.05
        phase2 *= 2**n / (1.6 * cfg.shadows.max_shots) * 4
    baselines = 2.5 * (phase1 + front + phase2) if (with_baselines and cfg.baselines.enabled) else 0.0
    return phase1 + front + phase2 + baselines


def shard_jobs(jobs: list, costs: list[float], shares: list[float], shard: int) -> list:
    """Deterministic weighted longest-processing-time partition: every machine
    computes the same assignment and keeps its own part."""
    load = [0.0] * len(shares)
    mine = []
    for i in sorted(range(len(jobs)), key=lambda i: (-costs[i], i)):
        k = min(range(len(shares)), key=lambda k: ((load[k] + costs[i]) / shares[k], k))
        load[k] += costs[i]
        if k == shard:
            mine.append(jobs[i])
    total = sum(load)
    print("shard loads: " + ", ".join(f"#{k}={100 * l / total:.1f}%" for k, l in enumerate(load)), flush=True)
    return mine


def _find_conf(suite, dataset: str, qubits: int) -> dict:
    for c in suite.configurations:
        if c["dataset"] == dataset and int(c["n_qubits"]) == qubits:
            return c
    raise SystemExit(f"{dataset} {qubits}q is not in the suite configuration")


def cmd_run(a) -> None:
    from .config import config_name, load_suite
    from .experiment import run_experiment, save_result

    suite = load_suite(a.config)
    overrides = json.loads(a.override) if a.override else None
    cfg = suite.run_config(_find_conf(suite, a.dataset, a.qubits), a.seed, a.variant, overrides)
    if a.workers:
        cfg.workers = a.workers
    out = Path(a.out) if a.out else _result_path(Path(a.results), a.variant, config_name(cfg), a.seed)
    if out.exists() and not a.force:
        print(f"exists, skipping: {out}")
        return
    result = run_experiment(cfg, run_baseline_methods=(a.variant == "full") and not a.no_baselines)
    result["variant"] = a.variant
    save_result(result, out)
    print(f"saved {out}")


def cmd_suite(a) -> None:
    from .config import load_suite

    from .data import prepare_datasets

    suite = load_suite(a.config)
    variants = list(dict.fromkeys(a.variants or suite.variants))
    unknown = set(variants) - set(suite.variants)
    if unknown:
        raise SystemExit(f"unknown variant(s) {sorted(unknown)}; known: {sorted(suite.variants)}")
    seeds = list(dict.fromkeys(a.seeds if a.seeds else suite.seeds))
    if a.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    names = {f"{c['dataset']}_{c['n_qubits']}q" for c in suite.configurations}
    if a.only and set(a.only) - names:
        raise SystemExit(f"--only: unknown configuration(s) {sorted(set(a.only) - names)}")
    results = Path(a.results)
    jobs = []
    for variant in variants:
        for conf in suite.configurations:
            name = f"{conf['dataset']}_{conf['n_qubits']}q"
            if a.only and name not in a.only:
                continue
            for seed in seeds:
                jobs.append((variant, conf, seed, name))
    if a.job_list:
        # explicit assignment: lines "variant config_name seed" ('#' comments allowed)
        wanted = set()
        for line in Path(a.job_list).read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                v, n, sd = line.split()
                wanted.add((v, n, int(sd)))
        known = {(v, n, sd) for v, _, sd, n in jobs}
        if wanted - known:
            raise SystemExit(f"--job-list: unknown entries {sorted(wanted - known)[:5]}")
        jobs = [j for j in jobs if (j[0], j[3], j[2]) in wanted]
    if a.shard is not None:
        # partition the *full* job list (independent of what is already done
        # locally) so that all machines agree on the assignment
        shares = [float(x) for x in a.shares.split(",")] if a.shares else [1.0] * a.num_shards
        if len(shares) != a.num_shards or not 0 <= a.shard < a.num_shards:
            raise SystemExit("--shard K needs 0 <= K < --num-shards and one --shares entry per shard")
        costs = [estimated_cost(suite.run_config(c, s, v), v == "full") for v, c, s, _ in jobs]
        jobs = shard_jobs(jobs, costs, shares, a.shard)
    jobs = [j for j in jobs if a.force or not _result_path(results, j[0], j[3], j[2]).exists()]
    workers = a.workers or int(suite.defaults.get("workers", 4))
    cores = os.cpu_count() or 1
    if a.jobs * workers > cores:
        print(f"warning: {a.jobs} jobs x {workers} workers = {a.jobs * workers} threads > {cores} cores", flush=True)
    # download / cache datasets once, so parallel runs never race on the files
    prepare_datasets([c for c in suite.configurations
                      if not a.only or f"{c['dataset']}_{c['n_qubits']}q" in a.only])
    # longest jobs first, so the tail of the schedule is short jobs
    jobs.sort(key=lambda j: -estimated_cost(suite.run_config(j[1], j[2], j[0]), j[0] == "full"))
    print(f"{len(jobs)} runs to do, {a.jobs} in parallel")
    log_dir = results / "_local" / "logs"       # per-run logs (not version-controlled)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **THREAD_ENV, PYTHONUNBUFFERED="1")   # live progress in results/_local/logs

    def launch(job):
        variant, conf, seed, name = job
        cmd = [sys.executable, "-m", "mogapvqnn", "run", "--config", str(a.config), "--dataset", conf["dataset"],
               "--qubits", str(conf["n_qubits"]), "--seed", str(seed), "--variant", variant,
               "--results", str(results), "--out", str(_result_path(results, variant, name, seed))]
        if a.workers:
            cmd += ["--workers", str(a.workers)]
        if a.force:
            cmd.append("--force")
        t0 = time.time()
        with open(log_dir / f"{variant}_{name}_seed{seed}.log", "w") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        return job, rc, time.time() - t0

    failures = 0
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futs = [pool.submit(launch, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            (variant, _, seed, name), rc, dt = f.result()
            failures += rc != 0
            print(f"[{i}/{len(jobs)}] {variant:15s} {name:18s} seed={seed} "
                  f"{'ok' if rc == 0 else f'FAILED rc={rc}'} ({dt / 60:.1f} min)", flush=True)
    if failures:
        raise SystemExit(f"{failures} run(s) failed; see {log_dir}")


def cmd_aggregate(a) -> None:
    from .report import aggregate

    files = aggregate(Path(a.results), Path(a.out))
    print(f"wrote {len(files)} files to {a.out}")
    print((Path(a.out) / "SUMMARY.md").read_text())


def cmd_validate(a) -> None:
    from .validation import validate

    r = validate(a.qubits, a.circuits, a.noise, a.shots)
    print(json.dumps(r, indent=2))
    if not r["passed"]:
        raise SystemExit(1)


def main(argv=None) -> None:
    # single-threaded BLAS (set before numpy is imported by any sub-command):
    # parallelism comes from the worker threads / suite jobs instead
    for k, v in THREAD_ENV.items():
        os.environ.setdefault(k, v)
    p = argparse.ArgumentParser(prog="mogapvqnn")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one (dataset, qubits, seed, variant)")
    r.add_argument("--config", default=str(ROOT / "configs" / "paper.yaml"))
    r.add_argument("--dataset", required=True)
    r.add_argument("--qubits", type=int, required=True)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--variant", default="full")
    r.add_argument("--workers", type=int, default=0)
    r.add_argument("--override", help="JSON dict merged into the run config")
    r.add_argument("--results", default=str(RESULTS))
    r.add_argument("--out")
    r.add_argument("--no-baselines", action="store_true")
    r.add_argument("--force", action="store_true")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("suite", help="run all configurations x seeds x variants")
    s.add_argument("config")
    s.add_argument("--jobs", type=int, default=4)
    s.add_argument("--workers", type=int, default=0)
    s.add_argument("--variants", nargs="*")
    s.add_argument("--seeds", type=int, nargs="+")
    s.add_argument("--only", nargs="*", help="config names such as fashion_mnist_6q")
    s.add_argument("--results", default=str(RESULTS))
    s.add_argument("--force", action="store_true")
    s.add_argument("--job-list", help="file with lines 'variant config_name seed' to run (subset of the suite)")
    s.add_argument("--shard", type=int, help="index of this machine (0-based) when splitting the suite")
    s.add_argument("--num-shards", type=int, default=1)
    s.add_argument("--shares", help="comma-separated relative machine speeds, e.g. 1,1.6")
    s.set_defaults(func=cmd_suite)

    g = sub.add_parser("aggregate", help="build tables and figure data")
    g.add_argument("--results", default=str(RESULTS / "runs"))
    g.add_argument("--out", default=str(RESULTS / "tables"))
    g.set_defaults(func=cmd_aggregate)

    v = sub.add_parser("validate", help="cross-check the simulator against PennyLane")
    v.add_argument("--qubits", type=int, default=4)
    v.add_argument("--circuits", type=int, default=5)
    v.add_argument("--noise", type=float, default=0.05)
    v.add_argument("--shots", type=int, default=4000)
    v.set_defaults(func=cmd_validate)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
