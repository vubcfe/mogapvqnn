"""Post-hoc studies on the ensembles selected in finished `full` runs.

    python scripts/posthoc.py [--runs results/runs] [--out results/posthoc] [--jobs 6] [--workers 4]

For every results/runs/full/<config>/seed<k>.json the task, feature
extractor and selected greedy ensemble are rebuilt deterministically (same
seeds, same circuits), and two studies are run on the *same* circuits:

* readout sweep: readout bit-flip q in --readout-levels, raw and
  robust-shadow mitigated, at every Phase-2 gate-noise level, test_p2 split.
* shot sweep: models trained and evaluated with T in --shots shadow rounds,
  plus exact expectation values (T = infinity), clean test split. Isolates
  the effect of shot noise from the choice of circuits.

Consistency check: at T = shadows.max_shots the rebuilt ensemble must
reproduce the stored clean test accuracy, and at the stored readout level
the stored readout numbers. Mismatches are reported and stored.
Output: results/posthoc/<config>/seed<k>.json (the run records are not modified).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]


def rebuild(run: dict):
    from mogapvqnn.circuits import from_json
    from mogapvqnn.config import _from_dict
    from mogapvqnn.data import make_task
    from mogapvqnn.experiment import _balanced_subset

    cfg = _from_dict(run["config"])
    task = make_task(cfg.dataset, cfg.classes, cfg.n_qubits, cfg.n_train, cfg.n_test, cfg.seed,
                     cfg.val_fraction, cfg.resize)
    iv = _balanced_subset(task.y_val, cfg.phase2.val_samples, cfg.seed)
    it = _balanced_subset(task.y_test, cfg.phase2.test_samples, cfg.seed + 1)
    extra = {"val_p2": task.X_val[iv], "test_p2": task.X_test[it]}
    labels = {"val_p2": task.y_val[iv], "test_p2": task.y_test[it]}
    members = [from_json(run["front"][i]["circuit"]) for i in run["ensembles"]["greedy"]["members"]]
    return cfg, task, extra, labels, members


def evaluator(cfg, task, extra, labels, estimator):
    from mogapvqnn.features import FeatureExtractor
    from mogapvqnn.search import CircuitEvaluator
    from mogapvqnn.shadows import make_observables

    obs = make_observables(cfg.n_qubits, cfg.shadows.observables)
    fx = FeatureExtractor(task, obs, cfg.seed, estimator, cfg.phase2.noise_model, extra,
                          cfg.phase2.readout_error, cfg.phase2.readout_mitigation)
    return CircuitEvaluator(task, fx, cfg, labels)


def process(path: str, out_root: str, readout_levels: list[float], shots: list[int], workers: int) -> str:
    from mogapvqnn.search import make_executor, noise_levels

    t0 = time.perf_counter()
    run = json.load(open(path))
    cfg, task, extra, labels, members = rebuild(run)
    levels = [0.0] + noise_levels(cfg)
    T = cfg.shadows.max_shots
    ev = evaluator(cfg, task, extra, labels, "shadow")
    out = {"name": run["name"], "seed": cfg.seed, "members": len(members), "checks": {}}
    with make_executor(workers) as pool:
        # ---- consistency with the stored run
        ev.prefetch(members, ["test"], T, [0.0], pool)
        rebuilt = ev.ensemble_acc(members, "test", T)
        stored = run["ensembles"]["greedy"]["test_acc"]
        out["checks"]["test_acc"] = {"stored": stored, "rebuilt": rebuilt, "match": abs(rebuilt - stored) < 1e-12}

        # ---- readout sweep (same circuits, same test_p2 samples)
        sweep = {}
        for q in readout_levels:
            entry = {}
            for mit in (False, True):
                ro = (q, mit)
                ev.prefetch(members, ["test_p2"], T, levels, pool, readout=ro)
                entry["mitigated" if mit else "raw"] = {
                    str(e): ev.ensemble_acc(members, "test_p2", T, e, readout=ro) for e in levels}
            sweep[str(q)] = entry
        out["readout_sweep"] = sweep
        ev.prefetch(members, ["test_p2"], T, [0.0], pool)
        out["clean_test_p2"] = ev.ensemble_acc(members, "test_p2", T)
        stored_ro = run["ensembles"]["greedy"].get("readout", {})
        for q, entry in stored_ro.items():
            if q in sweep:
                out["checks"][f"readout_{q}"] = {"match": entry == sweep[q]}

        # ---- shot sweep (same circuits; models refitted at every T)
        shot = {}
        for t in shots:
            ev.prefetch(members, ["test"], t, [0.0], pool)
            shot[str(t)] = {
                "ensemble": ev.ensemble_acc(members, "test", t),
                "members_mean": float(sum(ev.acc(c, "test", t) for c in members) / len(members)),
            }
        ex = evaluator(cfg, task, extra, labels, "exact")
        ex.prefetch(members, ["test"], T, [0.0], pool)
        shot["exact"] = {
            "ensemble": ex.ensemble_acc(members, "test", T),
            "members_mean": float(sum(ex.acc(c, "test", T) for c in members) / len(members)),
        }
        out["shots_sweep"] = shot
    out["seconds"] = time.perf_counter() - t0
    dest = Path(out_root) / run["name"] / Path(path).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, dest)
    bad = [k for k, v in out["checks"].items() if not v.get("match", True)]
    return f"{run['name']} seed={cfg.seed}: {out['seconds'] / 60:.1f} min" + (f"  CHECK MISMATCH {bad}" if bad else "  checks ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=str(ROOT / "results" / "runs"))
    p.add_argument("--out", default=str(ROOT / "results" / "posthoc"))
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--readout-levels", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    p.add_argument("--shots", type=int, nargs="+", default=[200, 1000, 5000])
    p.add_argument("--only", nargs="*", help="config names, e.g. fashion_mnist_8q")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    paths = sorted(str(x) for x in Path(a.runs, "full").glob("*/seed*.json"))
    if a.only:
        paths = [x for x in paths if Path(x).parent.name in a.only]
    todo = [x for x in paths if a.force or not (Path(a.out) / Path(x).parent.name / Path(x).name).exists()]
    # longest first: more qubits
    todo.sort(key=lambda x: -int(Path(x).parent.name.rsplit("_", 1)[1][:-1]))
    print(f"{len(todo)} runs to post-process ({len(paths) - len(todo)} done)", flush=True)
    failures = 0
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        futs = {pool.submit(process, x, a.out, a.readout_levels, a.shots, a.workers): x for x in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                print(f"[{i}/{len(todo)}] {f.result()}", flush=True)
            except Exception as e:  # keep going, report at the end
                failures += 1
                print(f"[{i}/{len(todo)}] FAILED {futs[f]}: {e!r}", flush=True)
    if failures:
        raise SystemExit(f"{failures} post-processing job(s) failed")


if __name__ == "__main__":
    main()
