"""Tuned classical references on the data of finished `full` runs.

    python scripts/classical_tuned.py [--runs results/runs] [--out results/posthoc_classical] [--jobs 8]

For every results/runs/full/<config>/seed<k>.json the amplitude-encoded task
is rebuilt deterministically (same seed, same fit/val/test split) and three
tuned classical SVMs are fitted (see mogapvqnn.baselines): linear_cv,
poly2_cv and rbf_cv. Hyperparameters are chosen on the validation split and
the model is trained on the fit split, as for the circuit selection of
MO-GA-PVQNN and Grid. No quantum simulation is involved.

Consistency check: the untuned `linear` and `poly2` references are refitted
and must reproduce the stored test accuracies exactly.
Output: results/posthoc_classical/<config>/seed<k>.json (run records are not
modified); report.py merges these records into the tables when present.
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
TUNED = ("linear", "poly2", "rbf")


def process(path: str, out_root: str) -> str:
    from sklearn.svm import SVC

    from mogapvqnn.baselines import tuned_svm
    from mogapvqnn.classifier import accuracy, fit_linear
    from mogapvqnn.config import _from_dict
    from mogapvqnn.data import make_task

    t0 = time.perf_counter()
    run = json.load(open(path))
    cfg = _from_dict(run["config"])
    task = make_task(cfg.dataset, cfg.classes, cfg.n_qubits, cfg.n_train, cfg.n_test, cfg.seed,
                     cfg.val_fraction, cfg.resize)
    X = {s: task.split(s)[0] for s in ("fit", "val", "test")}
    y = {s: task.split(s)[1] for s in ("fit", "val", "test")}

    # ---- consistency with the stored untuned references
    checks = {}
    lin = fit_linear(X["fit"], y["fit"], cfg.classifier.C, cfg.classifier.max_iter)
    poly = SVC(kernel="poly", degree=2, gamma=1.0, coef0=1.0, C=cfg.classifier.C).fit(X["fit"], y["fit"])
    for name, model in (("linear", lin), ("poly2", poly)):
        stored = run["baselines"].get(name, {}).get("test_acc")
        rebuilt = accuracy(model.predict(X["test"]), y["test"])
        checks[name] = {"stored": stored, "rebuilt": rebuilt,
                        "ok": stored is not None and abs(stored - rebuilt) < 1e-12}

    baselines = {}
    for kind in TUNED:
        t1 = time.perf_counter()
        model, params, val = tuned_svm(kind, X["fit"], y["fit"], X["val"], y["val"], cfg.classifier.max_iter)
        baselines[f"{kind}_cv"] = {"val_acc": val, "test_acc": accuracy(model.predict(X["test"]), y["test"]),
                                   "params": params, "classical": True, "seconds": time.perf_counter() - t1}

    out = {"name": run["name"], "seed": cfg.seed, "baselines": baselines, "checks": checks,
           "seconds": time.perf_counter() - t0}
    dst = Path(out_root) / run["name"] / f"seed{cfg.seed}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    bad = [k for k, v in checks.items() if not v["ok"]]
    accs = " ".join(f"{k}={v['test_acc']:.3f}" for k, v in baselines.items())
    return f"{run['name']} seed{cfg.seed}: {accs} ({out['seconds']:.0f}s){'  CHECK FAILED: ' + ','.join(bad) if bad else ''}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "results" / "runs"))
    ap.add_argument("--out", default=str(ROOT / "results" / "posthoc_classical"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", default="", help="comma-separated config names (default: all)")
    args = ap.parse_args()

    paths = sorted(Path(args.runs, "full").glob("*/seed*.json"))
    if args.only:
        keep = set(args.only.split(","))
        paths = [p for p in paths if p.parent.name in keep]
    if not paths:
        raise SystemExit(f"no full runs under {args.runs}")
    print(f"{len(paths)} runs, {args.jobs} jobs", flush=True)
    failed = 0
    with ProcessPoolExecutor(args.jobs) as pool:
        futs = {pool.submit(process, str(p), args.out): p for p in paths}
        for f in as_completed(futs):
            try:
                msg = f.result()
            except Exception as e:  # noqa: BLE001
                failed += 1
                msg = f"{futs[f]}: ERROR {e!r}"
            failed += "CHECK FAILED" in msg
            print(msg, flush=True)
    print("done" if not failed else f"done with {failed} problem(s)")


if __name__ == "__main__":
    main()
