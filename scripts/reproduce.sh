#!/usr/bin/env bash
# Reproduce every number, table and figure of the manuscript.
#
#   bash scripts/reproduce.sh            # 6 runs in parallel
#   JOBS=12 SEEDS="0 1 2 3 4 5 6 7 8 9" bash scripts/reproduce.sh
#
# Runs are resumable: finished runs (results/runs/<variant>/<config>/seedK.json)
# are skipped, so the script can be re-launched after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
JOBS=${JOBS:-6}
SEEDS=${SEEDS:-"0 1 2"}
CONFIG=${CONFIG:-configs/paper.yaml}

echo "== 1/5 unit tests"
"$PY" -m pytest -q

echo "== 2/5 simulator cross-check against PennyLane"
"$PY" -m mogapvqnn validate --qubits 4

echo "== 3/5 experiments (full model + baselines, then ablations)"
# shellcheck disable=SC2086
"$PY" -m mogapvqnn suite "$CONFIG" --jobs "$JOBS" --seeds $SEEDS --variants full
# shellcheck disable=SC2086
"$PY" -m mogapvqnn suite "$CONFIG" --jobs "$JOBS" --seeds $SEEDS --variants no_proxy measured_noise exact_features

echo "== post-hoc readout and shot sweeps on the selected ensembles"
"$PY" scripts/posthoc.py --jobs "$JOBS"

echo "== 4/5 tables"
"$PY" -m mogapvqnn aggregate

echo "== 5/5 figures"
"$PY" scripts/make_figures.py

echo "done: results/tables/SUMMARY.md, results/tables/latex/*.tex, results/figures/*.pdf"
