# MO-GA-PVQNN

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23104673.svg)](https://doi.org/10.5281/zenodo.23104673)

Multi-objective genetic architecture search and benchmarks for
**post-variational quantum neural networks** (PVQNN) using classical shadows.
This is the code, data and analysis for the article

> C. V. Bui, V.-N. Nguyen, H.-A.-V. Nguyen, *What Limits Post-Variational
> Quantum Neural Networks? Expressivity Bounds, Shot Noise and Noise-Aware
> Architecture Search under Amplitude Encoding*, submitted to Physical Review A
> (2026).

**Main findings.**
- With amplitude encoding, every PVQNN feature is a quadratic form of the
  input, so the models lie in the span of a degree-2 kernel
  (`tests/test_theory.py` checks this numerically).
- Untuned classical SVMs on the same inputs beat every quantum model.
  NSGA-II does no better than equal-budget random search.
- Shot noise and readout error, not circuit architecture, limit accuracy.
  Exact expectations recover up to 25 points; robust-shadow post-processing
  undoes readout errors.
- A measured-noise objective beats a structural noise proxy.

**What is in this repository**

| Path | Contents |
|---|---|
| `src/mogapvqnn/` | package: simulator, shadows, NSGA-II, baselines, experiment driver, reports |
| `configs/paper.yaml` | the exact experimental protocol |
| `results/runs/` | all 182 run records: 4 variants × 7 configurations; 10 seeds for `full` and `measured_noise`, 3 for `no_proxy` and `exact_features` |
| `results/posthoc/` | readout and shot sweeps on the 70 selected ensembles |
| `results/posthoc_classical/` | tuned linear, Poly-2 and RBF SVM references on the same 70 data splits |
| `results/tables/`, `results/figures/` | every table (LaTeX/Markdown) and figure of the article |
| `scripts/` | `reproduce.sh` (whole study), `posthoc.py` (readout/shot sweeps), `classical_tuned.py` (tuned classical SVMs), `make_figures.py`; `distributed/finish.sh` for a suite split across two machines |
| `docs/` | implementation details, deviations from the earlier manuscript, literature, code review |
| `tests/` | 56 unit, validation and theory tests |

The tables and figures can be regenerated from the stored records without
re-running any experiment:

```bash
.venv/bin/python -m mogapvqnn aggregate
.venv/bin/python scripts/make_figures.py
```

The pipeline:

1. **Amplitude encoding.** Each image is downsampled to √2ⁿ × √2ⁿ pixels and
   l2-normalised into an n-qubit state.
2. **Phase 1 (noiseless NSGA-II).** Searches fixed-length gate chromosomes over
   {RX, RZ, H, CNOT}. It minimises [1 − validation accuracy, depth, CNOT count,
   1 − Ŝ], where Ŝ is a complexity-based noise-stability proxy. Fitness uses
   classical-shadow features with a linear SVM read-out (the post-variational
   model).
3. **Front re-evaluation.** Pareto-front circuits are re-evaluated with a
   larger shot budget.
4. **Phase 2 (noise).** Every front circuit is evaluated under per-gate
   depolarizing noise at several strengths, on the same samples as the clean
   reference.
5. **Greedy ensemble.** Selection maximises (1 − β)·clean + β·robust
   validation accuracy of the majority vote.
6. **Baselines.** Random, equal-budget random search, Manual, HEA, PVQNN
   (ansatz expansion) and Grid search share the identical feature/read-out
   pipeline. Two classical references (linear SVM, degree-2 polynomial-kernel
   SVM on the same amplitude-encoded inputs) test whether the quantum
   pipeline adds anything over a classical model.

Design choices driven by recent literature (noise-aware NSGA-II QAS,
classical simulability of shadow models, equal-budget random search,
readout-robust shadows) are documented in
[docs/LITERATURE.md](docs/LITERATURE.md).

The original implementation used for the first manuscript draft was lost.
This repository is a from-scratch reimplementation that follows the
manuscript and **corrects several issues in it**. See
[docs/DEVIATIONS.md](docs/DEVIATIONS.md). All
numbers in the paper must be regenerated with this code.

---

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

Tested stack: Python 3.12, NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.4.2,
PennyLane 0.38.0 (needs `autoray==0.6.12`; newer autoray breaks PL 0.38).
PennyLane is only needed for `mogapvqnn validate` and `tests/test_pennylane.py`.
The experiments run on the built-in NumPy simulator, which is cross-checked
against PennyLane.

## Data

| Dataset | Classes | Source |
|---|---|---|
| Fashion-MNIST | 0 (T-shirt/top) vs 4 (Coat) | downloaded automatically (IDX files) |
| MNIST | 3 vs 8 | downloaded automatically (IDX files) |
| EuroSAT RGB | AnnualCrop vs Forest | local folder or Zenodo download |

Files go to `data/raw/` (override with `MOGAPVQNN_DATA=/path`). For EuroSAT,
point `EUROSAT_ROOT` at a folder containing `AnnualCrop/` and `Forest/`, or
symlink it:

```bash
ln -s /path/to/EuroSAT data/raw/EuroSAT_RGB
```

EuroSAT has no official split. A fixed pool split (80/20, seed `20240601`)
separates the train and test pools, and the run seed then draws the
class-balanced samples from those pools.

## Usage

```bash
# everything: tests, PennyLane cross-check, 7 configs x 3 seeds x 4 variants, tables, figures
bash scripts/reproduce.sh
JOBS=12 SEEDS="0 1 2 3 4 5 6 7 8 9" bash scripts/reproduce.sh   # 10 seeds (recommended)

# single run
.venv/bin/python -m mogapvqnn run --dataset fashion_mnist --qubits 6 --seed 0
.venv/bin/python -m mogapvqnn run --dataset fashion_mnist --qubits 6 --seed 0 --variant no_proxy
.venv/bin/python -m mogapvqnn run --dataset mnist --qubits 4 --seed 0 \
    --override '{"phase2": {"noise_model": "global"}}'

# a subset of the suite
.venv/bin/python -m mogapvqnn suite configs/paper.yaml --jobs 6 --only fashion_mnist_6q --variants full no_proxy

# tables (LaTeX + Markdown) and figures from whatever runs are finished
.venv/bin/python -m mogapvqnn aggregate
.venv/bin/python scripts/make_figures.py

# pipeline check on a tiny configuration
.venv/bin/python -m mogapvqnn suite configs/smoke.yaml --jobs 2 --results results/smoke
.venv/bin/python -m mogapvqnn aggregate --results results/smoke/runs --out results/smoke/tables

# tests and simulator validation
.venv/bin/python -m pytest -q
.venv/bin/python -m mogapvqnn validate --qubits 4
```

`suite` launches one subprocess per (variant, configuration, seed) with
`OMP_NUM_THREADS=1`. Each run uses `workers` threads (default 4) for
population evaluation, so `--jobs × workers` should not exceed the core
count. Runs are resumable: an existing `results/runs/<variant>/<config>/seedK.json`
is skipped unless you pass `--force`. Per-run logs go to `results/_local/logs/` (not version-controlled).

Run time has not been benchmarked for this reimplementation. The runtime
table is produced from the measured timings stored in every run record.

## Outputs

`results/runs/<variant>/<dataset>_<n>q/seed<k>.json` contains:

| Key | Contents |
|---|---|
| `config` | the fully resolved run configuration |
| `task` | sample counts, class names, image side |
| `phase1.history` | per generation: hypervolume, front size, best/mean validation accuracy, unique evaluations, cache hit rate, elapsed time |
| `pareto` | HV (in the objective space of the run), HV in the common 3-objective space (`hv_3obj`), front size, accuracy spread |
| `front[]` | every Pareto circuit: gates, depth, CNOTs, Ŝ, objectives, validation/test accuracy, Phase-2 accuracy at each ε |
| `ensembles` | `greedy` (clean + robust), `greedy_clean` (β = 0), `topk`, `best_single`: members, validation/test accuracy, noisy accuracies |
| `baselines` | per baseline: validation/test accuracy, noisy test accuracy, circuits, time; Grid also records the full sweep |
| `timings` | data, Phase 1, re-evaluation, Phase 2, ensembles, baselines, total (seconds) |
| `environment` | library versions, machine |

`scripts/posthoc.py` writes the readout and shot sweeps of every selected
ensemble to `results/posthoc/<config>/seed<k>.json`.
`scripts/classical_tuned.py` rebuilds the data split of every `full` run and
fits tuned linear, Poly-2 and RBF SVMs (hyperparameters selected on the
validation split, model trained on the fit split) to
`results/posthoc_classical/<config>/seed<k>.json`; `report.py` and
`make_figures.py` merge these records when present.

`mogapvqnn aggregate` writes every table twice, as
`results/tables/latex/<table>.tex` and `results/tables/markdown/<table>.md`, plus
a combined `results/tables/SUMMARY.md`:

| Table | Contents |
|---|---|
| `table_main` | test accuracy, all methods |
| `table_stats` | paired t-tests vs. best quantum and best classical baseline, Holm-adjusted |
| `table_shots` | same circuits refitted with T shadow rounds or exact expectations |
| `table_readout_sweep` | readout error q, raw vs robust-shadow mitigated |
| `table_noise` | ensemble accuracy at each ε on identical samples; δ for MO-GA, random search, HEA, Grid |
| `table_readout` | in-run readout study at q = 0.02 and the smallest ε |
| `table_ablation` | proxy off, measured noise, exact expectations; paired tests, HV₃, Phase-1 time |
| `table_ensemble` | greedy vs. greedy-clean vs. top-K vs. best single circuit |
| `table_pareto` | HV, front size, spread, generations, early stops |
| `table_runtime` | Phase 1 / Phase 2 / baselines / total, cache hit rate |

Figure data for pgfplots (`convergence_<config>.csv`, `accuracy_by_config.csv`)
go to `results/tables/data/`. `scripts/make_figures.py` writes the PDF figures
to `results/figures/`.

```
results/
├── runs/<variant>/<config>/seed<k>.json   run records (version-controlled)
├── posthoc/<config>/seed<k>.json          post-hoc sweeps (version-controlled)
├── posthoc_classical/<config>/seed<k>.json  tuned classical SVMs (version-controlled)
├── tables/{latex,markdown,data}/          generated tables and figure data
├── figures/                               generated PDF figures
└── _local/                                logs and scheduling files (git-ignored)
```

## Code map

| Manuscript | Module |
|---|---|
| Sec. 3.1 problem formulation, Eq. (1)–(2) objectives and proxy | `search.objective_vector`, `circuits.noise_proxy` |
| Sec. 3.2 amplitude encoding, Eq. (3) | `data.downsample`, `data.amplitude_encode` |
| Sec. 3.3 classical shadows, Eq. (4) | `shadows.shadow_features` |
| Sec. 3.4 chromosome, operators, NSGA-II, ε-dominance, early stop, cache, threads | `circuits`, `search.crossover/mutate/run_phase1`, `nsga2`, `features` |
| Sec. 3.5 Phase 2 noise evaluation | `simulator` (noise models), `search.phase2_evaluate` |
| Sec. 3.6 greedy ensemble, Eq. (7) | `search.greedy_ensemble`, `classifier.majority_vote` |
| Sec. 4.2 baselines | `baselines` |
| Sec. 4.4–4.10 tables and figures | `report`, `scripts/make_figures.py` |
| Table 1 hyperparameters | `configs/paper.yaml` |

```
src/mogapvqnn/
  circuits.py    genes, chromosomes, depth/CNOT/proxy, baseline ansätze, PennyLane export
  simulator.py   batched statevector + density-matrix simulator, depolarizing noise
  shadows.py     observables, classical-shadow and exact feature estimators
  features.py    deterministic, split-aware, thread-safe feature cache
  classifier.py  StandardScaler + LinearSVC read-out, majority vote
  nsga2.py       dominance, sorting, crowding, selection, exact hypervolume
  search.py      evaluator, operators, Phase 1, Phase 2, ensembles
  baselines.py   Random, Manual, HEA, PVQNN, Grid
  experiment.py  one full run -> JSON
  report.py      tables and statistics
  validation.py  cross-check against PennyLane
  cli.py         command-line interface
```

## Documentation

| File | Contents |
|---|---|
| [docs/METHODS.md](docs/METHODS.md) | implementation details: chromosome, operators, dominance, features, determinism, noise, variants, baselines, statistics |
| [docs/DEVIATIONS.md](docs/DEVIATIONS.md) | where this implementation intentionally differs from the earlier manuscript |
| [docs/LITERATURE.md](docs/LITERATURE.md) | related work (2023–2026) and the design changes it motivated |
| [docs/CODE_REVIEW.md](docs/CODE_REVIEW.md) | pre-run code review: findings, fixes, rejected findings |

## Tests

```bash
.venv/bin/python -m pytest -q
```

The tests cover:
- the qubit-ordering convention, unitarity, and density matrix vs. statevector
- validity of the noisy density matrix
- unbiasedness of clean and noisy shadows against the exact density matrix
  (both noise models)
- dominance and sorting against brute force
- crowding distance
- hypervolume: known values and a 4-D Monte-Carlo check
- genetic operators and gene validation
- the split-aware cache
- loading the paper configuration
- end-to-end determinism
- agreement with PennyLane
- readout noise and its mitigation against exact values
- the measured noise objective
- the random-search budget
- the classical references
- Holm correction
- building every table and figure from synthetic run records

## License and citation

Released under the [MIT License](LICENSE). Citation metadata is in
[CITATION.cff](CITATION.cff); please cite the article above. The archived
software is on Zenodo: version 1.0.0, doi:10.5281/zenodo.23104674 (all
versions: doi:10.5281/zenodo.23104673).
