# MO-GA-PVQNN

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
| `results/tables/`, `results/figures/` | every table (LaTeX/Markdown) and figure of the article |
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
[Differences from the earlier manuscript](#differences-from-the-earlier-qip-manuscript). All
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
is skipped unless you pass `--force`. Per-run logs go to `results/logs/`.

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

`mogapvqnn aggregate` writes the following to `results/tables/`, each as
`.tex` and `.md`, plus a combined `SUMMARY.md`:

| Table | Contents |
|---|---|
| `table_main` | test accuracy, all methods |
| `table_stats` | paired t-test vs. best baseline, CI of the mean and of the paired difference |
| `table_pareto` | HV, front size, spread, generations, early stops |
| `table_noise` | ensemble accuracy at each ε on identical samples; δ for MO-GA, random search, HEA, Grid |
| `table_readout` | the selected ensemble without readout error, with 2 % readout error, and with robust-shadow mitigation, at ε = 0 and at the smallest ε |
| `table_ensemble` | greedy vs. greedy-clean vs. top-K vs. best single circuit |
| `table_ablation` | proxy off, and exact expectations instead of shadows; paired tests, HV₃, Phase-1 time |
| `table_runtime` | Phase 1 / Phase 2 / baselines / total, cache hit rate |

It also writes `convergence_<config>.csv` and `accuracy_by_config.csv` for pgfplots.

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

## Implementation details

### Chromosome
- A chromosome has `n_genes = 8` genes.
- A gene is `(gate, wires, angle)` with gate ∈ {RX, RZ, H, CNOT}. Rotation
  angles are drawn from [−π, π).
- Mutation can also produce the no-op gene `I`, so circuits can shrink below
  8 gates. The manuscript says "up to d gates".
- Depth uses as-soon-as-possible layer scheduling.
- Objectives are normalised to [0, 1]: depth/8, CNOT/8, 1 − Ŝ, with
  Ŝ = exp(−0.05·(depth + 2·CNOT)).

### Operators
- Selection is a 3-way tournament on (rank, crowding distance).
- Crossover is single point with p_c = 1.
- Mutation: each gene's gate type is resampled with probability p_m.
  Otherwise, each rotation angle is perturbed by N(0, 0.3) with probability
  p_m and wrapped to [−π, π).
- Survivor selection is elitist (μ + λ). Duplicate circuits are removed
  before truncation.

### Dominance and hypervolume
- ε-dominance: a dominates b if a is no worse everywhere and better by more
  than ε = 0.005 in at least one normalised objective.
- Hypervolume is exact (recursive slicing), with reference point 1.1 per
  objective, normalised to [0, 1].
- Early stopping triggers when HV does not improve by more than 1e-4 for 5
  consecutive generations.

### Features and read-out
- Observables are X_i, Y_i, Z_i on every qubit plus nearest-neighbour
  X_iX_{i+1} and Z_iZ_{i+1}: 3n + 2(n − 1) features (`observables: local1`
  drops the 2-local terms).
- Phase 1 uses T = 200 shadow rounds. The front, the baselines and Phase 2
  use T = 1000.
- Each circuit gets one `StandardScaler + LinearSVC(C=1)` fitted on the fit
  split (80 % of the training sample). The validation split (20 %) drives
  fitness and ensemble selection. The test split is used only for reporting.

### Determinism
- Every feature evaluation draws from
  `default_rng([seed, blake2b(circuit), split, shots, ε])`.
- Results are therefore independent of thread scheduling and caching.
- Two runs with the same seed give identical results (tested).

### Noise model (default `local`)
- After every gate, a `DepolarizingChannel(ε)` acts on each qubit the gate
  touches, for ε ∈ {0.01, 0.03, 0.05}. The encoded input state is prepared
  noiselessly.
- Sampling uses quantum trajectories (random Pauli insertions). These give
  exactly the density-matrix measurement statistics at statevector memory
  cost, which makes 8 qubits feasible.
- `noise_model: global` instead applies ρ → (1 − ε)ρ + εI/2ⁿ once at the end
  (the formula printed in the manuscript).
- Clean and noisy accuracies are measured with the same clean-trained
  read-out, on the same samples, with the same shot budget.

### Variants (`configs/paper.yaml`)
| Variant | What changes | Purpose |
|---|---|---|
| `full` | nothing (structural proxy Ŝ as 4th objective); runs all baselines | main results |
| `no_proxy` | 3 objectives | does Ŝ help? |
| `measured_noise` | 4th objective = 1 − validation accuracy under ε = 0.01, simulated in the loop | free proxy vs NA-QAS-style measured noise |
| `exact_features` | exact expectations instead of shadows | cost of shot noise |

Readout noise is not a separate variant. Every `full` run re-evaluates its
selected ensemble with `phase2.readout_levels` (2 % bit-flips) at every
gate-noise level:
- raw, and
- with the robust-shadow correction, which is the *same* measurement record
  with each snapshot factor divided by 1 − 2q.

This gives a paired comparison without repeating the search
(`table_readout`).

### Baselines (all K = 8, same features, read-out and T)
| Baseline | Circuits |
|---|---|
| Random | K random chromosomes (no search) |
| Random search | as many unique random chromosomes as the GA evaluated in the same run, then the identical Pareto filter + greedy clean+robust ensemble |
| Manual | RY layer + linear CNOT chain, random angles |
| HEA | RY layer + all-to-all CNOT, random angles |
| PVQNN | the post-variational *ansatz expansion*: identity plus ±π/2 RY shifts on each qubit, all features concatenated into a single linear model; no search |
| Grid | sweep over gene count {2, 4, 6, 8} × CNOT fraction {0, 0.25, 0.5}; K random circuits per cell; best cell on validation |
| Linear SVM *(classical)* | StandardScaler + LinearSVC on the amplitude-encoded input vector |
| Poly-2 SVM *(classical)* | SVC with kernel (x·x′ + 1)²; spans every function a linear read-out on Pauli features of amplitude-encoded states can express |

### Statistics
- Sample std (ddof = 1).
- Student-t CIs with n − 1 degrees of freedom.
- Paired t-test over matched seeds (same data split), against the best
  quantum baseline and against the best classical reference.
- Holm–Bonferroni adjustment across the configurations of each table.
- The ablation compares hypervolume in the common accuracy–depth–CNOT space,
  since HV values from 3- and 4-objective spaces are not comparable.

## Differences from the earlier (QIP) manuscript

These are intentional and must be reflected in the revised text.

1. **Classical-shadow estimator (Sec. 3.3, Eq. 4).**
   - The manuscript draws bases from {H, RX(π/2)} (two bases) and writes
     ρ̂ = 3ⁿ/T Σ ⊗ U†|b⟩⟨b|U. Both are wrong for the Pauli shadow protocol.
   - The code uses three bases (X, Y, Z via H, HS†, I) and the unbiased
     estimator ρ̂ = ⊗ᵢ (3 Uᵢ†|bᵢ⟩⟨bᵢ|Uᵢ − I).
   - For a weight-k Pauli the snapshot value is ∏ 3·sᵢ·[basisᵢ = Pᵢ].
   - The sample complexity is O(3ᵏ log M / ε²), not "O(log M) independent of
     the observable count".
2. **Phase-2 evaluation (Tables 2 and 6).** The manuscript compares clean
   accuracy on the full test set with noisy accuracy on 3–10 samples using
   10–20 shots. That makes δ (e.g. −0.44 pp at 8 qubits) meaningless. The
   code evaluates clean and noisy accuracy on the same samples, by default
   the full validation and test splits, with T = 1000.
3. **Noise channel.** The manuscript writes a global channel but reports
   25–29 pp drops at ε = 0.01, which a global channel cannot produce. The
   code defaults to per-gate local depolarizing and offers `global` as an
   option. State which one the paper uses.
4. **Fashion-MNIST classes.** Label 4 is *Coat*, not *Dress* (that is 3).
   The original experiments and the ICMCSI-2026 version of this work used
   0 vs 4, so the code uses `classes: [0, 4]`. The manuscript must say
   "T-shirt/top vs Coat".
5. **Phase-2 coverage.** The manuscript evaluates noise on the top 1–4
   circuits of 4 configurations. The code evaluates the whole front on all
   7 configurations, and the baselines too, so robustness can be compared.
6. **Greedy objective.** Eq. 7 uses clean accuracy only, while Algorithm 1
   says "clean + robust". The code implements (1 − β)·clean + β·robust with
   β = 0.5 and also reports β = 0 and top-K.
7. **Statistics.**
   - The manuscript's std values are population std (ddof = 0) while its CIs
     use ddof = 1. The code uses ddof = 1 everywhere.
   - The ablation's HV comparison (0.809 vs 0.816) mixed 3- and 4-objective
     spaces. The code reports `hv_3obj` for both arms.
8. **Fitness and read-out splits.**
   - The manuscript says the SVM is "trained on D_val", which would make the
     fitness training accuracy. The code trains on the fit split and scores
     on the validation split.
   - Individual fitness is single-circuit validation accuracy, not "ensemble
     accuracy".
9. **New ablations.** Exact expectations vs. shadows (`exact_features`), and
   greedy vs. top-K / best single. The manuscript lists both as untested
   limitations.
10. **Simulator.** The experiments use a NumPy simulator validated against
    PennyLane 0.38 rather than PennyLane devices directly. Validated:
    - statevector against `default.qubit`: error 0
    - noisy expectations against `default.mixed` + `DepolarizingChannel`:
      error < 1e-12
    - shadows statistically against `qml.shadow_expval`

    The text should say "implemented in NumPy and validated against
    PennyLane 0.38".
11. **"PVQNN" means post-variational QNN** (Huang & Rebentrost,
    arXiv:2307.10560). Define it in the paper and cite that work for the
    PVQNN baseline instead of Beer et al. (2020).
12. **New baselines and variants not in the manuscript.** Equal-budget random
    search, the classical Linear and Poly-2 SVM references, measured in-loop
    noise, and readout noise with and without robust-shadow mitigation. The
    novelty statement must be narrowed in view of Li et al. 2026 (NSGA-II
    noise-aware QAS). The two prior conference versions of this work must
    be cited. See [docs/LITERATURE.md](docs/LITERATURE.md).

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
[CITATION.cff](CITATION.cff); please cite the article above.
