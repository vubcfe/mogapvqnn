# Implementation details

Reference for how each part of the method is implemented. Code locations are listed in the README code map.


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

