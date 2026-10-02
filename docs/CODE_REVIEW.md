# Pre-run code review (2026-10-02)

## Method
- An external review by DeepSeek (`deepseek-reasoner`, plus `deepseek-chat`
  where the reasoner ran out of tokens) covered every source file, split into
  ~200-line segments: 20 review requests in total.
- Every finding was then checked against the code and, where it was
  testable, against a test or a direct computation.

| Verdict | Meaning |
|---|---|
| **Fixed** | real problem, now fixed and covered by a regression test |
| **Hardened** | not reachable today; guarded anyway |
| **Rejected** | false positive (reason given) |

The test suite grew from 41 to 51 tests (`tests/test_review_fixes.py`), all
passing; `pyflakes` is clean.

## Correctness / reproducibility

| Area | Finding | Verdict |
|---|---|---|
| circuits | `chrom_key` distinguished no-op `I` genes on different wires and `-0.0` vs `0.0` angles → duplicate individuals, lower cache-hit rate | **Fixed** |
| data | one RNG stream sampled train, test and the fit/val split, so the test sample depended on `n_train` | **Fixed**: independent streams |
| data | parallel suite jobs could download the same IDX file or write the same EuroSAT cache concurrently (shared `.part` name) | **Fixed**: unique temp files + `os.replace`, corrupt files re-fetched, and `suite` prepares all datasets before launching jobs |
| data | `_stratified_holdout` could leave an empty validation split (round-half-even) | **Fixed**: both parts non-empty, round half up |
| data | EuroSAT cache ignored `data_dir` and the dataset root; extracted-zip folder name was hard-coded | **Fixed** |
| classifier | `warnings.catch_warnings()` is not thread-safe | **Fixed**: module-level filter |
| config | unknown variant names silently ran `full`; typos in `estimator` / `robust_objective` / `noise_model` silently selected another method | **Fixed**: validation raises |
| config | scalar / empty `seeds`, `null` sections, `seed` in overrides | **Fixed** |
| cli | `suite` decided "already done" from one path and `run` wrote another | **Fixed**: `suite` passes `--out` |
| cli | duplicate seeds, `--seeds` with no values, `--jobs 0`, unknown `--only`; BLAS threads unpinned for `run` | **Fixed** |
| features | concurrent requests for one key computed it repeatedly; cached arrays were writable; negative noise / unknown split / estimator typos not rejected; cache key used a float while the RNG used the quantised noise | **Fixed** |
| experiment | Phase-2 subset silently returned fewer samples than requested; `greedy_clean` missing when β = 0; result file not fsynced | **Fixed** |
| search | `0.0` listed in `noise_levels` was double-counted in robust accuracy | **Fixed**: `noise_levels()` helper used everywhere |
| search | `make_executor` returned two different types | **Fixed**: `SerialExecutor` |
| simulator | `run_statevector` silently ignored `noise_model="global"`; no input shape check | **Fixed** |
| nsga2 | NaN objectives would have been non-dominated | **Hardened**: NaN → +∞ |
| report / figures | noise tables assumed the literal key `"0.0"` and one noise grid per table; missing `hv_3obj` / baselines crashed | **Hardened** |
| paper.yaml | `readout` variant relied on the default for `readout_mitigation` | **Hardened**: explicit `false` |

## Performance (n = 8, T = 1000)

| Area | Change |
|---|---|
| simulator | density-matrix depolarizing via partial trace, (1 − 4p/3)ρ + (2p/3)Tr_w ρ ⊗ I: one pass instead of six matrix applications |
| simulator | trajectory Pauli insertion in place (no full-batch copy per gate) |
| shadows | exact expectations from 1-/2-qubit reduced density matrices computed once per support, instead of one full-size Pauli-applied copy per observable |
| baselines | Grid (96 circuits), Random, Manual, HEA, PVQNN and random-search ensembles now compute their feature matrices in parallel (`CircuitEvaluator.prefetch`); results identical |

## Second pass (own review)

| Area | Change |
|---|---|
| simulator | `apply_matrix` rewritten with a broadcast `matmul` on a (B, L, 2, R) view instead of `moveaxis` + `einsum`. Shadow sampling was ~90 % of the run time: now ~1.7× faster (8 q, B = 256, T = 200: 1.21 s → 0.67 s clean, 1.96 s → 1.29 s noisy) |
| compute layout | The `readout` / `readout_mitigated` variants repeated Phase 1, re-evaluation and Phase 2 of all 21 runs only to change the Phase-2 measurement (about ⅓ of the whole suite). Readout is now a post-hoc study inside every `full` run, on the selected ensemble |
| features | Mitigated readout features are derived from the cached raw record (÷ (1 − 2q)^k) instead of being re-simulated: exactly paired and half the cost |
| baselines | Random search used the Pareto front of *all* ~600 sampled circuits, giving it more selection candidates than the GA (front ≤ P) and a much costlier Phase-2 step. It now keeps P survivors by (rank, crowding) like NSGA-II, then uses their front |

## Rejected findings

| Finding | Why it is wrong |
|---|---|
| "Global depolarizing in `shadow_features` is bit-flip readout noise" | Replacing *all* bits of a shot by uniform bits with probability p is exactly measuring (1 − p)ρ + pI/2ⁿ; `test_noisy_trajectory_shadows_match_density_matrix[global]` checks it against the density matrix |
| "Hypervolume slicing direction is inverted" | For minimisation a slab at height z is dominated by points with z′ ≤ z. On DeepSeek's own example the code gives 0.425 = exact (inclusion–exclusion) = Monte Carlo 0.4249; a 4-D Monte-Carlo test also exists |
| "Validation tolerances 1e-10 are unrealistic / ops re-queued / StatePrep endianness" | Measured errors are 0.0 (statevector) and ~3e-13 (noisy expectations); the test compares directly with PennyLane |
| "`greedy_ensemble` contract / robust stage disabled in random search" | It returns indices and uses β with `val_p2` |
| "`linear` baseline lacks StandardScaler" | `fit_linear` builds StandardScaler + LinearSVC |
| "majority-vote tie-break should mean-centre" | The sign of an SVM decision value is meaningful; centring would change the vote |
| "front not deduplicated / empty front / process-pool cache loss" | Phase 1 deduplicates; the population is never empty; only threads are used |
| "unbounded cache ≈ 3 GB" | Realistic size is < 150 MB |
| "8-qubit budget smaller than 4/6 q" | Intentional: Table 1 of the manuscript |
| "`measured_noise` gives 5 objectives" | `robust_objective: measured` *replaces* the proxy |
