# Related work (2023–2026) and how it shaped this code

The literature search was done on 2026-10-01 (Consensus, papers from 2023 on).
Each section ends with the code change it motivated.

## 1. Our own prior versions of this work

| Ref | Paper | What it already contains |
|---|---|---|
| P1 | C. V. Bui et al., *Genetic Algorithm-Based Ansatz Optimization for Post-Variational Quantum Neural Networks*, ICMCSI 2026, doi:10.1109/icmcsi67283.2026.11412580 | GA, classical shadows, PVQNN; Fashion-MNIST **0 vs 4** at 8 qubits |
| P2 | C. V. Bui et al., *Evolutionary Ansatz Selection for Post-Variational Quantum Neural Networks*, CIoTSC 2025, doi:10.1109/ciotsc67482.2025.11413040 | GA ensemble selection for PV-QNN; Fashion-MNIST and MNIST (2 vs 6) |

**Consequences**
- The QIP manuscript must cite P1 and P2 and say exactly what is new:
  multiple objectives, noise awareness, Phase 2, statistics and the new
  datasets. Otherwise it risks a duplicate-publication flag.
- P1 uses Fashion-MNIST labels 0 vs 4, i.e. T-shirt/top vs **Coat**.
  The manuscript's "Dress" is a naming error.

→ Code: `classes: [0, 4]` everywhere (`configs/*.yaml`, `RunConfig`).

## 2. Noise-aware and multi-objective QAS: the novelty claim must be narrowed

| Ref | Paper | Relevance |
|---|---|---|
| L1 | C.-L. Li et al., *Noise-Aware Quantum Architecture Search Based on NSGA-II*, arXiv:2601.10965 (2026) | NSGA-II QAS with the noise model **inside** the search; classification tasks |
| L2 | K. Maleki et al., *QNAS*, arXiv:2604.07013 (2026) | NSGA-II over error, runtime and cutting cost; MNIST/Fashion-MNIST |
| L3 | Q. Ma et al., *Continuous evolution for efficient QAS*, EPJ Quantum Technol. 11 (2024), doi:10.1140/epjqt/s40507-024-00265-7 | NSGA-II with a SuperCircuit |
| L4 | H. Wang, *Genetic Transformer-Assisted QNNs*, arXiv:2506.09205 (2025) | NSGA-II: accuracy vs gate count |
| L5 | Y. J. Patel et al., *Curriculum RL for QAS under hardware errors*, arXiv:2402.03500 (2024) | noisy-environment QAS |
| L6 | S. Mousavi et al., *Training-free QAS under realistic noise*, Entropy 28 (2026), doi:10.3390/e28030330 | noise-free proxy (expressibility) that predicts noisy loss; same idea as our Ŝ |
| L7 | Y. Huang et al., *Adaptive diversity-based QAS*, Phys. Rev. Research 6 (2024), doi:10.1103/physrevresearch.6.033033 | noise- and mapping-aware QAS |
| L8 | J.-J. Su et al., *Topology-driven QAS*, Sci. China Inf. Sci. (2025), doi:10.1007/s11432-024-4486-x | noisy and noiseless validation |
| L9 | J. Chen et al., *Qubit-wise architecture search*, arXiv:2403.04268 (2024) | MNIST / Fashion-MNIST QAS benchmark |

**Consequences**
- "No prior work applies NSGA-II to QAS with a noise-aware objective" is
  false (L1). Our contribution has to be stated more narrowly: a
  **post-variational** model (no quantum training), shadow-based fitness,
  and a comparison of a free structural proxy against measured in-loop
  noise.
- L6 supports the idea that a noise-free proxy can rank noisy performance.
  The fair test is to compare it head-to-head with measured noise.

→ Code: `search.robust_objective: proxy | measured` with `loop_noise`. The
`measured_noise` variant runs NA-QAS-style search, where noisy validation
accuracy is the 4th objective. `table_ablation` compares proxy, measured and
no-proxy on matched seeds.

## 3. Post-variational QNNs and classical simulability

| Ref | Paper | Relevance |
|---|---|---|
| Q1 | P.-W. Huang, P. Rebentrost, *Post-variational quantum neural networks*, arXiv:2307.10560 (2023) | defines PVQNN (observable construction, ansatz expansion) |
| Q2 | P. Bermejo et al., *QCNNs are effectively classically simulable*, PRX Quantum (2024), doi:10.1103/8qt9-72ts | models that only use low-bodyness observables can be simulated with Pauli shadows; benchmarks are often "locally easy" |
| Q3 | S. Jerbi et al., *Shadows of quantum machine learning*, Nat. Commun. 15 (2024), doi:10.1038/s41467-024-49877-8 | shadow models have restricted learning capacity |
| Q4 | K. Yogaraj et al., *Post-variational classical quantum transfer learning*, Sci. Rep. (2025), doi:10.1038/s41598-025-08887-2 | PV applications |
| Q5 | M. Vandromme et al., *PVQNN on a hybrid HPC-QC system*, SC25-W (2025), doi:10.1145/3731599.3767540 | PV on real hardware |

**Consequences**
- With amplitude encoding, every feature ⟨ψ_x|U†PU|ψ_x⟩ is a quadratic form
  xᵀAx of the normalised input.
- A linear read-out over such features therefore lies inside the RKHS of the
  degree-2 polynomial kernel. Reviewers will ask (Q2, Q3) whether the
  quantum pipeline beats a classical model on the same inputs.

→ Code: the classical references `linear` (LinearSVC on x) and `poly2`
(SVC, kernel (x·x′ + 1)²) are part of every `full` run. `table_main` shows
them separately, and `table_stats` tests MO-GA-PVQNN against the best
classical reference as well as against the best quantum baseline.

## 4. Fair search baselines

| Ref | Paper | Relevance |
|---|---|---|
| B1 | K. Deligkaris, *PSO and Random Search for CNN NAS*, IEEE Access (2024), doi:10.1109/access.2024.3420870 | random search with the same budget is within 0.02–1.9 % of the evolutionary method |

**Consequences**
- The old "Random" baseline used only K = 8 circuits, while the GA evaluates
  hundreds. That comparison measures the search *budget*, not the search
  *strategy*.

→ Code: the `random_search` baseline samples as many unique random circuits
as the GA evaluated in the same run (`budget = phase1.unique_evaluations`).
It then applies the identical Pareto filter, max-shot re-evaluation and
greedy clean + robust ensemble. Any remaining gap is due to the NSGA-II
operators.

## 5. Noise in the measurement itself

| Ref | Paper | Relevance |
|---|---|---|
| N1 | H.-Y. Hu et al., *Robust and efficient quantum property learning with shallow shadows*, Nat. Commun. 16 (2025), doi:10.1038/s41467-025-57349-w | measurement noise biases shadow estimates; mitigation in post-processing |
| N2 | H. Jnane et al., *Quantum error mitigated classical shadows*, PRX Quantum 5 (2024), doi:10.1103/prxquantum.5.010324 | unbiased mitigated shadows at a sampling overhead |
| N3 | F. J. Kiwit et al., *Typical ML datasets as low-depth quantum circuits*, Quantum Sci. Technol. (2025), doi:10.1088/2058-9565/ae0123 | amplitude encoding is expensive; low-depth approximate loaders |

**Consequences**
- Gate noise is not the only error source for a shadow-based model. Readout
  error multiplies a weight-k estimate by (1 − 2q)ᵏ.
- The robust-shadow correction (divide each factor by 1 − 2q) removes the
  bias at a variance cost of (1 − 2q)⁻²ᵏ.
- Amplitude-encoding state preparation is assumed noiseless here (N3). This
  must be stated as a limitation.

→ Code: readout bit-flips are implemented in the shadow sampler and in the
exact estimator, and tested for unbiasedness. Every `full` run re-evaluates
its selected ensemble with `phase2.readout_levels`, both raw and mitigated.
The mitigated features are the raw measurement record rescaled by
(1 − 2q)^−k, so the comparison is exactly paired → `table_readout_sweep.tex`.

## 6. Statistics

Seven configurations are tested at once, so a single p < 0.05 is expected by
chance. `table_stats` therefore reports Holm–Bonferroni-adjusted p-values
next to the raw ones, and the README recommends 10 seeds.

## References in BibTeX-ready form

- Li C.-L. et al. (2026) Noise-Aware Quantum Architecture Search Based on NSGA-II Algorithm. arXiv:2601.10965
- Maleki K. et al. (2026) QNAS. arXiv:2604.07013
- Ma Q. et al. (2024) EPJ Quantum Technol. doi:10.1140/epjqt/s40507-024-00265-7
- Wang H. (2025) arXiv:2506.09205
- Patel Y. J. et al. (2024) arXiv:2402.03500
- Mousavi S. et al. (2026) Entropy 28:330, doi:10.3390/e28030330
- Huang Y. et al. (2024) Phys. Rev. Research 6:033033
- Su J.-J. et al. (2025) Sci. China Inf. Sci., doi:10.1007/s11432-024-4486-x
- Chen J. et al. (2024) arXiv:2403.04268
- Huang P.-W., Rebentrost P. (2023) arXiv:2307.10560
- Bermejo P. et al. (2024) PRX Quantum, doi:10.1103/8qt9-72ts
- Jerbi S. et al. (2024) Nat. Commun. 15, doi:10.1038/s41467-024-49877-8
- Yogaraj K. et al. (2025) Sci. Rep., doi:10.1038/s41598-025-08887-2
- Vandromme M. et al. (2025) SC25-W, doi:10.1145/3731599.3767540
- Deligkaris K. (2024) IEEE Access, doi:10.1109/access.2024.3420870
- Hu H.-Y. et al. (2025) Nat. Commun. 16, doi:10.1038/s41467-025-57349-w
- Jnane H. et al. (2024) PRX Quantum 5:010324
- Kiwit F. J. et al. (2025) Quantum Sci. Technol., doi:10.1088/2058-9565/ae0123
- Bui C. V. et al. (2026) ICMCSI, doi:10.1109/icmcsi67283.2026.11412580
- Bui C. V. et al. (2025) CIoTSC, doi:10.1109/ciotsc67482.2025.11413040
