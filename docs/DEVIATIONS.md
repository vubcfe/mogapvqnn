# Differences from the earlier (QIP) manuscript

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

