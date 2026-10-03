"""Draw the manuscript figures from finished runs.

    python scripts/make_figures.py [--runs results/runs] [--out results/figures]

Figures (PDF):
  fig_accuracy.pdf     grouped test accuracy per dataset / qubit count (with std error bars)
  fig_scaling.pdf      MO-GA-PVQNN vs best baseline as a function of qubit count
  fig_convergence.pdf  Phase-1 hypervolume per generation (mean +/- std over seeds)
  fig_pareto.pdf       Pareto front of one run (accuracy vs complexity, colour = robust accuracy)
  fig_noise.pdf        ensemble test accuracy vs depolarizing strength, MO-GA vs baselines
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Figures are drawn at the two-column width of Physical Review (7 in), so
# that they are reproduced at 1:1 scale and text stays legible.
FULL_WIDTH = 7.0
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5})
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mogapvqnn.report import (  # noqa: E402
    CLASSICAL_BASELINES,
    CLASSICAL_TUNED,
    CLASSICAL_UNTUNED,
    DATASET_LABELS,
    METHOD_LABELS,
    QUANTUM_BASELINES,
    _best,
    _sort_key,
    common_levels,
    load_classical,
    load_posthoc,
    load_runs,
    mean_sd,
    merge_classical,
    method_scores,
    present_methods,
)

COLORS = {"mo_ga": "#D62728", "random": "#AEC7E8", "random_search": "#1F77B4", "manual": "#FFBB78",
          "hea": "#98DF8A", "pvqnn": "#C5B0D5", "grid": "#C49C94", "linear": "#7F7F7F", "poly2": "#BCBD22",
          "linear_cv": "#7F7F7F", "poly2_cv": "#BCBD22", "rbf_cv": "#17BECF"}
DS_COLORS = {"fashion_mnist": "#D62728", "mnist": "#1F77B4", "eurosat": "#2CA02C"}
# Bars of the accuracy figure: tuned classical references replace the untuned
# ones once scripts/classical_tuned.py has been run (set in main()).
METHODS = ["mo_ga"] + QUANTUM_BASELINES + CLASSICAL_UNTUNED


def _label(m):
    return METHOD_LABELS.get(m, m).replace("~", " ").replace("$^\\ast$", " (tuned)")


def _scores(full, name):
    seeds = sorted(full[name])
    return {m: [method_scores(full[name][s]).get(m, np.nan) for s in seeds] for m in METHODS}


def fig_accuracy(full, out):
    datasets = [d for d in DATASET_LABELS if any(n.startswith(d + "_") for n in full)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(FULL_WIDTH, 2.7), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        names = sorted([n for n in full if n.rsplit("_", 1)[0] == ds], key=_sort_key)
        x = np.arange(len(names))
        w = 0.8 / len(METHODS)
        for k, m in enumerate(METHODS):
            ms = [mean_sd(_scores(full, n)[m]) for n in names]
            ax.bar(x + (k - (len(METHODS) - 1) / 2) * w, [100 * a for a, _ in ms], w,
                   yerr=[100 * s for _, s in ms], color=COLORS[m], edgecolor="black", linewidth=0.3,
                   capsize=1.5, label=_label(m), hatch="//" if m in CLASSICAL_BASELINES else None)
        ax.axhline(50, color="gray", lw=0.8, ls=":")
        ax.set_xticks(x, [n.rsplit("_", 1)[1] for n in names])
        ax.set_title(DATASET_LABELS[ds])
        ax.set_xlabel("Qubits")
        ax.grid(axis="y", ls="--", alpha=0.3)
    axes[0][0].set_ylabel("Test accuracy (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               handlelength=1.2, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "fig_accuracy.pdf")
    plt.close(fig)


def fig_scaling(full, out):
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.0))
    for ds in DATASET_LABELS:
        names = sorted([n for n in full if n.rsplit("_", 1)[0] == ds], key=_sort_key)
        if not names:
            continue
        q = [int(n.rsplit("_", 1)[1][:-1]) for n in names]
        ga = [mean_sd(_scores(full, n)["mo_ga"])[0] * 100 for n in names]
        best_q, best_c = [], []
        for n in names:
            ms = {m: mean_sd(v) for m, v in _scores(full, n).items()}
            bq, bc = _best(QUANTUM_BASELINES, ms), _best([m for m in CLASSICAL_BASELINES if m in ms], ms)
            best_q.append(100 * ms[bq][0] if bq else np.nan)
            best_c.append(100 * ms[bc][0] if bc else np.nan)
        col = DS_COLORS.get(ds, "black")
        ax.plot(q, ga, "-o", color=col, label=f"{DATASET_LABELS[ds]}: MO-GA-PVQNN")
        ax.plot(q, best_q, "--s", color=col, alpha=0.55, label=f"{DATASET_LABELS[ds]}: best quantum baseline")
        ax.plot(q, best_c, ":^", color=col, alpha=0.55, label=f"{DATASET_LABELS[ds]}: best classical")
    ax.axhline(50, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Qubits")
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(fontsize=7)
    ax.grid(ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_scaling.pdf")
    plt.close(fig)


def fig_convergence(full, out, dataset="fashion_mnist"):
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.6))
    for name in sorted([n for n in full if n.startswith(dataset + "_")], key=_sort_key):
        hs = [[h["hv"] for h in r["phase1"]["history"]] for r in full[name].values()]
        hs = [h for h in hs if h]
        if not hs:
            continue
        G = max(len(h) for h in hs)
        M = np.array([h + [h[-1]] * (G - len(h)) for h in hs])
        g = np.arange(G)
        m, s = M.mean(0), M.std(0, ddof=1) if len(M) > 1 else np.zeros(G)
        line, = ax.plot(g, m, label=name.rsplit("_", 1)[1])
        ax.fill_between(g, m - s, m + s, color=line.get_color(), alpha=0.2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume of $\\mathcal{F}_0$")
    ax.set_title(DATASET_LABELS.get(dataset, dataset))
    ax.legend(title="Qubits")
    ax.grid(ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_convergence.pdf")
    plt.close(fig)


def fig_pareto(full, out, name="fashion_mnist_6q", seed=0):
    if name not in full or seed not in full[name]:
        return
    rows = full[name][seed]["front"]
    x = [r["depth"] + 2 * r["cnot"] for r in rows]
    y = [100 * r["val_acc"] for r in rows]
    c = [100 * r["phase2"]["robust_val"] if "phase2" in r else np.nan for r in rows]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.6, 3.0))
    sc = ax.scatter(x, y, c=c, cmap="viridis", s=45, edgecolor="black", linewidth=0.4)
    fig.colorbar(sc, ax=ax, label="Robust val. accuracy (%)")
    ax.set_xlabel("Complexity (depth + 2 $\\times$ CNOT)")
    ax.set_ylabel("Clean val. accuracy (%)")
    ax.set_title(f"Pareto front, {name}, seed {seed}")
    ax.grid(ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_pareto.pdf")
    plt.close(fig)


def fig_noise(full, out):
    names = sorted(full, key=_sort_key)
    cols = min(4, len(names))
    rows_n = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(FULL_WIDTH, 1.9 * rows_n + 0.4), squeeze=False)
    for ax, name in zip(axes.flat, names):
        rs = list(full[name].values())
        levels, _ = common_levels(rs, lambda r: r["ensembles"]["greedy"].get("noisy_test", {}))
        if not levels:
            ax.axis("off")
            continue
        eps = [float(e) for e in levels]
        series = {"mo_ga": [[r["ensembles"]["greedy"]["noisy_test"][e] for e in levels] for r in rs]}
        for b in sorted({b for r in rs for b in r.get("baselines", {})}):
            nts = [r.get("baselines", {}).get(b, {}).get("noisy_test") for r in rs]
            if all(nt and all(e in nt for e in levels) for nt in nts):
                series[b] = [[nt[e] for e in levels] for nt in nts]
        for m, v in series.items():
            v = 100 * np.array(v)
            ax.errorbar(eps, v.mean(0), yerr=v.std(0, ddof=1) if len(v) > 1 else None, marker="o", ms=3,
                        color=COLORS.get(m, "black"), lw=2 if m == "mo_ga" else 1, capsize=2, label=_label(m))
        ds_, q_ = name.rsplit("_", 1)
        ax.set_title(f"{DATASET_LABELS.get(ds_, ds_)} {q_}")
        ax.set_xlabel("$\\epsilon$")
        ax.grid(ls="--", alpha=0.3)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for r in range(rows_n):
        axes[r][0].set_ylabel("Test accuracy (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "fig_noise.pdf")
    plt.close(fig)


def fig_shots(post, out):
    """Accuracy of the selected circuits vs number of shadow rounds T (exact = right-most point)."""
    if not post:
        return
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.0))
    for name in sorted(post, key=_sort_key):
        rs = list(post[name].values())
        Ts = sorted({int(k) for r in rs for k in r["shots_sweep"] if k != "exact"})
        xs = Ts + [Ts[-1] * 4]
        ys = [[r["shots_sweep"][str(t)]["ensemble"] for r in rs] for t in Ts] + \
             [[r["shots_sweep"]["exact"]["ensemble"] for r in rs]]
        m = [100 * np.mean(y) for y in ys]
        sd = [100 * np.std(y, ddof=1) if len(y) > 1 else 0 for y in ys]
        ds = name.rsplit("_", 1)[0]
        ax.errorbar(xs, m, yerr=sd, marker="o", ms=3, capsize=2, color=DS_COLORS.get(ds, "black"),
                    ls="-" if name.endswith("4q") else ("--" if name.endswith("6q") else ":"),
                    label=f"{DATASET_LABELS.get(ds, ds)} {name.rsplit('_', 1)[1]}")
    ax.set_xscale("log")
    ax.set_xticks(xs, [str(t) for t in Ts] + ["exact"])
    ax.set_xlabel("Shadow rounds $T$")
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.grid(ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_shots.pdf")
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=str(root / "results" / "runs"))
    p.add_argument("--out", default=str(root / "results" / "figures"))
    p.add_argument("--pareto-run", default="fashion_mnist_6q")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    full = load_runs(Path(a.runs)).get("full", {})
    if not full:
        raise SystemExit("no finished 'full' runs found")
    global METHODS
    if merge_classical(full, load_classical(Path(a.runs).parent / "posthoc_classical")):
        METHODS = ["mo_ga"] + QUANTUM_BASELINES + present_methods(full, CLASSICAL_TUNED)
    fig_accuracy(full, out)
    fig_scaling(full, out)
    fig_convergence(full, out)
    fig_pareto(full, out, a.pareto_run)
    fig_noise(full, out)
    fig_shots(load_posthoc(Path(a.runs).parent / "posthoc"), out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
