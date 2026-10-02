"""Aggregate run JSONs into the manuscript's tables (LaTeX + Markdown) and
figure data (CSV).

Conventions: std is the sample standard deviation (ddof = 1); confidence
intervals use Student-t with n - 1 degrees of freedom; comparisons across
methods use the paired t-test over matched seeds (same data split), with
Holm-Bonferroni adjustment across the configurations of one table.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

METHOD_LABELS = {
    "mo_ga": "MO-GA-PVQNN", "random": "Random", "random_search": "Rand.~search", "manual": "Manual",
    "hea": "HEA", "pvqnn": "PVQNN", "grid": "Grid", "linear": "Linear SVM", "poly2": "Poly-2 SVM",
}
QUANTUM_BASELINES = ["random", "random_search", "manual", "hea", "pvqnn", "grid"]
CLASSICAL_BASELINES = ["linear", "poly2"]
ALL_METHODS = ["mo_ga"] + QUANTUM_BASELINES + CLASSICAL_BASELINES
DATASET_LABELS = {"fashion_mnist": "Fashion-MNIST", "mnist": "MNIST", "eurosat": "EuroSAT"}
CONFIG_ORDER = ["fashion_mnist", "mnist", "eurosat"]


def load_runs(root: Path) -> dict[str, dict[str, dict[int, dict]]]:
    """runs[variant][config_name][seed] = result"""
    runs: dict = defaultdict(lambda: defaultdict(dict))
    for p in sorted(Path(root).glob("*/*/seed*.json")):
        variant, name = p.parent.parent.name, p.parent.name
        with open(p) as f:
            r = json.load(f)
        runs[variant][name][int(r["config"]["seed"])] = r
    return runs


def _sort_key(name: str):
    ds, q = name.rsplit("_", 1)
    return (CONFIG_ORDER.index(ds) if ds in CONFIG_ORDER else 99, int(q[:-1]))


def _label(name: str) -> str:
    ds, q = name.rsplit("_", 1)
    return f"{DATASET_LABELS.get(ds, ds)} {q}"


def mean_sd(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


def ci_half(x, level: float = 0.95) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan")
    return float(stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaNs are left untouched)."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    idx = [i for i in range(len(p)) if np.isfinite(p[i])]
    m = len(idx)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda j: p[j])):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()


def _best(methods, ms):
    """Method with the highest mean, ignoring methods without results."""
    valid = [m for m in methods if np.isfinite(ms[m][0])]
    return max(valid, key=lambda m: ms[m][0]) if valid else None


def paired(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out = {"n": len(d), "mean_diff": float(d.mean()), "ci_diff": ci_half(d)}
    if len(d) >= 2 and np.any(d != d[0]):
        t, p = stats.ttest_rel(a, b)
        out.update(t=float(t), p=float(p))
    else:
        out.update(t=float("nan"), p=float("nan"))
    return out


def method_scores(run: dict) -> dict[str, float]:
    s = {"mo_ga": run["ensembles"]["greedy"]["test_acc"]}
    for k, v in run.get("baselines", {}).items():
        s[k] = v["test_acc"]
    return s


def common_levels(runs: list[dict], getter) -> tuple[list[str], str | None]:
    """Noise-level keys present in every run (sorted by value) and the key of
    the clean level (value 0), robust to formatting and to differing grids."""
    sets = [set(getter(r)) for r in runs if getter(r)]
    if not sets:
        return [], None
    levels = sorted(set.intersection(*sets), key=float)
    clean = next((e for e in levels if float(e) == 0.0), None)
    return levels, clean


def _pct(m, s=None, digits=1):
    if s is None:
        return f"{100 * m:.{digits}f}"
    return f"{100 * m:.{digits}f}\\pm{100 * s:.{digits}f}"


def _write(out: Path, name: str, latex: str, md: str) -> None:
    """Tables go to <out>/latex/<name>.tex and <out>/markdown/<name>.md."""
    for sub, text, ext in (("latex", latex, "tex"), ("markdown", md, "md")):
        (out / sub).mkdir(parents=True, exist_ok=True)
        (out / sub / f"{name}.{ext}").write_text(text)


# --------------------------------------------------------------------------- #

def table_main(full: dict, out: Path) -> dict:
    methods = ALL_METHODS
    rows_tex, rows_md, summary = [], [], {}
    for name in sorted(full, key=_sort_key):
        seeds = sorted(full[name])
        sc = {m: [method_scores(full[name][s]).get(m, np.nan) for s in seeds] for m in methods}
        ms = {m: mean_sd(v) for m, v in sc.items()}
        best = _best(methods, ms)
        summary[name] = {"seeds": seeds, "scores": sc, "mean_sd": ms}
        cells = []
        for m in methods:
            c = f"${_pct(*ms[m])}$"
            cells.append(f"$\\mathbf{{{_pct(*ms[m])}}}$" if m == best else c)
        rows_tex.append(f"{_label(name)} & " + " & ".join(cells) + r" \\")
        rows_md.append(f"| {_label(name)} | " + " | ".join(
            (f"**{_pct(*ms[m])}**" if m == best else _pct(*ms[m])).replace("\\pm", " ± ") for m in methods) + " |")
    n_seeds = max(len(v["seeds"]) for v in summary.values()) if summary else 0
    head = " & ".join(METHOD_LABELS[m] for m in methods)
    tex = (
        "\\begin{table*}[t]\n\\caption{Test accuracy (\\%, mean $\\pm$ sample std over "
        f"{n_seeds} seeds). Quantum methods share features, shots and read-out; Linear/Poly-2 SVM are "
        "classical references on the same amplitude-encoded inputs. Bold: best mean in each row."
        "}\\label{tab:main}\n"
        f"\\scriptsize\\setlength\\tabcolsep{{3pt}}\n\\begin{{tabular}}{{@{{}}l{'c' * len(methods)}@{{}}}}\n\\toprule\n"
        f" & \\multicolumn{{{1 + len(QUANTUM_BASELINES)}}}{{c}}{{Quantum}} & "
        f"\\multicolumn{{{len(CLASSICAL_BASELINES)}}}{{c}}{{Classical}} \\\\\n"
        f"Configuration & {head} \\\\\n\\midrule\n" + "\n".join(rows_tex) + "\n\\botrule\n\\end{tabular}\n\\end{table*}\n"
    )
    md = "| Configuration | " + " | ".join(METHOD_LABELS[m].replace("~", " ") for m in methods) + " |\n|" + "---|" * (len(methods) + 1) + "\n" + "\n".join(rows_md) + "\n"
    _write(out, "table_main", tex, md)
    return summary


def table_stats(summary: dict, out: Path) -> None:
    """MO-GA-PVQNN vs the best quantum baseline and vs the best classical
    reference of each configuration; Holm-adjusted across configurations."""
    names = sorted(summary, key=_sort_key)
    comp = {}
    for name in names:
        sc, ms = summary[name]["scores"], summary[name]["mean_sd"]
        q = _best(QUANTUM_BASELINES, ms)
        c = _best(CLASSICAL_BASELINES, ms)
        comp[name] = {
            "q": (q, paired(sc["mo_ga"], sc[q]) if q else None),
            "c": (c, paired(sc["mo_ga"], sc[c]) if c else None),
            "ci": ci_half(sc["mo_ga"]),
        }
    for fam in ("q", "c"):
        adj = holm([comp[n][fam][1]["p"] if comp[n][fam][1] else np.nan for n in names])
        for n, a in zip(names, adj):
            if comp[n][fam][1]:
                comp[n][fam][1]["p_holm"] = a

    def cell(name, fam, tex=True):
        m, pt = comp[name][fam]
        if pt is None:
            return ["--"] * 4
        ms = summary[name]["mean_sd"]
        p = pt.get("p_holm", np.nan)
        ptxt = f"{p:.3f}" if np.isfinite(p) else "--"
        if tex and np.isfinite(p) and p < 0.05:
            ptxt = f"$\\mathbf{{{ptxt}}}$"
        lab = METHOD_LABELS[m] if tex else METHOD_LABELS[m].replace("~", " ")
        diff = f"{100 * pt['mean_diff']:+.2f}\\pm{100 * pt['ci_diff']:.2f}"
        return [f"{lab} ({_pct(ms[m][0], digits=1)})", f"${diff}$" if tex else diff.replace("\\pm", " ± "),
                f"{pt['p']:.3f}" if np.isfinite(pt["p"]) else "--", ptxt]

    rows_tex, rows_md = [], []
    for name in names:
        mo = summary[name]["mean_sd"]["mo_ga"][0]
        ci = comp[name]["ci"]
        rows_tex.append(f"{_label(name)} & {_pct(mo, digits=2)} & $\\pm{100 * ci:.2f}$ & "
                        + " & ".join(cell(name, "q") + cell(name, "c")) + r" \\")
        rows_md.append(f"| {_label(name)} | {_pct(mo, digits=2)} | ±{100 * ci:.2f} | "
                       + " | ".join(cell(name, "q", False) + cell(name, "c", False)) + " |")
    tex = (
        "\\begin{table*}[t]\n\\caption{Paired $t$-tests of MO-GA-PVQNN against the best quantum baseline and the "
        "best classical reference of each configuration (matched seeds). CI: 95\\% half-width of the MO-GA-PVQNN "
        "mean; $\\Delta$: mean paired difference with 95\\% CI (pp); $p_\\mathrm{H}$: Holm-adjusted over the "
        "configurations ($p_\\mathrm{H}<0.05$ in bold).}\\label{tab:stats}\n\\scriptsize\\setlength\\tabcolsep{3pt}\n"
        "\\begin{tabular}{@{}lcccccccccc@{}}\n\\toprule\n"
        " & & & \\multicolumn{4}{c}{vs.\\ best quantum baseline} & \\multicolumn{4}{c}{vs.\\ best classical} \\\\\n"
        "Configuration & MO-GA & 95\\% CI & Baseline & $\\Delta$ & $p$ & $p_\\mathrm{H}$ & Classical & $\\Delta$ & $p$ & "
        "$p_\\mathrm{H}$ \\\\\n\\midrule\n" + "\n".join(rows_tex) + "\n\\botrule\n\\end{tabular}\n\\end{table*}\n"
    )
    md = ("| Configuration | MO-GA | 95% CI | Best quantum | Δ (pp) | p | p Holm | Best classical | Δ (pp) | p | p Holm |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows_md) + "\n")
    _write(out, "table_stats", tex, md)

def table_pareto(full: dict, out: Path) -> None:
    rows_tex, rows_md = [], []
    for name in sorted(full, key=_sort_key):
        rs = list(full[name].values())
        hv = mean_sd([r["pareto"]["hv"] for r in rs])
        size = mean_sd([r["pareto"]["size"] for r in rs])
        spread = mean_sd([r["pareto"]["val_acc_spread_max_shots"] for r in rs])
        gens = mean_sd([r["phase1"]["generations_run"] for r in rs])
        early = sum(r["phase1"]["stopped_early"] for r in rs)
        rows_tex.append(f"{_label(name)} & ${hv[0]:.3f}\\pm{hv[1]:.3f}$ & ${size[0]:.1f}$ & "
                        f"${100 * spread[0]:.1f}$ pp & ${gens[0]:.1f}$ & {early}/{len(rs)} \\\\")
        rows_md.append(f"| {_label(name)} | {hv[0]:.3f} ± {hv[1]:.3f} | {size[0]:.1f} | {100 * spread[0]:.1f} pp | "
                       f"{gens[0]:.1f} | {early}/{len(rs)} |")
    tex = (
        "\\begin{table}[t]\n\\caption{Pareto-front quality (mean over seeds). HV: hypervolume of the final "
        "non-dominated front in normalised objective space (reference point 1.1 per objective, scaled to $[0,1]$). "
        "Spread: range of validation accuracy across the front. Gens: generations run; ES: runs stopped early."
        "}\\label{tab:pareto}\n\\footnotesize\n\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
        "Configuration & HV & $|\\mathcal{F}_0|$ & Spread & Gens & ES \\\\\n\\midrule\n"
        + "\n".join(rows_tex) + "\n\\botrule\n\\end{tabular}\n\\end{table}\n"
    )
    md = ("| Configuration | HV | |F0| | Spread | Gens | Early stop |\n|---|---|---|---|---|---|\n" + "\n".join(rows_md) + "\n")
    _write(out, "table_pareto", tex, md)


def table_noise(full: dict, out: Path, name: str = "table_noise", note: str = "") -> None:
    """Ensemble accuracy at every noise level on identical samples, and the
    accuracy change at the smallest non-zero level for MO-GA-PVQNN and the
    quantum baselines that were noise-evaluated."""
    rows_tex, rows_md = [], []
    compare = ["random_search", "hea", "grid"]
    all_runs = [r for c in full.values() for r in c.values()]
    # one noise grid for the whole table (header and every row agree)
    levels, clean = common_levels(all_runs, lambda r: r["ensembles"]["greedy"].get("noisy_test", {}))
    if not levels or clean is None:
        return
    nz = [e for e in levels if float(e) > 0]
    for cfg_name in sorted(full, key=_sort_key):
        rs = list(full[cfg_name].values())
        nt = [r["ensembles"]["greedy"]["noisy_test"] for r in rs]
        accs = {e: mean_sd([x[e] for x in nt]) for e in levels}
        delta = mean_sd([x[nz[0]] - x[clean] for x in nt]) if nz else (np.nan, np.nan)
        deltas = {}
        for b in compare:
            vals = [r["baselines"][b]["noisy_test"] for r in rs
                    if "noisy_test" in r.get("baselines", {}).get(b, {})]
            vals = [v for v in vals if nz and nz[0] in v and clean in v]
            deltas[b] = mean_sd([v[nz[0]] - v[clean] for v in vals]) if vals else (np.nan, np.nan)
        sizes = sorted({r["phase2_samples"]["test"] for r in rs})
        n_p2 = str(sizes[0]) if len(sizes) == 1 else "/".join(map(str, sizes))
        cells = " & ".join(f"${_pct(*accs[e])}$" for e in levels)
        dcells = " & ".join(f"${100 * deltas[b][0]:+.1f}$" if np.isfinite(deltas[b][0]) else "--" for b in compare)
        rows_tex.append(f"{_label(cfg_name)} & {n_p2} & {cells} & ${100 * delta[0]:+.1f}$ & {dcells} \\\\")
        rows_md.append(f"| {_label(cfg_name)} | {n_p2} | " + " | ".join(_pct(*accs[e]).replace("\\pm", " ± ") for e in levels)
                       + f" | {100 * delta[0]:+.1f} | " + " | ".join(
                           f"{100 * deltas[b][0]:+.1f}" if np.isfinite(deltas[b][0]) else "--" for b in compare) + " |")
    nz0 = nz[0] if nz else "--"
    head = " & ".join("clean" if float(e) == 0 else f"$\\epsilon={e}$" for e in levels)
    dhead = " & ".join(f"$\\delta_\\mathrm{{{METHOD_LABELS[b].replace('~', ' ')}}}$" for b in compare)
    tex = (
        "\\begin{table*}[t]\n\\caption{Phase~2 noise robustness of the MO-GA-PVQNN ensemble under per-gate "
        f"depolarizing noise{note} (test accuracy \\%, mean $\\pm$ std). Clean and noisy accuracies are measured on "
        f"the same $N$ test samples with the same shot budget. $\\delta$: accuracy change at $\\epsilon={nz0}$ (pp)."
        f"}}\\label{{tab:{name.replace('table_', '')}}}\n\\scriptsize\\setlength\\tabcolsep{{3pt}}\n"
        f"\\begin{{tabular}}{{@{{}}lc{'c' * len(levels)}c{'c' * len(compare)}@{{}}}}\n\\toprule\n"
        f"Configuration & $N$ & {head} & $\\delta$ & {dhead} \\\\\n\\midrule\n"
        + "\n".join(rows_tex) + "\n\\botrule\n\\end{tabular}\n\\end{table*}\n"
    )
    md = ("| Configuration | N | " + " | ".join("clean" if float(e) == 0 else f"ε={e}" for e in levels)
          + " | δ | " + " | ".join(f"δ {METHOD_LABELS[b].replace('~', ' ')}" for b in compare) + " |\n|"
          + "---|" * (len(levels) + 3 + len(compare)) + "\n" + "\n".join(rows_md) + "\n")
    _write(out, name, tex, md)

def table_readout(full: dict, out: Path) -> None:
    """Post-hoc readout study: the selected ensemble at noise 0 and at the
    smallest gate-noise level, without readout error, with readout error q
    (raw) and with robust-shadow mitigation. All columns use the same
    ensemble and test samples (paired)."""
    runs = [r for c in full.values() for r in c.values()]
    with_ro = [r for r in runs if r["ensembles"]["greedy"].get("readout")]
    if not with_ro:
        return
    qs = sorted(set.intersection(*[set(r["ensembles"]["greedy"]["readout"]) for r in with_ro]), key=float)
    levels, clean = common_levels(with_ro, lambda r: r["ensembles"]["greedy"]["noisy_test"])
    nz = [e for e in levels if float(e) > 0]
    if not qs or clean is None:
        return
    q = qs[0]
    cols = [(clean, None, "clean")]
    cols += [(clean, "raw", f"RO {q}"), (clean, "mitigated", "RO mit.")]
    if nz:
        e = nz[0]
        cols += [(e, None, f"$\\epsilon={e}$"), (e, "raw", "$\\epsilon$ + RO"), (e, "mitigated", "$\\epsilon$ + RO mit.")]

    def val(r, e, kind):
        g = r["ensembles"]["greedy"]
        return g["noisy_test"][e] if kind is None else g["readout"][q][kind][e]

    rows_tex, rows_md = [], []
    for name in sorted(full, key=_sort_key):
        rs = [r for r in full[name].values() if r["ensembles"]["greedy"].get("readout", {}).get(q)]
        if not rs:
            continue
        ms = [mean_sd([val(r, e, k) for r in rs]) for e, k, _ in cols]
        rows_tex.append(f"{_label(name)} & " + " & ".join(f"${_pct(*m)}$" for m in ms) + r" \\")
        rows_md.append(f"| {_label(name)} | " + " | ".join(_pct(*m).replace("\\pm", " ± ") for m in ms) + " |")
    head = " & ".join(h for _, _, h in cols)
    tex = (
        "\\begin{table*}[t]\n\\caption{Readout-error study of the selected MO-GA-PVQNN ensemble (test accuracy \\%, "
        f"mean $\\pm$ std). RO: symmetric readout bit-flip probability {q}; mit.: robust-shadow correction, i.e. the "
        "same measurement record with every snapshot factor divided by $1-2q$. All columns use the same ensemble "
        "and test samples.}\\label{tab:readout}\n\\footnotesize\n"
        f"\\begin{{tabular}}{{@{{}}l{'c' * len(cols)}@{{}}}}\n\\toprule\nConfiguration & {head} \\\\\n\\midrule\n"
        + "\n".join(rows_tex) + "\n\\botrule\n\\end{tabular}\n\\end{table*}\n"
    )
    md = ("| Configuration | " + " | ".join(h.replace("$", "").replace("\\epsilon", "ε") for _, _, h in cols)
          + " |\n|" + "---|" * (len(cols) + 1) + "\n" + "\n".join(rows_md) + "\n")
    _write(out, "table_readout", tex, md)


def table_ablation(runs: dict, out: Path) -> None:
    full = runs.get("full", {})
    blocks = [("no_proxy", "Proxy off (3 obj.)"), ("measured_noise", "Measured noise obj."),
              ("exact_features", "Exact expectations")]
    rows_tex, rows_md = [], []
    for variant, label in blocks:
        other = runs.get(variant, {})
        for name in sorted(set(full) & set(other), key=_sort_key):
            seeds = sorted(set(full[name]) & set(other[name]))
            if not seeds:
                continue
            a = [full[name][s]["ensembles"]["greedy"]["test_acc"] for s in seeds]
            b = [other[name][s]["ensembles"]["greedy"]["test_acc"] for s in seeds]
            fa = mean_sd([full[name][s]["pareto"]["size"] for s in seeds])
            fb = mean_sd([other[name][s]["pareto"]["size"] for s in seeds])
            ha = mean_sd([full[name][s]["pareto"].get("hv_3obj", np.nan) for s in seeds])
            hb = mean_sd([other[name][s]["pareto"].get("hv_3obj", np.nan) for s in seeds])
            ta = mean_sd([full[name][s]["timings"]["phase1_s"] for s in seeds])
            tb = mean_sd([other[name][s]["timings"]["phase1_s"] for s in seeds])
            pt = paired(a, b)
            p_txt = f"{pt['p']:.3f}" if np.isfinite(pt["p"]) else "--"
            ma, mb = mean_sd(a), mean_sd(b)
            rows_tex.append(
                f"{label} & {_label(name)} & ${_pct(*ma)}$ & ${_pct(*mb)}$ & ${100 * pt['mean_diff']:+.1f}$ & {p_txt} & "
                f"{fa[0]:.1f} / {fb[0]:.1f} & {ha[0]:.3f} / {hb[0]:.3f} & {ta[0] / 60:.1f} / {tb[0] / 60:.1f} \\\\")
            rows_md.append(
                f"| {label} | {_label(name)} | {_pct(*ma).replace(chr(92) + 'pm', ' ± ')} | {_pct(*mb).replace(chr(92) + 'pm', ' ± ')} | "
                f"{100 * pt['mean_diff']:+.1f} | {p_txt} | {fa[0]:.1f} / {fb[0]:.1f} | {ha[0]:.3f} / {hb[0]:.3f} | "
                f"{ta[0] / 60:.1f} / {tb[0] / 60:.1f} |")
    if not rows_tex:
        return
    tex = (
        "\\begin{table}[t]\n\\caption{Ablations (test accuracy \\%, mean $\\pm$ std over matched seeds). "
        "$\\Delta$: full minus variant (pp), paired $t$-test $p$. Front size, hypervolume (computed in the common "
        "accuracy--depth--CNOT space for both arms) and Phase~1 time (min) are given as full / variant."
        "}\\label{tab:ablation}\n\\footnotesize\\setlength\\tabcolsep{3pt}\n\\begin{tabular}{@{}llccccccc@{}}\n\\toprule\n"
        "Variant & Configuration & Full & Variant & $\\Delta$ & $p$ & $|\\mathcal{F}_0|$ & HV$_3$ & Time \\\\\n\\midrule\n"
        + "\n".join(rows_tex) + "\n\\botrule\n\\end{tabular}\n\\end{table}\n"
    )
    md = ("| Variant | Configuration | Full | Variant | Δ | p | |F0| | HV3 | Phase1 min |\n|---|---|---|---|---|---|---|---|---|\n"
          + "\n".join(rows_md) + "\n")
    _write(out, "table_ablation", tex, md)


def table_ensemble(full: dict, out: Path) -> None:
    kinds = [("greedy", "Greedy (clean+robust)"), ("greedy_clean", "Greedy (clean)"), ("topk", "Top-$K$"),
             ("best_single", "Best single")]
    rows_tex, rows_md = [], []
    for name in sorted(full, key=_sort_key):
        rs = [full[name][s] for s in sorted(full[name])]
        cells, cells_md = [], []
        for k, _ in kinds:
            v = [r["ensembles"][k]["test_acc"] for r in rs if k in r["ensembles"]]
            m = mean_sd(v) if v else (np.nan, np.nan)
            cells.append(f"${_pct(*m)}$")
            cells_md.append(_pct(*m).replace("\\pm", " ± "))
        rows_tex.append(f"{_label(name)} & " + " & ".join(cells) + r" \\")
        rows_md.append(f"| {_label(name)} | " + " | ".join(cells_md) + " |")
    tex = (
        "\\begin{table}[t]\n\\caption{Ensemble-selection ablation: test accuracy (\\%) of different ways of "
        "building the final classifier from the same Pareto front.}\\label{tab:ensemble}\n\\footnotesize\n"
        "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nConfiguration & " + " & ".join(l for _, l in kinds)
        + " \\\\\n\\midrule\n" + "\n".join(rows_tex) + "\n\\botrule\n\\end{tabular}\n\\end{table}\n"
    )
    md = ("| Configuration | " + " | ".join(l.replace("$", "") for _, l in kinds) + " |\n|---|---|---|---|---|\n"
          + "\n".join(rows_md) + "\n")
    _write(out, "table_ensemble", tex, md)


def table_runtime(full: dict, out: Path) -> None:
    rows_tex, rows_md = [], []
    env = None
    worker_counts = set()
    for name in sorted(full, key=_sort_key):
        rs = list(full[name].values())
        env = env or rs[0]["environment"]
        worker_counts |= {r["config"]["workers"] for r in rs}
        t = {k: mean_sd([r["timings"].get(k, 0.0) / 60 for r in rs])
             for k in ("phase1_s", "phase2_s", "baselines_s", "total_s")}
        hit = mean_sd([r["cache"]["hit_rate"] for r in rs])
        rows_tex.append(f"{_label(name)} & " + " & ".join(f"${m:.1f}\\pm{s:.1f}$" for m, s in t.values())
                        + f" & {100 * hit[0]:.0f}\\% \\\\")
        rows_md.append(f"| {_label(name)} | " + " | ".join(f"{m:.1f} ± {s:.1f}" for m, s in t.values())
                       + f" | {100 * hit[0]:.0f}% |")
    workers = "/".join(map(str, sorted(worker_counts))) or "?"
    hw = f"{env.get('processor', '?')}, {workers} worker threads" if env else ""
    tex = (
        "\\begin{table}[t]\n\\caption{Wall-clock time per run in minutes (mean $\\pm$ std; " + hw.replace("_", "\\_")
        + "). Cache: feature-cache hit rate.}\\label{tab:runtime}\n\\footnotesize\n\\begin{tabular}{@{}lccccc@{}}\n"
        "\\toprule\nConfiguration & Phase~1 & Phase~2 & Baselines & Total & Cache \\\\\n\\midrule\n"
        + "\n".join(rows_tex) + "\n\\botrule\n\\end{tabular}\n\\end{table}\n"
    )
    md = ("| Configuration | Phase 1 | Phase 2 | Baselines | Total | Cache hit |\n|---|---|---|---|---|---|\n"
          + "\n".join(rows_md) + "\n")
    _write(out, "table_runtime", tex, md)


def figure_data(full: dict, out: Path) -> None:
    """CSV files for pgfplots: convergence (mean HV per generation, runs that
    stopped early are carried forward at their final value) and scaling."""
    for name in sorted(full, key=_sort_key):
        hs = [[h["hv"] for h in r["phase1"]["history"]] for r in full[name].values()]
        hs = [h for h in hs if h]
        if not hs:
            continue
        G = max(len(h) for h in hs)
        M = np.array([h + [h[-1]] * (G - len(h)) for h in hs])
        (out / "data").mkdir(parents=True, exist_ok=True)
        with open(out / "data" / f"convergence_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["generation", "hv_mean", "hv_std", "hv_min", "hv_max"])
            for g in range(G):
                col = M[:, g]
                w.writerow([g, f"{col.mean():.5f}", f"{col.std(ddof=1) if len(col) > 1 else 0:.5f}",
                            f"{col.min():.5f}", f"{col.max():.5f}"])
    (out / "data").mkdir(parents=True, exist_ok=True)
    with open(out / "data" / "accuracy_by_config.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "dataset", "qubits", "method", "mean", "std"])
        for name in sorted(full, key=_sort_key):
            ds, q = name.rsplit("_", 1)
            seeds = sorted(full[name])
            for m in METHOD_LABELS:
                v = [method_scores(full[name][s]).get(m, np.nan) for s in seeds]
                mm, ss = mean_sd(v)
                w.writerow([name, ds, q[:-1], m, f"{100 * mm:.2f}", f"{100 * ss:.2f}"])


def load_posthoc(root: Path) -> dict[str, dict[int, dict]]:
    """posthoc[config][seed] from scripts/posthoc.py output (may be empty)."""
    out: dict = defaultdict(dict)
    for p in sorted(Path(root).glob("*/seed*.json")):
        with open(p) as f:
            r = json.load(f)
        out[r["name"]][int(r["seed"])] = r
    return out


def table_readout_sweep(post: dict, out: Path) -> None:
    """Readout-only error (no gate noise) on the selected ensemble: raw vs
    robust-shadow mitigated, for every readout level (paired, same circuits
    and test samples as the clean column)."""
    if not post:
        return
    qs = sorted({q for c in post.values() for r in c.values() for q in r["readout_sweep"]}, key=float)
    cols = [(None, None, "clean")] + [(q, k, f"$q={q}$ {'mit.' if k == 'mitigated' else 'raw'}")
                                      for q in qs for k in ("raw", "mitigated")]
    rows_tex, rows_md = [], []
    for name in sorted(post, key=_sort_key):
        rs = list(post[name].values())
        ms = []
        for q, k, _ in cols:
            if q is None:
                ms.append(mean_sd([r["clean_test_p2"] for r in rs if "clean_test_p2" in r]))
            else:
                ms.append(mean_sd([r["readout_sweep"][q][k]["0.0"] for r in rs if q in r["readout_sweep"]]))
        cell = lambda m, tex: "--" if not np.isfinite(m[0]) else (  # noqa: E731
            f"${_pct(*m)}$" if tex else _pct(*m).replace("\\pm", " ± "))
        rows_tex.append(f"{_label(name)} & {len(rs)} & " + " & ".join(cell(m, True) for m in ms) + r" \\")
        rows_md.append(f"| {_label(name)} | {len(rs)} | " + " | ".join(cell(m, False) for m in ms) + " |")
    head = " & ".join(h for _, _, h in cols)
    tex = ("\\begin{table*}[t]\n\\caption{Readout-error sweep on the selected ensembles (no gate noise; "
           "test accuracy \\%, mean $\\pm$ std over seeds; $n$: seeds). raw: unmitigated; mit.: robust-shadow "
           "correction of the same measurement record.}\\label{tab:readout-sweep}\n\\footnotesize\\setlength\\tabcolsep{3pt}\n"
           f"\\begin{{tabular}}{{@{{}}lc{'c' * len(cols)}@{{}}}}\n\\toprule\nConfiguration & $n$ & {head} \\\\\n\\midrule\n"
           + "\n".join(rows_tex) + "\n\\botrule\n\\end{tabular}\n\\end{table*}\n")
    md = ("| Configuration | n | " + " | ".join(h.replace("$", "") for _, _, h in cols) + " |\n|"
          + "---|" * (len(cols) + 2) + "\n" + "\n".join(rows_md) + "\n")
    _write(out, "table_readout_sweep", tex, md)


def table_shots(post: dict, full: dict, out: Path) -> None:
    """Same selected circuits, models refitted with T shadow rounds or exact
    expectations: isolates shot noise from circuit choice."""
    if not post:
        return
    keys = sorted({k for c in post.values() for r in c.values() for k in r["shots_sweep"] if k != "exact"}, key=int)
    keys.append("exact")
    rows_tex, rows_md = [], []
    for name in sorted(post, key=_sort_key):
        rs = list(post[name].values())
        ms = [mean_sd([r["shots_sweep"][k]["ensemble"] for r in rs if k in r["shots_sweep"]]) for k in keys]
        gap = paired([r["shots_sweep"]["exact"]["ensemble"] for r in rs],
                     [r["shots_sweep"][keys[0]]["ensemble"] for r in rs])
        p_txt = f"{gap['p']:.3f}" if np.isfinite(gap["p"]) else "--"
        rows_tex.append(f"{_label(name)} & {len(rs)} & " + " & ".join(f"${_pct(*m)}$" for m in ms)
                        + f" & ${100 * gap['mean_diff']:+.1f}$ & {p_txt} \\\\")
        rows_md.append(f"| {_label(name)} | {len(rs)} | " + " | ".join(_pct(*m).replace("\\pm", " ± ") for m in ms)
                       + f" | {100 * gap['mean_diff']:+.1f} | {p_txt} |")
    head = " & ".join("exact" if k == "exact" else f"$T={k}$" for k in keys)
    tex = ("\\begin{table*}[t]\n\\caption{Shot-noise study: the selected MO-GA-PVQNN circuits with read-outs "
           "refitted on classical-shadow features with $T$ rounds, or on exact expectation values (test accuracy \\%, "
           f"mean $\\pm$ std). $\\Delta$: exact minus $T={keys[0]}$ (pp), paired $t$-test.}}\\label{{tab:shots}}\n"
           "\\footnotesize\n"
           f"\\begin{{tabular}}{{@{{}}lc{'c' * len(keys)}cc@{{}}}}\n\\toprule\nConfiguration & $n$ & {head} & $\\Delta$ & $p$ \\\\\n"
           "\\midrule\n" + "\n".join(rows_tex) + "\n\\botrule\n\\end{tabular}\n\\end{table*}\n")
    md = ("| Configuration | n | " + " | ".join("exact" if k == "exact" else f"T={k}" for k in keys)
          + " | Δ | p |\n|" + "---|" * (len(keys) + 4) + "\n" + "\n".join(rows_md) + "\n")
    _write(out, "table_shots", tex, md)


def aggregate(results_root: Path, out_dir: Path) -> list[str]:
    runs = load_runs(results_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    full = runs.get("full", {})
    if not full:
        raise SystemExit(f"no 'full' runs found under {results_root}")
    summary = table_main(full, out_dir)
    table_stats(summary, out_dir)
    table_pareto(full, out_dir)
    table_noise(full, out_dir)
    table_readout(full, out_dir)
    post = load_posthoc(Path(results_root).parent / "posthoc")
    table_readout_sweep(post, out_dir)
    table_shots(post, full, out_dir)
    table_ensemble(full, out_dir)
    table_ablation(runs, out_dir)
    table_runtime(full, out_dir)
    figure_data(full, out_dir)
    parts = ["# Results summary\n", "Runs: " + ", ".join(f"{v}={sum(len(s) for s in c.values())}" for v, c in runs.items()) + "\n"]
    for t in ["table_main", "table_stats", "table_pareto", "table_noise", "table_readout",
              "table_readout_sweep", "table_shots", "table_ensemble", "table_ablation", "table_runtime"]:
        p = out_dir / "markdown" / f"{t}.md"
        if p.exists():
            parts.append(f"\n## {t}\n\n" + p.read_text())
    (out_dir / "SUMMARY.md").write_text("".join(parts))
    return sorted(p.name for p in out_dir.rglob("*") if p.is_file())
