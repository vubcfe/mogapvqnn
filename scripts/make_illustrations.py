"""Illustrations for the manuscript, built from the data and the run records.

    python scripts/make_illustrations.py [--runs results/runs] [--out results/figures]
                                         [--circuit-run full/fashion_mnist_6q/seed0]

* fig_datasets.pdf  one image per class of each dataset, as stored and after
                    the grayscale conversion and nearest-neighbour downsampling
                    used for 4, 6 and 8 qubits (shown as the amplitude vector
                    reshaped to the image grid, on a grey scale shared by the
                    two classes of each column).
* fig_circuits.tex  quantikz code of the circuits of one selected MO-GA-PVQNN
                    ensemble (gates placed in as-soon-as-possible layers),
                    plus summary statistics of all selected circuits on stdout.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mogapvqnn.data import (  # noqa: E402
    FASHION_LABELS,
    amplitude_encode,
    downsample,
    image_side,
    load_eurosat,
    load_idx_dataset,
)

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7})
ROOT = Path(__file__).resolve().parents[1]
DATASETS = [  # (name, title, classes, class labels)
    ("fashion_mnist", "Fashion-MNIST", (0, 4), ("T-shirt/top", "Coat")),
    ("mnist", "MNIST", (3, 8), ("3", "8")),
    ("eurosat", "EuroSAT", ("AnnualCrop", "Forest"), ("AnnualCrop", "Forest")),
]
QUBITS = (4, 6, 8)


def _examples(name, classes):
    """First training image of each class (deterministic)."""
    if name == "eurosat":
        X, y = load_eurosat(list(classes))
        return [X[np.flatnonzero(y == k)[0]] for k in (0, 1)]
    X, y, _, _ = load_idx_dataset(name)
    return [X[np.flatnonzero(y == c)[0]] for c in classes]


def fig_datasets(out: Path) -> None:
    cols = 1 + len(QUBITS)
    fig = plt.figure(figsize=(7.0, 1.55))
    outer = fig.add_gridspec(1, len(DATASETS), wspace=0.12, left=0.055, right=0.995, top=0.86, bottom=0.02)
    for b, (name, title, classes, labels) in enumerate(DATASETS):
        imgs = _examples(name, classes)
        enc = {c: [amplitude_encode(downsample(im[None], image_side(q), "nearest"))[0]
                   .reshape(image_side(q), image_side(q)) for im in imgs]
               for c, q in enumerate(QUBITS, start=1)}
        inner = outer[b].subgridspec(2, cols, wspace=0.06, hspace=0.08)
        for r, (img, lab) in enumerate(zip(imgs, labels)):
            for c in range(cols):
                ax = fig.add_subplot(inner[r, c])
                ax.set_xticks([]), ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_linewidth(0.4)
                if c == 0:
                    ax.imshow(img, cmap=None if img.ndim == 3 else "gray", interpolation="nearest")
                    ax.set_ylabel(lab, fontsize=6.5, labelpad=2)
                else:
                    # common grey scale for both classes of a column, so that
                    # amplitudes can be compared between the classes
                    v = enc[c][r]
                    vmax = max(e.max() for e in enc[c]) or 1.0
                    ax.imshow(v, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
                if r == 0:
                    ax.set_title("original" if c == 0 else f"$n={QUBITS[c - 1]}$", pad=2)
        x0 = outer[b].get_position(fig).x0
        x1 = outer[b].get_position(fig).x1
        fig.text((x0 + x1) / 2, 0.975, title, ha="center", va="top", fontsize=8, fontweight="bold")
    fig.savefig(out / "fig_datasets.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #

def _layers(circuit, n):
    """Place each gate in the earliest column after the previous gate on its own
    wires whose cells are all free; a CNOT also occupies the wires its vertical
    line crosses, so no gate is drawn on top of it. Gates on different wires
    commute, so the drawn circuit equals the chromosome."""
    last = [-1] * n
    used: set[tuple[int, int]] = set()
    placed = []
    for g in circuit:
        if g["gate"] == "I":
            continue
        w = g["wires"]
        span = range(min(w), max(w) + 1)
        col = max(last[q] for q in w) + 1
        while any((q, col) in used for q in span):
            col += 1
        used.update((q, col) for q in span)
        for q in w:
            last[q] = col
        placed.append((col, g))
    return placed, (max(c for c, _ in placed) + 1) if placed else 0


def _cell(g, q):
    if g["gate"] == "CNOT":
        c, t = g["wires"]
        return f"\\ctrl{{{t - c}}}" if q == c else "\\targ{}"
    if g["gate"] == "H":
        return "\\gate{H}"
    sub = g["gate"][1]
    return f"\\gate{{R_{sub}({g['angle']:.2f})}}"


def quantikz(circuit, n) -> str:
    placed, depth = _layers(circuit, n)
    grid = [["" for _ in range(depth)] for _ in range(n)]
    for col, g in placed:
        for q in g["wires"]:
            grid[q][col] = _cell(g, q)
    rows = []
    for q in range(n):
        cells = [f"\\lstick{{$q_{q}$}}"] + grid[q] + ["\\meter{}"]
        rows.append(" & ".join(cells))
    return ("\\begin{quantikz}[row sep={0.5cm,between origins}, column sep=0.25cm, font=\\scriptsize]\n"
            + " \\\\\n".join(rows) + "\n\\end{quantikz}")


def fig_circuits(runs: Path, out: Path, run_id: str) -> None:
    r = json.load(open(runs / f"{run_id}.json"))
    n = r["config"]["n_qubits"]
    members = r["ensembles"]["greedy"]["members"]
    parts = []
    for k, i in enumerate(members):
        f = r["front"][i]
        label = (f"({chr(97 + k)}) depth {f['depth']}, {f['cnot']} CNOT, "
                 f"validation {100 * f['val_acc']:.1f}\\%, test {100 * f['test_acc']:.1f}\\%")
        parts.append(f"\\begin{{minipage}}[t]{{0.49\\textwidth}}\\centering\n"
                     f"\\adjustbox{{max width=\\linewidth}}{{{quantikz(f['circuit'], n)}}}\\\\[2pt]\n"
                     f"{{\\footnotesize {label}}}\n\\end{{minipage}}")
    body = ""
    for k in range(0, len(parts), 2):
        body += "\\hfill\n".join(parts[k:k + 2]) + "\n\\par\\smallskip\n"
    header = (f"% generated by scripts/make_illustrations.py from results/runs/{run_id}.json\n"
              f"% ensemble test accuracy {100 * r['ensembles']['greedy']['test_acc']:.1f}%\n")
    (out / "fig_circuits.tex").write_text(header + body)

    # statistics of all selected circuits
    for variant in ("full", "measured_noise"):
        d, c = [], []
        for p in glob.glob(str(runs / variant / "*" / "seed*.json")):
            rr = json.load(open(p))
            for i in rr["ensembles"]["greedy"]["members"]:
                d.append(rr["front"][i]["depth"])
                c.append(rr["front"][i]["cnot"])
        if d:
            c = np.asarray(c)
            print(f"{variant}: {len(d)} selected circuits, mean depth {np.mean(d):.2f}, "
                  f"mean CNOT {c.mean():.2f}, no CNOT {100 * np.mean(c == 0):.0f}%, max CNOT {c.max()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "results" / "runs"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figures"))
    ap.add_argument("--circuit-run", default="full/fashion_mnist_6q/seed0")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_datasets(out)
    fig_circuits(Path(a.runs), out, a.circuit_run)
    print(f"illustrations written to {out}")


if __name__ == "__main__":
    main()
