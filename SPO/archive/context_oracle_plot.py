"""
The noisy-Y oracle as a yardstick, drawn from trials.csv.

An oracle handed the realized costs Y for the very context it is deciding on still pays
for chasing the noise: the path it picks, z*(Y), costs <f*, z*(Y)> in expectation, not
the <f*, z*(f*)> of the best path. That gap is a property of the DGP and the noise, not
of any model, so it is a fixed bar every solver can be held against -- and one no metric
in sweep.METRICS normalizes by, which is why it takes the raw sums to draw.

    oracle penalty = 100 * (<f*, z*(Y)> - <f*, z*(f*)>) / <f*, z*(f*)>

which is exactly regret_fstar's formula with the oracle's decision in place of a solver's,
so the bar and the boxes share one axis by construction.

Two figures:
  regret_fstar_vs_oracle_n{n}.png : the regret_fstar boxes with the bar over each degree
  oracle_win_rate.png             : how often each solver's decision beats the bar

    python context_oracle_plot.py [--csv trials.csv] [--outdir .] [--show]

Imports no PyEPO/Gurobi: the CSV holds everything these need.
"""
import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (after the backend is settled)

from sweep import DEGREES, SIZES, SERIES  # noqa: E402
from plots import RegretBoxPlot  # noqa: E402

ORACLE_COLOR = "tab:blue"   # no solver is blue (see sweep.SERIES), so the bar reads apart


def load(path):
    """
    Read trials.csv into the per-degree regrets and the oracle's penalty.

    Returns:
        (regret, oracle, beats): regret[n][deg][solver] -> per-trial regret_fstar (%);
            oracle[n][deg] -> per-trial oracle penalty (%), which is solver-independent
            and so read once per trial; beats[n][deg][solver] -> share of trials whose
            decision cost less under f* than the oracle's did.
    """
    regret = {n: {deg: defaultdict(list) for deg in DEGREES} for n in SIZES}
    oracle = {n: {deg: [] for deg in DEGREES} for n in SIZES}
    beaten = {n: {deg: defaultdict(list) for deg in DEGREES} for n in SIZES}
    seen = set()

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            n, deg, trial = (int(row["num_train"]), int(row["deg"]),
                             int(row["trial"]))
            if n not in regret or deg not in regret[n]:
                continue
            best = float(row["sum_fstar_zfstar"])    # <f*, z*(f*)>
            noisy = float(row["sum_fstar_zY"])       # <f*, z*(Y)>
            method = float(row["sum_fstar_zmethod"])  # <f*, z*(f(X))>

            regret[n][deg][row["solver"]].append(float(row["regret_fstar"]))
            beaten[n][deg][row["solver"]].append(method < noisy)
            # The oracle's penalty belongs to the trial, not the solver: count it once.
            if (n, deg, trial) not in seen:
                seen.add((n, deg, trial))
                oracle[n][deg].append(100 * (noisy - best) / best)

    if not seen:
        raise SystemExit(f"{path} has no rows for the sweep's degrees and sizes")
    return regret, oracle, beaten


def plot_boxes(regret, oracle, outdir, dpi=150):
    """One figure per training-set size: regret_fstar, with the oracle bar per degree."""
    paths = []
    for n in SIZES:
        plotter = RegretBoxPlot(
            groups=list(DEGREES), series=SERIES,
            xlabel="Polynomial degree of DGP",
            ylabel=r"Regret vs $f^*$ (%)",
            title="",
            figsize=(11, 6.5),
        )
        fig, ax = plotter.plot(regret[n])

        # One bar per degree, spanning its group: the oracle's penalty is a per-degree
        # property, and a single line across the axis would imply it is not.
        centers = np.arange(len(DEGREES)) * plotter.group_spacing
        half = plotter.group_spacing * 0.45
        for center, deg in zip(centers, DEGREES):
            ax.hlines(statistics.median(oracle[n][deg]), center - half, center + half,
                      color=ORACLE_COLOR, linestyle="--", linewidth=2, zorder=5)

        # boxplot patches are not labelled artists, so the legend has to be rebuilt by
        # hand rather than recovered with get_legend_handles_labels().
        handles = [matplotlib.patches.Patch(facecolor=color, alpha=plotter.alpha)
                   for _, _, color in SERIES]
        labels = [label for _, label, _ in SERIES]
        bar = matplotlib.lines.Line2D([], [], color=ORACLE_COLOR, linestyle="--",
                                      linewidth=2)
        ax.legend(handles + [bar], labels + ["noisy-Y oracle"], loc="upper left")
        ax.set_title(f"Regret vs $f^*$ against the noisy-Y oracle "
                     f"(50 trials per degree, training set size = {n})",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()

        path = outdir / f"regret_fstar_vs_oracle_n{n}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"wrote {path}")
        paths.append(path)
    return paths


def plot_win_rate(beaten, outdir, dpi=150):
    """How often each solver's decision beats the oracle's, by degree and size."""
    fig, axes = plt.subplots(1, len(SIZES), figsize=(7 * len(SIZES), 5.5), sharey=True)
    for ax, n in zip(np.atleast_1d(axes), SIZES):
        for key, label, color in SERIES:
            share = [100 * statistics.fmean(beaten[n][deg][key]) for deg in DEGREES]
            ax.plot(range(len(DEGREES)), share, marker="o", color=color, label=label)
        ax.axhline(50, color="black", linestyle=":", linewidth=1.2, zorder=0)
        ax.set_xticks(range(len(DEGREES)))
        ax.set_xticklabels(DEGREES)
        ax.set_xlabel("Polynomial degree of DGP")
        ax.set_ylabel("Trials beating the noisy-Y oracle (%)")
        ax.set_ylim(-5, 105)
        ax.set_title(f"training set size = {n}")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(loc="lower left")
    fig.suptitle("How often a solver's decision costs less, under $f^*$, than the "
                 "decision an oracle makes from one noisy draw",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = outdir / "oracle_win_rate.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"wrote {path}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="trials.csv",
                        help="Tidy per-trial CSV written by context_aggregate.py")
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    regret, oracle, beaten = load(args.csv)
    plot_boxes(regret, oracle, outdir)
    plot_win_rate(beaten, outdir)

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
