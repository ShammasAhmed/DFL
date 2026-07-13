"""
Collect the per-trial JSONs written by histogram_trial.py and draw the histograms.

For each training-set size: one figure per solver showing how often it picked a path
of each rank, and one figure putting all three side by side. The bars sit over the blue
curve of what those paths actually cost under f*, sorted ascending -- so rank 0 is the
true optimal path, and how bad a miss is can be read off the height of the curve above
the bar, which the counts alone cannot tell you.

Also writes a tidy CSV of the counts. Deliberately imports no PyEPO/Gurobi: the cost
curve is carried in the records, so this runs on a plain plotting node.

    python histogram_aggregate.py [--results results_histogram] [--outdir .] [--show]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

# Backend has to be settled before anything imports pyplot (plots.py does, below).
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sweep import (HIST_SIZES, HIST_NUM_TRIALS, HIST_RESULT_DIR,  # noqa: E402
                   SERIES, hist_result_path)
from plots import PathHistogramPlot  # noqa: E402

SOLVER_KEYS = [key for key, _, _ in SERIES]

# The trials all recompute the same context independently, so their cost curves agree
# to floating point rather than bit-for-bit.
COST_TOL = 1e-6


def load_records(results_dir):
    """Read every trial JSON. Returns (records, missing) with missing as (n, trial)."""
    records, missing = [], []
    for num_train in HIST_SIZES:
        for trial in range(HIST_NUM_TRIALS):
            path = hist_result_path(results_dir, num_train, trial)
            if not path.exists():
                missing.append((num_train, trial))
                continue
            records.append(json.loads(path.read_text()))
    return records, missing


def cost_curve(records):
    """
    The one true-cost curve every trial was scored against.

    Each record carries its own copy, and they must agree: a differing curve means that
    trial faced a different context or a different ground truth B, so its picks are not
    commensurable with the rest and counting them together would be meaningless. Loudly
    refuse rather than quietly average across two experiments.

    Raises:
        SystemExit: The records disagree about the context.
    """
    curves = np.array([rec["sorted_costs"] for rec in records])
    spread = np.abs(curves - curves[0]).max()
    if spread > COST_TOL:
        raise SystemExit(
            f"The trial records disagree about the true path costs (max difference "
            f"{spread:.3g}), so they did not all face the same fixed context. This "
            f"usually means {HIST_RESULT_DIR}/ mixes runs from different seeds or "
            f"degrees -- clear it and resubmit.")
    return curves[0]


def collect(records):
    """
    Count, per size and solver, how often a path of each rank was chosen.

    Returns:
        counts (dict): counts[num_train][solver_key] -> array over ranks
        trials (dict): trials[num_train] -> how many trials landed at that size
    """
    num_paths = len(records[0]["sorted_costs"])
    counts = {n: {key: np.zeros(num_paths, dtype=int) for key in SOLVER_KEYS}
              for n in HIST_SIZES}
    trials = {n: 0 for n in HIST_SIZES}

    for rec in records:
        n = rec["num_train"]
        trials[n] += 1
        for key in SOLVER_KEYS:
            counts[n][key][rec["chosen"][key]["rank"]] += 1
    return counts, trials


def plot_size(sorted_costs, counts, num_trials, num_train, outdir, show=False):
    """The per-solver histograms and the comparison figure, for one training size."""
    plotter = PathHistogramPlot(sorted_costs, num_trials, series=SERIES,
                                subtitle=f", training set size = {num_train}")

    figures = [(key, plotter.plot_solver(key, counts[key])) for key in SOLVER_KEYS]
    figures.append(("compare", plotter.plot_comparison(counts)))

    for name, fig in figures:
        path = outdir / f"path_histogram_n{num_train}_{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"wrote {path}")
    if show:
        plt.show()
    else:
        for _, fig in figures:
            plt.close(fig)


def write_csv(sorted_costs, counts, outdir):
    """One row per (size, rank): the path's true cost and each solver's count."""
    path = outdir / "path_histogram_counts.csv"
    lines = [",".join(["num_train", "rank", "true_cost", *SOLVER_KEYS])]
    for num_train in HIST_SIZES:
        for rank, cost in enumerate(sorted_costs):
            cells = [str(counts[num_train][key][rank]) for key in SOLVER_KEYS]
            lines.append(f"{num_train},{rank},{cost:.6f}," + ",".join(cells))
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def print_summary(sorted_costs, counts, trials):
    """Per size and solver: how often the true optimum was found, and mean regret."""
    for num_train in HIST_SIZES:
        n_trials = trials[num_train]
        plotter = PathHistogramPlot(sorted_costs, n_trials, series=SERIES)
        print(f"\nTraining set size {num_train} ({n_trials} trials)")
        header = f"{'solver':<10}{'optimal path':>15}{'median rank':>14}{'mean regret':>14}"
        print(header)
        print("-" * len(header))
        for key in SOLVER_KEYS:
            c = counts[num_train][key]
            if not c.sum():
                print(f"{key:<10}{'--':>15}{'--':>14}{'--':>14}")
                continue
            ranks = np.repeat(np.arange(len(c)), c)
            hit = 100 * c[0] / c.sum()
            print(f"{key:<10}{hit:>14.1f}%{np.median(ranks):>14.1f}"
                  f"{plotter.relative_regret(c):>13.2f}%")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(HIST_RESULT_DIR),
                        help="Directory of per-trial JSONs from histogram_trial.py")
    parser.add_argument("--outdir", default=".",
                        help="Where to write the PNGs and CSV")
    parser.add_argument("--show", action="store_true",
                        help="Display the figures (off by default; headless on a node)")
    args = parser.parse_args(argv)

    records, missing = load_records(args.results)
    if not records:
        print(f"No trial JSONs found in {args.results}/ -- has the array job run?",
              file=sys.stderr)
        return 1
    if missing:
        # Never silently draw a partial run as if it were complete.
        total = len(HIST_SIZES) * HIST_NUM_TRIALS
        print(f"WARNING: {len(missing)} of {total} trials are missing; the histograms "
              f"below count fewer selections.", file=sys.stderr)
        for n, trial in missing[:10]:
            print(f"  missing n={n} trial={trial}", file=sys.stderr)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more", file=sys.stderr)

    sorted_costs = cost_curve(records)
    counts, trials = collect(records)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print_summary(sorted_costs, counts, trials)
    write_csv(sorted_costs, counts, outdir)
    for num_train in HIST_SIZES:
        if trials[num_train]:
            plot_size(sorted_costs, counts[num_train], trials[num_train],
                      num_train, outdir, show=args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
