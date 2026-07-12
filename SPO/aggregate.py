"""
Collect the per-trial JSONs written by run_trial.py and draw the sweep's boxplots.

One figure per training-set size, one panel per selected metric, so everything
measured at that size sits in a single image. Within a panel, one group per DGP
degree and one box per solver, each box summarizing the NUM_TRIALS trials of that
cell.

run_trial.py stores every metric it computed, so which panels get drawn is a choice
made here, not there: --metrics re-plots an existing sweep under a different
selection without recomputing anything. It defaults to sweep.SHOW_METRICS, which
main.py's printed table also follows.

Also writes a tidy CSV of every trial and prints a median table. Deliberately imports
no PyEPO/Gurobi, so it runs on a plain plotting node.

    python aggregate.py [--results results] [--outdir .] [--metrics ...] [--show]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

# Backend has to be settled before anything imports pyplot (plots.py does, below).
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sweep import (DEGREES, SIZES, NUM_TRIALS, SERIES, METRICS,  # noqa: E402
                   SHOW_METRICS, RESULT_DIR, result_path)
from plots import RegretBoxPlot  # noqa: E402

SOLVER_KEYS = [key for key, _, _ in SERIES]


def load_records(results_dir):
    """Read every trial JSON. Returns (records, missing) with missing as cells."""
    records = []
    missing = []
    for num_train in SIZES:
        for deg in DEGREES:
            for trial in range(NUM_TRIALS):
                path = result_path(results_dir, deg, num_train, trial)
                if not path.exists():
                    missing.append((deg, num_train, trial))
                    continue
                records.append(json.loads(path.read_text()))
    return records, missing


def available_metrics(records):
    """
    Which metrics the records actually carry, in sweep.METRICS order.

    A run whose generator supplied no f* writes no f*-based metrics, so the records
    -- not sweep.METRICS -- are the authority on what can be plotted.
    """
    present = set()
    for rec in records:
        for key in SOLVER_KEYS:
            present.update(rec["metrics"].get(key, {}))
    return [metric for metric in METRICS if metric in present]


def collect(records, metrics):
    """
    Reshape into by_size[num_train][metric][deg][solver_key] -> list of trial values,
    which is exactly the data[group][series_key] layout RegretBoxPlot consumes.

    A metric absent from a record is skipped rather than faked, so a partial or
    mixed results/ directory yields thinner boxes instead of a KeyError.
    """
    by_size = {
        n: {metric: {deg: defaultdict(list) for deg in DEGREES}
            for metric in metrics}
        for n in SIZES
    }
    for rec in records:
        n, deg = rec["num_train"], rec["deg"]
        for metric in metrics:
            for key in SOLVER_KEYS:
                value = rec["metrics"].get(key, {}).get(metric)
                if value is not None:
                    by_size[n][metric][deg][key].append(value)
    return by_size


def plot_size(by_size, metrics, num_train, outdir, show=False):
    """Side-by-side boxplot panels, one per selected metric, for one training size."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(9 * len(metrics), 6.5),
                             squeeze=False)

    for ax, metric in zip(axes[0], metrics):
        label = METRICS[metric].label
        plotter = RegretBoxPlot(
            groups=list(DEGREES),
            series=SERIES,
            xlabel="Polynomial degree of DGP",
            ylabel=label,
            title=label,
        )
        plotter.plot(by_size[num_train][metric], ax=ax)

    fig.suptitle(
        f"Pooled context regret over {NUM_TRIALS} trials per degree "
        f"(5x5 grid shortest path, training set size = {num_train})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    path = outdir / f"regret_boxplots_n{num_train}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def write_csv(records, metrics, outdir):
    """
    Tidy one-row-per-(trial, solver) CSV, for whatever downstream stats you want.

    Carries every metric the records hold, not just the plotted selection -- the CSV
    is the archive, the panels are the view.
    """
    path = outdir / "trials.csv"
    lines = [",".join(["num_train", "deg", "trial", "seed", "solver", *metrics])]
    for rec in sorted(records, key=lambda r: (r["num_train"], r["deg"], r["trial"])):
        for key in SOLVER_KEYS:
            m = rec["metrics"][key]
            values = [f"{m[metric]:.6f}" if metric in m else ""
                      for metric in metrics]
            lines.append(
                f"{rec['num_train']},{rec['deg']},{rec['trial']},{rec['seed']},"
                f"{key}," + ",".join(values)
            )
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def print_medians(by_size, metrics):
    for num_train in SIZES:
        for metric in metrics:
            print(f"\n{METRICS[metric].label} -- training set size {num_train} "
                  f"(median over trials)")
            header = f"{'deg':>5}" + "".join(f"{key:>16}" for key in SOLVER_KEYS)
            print(header)
            print("-" * len(header))
            for deg in DEGREES:
                cells = by_size[num_train][metric][deg]
                row = "".join(
                    f"{np.median(cells[key]):>15.4f}%" if cells[key] else f"{'--':>16}"
                    for key in SOLVER_KEYS
                )
                print(f"{deg:>5}" + row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(RESULT_DIR),
                        help="Directory of per-trial JSONs from run_trial.py")
    parser.add_argument("--outdir", default=".",
                        help="Where to write the PNGs and CSV")
    parser.add_argument("--metrics", default=",".join(SHOW_METRICS),
                        help="Comma-separated metrics to plot, one panel each "
                             f"(default: {','.join(SHOW_METRICS)}; "
                             f"known: {','.join(METRICS)})")
    parser.add_argument("--show", action="store_true",
                        help="Display the figures (off by default; headless on a node)")
    args = parser.parse_args(argv)

    records, missing = load_records(args.results)
    if not records:
        print(f"No trial JSONs found in {args.results}/ -- has the array job run?",
              file=sys.stderr)
        return 1
    if missing:
        # Never silently plot a partial sweep as if it were complete.
        total = len(SIZES) * len(DEGREES) * NUM_TRIALS
        print(f"WARNING: {len(missing)} of {total} trials are missing; the boxes "
              f"below are drawn from fewer samples.", file=sys.stderr)
        for deg, n, trial in missing[:10]:
            print(f"  missing deg={deg} n={n} trial={trial}", file=sys.stderr)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more", file=sys.stderr)

    stored = available_metrics(records)
    requested = [m.strip() for m in args.metrics.split(",") if m.strip()]

    unknown = [m for m in requested if m not in METRICS]
    if unknown:
        print(f"Unknown metric(s): {', '.join(unknown)}. Known metrics are "
              f"{', '.join(METRICS)}.", file=sys.stderr)
        return 1

    # A metric nobody stored cannot be plotted -- say so rather than draw an empty
    # panel. The usual cause is a sweep run with a generator that had no f*.
    absent = [m for m in requested if m not in stored]
    if absent:
        print(f"Metric(s) {', '.join(absent)} are not in these records (stored: "
              f"{', '.join(stored)}); rerun the sweep to compute them.",
              file=sys.stderr)
        return 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    by_size = collect(records, requested)
    print_medians(by_size, requested)
    write_csv(records, stored, outdir)
    for num_train in SIZES:
        plot_size(by_size, requested, num_train, outdir, show=args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())