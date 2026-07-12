"""
Collect the per-trial JSONs written by run_trial.py and draw the sweep's boxplots.

One figure per training-set size, each with two panels -- Regret vs f* and Regret
vs Y -- so everything measured at that size sits in a single image. Within a panel,
one group per DGP degree and one box per solver, each box summarizing the NUM_TRIALS
trials of that cell.

Also writes a tidy CSV of every trial and prints a median table. Deliberately imports
no PyEPO/Gurobi, so it runs on a plain plotting node.

    python aggregate.py [--results results] [--outdir .] [--show]
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
                   RESULT_DIR, result_path)
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


def collect(records):
    """
    Reshape into by_size[num_train][metric][deg][solver_key] -> list of trial values,
    which is exactly the data[group][series_key] layout RegretBoxPlot consumes.
    """
    by_size = {
        n: {metric: {deg: defaultdict(list) for deg in DEGREES}
            for metric, _ in METRICS}
        for n in SIZES
    }
    for rec in records:
        n, deg = rec["num_train"], rec["deg"]
        for metric, _ in METRICS:
            for key in SOLVER_KEYS:
                by_size[n][metric][deg][key].append(rec["metrics"][key][metric])
    return by_size


def plot_size(by_size, num_train, outdir, show=False):
    """Two side-by-side boxplot panels (one per metric) for a single training size."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))

    for ax, (metric, metric_label) in zip(axes, METRICS):
        plotter = RegretBoxPlot(
            groups=list(DEGREES),
            series=SERIES,
            xlabel="Polynomial degree of DGP",
            ylabel=metric_label,
            title=metric_label,
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


def write_csv(records, outdir):
    """Tidy one-row-per-(trial, solver) CSV, for whatever downstream stats you want."""
    path = outdir / "trials.csv"
    lines = ["num_train,deg,trial,seed,solver,loss_Y,regret_fstar,regret_Y"]
    for rec in sorted(records, key=lambda r: (r["num_train"], r["deg"], r["trial"])):
        for key in SOLVER_KEYS:
            m = rec["metrics"][key]
            lines.append(
                f"{rec['num_train']},{rec['deg']},{rec['trial']},{rec['seed']},"
                f"{key},{m['loss_Y']:.6f},{m['regret_fstar']:.6f},{m['regret_Y']:.6f}"
            )
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def print_medians(by_size):
    for num_train in SIZES:
        for metric, metric_label in METRICS:
            print(f"\n{metric_label} -- training set size {num_train} "
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

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    by_size = collect(records)
    print_medians(by_size)
    write_csv(records, outdir)
    for num_train in SIZES:
        plot_size(by_size, num_train, outdir, show=args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())