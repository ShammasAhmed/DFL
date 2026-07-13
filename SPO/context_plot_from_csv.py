"""
Redraw the sweep's regret boxplots straight from trials.csv, without needing the
per-trial JSONs (or any of the optimization stack) around.

Draws exactly what context_aggregate.py draws -- one figure per training-set size, one
panel per metric, every panel on a y-axis shared across both figures (see
plots.plot_regret_boxplots) -- just sourced from the CSV rather than the JSONs. The
metrics are selected by --metrics and default to sweep.SHOW_METRICS.

    python context_plot_from_csv.py [--csv trials.csv] [--outdir .] [--metrics ...] [--show]
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

from sweep import DEGREES, SIZES, SERIES, METRICS, SHOW_METRICS  # noqa: E402
from plots import plot_regret_boxplots  # noqa: E402

SOLVER_KEYS = [key for key, _, _ in SERIES]


def load_csv(path, metrics):
    """
    Read the tidy CSV into by_size[num_train][metric][deg][solver_key] -> list of
    trial values, the data[group][series_key] layout RegretBoxPlot consumes.

    Inputs:
        path (str): The tidy per-trial CSV written by context_aggregate.py
        metrics (list): Metric columns to pull out, as keys of sweep.METRICS

    Returns:
        by_size (dict): The nested layout described above

    Raises:
        SystemExit: The CSV has no usable rows, or lacks a requested metric column.
    """
    by_size = {
        n: {metric: {deg: defaultdict(list) for deg in DEGREES}
            for metric in metrics}
        for n in SIZES
    }
    n_rows = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        absent = [m for m in metrics if m not in (reader.fieldnames or [])]
        if absent:
            raise SystemExit(
                f"{path} has no column(s) {', '.join(absent)}; it holds "
                f"{', '.join(reader.fieldnames or [])}")
        for row in reader:
            n, deg, key = int(row["num_train"]), int(row["deg"]), row["solver"]
            if n not in by_size or deg not in DEGREES or key not in SOLVER_KEYS:
                continue  # a row outside the sweep grid we know how to plot
            for metric in metrics:
                if row[metric] != "":
                    by_size[n][metric][deg][key].append(float(row[metric]))
            n_rows += 1
    if not n_rows:
        raise SystemExit(f"No usable rows in {path}")
    return by_size


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="trials.csv",
                        help="Tidy per-trial CSV written by context_aggregate.py")
    parser.add_argument("--outdir", default=".", help="Where to write the PNGs")
    parser.add_argument("--metrics", default=",".join(SHOW_METRICS),
                        help="Comma-separated metrics to plot, one panel each "
                             f"(default: {','.join(SHOW_METRICS)}; "
                             f"known: {','.join(METRICS)})")
    parser.add_argument("--show", action="store_true", help="Display the figures")
    args = parser.parse_args(argv)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        print(f"Unknown metric(s): {', '.join(unknown)}. Known metrics are "
              f"{', '.join(METRICS)}.", file=sys.stderr)
        return 1

    by_size = load_csv(args.csv, metrics)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_regret_boxplots(
        by_size, SIZES, [(m, METRICS[m].label) for m in metrics], list(DEGREES),
        SERIES, outdir,
        xlabel="Polynomial degree of DGP",
        suptitle=lambda n: ("Pooled context regret per degree "
                            f"(5x5 grid shortest path, training set size = {n})"),
        show=args.show,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())