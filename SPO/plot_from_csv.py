"""
Redraw the sweep's regret boxplots straight from trials.csv, without needing the
per-trial JSONs (or any of the optimization stack) around.

Same layout as aggregate.py -- one figure per training-set size, one panel per
selected metric (--metrics, defaulting to sweep.SHOW_METRICS) -- with two
differences:
  * the panels of a figure share one y-axis, so a box in one panel is directly
    comparable in height to a box in another. That is exactly the comparison
    regret_Y and regret_Y_lowvar are built for: they pool against the same
    denominator, so the shared scale is meaningful rather than coincidental.
  * a dotted line marks y = 0, separating positive from negative regret --
    regret_Y_lowvar routinely goes negative, since z*(Y) chases the noise and is
    beatable under f*.

    python plot_from_csv.py [--csv trials.csv] [--outdir .] [--metrics ...] [--show]
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sweep import DEGREES, SIZES, SERIES, METRICS, SHOW_METRICS  # noqa: E402
from plots import RegretBoxPlot  # noqa: E402

SOLVER_KEYS = [key for key, _, _ in SERIES]


def load_csv(path, metrics):
    """
    Read the tidy CSV into by_size[num_train][metric][deg][solver_key] -> list of
    trial values, the data[group][series_key] layout RegretBoxPlot consumes.

    Inputs:
        path (str): The tidy per-trial CSV written by aggregate.py
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


def plot_size(by_size, metrics, num_train, outdir, show=False):
    """Boxplot panels for one training size, sharing a y-axis, with a y=0 line."""
    # sharey ties the panels' limits together, so all autoscale to the union of their
    # data and the same box height means the same regret in any of them.
    fig, axes = plt.subplots(1, len(metrics), figsize=(9 * len(metrics), 6.5),
                             sharey=True, squeeze=False)

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
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1.2, zorder=0)
        # sharey blanks the later panels' tick labels; keep the ylabel, it differs.
        ax.set_ylabel(label)

    fig.suptitle(
        f"Pooled context regret per degree "
        f"(5x5 grid shortest path, training set size = {num_train})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    path = outdir / f"regret_boxplots_shared_n{num_train}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="trials.csv",
                        help="Tidy per-trial CSV written by aggregate.py")
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
    for num_train in SIZES:
        plot_size(by_size, metrics, num_train, outdir, show=args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())