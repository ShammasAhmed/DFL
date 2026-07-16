"""
Collect the per-trial JSONs written by context_trial.py and draw the sweep's boxplots.

One figure per training-set size, one panel per selected metric, so everything
measured at that size sits in a single image. Within a panel, one group per DGP
degree and one box per solver, each box summarizing the NUM_TRIALS trials of that
cell. Every panel of both figures sits on one shared y-axis, so the sizes can be
compared by eye -- see plots.plot_regret_boxplots, which context_plot_from_csv.py
draws through too.

context_trial.py stores every metric it computed, so which panels get drawn is a choice
made here, not there: --metrics re-plots an existing sweep under a different
selection without recomputing anything. It defaults to sweep.SHOW_METRICS, which
main.py's printed table also follows.

Also writes a tidy CSV of every trial and prints a median table. Deliberately imports
no PyEPO/Gurobi, so it runs on a plain plotting node.

    python context_aggregate.py [--results results] [--outdir .] [--metrics ...] [--show]
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

from sweep import (DEGREES, SIZES, NUM_TRIALS, SERIES, METRICS,  # noqa: E402
                   SHOW_METRICS, NOISE_WIDTH, RESULT_DIR, result_path)
from plots import plot_regret_boxplots  # noqa: E402

SOLVER_KEYS = [key for key, _, _ in SERIES]


def load_records(results_dir, h=NOISE_WIDTH):
    """Read every trial JSON at noise width h. Returns (records, missing) as cells."""
    records = []
    missing = []
    for num_train in SIZES:
        for deg in DEGREES:
            for trial in range(NUM_TRIALS):
                path = result_path(results_dir, deg, num_train, trial, h)
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


# The raw cost sums beside the metrics, as (column, where it lives in the record, term).
# Writing sum <a, b> for the sum over a trial's contexts of <a, b>, and z*(c) for the
# path that minimizes c:
#
#   sum_Y_zmethod      <Y,  z*(f(X))>   what a method costs in practice, on this draw
#   sum_fstar_zmethod  <f*, z*(f(X))>   what it costs in expectation
#   sum_Y_zY           <Y,  z*(Y)>      an oracle told this draw's realized costs
#   sum_fstar_zY       <f*, z*(Y)>      what that oracle's path really costs
#   sum_fstar_zfstar   <f*, z*(f*)>     the best attainable: what every method aims at
#
# Every metric column is a ratio of differences of these (see sweep.METRICS), so the
# normalization is recoverable, replaceable, or droppable from the CSV alone. The first
# two vary by solver; the last three are the same for all three rows of a trial.
SUM_COLUMNS = [
    ("sum_Y_zmethod",     "terms",  "loss_Y"),
    ("sum_fstar_zmethod", "terms",  "fstar_at_hat"),
    ("sum_Y_zY",          "totals", "opt_Y"),
    ("sum_fstar_zY",      "totals", "fstar_at_star_Y"),
    ("sum_fstar_zfstar",  "totals", "opt_fstar"),
]


def write_csv(records, metrics, outdir):
    """
    Tidy one-row-per-(trial, solver) CSV, for whatever downstream stats you want.

    Carries every metric the records hold, not just the plotted selection, and the raw
    cost sums behind them -- the CSV is the archive, the panels are the view. A record
    missing a metric or a sum leaves that cell empty rather than failing, so a results/
    directory of mixed vintage still aggregates.
    """
    path = outdir / "trials.csv"
    columns = ["num_train", "deg", "trial", "seed", "h", "solver", "num_contexts",
               *metrics, *(name for name, _, _ in SUM_COLUMNS)]
    lines = [",".join(columns)]
    for rec in sorted(records, key=lambda r: (r["num_train"], r["deg"], r["trial"])):
        for key in SOLVER_KEYS:
            m = rec["metrics"][key]
            values = [f"{m[metric]:.6f}" if metric in m else ""
                      for metric in metrics]
            for _, where, term in SUM_COLUMNS:
                source = rec.get(where, {})
                if where == "terms":
                    source = source.get(key, {})
                value = source.get(term)
                values.append(f"{value:.6f}" if value is not None else "")
            lines.append(
                f"{rec['num_train']},{rec['deg']},{rec['trial']},{rec['seed']},"
                f"{rec.get('h', '')},{key},{rec.get('num_contexts', '')}," + ",".join(values)
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
                        help="Directory of per-trial JSONs from context_trial.py")
    parser.add_argument("--h", type=float, default=NOISE_WIDTH,
                        help="Noise half-width to read (files are named by h, so one "
                             f"results dir may hold several; default: {NOISE_WIDTH}).")
    parser.add_argument("--outdir", default=".",
                        help="Where to write the PNGs and CSV")
    parser.add_argument("--metrics", default=",".join(SHOW_METRICS),
                        help="Comma-separated metrics to plot, one panel each "
                             f"(default: {','.join(SHOW_METRICS)}; "
                             f"known: {','.join(METRICS)})")
    parser.add_argument("--show", action="store_true",
                        help="Display the figures (off by default; headless on a node)")
    args = parser.parse_args(argv)

    records, missing = load_records(args.results, args.h)
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
    plot_regret_boxplots(
        by_size, SIZES, [(m, METRICS[m].label) for m in requested], list(DEGREES),
        SERIES, outdir,
        xlabel="Polynomial degree of DGP",
        suptitle=lambda n: (f"Pooled context regret over {NUM_TRIALS} trials per degree "
                            f"(5x5 grid shortest path, training set size = {n})"),
        show=args.show,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())