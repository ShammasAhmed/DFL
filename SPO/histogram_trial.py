"""
One trial of the histogram experiment: one training draw, at one training-set size.

A trial trains all three solvers on its own draw, shows each of them the single fixed
test context, and records which path each one picked -- both the path index and its
rank among all paths by true expected cost, rank 0 being the true optimal path. Those
ranks are what histogram_aggregate.py counts into the histogram.

Every trial recomputes the fixed context and the path ranking itself. That is safe
precisely because the generator's ground truth B is pinned (main.fixed_dgp_gen_for):
the context, the f* ranking the paths, and every trial's training data all come from
one DGP, so 1000 tasks that never speak to each other still agree on what they are
being scored against. Each record carries the resulting cost curve, and the aggregator
checks they match -- a task that somehow faced a different context gets caught rather
than averaged in.

Invoked once per Slurm array task by slurm/histogram_trials.sbatch.

    python histogram_trial.py --num-train 100 --trial 0
"""
import argparse
import json
import os
import sys

from main import histogram_experiment, SOLVERS
from sweep import (HIST_DEG, HIST_SIZES, HIST_NUM_TRIALS, HIST_RESULT_DIR,
                   HIST_DGP_SEED, HIST_CONTEXT_SEED, hist_seed_for,
                   hist_result_path)

SOLVER_KEYS = [key for key, _ in SOLVERS]


def run_trial(num_train, trial, deg=HIST_DEG, outdir=HIST_RESULT_DIR):
    """Run one trial and write its record to JSON. Returns the record."""
    experiment = histogram_experiment(deg=deg, num_train=num_train,
                                      NUM_TRIALS=HIST_NUM_TRIALS)
    chosen = experiment.run_trial(trial)

    sorted_costs = experiment.true_path_costs[experiment.sorted_indices]
    record = {
        "deg": deg,
        "num_train": num_train,
        "trial": trial,
        "seed": hist_seed_for(trial),
        "dgp_seed": HIST_DGP_SEED,
        "context_seed": HIST_CONTEXT_SEED,
        # The blue curve. Identical in every record by construction; stored per record
        # so the aggregator can draw it without PyEPO, and cross-check it while it does.
        "sorted_costs": [float(c) for c in sorted_costs],
        "chosen": chosen,
    }

    path = hist_result_path(outdir, num_train, trial)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename: a task killed mid-write leaves no half-parsed JSON behind.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2))
    os.replace(tmp, path)
    return record


def is_reusable(path):
    """
    Whether an existing trial JSON can stand in for recomputing this trial.

    It can only if it holds a decision for every solver we run today, so a results
    directory written before a solver was added is recomputed rather than silently
    kept and later aggregated into a histogram missing a bar.
    """
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return False  # missing, truncated, or not JSON -- recompute
    chosen = record.get("chosen", {})
    return (bool(record.get("sorted_costs"))
            and all(key in chosen and "rank" in chosen[key] for key in SOLVER_KEYS))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-train", type=int, required=True, choices=HIST_SIZES)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--deg", type=int, default=HIST_DEG)
    parser.add_argument("--outdir", default=str(HIST_RESULT_DIR))
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if this trial's JSON already exists "
                             "(default: skip, so a resubmitted array resumes).")
    args = parser.parse_args(argv)
    if not 0 <= args.trial < HIST_NUM_TRIALS:
        parser.error(f"--trial must be in [0, {HIST_NUM_TRIALS})")
    return args


def main(argv=None):
    args = parse_args(argv)
    path = hist_result_path(args.outdir, args.num_train, args.trial)
    if path.exists() and not args.overwrite:
        if is_reusable(path):
            print(f"{path} exists, skipping (pass --overwrite to recompute)")
            return 0
        print(f"{path} is unreadable or predates the current solver set, recomputing")

    record = run_trial(args.num_train, args.trial, deg=args.deg, outdir=args.outdir)

    picks = "  |  ".join(
        f"{key} path={record['chosen'][key]['path']} "
        f"rank={record['chosen'][key]['rank']}"
        for key in SOLVER_KEYS)
    print(f"n={record['num_train']} trial={record['trial']} "
          f"seed={record['seed']}: {picks}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
