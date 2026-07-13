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

import numpy as np

from main import histogram_experiment, SOLVERS
from sweep import (HIST_DEG, HIST_SIZES, HIST_NUM_TRIALS, HIST_RESULT_DIR,
                   HIST_DGP_SEED, HIST_CONTEXT_SEED, HIST_CONTEXT_MARGIN,
                   HIST_CONTEXT_MARGIN_MAX, hist_seed_for, hist_result_path)

SOLVER_KEYS = [key for key, _ in SOLVERS]


def sorted_costs_of(experiment):
    """The true path costs of the experiment's fixed context, cheapest first."""
    return [float(c) for c in
            experiment.true_path_costs[experiment.sorted_indices]]


def run_trial(experiment, num_train, trial, deg=HIST_DEG, outdir=HIST_RESULT_DIR):
    """Run one trial and write its record to JSON. Returns the record."""
    chosen = experiment.run_trial(trial)

    record = {
        "deg": deg,
        "num_train": num_train,
        "trial": trial,
        "seed": hist_seed_for(trial),
        "dgp_seed": HIST_DGP_SEED,
        "context_seed": HIST_CONTEXT_SEED,
        # Which candidate context was selected, and by how much its best path beats the
        # second-best. Both follow from HIST_CONTEXT_MARGIN, so they pin down what this
        # trial was actually graded against.
        "context_index": experiment.context_index,
        "context_margin": experiment.margin,
        # The blue curve. Identical in every record by construction; stored per record
        # so the aggregator can draw it without PyEPO, and cross-check it while it does.
        "sorted_costs": sorted_costs_of(experiment),
        "chosen": chosen,
    }

    path = hist_result_path(outdir, num_train, trial)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename: a task killed mid-write leaves no half-parsed JSON behind.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2))
    os.replace(tmp, path)
    return record


def is_reusable(path, experiment):
    """
    Whether an existing trial JSON can stand in for recomputing this trial.

    It can only if it holds a decision for every solver we run today AND was graded
    against the context we would use today. The second check is what makes changing
    HIST_CONTEXT_MARGIN safe: a new margin picks a different context, and without it a
    resubmitted array would skip every trial whose JSON already exists and hand you a
    histogram of the OLD context's ranks under a new context's cost curve -- wrong, and
    silently so. A stale, unreadable, or differently-graded record is recomputed.
    """
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return False  # missing, truncated, or not JSON -- recompute
    chosen = record.get("chosen", {})
    if not all(key in chosen and "rank" in chosen[key] for key in SOLVER_KEYS):
        return False
    stored = record.get("sorted_costs")
    current = sorted_costs_of(experiment)
    return (stored is not None and len(stored) == len(current)
            and np.allclose(stored, current, rtol=1e-6))


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

    # Built before the reuse check, since deciding whether a cached record is still
    # valid means knowing which context we would grade against today.
    experiment = histogram_experiment(deg=args.deg, num_train=args.num_train,
                                      NUM_TRIALS=HIST_NUM_TRIALS)
    print(f"context #{experiment.context_index}: best path beats second-best by "
          f"{experiment.margin:.4f}% (window "
          f"[{HIST_CONTEXT_MARGIN}%, {HIST_CONTEXT_MARGIN_MAX}%])")

    path = hist_result_path(args.outdir, args.num_train, args.trial)
    if path.exists() and not args.overwrite:
        if is_reusable(path, experiment):
            print(f"{path} exists, skipping (pass --overwrite to recompute)")
            return 0
        print(f"{path} is unreadable, predates the current solver set, or was graded "
              f"against a different context, recomputing")

    record = run_trial(experiment, args.num_train, args.trial, deg=args.deg,
                       outdir=args.outdir)

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
