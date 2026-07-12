"""
One trial of the (degree x training-set-size) sweep.

A trial is a single ContextExperiment: train the solvers once on a fresh DGP draw
at (deg, num_train, seed), face NUM_CONTEXTS test contexts, and pool the metrics --
so each trial collapses to one number per (solver, metric). That scalar is the thing
the boxplots take their distribution over.

Every metric the generator supports is written to the JSON, not merely the ones
sweep.SHOW_METRICS currently displays. The columns are then a plotting-time choice
(aggregate.py --metrics ...), so changing your mind about them never costs a rerun
of the array.

Invoked once per Slurm array task by slurm/trials.sbatch; aggregate.py reads the
JSONs back and draws the plots.

    python run_trial.py --deg 4 --num-train 100 --trial 0
"""
import argparse
import json
import os
import sys

from main import optmodel, SOLVERS, gen_for
from experiments import ContextExperiment, metrics_for
from sweep import (DEGREES, SIZES, NUM_TRIALS, NUM_CONTEXTS, RESULT_DIR,
                   SHOW_METRICS, seed_for, result_path)


def run_trial(deg, num_train, trial, num_contexts=NUM_CONTEXTS, outdir=RESULT_DIR):
    """Run one trial and write its record to JSON. Returns the record."""
    seed = seed_for(deg, trial)
    experiment = ContextExperiment(
        optmodel=optmodel,
        solvers=SOLVERS,
        gen=gen_for(deg),
        num_contexts=num_contexts,
        shared_models=True,
        num_train=num_train,
        rng_seed=seed,
    )
    table = experiment.run()

    record = {
        "deg": deg,
        "num_train": num_train,
        "trial": trial,
        "seed": seed,
        "num_contexts": num_contexts,
        # Persist everything computed; aggregate.py selects columns at plot time.
        "metrics": {key: dict(table[key]) for key, _ in SOLVERS},
    }

    path = result_path(outdir, deg, num_train, trial)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename: a task killed mid-write leaves no half-parsed JSON for
    # aggregate.py to trip over.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2))
    os.replace(tmp, path)
    return record


def is_reusable(path, gen):
    """
    Whether an existing trial JSON can stand in for recomputing this trial.

    It can only if it already holds every metric a run today would compute. This is
    what makes a resubmitted array safe: the resume path skips any trial whose JSON
    exists, so a results/ directory written before a metric existed would otherwise
    survive the resubmission untouched and go unnoticed until aggregation refused the
    missing column -- two jobs and several hours later, with the array reporting
    success. A stale or unreadable record is recomputed instead.

    Inputs:
        path (Path): Where this trial's JSON would live
        gen (callable): The generator this trial would run on

    Returns:
        reusable (bool): True if the cached record is complete and current
    """
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return False  # missing, truncated, or not JSON -- recompute
    expected = set(metrics_for(gen(1, 0).fstar is not None))
    stored = record.get("metrics", {})
    return all(expected <= set(stored.get(key, {})) for key, _ in SOLVERS)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deg", type=int, required=True, choices=DEGREES)
    parser.add_argument("--num-train", type=int, required=True, choices=SIZES)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--num-contexts", type=int, default=NUM_CONTEXTS)
    parser.add_argument("--outdir", default=str(RESULT_DIR))
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if this trial's JSON already exists and "
                             "holds every current metric (default: skip, so a "
                             "resubmitted array resumes).")
    args = parser.parse_args(argv)
    if not 0 <= args.trial < NUM_TRIALS:
        parser.error(f"--trial must be in [0, {NUM_TRIALS})")
    return args


def main(argv=None):
    args = parse_args(argv)
    path = result_path(args.outdir, args.deg, args.num_train, args.trial)
    if path.exists() and not args.overwrite:
        if is_reusable(path, gen_for(args.deg)):
            print(f"{path} exists, skipping (pass --overwrite to recompute)")
            return 0
        print(f"{path} is unreadable or predates the current metric set, recomputing")

    record = run_trial(args.deg, args.num_train, args.trial,
                       num_contexts=args.num_contexts, outdir=args.outdir)
    # The JSON holds every metric; the log line shows the selected ones.
    parts = [
        f"{key} " + " ".join(
            f"{metric}={m[metric]:.4f}" for metric in SHOW_METRICS if metric in m)
        for key, m in record["metrics"].items()
    ]
    print(f"deg={record['deg']} n={record['num_train']} trial={record['trial']} "
          f"seed={record['seed']}: " + "  |  ".join(parts))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())