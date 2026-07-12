"""
One trial of the (degree x training-set-size) sweep.

A trial is a single ContextExperiment: train the solvers once on a fresh DGP draw
at (deg, num_train, seed), face NUM_CONTEXTS test contexts, and pool the metrics --
so each trial collapses to one Regret vs f* and one Regret vs Y number per solver.
That scalar is the thing the boxplots take their distribution over.

Invoked once per Slurm array task by slurm/trials.sbatch; aggregate.py reads the
JSONs back and draws the plots.

    python run_trial.py --deg 4 --num-train 100 --trial 0
"""
import argparse
import json
import os
import sys

from main import optmodel, SOLVERS, P, h
from experiments import ContextExperiment
from sweep import (DEGREES, SIZES, NUM_TRIALS, NUM_CONTEXTS, RESULT_DIR,
                   seed_for, result_path)


def run_trial(deg, num_train, trial, num_contexts=NUM_CONTEXTS, outdir=RESULT_DIR):
    """Run one trial and write its record to JSON. Returns the record."""
    seed = seed_for(deg, trial)
    experiment = ContextExperiment(
        optmodel=optmodel,
        solvers=SOLVERS,
        deg=deg,
        num_contexts=num_contexts,
        shared_models=True,
        P=P,
        h=h,
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
        "metrics": {
            key: {
                "loss_Y": table[key]["loss_Y"],
                "regret_fstar": table[key]["regret_fstar"],
                "regret_Y": table[key]["regret_Y"],
            }
            for key, _ in SOLVERS
        },
    }

    path = result_path(outdir, deg, num_train, trial)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename: a task killed mid-write leaves no half-parsed JSON for
    # aggregate.py to trip over.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2))
    os.replace(tmp, path)
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deg", type=int, required=True, choices=DEGREES)
    parser.add_argument("--num-train", type=int, required=True, choices=SIZES)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--num-contexts", type=int, default=NUM_CONTEXTS)
    parser.add_argument("--outdir", default=str(RESULT_DIR))
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if this trial's JSON already exists "
                             "(default: skip, so a resubmitted array resumes).")
    args = parser.parse_args(argv)
    if not 0 <= args.trial < NUM_TRIALS:
        parser.error(f"--trial must be in [0, {NUM_TRIALS})")
    return args


def main(argv=None):
    args = parse_args(argv)
    path = result_path(args.outdir, args.deg, args.num_train, args.trial)
    if path.exists() and not args.overwrite:
        print(f"{path} exists, skipping (pass --overwrite to recompute)")
        return 0

    record = run_trial(args.deg, args.num_train, args.trial,
                       num_contexts=args.num_contexts, outdir=args.outdir)
    parts = [f"{key} f*={m['regret_fstar']:.4f}% Y={m['regret_Y']:.4f}%"
             for key, m in record["metrics"].items()]
    print(f"deg={record['deg']} n={record['num_train']} trial={record['trial']} "
          f"seed={record['seed']}: " + "  |  ".join(parts))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())