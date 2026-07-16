"""
One trial of the (degree x training-set-size) sweep.

A trial is a single ContextExperiment: train the solvers once on a fresh DGP draw
at (deg, num_train, seed), face NUM_CONTEXTS test contexts, and pool the metrics --
so each trial collapses to one number per (solver, metric). That scalar is the thing
the boxplots take their distribution over.

Every metric the generator supports is written to the JSON, not merely the ones
sweep.SHOW_METRICS currently displays. The columns are then a plotting-time choice
(context_aggregate.py --metrics ...), so changing your mind about them never costs a rerun
of the array.

Invoked once per Slurm array task by slurm/context_trials.sbatch; context_aggregate.py reads the
JSONs back and draws the plots.

    python context_trial.py --deg 4 --num-train 100 --trial 0
"""
import argparse
import json
import os
import sys

from main import optmodel, SOLVERS, gen_for
from experiments import (ContextExperiment, metrics_for, terms_for,
                         SOLVER_TERMS, BENCHMARK_TERMS)
from sweep import (DEGREES, SIZES, NUM_TRIALS, NUM_CONTEXTS, NOISE_WIDTH,
                   RESULT_DIR, SHOW_METRICS, seed_for, result_path)


def run_trial(deg, num_train, trial, noise_width=NOISE_WIDTH,
              num_contexts=NUM_CONTEXTS, outdir=RESULT_DIR):
    """Run one trial and write its record to JSON. Returns the record."""
    seed = seed_for(deg, trial)
    experiment = ContextExperiment(
        optmodel=optmodel,
        solvers=SOLVERS,
        gen=gen_for(deg, noise_width),
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
        "h": noise_width,
        "num_contexts": num_contexts,
        # Persist everything computed; context_aggregate.py selects columns at plot time.
        "metrics": {key: dict(table[key]) for key, _ in SOLVERS},
        # And the raw cost sums the metrics are ratios of, so that renormalizing them --
        # or dropping the normalization -- is arithmetic on the CSV rather than a rerun.
        "terms": {key: dict(experiment.terms[key]) for key, _ in SOLVERS},
        "totals": dict(experiment.totals),
    }

    path = result_path(outdir, deg, num_train, trial, noise_width)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename: a task killed mid-write leaves no half-parsed JSON for
    # context_aggregate.py to trip over.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2))
    os.replace(tmp, path)
    return record


def is_reusable(path, gen):
    """
    Whether an existing trial JSON can stand in for recomputing this trial.

    It can only if it already holds every metric AND every raw cost sum a run today
    would compute. This is what makes a resubmitted array safe: the resume path skips
    any trial whose JSON exists, so a results/ directory written before a metric existed
    would otherwise survive the resubmission untouched and go unnoticed until aggregation
    refused the missing column -- two jobs and several hours later, with the array
    reporting success. A stale or unreadable record is recomputed instead.

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
    has_fstar = gen(1, 0).fstar is not None
    expected = set(metrics_for(has_fstar))
    expected_terms = set(terms_for(has_fstar, SOLVER_TERMS))
    expected_totals = set(terms_for(has_fstar, BENCHMARK_TERMS))

    metrics = record.get("metrics", {})
    terms = record.get("terms", {})
    return (expected_totals <= set(record.get("totals", {}))
            and all(expected <= set(metrics.get(key, {}))
                    and expected_terms <= set(terms.get(key, {}))
                    for key, _ in SOLVERS))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deg", type=int, required=True, choices=DEGREES)
    parser.add_argument("--num-train", type=int, required=True, choices=SIZES)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--noise-width", type=float, default=NOISE_WIDTH,
                        help="Multiplicative noise half-width h (epsilon ~ U[1-h, 1+h]). "
                             "Recorded in the JSON and in the filename, so trials at "
                             f"different h coexist in one results dir (default: {NOISE_WIDTH}).")
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
    path = result_path(args.outdir, args.deg, args.num_train, args.trial,
                       args.noise_width)
    if path.exists() and not args.overwrite:
        if is_reusable(path, gen_for(args.deg, args.noise_width)):
            print(f"{path} exists, skipping (pass --overwrite to recompute)")
            return 0
        print(f"{path} is unreadable or predates the current metric set, recomputing")

    record = run_trial(args.deg, args.num_train, args.trial,
                       noise_width=args.noise_width,
                       num_contexts=args.num_contexts, outdir=args.outdir)
    # The JSON holds every metric; the log line shows the selected ones.
    parts = [
        f"{key} " + " ".join(
            f"{metric}={m[metric]:.4f}" for metric in SHOW_METRICS if metric in m)
        for key, m in record["metrics"].items()
    ]
    print(f"deg={record['deg']} n={record['num_train']} trial={record['trial']} "
          f"h={record['h']} seed={record['seed']}: " + "  |  ".join(parts))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())