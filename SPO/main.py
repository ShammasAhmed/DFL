"""
Entry point for the 5x5 grid shortest-path DFL experiments.

Three experiments are available, sharing the same optmodel and solver set:
  - run_sweep()    : NUM_TRIALS trials per DGP degree, drawn as a grouped boxplot
                     (RegretExperiment + RegretBoxPlot)
  - run_contexts() : train once, face num_contexts test contexts, print a table of
                     the sweep.SHOW_METRICS columns (ContextExperiment)
  - run_histogram(): vary the training set against one fixed context, plot which
                     paths each solver picks (HistogramExperiment)

The DGP is passed in rather than baked in: an experiment only sees
`gen(n, seed) -> Sample`. Which metrics a run can report follows from whether that
generator supplies f*, so run_sweep -- which never needs it -- asks for a cheaper
generator that skips it.

Two generators, one cost model. gen_for wraps PyEPO's genData, whose B is drawn from
the sample seed; the sweeps use it, since each of their trials is a single draw split
into train and test and so shares a B within itself. fixed_dgp_gen_for pins B at
build time instead; run_histogram needs that, because it holds one context's f* fixed
while varying the training draw, and the two are only comparable under one B.

Solvers compared:
  - GBMTwoStage  : prediction-focused (two-stage) HistGradientBoosting baseline
  - LASSOTwoStage: prediction-focused (two-stage) least-squares LASSO baseline
  - LinearSPOPlus : decision-focused linear model trained with PyEPO's SPO+ loss

Pick which to run in the __main__ block at the bottom.
"""
from pyepo.model.grb import shortestPathModel

from datagen import shortest_path_gen, numpy_shortest_path_gen
from solvers import GBMTwoStage, LASSOTwoStage, LinearSPOPlus
from experiments import RegretExperiment, ContextExperiment, HistogramExperiment
from plots import RegretBoxPlot, plot_noise_discount, plot_regret_from_csv
from sweep import (RNG_SEED, SERIES, SHOW_METRICS,
                   HIST_DEG, HIST_NUM_TRIALS, HIST_DGP_SEED, HIST_CONTEXT_SEED,
                   HIST_RANK_GAPS, HIST_RANK_TARGET, HIST_CONTEXT_POOL,
                   HIST_DRAW, hist_seed_for, parse_gap_spec)

# Shared configuration ------------------------------------------------------- #
GRID = (5, 5)
P = 5
h = 0.5
NUM_TRAIN = 100

# (key, solver_cls) for the experiments. The matching (key, label, color) plot
# series live in sweep.py, so the Slurm sweep and these local runs share them.
SOLVERS = [
    ("gbm", GBMTwoStage),
    ("lasso", LASSOTwoStage),
    ("spo", LinearSPOPlus),
]

optmodel = shortestPathModel(grid=GRID)


def gen_for(deg, noise_width=h, with_fstar=True):
    """The DGP generator for one polynomial degree, at this run's shared config.

    noise_width defaults to the module-level h; the sweep passes it explicitly so the
    noise half-width can be set per run from the sbatch.
    """
    return shortest_path_gen(GRID, P, noise_width, deg, with_fstar=with_fstar)


def fixed_dgp_gen_for(deg, dgp_seed=HIST_DGP_SEED, with_fstar=True):
    """
    The same DGP, same cost model, but with the ground truth B pinned by dgp_seed
    instead of redrawn from every sample seed.

    HistogramExperiment needs this: it compares decisions made by models trained on
    many different draws against a single context's f*, which is only meaningful if
    all of them share one B. The sweeps do not -- each of their trials is a single
    draw split into train and test, so B is shared within a trial by construction --
    and they stay on gen_for, whose numbers are already in results/.
    """
    return numpy_shortest_path_gen(GRID, P, h, deg, dgp_seed=dgp_seed,
                                   with_fstar=with_fstar)


def run_sweep(NUM_TRIALS=50, degrees=(1, 2, 4, 6, 8), num_test=1000):
    """Degree sweep of per-trial test regret, saved/shown as a grouped boxplot."""
    # Scored by pyepo.metric.regret, which never looks at f*, so don't pay to draw it.
    groups = [(deg, gen_for(deg, with_fstar=False)) for deg in degrees]

    experiment = RegretExperiment(
        optmodel=optmodel,
        solvers=SOLVERS,
        groups=groups,
        num_train=NUM_TRAIN,
        num_test=num_test,
        NUM_TRIALS=NUM_TRIALS,
        rng_seed=RNG_SEED,
    )
    results = experiment.run()

    plotter = RegretBoxPlot(
        groups=list(degrees),
        series=SERIES,
        xlabel="Polynomial degree of DGP",
        ylabel="Test regret (%)",
        title=f"Two-stage GBM vs LASSO vs SPO+ linear over {NUM_TRIALS} trials "
              f"(5x5 grid shortest path)",
    )
    plotter.plot(results)
    plotter.save("regret_boxplot.png")
    print("Saved boxplot to regret_boxplot.png")
    plotter.show()
    return results


def run_contexts(deg=4, num_contexts=200, shared_models=True, rng_seed=RNG_SEED,
                 metrics=SHOW_METRICS):
    """
    Train once, face num_contexts test contexts, print the pooled metric table.

    `metrics` picks the columns; it defaults to sweep.SHOW_METRICS, so editing that
    one tuple moves this table and context_aggregate.py's boxplot panels together.
    Every metric the generator supports is computed regardless -- the selection only
    decides what gets printed.
    """
    experiment = ContextExperiment(
        optmodel=optmodel,
        solvers=SOLVERS,
        gen=gen_for(deg),
        num_contexts=num_contexts,
        shared_models=shared_models,
        num_train=NUM_TRAIN,
        metrics=metrics,
        rng_seed=rng_seed,
    )
    return experiment.print_table()

def histogram_experiment(deg=HIST_DEG, num_train=NUM_TRAIN,
                         NUM_TRIALS=HIST_NUM_TRIALS, rank_gaps=None, rank_target=None):
    """
    The configured HistogramExperiment, built the same way here and on the cluster.

    histogram_trial.py calls this for its one trial and main's run_histogram calls it
    for all of them, so a local run and the Slurm array face the same fixed context and
    train on the same draws.

    `rank_gaps` / `rank_target` shape the fixed context's sorted cost curve (see
    HistogramExperiment); pass exactly one to try a shape out locally. Each accepts the
    same JSON string the sbatch uses or the parsed tuples; both None uses the sweep
    defaults, which is what the array runs -- changing what the cluster runs means editing
    sweep.py (or the sbatch override), since all tasks must agree on the context.
    """
    if isinstance(rank_gaps, str):
        rank_gaps = parse_gap_spec(rank_gaps)
    if isinstance(rank_target, str):
        rank_target = parse_gap_spec(rank_target)
    if rank_gaps is None and rank_target is None:
        rank_gaps, rank_target = HIST_RANK_GAPS, HIST_RANK_TARGET
    return HistogramExperiment(
        optmodel=optmodel,
        solvers=SOLVERS,
        gen=fixed_dgp_gen_for(deg),
        NUM_TRIALS=NUM_TRIALS,
        num_train=num_train,
        draw_size=HIST_DRAW,
        context_seed=HIST_CONTEXT_SEED,
        rank_gaps=rank_gaps,
        rank_target=rank_target,
        context_pool=HIST_CONTEXT_POOL,
        seed_fn=hist_seed_for,
    )


def preview_context(rank_gaps=None, rank_target=None, deg=HIST_DEG):
    """
    Select the fixed context for a shape spec and print its cost curve -- no training.

    The pre-flight for a cluster run: paste the exact JSON you'll put in the sbatch's
    RANK_GAPS / RANK_TARGET here first (both accept that string). A hard spec that finds
    nothing fails every array task, so confirm it selects and lands the gaps you wanted
    before submitting. Seconds to run, and needs no Gurobi -- it never trains a model.
    """
    try:
        experiment = histogram_experiment(deg=deg, NUM_TRIALS=1,
                                          rank_gaps=rank_gaps, rank_target=rank_target)
    except ValueError as e:
        print(f"no context selected: {e}")
        return None

    sorted_costs = experiment.true_path_costs[experiment.sorted_indices]
    opt = sorted_costs[0]
    print(f"context #{experiment.context_index} of {experiment.context_pool} "
          f"(deg={deg}), {len(sorted_costs)} paths")
    print("rank      cost   % above opt")
    for rank, c in enumerate(sorted_costs):
        print(f"{rank:>4}  {c:>9.4f}   {100 * (c - opt) / opt:>8.4f}%")
    print("requested gaps achieved:")
    for lo, hi, g in experiment.rank_gaps_achieved:
        print(f"  rank {lo}->{hi}: {g:.4f}%")
    return experiment


def run_histogram(deg=HIST_DEG, num_train=NUM_TRAIN, NUM_TRIALS=5,
                  rank_gaps=None, rank_target=None):
    """
    Vary training sets for a single fixed context and plot path selection histograms.

    Runs every trial in this process, so keep NUM_TRIALS small: the full
    HIST_NUM_TRIALS x HIST_SIZES grid is what the Slurm array is for
    (bash slurm/run_histogram.sh).
    """
    experiment = histogram_experiment(deg=deg, num_train=num_train,
                                      NUM_TRIALS=NUM_TRIALS, rank_gaps=rank_gaps,
                                      rank_target=rank_target)
    gaps = ", ".join(f"rank {lo}->{hi}: {g:.4f}%"
                     for lo, hi, g in experiment.rank_gaps_achieved)
    print(f"Fixed context #{experiment.context_index} ({gaps})")

    print(f"Running histogram experiment over {NUM_TRIALS} independent training "
          f"trials (n={num_train})...")
    experiment.run()

    print("Generating visualizations...")
    experiment.plot_histogram(subtitle=f", n={num_train}")

    return experiment.results


if __name__ == "__main__":
    # run_contexts(deg=1, num_contexts=1000)
    # run_sweep(NUM_TRIALS=25)

    # Redraw figures straight from a trials CSV -- no training, no Gurobi.
    # plot_noise_discount("trials_h05.csv", suptitle="Noise discount (5x5 grid shortest path, noise h=0.5)")
    # plot_regret_from_csv("trials_h05.csv")

    # Pre-flight a cluster shape locally -- no training, seconds to run. Paste the same
    # JSON you'll put in slurm/histogram_trials.sbatch; set one spec, leave the other out.
    # preview_context(rank_gaps='[[0,1,0.0,0.1],[4,5,15,null]]')
    # preview_context(rank_target='[[0,1,0.05],[4,5,20,2]]')

    # A small end-to-end run (trains models; needs Gurobi). Same spec argument as above.
    # The full 500-trial x {100, 1000} grid trains 3000 models -- that belongs on the
    # cluster: bash slurm/run_histogram.sh
    run_histogram(deg=4, NUM_TRIALS=25, rank_target='[[0,1,0.05],[4,5,20,2]]')
