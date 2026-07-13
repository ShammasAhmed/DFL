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
from plots import RegretBoxPlot
from sweep import (RNG_SEED, SERIES, SHOW_METRICS,
                   HIST_DEG, HIST_NUM_TRIALS, HIST_DGP_SEED, HIST_CONTEXT_SEED,
                   HIST_DRAW, hist_seed_for)

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


def gen_for(deg, with_fstar=True):
    """The DGP generator for one polynomial degree, at this run's shared config."""
    return shortest_path_gen(GRID, P, h, deg, with_fstar=with_fstar)


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
                         NUM_TRIALS=HIST_NUM_TRIALS):
    """
    The configured HistogramExperiment, built the same way here and on the cluster.

    histogram_trial.py calls this for its one trial and main's run_histogram calls it
    for all of them, so a local run and the Slurm array face the same fixed context and
    train on the same draws.
    """
    return HistogramExperiment(
        optmodel=optmodel,
        solvers=SOLVERS,
        gen=fixed_dgp_gen_for(deg),
        NUM_TRIALS=NUM_TRIALS,
        num_train=num_train,
        draw_size=HIST_DRAW,
        context_seed=HIST_CONTEXT_SEED,
        seed_fn=hist_seed_for,
    )


def run_histogram(deg=HIST_DEG, num_train=NUM_TRAIN, NUM_TRIALS=5):
    """
    Vary training sets for a single fixed context and plot path selection histograms.

    Runs every trial in this process, so keep NUM_TRIALS small: the full
    HIST_NUM_TRIALS x HIST_SIZES grid is what the Slurm array is for
    (bash slurm/run_histogram.sh).
    """
    experiment = histogram_experiment(deg=deg, num_train=num_train,
                                      NUM_TRIALS=NUM_TRIALS)

    print(f"Running histogram experiment over {NUM_TRIALS} independent training "
          f"trials (n={num_train})...")
    experiment.run()

    print("Generating visualizations...")
    experiment.plot_histogram(subtitle=f", n={num_train}")

    return experiment.results


if __name__ == "__main__":
    # run_contexts(deg=8, num_contexts=1000)
    # run_sweep(NUM_TRIALS=25)

    # A small local run. The full 500-trial x {100, 1000} grid trains 3000 models --
    # that one belongs on the cluster: bash slurm/run_histogram.sh
    run_histogram(NUM_TRIALS=10)
