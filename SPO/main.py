"""
Entry point for the 5x5 grid shortest-path DFL experiments.

Three experiments are available, sharing the same optmodel and solver set:
  - run_sweep()    : NUM_TRIALS trials per DGP degree, drawn as a grouped boxplot
                     (RegretExperiment + RegretBoxPlot)
  - run_contexts() : train once, face num_contexts test contexts, print a table of
                     the sweep.SHOW_METRICS columns (ContextExperiment)
  - run_histogram(): vary the training set against one fixed context, plot which
                     paths each solver picks (HistogramExperiment)

The DGP is passed in rather than baked in: gen_for(deg) builds the generator, and an
experiment only sees `gen(n, seed) -> Sample`. Which metrics a run can report follows
from whether that generator supplies f*, so run_sweep -- which never needs it -- asks
for a cheaper generator that skips it.

Solvers compared:
  - GBMTwoStage  : prediction-focused (two-stage) HistGradientBoosting baseline
  - LASSOTwoStage: prediction-focused (two-stage) least-squares LASSO baseline
  - LinearSPOPlus : decision-focused linear model trained with PyEPO's SPO+ loss

Pick which to run in the __main__ block at the bottom.
"""
from pyepo.model.grb import shortestPathModel

from datagen import shortest_path_gen
from solvers import GBMTwoStage, LASSOTwoStage, LinearSPOPlus
from experiments import RegretExperiment, ContextExperiment, HistogramExperiment
from plots import RegretBoxPlot
from sweep import RNG_SEED, SERIES, SHOW_METRICS

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
    one tuple moves this table and aggregate.py's boxplot panels together. Every
    metric the generator supports is computed regardless -- the selection only
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

def run_histogram(deg=4, NUM_TRIALS=5, rng_seed=RNG_SEED):
    """Vary training sets for a single fixed context and plot path distribution histograms."""
    experiment = HistogramExperiment(
        optmodel=optmodel,
        solvers=SOLVERS,
        gen=gen_for(deg),
        NUM_TRIALS=NUM_TRIALS,
        num_train=NUM_TRAIN,
        rng_seed=rng_seed,
    )

    print(f"Running histogram experiment over {NUM_TRIALS} independent training trials...")
    experiment.run()

    print("Generating visualizations...")
    experiment.plot_histogram()

    return experiment.results


if __name__ == "__main__":
    run_contexts(deg=8, num_contexts=1000)
    # run_sweep(NUM_TRIALS=25)
    
    # # --- Execute the New Histogram Experiment ---
    # print("\n" + "=" * 50)
    # print("Running Histogram Experiment...")
    # print("=" * 50)
    # results = run_histogram(deg=4, NUM_TRIALS=50, rng_seed=RNG_SEED)
    # print("Done!")
