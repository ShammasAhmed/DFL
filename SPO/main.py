"""
Entry point for the 5x5 grid shortest-path DFL experiments.

Two experiments are available, sharing the same optmodel and solver set:
  - run_sweep()    : NUM_TRIALS trials per DGP degree, drawn as a grouped boxplot
                     (RegretExperiment + RegretBoxPlot)
  - run_contexts() : train once, face num_contexts test contexts, print a table of
                     Decision Loss / Regret vs f* / Regret vs Y (ContextExperiment)

Solvers compared:
  - GBMTwoStage  : prediction-focused (two-stage) HistGradientBoosting baseline
  - LASSOTwoStage: prediction-focused (two-stage) least-squares LASSO baseline
  - LinearSPOPlus : decision-focused linear model trained with PyEPO's SPO+ loss

Pick which to run in the __main__ block at the bottom.
"""
from pyepo.model.grb import shortestPathModel

from solvers import GBMTwoStage, LASSOTwoStage, LinearSPOPlus
from experiments import RegretExperiment, ContextExperiment, HistogramExperiment
from plots import RegretBoxPlot
from sweep import RNG_SEED, SERIES

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


def run_sweep(NUM_TRIALS=50, degrees=(1, 2, 4, 6, 8), num_test=1000):
    """Degree sweep of per-trial test regret, saved/shown as a grouped boxplot."""
    experiment = RegretExperiment(
        optmodel=optmodel,
        solvers=SOLVERS,
        P=P,
        h=h,
        num_train=NUM_TRAIN,
        num_test=num_test,
        NUM_TRIALS=NUM_TRIALS,
        degrees=degrees,
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


def run_contexts(deg=4, num_contexts=200, shared_models=True, rng_seed=RNG_SEED):
    """Train once, face num_contexts test contexts, print the averaged metric table."""
    experiment = ContextExperiment(
        optmodel=optmodel,
        solvers=SOLVERS,
        deg=deg,
        num_contexts=num_contexts,
        shared_models=shared_models,
        P=P,
        h=h,
        num_train=NUM_TRAIN,
        rng_seed=rng_seed,
    )
    return experiment.print_table()

def run_histogram(deg=4, NUM_TRIALS=5, rng_seed=RNG_SEED):
    """Vary training sets for a single fixed context and plot path distribution histograms."""
    experiment = HistogramExperiment(
        optmodel=optmodel,  # <-- Pass the real optmodel here
        solvers=SOLVERS,
        deg=deg,
        NUM_TRIALS=NUM_TRIALS,
        P=P,
        h=h,
        num_train=NUM_TRAIN,
        rng_seed=rng_seed,
    )
    
    print(f"Building path matrix for grid {GRID}...")
    experiment.compute_paths(grid=GRID)
    
    print(f"Running histogram experiment over {NUM_TRIALS} independent training trials...")
    experiment.run(grid=GRID)
    
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
