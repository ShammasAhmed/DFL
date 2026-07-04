"""
Entry point for the 5x5 grid shortest-path DFL experiments.

Two experiments are available, sharing the same optmodel and solver set:
  - run_sweep()    : NUM_TRIALS trials per DGP degree, drawn as a grouped boxplot
                     (RegretExperiment + RegretBoxPlot)
  - run_contexts() : train once, face N_CONTEXTS test contexts, print a table of
                     Decision Loss / Regret vs f* / Regret vs Y (ContextExperiment)

Solvers compared:
  - GBM_twostage  : prediction-focused (two-stage) HistGradientBoosting baseline
  - LASSO_twostage: prediction-focused (two-stage) least-squares LASSO baseline
  - LinearSPOPlus : decision-focused linear model trained with PyEPO's SPO+ loss

Pick which to run in the __main__ block at the bottom.
"""
from pyepo.model.grb import shortestPathModel

from solvers import GBM_twostage, LASSO_twostage, LinearSPOPlus
from experiments import RegretExperiment, ContextExperiment, HistogramExperiment
from plots import RegretBoxPlot

# Shared configuration ------------------------------------------------------- #
GRID = (5, 5)
NUM_FEATURES = 5
NOISE_WIDTH = 0.5
NUM_TRAIN = 100
RNG_SEED = 143
TEST_RNG_SEED = 42

# (key, solver_cls) for the experiments; (key, label, color) for the plot.
SOLVERS = [
    ("gbm", GBM_twostage),
    ("lasso", LASSO_twostage),
    ("spo", LinearSPOPlus),
]
SERIES = [
    ("gbm", "2-stage GBM", "tab:blue"),
    ("lasso", "2-stage LASSO", "tab:green"),
    ("spo", "SPO+ linear", "tab:orange"),
]

optmodel = shortestPathModel(grid=GRID)


def run_sweep(num_trials=50, degrees=(1, 2, 4, 6, 8), num_test=1000):
    """Degree sweep of per-trial test regret, saved/shown as a grouped boxplot."""
    experiment = RegretExperiment(
        optmodel=optmodel,
        solvers=SOLVERS,
        num_features=NUM_FEATURES,
        noise_width=NOISE_WIDTH,
        num_train=NUM_TRAIN,
        num_test=num_test,
        num_trials=num_trials,
        degrees=degrees,
        rng_seed=RNG_SEED,
    )
    results = experiment.run()

    plotter = RegretBoxPlot(
        groups=list(degrees),
        series=SERIES,
        xlabel="Polynomial degree of DGP",
        ylabel="Test regret (%)",
        title=f"Two-stage GBM vs LASSO vs SPO+ linear over {num_trials} trials "
              f"(5x5 grid shortest path)",
    )
    plotter.plot(results)
    plotter.save("regret_boxplot.png")
    print("Saved boxplot to regret_boxplot.png")
    plotter.show()
    return results


def run_contexts(degree=4, n_contexts=200, shared_models=True, seed=RNG_SEED):
    """Train once, face n_contexts test contexts, print the averaged metric table."""
    experiment = ContextExperiment(
        optmodel=optmodel,
        solvers=SOLVERS,
        degree=degree,
        n_contexts=n_contexts,
        shared_models=shared_models,
        num_features=NUM_FEATURES,
        noise_width=NOISE_WIDTH,
        num_train=NUM_TRAIN,
        rng_seed=seed,
    )
    return experiment.print_table()

def run_histogram(degree=4, n_trials=5, seed=RNG_SEED, test_seed=TEST_RNG_SEED):
    """Vary training sets for a single fixed context and plot path distribution histograms."""
    experiment = HistogramExperiment(
        optmodel=optmodel,  # <-- Pass the real optmodel here
        solvers=SOLVERS,
        degree=degree,
        n_trials=n_trials,
        num_features=NUM_FEATURES,
        noise_width=NOISE_WIDTH,
        num_train=NUM_TRAIN,
        rng_seed=seed,
        test_rng_seed=test_seed
    )
    
    print(f"Building path matrix for grid {GRID}...")
    experiment.compute_all_paths(grid=GRID)
    
    print(f"Running histogram experiment over {n_trials} independent training trials...")
    experiment.run(grid=GRID)
    
    print("Generating visualizations...")
    experiment.plot_histogram()
    
    return experiment.results


if __name__ == "__main__":
    # run_contexts(degree=8, n_contexts=1)
    # run_sweep(num_trials=50)
    
    # --- Execute the New Histogram Experiment ---
    print("\n" + "=" * 50)
    print("Running Histogram Experiment...")
    print("=" * 50)
    results = run_histogram(degree=2, n_trials=2, seed=RNG_SEED, test_seed=TEST_RNG_SEED)
    print("Done!")
