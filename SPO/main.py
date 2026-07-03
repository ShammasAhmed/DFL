"""
PyEPO-based comparison on the 5x5 grid shortest-path benchmark.

Runs NUM_TRIALS independent trials per DGP degree and compares:
  - GBM_twostage  : prediction-focused (two-stage) HistGradientBoosting baseline
  - LASSO_twostage: prediction-focused (two-stage) least-squares LASSO baseline
  - LinearSPOPlus : decision-focused linear model trained with PyEPO's SPO+ loss

The sweep is orchestrated by RegretExperiment (experiments.py) and the per-trial
test regrets are drawn as a grouped boxplot by RegretBoxPlot (plots.py).
"""
from pyepo.model.grb import shortestPathModel

from solvers import GBM_twostage, LASSO_twostage, LinearSPOPlus
from experiments import RegretExperiment
from plots import RegretBoxPlot

NUM_TRIALS = 50
DEGREES = [1, 2, 4, 6, 8]

optmodel = shortestPathModel(grid=(5, 5))

experiment = RegretExperiment(
    optmodel=optmodel,
    solvers=[
        ("gbm", GBM_twostage),
        ("lasso", LASSO_twostage),
        ("spo", LinearSPOPlus),
    ],
    num_features=5,
    noise_width=0.5,
    num_train=100,
    num_test=1000,
    num_trials=NUM_TRIALS,
    degrees=DEGREES,
    rng_seed=42,
)
results = experiment.run()

plotter = RegretBoxPlot(
    groups=DEGREES,
    series=[
        ("gbm", "2-stage GBM", "tab:blue"),
        ("lasso", "2-stage LASSO", "tab:green"),
        ("spo", "SPO+ linear", "tab:orange"),
    ],
    xlabel="Polynomial degree of DGP",
    ylabel="Test regret (%)",
    title=f"Two-stage GBM vs LASSO vs SPO+ linear over {NUM_TRIALS} trials "
          f"(5x5 grid shortest path)",
)
plotter.plot(results)
plotter.save("regret_boxplot.png")
print("Saved boxplot to regret_boxplot.png")
plotter.show()
