"""
The (degree x training-set-size) sweep grid, shared by every piece of the pipeline.

Kept free of PyEPO/Gurobi imports on purpose: run_trial.py (compute) needs the
optimization stack, aggregate.py (plotting) must not, so the things they agree on
-- which cells exist, which seed each cell gets, and what the metrics mean -- live
here.
"""
from pathlib import Path
from typing import NamedTuple

DEGREES = (1, 2, 4, 6, 8)
SIZES = (100, 1000)
NUM_TRIALS = 50

# Base seed; matches RNG_SEED in main.py.
RNG_SEED = 143

# Number of test contexts each trial pools its regrets over.
NUM_CONTEXTS = 1000

RESULT_DIR = Path("results")

# (key, label, color) per solver, for plot legends. Imported by main.py too, so
# the plots stay in sync with the sweep.
SERIES = [
    ("gbm", "2-stage GBM", "tab:blue"),
    ("lasso", "2-stage LASSO", "tab:green"),
    ("spo", "SPO+ linear", "tab:orange"),
]

class Metric(NamedTuple):
    """
    One quantity ContextExperiment knows how to compute.

    Attributes:
        header (str): Plain-text column title, for the printed table
        label (str): Axis title, for plots (matplotlib mathtext allowed)
        needs_fstar (bool): Whether computing it requires the conditional mean.
            A generator that returns fstar=None makes these metrics unavailable.
        denom (str): What the per-context values are pooled against --
            "count" for a plain mean, or "opt_Y" / "opt_fstar" for
            100 * (sum of values) / (sum of that optimal cost).
    """
    header: str
    label: str
    needs_fstar: bool
    denom: str


# Every metric ContextExperiment can compute. For a decision w_hat = z*(f(X)) made
# from a solver's predicted costs, against realized costs Y and conditional mean f*:
#
#   loss_Y          <Y, w_hat>                                 realized cost
#   regret_Y        <Y, w_hat>  - <Y, z*(Y)>                   SPO regret
#   regret_Y_lowvar <f*, w_hat> - <f*, z*(Y)>                  same decision pair as
#                                                              regret_Y, scored under
#                                                              f* instead of noisy Y,
#                                                              hence lower variance
#   regret_fstar    <f*, w_hat> - <f*, z*(f*)>                 gap to the best policy
#
# regret_Y_lowvar shares regret_Y's denominator so the two sit on one scale and can be
# read side by side. Every entry is computed and persisted whenever f* is available --
# selection happens at display time, so changing SHOW_METRICS never costs a rerun.
METRICS = {
    "loss_Y":          Metric("Decision Loss", "Decision loss",
                              needs_fstar=False, denom="count"),
    "regret_Y":        Metric("Regret vs Y", r"Regret vs Y (%)",
                              needs_fstar=False, denom="opt_Y"),
    "regret_Y_lowvar": Metric("Regret vs Y (f*)", r"Regret vs Y, $f^*$-scored (%)",
                              needs_fstar=True, denom="opt_Y"),
    "regret_fstar":    Metric("Regret vs f*", r"Regret vs $f^*$ (%)",
                              needs_fstar=True, denom="opt_fstar"),
}

# THE SWITCH: which metrics to display. Drives both the table main.py prints and the
# boxplot panels aggregate.py draws, so the two stay in step. Every metric is stored
# in the trial JSONs regardless, so you can re-plot an existing sweep under a
# different selection (aggregate.py --metrics ...) without recomputing anything.
SHOW_METRICS = ("regret_fstar", "regret_Y_lowvar", "regret_Y")


def seed_for(deg, trial):
    """
    Seed for one trial. Distinct for every (deg, trial), and deliberately shared
    between the two training-set sizes so that n=100 and n=1000 face the same DGP
    draw at the same (deg, trial) -- the size comparison is then paired rather than
    confounded by which B happened to be sampled.
    """
    return RNG_SEED + 1000 * deg + trial


def result_path(outdir, deg, num_train, trial):
    """One JSON per trial, so the array job is restartable cell by cell."""
    return Path(outdir) / f"deg{deg}_n{num_train}_t{trial}.json"