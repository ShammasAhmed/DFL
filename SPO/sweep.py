"""
The experiment grids, shared by every piece of the pipeline: the context sweep's
(degree x training-set-size) cells, and the histogram experiment's (size x trial)
cells further down.

Kept free of PyEPO/Gurobi imports on purpose: the *_trial.py scripts (compute) need
the optimization stack, the *_aggregate.py scripts (plotting) must not, so the things
they agree on -- which cells exist, which seed each cell gets, and what the metrics
mean -- live here.
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

# Default multiplicative noise half-width (epsilon ~ U[1-h, 1+h]). The sbatch can
# override it per run via --noise-width; this is the fallback for a bare context_trial.py
# invocation and the value context_aggregate.py assumes when selecting which files to read.
NOISE_WIDTH = 0.5

RESULT_DIR = Path("results")

# (key, label, color) per solver, for plot legends. Imported by main.py too, so
# the plots stay in sync with the sweep.
#
# No solver is blue: the histogram plots reserve blue for the true-cost curve, and a
# blue bar sitting on a blue line reads as part of it. One color per solver across
# every figure is worth more than any individual color choice.
SERIES = [
    ("gbm", "2-stage GBM", "tab:red"),
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
#   regret_Y_lowvar <f*, w_hat> - <Y, z*(Y)>                   regret_Y with only the
#                                                              solver's term scored under
#                                                              f*; E<Y, w_hat> =
#                                                              <f*, w_hat>, so the same
#                                                              estimand with less noise
#   regret_fstar    <f*, w_hat> - <f*, z*(f*)>                 gap to the best policy
#
# regret_Y_lowvar shares regret_Y's benchmark, and so its denominator: the two sit on one
# scale and can be read side by side. Every entry is computed and persisted whenever f* is
# available -- selection happens at display time, so changing SHOW_METRICS never costs a
# rerun.
METRICS = {
    "loss_Y":          Metric("Decision Loss", "Decision loss",
                              needs_fstar=False, denom="count"),
    "regret_Y":        Metric("Regret vs Y", r"Regret vs Y (%)",
                              needs_fstar=False, denom="opt_Y"),
    "regret_Y_lowvar": Metric("Regret vs Y (lv)", r"Regret vs Y, low-variance (%)",
                              needs_fstar=True, denom="opt_Y"),
    "regret_fstar":    Metric("Regret vs f*", r"Regret vs $f^*$ (%)",
                              needs_fstar=True, denom="opt_fstar"),
}

# THE SWITCH: which metrics to display. Drives both the table main.py prints and the
# boxplot panels context_aggregate.py draws, so the two stay in step. Every metric is
# stored in the trial JSONs regardless, so you can re-plot an existing sweep under a
# different selection (context_aggregate.py --metrics ...) without recomputing it.
SHOW_METRICS = ("regret_fstar", "regret_Y_lowvar", "regret_Y")


# --- Histogram experiment ---------------------------------------------------- #
# A single fixed test context, faced by HIST_NUM_TRIALS independently trained models
# at each training-set size. All of it shares one ground truth B, fixed by
# HIST_DGP_SEED (see datagen.numpy_shortest_path_gen); the per-trial seeds below vary
# only the training draw.
HIST_DEG = 4
HIST_SIZES = (100, 1000)
HIST_NUM_TRIALS = 500

HIST_DGP_SEED = RNG_SEED        # fixes B, and so the true path costs
HIST_CONTEXT_SEED = RNG_SEED    # draws the candidate contexts to choose from

# The % by which the context's best path must beat the second-best must fall in
# [HIST_CONTEXT_MARGIN, HIST_CONTEXT_MARGIN_MAX]. Raise the floor to plant an obvious
# winner; lower the ceiling to plant a near-tie. Selection sees only f*; the training
# draws are untouched. (0.0, None) is an ordinary random context.
HIST_CONTEXT_MARGIN = 0.0
HIST_CONTEXT_MARGIN_MAX = 0.1
HIST_CONTEXT_POOL = 200_000     # candidates scanned for the first one in the window

# Every trial draws this many rows regardless of training-set size, then takes the
# first num_train for training and the next num_train // 4 for validation. Sizing the
# draw by the largest size makes n=100 a prefix of n=1000 at the same trial -- the two
# arms are then nested draws rather than unrelated ones, so the size comparison is
# paired. Must be at least max(HIST_SIZES) * 5 // 4.
HIST_DRAW = max(HIST_SIZES) + max(HIST_SIZES) // 4

HIST_RESULT_DIR = Path("results_histogram")


def hist_seed_for(trial):
    """
    Seed for one histogram trial's training draw. Shared between the two training-set
    sizes (see HIST_DRAW), and distinct from HIST_CONTEXT_SEED so that no trial trains
    on the context it is tested against.
    """
    return RNG_SEED + 1000 + trial


def hist_result_path(outdir, num_train, trial):
    """One JSON per (size, trial), so the array job is restartable cell by cell."""
    return Path(outdir) / f"hist_n{num_train}_t{trial}.json"


def seed_for(deg, trial):
    """
    Seed for one trial. Distinct for every (deg, trial), and deliberately shared
    between the two training-set sizes so that n=100 and n=1000 face the same DGP
    draw at the same (deg, trial) -- the size comparison is then paired rather than
    confounded by which B happened to be sampled.
    """
    return RNG_SEED + 1000 * deg + trial


def result_path(outdir, deg, num_train, trial, h=NOISE_WIDTH):
    """
    One JSON per trial, so the array job is restartable cell by cell.

    The noise half-width h is in the name so trials at different noise levels can share
    one results/ directory without colliding -- and so the resume check never mistakes a
    trial run at one h for the same (deg, size, trial) run at another.
    """
    return Path(outdir) / f"deg{deg}_n{num_train}_h{h}_t{trial}.json"