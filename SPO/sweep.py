"""
The (degree x training-set-size) sweep grid, shared by every piece of the pipeline.

Kept free of PyEPO/Gurobi imports on purpose: run_trial.py (compute) needs the
optimization stack, aggregate.py (plotting) must not, so the one thing they agree
on -- which cells exist, and which seed each cell gets -- lives here.
"""
from pathlib import Path

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

# The two pooled regret metrics ContextExperiment reports, and how to title them.
METRICS = [
    ("regret_fstar", r"Regret vs $f^*$ (%)"),
    ("regret_Y", "Regret vs Y (%)"),
]


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


def cells():
    """Every (deg, num_train, trial) triple in the sweep, in array-task order."""
    for num_train in SIZES:
        for deg in DEGREES:
            for trial in range(NUM_TRIALS):
                yield deg, num_train, trial