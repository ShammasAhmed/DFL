"""
Original numpy/sklearn shortest-path driver (pre-PyEPO).

Kept as a reference baseline: it uses the hand-rolled path enumeration in
utils.py and the GBMNonparametric two-stage solver in solvers.py. The PyEPO
based comparison lives in main.py.
"""
import numpy as np
from itertools import combinations

from solvers import GBMNonparametric
import utils

rng = np.random.default_rng(seed=42)
D = 40
P = 5
h = 0.5
# deg = 4
NUM_TRIALS = 1
NUM_TRAIN = 100
NUM_VAL = int(NUM_TRAIN / 4)
NUM_TEST = 1000
RNG_SEED = 42
B_FIXED = True

B = []
if B_FIXED:
    B.append(rng.binomial(1, 0.5, size=(D, P)))
else:
    for _ in range(NUM_TRIALS):
        B.append(rng.binomial(1, 0.5, size=(D, P)))

PATHS = utils.create_paths()
path_matrix = utils.compute_paths(PATHS, D)

for deg in [1, 2, 4, 6, 8]:
    DATASETS = utils.generate_test_val_train_data(D, P, B[0], deg, h, NUM_TRAIN, NUM_VAL, NUM_TEST, rng)

    X_TRAIN, Y_TRAIN = DATASETS[0], DATASETS[1]
    X_VAL, Y_VAL = DATASETS[2], DATASETS[3]
    X_TEST, Y_TEST = DATASETS[4], DATASETS[5]

    GBM_solver = GBMNonparametric(D, P, h, B, PATHS, path_matrix,
                                X_TRAIN, Y_TRAIN, X_VAL, Y_VAL,
                                NUM_TRIALS)
    gbm_model = GBM_solver.model
    gbm_regret = GBM_solver.test_decision_regret_Y(X_TEST, Y_TEST)

    print(f"THE REGRET FOR deg = {deg} IS {round(gbm_regret, 4)}%")
