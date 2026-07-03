"""
Original numpy/sklearn shortest-path driver (pre-PyEPO).

Kept as a reference baseline: it uses the hand-rolled path enumeration in
utils.py and the GBM_nonparametric two-stage solver in solvers.py. The PyEPO
based comparison lives in main.py.
"""
import numpy as np
from itertools import combinations

from solvers import GBM_nonparametric
import utils

rng = np.random.default_rng(seed=42)
D = 40
P = 5
H = 0.5
# DEG = 4
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
PATH_MATRIX = utils.create_path_matrix(PATHS, D)

for DEG in [1, 2, 4, 6, 8]:
    DATASETS = utils.generate_test_val_train_data(D, P, B[0], DEG, H, NUM_TRAIN, NUM_VAL, NUM_TEST, rng)

    X_TRAIN, C_TRAIN = DATASETS[0], DATASETS[1]
    X_VAL, C_VAL = DATASETS[2], DATASETS[3]
    X_TEST, C_TEST = DATASETS[4], DATASETS[5]

    GBM_solver = GBM_nonparametric(D, P, H, B, PATHS, PATH_MATRIX,
                                X_TRAIN, C_TRAIN, X_VAL, C_VAL,
                                NUM_TRIALS)
    gbm_model = GBM_solver.model
    gbm_regret = GBM_solver.test_decision_regret_Y(X_TEST, C_TEST)

    print(f"THE REGRET FOR DEG = {DEG} IS {round(gbm_regret, 4)}%")
