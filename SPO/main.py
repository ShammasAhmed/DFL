import numpy as np
from itertools import combinations

from solvers import GBM_nonparametric
import utils

rng = np.random.default_rng(seed=42)
D = 40
P = 5
H = 0.5
DEG = 4
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

DATASETS = utils.generate_test_val_train_data(D, P, B[0], DEG, H, NUM_TRAIN, NUM_VAL, NUM_TEST, rng)

X_TRAIN, C_TRAIN = DATASETS[0], DATASETS[1]
X_VAL, C_VAL = DATASETS[2], DATASETS[3]
X_TEST, C_TEST = DATASETS[4], DATASETS[5]

GBM_solver = GBM_nonparametric(D, P, H, B, NUM_TRAIN, NUM_TEST, NUM_TRIALS, PATHS, PATH_MATRIX)