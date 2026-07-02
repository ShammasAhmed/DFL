import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning

from utils import *

class GBM_nonparametric:
    """
    This class creates a nonparametric GBM solver that can solve a decision problem given sample 
    data. It takes an instance with training and test datasets, trains a nonparametric GBM learner 
    and computes an upper bound on the maximum regret improvement that a DFL method can provide. It 
    does this by evaluating the decision loss against an oracle, the decision regret relative to f* 
    (true conditional mean), and decision regret relative to Y (the response).
    
    """

    def __init__(self, D: int, P: int, H: float, B: list, NUM_TRAIN: int, NUM_TEST: int, NUM_TRIALS: int, PATHS: list, PATH_MATRIX: np.ndarray, RNG_SEED: int = 42):
        """
        Initializes a nonparametric GBM solver.

        Inputs: 
            D (int): dim(Y)
            P (int): dim(X)
            H (float): Half noise width
            B (list): List of all ground truth matrices B
            NUM_TRAIN (int): Training set size
            NUM_TEST (int):  Test set size
            NUM_TRIALS (int): How many trials to conduct and average over
            PATHS (list): List of all unique paths
            PATH_MATRIX (np.ndarray): Path-edge incidence matrix for fast vector operations
            RNG_SEED (int): A global seed for the data generation method
        
        Returns:
            A nonparametric GBM solver object
        """
        self.D = D
        self.P = P
        self.H = H
        self.B = B
        self.PATHS = PATHS
        self.PATH_MATRIX = PATH_MATRIX
        self.NUM_TRAIN = NUM_TRAIN
        self.NUM_TEST = NUM_TEST
        self.NUM_TRIALS = NUM_TRIALS
        self.RNG_SEED = RNG_SEED

        print("INITIALIZED")

    

        
            


