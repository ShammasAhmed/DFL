import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

import torch
from torch import nn
from torch.utils.data import DataLoader
import pyepo
from pyepo.data.dataset import optDataset
from pyepo.func import SPOPlus

from utils import *

class GBM_nonparametric:
    """
    This class creates a nonparametric GBM solver that can solve a decision problem given sample 
    data. It takes an instance with training and test datasets, trains a nonparametric GBM learner 
    and computes an upper bound on the maximum regret improvement that a DFL method can provide. It 
    does this by evaluating the decision loss against an oracle, the decision regret relative to f* 
    (true conditional mean), and decision regret relative to Y (the response).
    
    """

    def __init__(self, D: int, P: int, H: float, B: list, PATHS: list, PATH_MATRIX: np.ndarray,
                 X_TRAIN: np.ndarray, C_TRAIN: np.ndarray, X_VAL: np.ndarray, C_VAL: np.ndarray,
                 NUM_TRIALS: int, RNG_SEED: int = 42):
        """
        Initializes a nonparametric GBM solver bound to a given training and validation set.

        Inputs:
            D (int): dim(Y)
            P (int): dim(X)
            H (float): Half noise width
            B (list): List of all ground truth matrices B
            PATHS (list): List of all unique paths
            PATH_MATRIX (np.ndarray): Path-edge incidence matrix for fast vector operations
            X_TRAIN (np.ndarray): Training covariates the solver is trained on
            C_TRAIN (np.ndarray): Training costs the solver is trained on
            X_VAL (np.ndarray): Validation covariates
            C_VAL (np.ndarray): Validation costs
            NUM_TRIALS (int): How many trials to conduct and average over
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

        self.X_TRAIN = X_TRAIN
        self.C_TRAIN = C_TRAIN
        self.X_VAL = X_VAL
        self.C_VAL = C_VAL

        self.NUM_TRAIN = X_TRAIN.shape[0]
        self.NUM_VAL = X_VAL.shape[0]
        self.NUM_TRIALS = NUM_TRIALS
        self.RNG_SEED = RNG_SEED

        self.scale_factor = np.max(np.abs(self.C_TRAIN))
        self.model = self.train()

    def train(self):
        C_TRAIN_SCALED = self.C_TRAIN / self.scale_factor

        gbm_l2_grid = np.logspace(-3, 3, 7)

        best_model, best_err = None, np.inf
        for l2 in gbm_l2_grid:
            base = HistGradientBoostingRegressor(
                l2_regularization=l2,
                learning_rate=0.1,
                max_iter=200,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                random_state=0,
            )
            m = MultiOutputRegressor(base)
            m.fit(self.X_TRAIN, C_TRAIN_SCALED)
            reg = self.validation_spo_regret(m, self.X_VAL, self.C_VAL, self.scale_factor)
            if reg < best_err:
                best_err, best_model = reg, m
        return best_model

    def validation_spo_regret(self, model, X, C, scale_factor):
        """
        Total SPO regret of a fitted sklearn regressor on scaled costs, evaluated
        over the given covariate/cost set using this solver's path-edge incidence matrix.

        Inputs:
            model: A fitted sklearn regressor mapping covariates to edge costs
            X (np.ndarray): Covariates to evaluate on
            C (np.ndarray): True costs for X
            scale_factor (float): Rescales model predictions back to cost units

        Returns:
            reg (float): The total regret of the model over the given set
        """
        reg = 0.0
        for v in range(X.shape[0]):
            c_v = C[v]
            z_v = evaluate_path_cost(solve_shortest_path(self.PATH_MATRIX, c_v), c_v)
            p_v = np.clip(model.predict(X[v].reshape(1, -1)).flatten() * scale_factor, 0.001, None)
            reg += evaluate_path_cost(solve_shortest_path(self.PATH_MATRIX, p_v), c_v) - z_v
        return reg

    def test_decision_regret_Y(self, X_test, C_test):
        """
        Percentage decision regret of the trained model relative to Y (the realized
        costs) over the test set.

        For each test point the model predicts edge costs, a path is chosen under those
        predictions, and its loss is measured against the realized costs C_test. The
        reported figure aggregates over all test points:

            100 * (sum of per-test decision losses) / (sum of per-test optimal costs)

        Inputs:
            X_test (np.ndarray): Test covariates
            C_test (np.ndarray): Realized (true) test costs

        Returns:
            regret_pct (float): Total decision loss as a percentage of total optimal cost
        """
        total_loss = 0.0
        total_optimal = 0.0
        for t in range(X_test.shape[0]):
            c_t = C_test[t]
            z_t = evaluate_path_cost(solve_shortest_path(self.PATH_MATRIX, c_t), c_t)
            p_t = np.clip(self.model.predict(X_test[t].reshape(1, -1)).flatten() * self.scale_factor, 0.001, None)
            realized_cost = evaluate_path_cost(solve_shortest_path(self.PATH_MATRIX, p_t), c_t)
            total_loss += realized_cost - z_t
            total_optimal += z_t
        return 100 * total_loss / total_optimal


class SklearnPredictor(nn.Module):
    """
    Adapts a fitted sklearn regressor to PyEPO's nn.Module predictor interface so
    that a two-stage learner can be scored with pyepo.metric.regret alongside the
    decision-focused (torch) models.

    Inputs:
        sk_model: A fitted sklearn regressor mapping covariates to edge costs
        scale_factor (float): Rescales predictions back to cost units (regret is
            invariant to a positive scale, but this keeps predictions in cost units)
    """

    def __init__(self, sk_model, scale_factor: float = 1.0):
        super().__init__()
        self.sk_model = sk_model
        self.scale_factor = scale_factor
        # PyEPO reads next(predmodel.parameters()).device to place tensors; the
        # sklearn model has no torch parameters, so register a dummy one so that
        # call succeeds. It is never used in forward and carries no gradient.
        self._device_ref = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        device = self._device_ref.device
        preds = self.sk_model.predict(x.detach().cpu().numpy()) * self.scale_factor
        return torch.as_tensor(preds, dtype=torch.float32, device=device)


class GBM_twostage:
    """
    Prediction-focused (two-stage) GBM baseline, evaluated through PyEPO.

    A HistGradientBoosting regressor is trained with plain MSE, its L2 strength is
    selected by PyEPO validation regret against the shared optimization model, and
    the selected predictor (wrapped as an nn.Module) is exposed at self.model so it
    can be scored with pyepo.metric.regret exactly like the DFL solvers.
    """

    def __init__(self, optmodel, X_TRAIN, C_TRAIN, X_VAL, C_VAL,
                 l2_grid=None, seed: int = 0):
        """
        Inputs:
            optmodel: A PyEPO optModel (the shared shortest-path optimizer)
            X_TRAIN, C_TRAIN (np.ndarray): Training covariates / costs
            X_VAL, C_VAL (np.ndarray): Validation covariates / costs
            l2_grid (iterable): Candidate L2 regularization strengths
            seed (int): Seed for the underlying regressor
        """
        self.optmodel = optmodel
        self.X_TRAIN, self.C_TRAIN = X_TRAIN, C_TRAIN
        self.X_VAL, self.C_VAL = X_VAL, C_VAL
        self.l2_grid = np.logspace(-3, 3, 7) if l2_grid is None else l2_grid
        self.seed = seed
        self.scale_factor = np.max(np.abs(C_TRAIN))

        val_set = optDataset(optmodel, X_VAL, C_VAL)
        self.val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)

        self.model = self.train()

    def train(self):
        C_TRAIN_SCALED = self.C_TRAIN    / self.scale_factor

        best_model, best_reg = None, np.inf
        for l2 in self.l2_grid:
            base = HistGradientBoostingRegressor(
                l2_regularization=l2,
                learning_rate=0.1,
                max_iter=200,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                random_state=self.seed,
            )
            m = MultiOutputRegressor(base)
            m.fit(self.X_TRAIN, C_TRAIN_SCALED)
            predictor = SklearnPredictor(m, self.scale_factor)
            reg = pyepo.metric.regret(predictor, self.optmodel, self.val_loader)
            if reg < best_reg:
                best_reg, best_model = reg, predictor
        return best_model


class LASSO_twostage:
    """
    Prediction-focused (two-stage) least-squares LASSO baseline, evaluated
    through PyEPO.

    A LASSO (L1-regularized least squares) regressor is fit for each candidate
    regularization strength lambda; the strength minimizing PyEPO validation
    regret against the shared optimization model is selected. The chosen
    predictor (wrapped as an nn.Module) is exposed at self.model so it can be
    scored with pyepo.metric.regret exactly like the other solvers.
    """

    def __init__(self, optmodel, X_TRAIN, C_TRAIN, X_VAL, C_VAL,
                 lambdas=None, seed: int = 0):
        """
        Inputs:
            optmodel: A PyEPO optModel (the shared shortest-path optimizer)
            X_TRAIN, C_TRAIN (np.ndarray): Training covariates / costs
            X_VAL, C_VAL (np.ndarray): Validation covariates / costs
            lambdas (iterable): Candidate L1 regularization strengths
            seed (int): Seed for the underlying regressor
        """
        self.optmodel = optmodel
        self.X_TRAIN, self.C_TRAIN = X_TRAIN, C_TRAIN
        self.X_VAL, self.C_VAL = X_VAL, C_VAL
        self.lambdas = np.logspace(-6, 0, 10) if lambdas is None else lambdas
        self.seed = seed
        self.scale_factor = np.max(np.abs(C_TRAIN))

        val_set = optDataset(optmodel, X_VAL, C_VAL)
        self.val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)

        self.model = self.train()

    def train(self):
        C_TRAIN_SCALED = self.C_TRAIN / self.scale_factor

        best_model, best_reg = None, np.inf
        for lam in self.lambdas:
            base = Lasso(alpha=lam, max_iter=10000)
            m = MultiOutputRegressor(base)
            m.fit(self.X_TRAIN, C_TRAIN_SCALED)
            predictor = SklearnPredictor(m, self.scale_factor)
            reg = pyepo.metric.regret(predictor, self.optmodel, self.val_loader)
            if reg < best_reg:
                best_reg, best_model = reg, predictor
        return best_model


class LinearSPOPlus:
    """
    Decision-focused (DFL) solver: a linear predictor trained end-to-end with
    PyEPO's SPO+ surrogate loss.

    Trains on init and exposes the trained predictor at self.model, so it can be
    scored with pyepo.metric.regret just like the two-stage GBM baseline.
    """

    def __init__(self, optmodel, X_TRAIN, C_TRAIN, X_VAL, C_VAL,
                 num_epochs: int = 20, lr: float = 1e-2, batch_size: int = 32,
                 seed: int = 42):
        """
        Inputs:
            optmodel: A PyEPO optModel (the shared shortest-path optimizer)
            X_TRAIN, C_TRAIN (np.ndarray): Training covariates / costs
            X_VAL, C_VAL (np.ndarray): Validation covariates / costs (reserved for
                early stopping / tuning; unused in this basic loop)
            num_epochs (int): Training epochs
            lr (float): Adam learning rate
            batch_size (int): Minibatch size
            seed (int): Torch seed for reproducible initialization
        """
        self.optmodel = optmodel
        self.X_TRAIN, self.C_TRAIN = X_TRAIN, C_TRAIN
        self.X_VAL, self.C_VAL = X_VAL, C_VAL
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.P = X_TRAIN.shape[1]
        self.D = C_TRAIN.shape[1]

        self.model = self.train()

    def train(self):
        torch.manual_seed(self.seed)
        predmodel = nn.Linear(self.P, self.D)
        optimizer = torch.optim.Adam(predmodel.parameters(), lr=self.lr)
        spop = SPOPlus(self.optmodel)

        train_set = optDataset(self.optmodel, self.X_TRAIN, self.C_TRAIN)
        loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        predmodel.train()
        for _ in range(self.num_epochs):
            for x, c, w, z in loader:
                cp = predmodel(x)
                loss = spop(cp, c, w, z)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        predmodel.eval()
        return predmodel



