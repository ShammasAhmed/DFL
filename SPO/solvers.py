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

class GBMNonparametric:
    """
    This class creates a nonparametric GBM solver that can solve a decision problem given sample 
    data. It takes an instance with training and test datasets, trains a nonparametric GBM learner 
    and computes an upper bound on the maximum regret improvement that a DFL method can provide. It 
    does this by evaluating the decision loss against an oracle, the decision regret relative to f* 
    (true conditional mean), and decision regret relative to Y (the response).
    
    """

    def __init__(self, D: int, P: int, h: float, B: list, PATHS: list, path_matrix: np.ndarray,
                 X_TRAIN: np.ndarray, Y_TRAIN: np.ndarray, X_VAL: np.ndarray, Y_VAL: np.ndarray,
                 NUM_TRIALS: int, rng_seed: int = 42):
        """
        Initializes a nonparametric GBM solver bound to a given training and validation set.

        Inputs:
            D (int): dim(Y)
            P (int): dim(X)
            h (float): Half noise width
            B (list): List of all ground truth matrices B
            PATHS (list): List of all unique paths
            path_matrix (np.ndarray): Path-edge incidence matrix for fast vector operations
            X_TRAIN (np.ndarray): Training covariates the solver is trained on
            Y_TRAIN (np.ndarray): Training costs the solver is trained on
            X_VAL (np.ndarray): Validation covariates
            Y_VAL (np.ndarray): Validation costs
            NUM_TRIALS (int): How many trials to conduct and average over
            rng_seed (int): A global rng_seed for the data generation method

        Returns:
            A nonparametric GBM solver object
        """
        self.D = D
        self.P = P
        self.h = h
        self.B = B
        self.PATHS = PATHS
        self.path_matrix = path_matrix

        self.X_TRAIN = X_TRAIN
        self.Y_TRAIN = Y_TRAIN
        self.X_VAL = X_VAL
        self.Y_VAL = Y_VAL

        self.NUM_TRAIN = X_TRAIN.shape[0]
        self.NUM_VAL = X_VAL.shape[0]
        self.NUM_TRIALS = NUM_TRIALS
        self.rng_seed = rng_seed

        self.scale_factor = np.max(np.abs(self.Y_TRAIN))
        self.model = self.train()

    def train(self):
        Y_TRAIN_SCALED = self.Y_TRAIN / self.scale_factor

        gbm_l2_grid = np.logspace(-3, 3, 7)

        best_model, best_regret = None, np.inf
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
            m.fit(self.X_TRAIN, Y_TRAIN_SCALED)
            regret = self.validation_spo_regret(m, self.X_VAL, self.Y_VAL, self.scale_factor)
            if regret < best_regret:
                best_regret, best_model = regret, m
        return best_model

    def validation_spo_regret(self, model, X, Y, scale_factor):
        """
        Total SPO regret of a fitted sklearn regressor on scaled costs, evaluated
        over the given covariate/cost set using this solver's path-edge incidence matrix.

        Inputs:
            model: A fitted sklearn regressor mapping covariates to edge costs
            X (np.ndarray): Covariates to evaluate on
            Y (np.ndarray): True costs for X
            scale_factor (float): Rescales model predictions back to cost units

        Returns:
            regret (float): The total regret of the model over the given set
        """
        regret = 0.0
        for v in range(X.shape[0]):
            y_v = Y[v]
            z_v = evaluate_path_cost(solve_shortest_path(self.path_matrix, y_v), y_v)
            Y_hat = np.clip(model.predict(X[v].reshape(1, -1)).flatten() * scale_factor, 0.001, None)
            regret += evaluate_path_cost(solve_shortest_path(self.path_matrix, Y_hat), y_v) - z_v
        return regret

    def test_decision_regret_Y(self, X_TEST, Y_TEST):
        """
        Percentage decision regret of the trained model relative to Y (the realized
        costs) over the test set.

        For each test point the model predicts edge costs, a path is chosen under those
        predictions, and its loss is measured against the realized costs Y_TEST. The
        reported figure aggregates over all test points:

            100 * (sum of per-test decision losses) / (sum of per-test optimal costs)

        Inputs:
            X_TEST (np.ndarray): Test covariates
            Y_TEST (np.ndarray): Realized (true) test costs

        Returns:
            regret_pct (float): Total decision loss as a percentage of total optimal cost
        """
        total_regret = 0.0
        total_optimal = 0.0
        for t in range(X_TEST.shape[0]):
            y_t = Y_TEST[t]
            z_t = evaluate_path_cost(solve_shortest_path(self.path_matrix, y_t), y_t)
            Y_hat = np.clip(self.model.predict(X_TEST[t].reshape(1, -1)).flatten() * self.scale_factor, 0.001, None)
            loss_Y = evaluate_path_cost(solve_shortest_path(self.path_matrix, Y_hat), y_t)
            total_regret += loss_Y - z_t
            total_optimal += z_t
        return 100 * total_regret / total_optimal


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
        # PyEPO reads next(model.parameters()).device to place tensors; the
        # sklearn model has no torch parameters, so register a dummy one so that
        # call succeeds. It is never used in forward and carries no gradient.
        self._device_ref = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        device = self._device_ref.device
        Y_hat = self.sk_model.predict(x.detach().cpu().numpy()) * self.scale_factor
        return torch.as_tensor(Y_hat, dtype=torch.float32, device=device)


class GBMTwoStage:
    """
    Prediction-focused (two-stage) GBM baseline, evaluated through PyEPO.

    A HistGradientBoosting regressor is trained with plain MSE, its L2 strength is
    selected by PyEPO validation regret against the shared optimization model, and
    the selected predictor (wrapped as an nn.Module) is exposed at self.model so it
    can be scored with pyepo.metric.regret exactly like the DFL solvers.
    """

    def __init__(self, optmodel, X_TRAIN, Y_TRAIN, X_VAL, Y_VAL,
                 l2_grid=None, rng_seed: int = 0):
        """
        Inputs:
            optmodel: A PyEPO optModel (the shared shortest-path optimizer)
            X_TRAIN, Y_TRAIN (np.ndarray): Training covariates / costs
            X_VAL, Y_VAL (np.ndarray): Validation covariates / costs
            l2_grid (iterable): Candidate L2 regularization strengths
            rng_seed (int): Seed for the underlying regressor
        """
        self.optmodel = optmodel
        self.X_TRAIN, self.Y_TRAIN = X_TRAIN, Y_TRAIN
        self.X_VAL, self.Y_VAL = X_VAL, Y_VAL
        self.l2_grid = np.logspace(-3, 3, 7) if l2_grid is None else l2_grid
        self.rng_seed = rng_seed
        self.scale_factor = np.max(np.abs(Y_TRAIN))

        val_set = optDataset(optmodel, X_VAL, Y_VAL)
        self.val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)

        self.model = self.train()

    def train(self):
        Y_TRAIN_SCALED = self.Y_TRAIN    / self.scale_factor

        best_model, best_regret = None, np.inf
        for l2 in self.l2_grid:
            base = HistGradientBoostingRegressor(
                l2_regularization=l2,
                learning_rate=0.1,
                max_iter=200,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                random_state=self.rng_seed,
            )
            m = MultiOutputRegressor(base)
            m.fit(self.X_TRAIN, Y_TRAIN_SCALED)
            model = SklearnPredictor(m, self.scale_factor)
            regret = pyepo.metric.regret(model, self.optmodel, self.val_loader)
            if regret < best_regret:
                best_regret, best_model = regret, model
        return best_model


class LASSOTwoStage:
    """
    Prediction-focused (two-stage) least-squares LASSO baseline, evaluated
    through PyEPO.

    A LASSO (L1-regularized least squares) regressor is fit for each candidate
    regularization strength lambda; the strength minimizing PyEPO validation
    regret against the shared optimization model is selected. The chosen
    predictor (wrapped as an nn.Module) is exposed at self.model so it can be
    scored with pyepo.metric.regret exactly like the other solvers.
    """

    def __init__(self, optmodel, X_TRAIN, Y_TRAIN, X_VAL, Y_VAL,
                 lambdas=None, rng_seed: int = 0):
        """
        Inputs:
            optmodel: A PyEPO optModel (the shared shortest-path optimizer)
            X_TRAIN, Y_TRAIN (np.ndarray): Training covariates / costs
            X_VAL, Y_VAL (np.ndarray): Validation covariates / costs
            lambdas (iterable): Candidate L1 regularization strengths
            rng_seed (int): Seed for the underlying regressor
        """
        self.optmodel = optmodel
        self.X_TRAIN, self.Y_TRAIN = X_TRAIN, Y_TRAIN
        self.X_VAL, self.Y_VAL = X_VAL, Y_VAL
        self.lambdas = np.logspace(-6, 0, 10) if lambdas is None else lambdas
        self.rng_seed = rng_seed
        self.scale_factor = np.max(np.abs(Y_TRAIN))

        val_set = optDataset(optmodel, X_VAL, Y_VAL)
        self.val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)

        self.model = self.train()

    def train(self):
        Y_TRAIN_SCALED = self.Y_TRAIN / self.scale_factor

        best_model, best_regret = None, np.inf
        for lam in self.lambdas:
            base = Lasso(alpha=lam, max_iter=10000)
            m = MultiOutputRegressor(base)
            m.fit(self.X_TRAIN, Y_TRAIN_SCALED)
            model = SklearnPredictor(m, self.scale_factor)
            regret = pyepo.metric.regret(model, self.optmodel, self.val_loader)
            if regret < best_regret:
                best_regret, best_model = regret, model
        return best_model


class LinearSPOPlus:
    """
    Decision-focused (DFL) solver: a linear predictor trained end-to-end with
    PyEPO's SPO+ surrogate loss.

    Trains on init and exposes the trained predictor at self.model, so it can be
    scored with pyepo.metric.regret just like the two-stage GBM baseline.
    """

    def __init__(self, optmodel, X_TRAIN, Y_TRAIN, X_VAL, Y_VAL,
                 num_epochs: int = 20, lr: float = 1e-2, batch_size: int = 32,
                 rng_seed: int = 42):
        """
        Inputs:
            optmodel: A PyEPO optModel (the shared shortest-path optimizer)
            X_TRAIN, Y_TRAIN (np.ndarray): Training covariates / costs
            X_VAL, Y_VAL (np.ndarray): Validation covariates / costs (reserved for
                early stopping / tuning; unused in this basic loop)
            num_epochs (int): Training epochs
            lr (float): Adam learning rate
            batch_size (int): Minibatch size
            rng_seed (int): Torch rng_seed for reproducible initialization
        """
        self.optmodel = optmodel
        self.X_TRAIN, self.Y_TRAIN = X_TRAIN, Y_TRAIN
        self.X_VAL, self.Y_VAL = X_VAL, Y_VAL
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.rng_seed = rng_seed
        self.P = X_TRAIN.shape[1]
        self.D = Y_TRAIN.shape[1]

        self.model = self.train()

    def train(self):
        torch.manual_seed(self.rng_seed)
        model = nn.Linear(self.P, self.D)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        spop = SPOPlus(self.optmodel)

        train_set = optDataset(self.optmodel, self.X_TRAIN, self.Y_TRAIN)
        loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        model.train()
        for _ in range(self.num_epochs):
            for x, y, w, z in loader:
                Y_hat = model(x)
                loss = spop(Y_hat, y, w, z)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.eval()
        return model



