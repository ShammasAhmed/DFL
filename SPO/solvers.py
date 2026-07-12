import numpy as np
from sklearn.linear_model import Lasso
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

import torch
from torch import nn
from torch.utils.data import DataLoader
import pyepo
from pyepo.data.dataset import optDataset
from pyepo.func import SPOPlus


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
        Y_TRAIN_SCALED = self.Y_TRAIN / self.scale_factor

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



