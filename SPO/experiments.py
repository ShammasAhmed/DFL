"""
Experiment orchestration for the shortest-path DFL comparison.

RegretExperiment sweeps a set of polynomial degrees, runs independent trials with
fresh data at each degree, trains each supplied solver, and collects the per-trial
test regret (%). It owns the shared optimization model and the data generation, so
the entry point only has to declare which solvers to compare and the config.
"""
import numpy as np

import pyepo
from pyepo.data.dataset import optDataset
from torch.utils.data import DataLoader


class RegretExperiment:
    """
    Runs the degree sweep x trials and returns per-trial regrets for each solver.

    Inputs:
        optmodel: The shortest-path optimization model, passed in by the caller.
            Its `.grid` attribute is used to generate matching cost data, so the
            optmodel is the single source of truth for the problem geometry.
        solvers (list): Ordered list of (key, solver_cls) pairs. Each solver_cls is
            constructed as solver_cls(optmodel, x_train, c_train, x_val, c_val,
            seed=seed) and must expose a `.model` scoreable by pyepo.metric.regret.
        num_features (int): Feature dimension P
        noise_width (float): Multiplicative noise half-width H
        num_train, num_val, num_test (int): Split sizes. num_val defaults to
            num_train // 4 when left as None.
        num_trials (int): Independent trials per degree
        degrees (iterable): Polynomial degrees of the DGP to sweep
        rng_seed (int): Base seed; each (degree, trial) gets a distinct seed
        verbose (bool): Print median regrets per degree as the sweep runs
    """

    def __init__(self, optmodel, solvers, num_features=5, noise_width=0.5,
                 num_train=100, num_val=None, num_test=1000, num_trials=50,
                 degrees=(1, 2, 4, 6, 8), rng_seed=42, verbose=True):
        self.optmodel = optmodel
        self.grid = optmodel.grid
        self.solvers = list(solvers)
        self.num_features = num_features
        self.noise_width = noise_width
        self.num_train = num_train
        self.num_val = num_train // 4 if num_val is None else num_val
        self.num_test = num_test
        self.num_trials = num_trials
        self.degrees = list(degrees)
        self.rng_seed = rng_seed
        self.verbose = verbose

        self.results = None

    def run(self):
        """
        Execute the sweep.

        Returns:
            results (dict): results[degree][solver_key] -> list of per-trial
                test regrets (%). Also stored on self.results.
        """
        results = {deg: {key: [] for key, _ in self.solvers} for deg in self.degrees}
        n = self.num_train + self.num_val + self.num_test

        for deg in self.degrees:
            for trial in range(self.num_trials):
                # Distinct DGP per (degree, trial); one genData call so B is
                # shared across the train/val/test split within a trial.
                seed = self.rng_seed + 1000 * deg + trial
                x, c = pyepo.data.shortestpath.genData(
                    n, self.num_features, self.grid,
                    deg=deg, noise_width=self.noise_width, seed=seed,
                )
                x_train, c_train, x_val, c_val, x_test, c_test = self._split(x, c)

                test_set = optDataset(self.optmodel, x_test, c_test)
                test_loader = DataLoader(test_set, batch_size=len(test_set),
                                         shuffle=False)

                for key, solver_cls in self.solvers:
                    solver = solver_cls(self.optmodel, x_train, c_train,
                                        x_val, c_val, seed=seed)
                    regret = 100 * pyepo.metric.regret(
                        solver.model, self.optmodel, test_loader)
                    results[deg][key].append(regret)

            if self.verbose:
                self._print_medians(deg, results)

        self.results = results
        return results

    def _split(self, x, c):
        ntr, nv = self.num_train, self.num_val
        return (x[:ntr], c[:ntr],
                x[ntr:ntr + nv], c[ntr:ntr + nv],
                x[ntr + nv:], c[ntr + nv:])

    def _print_medians(self, deg, results):
        parts = [f"{key} median {np.median(results[deg][key]):7.4f}%"
                 for key, _ in self.solvers]
        print(f"DEG {deg:>2}: " + "  |  ".join(parts) +
              f"  ({self.num_trials} trials)")
