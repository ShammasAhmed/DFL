"""
Experiment orchestration for the shortest-path DFL comparison.

RegretExperiment sweeps a set of polynomial degrees, runs independent trials with
fresh data at each degree, trains each supplied solver, and collects the per-trial
test regret (%). It owns the shared optimization model and the data generation, so
the entry point only has to declare which solvers to compare and the config.
"""
import numpy as np
import torch
from itertools import combinations
import matplotlib.pyplot as plt

import pyepo
from pyepo.model.opt import optModel
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


class ContextExperiment:
    """
    Individual experiment: train each solver on shared train/val data, then have the
    models face one or more test contexts and report, per solver, the average over
    contexts of three quantities.

    For a test context X with realized costs Y and noiseless conditional mean
    f* = E[Y | X], let w(c) = argmin_w c^T w be the optimal decision under cost c and
    w_hat = w(c_hat) the decision the model makes from its predicted costs:

        Decision Loss     : Y^T w_hat                  (realized cost of the path)
        Regret rel. f*    : f*^T w_hat - f*^T w(f*)    (gap to the best policy)
        Regret rel. Y     : Y^T w_hat  - Y^T w(Y)      (gap to the clairvoyant oracle)

    f* and Y for the same X are obtained from PyEPO's genData with noise_width 0 and
    H respectively at a shared seed (same features and ground-truth B, differing only
    in noise).

    With shared_models=True (default) the solvers are trained once and every context
    is evaluated against those same models. With shared_models=False the solvers are
    retrained on a fresh DGP draw for each context.

    Inputs:
        optmodel: The shortest-path optimization model (single source of geometry)
        solvers (list): Ordered list of (key, solver_cls) pairs, constructed as
            solver_cls(optmodel, x_train, c_train, x_val, c_val, seed=seed)
        degree (int): Polynomial degree of the DGP
        n_contexts (int): Number of test contexts to average over
        shared_models (bool): Reuse one trained set of models across all contexts
        num_features (int): Feature dimension P
        noise_width (float): Multiplicative noise half-width H
        num_train, num_val (int): Split sizes for the training data. num_val defaults
            to num_train // 4 when left as None.
        rng_seed (int): Seed for data generation and solver initialization
    """

    def __init__(self, optmodel, solvers, degree, n_contexts=1, shared_models=True,
                 num_features=5, noise_width=0.5, num_train=100, num_val=None,
                 rng_seed=42):
        self.optmodel = optmodel
        self.grid = optmodel.grid
        self.solvers = list(solvers)
        self.degree = degree
        self.n_contexts = n_contexts
        self.shared_models = shared_models
        self.num_features = num_features
        self.noise_width = noise_width
        self.num_train = num_train
        self.num_val = num_train // 4 if num_val is None else num_val
        self.rng_seed = rng_seed
        self.table = None

    def _generate(self, seed, n_test):
        """Generate train/val data plus n_test contexts (features, Y, f*)."""
        ntr, nv = self.num_train, self.num_val
        n = ntr + nv + n_test
        # Same seed, two noise levels: noisy costs Y and noiseless mean f*.
        x, y = pyepo.data.shortestpath.genData(
            n, self.num_features, self.grid,
            deg=self.degree, noise_width=self.noise_width, seed=seed)
        _, fstar = pyepo.data.shortestpath.genData(
            n, self.num_features, self.grid,
            deg=self.degree, noise_width=0.0, seed=seed)
        train = (x[:ntr], y[:ntr], x[ntr:ntr + nv], y[ntr:ntr + nv])
        contexts = (x[ntr + nv:], y[ntr + nv:], fstar[ntr + nv:])
        return train, contexts

    def _train(self, train):
        x_tr, y_tr, x_val, y_val = train
        return {key: cls(self.optmodel, x_tr, y_tr, x_val, y_val, seed=self.rng_seed)
                for key, cls in self.solvers}

    def _decision(self, cost_vec):
        """Optimal decision (path incidence vector) under the given cost vector."""
        self.optmodel.setObj(np.asarray(cost_vec))
        sol, _ = self.optmodel.solve()
        return np.asarray(sol)

    def _predict(self, solver, x_row):
        """Predicted cost vector for one context's features."""
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(x_row).reshape(1, -1), dtype=torch.float32)
            c_hat = solver.model(x)
        return c_hat.detach().cpu().numpy().flatten()

    def _eval_context(self, trained, x_ctx, y_ctx, fstar_ctx):
        """
        Raw per-context pieces for one context.

        Returns:
            (opt_fstar, opt_Y): the two benchmark optimal costs, and
            per_solver: key -> (decision_loss, loss_fstar, loss_Y), where
                loss_fstar/loss_Y are the un-normalized regrets (numerators).
        """
        opt_fstar = float(fstar_ctx @ self._decision(fstar_ctx))
        opt_Y = float(y_ctx @ self._decision(y_ctx))
        per_solver = {}
        for key, solver in trained.items():
            w_hat = self._decision(self._predict(solver, x_ctx))
            realized = float(y_ctx @ w_hat)
            per_solver[key] = (realized,
                               float(fstar_ctx @ w_hat) - opt_fstar,
                               realized - opt_Y)
        return (opt_fstar, opt_Y), per_solver

    def _context_iter(self):
        """Yield (trained_models, x, y, f*) for each context."""
        if self.shared_models:
            train, (x_te, y_te, f_te) = self._generate(self.rng_seed, self.n_contexts)
            trained = self._train(train)
            for i in range(self.n_contexts):
                yield trained, x_te[i], y_te[i], f_te[i]
        else:
            for i in range(self.n_contexts):
                train, (x_te, y_te, f_te) = self._generate(self.rng_seed + i, 1)
                trained = self._train(train)
                yield trained, x_te[0], y_te[0], f_te[0]

    def run(self):
        """
        Train and evaluate over all contexts.

        Returns:
            table (dict): key -> {"decision_loss", "regret_fstar", "regret_Y"}.
                decision_loss is the mean realized cost; the regrets are percentages,
                pooled as 100 * (sum of regrets) / (sum of optimal costs) over the
                contexts. Also stored on self.table.
        """
        sums = {key: {"decision_loss": 0.0, "loss_fstar": 0.0, "loss_Y": 0.0}
                for key, _ in self.solvers}
        opt_fstar_total = 0.0
        opt_Y_total = 0.0
        n = 0

        for trained, x_ctx, y_ctx, f_ctx in self._context_iter():
            (opt_fstar, opt_Y), per_solver = self._eval_context(
                trained, x_ctx, y_ctx, f_ctx)
            opt_fstar_total += opt_fstar
            opt_Y_total += opt_Y
            for key, (dl, lf, ly) in per_solver.items():
                sums[key]["decision_loss"] += dl
                sums[key]["loss_fstar"] += lf
                sums[key]["loss_Y"] += ly
            n += 1

        self.table = {
            key: {
                "decision_loss": sums[key]["decision_loss"] / n,
                "regret_fstar": 100 * sums[key]["loss_fstar"] / opt_fstar_total,
                "regret_Y": 100 * sums[key]["loss_Y"] / opt_Y_total,
            }
            for key, _ in self.solvers
        }
        return self.table

    def print_table(self):
        """Run the experiment (if needed) and print the averaged results table."""
        table = self.table if self.table is not None else self.run()
        header = f"{'Model':<10}{'Decision Loss':>15}{'Regret vs f*':>15}{'Regret vs Y':>15}"
        print(header)
        print("-" * len(header))
        for key, _ in self.solvers:
            row = table[key]
            print(f"{key:<10}{row['decision_loss']:>15.4f}"
                  f"{row['regret_fstar']:>14.4f}%{row['regret_Y']:>14.4f}%")
        mode = "shared" if self.shared_models else "per-context"
        print(f"(over {self.n_contexts} contexts, {mode} models; "
              f"decision loss is mean realized cost, regrets are pooled %)")
        return table

class HistogramExperiment:
    """
    Evaluates how often a learner selects specific paths across varying training sets
    for a SINGLE, FIXED test context.
    
    Uses standard PyEPO solvers for training, but completely bypasses Gurobi during 
    evaluation using a combinatorial path-edge incidence matrix.
    """

    def __init__(self, optmodel, solvers, degree, n_trials=50,
                 num_features=5, noise_width=0.5, num_train=100, num_val=None,
                 rng_seed=42):
        self.optmodel = optmodel # Pass the real PyEPO optmodel here
        self.solvers = list(solvers)
        self.degree = degree
        self.n_trials = n_trials
        self.num_features = num_features
        self.noise_width = noise_width
        self.num_train = num_train
        self.num_val = num_train // 4 if num_val is None else num_val
        self.rng_seed = rng_seed
        
        self.all_paths = None
        self.n_paths = None
        self.fixed_x = None
        self.fixed_y = None
        self.fixed_fstar = None
        self.results = None

    def compute_all_paths(self, grid):
        """Enumerate all monotone paths using your combinatorial layout."""
        height, width = grid
        right_moves = width - 1
        down_moves = height - 1
        total_moves = right_moves + down_moves
        total_edges = height * right_moves + down_moves * width
        
        paths_list = []
        for positions in combinations(range(total_moves), down_moves):
            path = ['R'] * total_moves
            for pos in positions:
                path[pos] = 'D'
            paths_list.append(''.join(path))
            
        path_matrix = np.zeros((len(paths_list), total_edges))
        vertical_offset = height * right_moves
        
        for idx, path in enumerate(paths_list):
            row, col = 0, 0
            for move in path:
                if move == 'R':
                    path_matrix[idx, row * right_moves + col] = 1
                    col += 1
                elif move == 'D':
                    path_matrix[idx, vertical_offset + row * width + col] = 1
                    row += 1
                    
        self.all_paths = path_matrix
        self.n_paths = len(paths_list)
        return self.all_paths

    def _generate_fixed_context(self, grid):
        x, y = pyepo.data.shortestpath.genData(
            1, self.num_features, grid, deg=self.degree, noise_width=self.noise_width, seed=self.rng_seed
        )
        _, fstar = pyepo.data.shortestpath.genData(
            1, self.num_features, grid, deg=self.degree, noise_width=0.0, seed=self.rng_seed
        )
        self.fixed_x = x[0]
        self.fixed_y = y[0]
        self.fixed_fstar = fstar[0]

    def _generate_train_data(self, seed, grid):
        ntr, nv = self.num_train, self.num_val
        n = ntr + nv
        x, y = pyepo.data.shortestpath.genData(
            n, self.num_features, grid, deg=self.degree, noise_width=self.noise_width, seed=seed
        )
        return x[:ntr], y[:ntr], x[ntr:], y[ntr:]

    def _predict(self, solver, x_row):
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(x_row).reshape(1, -1), dtype=torch.float32)
            c_hat = solver.model(x)
        return c_hat.detach().cpu().numpy().flatten()

    def _get_shortest_path_index(self, cost_vec):
        """Bypasses Gurobi entirely using vectorized matrix multiplication."""
        path_costs = self.all_paths @ cost_vec
        return np.argmin(path_costs)

    def run(self, grid):
        if self.all_paths is None:
            self.compute_all_paths(grid)
            
        self._generate_fixed_context(grid)
        
        true_path_costs = self.all_paths @ self.fixed_fstar
        sorted_indices = np.argsort(true_path_costs)
        
        path_counts = {key: np.zeros(self.n_paths, dtype=int) for key, _ in self.solvers}

        for trial in range(self.n_trials):
            trial_seed = self.rng_seed + 1000 + trial
            x_tr, y_tr, x_val, y_val = self._generate_train_data(trial_seed, grid)
            
            for key, cls in self.solvers:
                # 1. Pass the actual, real optmodel into PyEPO solvers for training
                solver = cls(self.optmodel, x_tr, y_tr, x_val, y_val, seed=trial_seed)
                
                # 2. Get predictions
                c_hat = self._predict(solver, self.fixed_x)
                
                # 3. Solve using matrix algebra argmin instead of calling Gurobi
                chosen_idx = self._get_shortest_path_index(c_hat)
                path_counts[key][chosen_idx] += 1

        self.results = {
            "true_path_costs": true_path_costs,
            "sorted_indices": sorted_indices,
            "counts": path_counts,
            "true_optimal_idx": sorted_indices[0]
        }
        return self.results

    def plot_histogram(self):
        """
        Generates the superimposed line graph (true costs) and histogram (selections)
        separately for each solver evaluated. 
        
        Features:
          - Blue 'x' markers indicating exact costs on the line curve.
          - Total relative regret percentage included dynamically in the title.
        """
        if self.results is None:
            raise ValueError("Experiment has not been run yet. Please execute run(grid) first.")
            
        res = self.results
        sorted_idx = res["sorted_indices"]
        sorted_costs = res["true_path_costs"][sorted_idx]
        x_axis = np.arange(self.n_paths)
        
        # The cost of the absolutely optimal path under f*
        z_star = sorted_costs[0]

        for key in res["counts"].keys():
            # Counts of how many times this solver picked each path across all trials
            sorted_counts = res["counts"][key][sorted_idx]
            
            # --- Regret Calculation ---
            # Total cost incurred by the model across all trials
            total_realized_cost = np.sum(sorted_counts * sorted_costs)
            # If the model chose perfectly every trial, the total cost would be:
            total_optimal_cost = self.n_trials * z_star
            
            # Relative Regret % = (Sum(Loss) / Sum(Optimals)) * 100
            total_loss = total_realized_cost - total_optimal_cost
            relative_regret_pct = (total_loss / total_optimal_cost) * 100 if total_optimal_cost > 0 else 0.0
            
            # --- Plot Generation ---
            fig, ax1 = plt.subplots(figsize=(11, 5))

            # Primary Axis (Left): Line Graph of True Path Costs
            color_line = 'tab:blue'
            ax1.set_xlabel('Paths (Sorted by True Expected Cost Ascending)', fontsize=11)
            ax1.set_ylabel('True Expected Path Cost ($f^{*T} w$)', color=color_line, fontsize=11)
            
            # Plot the line and superimpose blue 'x' markers at each path cost coordinate
            ax1.plot(x_axis, sorted_costs, color=color_line, linewidth=2.5, label='Path Cost Curve')
            ax1.scatter(x_axis, sorted_costs, color='blue', marker='x', s=40, zorder=3, label='Path Cost Point')
            
            ax1.tick_params(axis='y', labelcolor=color_line)
            ax1.grid(True, alpha=0.3, linestyle=':')
            
            # Highlight the true optimal path (always index 0 on the sorted plot)
            ax1.axvline(x=0, color='crimson', linestyle='--', alpha=0.8, label='True Optimal Path')

            # Secondary Axis (Right): Histogram of Model Selections
            ax2 = ax1.twinx()  
            color_bar = 'tab:orange'
            ax2.set_ylabel('Selection Count (across training sets)', color=color_bar, fontsize=11)
            ax2.bar(x_axis, sorted_counts, color=color_bar, alpha=0.5, width=0.8, label='Model Selections')
            ax2.tick_params(axis='y', labelcolor=color_bar)

            # Title incorporating the relative regret percentage format
            plt.title(f"Solver Path Selection Profile: {key}\n"
                      f"Relative Regret: {relative_regret_pct:.2f}% | ({self.n_trials} Trials, Single Fixed Context)", 
                      fontsize=13, fontweight='bold')
            
            fig.tight_layout()
            plt.show()