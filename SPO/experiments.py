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
            constructed as solver_cls(optmodel, X_TRAIN, Y_TRAIN, X_VAL, Y_VAL,
            rng_seed=rng_seed) and must expose a `.model` scoreable by pyepo.metric.regret.
        P (int): Covariate dimension
        h (float): Multiplicative noise half-width
        num_train, num_val, num_test (int): Split sizes. num_val defaults to
            num_train // 4 when left as None.
        NUM_TRIALS (int): Independent trials per degree
        degrees (iterable): Polynomial degrees of the DGP to sweep
        rng_seed (int): Base seed; each (degree, trial) gets a distinct seed
        verbose (bool): Print median regrets per degree as the sweep runs
    """

    def __init__(self, optmodel, solvers, P=5, h=0.5,
                 num_train=100, num_val=None, num_test=1000, NUM_TRIALS=50,
                 degrees=(1, 2, 4, 6, 8), rng_seed=42, verbose=True):
        self.optmodel = optmodel
        self.grid = optmodel.grid
        self.solvers = list(solvers)
        self.P = P
        self.h = h
        self.num_train = num_train
        self.num_val = num_train // 4 if num_val is None else num_val
        self.num_test = num_test
        self.NUM_TRIALS = NUM_TRIALS
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
            for trial in range(self.NUM_TRIALS):
                # Distinct DGP per (degree, trial); one genData call so B is
                # shared across the train/val/test split within a trial.
                seed = self.rng_seed + 1000 * deg + trial
                X, Y = pyepo.data.shortestpath.genData(
                    n, self.P, self.grid,
                    deg=deg, noise_width=self.h, seed=seed,
                )
                X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, X_TEST, Y_TEST = self._split(X, Y)

                test_set = optDataset(self.optmodel, X_TEST, Y_TEST)
                test_loader = DataLoader(test_set, batch_size=len(test_set),
                                         shuffle=False)

                for key, solver_cls in self.solvers:
                    solver = solver_cls(self.optmodel, X_TRAIN, Y_TRAIN,
                                        X_VAL, Y_VAL, rng_seed=seed)
                    regret = 100 * pyepo.metric.regret(
                        solver.model, self.optmodel, test_loader)
                    results[deg][key].append(regret)

            if self.verbose:
                self._print_medians(deg, results)

        self.results = results
        return results

    def _split(self, X, Y):
        ntr, nv = self.num_train, self.num_val
        return (X[:ntr], Y[:ntr],
                X[ntr:ntr + nv], Y[ntr:ntr + nv],
                X[ntr + nv:], Y[ntr + nv:])

    def _print_medians(self, deg, results):
        parts = [f"{key} median {np.median(results[deg][key]):7.4f}%"
                 for key, _ in self.solvers]
        print(f"DEG {deg:>2}: " + "  |  ".join(parts) +
              f"  ({self.NUM_TRIALS} trials)")


class ContextExperiment:
    """
    Individual experiment: train each solver on shared train/val data, then have the
    models face one or more test contexts and report, per solver, the average over
    contexts of three quantities.

    For a test context X with realized costs Y and noiseless conditional mean
    f* = E[Y | X], let w(c) = argmin_w c^T w be the optimal decision under cost c and
    w_hat = w(Y_hat) the decision the model makes from its predicted costs:

        Decision Loss     : Y^T w_hat                  (realized cost of the path)
        Regret rel. f*    : f*^T w_hat - f*^T w(f*)    (gap to the best policy)
        Regret rel. Y     : Y^T w_hat  - Y^T w(Y)      (gap to the clairvoyant oracle)

    f* and Y for the same X are obtained from PyEPO's genData with noise_width 0 and
    h respectively at a shared seed (same covariates and ground-truth B, differing only
    in noise).

    With shared_models=True (default) the solvers are trained once and every context
    is evaluated against those same models. With shared_models=False the solvers are
    retrained on a fresh DGP draw for each context.

    Inputs:
        optmodel: The shortest-path optimization model (single source of geometry)
        solvers (list): Ordered list of (key, solver_cls) pairs, constructed as
            solver_cls(optmodel, X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, rng_seed=rng_seed)
        deg (int): Polynomial degree of the DGP
        num_contexts (int): Number of test contexts to average over
        shared_models (bool): Reuse one trained set of models across all contexts
        P (int): Covariate dimension
        h (float): Multiplicative noise half-width
        num_train, num_val (int): Split sizes for the training data. num_val defaults
            to num_train // 4 when left as None.
        rng_seed (int): Seed for data generation and solver initialization
    """

    def __init__(self, optmodel, solvers, deg, num_contexts=1, shared_models=True,
                 P=5, h=0.5, num_train=100, num_val=None,
                 rng_seed=42):
        self.optmodel = optmodel
        self.grid = optmodel.grid
        self.solvers = list(solvers)
        self.deg = deg
        self.num_contexts = num_contexts
        self.shared_models = shared_models
        self.P = P
        self.h = h
        self.num_train = num_train
        self.num_val = num_train // 4 if num_val is None else num_val
        self.rng_seed = rng_seed
        self.table = None

    def _generate(self, seed, num_test):
        """Generate train/val data plus num_test contexts (covariates, Y, f*)."""
        ntr, nv = self.num_train, self.num_val
        n = ntr + nv + num_test
        # Same seed, two noise levels: noisy costs Y and noiseless mean f*.
        X, Y = pyepo.data.shortestpath.genData(
            n, self.P, self.grid,
            deg=self.deg, noise_width=self.h, seed=seed)
        _, fstar_X = pyepo.data.shortestpath.genData(
            n, self.P, self.grid,
            deg=self.deg, noise_width=0.0, seed=seed)
        train = (X[:ntr], Y[:ntr], X[ntr:ntr + nv], Y[ntr:ntr + nv])
        test = (X[ntr + nv:], Y[ntr + nv:], fstar_X[ntr + nv:])
        return train, test

    def _train(self, train):
        X_TR, Y_TR, X_VAL, Y_VAL = train
        return {key: cls(self.optmodel, X_TR, Y_TR, X_VAL, Y_VAL, rng_seed=self.rng_seed)
                for key, cls in self.solvers}

    def _decision(self, cost_vec):
        """Optimal decision (path incidence vector) under the given cost vector."""
        self.optmodel.setObj(np.asarray(cost_vec))
        sol, _ = self.optmodel.solve()
        return np.asarray(sol)

    def _compute_paths(self, grid):
        """
        Enumerate every monotone (right/down) source-to-sink path on the grid as an
        (num_paths x num_edges) incidence matrix, cached on self.path_matrix.

        Columns are indexed straight from optmodel.arcs (the cost-vector edge order),
        so a row of this matrix is directly comparable to the incidence vector returned
        by the Gurobi-backed _decision, regardless of PyEPO's internal edge layout.
        Nodes are row-major (node = row*width + col); a right move is arc (node, node+1)
        and a down move is arc (node, node+width).
        """
        height, width = grid
        arc_index = {tuple(arc): j for j, arc in enumerate(self.optmodel.arcs)}
        num_edges = len(self.optmodel.arcs)

        right_moves = width - 1
        down_moves = height - 1
        total_moves = right_moves + down_moves

        paths_list = []
        for positions in combinations(range(total_moves), down_moves):
            path = ['R'] * total_moves
            for pos in positions:
                path[pos] = 'D'
            paths_list.append(path)

        path_matrix = np.zeros((len(paths_list), num_edges))
        for idx, path in enumerate(paths_list):
            row, col = 0, 0
            for move in path:
                node = row * width + col
                if move == 'R':
                    path_matrix[idx, arc_index[(node, node + 1)]] = 1
                    col += 1
                else:  # 'D'
                    path_matrix[idx, arc_index[(node, node + width)]] = 1
                    row += 1

        self.path_matrix = path_matrix
        return path_matrix

    def _decision_argmin(self, cost_vec):
        """
        Optimal decision via the precomputed path enumeration instead of Gurobi.

        Minimizes path cost over self.path_matrix and returns the winning path's
        incidence vector. Builds the matrix on first use. Intended as a drop-in
        replacement for _decision; use _verify_decision to confirm they agree before
        switching over.
        """
        if getattr(self, "path_matrix", None) is None:
            self._compute_paths(self.grid)
        cost_vec = np.asarray(cost_vec)
        return self.path_matrix[np.argmin(self.path_matrix @ cost_vec)]

    def _verify_decision(self, num_checks=100, seed=0):
        """
        Sanity check that _decision_argmin reproduces the Gurobi _decision exactly on
        random cost vectors. Returns True if every incidence vector matches. Uses
        continuous costs so optimal-path ties (measure zero) don't cause spurious
        mismatches.
        """
        num_edges = self._compute_paths(self.grid).shape[1]
        rng = np.random.default_rng(seed)
        for _ in range(num_checks):
            cost = rng.random(num_edges)
            if not np.array_equal(self._decision(cost), self._decision_argmin(cost)):
                return False
        return True

    def _predict(self, solver, x_row):
        """Predicted cost vector for one context's covariates."""
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(x_row).reshape(1, -1), dtype=torch.float32)
            Y_hat = solver.model(x)
        return Y_hat.detach().cpu().numpy().flatten()

    # def _eval_context(self, trained, x_ctx, y_ctx, fstar_X):
    #     """
    #     Raw per-context pieces for one context.

    #     Returns:
    #         (opt_fstar, opt_Y): the two benchmark optimal costs, and
    #         per_solver: key -> (loss_Y, regret_fstar, regret_Y), where
    #             regret_fstar/regret_Y are the un-normalized regrets (numerators).
    #     """
    #     opt_fstar = float(fstar_X @ self._decision(fstar_X))
    #     opt_Y = float(y_ctx @ self._decision(y_ctx))
    #     per_solver = {}
    #     for key, solver in trained.items():
    #         w_hat = self._decision(self._predict(solver, x_ctx))
    #         loss_Y = float(y_ctx @ w_hat)
    #         per_solver[key] = (loss_Y,
    #                            float(fstar_X @ w_hat) - opt_fstar,
    #                            realized - opt_Y)
    #     return (opt_fstar, opt_Y), per_solver

    def _eval_context(self, trained, x_ctx, y_ctx, fstar_X):
        """
        Raw per-context pieces for one context.

        Returns:
            (opt_fstar, opt_Y): the two benchmark optimal costs, and
            per_solver: key -> (loss_Y, regret_fstar, regret_Y), where
                regret_fstar/regret_Y are the un-normalized regrets (numerators).
        """
        # <E[Y|X], z*(E[Y|X])>
        opt_fstar = float(fstar_X @ self._decision_argmin(fstar_X))

        # z*(Y)
        w_hat_y = self._decision_argmin(y_ctx)

        # <Y, z*(Y)>
        opt_Y = float(y_ctx @ w_hat_y)

        per_solver = {}
        for key, solver in trained.items():
            # z*(f^{Method}(X))
            w_hat = self._decision_argmin(self._predict(solver, x_ctx))

            # <Y, z*(f^{Method)(X)>
            loss_Y = float(y_ctx @ w_hat)

            # 1. <Y, z*(f^{method}(X))>
            # 2. <E[Y|X], z*(f^{method}(X))> - <E[Y|X], z*(E[Y|X])>
            # 3. <E[Y|X], z*(f^{method}(X))> - <E[Y|X], z*(Y)>
            per_solver[key] = (loss_Y,
                               float(fstar_X @ w_hat) - opt_fstar,
                               float(fstar_X @ w_hat) - float(fstar_X @ w_hat_y))
            # per_solver[key] = (loss_Y,
            #                    float(fstar_X @ w_hat) - opt_fstar,
            #                    float(fstar_X @ w_hat) - float(fstar_X @ w_hat_y))
        return (opt_fstar, opt_Y), per_solver

    def _context_iter(self):
        """Yield (trained_models, x, y, f*) for each context."""
        if self.shared_models:
            train, (X_TE, Y_TE, fstar_X) = self._generate(self.rng_seed, self.num_contexts)
            trained = self._train(train)
            for i in range(self.num_contexts):
                yield trained, X_TE[i], Y_TE[i], fstar_X[i]
        else:
            for i in range(self.num_contexts):
                train, (X_TE, Y_TE, fstar_X) = self._generate(self.rng_seed + i, 1)
                trained = self._train(train)
                yield trained, X_TE[0], Y_TE[0], fstar_X[0]

    def run(self):
        """
        Train and evaluate over all contexts.

        Returns:
            table (dict): key -> {"loss_Y", "regret_fstar", "regret_Y"}.
                loss_Y is the mean realized cost; the regrets are percentages,
                pooled as 100 * (sum of regrets) / (sum of optimal costs) over the
                contexts. Also stored on self.table.
        """
        sums = {key: {"loss_Y": 0.0, "regret_fstar": 0.0, "regret_Y": 0.0}
                for key, _ in self.solvers}
        opt_fstar_total = 0.0
        opt_Y_total = 0.0
        n = 0

        for trained, x_ctx, y_ctx, fstar_X in self._context_iter():
            (opt_fstar, opt_Y), per_solver = self._eval_context(
                trained, x_ctx, y_ctx, fstar_X)
            opt_fstar_total += opt_fstar
            opt_Y_total += opt_Y
            for key, (loss_Y, regret_fstar, regret_Y) in per_solver.items():
                sums[key]["loss_Y"] += loss_Y
                sums[key]["regret_fstar"] += regret_fstar
                sums[key]["regret_Y"] += regret_Y
            n += 1

        self.table = {
            key: {
                "loss_Y": sums[key]["loss_Y"] / n,
                "regret_fstar": 100 * sums[key]["regret_fstar"] / opt_fstar_total,
                "regret_Y": 100 * sums[key]["regret_Y"] / opt_Y_total,
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
            print(f"{key:<10}{row['loss_Y']:>15.4f}"
                  f"{row['regret_fstar']:>14.4f}%{row['regret_Y']:>14.4f}%")
        mode = "shared" if self.shared_models else "per-context"
        print(f"(over {self.num_contexts} contexts, {mode} models; "
              f"decision loss is mean realized cost, regrets are pooled %)")
        return table

class HistogramExperiment:
    """
    Evaluates how often a learner selects specific paths across varying training sets
    for a SINGLE, FIXED test context.
    
    Uses standard PyEPO solvers for training, but completely bypasses Gurobi during 
    evaluation using a combinatorial path-edge incidence matrix.
    """

    def __init__(self, optmodel, solvers, deg, NUM_TRIALS=50,
                 P=5, h=0.5, num_train=100, num_val=None,
                 rng_seed=42):
        self.optmodel = optmodel # Pass the real PyEPO optmodel here
        self.solvers = list(solvers)
        self.deg = deg
        self.NUM_TRIALS = NUM_TRIALS
        self.P = P
        self.h = h
        self.num_train = num_train
        self.num_val = num_train // 4 if num_val is None else num_val
        self.rng_seed = rng_seed
        
        self.path_matrix = None
        self.num_paths = None
        self.fixed_x = None
        self.fixed_y = None
        self.fstar_X = None
        self.results = None

    def compute_paths(self, grid):
        """
        Enumerate all monotone (right/down) paths as a (num_paths x num_edges)
        incidence matrix, cached on self.path_matrix.

        Column order reproduces PyEPO's shortestPathModel arc ordering purely from the
        grid (no PyEPO object needed): edges are laid out interleaved by row -- for each
        row, its horizontal edges first, then the vertical edges leaving that row. Nodes
        are row-major (node = row*width + col), so a right move is arc (node, node+1) and
        a down move is arc (node, node+width). This matches the cost-vector edge order,
        so a row of this matrix is comparable to a Gurobi shortest-path solution.
        """
        height, width = grid
        right_moves = width - 1
        down_moves = height - 1
        total_moves = right_moves + down_moves

        # Column index for every directed arc, in PyEPO's edge order.
        arc_index = {}
        j = 0
        for r in range(height):
            for c in range(width - 1):
                node = r * width + c
                arc_index[(node, node + 1)] = j
                j += 1
            if r < height - 1:
                for c in range(width):
                    node = r * width + c
                    arc_index[(node, node + width)] = j
                    j += 1
        num_edges = j

        paths_list = []
        for positions in combinations(range(total_moves), down_moves):
            path = ['R'] * total_moves
            for pos in positions:
                path[pos] = 'D'
            paths_list.append(path)

        path_matrix = np.zeros((len(paths_list), num_edges))
        for idx, path in enumerate(paths_list):
            row, col = 0, 0
            for move in path:
                node = row * width + col
                if move == 'R':
                    path_matrix[idx, arc_index[(node, node + 1)]] = 1
                    col += 1
                else:  # 'D'
                    path_matrix[idx, arc_index[(node, node + width)]] = 1
                    row += 1

        self.path_matrix = path_matrix
        self.num_paths = len(paths_list)
        return self.path_matrix

    def _generate_fixed_context(self, grid):
        X, Y = pyepo.data.shortestpath.genData(
            1, self.P, grid, deg=self.deg, noise_width=self.h, seed=self.rng_seed
        )
        _, fstar_X = pyepo.data.shortestpath.genData(
            1, self.P, grid, deg=self.deg, noise_width=0.0, seed=self.rng_seed
        )
        self.fixed_x = X[0]
        self.fixed_y = Y[0]
        self.fstar_X = fstar_X[0]

    def _generate_train_data(self, seed, grid):
        ntr, nv = self.num_train, self.num_val
        n = ntr + nv
        X, Y = pyepo.data.shortestpath.genData(
            n, self.P, grid, deg=self.deg, noise_width=self.h, seed=seed
        )
        return X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]

    def _predict(self, solver, x_row):
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(x_row).reshape(1, -1), dtype=torch.float32)
            Y_hat = solver.model(x)
        return Y_hat.detach().cpu().numpy().flatten()

    def _get_shortest_path_index(self, cost_vec):
        """Bypasses Gurobi entirely using vectorized matrix multiplication."""
        path_costs = self.path_matrix @ cost_vec
        return np.argmin(path_costs)

    def run(self, grid):
        if self.path_matrix is None:
            self.compute_paths(grid)
            
        self._generate_fixed_context(grid)
        
        true_path_costs = self.path_matrix @ self.fstar_X
        sorted_indices = np.argsort(true_path_costs)
        
        path_counts = {key: np.zeros(self.num_paths, dtype=int) for key, _ in self.solvers}

        for trial in range(self.NUM_TRIALS):
            trial_seed = self.rng_seed + 1000 + trial
            X_TR, Y_TR, X_VAL, Y_VAL = self._generate_train_data(trial_seed, grid)
            
            for key, cls in self.solvers:
                # 1. Pass the actual, real optmodel into PyEPO solvers for training
                solver = cls(self.optmodel, X_TR, Y_TR, X_VAL, Y_VAL, rng_seed=trial_seed)
                
                # 2. Get predictions
                Y_hat = self._predict(solver, self.fixed_x)
                
                # 3. Solve using matrix algebra argmin instead of calling Gurobi
                chosen_idx = self._get_shortest_path_index(Y_hat)
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
        
        Also generates a final summary plot superimposing all solvers' selections
        on a single graph for direct comparison.
        
        Features:
          - Blue 'x' markers indicating exact costs on the line curve.
          - Total relative regret percentage included dynamically in the title.
        """
        if self.results is None:
            raise ValueError("Experiment has not been run yet. Please execute run(grid) first.")
            
        res = self.results
        sorted_idx = res["sorted_indices"]
        sorted_costs = res["true_path_costs"][sorted_idx]
        x_axis = np.arange(self.num_paths)
        
        # The cost of the absolutely optimal path under f*
        z_star = sorted_costs[0]

        # --- Part 1: Individual Solver Plots ---
        for key in res["counts"].keys():
            # Counts of how many times this solver picked each path across all trials
            sorted_counts = res["counts"][key][sorted_idx]
            
            # --- Regret Calculation ---
            total_loss_fstar = np.sum(sorted_counts * sorted_costs)
            total_optimal_cost = self.NUM_TRIALS * z_star
            total_regret = total_loss_fstar - total_optimal_cost
            relative_regret_pct = (total_regret / total_optimal_cost) * 100 if total_optimal_cost > 0 else 0.0
            
            # --- Plot Generation ---
            fig, ax1 = plt.subplots(figsize=(11, 5))

            # Primary Axis (Left): Line Graph of True Path Costs
            color_line = 'tab:blue'
            ax1.set_xlabel('Paths (Sorted by True Expected Cost Ascending)', fontsize=11)
            ax1.set_ylabel('True Expected Path Cost ($f^{*T} w$)', color=color_line, fontsize=11)
            
            ax1.plot(x_axis, sorted_costs, color=color_line, linewidth=2.5, label='Path Cost Curve')
            ax1.scatter(x_axis, sorted_costs, color='blue', marker='x', s=40, zorder=3, label='Path Cost Point')
            
            ax1.tick_params(axis='y', labelcolor=color_line)
            ax1.grid(True, alpha=0.3, linestyle=':')
            ax1.axvline(x=0, color='crimson', linestyle='--', alpha=0.8, label='True Optimal Path')

            # Secondary Axis (Right): Histogram of Model Selections
            ax2 = ax1.twinx()  
            color_bar = 'tab:orange'
            ax2.set_ylabel('Selection Count (across training sets)', color=color_bar, fontsize=11)
            ax2.bar(x_axis, sorted_counts, color=color_bar, alpha=0.5, width=0.8, label='Model Selections')
            ax2.tick_params(axis='y', labelcolor=color_bar)

            plt.title(f"Solver Path Selection Profile: {key}\n"
                      f"Relative Regret: {relative_regret_pct:.2f}% | ({self.NUM_TRIALS} Trials, Single Fixed Context)", 
                      fontsize=13, fontweight='bold')
            
            fig.tight_layout()
            plt.show()

        # --- Part 2: Side-by-Side Multi-Solver Comparison Plot ---
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Left axis remains the continuous Path Cost Curve
        color_line = 'tab:blue'
        ax1.set_xlabel('Paths (Sorted by True Expected Cost Ascending)', fontsize=11)
        ax1.set_ylabel('True Expected Path Cost ($f^{*T} w$)', color=color_line, fontsize=11)
        
        ax1.plot(x_axis, sorted_costs, color=color_line, linewidth=2.5, label='Path Cost Curve')
        ax1.scatter(x_axis, sorted_costs, color='blue', marker='x', s=40, zorder=3)
        ax1.axvline(x=0, color='crimson', linestyle='--', alpha=0.8, label='True Optimal Path')
        ax1.tick_params(axis='y', labelcolor=color_line)
        ax1.grid(True, alpha=0.3, linestyle=':')

        # Right axis hosts all the histograms side-by-side at full opacity
        ax2 = ax1.twinx()
        ax2.set_ylabel('Selection Count (All Solvers Comparison)', color='black', fontsize=11)
        
        comparison_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # --- Side-by-Side Width Geometry ---
        num_solvers = len(res["counts"])
        total_group_width = 0.8  # Total space allocated for all bars combined per path
        individual_bar_width = total_group_width / num_solvers
        
        for idx, (key, counts) in enumerate(res["counts"].items()):
            sorted_counts = counts[sorted_idx]
            color = comparison_colors[idx % len(comparison_colors)]
            
            # Compute the offset shift for this specific solver's bar
            # Centers the grouped cluster over the true x-coordinate index
            offset = (idx - (num_solvers - 1) / 2) * individual_bar_width
            
            # alpha=1.0 keeps the colors solid, vivid, and completely unmixed
            ax2.bar(x_axis + offset, sorted_counts, color=color, alpha=1.0, 
                    width=individual_bar_width, label=f"Selections: {key}")
            
        ax2.tick_params(axis='y', labelcolor='black')

        # Combine legends from both independent axes safely
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9)

        plt.title(f"Comparative Solver Path Selection Profile\n"
                  f"Side-by-Side Histograms ({self.NUM_TRIALS} Trials, Single Fixed Context)", 
                  fontsize=13, fontweight='bold')
        
        fig.tight_layout()
        plt.show()