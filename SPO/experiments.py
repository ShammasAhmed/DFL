"""
Experiment orchestration for the shortest-path DFL comparison.

None of these classes generate their own data: each takes a generator from
datagen.py, a pure `gen(n, seed) -> Sample` that hands back covariates, realized
costs, and (when the DGP can supply it) the conditional mean f* = E[Y | X]. The
DGP is therefore swappable without touching an experiment, and whether f* exists
is a property of the generator rather than something an experiment assumes.

RegretExperiment sweeps a set of generators (degrees, by default), runs independent
trials at each, and collects the per-trial test regret (%).

ContextExperiment trains once and faces a batch of test contexts, reporting the
metrics in sweep.METRICS -- all of them when the generator supplies f*, and only
the ones that do not need it otherwise.

HistogramExperiment varies the training set for a single fixed context and records
which path each solver picks. It requires f*.
"""
import numpy as np
import torch
from itertools import combinations
import matplotlib.pyplot as plt

import pyepo
from pyepo.data.dataset import optDataset
from torch.utils.data import DataLoader

from datagen import split
from sweep import METRICS


def metrics_for(has_fstar):
    """
    The metrics a ContextExperiment can compute, given whether it has f*.

    Inputs:
        has_fstar (bool): Whether the generator supplies the conditional mean

    Returns:
        metrics (tuple): Computable keys of sweep.METRICS, in registry order
    """
    return tuple(key for key, m in METRICS.items()
                 if has_fstar or not m.needs_fstar)


def build_path_matrix(optmodel):
    """
    Enumerate every monotone (right/down) source-to-sink path on the grid as a
    (num_paths x num_edges) incidence matrix.

    Columns are indexed straight from optmodel.arcs -- the cost-vector edge order --
    so a row of this matrix is directly comparable to the incidence vector a Gurobi
    solve returns, regardless of PyEPO's internal edge layout. Nodes are row-major
    (node = row*width + col); a right move is arc (node, node+1) and a down move is
    arc (node, node+width).

    Inputs:
        optmodel: The shortest-path optimization model, supplying both the grid and
            the arc ordering

    Returns:
        path_matrix (np.ndarray): (num_paths x num_edges) incidence matrix
    """
    height, width = optmodel.grid
    arc_index = {tuple(arc): j for j, arc in enumerate(optmodel.arcs)}
    num_edges = len(optmodel.arcs)

    down_moves = height - 1
    total_moves = (width - 1) + down_moves

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
    return path_matrix


def predict(solver, x_row):
    """Predicted cost vector for one context's covariates."""
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(x_row).reshape(1, -1), dtype=torch.float32)
        Y_hat = solver.model(x)
    return Y_hat.detach().cpu().numpy().flatten()


class RegretExperiment:
    """
    Runs the sweep (one group per generator) x trials and returns per-trial regrets.

    The regret here is pyepo.metric.regret, i.e. the SPO regret against realized
    costs Y -- the same estimand as ContextExperiment's `regret_Y`. It never looks
    at f*, so the generators handed to it can safely be built with with_fstar=False.

    Inputs:
        optmodel: The shortest-path optimization model, passed in by the caller.
            The single source of truth for the problem geometry.
        solvers (list): Ordered list of (key, solver_cls) pairs. Each solver_cls is
            constructed as solver_cls(optmodel, X_TRAIN, Y_TRAIN, X_VAL, Y_VAL,
            rng_seed=rng_seed) and must expose a `.model` scoreable by
            pyepo.metric.regret.
        groups (list): Ordered list of (label, gen) pairs -- one sweep group per
            generator, labelled by whatever the sweep varies (polynomial degree, as
            things stand). The label is the plot's x-axis group and, by default,
            part of the seed.
        num_train, num_val, num_test (int): Split sizes. num_val defaults to
            num_train // 4 when left as None.
        NUM_TRIALS (int): Independent trials per group
        rng_seed (int): Base seed; each (label, trial) gets a distinct seed
        seed_fn (callable): seed_fn(label, trial) -> int, overriding how a trial's
            seed is derived. Defaults to rng_seed + 1000 * label + trial, which
            requires integer labels.
        verbose (bool): Print median regrets per group as the sweep runs
    """

    def __init__(self, optmodel, solvers, groups,
                 num_train=100, num_val=None, num_test=1000, NUM_TRIALS=50,
                 rng_seed=42, seed_fn=None, verbose=True):
        self.optmodel = optmodel
        self.solvers = list(solvers)
        self.groups = list(groups)
        self.num_train = num_train
        self.num_val = num_train // 4 if num_val is None else num_val
        self.num_test = num_test
        self.NUM_TRIALS = NUM_TRIALS
        self.rng_seed = rng_seed
        self.seed_fn = seed_fn or (
            lambda label, trial: self.rng_seed + 1000 * label + trial)
        self.verbose = verbose

        self.results = None

    def run(self):
        """
        Execute the sweep.

        Returns:
            results (dict): results[label][solver_key] -> list of per-trial test
                regrets (%). Also stored on self.results.
        """
        results = {label: {key: [] for key, _ in self.solvers}
                   for label, _ in self.groups}
        n = self.num_train + self.num_val + self.num_test

        for label, gen in self.groups:
            for trial in range(self.NUM_TRIALS):
                # Distinct DGP per (label, trial); one draw, so the ground truth B is
                # shared across the train/val/test split within a trial.
                seed = self.seed_fn(label, trial)
                train, val, test = split(gen(n, seed), self.num_train, self.num_val)

                test_set = optDataset(self.optmodel, test.X, test.Y)
                test_loader = DataLoader(test_set, batch_size=len(test_set),
                                         shuffle=False)

                for key, solver_cls in self.solvers:
                    solver = solver_cls(self.optmodel, train.X, train.Y,
                                        val.X, val.Y, rng_seed=seed)
                    regret = 100 * pyepo.metric.regret(
                        solver.model, self.optmodel, test_loader)
                    results[label][key].append(regret)

            if self.verbose:
                self._print_medians(label, results)

        self.results = results
        return results

    def _print_medians(self, label, results):
        parts = [f"{key} median {np.median(results[label][key]):7.4f}%"
                 for key, _ in self.solvers]
        print(f"GROUP {label:>2}: " + "  |  ".join(parts) +
              f"  ({self.NUM_TRIALS} trials)")


class ContextExperiment:
    """
    Individual experiment: train each solver on shared train/val data, then have the
    models face one or more test contexts and report, per solver, the pooled metrics.

    For a test context X with realized costs Y and (where available) the noiseless
    conditional mean f* = E[Y | X], let z*(c) = argmin_w c^T w be the optimal decision
    under cost c, and w_hat = z*(f(X)) the decision a solver makes from its predicted
    costs. The metrics are the entries of sweep.METRICS:

        loss_Y          : <Y, w_hat>                     realized cost of the path
        regret_Y        : <Y, w_hat>  - <Y, z*(Y)>       SPO regret
        regret_Y_lowvar : <f*, w_hat> - <f*, z*(Y)>      the same decision pair as
                                                         regret_Y, but scored under f*
                                                         rather than the noisy Y, so a
                                                         lower-variance estimate of it
        regret_fstar    : <f*, w_hat> - <f*, z*(f*)>     gap to the best policy

    The last two need f*, so they exist only when the generator supplies it. Everything
    available is always computed and returned; `metrics` selects only what print_table
    shows, which is why changing the selection never requires recomputing a sweep.

    loss_Y is a plain mean over contexts. The regrets are pooled as
    100 * (sum of regrets) / (sum of the corresponding optimal cost), the denominator
    being sweep.METRICS[...].denom -- regret_Y and regret_Y_lowvar deliberately share
    one, so the two sit on a single scale and can be read side by side.

    With shared_models=True (default) the solvers are trained once and every context is
    evaluated against those same models. With shared_models=False the solvers are
    retrained on a fresh DGP draw for each context.

    Inputs:
        optmodel: The shortest-path optimization model (single source of geometry)
        solvers (list): Ordered list of (key, solver_cls) pairs, constructed as
            solver_cls(optmodel, X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, rng_seed=rng_seed)
        gen (callable): A datagen generator, gen(n, seed) -> Sample. Whether it
            supplies f* decides which metrics are available.
        num_contexts (int): Number of test contexts to pool over
        shared_models (bool): Reuse one trained set of models across all contexts
        num_train, num_val (int): Split sizes for the training data. num_val defaults
            to num_train // 4 when left as None.
        metrics (iterable): Which metrics print_table displays, as keys of
            sweep.METRICS. Defaults to every metric this generator makes available.
            Naming one that needs f* against a generator without it is an error
            rather than a silent drop.
        rng_seed (int): Seed for data generation and solver initialization
    """

    def __init__(self, optmodel, solvers, gen, num_contexts=1, shared_models=True,
                 num_train=100, num_val=None, metrics=None,
                 rng_seed=42):
        self.optmodel = optmodel
        self.solvers = list(solvers)
        self.gen = gen
        self.num_contexts = num_contexts
        self.shared_models = shared_models
        self.num_train = num_train
        self.num_val = num_train // 4 if num_val is None else num_val
        self.rng_seed = rng_seed

        # One cheap draw settles what this generator can offer, so an impossible
        # metric selection fails here rather than midway through a sweep.
        self.has_fstar = gen(1, rng_seed).fstar is not None
        self.available = metrics_for(self.has_fstar)
        self.metrics = self._resolve_metrics(metrics)

        self.path_matrix = build_path_matrix(optmodel)
        self.table = None

    def _resolve_metrics(self, metrics):
        """
        Validate the display selection against what this generator can support.

        Inputs:
            metrics (iterable | None): Requested metric keys, or None for every
                metric available

        Returns:
            metrics (tuple): The validated selection

        Raises:
            ValueError: An unknown key, or one needing an f* the generator lacks.
        """
        if metrics is None:
            return self.available
        metrics = tuple(metrics)
        for key in metrics:
            if key not in METRICS:
                raise ValueError(
                    f"unknown metric {key!r}; known metrics are "
                    f"{', '.join(METRICS)}")
            if key not in self.available:
                raise ValueError(
                    f"metric {key!r} needs f*, but this generator does not supply "
                    f"it (build it with with_fstar=True, or drop {key!r} from the "
                    f"selection)")
        return metrics

    def _generate(self, seed, num_test):
        """
        Draw train/val data plus num_test contexts.

        Inputs:
            seed (int): Seed handed to the generator
            num_test (int): Number of test contexts to carve off the draw

        Returns:
            (train, val, test) (tuple of Sample): One draw, split three ways, so the
                ground truth is shared across the splits within a trial.
        """
        n = self.num_train + self.num_val + num_test
        return split(self.gen(n, seed), self.num_train, self.num_val)

    def _train(self, train, val):
        return {key: cls(self.optmodel, train.X, train.Y, val.X, val.Y,
                         rng_seed=self.rng_seed)
                for key, cls in self.solvers}

    def _decision_argmin(self, cost_vec):
        """
        Optimal decision (path incidence vector) under the given cost vector.

        Minimizes over the enumerated paths rather than calling Gurobi.
        """
        cost_vec = np.asarray(cost_vec)
        return self.path_matrix[np.argmin(self.path_matrix @ cost_vec)]

    def _eval_context(self, trained, x_ctx, y_ctx, fstar):
        """
        Un-normalized per-solver numerators, and the denominators to pool them against,
        for a single context.

        The f*-dependent metrics are skipped entirely when fstar is None; every metric
        the generator does support is computed, since each is only a dot product against
        a decision already made here.

        Inputs:
            trained (dict): key -> trained solver
            x_ctx (np.ndarray): The context's covariates
            y_ctx (np.ndarray): The context's realized costs Y
            fstar (np.ndarray | None): The context's conditional mean E[Y | X]

        Returns:
            (denoms, per_solver): denoms maps a sweep.METRICS denom name to this
                context's contribution to it; per_solver maps key -> {metric: value},
                the values being the un-normalized numerators.
        """
        w_star_Y = self._decision_argmin(y_ctx)          # z*(Y)
        opt_Y = float(y_ctx @ w_star_Y)                  # <Y, z*(Y)>
        denoms = {"count": 1.0, "opt_Y": opt_Y, "opt_fstar": 0.0}

        if fstar is not None:
            w_star_fstar = self._decision_argmin(fstar)  # z*(f*)
            opt_fstar = float(fstar @ w_star_fstar)      # <f*, z*(f*)>
            fstar_at_star_Y = float(fstar @ w_star_Y)    # <f*, z*(Y)>
            denoms["opt_fstar"] = opt_fstar

        per_solver = {}
        for key, solver in trained.items():
            w_hat = self._decision_argmin(predict(solver, x_ctx))  # z*(f(X))
            loss_Y = float(y_ctx @ w_hat)                                # <Y, z*(f(X))>

            values = {"loss_Y": loss_Y,
                      "regret_Y": loss_Y - opt_Y}
            if fstar is not None:
                fstar_at_hat = float(fstar @ w_hat)      # <f*, z*(f(X))>
                values["regret_Y_lowvar"] = fstar_at_hat - fstar_at_star_Y
                values["regret_fstar"] = fstar_at_hat - opt_fstar
            per_solver[key] = values

        return denoms, per_solver

    def _context_iter(self):
        """Yield (trained_models, x, y, f*) for each context; f* is None without one."""
        def at(test, i):
            return (test.X[i], test.Y[i],
                    None if test.fstar is None else test.fstar[i])

        if self.shared_models:
            train, val, test = self._generate(self.rng_seed, self.num_contexts)
            trained = self._train(train, val)
            for i in range(self.num_contexts):
                yield (trained, *at(test, i))
        else:
            for i in range(self.num_contexts):
                train, val, test = self._generate(self.rng_seed + i, 1)
                trained = self._train(train, val)
                yield (trained, *at(test, 0))

    def run(self):
        """
        Train and evaluate over all contexts.

        Returns:
            table (dict): solver_key -> {metric: value} over every metric this
                generator makes available (self.available), not merely the ones
                selected for display. loss_Y is the mean realized cost; the regrets
                are percentages pooled over the contexts against their denominators.
                Also stored on self.table.
        """
        sums = {key: {metric: 0.0 for metric in self.available}
                for key, _ in self.solvers}
        totals = {"count": 0.0, "opt_Y": 0.0, "opt_fstar": 0.0}

        for trained, x_ctx, y_ctx, fstar in self._context_iter():
            denoms, per_solver = self._eval_context(trained, x_ctx, y_ctx, fstar)
            for name, value in denoms.items():
                totals[name] += value
            for key, values in per_solver.items():
                for metric, value in values.items():
                    sums[key][metric] += value

        def pool(key, metric):
            denom = METRICS[metric].denom
            if denom == "count":
                return sums[key][metric] / totals["count"]
            return 100 * sums[key][metric] / totals[denom]

        self.table = {
            key: {metric: pool(key, metric) for metric in self.available}
            for key, _ in self.solvers
        }
        return self.table

    def print_table(self):
        """
        Run the experiment (if needed) and print the pooled results table.

        Only the metrics in self.metrics are shown; self.table always holds every
        metric the generator supports.

        Returns:
            table (dict): The full results table, as returned by run()
        """
        table = self.table if self.table is not None else self.run()

        header = f"{'Model':<10}" + "".join(
            f"{METRICS[m].header:>18}" for m in self.metrics)
        print(header)
        print("-" * len(header))
        for key, _ in self.solvers:
            row = "".join(
                f"{table[key][m]:>18.4f}" if METRICS[m].denom == "count"
                else f"{table[key][m]:>17.4f}%"
                for m in self.metrics)
            print(f"{key:<10}" + row)

        mode = "shared" if self.shared_models else "per-context"
        print(f"(over {self.num_contexts} contexts, {mode} models; "
              f"decision loss is mean realized cost, regrets are pooled %)")
        if not self.has_fstar:
            print("(generator supplies no f*; f*-based metrics unavailable)")
        return table

class HistogramExperiment:
    """
    Evaluates how often a learner selects specific paths across varying training sets
    for a SINGLE, FIXED test context.

    Uses standard PyEPO solvers for training, but completely bypasses Gurobi during
    evaluation using a combinatorial path-edge incidence matrix.

    Requires a generator that supplies f*: the paths are ranked by their true expected
    cost <f*, w>, which is the curve the selection histograms are read against.

    Inputs:
        optmodel: The shortest-path optimization model, used to train the solvers
        solvers (list): Ordered list of (key, solver_cls) pairs, constructed as
            solver_cls(optmodel, X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, rng_seed=rng_seed)
        gen (callable): A datagen generator, gen(n, seed) -> Sample. Must supply f*.
        NUM_TRIALS (int): Independent training sets the fixed context is faced with
        num_train, num_val (int): Split sizes for the training data. num_val defaults
            to num_train // 4 when left as None.
        rng_seed (int): Seed for the fixed context; each trial's training draw is
            seeded off it
    """

    def __init__(self, optmodel, solvers, gen, NUM_TRIALS=50,
                 num_train=100, num_val=None,
                 rng_seed=42):
        self.optmodel = optmodel
        self.solvers = list(solvers)
        self.gen = gen
        self.NUM_TRIALS = NUM_TRIALS
        self.num_train = num_train
        self.num_val = num_train // 4 if num_val is None else num_val
        self.rng_seed = rng_seed

        self.path_matrix = build_path_matrix(optmodel)
        self.num_paths = len(self.path_matrix)
        self.fixed_x = None
        self.fstar_X = None
        self.results = None

    def _generate_fixed_context(self):
        """
        Draw the one context every trial is evaluated against, and cache it.

        Raises:
            ValueError: The generator supplies no f*, so the paths cannot be ranked
                by their true expected cost.
        """
        sample = self.gen(1, self.rng_seed)
        if sample.fstar is None:
            raise ValueError(
                "HistogramExperiment ranks paths by their true expected cost, so it "
                "needs f*; build the generator with with_fstar=True")
        self.fixed_x = sample.X[0]
        self.fstar_X = sample.fstar[0]

    def _generate_train_data(self, seed):
        """One trial's training draw, split into train and validation Samples."""
        n = self.num_train + self.num_val
        train, val, _ = split(self.gen(n, seed), self.num_train, self.num_val)
        return train, val

    def _get_shortest_path_index(self, cost_vec):
        """Bypasses Gurobi entirely using vectorized matrix multiplication."""
        path_costs = self.path_matrix @ cost_vec
        return np.argmin(path_costs)

    def run(self):
        """
        Face the fixed context with NUM_TRIALS independently trained solver sets.

        Returns:
            results (dict): The true path costs under f*, the ordering that sorts them
                ascending, per-solver selection counts, and the true optimal path's
                index. Also stored on self.results.
        """
        self._generate_fixed_context()

        true_path_costs = self.path_matrix @ self.fstar_X
        sorted_indices = np.argsort(true_path_costs)

        path_counts = {key: np.zeros(self.num_paths, dtype=int) for key, _ in self.solvers}

        for trial in range(self.NUM_TRIALS):
            trial_seed = self.rng_seed + 1000 + trial
            train, val = self._generate_train_data(trial_seed)

            for key, cls in self.solvers:
                solver = cls(self.optmodel, train.X, train.Y, val.X, val.Y,
                             rng_seed=trial_seed)
                chosen_idx = self._get_shortest_path_index(
                    predict(solver, self.fixed_x))
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
            raise ValueError("Experiment has not been run yet. Please execute run() first.")
            
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