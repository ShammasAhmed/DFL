"""
Plotting utilities for the shortest-path DFL experiments.
"""
import numpy as np
import matplotlib.pyplot as plt


class RegretBoxPlot:
    """
    Grouped boxplot for comparing several methods across a set of groups.

    Typical use is one group per DGP degree, with one box per solver inside each
    group, drawn from the per-trial regret distribution.

    Inputs:
        groups (list): The group labels along the x-axis (e.g. polynomial degrees)
        series (list): Ordered list of (key, label, color) tuples, one per method.
            `key` indexes into the data dict; `label` is the legend text; `color`
            is any matplotlib color.
        xlabel, ylabel, title (str): Axis labels and title
        box_width (float): Center-to-center spacing between boxes within a group
        group_spacing (float): Center-to-center spacing between groups. Defaults to
            a value that leaves a gap between adjacent groups.
        alpha (float): Box face transparency
        figsize (tuple): Figure size
        showfliers (bool): Whether to draw outlier points
    """

    def __init__(self, groups, series, xlabel="", ylabel="", title="",
                 box_width=0.5, group_spacing=None, alpha=0.7,
                 figsize=(11, 6), showfliers=False):
        self.groups = list(groups)
        self.series = list(series)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.box_width = box_width
        self.alpha = alpha
        self.figsize = figsize
        self.showfliers = showfliers

        num_series = len(self.series)
        if group_spacing is None:
            # Leave roughly one box-width of gap between adjacent groups.
            group_spacing = (num_series + 1) * box_width
        self.group_spacing = group_spacing

        self.fig = None
        self.ax = None

    def plot(self, data, ax=None):
        """
        Build the grouped boxplot.

        Inputs:
            data (dict): data[group][series_key] -> list of values (one per trial)
            ax: Optional existing axes to draw into, for panelling several of these
                into one figure. When omitted a new figure is created.

        Returns:
            (fig, ax): The matplotlib figure and axes
        """
        num_series = len(self.series)
        centers = np.arange(len(self.groups)) * self.group_spacing
        # Offsets place the series symmetrically around each group center.
        offsets = (np.arange(num_series) - (num_series - 1) / 2) * self.box_width
        width = self.box_width * 0.85

        owns_figure = ax is None
        if owns_figure:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        handles = []
        for (key, _label, color), offset in zip(self.series, offsets):
            series_data = [data[g][key] for g in self.groups]
            bp = ax.boxplot(series_data, positions=centers + offset, widths=width,
                            patch_artist=True, showfliers=self.showfliers)
            for box in bp["boxes"]:
                box.set_facecolor(color)
                box.set_alpha(self.alpha)
            for median in bp["medians"]:
                median.set_color("black")
            handles.append(bp["boxes"][0])

        ax.set_xticks(centers)
        ax.set_xticklabels(self.groups)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(self.title)
        ax.legend(handles, [label for _, label, _ in self.series], loc="upper left")
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        if owns_figure:
            fig.tight_layout()
        self.fig, self.ax = fig, ax
        return fig, ax

    def save(self, path, dpi=150):
        """Save the current figure to `path`. Call after plot()."""
        if self.fig is None:
            raise RuntimeError("Call plot() before save().")
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight")

    def show(self):
        """Display the figure."""
        plt.show()


class PathHistogramPlot:
    """
    How often each solver picked a path of each rank, drawn against the curve of what
    those paths actually cost.

    The x-axis is every path, sorted by true expected cost <f*, w> ascending, so rank 0
    is the true optimal path. The blue line (left axis) is that cost curve; the bars
    (right axis) are selection counts. Reading the two together is the point: bars far
    to the right are bad picks, and *how* bad is the height of the blue line above them.
    A solver that misses the optimum onto a nearly-as-cheap path is doing something
    quite different from one that misses onto an expensive one, and a histogram alone
    cannot tell you which happened.

    Used by both the local run (experiments.HistogramExperiment) and the Slurm
    aggregation (histogram_aggregate.py), so the two draw the same figure.

    Inputs:
        sorted_costs (np.ndarray): True expected cost of every path, ascending
        num_trials (int): Trials the counts are drawn from, per solver
        series (list): Ordered (key, label, color) per solver, as in sweep.SERIES
        subtitle (str): Appended to every title, e.g. the training-set size
    """

    def __init__(self, sorted_costs, num_trials, series, subtitle=""):
        self.sorted_costs = np.asarray(sorted_costs)
        self.num_trials = num_trials
        self.series = list(series)
        self.subtitle = subtitle
        self.num_paths = len(self.sorted_costs)
        self.x = np.arange(self.num_paths)

    def relative_regret(self, rank_counts):
        """
        Mean regret of the chosen paths, as a percentage of the true optimal cost.

        Inputs:
            rank_counts (np.ndarray): Times a path of each rank was chosen

        Returns:
            pct (float): 100 * (mean chosen cost - optimal cost) / optimal cost
        """
        counts = np.asarray(rank_counts)
        trials = counts.sum()
        z_star = self.sorted_costs[0]
        if trials == 0 or z_star <= 0:
            return 0.0
        mean_cost = float(counts @ self.sorted_costs) / trials
        return 100 * (mean_cost - z_star) / z_star

    def _cost_curve(self, ax):
        """The blue true-cost curve, shared by both figures."""
        color = "tab:blue"
        ax.set_xlabel("Paths (sorted by true expected cost, ascending)", fontsize=11)
        ax.set_ylabel(r"True expected path cost ($f^{*T} w$)", color=color, fontsize=11)
        ax.plot(self.x, self.sorted_costs, color=color, linewidth=2.5,
                label="Path cost curve")
        ax.scatter(self.x, self.sorted_costs, color=color, marker="x", s=40, zorder=3)
        ax.axvline(x=0, color="crimson", linestyle="--", alpha=0.8,
                   label="True optimal path")
        ax.tick_params(axis="y", labelcolor=color)
        ax.grid(True, alpha=0.3, linestyle=":")

    def plot_solver(self, key, rank_counts):
        """One solver's selection histogram over the cost curve."""
        label, color = next((lab, col) for k, lab, col in self.series if k == key)

        fig, ax1 = plt.subplots(figsize=(11, 5))
        self._cost_curve(ax1)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Selection count (across training sets)", color=color,
                       fontsize=11)
        ax2.bar(self.x, rank_counts, color=color, alpha=0.6, width=0.8,
                label="Model selections")
        ax2.tick_params(axis="y", labelcolor=color)

        ax1.set_title(
            f"Path selection profile: {label}\n"
            f"Mean relative regret: {self.relative_regret(rank_counts):.2f}%  |  "
            f"{self.num_trials} trials, single fixed context{self.subtitle}",
            fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig

    def plot_comparison(self, rank_counts_by_key):
        """Every solver's histogram side by side, over the one cost curve."""
        fig, ax1 = plt.subplots(figsize=(12, 6))
        self._cost_curve(ax1)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Selection count (all solvers)", color="black", fontsize=11)

        drawn = [(key, label, color) for key, label, color in self.series
                 if key in rank_counts_by_key]
        # Bars sit side by side rather than overlapping, so no solver hides another.
        group_width = 0.8
        bar_width = group_width / max(len(drawn), 1)
        for idx, (key, label, color) in enumerate(drawn):
            offset = (idx - (len(drawn) - 1) / 2) * bar_width
            regret = self.relative_regret(rank_counts_by_key[key])
            ax2.bar(self.x + offset, rank_counts_by_key[key], color=color, alpha=1.0,
                    width=bar_width, label=f"{label} ({regret:.2f}% regret)")
        ax2.tick_params(axis="y", labelcolor="black")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
                   framealpha=0.9)

        ax1.set_title(
            f"Comparative path selection profile\n"
            f"{self.num_trials} trials, single fixed context{self.subtitle}",
            fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig
