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
