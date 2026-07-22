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


def plot_regret_boxplots(by_size, sizes, panels, groups, series, outdir,
                         xlabel="", suptitle=None, filename="regret_boxplots_n{n}.png",
                         show=False, dpi=150):
    """
    The sweep's regret boxplots: one figure per training-set size, one panel per metric.

    Every panel of every figure lands on the same y-axis, so a box's height means the
    same regret wherever it appears -- across metrics, and across the figures. The
    n=100 vs n=1000 comparison is the point of drawing these together, and it only
    holds if the two figures cannot autoscale apart; regret_Y and regret_Y_lowvar pool
    against the same denominator, so the shared scale is meaningful rather than
    coincidental. The limits are the union of what each figure would have autoscaled to
    on its own, which is why the figures are all built before any of them is saved.

    The cost of one scale is that a small-spread metric gets squashed by a large-spread
    one; regret_Y runs several times larger than regret_fstar, so read the small-scale
    panels for their position relative to each other rather than for their internals.

    A dotted line marks y = 0, separating positive from negative regret --
    regret_Y_lowvar routinely goes negative, since z*(Y) chases the noise and is
    beatable under f*.

    Both context_aggregate.py (from the per-trial JSONs) and plot_regret_from_csv
    (from a trials CSV) draw through here, so the two produce the same figures.

    Inputs:
        by_size (dict): by_size[num_train][metric][group][series_key] -> trial values
        sizes (list): Training-set sizes, one figure each
        panels (list): Ordered (metric_key, axis_label) tuples, one panel each
        groups (list): Group labels along the x-axis (the DGP degrees)
        series (list): Ordered (key, label, color) per solver, as in sweep.SERIES
        outdir (Path): Where the PNGs are written
        xlabel (str): X-axis label, shared by every panel
        suptitle (callable): num_train -> figure title. Omit for no title.
        filename (str): Output name, formatted with the size as `n`
        show (bool): Display the figures instead of closing them
        dpi (int): Resolution of the saved PNGs

    Returns:
        paths (list): The files written, in `sizes` order
    """
    figures = []
    for num_train in sizes:
        fig, axes = plt.subplots(1, len(panels), figsize=(9 * len(panels), 6.5),
                                 sharey=True, squeeze=False)
        for ax, (metric, label) in zip(axes[0], panels):
            plotter = RegretBoxPlot(
                groups=list(groups),
                series=series,
                xlabel=xlabel,
                ylabel=label,
                title=label,
            )
            plotter.plot(by_size[num_train][metric], ax=ax)
            ax.axhline(0.0, color="black", linestyle=":", linewidth=1.2, zorder=0)
            # sharey blanks the later panels' tick labels; keep the ylabel, it differs.
            ax.set_ylabel(label)
        if suptitle is not None:
            fig.suptitle(suptitle(num_train), fontsize=13, fontweight="bold")
        figures.append((num_train, fig, axes[0]))

    # Widen every figure to the union of what each autoscaled to, which is what ties
    # the sizes to one scale. sharey has already done this within a figure.
    limits = [ax.get_ylim() for _, _, axs in figures for ax in axs]
    ylim = (min(low for low, _ in limits), max(high for _, high in limits))

    paths = []
    for num_train, fig, axs in figures:
        for ax in axs:
            ax.set_ylim(ylim)
        fig.tight_layout()  # after set_ylim: the tick labels it lays out around change
        path = outdir / filename.format(n=num_train)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"wrote {path}")
        paths.append(path)

    if show:
        plt.show()
    else:
        for _, fig, _ in figures:
            plt.close(fig)
    return paths


# Raw regret is the numerator of each normalized metric -- the method's cost minus its
# benchmark, before the division by the optimal cost (see context_aggregate.SUM_COLUMNS).
# Mapped metric -> (method sum column, benchmark sum column).
RAW_REGRET_TERMS = {
    "regret_fstar":    ("sum_fstar_zmethod", "sum_fstar_zfstar"),
    "regret_Y_lowvar": ("sum_fstar_zmethod", "sum_Y_zY"),
    "regret_Y":        ("sum_Y_zmethod",     "sum_Y_zY"),
}


def plot_raw_regret_boxplots(by_sums, sizes, panels, groups, series, outdir,
                             xlabel="", suptitle=None,
                             filename="raw_regret_boxplots_n{n}.png",
                             show=False, dpi=150):
    """
    plot_regret_boxplots, but each box is raw regret in cost units -- the numerator
    alone, averaged per context -- instead of a percentage of the optimal cost.

    Un-normalizing is the whole difference: for each trial it takes the method's cost
    sum minus its benchmark (RAW_REGRET_TERMS) and divides by that trial's context
    count, then hands the result to plot_regret_boxplots to draw. Every panel of both
    figures still shares one y-axis, and here that is exact rather than a compromise,
    since all three raw regrets are in the same cost units.

    Inputs:
        by_sums (dict): by_sums[num_train][deg][series_key] -> list of per-trial dicts,
            each holding the sum columns named in RAW_REGRET_TERMS plus "num_contexts".
        sizes, panels, groups, series, outdir, xlabel, suptitle, filename, show, dpi:
            as in plot_regret_boxplots. `panels` labels should read in cost units, not %.
    """
    by_size = {
        n: {metric: {g: {} for g in groups} for metric, _ in panels}
        for n in sizes
    }
    for n in sizes:
        for metric, _label in panels:
            method_col, bench_col = RAW_REGRET_TERMS[metric]
            for g in groups:
                for key, _lbl, _color in series:
                    by_size[n][metric][g][key] = [
                        (t[method_col] - t[bench_col]) / t["num_contexts"]
                        for t in by_sums[n][g].get(key, [])]
    return plot_regret_boxplots(by_size, sizes, panels, groups, series, outdir,
                                xlabel=xlabel, suptitle=suptitle, filename=filename,
                                show=show, dpi=dpi)


class PathHistogramPlot:
    """
    How often each solver picked a path of each rank, drawn against the curve of what
    picking that path costs you.

    The x-axis is every path, sorted by true expected cost <f*, w> ascending, so rank 0
    is the true optimal path. Both axes are relative, which is what makes the two
    readable against each other:

      left  (blue line): how much more a path costs than the optimal one, as a
                         percentage. Rank 0 sits at 0% by construction, so the curve
                         reads directly as the regret you incur by landing on a given
                         rank -- no mental subtraction of a baseline.
      right (bars):      what percentage of trials landed on that rank. The bars sum
                         to 100% per solver, so a solver's histogram is a density over
                         ranks and two solvers are comparable even from different
                         numbers of trials.

    Reading the two together is the point, and it is why neither axis is a raw count.
    A solver's mean regret is literally the bars integrated against the curve -- the
    density weighted by what each rank costs -- so a tall bar out where the blue line
    is high is exactly what a bad solver looks like. A solver that misses the optimum
    onto a nearly-as-cheap path is doing something quite different from one that misses
    onto an expensive one, and a histogram alone cannot tell you which happened.

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

        # Cost of every path as a % increase over the optimal one. This is the left
        # axis, and it is also the per-rank regret the densities get weighted by.
        z_star = self.sorted_costs[0]
        self.cost_pct = 100 * (self.sorted_costs - z_star) / z_star

    def density(self, rank_counts):
        """
        Selection counts as a percentage of the trials that produced them.

        Inputs:
            rank_counts (np.ndarray): Times a path of each rank was chosen

        Returns:
            density (np.ndarray): Percentages over ranks, summing to 100
        """
        counts = np.asarray(rank_counts, dtype=float)
        total = counts.sum()
        return np.zeros_like(counts) if total == 0 else 100 * counts / total

    def relative_regret(self, rank_counts):
        """
        Mean regret of the chosen paths, as a percentage of the true optimal cost.

        The density integrated against the cost curve -- i.e. the number you would get
        by reading the two axes of this plot against each other, which is the whole
        reason they are both relative.

        Inputs:
            rank_counts (np.ndarray): Times a path of each rank was chosen

        Returns:
            pct (float): 100 * (mean chosen cost - optimal cost) / optimal cost
        """
        return float(self.density(rank_counts) @ self.cost_pct) / 100

    def _cost_curve(self, ax):
        """The blue cost-increase curve, shared by both figures."""
        color = "tab:blue"
        ax.set_xlabel("Paths (sorted by true expected cost, ascending)", fontsize=11)
        ax.set_ylabel("Increase in true expected path cost over optimal (%)",
                      color=color, fontsize=11)
        ax.plot(self.x, self.cost_pct, color=color, linewidth=2.5,
                label="Path cost curve")
        ax.scatter(self.x, self.cost_pct, color=color, marker="x", s=40, zorder=3)
        ax.axvline(x=0, color="crimson", linestyle="--", alpha=0.8,
                   label="True optimal path (0%)")
        ax.tick_params(axis="y", labelcolor=color)
        ax.grid(True, alpha=0.3, linestyle=":")

    def plot_solver(self, key, rank_counts):
        """One solver's selection density over the cost curve."""
        label, color = next((lab, col) for k, lab, col in self.series if k == key)

        fig, ax1 = plt.subplots(figsize=(11, 5))
        self._cost_curve(ax1)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Selection density (% of trials)", color=color, fontsize=11)
        ax2.bar(self.x, self.density(rank_counts), color=color, alpha=0.6, width=0.8,
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
        """Every solver's density side by side, over the one cost curve."""
        fig, ax1 = plt.subplots(figsize=(12, 6))
        self._cost_curve(ax1)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Selection density (% of trials)", color="black", fontsize=11)

        drawn = [(key, label, color) for key, label, color in self.series
                 if key in rank_counts_by_key]
        # Bars sit side by side rather than overlapping, so no solver hides another.
        group_width = 0.8
        bar_width = group_width / max(len(drawn), 1)
        for idx, (key, label, color) in enumerate(drawn):
            offset = (idx - (len(drawn) - 1) / 2) * bar_width
            counts = rank_counts_by_key[key]
            ax2.bar(self.x + offset, self.density(counts), color=color, alpha=1.0,
                    width=bar_width,
                    label=f"{label} ({self.relative_regret(counts):.2f}% regret)")
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


def plot_regret_from_csv(csv_path, tag="h05", noise_h=0.5, outdir=".",
                         show=False, dpi=150):
    """
    Redraw the sweep's regret boxplots straight from a trials CSV -- both the normalized
    regret (% of optimal) and the raw regret in cost units. Same figures
    context_aggregate.py draws from the per-trial JSONs (plot_regret_boxplots /
    plot_raw_regret_boxplots); this is the from-CSV path, with no optimization stack.

    Inputs:
        csv_path (str): The trials CSV written by the sweep
        tag (str): Filename tag keeping this noise level's figures distinct
        noise_h (float): Noise half-width, for the figure titles only
        outdir (Path|str): Where the PNGs are written
        show (bool): Display the figures instead of closing them
        dpi (int): Resolution of the saved PNGs

    Returns:
        paths (list): The files written
    """
    import csv
    from collections import defaultdict
    from pathlib import Path
    from sweep import DEGREES, SIZES, SERIES, METRICS, SHOW_METRICS

    solver_keys = [key for key, _, _ in SERIES]
    sum_cols = sorted({col for pair in RAW_REGRET_TERMS.values() for col in pair})

    # normalized: by_size[num_train][metric][deg][solver] -> list of trial values
    # raw sums:   by_sums[num_train][deg][solver]         -> list of per-trial {col: val}
    by_size = {n: {m: {deg: defaultdict(list) for deg in DEGREES} for m in SHOW_METRICS}
               for n in SIZES}
    by_sums = {n: {deg: defaultdict(list) for deg in DEGREES} for n in SIZES}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            n, deg, key = int(row["num_train"]), int(row["deg"]), row["solver"]
            if n in by_size and deg in DEGREES and key in solver_keys:  # skip stray rows
                for m in SHOW_METRICS:
                    by_size[n][m][deg][key].append(float(row[m]))
                by_sums[n][deg][key].append(
                    {c: float(row[c]) for c in (*sum_cols, "num_contexts")})

    outdir = Path(outdir)
    paths = plot_regret_boxplots(
        by_size, SIZES, [(m, METRICS[m].label) for m in SHOW_METRICS], list(DEGREES),
        SERIES, outdir,
        xlabel="Polynomial degree of DGP",
        filename=f"regret_boxplots_{tag}_n{{n}}.png",
        suptitle=lambda n: ("Pooled context regret over 50 trials per degree "
                            f"(5x5 grid shortest path, noise h={noise_h}, "
                            f"training set size = {n})"),
        show=show, dpi=dpi,
    )
    paths += plot_raw_regret_boxplots(
        by_sums, SIZES,
        [(m, METRICS[m].label.replace("(%)", "(mean cost/context)")) for m in SHOW_METRICS],
        list(DEGREES), SERIES, outdir,
        xlabel="Polynomial degree of DGP",
        filename=f"raw_regret_boxplots_{tag}_n{{n}}.png",
        suptitle=lambda n: ("Raw context regret over 50 trials per degree "
                            f"(5x5 grid shortest path, noise h={noise_h}, "
                            f"training set size = {n})"),
        show=show, dpi=dpi,
    )
    return paths


def plot_noise_discount(csv_path, tag="h05", outdir=".",
                        filename="noise_discount_{tag}.png", suptitle=None,
                        show=False, dpi=150):
    """
    Line plot of the noise discount: how much cheaper the best decision that knows the
    realized cost Y is than the best decision that knows only E[Y|X]=f*.

    Per context and trial this is (Regret vs Y, low-variance) - (Regret vs f*), which
    reduces to (sum_fstar_zfstar - sum_Y_zY) / num_contexts: the solver's own decision
    cancels, and so does the trained model -- both z*(Y) and z*(f*) are model-free, so
    the discount is a property of the DGP alone. It is therefore pooled across every
    training-set size into one line over DGP degree. Unnormalized (raw cost per context,
    averaged over contexts and trials, never divided by an optimal cost) and
    non-negative -- knowing Y only helps.

    Inputs:
        csv_path (str): The trials CSV written by the sweep
        tag (str): Filename tag distinguishing this noise level's figure
        outdir (Path|str): Where the PNG is written
        filename (str): Output name, formatted with `tag`
        suptitle (str): Figure title. Omit for none.
        show (bool): Display instead of closing
        dpi (int): Resolution of the saved PNG

    Returns:
        path (Path): The file written
    """
    import csv
    from pathlib import Path
    from sweep import DEGREES, SERIES

    rep_solver = SERIES[0][0]   # discount is solver-independent; read one solver's rows

    # discounts[deg] -> per-trial discounts (raw cost per context), pooled over sizes
    discounts = {deg: [] for deg in DEGREES}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["solver"] != rep_solver:
                continue
            deg = int(row["deg"])
            if deg in discounts:
                discounts[deg].append(
                    (float(row["sum_fstar_zfstar"]) - float(row["sum_Y_zY"]))
                    / float(row["num_contexts"]))

    degrees = list(DEGREES)
    means = [np.mean(discounts[deg]) for deg in degrees]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(degrees, means, marker="o", color="tab:purple")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0, zorder=0)
    ax.set_xlabel("Polynomial degree of DGP")
    ax.set_ylabel("Noise discount (mean cost/context)")
    ax.set_xticks(degrees)
    ax.grid(True, linestyle=":", alpha=0.5)
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = Path(outdir) / filename.format(tag=tag)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"wrote {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path
