"""
Cost spread across the 40 grid edges, drawn against DGP degree.

For each degree we draw 1000 trials (contexts) from one fixed ground truth B, average
each edge's cost over those trials (normalize = sum / 1000), and box the resulting 40
per-edge means. The box's height is the spread of mean cost across edges, so a wider box
means the degree makes edge costs more heterogeneous.

Two panels off one draw: the noiseless cost f* = E[Y|X] and the noisy cost Y at h=0.5.
Same seed, so both share X and B and differ only in the multiplicative noise; on a shared
y-axis the pair shows how little averaging 1000 mean-1 noise factors moves the edge means.

SAME_X (below) controls whether the degrees share their X draw: True isolates the effect
of degree on cost (every degree sees the identical 1000 contexts), False gives each its
own draw so the spread also reflects X sampling.

    python costvar.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datagen import numpy_shortest_path_gen
from sweep import RNG_SEED

GRID = (5, 5)
P = 5
h = 0.5
DEGREES = (1, 2, 4, 6, 8)
NUM_TRIALS = 1000
SEED = RNG_SEED          # one draw shared by both panels, fixed across degrees
SAME_X = True            # True: every degree sees the same X draw; False: one draw each


def per_edge_means(deg, seed):
    """The 40 per-edge mean costs at `deg`, noiseless and noisy, from one shared draw."""
    gen = numpy_shortest_path_gen(GRID, P, h, deg, dgp_seed=RNG_SEED, with_fstar=True)
    sample = gen(NUM_TRIALS, seed)
    # Average over the 1000 trials -> one mean cost per edge (the "normalize by 1000").
    return sample.fstar.mean(axis=0), sample.Y.mean(axis=0)


# Same seed across degrees -> identical X (drawn before deg is applied); a per-degree
# seed gives each its own X. B stays pinned by dgp_seed either way.
draws = [per_edge_means(deg, SEED if SAME_X else SEED + i)
         for i, deg in enumerate(DEGREES)]
noiseless = [f for f, _ in draws]
noisy = [y for _, y in draws]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, data, title in ((axes[0], noiseless, "Noiseless cost  $f^*=E[Y|X]$"),
                        (axes[1], noisy, f"Noisy cost  $Y$  (h={h})")):
    ax.boxplot(data, positions=DEGREES, widths=0.6, patch_artist=True,
               boxprops=dict(facecolor="tab:purple", alpha=0.6),
               medianprops=dict(color="black"))
    ax.set_xlabel("Polynomial degree of DGP")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
axes[0].set_ylabel("Mean edge cost across 1000 trials")
fig.suptitle("Cost spread across the 40 grid edges (5x5 shortest path, fixed ground truth)",
             fontsize=13, fontweight="bold")
fig.tight_layout()

fig.savefig("cost_boxplot.png", dpi=150, bbox_inches="tight")
print("wrote cost_boxplot.png")