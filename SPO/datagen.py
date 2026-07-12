"""
Data generation for the shortest-path experiments.

An experiment never draws its own data. It is handed a generator: a pure function

    gen(n, seed) -> Sample(X, Y, fstar)

returning n fresh covariate/cost pairs at the given seed. `Sample.fstar` is the
noiseless conditional mean E[Y | X] when the DGP can supply it, and None when it
cannot -- the experiments key their available metrics off exactly that, so a
generator without f* silently restricts them to the metrics that do not need it.

Holding generation behind this one signature is what makes the DGP swappable
without touching an experiment: shortest_path_gen wraps PyEPO's genData today,
and a pure-numpy generator can take its place later with no change on the
experiment side.
"""
from typing import Callable, NamedTuple, Optional

import numpy as np
import pyepo


class Sample(NamedTuple):
    """
    One draw from a DGP.

    Attributes:
        X (np.ndarray): (n x P) covariates
        Y (np.ndarray): (n x D) realized costs
        fstar (np.ndarray | None): (n x D) noiseless conditional mean E[Y | X],
            or None when the generator cannot supply it
    """
    X: np.ndarray
    Y: np.ndarray
    fstar: Optional[np.ndarray] = None


def shortest_path_gen(grid: tuple, P: int, h: float, deg: int,
                      with_fstar: bool = True) -> Callable[[int, int], Sample]:
    """
    Build a generator for PyEPO's polynomial shortest-path DGP.

    Costs are c = ((B x / sqrt(P) + 3) ** deg + 1) / 3.5 ** deg scaled by
    multiplicative uniform noise on [1 - h, 1 + h], with B ~ Bernoulli(0.5) and
    x ~ N(0, I). E[Y | X] is that same expression with the noise factor dropped,
    which is what genData returns at noise_width 0.

    f* is therefore obtained by calling genData a second time at the SAME seed with
    noise_width 0. genData draws B, then x, then the noise, in that order from a
    fresh RandomState, so the two calls agree on B and x and differ only in the
    noise factor: the f* returned really is the conditional mean of the Y returned
    alongside it, not an unrelated draw. Any replacement generator must preserve
    that draw order for f* to keep meaning what it says.

    Inputs:
        grid (tuple): Size of the grid network, e.g. (5, 5)
        P (int): Covariate dimension
        h (float): Multiplicative noise half-width
        deg (int): Polynomial degree of the feature-to-cost mapping
        with_fstar (bool): Also return E[Y | X]. Costs a second genData call, so
            pass False for experiments that never look at f*.

    Returns:
        gen (callable): gen(n, seed) -> Sample
    """
    def gen(n: int, seed: int) -> Sample:
        X, Y = pyepo.data.shortestpath.genData(
            n, P, grid, deg=deg, noise_width=h, seed=seed)
        fstar = None
        if with_fstar:
            _, fstar = pyepo.data.shortestpath.genData(
                n, P, grid, deg=deg, noise_width=0.0, seed=seed)
        return Sample(X, Y, fstar)

    return gen


def split(sample: Sample, num_train: int, num_val: int) -> tuple:
    """
    Split one draw into disjoint train / validation / test Samples.

    The test split takes whatever is left over, so a caller sizes it by asking the
    generator for num_train + num_val + num_test points in the first place.

    Inputs:
        sample (Sample): A draw from a generator
        num_train (int): Size of the training split
        num_val (int): Size of the validation split

    Returns:
        (train, val, test) (tuple of Sample): The three disjoint splits. fstar is
            sliced alongside X and Y, or stays None across all three.
    """
    def take(lo, hi):
        fstar = None if sample.fstar is None else sample.fstar[lo:hi]
        return Sample(sample.X[lo:hi], sample.Y[lo:hi], fstar)

    end_train = num_train
    end_val = num_train + num_val
    return (take(0, end_train),
            take(end_train, end_val),
            take(end_val, len(sample.X)))
