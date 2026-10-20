"""Known-fraction benchmarks for the pool fraction estimators, and the purity line they fit.

Mixes two pure pools at known proportions (`sweep_mixtures`), regresses the recovered fraction on the
known one, and reads the two pure-pool anchors off the line (`fit_purity` -> `Purity`). Under the model
the regression is a straight line

    p_hat = f_B + p_true (f_A - f_B),   separation = f_A - f_B,   midpoint = (f_A + f_B) / 2

with `f_A = 1`, `f_B = 0` under a perfect forward model. A compression (`separation < 1`) is
symmetric template error; an offset (`midpoint != 1/2`) is directional, one bank favored.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from cryodisc.statistics.composition import (
    fraction_no_pool,
    fraction_one_pool,
    fraction_two_pool,
)
from reproduction.analysis.fits import line_fit


@dataclass(frozen=True)
class Purity:
    """What a known-fraction sweep says the two pure pools actually read.

    Args:
        f_a: Where a pure state-A pool sits on the axis. One under a perfect forward model.
        f_b: Where a pure state-B pool sits. Zero under a perfect forward model.
        f_a_err: Standard error of `f_a`, from the regression residuals.
        f_b_err: Standard error of `f_b`.
        separation_err: Standard error of `separation`, which is the fitted SLOPE, so this is
            `sqrt(cov[1, 1])` and not the two anchors' errors combined -- they are correlated.
        midpoint_err: Standard error of `midpoint`, likewise carrying the covariance term.
        n_points: Mixtures the line was fitted to.
        residual_rms: RMS departure of `p_hat` from the fitted line, in fraction units. The check
            that the model is a line at all: a subpopulation that is inert or contaminating changes
            the SHAPE, and shows up here rather than in either anchor.
    """

    f_a: float
    f_b: float
    f_a_err: float
    f_b_err: float
    separation_err: float
    midpoint_err: float
    n_points: int
    residual_rms: float

    @property
    def separation(self) -> float:
        """`f_a - f_b`, one under a perfect forward model.

        Returns:
            The span of the axis. Below one is a COMPRESSION, which template error of comparable
            size on the two states produces, and it enters both anchors with the same sign.
        """
        return self.f_a - self.f_b

    @property
    def midpoint(self) -> float:
        """`(f_a + f_b) / 2`, one half under a perfect forward model.

        Returns:
            The center of the axis. Away from one half is an OFFSET, which enters the two anchors
            with OPPOSITE signs and so cannot be produced by anything that merely rescales the
            score -- it is directional, and it says one bank is favored over the other.
        """
        return 0.5 * (self.f_a + self.f_b)


def fit_purity(
    p_true: Float[np.ndarray, " m"] | list[float],
    p_hat: Float[np.ndarray, " m"] | list[float],
) -> Purity:
    """Regress recovered fraction on known fraction, and read the two pure-pool anchors off the line.

    Args:
        p_true: Known state-A fraction of each mixture, exact by construction from the pool sizes.
        p_hat: What the fraction estimator recovered from each.

    Returns:
        The anchors and the line's quality.

    Raises:
        ValueError: If the two disagree in length, if fewer than three mixtures are given, or if
            every mixture sat at the same fraction -- a line through one abscissa is not determined,
            and fitting it anyway would report an anchor with no support.
    """
    known = np.asarray(p_true, float).ravel()
    recovered = np.asarray(p_hat, float).ravel()
    if known.shape != recovered.shape:
        raise ValueError(f"p_true and p_hat must agree, got {known.shape} and {recovered.shape}")
    # three, not two: two points fit any line exactly and leave no residual to read the shape from
    if known.size < 3:
        raise ValueError(f"a purity fit needs at least three mixtures, got {known.size}")
    if np.ptp(known) == 0.0:
        raise ValueError(f"every mixture sat at p_true = {known[0]}, which determines no line")

    # regressed on the known fraction itself, so the intercept is a pure state-B pool's reading
    (intercept, slope), covariance, residual_rms = line_fit(known, recovered)
    # f_a = intercept + slope, so its variance carries the covariance term and is not the sum
    f_a_variance = covariance[0, 0] + 2.0 * covariance[0, 1] + covariance[1, 1]
    # separation IS the slope and midpoint is `intercept + slope/2`, so both errors come off the
    # covariance the fit already computed rather than from the anchors, which are correlated
    midpoint_variance = covariance[0, 0] + covariance[0, 1] + 0.25 * covariance[1, 1]
    return Purity(
        f_a=float(intercept + slope),
        f_b=float(intercept),
        f_a_err=float(np.sqrt(max(f_a_variance, 0.0))),
        f_b_err=float(np.sqrt(max(covariance[0, 0], 0.0))),
        separation_err=float(np.sqrt(max(covariance[1, 1], 0.0))),
        midpoint_err=float(np.sqrt(max(midpoint_variance, 0.0))),
        n_points=int(known.size),
        residual_rms=residual_rms,
    )


def sweep_mixtures(
    log_ratio_a: Float[Array, " a"],
    log_ratio_b: Float[Array, " b"],
    fractions: Sequence[float],
    *,
    key: Array,
    estimator: Callable[[Array], float] = fraction_no_pool,
) -> tuple[list[float], list[float]]:
    """Recover the fraction from pools mixed at known proportions, at a constant pool size.

    The abscissa `fit_purity` regresses on. Pool size is held CONSTANT across the sweep so every point
    carries the same sampling error; growing it with the fraction would tilt the fit.

    Args:
        log_ratio_a: Lambda for the pool taken as state A.
        log_ratio_b: Lambda for the pool taken as state B.
        fractions: The `p_true` to mix at.
        key: PRNG key, split per mixture.
        estimator: Reads `p_hat` off a mixed pool's Lambda. Default is the 0-pool `fraction_no_pool`;
            a caller with two pure pools passes `lambda mix: fraction_two_pool(mix, pool_a, pool_b)`.

    Returns:
        `(p_true, p_hat)`, one entry per mixture. `p_true` is recomputed from the counts actually
        drawn rather than echoed from the request, so a rounding to whole particles is visible in
        the truth instead of quietly landing in the residual.
    """
    size = min(log_ratio_a.shape[0], log_ratio_b.shape[0])
    known, recovered = [], []
    for fraction, subkey in zip(fractions, jax.random.split(key, len(fractions)), strict=True):
        n_a = round(fraction * size)
        key_a, key_b = jax.random.split(subkey)
        drawn_a = jax.random.choice(key_a, log_ratio_a, (n_a,))
        drawn_b = jax.random.choice(key_b, log_ratio_b, (size - n_a,))
        known.append(n_a / size)
        recovered.append(estimator(jnp.concatenate([drawn_a, drawn_b])))
    return known, recovered


SWEEP_FRACTIONS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
"""The mixtures the calibration-route and coherence-slope sweeps draw.

Shared so drivers quoting purity lines stay on one abscissa; importing it is what opts a driver in.
"""

SWEEP_REPEATS = 200
"""Independent sweeps behind an error bar on a calibration route."""


def sweep_mixtures_repeated(
    log_ratio_a: Float[Array, " a"],
    log_ratio_b: Float[Array, " b"],
    fractions: Sequence[float],
    *,
    key: Array,
    repeats: int = SWEEP_REPEATS,
    estimator: Callable[[Array], float] = fraction_no_pool,
) -> tuple[list[float], list[float], list[float]]:
    """`sweep_mixtures` run many times, to put an error bar on each recovered fraction.

    Repeating the whole sweep under fresh keys is the bootstrap; `sweep_mixtures` already samples the
    pools with replacement. The spread is over WHICH PARTICLES land in the pool -- the error a repeat at
    the same pool size would see, NOT the error on the underlying discriminability.

    Args:
        log_ratio_a: Lambda for the pool taken as state A.
        log_ratio_b: Lambda for the pool taken as state B.
        fractions: The `p_true` to mix at.
        key: PRNG key, split per repeat.
        repeats: How many independent sweeps to draw.
        estimator: Reads `p_hat` off a mixed pool's Lambda, forwarded to `sweep_mixtures`.

    Returns:
        `(p_true, p_hat_mean, p_hat_sd)`, one entry per mixture.

    Raises:
        ValueError: If `repeats` is under two, a standard deviation over one draw being undefined.
    """
    if repeats < 2:
        raise ValueError(f"a standard deviation needs at least two sweeps, got {repeats}")
    known: list[float] = []
    draws: list[list[float]] = []
    for subkey in jax.random.split(key, repeats):
        p_true, p_hat = sweep_mixtures(log_ratio_a, log_ratio_b, fractions, key=subkey, estimator=estimator)
        known = p_true
        draws.append([float(value) for value in p_hat])
    stacked = np.asarray(draws, dtype=float)
    return known, [float(v) for v in stacked.mean(axis=0)], [float(v) for v in stacked.std(axis=0, ddof=1)]


ESTIMATORS = ("no_pool", "one_pool_a", "one_pool_b", "two_pool")
"""The fraction estimators `fraction_estimator` names, by how many pure pools each reads."""


def fraction_estimator(name: str, log_ratio_a: np.ndarray, log_ratio_b: np.ndarray) -> Callable[[Array], float]:
    """The callable `sweep_mixtures` reads each `p_hat` with, for one of `ESTIMATORS`.

    The pure pools supply one reflected density (1-pool) or both measured densities (2-pool).
    """
    if name == "no_pool":
        return fraction_no_pool
    if name == "one_pool_a":
        return lambda mix: fraction_one_pool(np.asarray(mix), log_ratio_a)
    if name == "one_pool_b":
        return lambda mix: fraction_one_pool(np.asarray(mix), log_ratio_b)
    return lambda mix: fraction_two_pool(np.asarray(mix), log_ratio_a, log_ratio_b)


def holdout_split(log_ratio_a: Array, log_ratio_b: Array, key: Array) -> tuple[Array, Array, Array, Array]:
    """Split each pool into a draw half (mixed into the sweep) and a reference half (the estimator's).

    Disjoint by construction, so no particle mixed into a sweep is also one of its references.
    """
    split = []
    for pool, subkey in zip((log_ratio_a, log_ratio_b), jax.random.split(key)):
        index = jax.random.permutation(subkey, pool.shape[0])
        half = pool.shape[0] // 2
        split.append((pool[index[:half]], pool[index[half:]]))
    (draw_a, ref_a), (draw_b, ref_b) = split
    return draw_a, draw_b, ref_a, ref_b
