"""The recovery estimators the package did NOT keep -- the comparison that justifies the ones it did.

`statistics.composition` keeps the pool-count trio `fraction_no_pool` / `fraction_one_pool` /
`fraction_two_pool`. This module holds the retired alternatives a driver uses to show why those won:
reference-based calibration (`calibrate_temperature`), the anchor routes (`anchor_from_moments`,
`anchors_from_pools`, `observed_anchors`, `stretch`), and a matched estimator (`efficient_fraction`).
It imports the soft-EM internals and the reference grid it shares with the package.
"""

import numpy as np

from cryodisc.statistics.composition import (
    _BISECTION_STEPS,
    _LOGIT_LIMIT,
    _reference_grid,
)

_GRID_NODES = 2001
"""Points the Gaussian in `anchor_from_moments` is discretized on. Trapezoid rather than
Gauss-Hermite, whose weights overflow at the node counts a smooth integrand over +-12 sigma wants."""

_GRID_SPAN = 12.0
"""Half-width of that grid, in standard deviations. The anchor is a TAIL quantity, so a narrow
window would truncate the very region that sets it."""


def _logsumexp(values: np.ndarray) -> float:
    """`log(sum(exp(values)))`, shifted by the maximum so it does not overflow.

    Args:
        values: Log-space quantities.

    Returns:
        The log of their sum.
    """
    peak = float(np.max(values))
    return peak + float(np.log(np.sum(np.exp(values - peak))))


def anchor_from_moments(mean: float, variance: float) -> float:
    """Where a PURE pool with these two moments lands on the fraction axis, derived not fitted.

    `_em_fraction` solves `logit(mean_i sigmoid(Lambda_i + u)) = u`; for large `u` the residual tends to
    `-log Z` with `Z = E[exp(-Lambda)]`, so `Z < 1` reads exactly one, `Z > 1` reads below one, and
    `Z = 1` is calibrated. For a Gaussian `Lambda ~ N(mu, s^2)`, `Z = exp(-mu + s^2/2)`, hence

        log Z = (s^2 / 2) (1 - coherence),        coherence = 2 mu / s^2

    which carries the SIGN of `log Z` exactly. The Gaussian is deliberate: `E[exp(-Lambda)]` is
    dominated by a few extreme particles and not estimable from a finite pool, so substituting each
    pool's measured `log Z` is far worse than the Gaussian form.

    Args:
        mean: Mean of `Lambda` over the pool. `score_pool` records it under `coherence/mean`.
        variance: Variance of `Lambda` over the pool, under `coherence/variance`.

    Returns:
        `p_hat` for a pure pool, in `[0, 1]`. One means the readout saturates, which is what a pool
        whose coherence exceeds one does.

    Raises:
        ValueError: If the variance is not positive, which no pool of more than one particle has.
    """
    if not variance > 0:
        raise ValueError(f"variance must be positive, got {variance}")
    sigma = float(np.sqrt(variance))
    lam = float(mean) + sigma * np.linspace(-_GRID_SPAN, _GRID_SPAN, _GRID_NODES)
    log_weight = -0.5 * ((lam - float(mean)) / sigma) ** 2
    log_weight -= _logsumexp(log_weight)

    def residual(u: float) -> float:
        """`logit(mean sigmoid(Lambda + u)) - u`, in log space so neither tail underflows."""
        shifted = lam + u
        log_p = _logsumexp(log_weight - np.logaddexp(0.0, -shifted))
        log_q = _logsumexp(log_weight - np.logaddexp(0.0, shifted))
        return float(log_p - log_q - u)

    # the sign of log Z decides whether a finite root exists at all, and this IS the derivation
    if -float(mean) + variance / 2.0 <= 0.0:
        return 1.0
    low, high = -_LOGIT_LIMIT, _LOGIT_LIMIT
    if residual(low) < 0.0:
        return 0.0
    if residual(high) > 0.0:
        return 1.0
    # the residual is non-increasing in u, so bisection is valid -- the same argument _em_fraction uses
    for _ in range(_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if residual(middle) > 0.0 else (low, middle)
    return float(1.0 / (1.0 + np.exp(-0.5 * (low + high))))


def stretch(p_hat: np.ndarray, f_a: float, f_b: float) -> np.ndarray:
    """Undo the compression, rescaling so that `f_b` maps to zero and `f_a` to one.

    A mixture reading `p_hat` came from `(p_hat - f_b)/(f_a - f_b)`, and `anchors_from_pools` gets both
    anchors from the pools' own two moments, so this costs no fitted parameter. The argument order is
    `(f_a, f_b)`, NOT the `(f_b, f_a)` the formula reads in: swapping them silently runs the axis
    backwards while every mixture still lands on a plausible line.

    Args:
        p_hat: Recovered fractions, uncalibrated.
        f_a: Where a pure state-A pool reads.
        f_b: Where a pure state-B pool reads.

    Returns:
        The calibrated fractions.
    """
    return (np.asarray(p_hat, float) - f_b) / (f_a - f_b)


def anchors_from_pools(pool_a: np.ndarray, pool_b: np.ndarray) -> tuple[float, float]:
    """Where the two pure pools land, from their own two moments and nothing else.

    The B side is MIRRORED -- `anchor_from_moments` is written for a pool whose mean favors state A,
    and `anchor(-mu) = 1 - anchor(mu)` is what carries it to the other pool. Getting that backwards
    puts both anchors on the same side of a half and the stretch inverts, so the mirror lives here
    once rather than in each driver that needs a pair.

    Args:
        pool_a: Per-particle log-ratios, state-A pool.
        pool_b: Per-particle log-ratios, state-B pool.

    Returns:
        `(f_a, f_b)` predicted with no fitted parameter.
    """
    f_a = float(anchor_from_moments(float(pool_a.mean()), float(pool_a.var())))
    f_b = 1.0 - float(anchor_from_moments(float(-pool_b.mean()), float(pool_b.var())))
    return f_a, f_b


def observed_anchors(p_true: np.ndarray, p_hat: np.ndarray) -> tuple[float, float]:
    """Where the two pure pools land, read off a sweep's own two pure-end rows.

    The OBSERVED calibration route: no fitted parameter and no use of the known fractions beyond
    knowing which two rows are pure. It sits between `anchors_from_pools`, which predicts the anchors
    from the pools' moments, and `fit_purity`, which regresses on every known fraction.

    Args:
        p_true: Known state-A fraction of each mixture.
        p_hat: What the fraction estimator recovered from each.

    Returns:
        `(f_a, f_b)`, the readings at `p_true = 1` and `p_true = 0`.

    Raises:
        ValueError: If the sweep does not reach both pure ends. Reading the extreme rows of a grid
            without them returns anchors taken off MIXTURES, and the stretch between those is a wrong
            calibration that raises nothing on its own.
    """
    known = np.asarray(p_true, float)
    recovered = np.asarray(p_hat, float)
    if known.min() != 0.0 or known.max() != 1.0:
        raise ValueError(f"the observed route needs both pure pools in the sweep, got {np.unique(known)}")
    return float(recovered[known.argmax()]), float(recovered[known.argmin()])


def calibrate_temperature(pool_a: np.ndarray, pool_b: np.ndarray) -> tuple[float, float]:
    """The affine map `s Lambda + c` that makes an overdispersed score self-normalize.

    An honest log-ratio has `mean gap = variance` on its two pure pools; a real score has the gap two to
    three times short. Scaling by `s = gap / variance` restores it, and the shift `c` recenters the
    midpoint to zero. The exact root of `log Z = 0` is deliberately not used -- see `anchor_from_moments`
    on why the exponential tilt is not estimable from a finite pool. Applying `s Lambda + c` then the
    bare fixed point is a pool-requiring calibration route, retired in favor of `fraction_one_pool`
    (one pool's density mirrored about the boundary) and `fraction_two_pool` (both pools' full densities).

    Args:
        pool_a: Per-particle `Lambda` on a pool that is state A.
        pool_b: Per-particle `Lambda` on a pool that is state B.

    Returns:
        `(scale, shift)`. The calibrated score of any pool is `scale * Lambda + shift`.

    Raises:
        ValueError: If the pooled variance is not positive, or if `gap = mean(A) - mean(B)` is not
            positive -- the pools are unseparated or were passed in the wrong order, either of which
            would collapse or sign-flip the calibrated score.
    """
    a = np.asarray(pool_a, float).ravel()
    b = np.asarray(pool_b, float).ravel()
    gap = float(a.mean() - b.mean())
    variance = float(0.5 * (a.var() + b.var()))
    if not variance > 0.0:
        raise ValueError(f"pooled variance must be positive, got {variance}")
    if not gap > 0.0:
        raise ValueError(f"mean(A) must exceed mean(B) to calibrate; got gap {gap} (pools swapped or unseparated)")
    scale = gap / variance
    shift = -scale * 0.5 * float(a.mean() + b.mean())
    return float(scale), float(shift)


def _kde(reference: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """A Gaussian kernel density of `reference` on `grid`, Silverman bandwidth, floored positive."""
    spread = reference.std()
    iqr = (np.percentile(reference, 75) - np.percentile(reference, 25)) / 1.349
    bandwidth = max(0.9 * min(spread, iqr) * reference.size ** (-0.2), 1e-3 * max(spread, 1.0))
    z = (grid[None, :] - reference[:, None]) / bandwidth
    density = np.exp(-0.5 * z * z).sum(0) / (reference.size * bandwidth * np.sqrt(2.0 * np.pi))
    return np.maximum(density, 1e-9)


def efficient_fraction(log_ratio: np.ndarray, pool_a: np.ndarray, pool_b: np.ndarray, *, nodes: int = 400) -> float:
    """Fraction in state A by the one-step semiparametric-efficient estimator.

    Every `p_hat = (E_mix[h] - E_B[h]) / (E_A[h] - E_B[h])` is exact for any per-particle `h`; the
    variance-minimizing choice is the efficient influence function
    `h* = (f_A - f_B) / (p f_A + (1 - p) f_B)`, evaluated at a `pacc` pilot (one Newton-like step reaches
    efficiency) and plugged into the same ratio. Attains the Cramer-Rao bound the moment estimator
    cannot, and edges the reference mixture MLE at a few hundred particles.

    Args:
        log_ratio: `Lambda` for every particle in the mixed pool.
        pool_a: `Lambda` on the pure state-A pool.
        pool_b: `Lambda` on the pure state-B pool.
        nodes: Grid points the densities are evaluated on.

    Returns:
        `p_hat` in `[0, 1]`, clipped because a plug-in ratio is not confined to the simplex.

    Raises:
        ValueError: If the two pools read the same mean under `h*`, when no fraction is identifiable.
    """
    lam = np.asarray(log_ratio, float).ravel()
    a = np.asarray(pool_a, float).ravel()
    b = np.asarray(pool_b, float).ravel()
    grid = _reference_grid(a, b, lam, nodes=nodes)
    density_a, density_b = _kde(a, grid), _kde(b, grid)

    # the pacc pilot: h = sigmoid(Lambda), the bounded score, gives a p with no tail sensitivity
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + np.tanh(0.5 * x))

    pilot_a, pilot_b = float(_sigmoid(a).mean()), float(_sigmoid(b).mean())
    if abs(pilot_a - pilot_b) < 1e-12:
        # the bounded pilot is uninformative (the pools share a sigmoid mean); h* still separates
        # them by their higher moments, so fall back to a neutral pilot rather than declaring failure
        pilot = 0.5
    else:
        pilot = float(np.clip((float(_sigmoid(lam).mean()) - pilot_b) / (pilot_a - pilot_b), 0.02, 0.98))

    h_star = (density_a - density_b) / (pilot * density_a + (1.0 - pilot) * density_b)
    read_mix = float(np.interp(lam, grid, h_star).mean())
    read_a = float(np.interp(a, grid, h_star).mean())
    read_b = float(np.interp(b, grid, h_star).mean())
    if abs(read_a - read_b) < 1e-12:
        raise ValueError("the two pools read the same mean under h*; no fraction is identifiable")
    return float(np.clip((read_mix - read_b) / (read_a - read_b), 0.0, 1.0))
