"""Recover a population fraction from two pure pools, and ask whether the predicted count was right.

Used by `panel/survey_fraction.py`, which runs the same draw, estimate and measure loop over a whole
survey; here rather than in the driver so any benchmark of one pair draws it the same way.

**The test is prospective.** The count comes from `d^2` alone, before any draw, and the draws are
then asked whether they meet it. Nothing here fits a count to an observed scatter.
"""

import numpy as np
from jaxtyping import Float

from cryodisc.statistics.composition import (
    fraction_no_pool,
    fraction_two_pool,
    information_from_pools,
)


def gap_from_pools(log_ratio_a: Float[np.ndarray, " a"], log_ratio_b: Float[np.ndarray, " b"]) -> float:
    """The signed mean gap between the two pure pools -- the direction check, not the discriminability.

    This is `mean(Lambda_A) - mean(Lambda_B)` (the Jeffreys divergence when Lambda is the exact LLR).
    It is NOT the reported currency -- that is the deflection `gap^2/within`, and the count comes from
    the pools' Eq. 6 information (`count_from_pools`). Kept only to reject a pair whose state A reads
    lower, since the count map assumes A reads higher.

    Args:
        log_ratio_a: `Lambda` on a pool that is state A.
        log_ratio_b: `Lambda` on a pool that is state B.

    Returns:
        `mean(Lambda_A) - mean(Lambda_B)`, positive when state A reads higher.
    """
    return float(np.asarray(log_ratio_a).mean() - np.asarray(log_ratio_b).mean())


def predicted_count(d2: float, *, fraction: float, sigma: float) -> float:
    """Particles for `sigma` from a single deflection scalar, via the Gaussian-model `I_p` map.

    The design-tool path, for when only a `d^2` scalar is in hand (no pools) -- it assumes the
    calibrated equal-variance Gaussian of §4. When the two pure pools ARE available, prefer
    `count_from_pools`, which uses the exact Eq. 6 information and is calibration-free. Pass the
    DEFLECTION `gap^2/within` here, never the raw gap.

    Args:
        d2: Per-particle deflection `gap^2/within`.
        fraction: Population fraction of state A.
        sigma: Target standard error on that fraction.

    Returns:
        The count.
    """
    import jax.numpy as jnp

    from cryodisc.statistics.power import n_for_error

    return float(n_for_error(jnp.asarray(d2), error_target=sigma, fraction=fraction))


def count_from_pools(
    log_ratio_a: Float[np.ndarray, " a"],
    log_ratio_b: Float[np.ndarray, " b"],
    *,
    fraction: float,
    sigma: float,
) -> tuple[float, float, float]:
    """Particles for `sigma` from the two pools' own Eq. 6 information -- no Gaussian map.

    Uses `information_from_pools` (derivations §9b(e)): exact for a calibrated bank, calibration-free
    on a real pool. A returned gate far from `fraction` warns the raw score is miscalibrated, so
    recover with the two-pool MLE (`measure_scatter(..., two_pool=True)`), not the 0-pool fixed point.

    Args:
        log_ratio_a: `Lambda` on the pure state-A pool.
        log_ratio_b: The same for state B.
        fraction: Population fraction of state A.
        sigma: Target standard error on that fraction.

    Returns:
        `(count, information, calibration_gate)`.
    """
    information, gate = information_from_pools(log_ratio_a, log_ratio_b, fraction=fraction)
    return 1.0 / (sigma**2 * information), information, gate


def fraction_estimate(log_ratio: Float[np.ndarray, " n"]) -> float:
    """Estimate the state-A fraction of one mixed pool from its per-particle `Lambda`, no reference.

    Delegates to `statistics.composition.fraction_no_pool`, the package's soft-EM fixed point (the root
    of `logit(mean(sigma(Lambda + u))) - u`), so it tests the tool and not a copy.

    Args:
        log_ratio: `Lambda` for every particle in the mixed pool.

    Returns:
        The recovered fraction in (0, 1).
    """
    return fraction_no_pool(log_ratio)


def measure_scatter(
    log_ratio_a: Float[np.ndarray, " a"],
    log_ratio_b: Float[np.ndarray, " b"],
    *,
    n: int,
    fraction: float,
    repeats: int,
    rng: np.random.Generator,
    two_pool: bool = False,
) -> tuple[float, float]:
    """Draw `n`-particle mixtures `repeats` times and measure how much the recovered fraction moves.

    **`sigma_p` is measured by REPETITION, not from a fit's own error bar**, those being different
    numbers and only one of them what was predicted. The repeats resample the two scored pure pools
    WITH REPLACEMENT rather than re-simulating, so they see the counting statistics and not anything
    varying between pools -- ice thickness, a per-session defocus error.

    Args:
        log_ratio_a: `Lambda` on the pure state-A pool.
        log_ratio_b: `Lambda` on the pure state-B pool.
        n: Particles per drawn mixture.
        fraction: True state-A fraction each mixture is drawn at.
        repeats: How many mixtures to draw. Sets how well `sigma_p` itself is resolved.
        rng: Source of the draws.
        two_pool: Recover with the reference mixture MLE against the two pure pools instead of the
            0-pool fixed point. The pools themselves are the references.

    Returns:
        `(sigma_p, bias)` -- the standard deviation of the recovered fractions across repeats, and
        their mean minus the truth.
    """
    lam_a = np.asarray(log_ratio_a, dtype=float)
    lam_b = np.asarray(log_ratio_b, dtype=float)
    estimates = np.empty(repeats)
    for repeat in range(repeats):
        # binomial, not round(p * n): a real pool's composition is itself a draw, and that
        # `p(1-p)/N` variance is a term being tested, which fixing the count would remove
        n_a = int(rng.binomial(n, fraction))
        draw = np.concatenate(
            [rng.choice(lam_a, size=n_a, replace=True), rng.choice(lam_b, size=n - n_a, replace=True)]
        )
        estimates[repeat] = fraction_two_pool(draw, lam_a, lam_b) if two_pool else fraction_no_pool(draw)
    return float(estimates.std(ddof=1)), float(estimates.mean() - fraction)
