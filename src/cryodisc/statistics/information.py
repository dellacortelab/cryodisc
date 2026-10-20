"""Reading a per-particle log-ratio as an amount of information.

Every rung yields `Lambda = log p(Y|A) / p(Y|B)` per particle, read two ways:
    deflection            the canonical discriminability d^2 = gap^2 / within-variance
    fisher_from_deflection    I_p about the population fraction that d^2 implies, capped at perfect labels
The deflection is intrinsic to the pair and free of the population fraction; `fisher_from_deflection`
is the only place the fraction enters, converting d^2 to a particle count `sigma(p_hat) = 1/sqrt(N I_p)`.
`deflection_standard_error` gives the error bar for ordering a measured rung against a model rung.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


def information_ceiling(fraction: Float[Array, ""] | float = 0.5) -> Float[Array, ""] | float:
    """The most information one particle can carry about the fraction.

    Args:
        fraction: Population fraction of state A.

    Returns:
        `1 / [p (1 - p)]`, which only perfect per-particle labels reach.
    """
    return 1.0 / (fraction * (1.0 - fraction))


@eqx.filter_jit
def deflection(
    log_ratio_a: Float[Array, " na"],
    log_ratio_b: Float[Array, " nb"],
) -> Float[Array, ""]:
    """The canonical discriminability d^2: squared mean gap over the pooled within-class variance.

    Normalizing by `within` divides out the scale of an LLR that is not calibrated (gap != within),
    which a raw mean gap cannot; the deflection is therefore affine-invariant (= signal-detection
    d'^2) and free of the population fraction, a property of the pair alone. It is non-negative by
    construction. The pool and pose grid cover the population, so there is no subsample bias to
    correct: what is measured is the true population d^2 once the grid has converged.

    Args:
        log_ratio_a: Lambda on particles whose true state is A.
        log_ratio_b: Lambda on particles whose true state is B.

    Returns:
        The deflection, non-negative and invariant to any affine rescaling of Lambda.
    """
    n_a, n_b = log_ratio_a.shape[0], log_ratio_b.shape[0]
    if n_a < 2 or n_b < 2:
        raise ValueError(f"each class needs >=2 particles for a within-class variance; got {n_a} and {n_b}")
    within = (log_ratio_a.var(ddof=1) * n_a + log_ratio_b.var(ddof=1) * n_b) / (n_a + n_b)
    gap = log_ratio_a.mean() - log_ratio_b.mean()
    return gap**2 / jnp.maximum(within, 1e-12)


@eqx.filter_jit
def deflection_standard_error(
    log_ratio_a: Float[Array, " na"],
    log_ratio_b: Float[Array, " nb"],
) -> Float[Array, ""]:
    """Sampling error of `deflection` on the pool it was measured on.

    Keep the `filter_jit`, here and on `deflection`: it fuses the primitives into one kernel rather
    than recompiling on pools whose length is a random draw. The estimate is a ratio `gap^2 / within`,
    so its variance combines the gap's sampling variance with the pooled variance's own scatter, and
    stays nonzero at zero signal -- a rung at d^2 = 0 still fluctuates against its own noise.

    Args:
        log_ratio_a: Lambda on particles whose true state is A.
        log_ratio_b: Lambda on particles whose true state is B.

    Returns:
        The delta-method standard error of the deflection.
    """
    n_a, n_b = log_ratio_a.shape[0], log_ratio_b.shape[0]
    if n_a < 2 or n_b < 2:
        raise ValueError(f"each class needs >=2 particles for a within-class variance; got {n_a} and {n_b}")
    var_a, var_b = log_ratio_a.var(ddof=1), log_ratio_b.var(ddof=1)
    within = jnp.maximum((var_a * n_a + var_b * n_b) / (n_a + n_b), 1e-12)
    floor_scale = (var_a / n_a + var_b / n_b) / within  # Var(gap)/within, the zero-signal SE scale
    d2 = deflection(log_ratio_a, log_ratio_b)
    variance = 4.0 * d2 * floor_scale + 2.0 * floor_scale**2 + 2.0 * d2**2 / (n_a + n_b)
    return jnp.sqrt(variance)


_GAUSSIAN_POINTS = 4001
"""Quadrature points in `fisher_from_deflection`, whose integrand is two known Gaussians."""

_GAUSSIAN_SPAN = 8.0
"""Half-width of that grid in units of the common standard deviation, beyond the two means.

Eight sigma, so the tails the integral misses are below float32 resolution. Both are constants rather
than parameters: they are properties of the quadrature and no caller ever set one.
"""


@eqx.filter_jit
def fisher_from_deflection(
    deflection: Float[Array, ""] | float,
    fraction: float = 0.5,
) -> Float[Array, ""]:
    """The I_p a deflection implies under the equal-variance Gaussian model.

    Under that model `Lambda|A ~ N(+d^2/2, d^2)` and `Lambda|B ~ N(-d^2/2, d^2)`. This is what puts
    a deterministic rung, which has no pool to measure, on the same scale as a measured one, and is
    the sole converter of the fraction-free d^2 into a fraction-dependent particle count.

    Args:
        deflection: d^2.
        fraction: Population fraction of state A.

    Returns:
        I_p.
    """
    safe = jnp.maximum(jnp.asarray(deflection), 1e-12)
    sigma = jnp.sqrt(safe)
    reach = 0.5 * safe + _GAUSSIAN_SPAN * sigma
    grid = jnp.linspace(-reach, reach, _GAUSSIAN_POINTS)

    def gaussian(mean: Array) -> Array:
        return jnp.exp(-0.5 * ((grid - mean) / sigma) ** 2) / (sigma * jnp.sqrt(2.0 * jnp.pi))

    information = _mixture_information(gaussian(0.5 * safe), gaussian(-0.5 * safe), grid, fraction)
    ceiling = information_ceiling(fraction)
    # the 300 is where a fixed point count stops resolving peaks of width sqrt(d^2) on a grid
    # spanning d^2; past it the ceiling deficit is below fp64 anyway, falling as exp(-d^2 / 8)
    return jnp.where(deflection <= 0.0, 0.0, jnp.where(deflection >= 300.0, ceiling, information))


def _mixture_information(
    density_a: Float[Array, " g"],
    density_b: Float[Array, " g"],
    grid: Float[Array, " g"],
    fraction: float,
) -> Float[Array, ""]:
    """The Fisher integral of two densities about their mixing fraction, capped at perfect labels."""
    mixture = fraction * density_a + (1.0 - fraction) * density_b
    # the floor must be representable in the working dtype: 1e-300 flushes to zero in float32, and a
    # well-separated pair has whole stretches of grid where both densities underflow, so the
    # integrand would be 0/0 exactly where the answer is the ceiling
    information = jnp.trapezoid(
        (density_a - density_b) ** 2 / jnp.maximum(mixture, jnp.finfo(mixture.dtype).tiny), grid
    )
    # the ceiling is a theorem, so enforce it rather than letting quadrature error out-vote it
    return jnp.clip(information, 0.0, information_ceiling(fraction))
