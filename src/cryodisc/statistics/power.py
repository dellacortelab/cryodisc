"""Reading a per-particle deflection as a particle count.

Under the equal-variance Gaussian model a particle of deflection `d^2` carries Fisher information
`I_p(d^2)` about the population fraction, information adds over particles, and the Cramer-Rao bound
`Var(p_hat) >= 1 / (N I_p)` is what every function here rearranges. `I_p` is capped at `1/[p(1-p)]`,
so near the cap a count is a bound rather than a precision -- read `saturation` alongside it.
`efficiency` scales `I_p` to what a real pipeline gets and defaults to 1.0, the theoretical ceiling.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from cryodisc.statistics.information import fisher_from_deflection, information_ceiling


@eqx.filter_jit
def information_per_particle(
    deflection: Float[Array, ""] | float,
    *,
    fraction: float = 0.5,
    efficiency: float = 1.0,
) -> Float[Array, ""]:
    """I_p for one particle of deflection `d^2`, scaled by the pipeline efficiency.

    Args:
        deflection: d^2, from a rung or from a measured pool.
        fraction: Population fraction of state A.
        efficiency: Achieved over predicted information for the nearest measured operating point.

    Returns:
        I_p, capped at the perfect-label ceiling.
    """
    _check(fraction, efficiency)
    return _capped(efficiency * fisher_from_deflection(deflection, fraction), fraction)


def saturation(
    information: Float[Array, ""] | float,
    *,
    fraction: float = 0.5,
) -> Float[Array, ""]:
    """How close an I_p sits to the ceiling.

    Args:
        information: I_p.
        fraction: Population fraction of state A.

    Returns:
        The ratio to `information_ceiling(fraction)`. At 0.99 or above every count here is limited
        by the ceiling rather than by d^2, and d^2 is no longer recoverable from I_p.
    """
    return jnp.asarray(information) / information_ceiling(fraction)


@eqx.filter_jit
def fraction_error(
    deflection: Float[Array, ""] | float,
    *,
    n_particles: Float[Array, ""] | float,
    fraction: float = 0.5,
    efficiency: float = 1.0,
) -> Float[Array, ""]:
    """The Cramer-Rao standard error on the recovered mixing fraction.

    Args:
        deflection: d^2.
        n_particles: Particles collected.
        fraction: Population fraction of state A.
        efficiency: Achieved over predicted information.

    Returns:
        `1 / sqrt(N I_p)`, infinite where the pair carries no information.
    """
    _check(fraction, efficiency)
    total = n_particles * information_per_particle(deflection, fraction=fraction, efficiency=efficiency)
    return jnp.where(total > 0.0, 1.0 / jnp.sqrt(jnp.maximum(total, 1e-300)), jnp.inf)


@eqx.filter_jit
def n_for_error(
    deflection: Float[Array, ""] | float,
    *,
    error_target: float,
    fraction: float = 0.5,
    efficiency: float = 1.0,
) -> Float[Array, ""]:
    """Particles needed to pin the fraction to `error_target`.

    Args:
        deflection: d^2.
        error_target: The standard error on the fraction that counts as success.
        fraction: Population fraction of state A.
        efficiency: Achieved over predicted information.

    Returns:
        `N = 1 / (I_p sigma^2)`, infinite where the pair carries no information.
    """
    _check(fraction, efficiency)
    return n_for_error_from_information(
        fisher_from_deflection(deflection, fraction),
        error_target=error_target,
        fraction=fraction,
        efficiency=efficiency,
    )


@eqx.filter_jit
def n_for_error_from_information(
    information: Float[Array, ""] | float,
    *,
    error_target: float,
    fraction: float = 0.5,
    efficiency: float = 1.0,
) -> Float[Array, ""]:
    """`n_for_error` when I_p is already in hand rather than a single deflection.

    Information adds while `I_p(d^2)` is nonlinear, so a pool whose deflection varies over pose or
    defocus carries `E[I_p(d^2)]`, not `I_p(E[d^2])`: average the information and come in here. The
    Jensen gap changes sign with `fraction`, since I_p is concave, and so the mean deflection is
    optimistic only for `0.2113 < p < 0.7887`.

    Args:
        information: I_p, already averaged over whatever the pool varies in.
        error_target: The standard error on the fraction that counts as success.
        fraction: Population fraction of state A.
        efficiency: Achieved over predicted information.

    Returns:
        `N = 1 / (I_p sigma^2)`, infinite where the information is zero.
    """
    _check(fraction, efficiency)
    capped = _capped(efficiency * jnp.asarray(information), fraction)
    return jnp.where(capped > 0.0, 1.0 / (jnp.maximum(capped, 1e-300) * error_target**2), jnp.inf)


def _check(fraction: float, efficiency: float) -> None:
    """Reject the parameters that would make every count here meaningless."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be strictly between 0 and 1, got {fraction}")
    if efficiency <= 0.0:
        raise ValueError(f"efficiency must be positive, got {efficiency}")


def _capped(information: Array, fraction: Float[Array, ""] | float) -> Float[Array, ""]:
    """Clip to the perfect-label ceiling, which a measured efficiency above one can otherwise breach."""
    return jnp.clip(information, 0.0, information_ceiling(fraction))
