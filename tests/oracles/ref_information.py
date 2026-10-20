"""Independent reference for the Fisher information about the population fraction, and the counts.

Nothing here imports `cryodisc`; every integral is adaptive quadrature and every
inverse a root-find, so agreement is between two methods rather than a method and itself.

    I_p = integral (f_A - f_B)^2 / (p f_A + (1-p) f_B) dLambda                              (C5)

Densities from `ref_estimators` (ref-A8); the overlap identity makes the ceiling `I_p <= 1/[p(1-p)]`
a theorem. Derivations: docs/rationale/tests/oracles.md
"""

import itertools
import math
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq

__all__ = [
    "d2_from_information",
    "information_by_overlap",
    "information_by_quadrature",
    "information_ceiling",
    "information_large_d2",
    "information_small_d2",
    "n_for_error",
    "standard_error_of_fraction",
]

_QUAD = {"limit": 400, "epsabs": 1e-12, "epsrel": 1e-11}


def _integrate(function, low: float, high: float) -> float:
    """`scipy.integrate.quad` with QUADPACK's roundoff warning silenced.

    The warning fires only at very small `d^2`, where the answer is `d^2` to many digits by the
    series above, so it is noise rather than information.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        value, _ = quad(function, low, high, **_QUAD)
    return float(value)


def information_ceiling(fraction: float = 0.5) -> float:
    """`1 / [p (1 - p)]`, the most information one particle can carry about the fraction.

    Args:
        fraction: Population fraction of state A.

    Returns:
        The ceiling, 4 at `fraction = 0.5`. Reached only by perfect per-particle labels; see the
        overlap identity above for why this is a theorem.
    """
    return 1.0 / (fraction * (1.0 - fraction))


def _densities(x: np.ndarray, d: float) -> tuple[np.ndarray, np.ndarray]:
    """The two standardized conditional densities `phi(x -+ d/2)`."""
    normalization = 1.0 / math.sqrt(2.0 * math.pi)
    return (
        normalization * np.exp(-0.5 * (x - 0.5 * d) ** 2),
        normalization * np.exp(-0.5 * (x + 0.5 * d) ** 2),
    )


def _range(d: float) -> tuple[float, float, list[float]]:
    """Integration limits and interior break points, in the standardized variable.

    Limits reach far into the tails; the break points sit at the two means, where the integrand's
    structure is.
    """
    reach = 0.5 * d + 40.0
    return -reach, reach, [-0.5 * d, 0.0, 0.5 * d]


def information_by_quadrature(d2: float, fraction: float = 0.5) -> float:
    """(C5) by adaptive quadrature in the standardized variable. The reference forward map.

    Args:
        d2: The index.
        fraction: Population fraction of state A.

    Returns:
        `I_p`.
    """
    if d2 <= 0.0:
        return 0.0
    d = math.sqrt(d2)
    other = 1.0 - fraction

    def integrand(x: float) -> float:
        density_a, density_b = _densities(np.asarray(x), d)
        mixture = fraction * density_a + other * density_b
        return float((density_a - density_b) ** 2 / mixture) if mixture > 0.0 else 0.0

    low, high, breaks = _range(d)
    edges = [low, *breaks, high]
    return sum(_integrate(integrand, a, b) for a, b in itertools.pairwise(edges))


def information_by_overlap(d2: float, fraction: float = 0.5) -> float:
    """(C5) by the exact overlap identity. An algebraically independent second route.

    Args:
        d2: The index.
        fraction: Population fraction of state A.

    Returns:
        `I_p`, which must agree with `information_by_quadrature` to quadrature precision. It is a
        different arrangement of the same integral, so a disagreement is an arithmetic error in one
        of the two derivations and nothing else.
    """
    if d2 <= 0.0:
        return 0.0
    d = math.sqrt(d2)
    other = 1.0 - fraction

    def integrand(x: float) -> float:
        density_a, density_b = _densities(np.asarray(x), d)
        mixture = fraction * density_a + other * density_b
        return float(density_a * density_b / mixture) if mixture > 0.0 else 0.0

    low, high, breaks = _range(d)
    edges = [low, *breaks, high]
    overlap = sum(_integrate(integrand, a, b) for a, b in itertools.pairwise(edges))
    return (1.0 - overlap) / (fraction * (1.0 - fraction))


def information_small_d2(d2: float, fraction: float = 0.5) -> float:
    """The weak-signal series `I_p = t + (3 a^2 - 1/4) t^2`, derived in the module docstring.

    Args:
        d2: The index.
        fraction: Population fraction of state A.

    Returns:
        The two-term expansion. Pins a COEFFICIENT rather than a power, which is what makes it a
        test rather than a sanity check.
    """
    asymmetry = 0.5 * (1.0 - 2.0 * fraction)
    return d2 + (3.0 * asymmetry * asymmetry - 0.25) * d2 * d2


def information_large_d2(d2: float, fraction: float = 0.5) -> float:
    """The saturation asymptote, derived in the module docstring.

    Args:
        d2: The index.
        fraction: Population fraction of state A.

    Returns:
        `ceiling - deficit`, leading order. The relative correction is `O(1/d^2)`, so compare the
        DEFICIT rather than `I_p`.
    """
    product = fraction * (1.0 - fraction)
    deficit = math.sqrt(math.pi / 2.0) * math.exp(-d2 / 8.0) / (math.sqrt(product) * math.sqrt(d2))
    return (1.0 - deficit) / product


def d2_from_information(information: float, fraction: float = 0.5, upper: float = 1.0e4) -> float:
    """Invert `I_p(d^2)` by root-finding on the forward map. No closed form and no series.

    `I_p` is strictly increasing in `d^2` -- the two densities separate monotonically and the
    integrand is non-negative -- so the root is unique.

    Args:
        information: The `I_p` to invert.
        fraction: Population fraction of state A.
        upper: Largest `d^2` searched.

    Returns:
        `d^2`.

    Raises:
        ValueError: If the information is at or above the ceiling, where the inverse is unbounded.
    """
    ceiling = information_ceiling(fraction)
    if information <= 0.0:
        return 0.0
    if information >= ceiling:
        raise ValueError(f"I_p = {information} is at or above the ceiling {ceiling}; d^2 is unbounded there")
    return float(brentq(lambda t: information_by_quadrature(t, fraction) - information, 1e-12, upper, xtol=1e-12))


def standard_error_of_fraction(d2: float, n_particles: float, fraction: float = 0.5) -> float:
    """The Cramer-Rao standard error on the recovered fraction, from its definition.

    Args:
        d2: The index.
        n_particles: Particles collected.
        fraction: Population fraction of state A.

    Returns:
        `1 / sqrt(N I_p)`, infinite where the pair carries no information.
    """
    total = n_particles * information_by_quadrature(d2, fraction)
    return math.inf if total <= 0.0 else 1.0 / math.sqrt(total)


def n_for_error(d2: float, error_target: float, fraction: float = 0.5) -> float:
    """Particles needed to reach a target standard error, by root-finding on the forward map.

    Deliberately NOT the closed form `1 / (I_p sigma^2)`: solving `standard_error(N) = target`
    numerically checks the rearrangement production performs rather than repeating it.

    Args:
        d2: The index.
        error_target: The standard error that counts as success.
        fraction: Population fraction of state A.

    Returns:
        The particle count, infinite where the pair carries no information.
    """
    if information_by_quadrature(d2, fraction) <= 0.0:
        return math.inf
    return float(brentq(lambda n: standard_error_of_fraction(d2, n, fraction) - error_target, 1e-6, 1e18, xtol=1e-6))
