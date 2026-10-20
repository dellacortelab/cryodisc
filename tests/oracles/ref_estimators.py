"""Independent reference for the discriminability estimators, derived from the mathematics.

Nothing here imports `cryodisc`. The whitened inner product is
`<x, y>_w = sum_k Re[x_k conj(y_k)] / sigma^2_k`, the known-pose log-ratio is
`Lambda(Y) = <mu_A - mu_B, Y - (mu_A + mu_B)/2>_w`, and the discriminability is the mean gap
`d^2 = E[Lambda|A] - E[Lambda|B]`. Hence

    Lambda|A ~ N(+d^2/2, d^2),    Lambda|B ~ N(-d^2/2, d^2)                               (ref-A8)

The finite-sample expectations are exact, not asymptotic. Derivations: docs/rationale/tests/oracles.md
"""

import numpy as np

__all__ = [
    "expected_pooled_deflection",
    "kl_equal_covariance",
    "llr_gaussian",
    "make_strata",
    "ref_deflection",
    "ref_gaussian_information",
    "ref_mean_gap",
    "sample_llr",
    "sample_llr_strata",
    "true_d2_gaussian",
    "whitened_inner",
]


def whitened_inner(x: np.ndarray, y: np.ndarray, noise_variance: np.ndarray) -> float:
    """The whitened inner product `sum_k Re[x conj(y)] / sigma^2_k`.

    Args:
        x: First vector, possibly complex.
        y: Second vector.
        noise_variance: `sigma^2` per mode.

    Returns:
        The inner product.
    """
    return float(np.sum(np.real(np.asarray(x) * np.conj(np.asarray(y))) / np.asarray(noise_variance)))


def true_d2_gaussian(mean_a: np.ndarray, mean_b: np.ndarray, noise_variance: np.ndarray) -> float:
    """(ref-1), the squared Mahalanobis separation for a diagonal covariance.

    Args:
        mean_a: `mu_A`.
        mean_b: `mu_B`.
        noise_variance: `sigma^2` per mode.

    Returns:
        `d^2`. No factor of 4 and no factor of 1/2.
    """
    difference = np.asarray(mean_a) - np.asarray(mean_b)
    return whitened_inner(difference, difference, noise_variance)


def kl_equal_covariance(mean_a: np.ndarray, mean_b: np.ndarray, noise_variance: np.ndarray) -> float:
    """`KL(P_A||P_B) = d^2/2`, so that the symmetrized sum comes to `d^2`.

    Args:
        mean_a: `mu_A`.
        mean_b: `mu_B`.
        noise_variance: `sigma^2` per mode.

    Returns:
        The one-way divergence.
    """
    return 0.5 * true_d2_gaussian(mean_a, mean_b, noise_variance)


def llr_gaussian(data: np.ndarray, mean_a: np.ndarray, mean_b: np.ndarray, noise_variance: np.ndarray) -> np.ndarray:
    """`Lambda = <mu_A - mu_B, Y - (mu_A + mu_B)/2>_w` for a stack of observations.

    Args:
        data: `(n, npix)` observations, possibly complex.
        mean_a: `mu_A`, `(npix,)`.
        mean_b: `mu_B`, `(npix,)`.
        noise_variance: `sigma^2` per mode.

    Returns:
        `(n,)` log-ratios.
    """
    observations = np.atleast_2d(np.asarray(data))
    difference = np.asarray(mean_a) - np.asarray(mean_b)
    midpoint = 0.5 * (np.asarray(mean_a) + np.asarray(mean_b))
    weight = difference / np.asarray(noise_variance)
    return np.real((observations - midpoint) @ np.conj(weight))


def sample_llr(d2: float, n_a: int, n_b: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw log-ratios from (ref-A8) at a single true index.

    Args:
        d2: The true index.
        n_a: Particles in state A.
        n_b: Particles in state B.
        rng: Random generator.

    Returns:
        `(log_ratio, is_a)`.
    """
    spread = np.sqrt(d2)
    values = np.concatenate([rng.normal(+0.5 * d2, spread, n_a), rng.normal(-0.5 * d2, spread, n_b)])
    is_a = np.zeros(n_a + n_b, dtype=bool)
    is_a[:n_a] = True
    return values, is_a


def make_strata(mean_index: float, cv: float, n_strata: int) -> np.ndarray:
    """Per-stratum indices with EXACTLY the given mean and coefficient of variation.

    So the truth of a mixed pool is an input rather than a fit. Built from an equally spaced
    template standardized to mean 0 and POPULATION sd 1 -- population, because the strata are the
    whole population of poses here -- then mapped affinely. Only the mean and the variance enter
    (ref-5), so the shape is immaterial; equal spacing keeps every stratum positive up to
    `cv = 1/sqrt(3)`.

    Args:
        mean_index: `E[D]`.
        cv: Coefficient of variation of `D`.
        n_strata: Number of strata.

    Returns:
        `(n_strata,)` indices.

    Raises:
        ValueError: If the spread would make a stratum non-positive.
    """
    template = np.linspace(-1.0, 1.0, n_strata)
    standardized = (template - template.mean()) / template.std()
    strata = mean_index * (1.0 + cv * standardized)
    if np.any(strata <= 0.0):
        raise ValueError(f"cv={cv} makes a non-positive stratum at mean_index={mean_index}")
    return strata


def sample_llr_strata(strata: np.ndarray, n_per: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    """A mixed-pose pool: `n_per` particles of each state at every stratum.

    Every particle is (ref-A8) at its OWN index, which is what a pose-known score on a pool of
    varying true poses is.

    Args:
        strata: Per-stratum indices, from `make_strata`.
        n_per: Particles of each state per stratum.
        rng: Random generator.

    Returns:
        `(log_ratio, is_a, mean_index)`.
    """
    indices = np.asarray(strata, dtype=float)
    pool_a = np.concatenate([rng.normal(+0.5 * d, np.sqrt(d), n_per) for d in indices])
    pool_b = np.concatenate([rng.normal(-0.5 * d, np.sqrt(d), n_per) for d in indices])
    values = np.concatenate([pool_a, pool_b])
    is_a = np.zeros(values.size, dtype=bool)
    is_a[: pool_a.size] = True
    return values, is_a, float(indices.mean())


def ref_mean_gap(log_ratio: np.ndarray, is_a: np.ndarray) -> float:
    """The mean gap read directly, which the discriminability is built from and equals when calibrated.

    Args:
        log_ratio: Per-particle `Lambda`.
        is_a: True where the particle is state A.

    Returns:
        `mean(Lambda|A) - mean(Lambda|B)`.
    """
    values = np.asarray(log_ratio, dtype=float)
    mask = np.asarray(is_a, dtype=bool)
    return float(values[mask].mean() - values[~mask].mean())


def ref_deflection(log_ratio: np.ndarray, is_a: np.ndarray) -> float:
    """The discriminability `gap^2 / within`, pooled WITHIN class at ddof=1.

    Within-class pooling is what (ref-5) requires: a grand variance would also absorb the
    between-state separation `gap^2/4`, which is signal and not dispersion. No sampling correction is
    applied, so the estimate is non-negative and carries a `~2/n` floor that a converged pool removes.

    Args:
        log_ratio: Per-particle `Lambda`.
        is_a: True where the particle is state A.

    Returns:
        The discriminability, invariant to any affine rescaling of `Lambda`.
    """
    values = np.asarray(log_ratio, dtype=float)
    mask = np.asarray(is_a, dtype=bool)
    pool_a, pool_b = values[mask], values[~mask]
    var_a, var_b = pool_a.var(ddof=1), pool_b.var(ddof=1)
    gap = pool_a.mean() - pool_b.mean()
    within = (var_a * pool_a.size + var_b * pool_b.size) / values.size
    return float(gap * gap / within)


def ref_gaussian_information(d2: float, fraction: float = 0.5, n_points: int = 8001) -> float:
    """The exact Fisher information about the fraction under (ref-A8): the two Gaussians `N(+-d^2/2, d^2)`.

    `I_p = integral (f_A - f_B)^2 / (p f_A + (1 - p) f_B) dLambda`, capped at the perfect-label
    `1 / [p (1 - p)]`. The Gaussian reference `fisher_from_deflection` must reproduce.

    Args:
        d2: The discriminability.
        fraction: Population fraction of state A.
        n_points: Quadrature points across an eight-sigma span.

    Returns:
        `I_p`.
    """
    sigma = np.sqrt(d2)
    reach = 0.5 * d2 + 8.0 * sigma
    grid = np.linspace(-reach, reach, n_points)

    def gaussian(mean: float) -> np.ndarray:
        return np.exp(-0.5 * ((grid - mean) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))

    density_a, density_b = gaussian(0.5 * d2), gaussian(-0.5 * d2)
    mixture = fraction * density_a + (1.0 - fraction) * density_b
    integrand = (density_a - density_b) ** 2 / np.maximum(mixture, np.finfo(mixture.dtype).tiny)
    ceiling = 1.0 / (fraction * (1.0 - fraction))
    return float(min(np.trapezoid(integrand, grid), ceiling))


def expected_pooled_deflection(mean_index: float, cv: float) -> float:
    """(ref-5), the population deflection of a mixed-pose pool.

    Args:
        mean_index: `E[D]`.
        cv: Coefficient of variation of `D` across the pool.

    Returns:
        The expectation. It is BELOW `mean_index` whenever `cv > 0`: a pose-marginalized pool reads
        its own mixture, not the mean per-pose separation.
    """
    variance = mean_index * (1.0 + cv * cv * mean_index / 4.0)
    return mean_index * mean_index / variance
