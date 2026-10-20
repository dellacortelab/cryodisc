"""Checks on the canonical discriminability and the count it converts to.

The equal-variance Gaussian model gives the deflection and its Fisher map a closed form, so most of
these compare against it. The rest pin the properties that make the deflection the right d^2: it is
non-negative with a sampling floor that vanishes as the pool grows, affine-invariant, and symmetric
under swapping the two pools.
"""

import numpy as np
import pytest
from conftest import DEFLECTIONS, gaussian_pools
from jax import numpy as jnp

from cryodisc.statistics.information import (
    deflection,
    deflection_standard_error,
    fisher_from_deflection,
)

POOL = 40_000
"""Particles per pool where the size is not itself the subject; tests that turn on `n` state their own."""


@pytest.mark.parametrize("squared_distance", DEFLECTIONS)
def test_gaussian_pools_recover_their_own_parameter(squared_distance):
    """On a single-pose pool the deflection reads the separation it was built from."""
    from_a, from_b = gaussian_pools(np.random.default_rng(0), squared_distance, POOL)
    assert float(deflection(from_a, from_b)) == pytest.approx(squared_distance, rel=0.03)


@pytest.mark.parametrize("squared_distance", DEFLECTIONS)
def test_the_error_bar_is_the_spread_of_its_own_estimate(squared_distance):
    """The reported error has to be the sd of the estimator, so a 3-sigma test means 3 sigma.

    The deflection is a ratio, so it has no tidy closed form the way a mean gap does; check the
    reported bar against the scatter of independent repeats, which is what it claims to be.
    """
    n = 400
    rng = np.random.default_rng(7)
    estimates, reported = [], []
    for _ in range(200):
        from_a, from_b = gaussian_pools(rng, squared_distance, n)
        estimates.append(float(deflection(from_a, from_b)))
        reported.append(float(deflection_standard_error(from_a, from_b)))

    assert float(np.mean(reported)) == pytest.approx(float(np.std(estimates, ddof=1)), rel=0.15)


def test_the_error_bar_shrinks_as_the_root_of_the_pool_size():
    """Quadrupling a pool must halve its error bar, which is what makes 3 sigma mean anything."""
    rng = np.random.default_rng(8)
    small = deflection_standard_error(*gaussian_pools(rng, 4.0, 1_000))
    large = deflection_standard_error(*gaussian_pools(rng, 4.0, 4_000))
    assert float(small / large) == pytest.approx(2.0, rel=0.1)


def test_deflection_reads_the_mixture_of_a_mixed_pose_pool():
    """Half the pool at d^2 = 1 and half at 9, so the means average to a gap of 5, but the pooled
    variance also picks up the spread of those means (5 + 4 = 9), giving a deflection of 25/9. This
    is the honest reading of a pose-marginalized pool, whose per-particle informativeness varies."""
    rng = np.random.default_rng(1)
    parts_a, parts_b = zip(*(gaussian_pools(rng, d2, 60_000) for d2 in (1.0, 9.0)))
    from_a, from_b = jnp.concatenate(parts_a), jnp.concatenate(parts_b)

    assert float(deflection(from_a, from_b)) == pytest.approx(25.0 / 9.0, rel=0.03)


def test_deflection_reads_its_sampling_floor_at_zero_signal():
    """Two independent pools from the same distribution have true d^2 = 0, where the squared gap reads
    its sampling floor `~2/n` rather than zero. No correction is applied: the mean over repeats sits
    at the floor and is never negative -- the floor is a convergence artifact removed by more particles,
    not a bias subtracted into spurious negatives."""
    rng = np.random.default_rng(9)
    n = 2000
    estimates = [
        float(deflection(jnp.asarray(rng.normal(size=n)), jnp.asarray(rng.normal(size=n)))) for _ in range(400)
    ]
    assert float(np.mean(estimates)) == pytest.approx(2.0 / n, rel=0.25)
    assert float(np.min(estimates)) >= 0.0


def test_the_deflection_floor_shrinks_as_the_pool_grows():
    """The naive deflection carries a `+2/n` floor from the squared gap, so at small `n` its mean sits
    above the truth by that much; quadrupling the pool quarters the excess. Convergence removes it,
    which is why the population d^2 is what a converged pool measures."""
    rng = np.random.default_rng(10)
    small = float(np.mean([float(deflection(*gaussian_pools(rng, 1.0, 200))) for _ in range(500)]))
    large = float(np.mean([float(deflection(*gaussian_pools(rng, 1.0, 4_000))) for _ in range(500)]))
    assert small > 1.0  # reads high by the +2/n floor of the squared gap
    assert (large - 1.0) < (small - 1.0)  # the floor shrinks with the pool
    assert large == pytest.approx(1.0, abs=0.005)  # negligible once the pool is large


def test_deflection_is_affine_invariant():
    """The property that lets it tolerate a score only proportional to the true log-ratio: rescaling
    and shifting Lambda leaves the deflection unchanged (it is signal-detection d'^2)."""
    from_a, from_b = gaussian_pools(np.random.default_rng(6), 2.0, POOL)
    scaled_a, scaled_b = 3.0 * from_a + 7.0, 3.0 * from_b + 7.0
    assert float(deflection(scaled_a, scaled_b)) == pytest.approx(float(deflection(from_a, from_b)), rel=1e-4)


def test_deflection_is_symmetric_under_swapping_the_pools():
    """d^2 is a separation, not a signed gap, so which pool is called A cannot change it."""
    from_a, from_b = gaussian_pools(np.random.default_rng(5), 2.0, POOL)
    assert float(deflection(from_b, from_a)) == pytest.approx(float(deflection(from_a, from_b)), rel=1e-5)


def test_fisher_from_deflection_is_the_deflection_in_the_weak_signal_limit():
    """In the experimental regime a particle's information about the fraction IS its deflection:
    `I_p -> d^2` as `d^2 -> 0` at p = 1/2, so a count reads straight off d^2 there."""
    for small in (0.01, 0.04):
        assert float(fisher_from_deflection(small, 0.5)) == pytest.approx(small, rel=0.02)
    increasing = [float(fisher_from_deflection(d2, 0.5)) for d2 in (0.5, 1.0, 2.0)]
    assert increasing == sorted(increasing)


@pytest.mark.parametrize("fraction", [0.5, 0.3, 0.1])
def test_fisher_respects_the_perfect_label_ceiling(fraction):
    """Enforced, not trusted: the quadrature loses the two peaks at large d^2 and would otherwise
    return numbers above a ceiling that is a theorem."""
    ceiling = 1.0 / (fraction * (1.0 - fraction))
    assert float(fisher_from_deflection(1e4, fraction)) == pytest.approx(ceiling)
    assert float(fisher_from_deflection(1e9, fraction)) == pytest.approx(ceiling)


def test_degenerate_inputs_return_zero():
    """No information at zero separation, and identical pools carry no discriminability."""
    assert float(fisher_from_deflection(0.0)) == 0.0
    assert float(fisher_from_deflection(-1.0)) == 0.0

    identical = jnp.asarray(np.random.default_rng(4).normal(size=2000))
    assert float(deflection(identical, identical)) == pytest.approx(0.0, abs=1e-2)
