"""Checks on the retired estimators kept for the SI comparison, in `reproduction/analysis/si_estimators`.

These are the routes the package did not keep -- the derived anchor, the pool calibration, the second
matched estimator, and the stretch that rescales between anchors. They still have to be correct for the
bake-off that justifies the two survivors to be a fair fight.
"""

import numpy as np
import pytest
from jax import numpy as jnp

from cryodisc.statistics.composition import fraction_no_pool
from reproduction.analysis.si_estimators import (
    anchor_from_moments,
    anchors_from_pools,
    calibrate_temperature,
    efficient_fraction,
    observed_anchors,
    stretch,
)


def pure_pools(rng, squared_distance=6.0, n=40_000, overdispersion=1.0):
    """Lambda for a pure state-A and state-B pool, optionally overdispersed by a scale on the score."""
    spread = np.sqrt(squared_distance)
    pool_a = 0.5 * squared_distance + spread * rng.normal(size=n)
    pool_b = -0.5 * squared_distance + spread * rng.normal(size=n)
    return overdispersion * pool_a, overdispersion * pool_b


def _mixture(rng, pool_a, pool_b, fraction, n=8_000):
    n_a = round(fraction * n)
    return np.concatenate([rng.choice(pool_a, n_a), rng.choice(pool_b, n - n_a)])


def test_a_calibrated_pool_is_derived_to_read_exactly_one():
    """`E[exp(-Lambda)] = 1` (coherence one for a Gaussian) is where the derived anchor must saturate,
    for any `d^2` -- a calibrated log-ratio recovers a pure pool exactly however weak the signal."""
    for d2 in (0.01, 1.0, 4.0, 100.0):
        assert anchor_from_moments(d2 / 2, d2) == 1.0
    assert anchor_from_moments(3.0, 4.0) == 1.0  # over-confident saturates too
    assert anchor_from_moments(1.0, 4.0) < 1.0  # under-confident does not


def test_the_derived_anchor_falls_as_the_pool_gets_less_coherent():
    """At fixed variance the anchor is monotone in the mean, reaching the 1.000 corner a linear model
    in coherence could not."""
    anchors = [anchor_from_moments(mean, 4.0) for mean in (0.0, 0.5, 1.0, 1.5, 2.0)]
    assert anchors == sorted(anchors)
    assert anchors[0] == pytest.approx(0.5, abs=0.02)
    assert anchors[-1] == 1.0
    assert anchor_from_moments(0.5, 2.0) != pytest.approx(anchor_from_moments(1.0, 4.0), abs=1e-3)


def test_the_derived_anchor_refuses_a_variance_no_pool_has():
    with pytest.raises(ValueError):
        anchor_from_moments(1.0, 0.0)


@pytest.mark.parametrize("fraction", [0.2, 0.5, 0.8])
def test_efficient_fraction_recovers_a_known_fraction(fraction):
    rng = np.random.default_rng(0)
    pool_a, pool_b = pure_pools(rng)
    mixture = _mixture(rng, pool_a, pool_b, fraction)
    assert efficient_fraction(mixture, pool_a, pool_b) == pytest.approx(fraction, abs=0.03)


def test_efficient_fraction_is_scale_invariant_where_the_fixed_point_is_biased():
    rng = np.random.default_rng(1)
    pool_a, pool_b = pure_pools(rng, overdispersion=4.0)
    mixture = _mixture(rng, pool_a, pool_b, 0.25)
    assert efficient_fraction(mixture, pool_a, pool_b) == pytest.approx(0.25, abs=0.02)
    assert abs(float(fraction_no_pool(jnp.asarray(mixture))) - 0.25) > 0.025


def test_calibrate_temperature_unbiases_the_fixed_point_on_an_overdispersed_score():
    rng = np.random.default_rng(2)
    pool_a, pool_b = pure_pools(rng, overdispersion=3.0)
    scale, shift = calibrate_temperature(pool_a, pool_b)
    calibrated_a, calibrated_b = scale * pool_a + shift, scale * pool_b + shift
    gap = float(calibrated_a.mean() - calibrated_b.mean())
    variance = float(0.5 * (calibrated_a.var() + calibrated_b.var()))
    assert gap == pytest.approx(variance, rel=0.05)  # honest-Gaussian self-normalization restored
    mixture = _mixture(rng, pool_a, pool_b, 0.25)
    recovered = float(fraction_no_pool(jnp.asarray(scale * mixture + shift)))
    assert recovered == pytest.approx(0.25, abs=0.03)


def test_calibrate_temperature_refuses_a_variance_no_pool_has():
    constant = np.full(1_000, 2.0)
    with pytest.raises(ValueError, match="variance"):
        calibrate_temperature(constant, constant)


def test_calibrate_temperature_refuses_pools_in_the_wrong_order():
    """A swapped pair sign-flips the calibrated score, which would make the fixed point read 1 - p."""
    rng = np.random.default_rng(5)
    pool_a, pool_b = pure_pools(rng)
    with pytest.raises(ValueError, match="gap"):
        calibrate_temperature(pool_b, pool_a)


def test_the_stretch_inverts_the_anchors_it_is_given():
    """`stretch` maps `f_b -> 0` and `f_a -> 1`; `anchors_from_pools` supplies both from the moments."""
    rng = np.random.default_rng(6)
    pool_a, pool_b = pure_pools(rng)
    f_a, f_b = anchors_from_pools(pool_a, pool_b)
    assert f_b < 0.5 < f_a  # the B anchor is mirrored to the low side
    assert stretch(np.array([f_b, f_a]), f_a, f_b) == pytest.approx([0.0, 1.0])


def test_observed_anchors_need_both_pure_ends_in_the_sweep():
    with pytest.raises(ValueError, match="both pure pools"):
        observed_anchors(np.array([0.25, 0.5, 0.75]), np.array([0.3, 0.5, 0.7]))
