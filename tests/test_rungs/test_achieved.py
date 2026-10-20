"""Checks on reading responsibilities back as information.

The identities are exact, so most of these are one line. The rest pin the two failures the readout
exists to expose, and which fraction the score has to be evaluated at to see each.
"""

import jax
import jax.numpy as jnp
import pytest

from cryodisc.rungs.achieved import (
    achieved_information,
    calibration_curve,
    fraction_bias,
    pipeline_efficiency,
)
from cryodisc.statistics.information import fisher_from_deflection, information_ceiling
from cryodisc.statistics.power import saturation


@pytest.mark.parametrize("fraction", [0.5, 0.25, 0.01])
def test_true_labels_reach_the_ceiling_exactly(fraction):
    """`sum (r - p)^2 = n p (1 - p)` identically at 0/1 labels, so perfect recovery hits the cap."""
    n = 10_000
    labels = (jnp.arange(n) < fraction * n).astype(float)
    assert float(achieved_information(labels)) == pytest.approx(float(information_ceiling(fraction)), rel=1e-5)


def test_responsibilities_that_separate_nothing_carry_nothing():
    assert float(achieved_information(jnp.full(1000, 0.3))) < 1e-9


def test_a_true_posterior_recovers_the_predicted_information():
    """Under the equal-variance Gaussian model `r = sigmoid(Lambda)` is exact, so efficiency is one."""
    d2 = 1.0
    key = jax.random.key(0)
    noise = jax.random.normal(key, (200_000,)) * jnp.sqrt(d2)
    log_ratio = noise + jnp.where(jnp.arange(200_000) < 100_000, 0.5 * d2, -0.5 * d2)
    responsibilities = jax.nn.sigmoid(log_ratio)
    assert float(pipeline_efficiency(responsibilities, deflection=d2, fraction=0.5)) == pytest.approx(1.0, rel=0.02)


def test_a_degraded_posterior_loses_information():
    """Shrinking the responsibilities toward the fraction is what a lossy reconstruction does."""
    labels = (jnp.arange(1000) < 500).astype(float)
    assert float(pipeline_efficiency(0.5 + 0.3 * (labels - 0.5), deflection=1e4, fraction=0.5)) < 0.5


def test_the_default_reading_cannot_report_its_own_miscalibration():
    """It sits at the ceiling of whichever fraction it names, which is why it is blind to the wrong one."""
    responsibilities = jnp.concatenate([jnp.ones(10), jnp.zeros(990)])  # separates perfectly, at 0.01
    information = achieved_information(responsibilities)
    assert float(information) == pytest.approx(float(information_ceiling(0.01)), rel=1e-5)
    assert float(saturation(information, fraction=0.01)) == pytest.approx(1.0, rel=1e-5)
    # 25x what the true even mixture permits, and no reading at 0.01 can say so
    assert float(information) / float(information_ceiling(0.5)) == pytest.approx(25.25, rel=1e-3)
    assert abs(float(fraction_bias(responsibilities, fraction=0.5))) > 30.0


def test_anticalibrated_responsibilities_claim_impossible_information():
    """A swapped label convention against a known fraction, which reads as a negative loss."""
    responsibilities = jnp.concatenate([jnp.zeros(10), jnp.ones(990)])  # the 1% that are A get 0
    information = achieved_information(responsibilities, fraction=0.01)
    assert float(saturation(information, fraction=0.01)) == pytest.approx(98.0, rel=0.01)
    assert float(pipeline_efficiency(responsibilities, deflection=1e4, fraction=0.01)) > 1.0


def test_scoring_at_the_estimators_own_fraction_stays_under_the_ceiling():
    """The bound that makes the default self-limiting, on responsibilities with no structure at all."""
    responsibilities = jax.random.uniform(jax.random.key(1), (5000,))
    fraction = float(responsibilities.mean())
    assert float(saturation(achieved_information(responsibilities), fraction=fraction)) <= 1.0


def test_a_degenerate_fraction_carries_no_information_rather_than_dividing_by_zero():
    assert float(achieved_information(jnp.zeros(100))) == 0.0
    assert float(achieved_information(jnp.ones(100), fraction=1.0)) == 0.0


def test_the_curve_is_flat_for_responsibilities_the_labels_agree_with():
    """Labels drawn at rate r_i are calibrated by construction, so predicted and empirical coincide."""
    key_r, key_y = jax.random.split(jax.random.key(2))
    responsibilities = jax.random.uniform(key_r, (200_000,))
    labels = (jax.random.uniform(key_y, (200_000,)) < responsibilities).astype(int)
    curve = calibration_curve(responsibilities, labels)
    assert float(curve.expected_error) < 0.01
    assert float(jnp.nanmax(jnp.abs(curve.predicted - curve.empirical))) < 0.02
    assert float(curve.expected_error) < float(calibration_curve(responsibilities, 1 - labels).expected_error)


def test_the_curve_accounts_for_every_particle_and_marks_empty_bins():
    responsibilities = jnp.concatenate([jnp.full(60, 0.05), jnp.full(40, 0.95)])
    curve = calibration_curve(responsibilities, jnp.arange(100) >= 60)
    assert int(curve.count.sum()) == 100
    assert int(curve.count[0]) == 60 and int(curve.count[-1]) == 40
    assert bool(jnp.isnan(curve.predicted[5])) and bool(jnp.isnan(curve.empirical[5]))
    assert float(curve.expected_error) == pytest.approx(0.05, rel=1e-5)


def test_the_deflection_and_the_responsibility_readings_agree_on_a_saturated_pair():
    """Both are capped by perfect labels, so a separable pair at either reading is at 1/[p(1-p)]."""
    labels = (jnp.arange(1000) < 250).astype(float)
    assert float(achieved_information(labels)) == pytest.approx(float(fisher_from_deflection(1e3, 0.25)), rel=1e-6)
