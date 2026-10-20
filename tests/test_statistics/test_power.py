"""Checks on the deflection-to-particle-count conversion.

Every count is a rearrangement of `Var(p_hat) = 1/(N I_p)`, so the round trips are exact and most of
these are worth only one line. The rest pin the two behaviors that make the counts misreadable: the
perfect-label ceiling, and the Jensen gap between averaging deflections and averaging information.
"""

import jax.numpy as jnp
import pytest

from cryodisc.statistics.information import fisher_from_deflection, information_ceiling
from cryodisc.statistics.power import (
    fraction_error,
    information_per_particle,
    n_for_error,
    n_for_error_from_information,
    saturation,
)

WEAK = 0.5
SATURATING = 100.0


@pytest.mark.parametrize("error_target", [0.05, 0.02, 0.005])
def test_the_count_and_the_error_bar_invert_each_other(error_target):
    n = n_for_error(WEAK, error_target=error_target)
    assert float(fraction_error(WEAK, n_particles=n)) == pytest.approx(error_target, rel=1e-6)


def test_efficiency_scales_the_count_and_is_applied_once():
    """The scale factor lives in one place, so half the information is exactly twice the particles."""
    full = n_for_error(WEAK, error_target=0.02)
    half = n_for_error(WEAK, error_target=0.02, efficiency=0.5)
    assert float(half) == pytest.approx(2.0 * float(full), rel=1e-6)


def test_a_measured_efficiency_above_one_cannot_breach_the_ceiling():
    """A transfer factor above one is a real measurement, but perfect labels are still the limit."""
    information = information_per_particle(SATURATING, efficiency=2.0)
    assert float(information) <= float(information_ceiling(0.5))
    assert float(saturation(information)) == pytest.approx(1.0, abs=1e-9)


def test_a_saturated_pair_stops_responding_to_its_deflection():
    """Past the ceiling the count measures 1/[p(1-p)], not the physics, so it is a bound."""
    assert float(saturation(information_per_particle(SATURATING))) > 0.99
    near = n_for_error(SATURATING, error_target=0.02)
    far = n_for_error(1.5 * SATURATING, error_target=0.02)
    assert float(far) == pytest.approx(float(near), rel=1e-3)
    # while an unsaturated pair still moves nearly in proportion
    assert float(n_for_error(0.2, error_target=0.02)) > 2.0 * float(n_for_error(0.5, error_target=0.02))


@pytest.mark.parametrize(("fraction", "concave"), [(0.5, True), (0.35, True), (0.05, False), (0.95, False)])
def test_the_jensen_gap_changes_sign_across_the_documented_band(fraction, concave):
    """I_p is concave, so a mean deflection optimistic, only for 0.2113 < p < 0.7887."""
    deflections = jnp.array([0.05, 0.45])
    pointwise = float(jnp.mean(jnp.array([fisher_from_deflection(d, fraction) for d in deflections])))
    at_mean = float(fisher_from_deflection(deflections.mean(), fraction))
    assert (at_mean > pointwise) == concave

    # and so the count from a pose-averaged information differs from the count at a pose-averaged d^2
    honest = float(n_for_error_from_information(pointwise, error_target=0.02, fraction=fraction))
    naive = float(n_for_error(deflections.mean(), error_target=0.02, fraction=fraction))
    assert (naive < honest) == concave


def test_a_pair_carrying_no_information_needs_infinitely_many_particles():
    assert jnp.isinf(n_for_error(0.0, error_target=0.02))
    assert jnp.isinf(fraction_error(0.0, n_particles=1000.0))


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_an_impossible_fraction_is_rejected(fraction):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        n_for_error(WEAK, error_target=0.02, fraction=fraction)


def test_a_nonpositive_efficiency_is_rejected():
    with pytest.raises(ValueError, match="efficiency must be positive"):
        n_for_error(WEAK, error_target=0.02, efficiency=0.0)


@pytest.mark.parametrize(
    ("fraction", "error_target", "floor"), [(0.5, 0.02, 625.0), (0.5, 0.05, 100.0), (0.2, 0.02, 400.0)]
)
def test_the_count_bottoms_out_at_the_binomial_counting_floor(fraction, error_target, floor):
    """`N` cannot fall below `p(1-p)/sigma^2` however discriminable the pair is.

    With perfect per-particle labeling the fraction is still a binomial proportion, so pinning it to
    `sigma` needs `p(1-p)/sigma^2` particles no matter how large `d^2` grows. A flat column at the
    floor is the error target being quoted back, not a result. Pinned as an identity in `p` and
    `sigma` rather than a single number, so a reader can tell the floor from a coincidence.
    """
    enormous = n_for_error(1e6, error_target=error_target, fraction=fraction)

    assert float(enormous) == pytest.approx(floor, rel=1e-3)
    assert float(enormous) == pytest.approx(fraction * (1.0 - fraction) / error_target**2, rel=1e-3)

    # and it is a FLOOR: nothing below it is reachable, from either side of the ceiling
    for deflection in (1.0, 10.0, 100.0, 1e4, 1e8):
        assert float(n_for_error(deflection, error_target=error_target, fraction=fraction)) >= floor * (1 - 1e-6)

    # the ceiling it saturates against is the information one, so the two agree on where the floor is
    assert float(enormous) == pytest.approx(1.0 / (float(information_ceiling(fraction)) * error_target**2), rel=1e-3)
