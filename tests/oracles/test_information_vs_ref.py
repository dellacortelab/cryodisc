"""`statistics.information.fisher_from_deflection` and `statistics.power` against the reference.

Production evaluates the Fisher integral on a fixed grid and inverts it in closed form; the reference
evaluates it adaptively via an independent identity and inverts it by root-finding, so agreement is
about the quantity rather than a shared implementation. Production's float32 against the reference's
float64 sets the tolerances, each written next to its reason.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from cryodisc.statistics.information import fisher_from_deflection, information_ceiling
from cryodisc.statistics.power import (
    fraction_error,
    information_per_particle,
    n_for_error,
    saturation,
)

from . import ref_information as ref

FLOAT32_QUADRATURE = 3e-3
"""Agreement a float32 fixed-grid trapezoid can reach against float64 adaptive quadrature."""


@pytest.mark.parametrize("d2", [0.01, 0.5, 2.0, 8.0, 30.0])
@pytest.mark.parametrize("fraction", [0.5, 0.25])
def test_the_two_reference_routes_agree_before_production_is_compared_to_either(d2, fraction):
    """The direct integral and the overlap identity are two arrangements of the same quantity, so a
    disagreement is an arithmetic error in one of the derivations and nothing else. Both are float64
    adaptive, so they agree to quadrature precision."""
    direct = ref.information_by_quadrature(d2, fraction)
    overlap = ref.information_by_overlap(d2, fraction)
    assert direct == pytest.approx(overlap, rel=1e-8)


@pytest.mark.parametrize("d2", [0.01, 0.1, 0.5, 2.0, 8.0, 30.0, 100.0])
def test_the_fisher_map_matches_adaptive_quadrature(d2):
    """The central comparison: production's fixed-grid trapezoid against the reference integral."""
    produced = float(fisher_from_deflection(jnp.asarray(d2)))
    assert produced == pytest.approx(ref.information_by_quadrature(d2), rel=FLOAT32_QUADRATURE)


@pytest.mark.parametrize("fraction", [0.2, 0.35, 0.5, 0.8])
def test_the_fisher_map_matches_at_an_asymmetric_fraction(fraction):
    """The information about a RARE state is not the information about an even mixture, and a
    formula that dropped `fraction` from the mixture density would still pass at 0.5."""
    produced = float(fisher_from_deflection(jnp.asarray(3.0), fraction=fraction))
    assert produced == pytest.approx(ref.information_by_quadrature(3.0, fraction), rel=FLOAT32_QUADRATURE)


def test_the_weak_signal_limit_has_the_derived_quadratic_coefficient():
    """`I_p = t - t^2/4 + O(t^3)` at p = 0.5, derived by expanding both shifted Gaussians. This pins
    a COEFFICIENT rather than a power: it says the information falls BELOW the deflection
    immediately, so I_p is concave at the origin and not merely near the ceiling."""
    for d2 in (1e-3, 1e-2, 5e-2):
        produced = float(fisher_from_deflection(jnp.asarray(d2)))
        assert produced == pytest.approx(ref.information_small_d2(d2), rel=5e-3)
        assert produced < d2  # strictly below the deflection, which is the content of the sign


def test_the_saturation_approaches_the_ceiling_at_the_derived_exponential_rate():
    """The deficit falls as `exp(-d^2/8)/sqrt(d^2)`, whose exponent is the Bhattacharyya error
    exponent of the same model -- an external consistency check on the derivation. Compared on the
    DEFICIT, since the relative correction to the asymptote is O(1/d^2)."""
    ceiling = float(information_ceiling(0.5))
    for d2 in (50.0, 80.0):
        produced_deficit = ceiling - float(fisher_from_deflection(jnp.asarray(d2)))
        reference_deficit = ceiling - ref.information_large_d2(d2)
        assert produced_deficit == pytest.approx(reference_deficit, rel=0.15)


def test_the_map_is_monotone_and_bounded():
    """Both are theorems -- the densities separate monotonically and the overlap is non-negative --
    so they are enforced rather than sampled."""
    grid = np.geomspace(1e-4, 500.0, 60)
    values = np.array([float(fisher_from_deflection(jnp.asarray(t))) for t in grid])
    assert np.all(np.diff(values) >= -1e-6)
    assert np.all(values <= float(information_ceiling(0.5)) + 1e-9)
    assert float(fisher_from_deflection(jnp.asarray(0.0))) == 0.0


def test_the_map_inverts_through_the_reference_root_finder():
    """A round trip that goes out through production and back through `brentq` on the reference
    forward map. It closes only if the two maps agree everywhere between, not just at the endpoint."""
    for d2 in (0.2, 1.0, 5.0, 15.0):
        produced = float(fisher_from_deflection(jnp.asarray(d2)))
        recovered = ref.d2_from_information(produced)
        assert recovered == pytest.approx(d2, rel=0.02)


def test_the_efficiency_scales_the_information_and_the_cap_still_holds():
    """A measured efficiency above one can otherwise breach a ceiling that is a theorem."""
    d2 = 6.0
    full = float(information_per_particle(jnp.asarray(d2)))
    half = float(information_per_particle(jnp.asarray(d2), efficiency=0.5))
    assert half == pytest.approx(0.5 * full, rel=1e-4)
    assert float(information_per_particle(jnp.asarray(d2), efficiency=100.0)) == pytest.approx(
        float(information_ceiling(0.5)), rel=1e-6
    )


def test_the_saturation_reading_is_the_fraction_of_the_ceiling():
    """The column that says whether a count is limited by `d^2` or by perfect labels."""
    d2 = 10.0
    information = information_per_particle(jnp.asarray(d2))
    assert float(saturation(information)) == pytest.approx(
        float(information) / float(information_ceiling(0.5)), rel=1e-6
    )


@pytest.mark.parametrize("d2", [0.1, 1.0, 10.0])
def test_the_standard_error_matches_the_reference_definition(d2):
    """`1/sqrt(N I_p)`, with the reference's `I_p` from adaptive quadrature."""
    n_particles = 5000.0
    produced = float(fraction_error(jnp.asarray(d2), n_particles=n_particles))
    assert produced == pytest.approx(ref.standard_error_of_fraction(d2, n_particles), rel=FLOAT32_QUADRATURE)


@pytest.mark.parametrize("d2", [0.1, 1.0, 10.0])
def test_the_particle_count_matches_a_root_found_one(d2):
    """Production rearranges `sigma = 1/sqrt(N I_p)` in closed form; the reference SOLVES it. So
    this checks the rearrangement rather than repeating it."""
    target = 0.02
    produced = float(n_for_error(jnp.asarray(d2), error_target=target))
    assert produced == pytest.approx(ref.n_for_error(d2, target), rel=FLOAT32_QUADRATURE)


def test_the_count_scales_as_the_inverse_square_of_the_target():
    """An exact invariance of the definition, independent of the Fisher integral entirely."""
    d2 = 2.0
    coarse = float(n_for_error(jnp.asarray(d2), error_target=0.04))
    fine = float(n_for_error(jnp.asarray(d2), error_target=0.02))
    assert fine == pytest.approx(4.0 * coarse, rel=1e-5)


def test_a_pair_with_no_information_needs_infinitely_many_particles():
    """The boundary case, which a formula that divided by a floored information would report as a
    large finite number."""
    assert float(n_for_error(jnp.asarray(0.0), error_target=0.05)) == np.inf
    assert float(fraction_error(jnp.asarray(0.0), n_particles=1000.0)) == np.inf
    assert ref.n_for_error(0.0, 0.05) == np.inf
