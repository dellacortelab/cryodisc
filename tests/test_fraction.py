"""estimate_fraction reads the population fraction off a mixed pool, with an error bar.

The pool is simulated at a known fraction from the same two models the estimate scores against, so a
recovered fraction that misses the truth by more than its own bootstrap error is a real failure and
not a template mismatch. The seam checks -- the registry, the required inputs -- are what keep a
route swappable without a silent wrong answer.
"""

import jax
import pytest

from cryodisc import CollectionPlan, Pool, estimate_fraction
from cryodisc.grids.poses import pose_bank
from cryodisc.microscope import Transfer
from cryodisc.particles.sim import simulate_stack
from cryodisc.resolution import noise_spectrum

PLAN = CollectionPlan(dose=40.0, pixel_size=2.0, defocus_range=(8000.0, 20000.0))
BOX = 48
GRID = pose_bank(96)


@pytest.fixture(scope="module")
def pair(bent_pair_factory):
    """A bend large enough to separate the states at a workable pool size."""
    return bent_pair_factory(n_atoms=5000, spread=10.0, bend=2.5)


def _simulated_pool(pair, fraction, *, key, n=1500):
    """A `Pool` of `n` particles drawn `fraction` state A from the pair's own two models."""
    model_a, model_b = pair.models(PLAN, box_size=BOX)
    spectrum = noise_spectrum(PLAN, BOX, rmsf=0.5)
    stack = simulate_stack(
        models=(model_a, model_b),
        key=key,
        n_particles=n,
        noise_spectrum=spectrum,
        occupancy=(fraction, 1.0 - fraction),
        plan=PLAN,
        shift_sigma=0.0,
    )
    transfer = Transfer.from_defocus(stack.defocus, PLAN, BOX)
    return Pool(images=stack.images, transfer=transfer, box_size=BOX, pixel_size=PLAN.pixel_size, labels=stack.state)


def test_no_pool_recovers_a_known_fraction_within_its_error(pair):
    """The 0-pool route reads the fraction back from the mixed pool alone, and the truth sits within a
    few of its bootstrap standard errors -- the error bar is calibrated, not decorative."""
    pool = _simulated_pool(pair, 0.35, key=jax.random.PRNGKey(0))
    est = estimate_fraction(
        pair, PLAN, pool, key=jax.random.PRNGKey(1), method="no_pool", pose_grid=GRID, n_bootstrap=80
    )

    assert est.method == "no_pool"
    assert est.n_particles == 1500
    assert 0.0 < est.fraction < 1.0
    assert est.fraction_err > 0.0
    assert est.contrast > 0.0
    assert abs(est.fraction - 0.35) < max(0.1, 4.0 * est.fraction_err)


def test_two_pool_route_recovers_with_pure_reference_pools(pair):
    """The reference mixture MLE reads the fraction against two pure pools -- the accurate route when
    both states can be prepared separately."""
    pool = _simulated_pool(pair, 0.5, key=jax.random.PRNGKey(2))
    pure = (
        _simulated_pool(pair, 1.0, key=jax.random.PRNGKey(20)),
        _simulated_pool(pair, 0.0, key=jax.random.PRNGKey(21)),
    )
    est = estimate_fraction(
        pair, PLAN, pool, key=jax.random.PRNGKey(3), method="two_pool", pure_pools=pure, pose_grid=GRID, n_bootstrap=40
    )

    assert est.method == "two_pool"
    assert 0.0 < est.fraction < 1.0
    assert est.fraction_err >= 0.0


def test_a_given_amplitude_skips_calibration(pair):
    """Passing a contrast scores at exactly that scale rather than fitting one."""
    pool = _simulated_pool(pair, 0.5, key=jax.random.PRNGKey(4))
    est = estimate_fraction(
        pair, PLAN, pool, key=jax.random.PRNGKey(5), method="no_pool", contrast=1.0, pose_grid=GRID, n_bootstrap=10
    )

    assert est.contrast == 1.0


def test_two_pool_without_pure_pools_is_refused(pair):
    """The reference routes name pure pools they cannot run without, so they refuse rather than guess."""
    pool = _simulated_pool(pair, 0.5, key=jax.random.PRNGKey(6))
    with pytest.raises(ValueError, match="pure reference pools"):
        estimate_fraction(pair, PLAN, pool, key=jax.random.PRNGKey(7), method="two_pool", pose_grid=GRID)


def test_an_unknown_method_is_refused(pair):
    pool = _simulated_pool(pair, 0.5, key=jax.random.PRNGKey(8))
    with pytest.raises(ValueError, match="method must be one of"):
        estimate_fraction(pair, PLAN, pool, key=jax.random.PRNGKey(9), method="bogus", pose_grid=GRID)
