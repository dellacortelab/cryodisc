"""Checks on the three fraction estimators and the purity line they feed.

The estimators are checked against pools drawn at a known fraction from the equal-covariance model,
indexed by how many pure pools they use (0, 1, 2); the line against anchors planted by construction.
"""

import numpy as np
import pytest
from jax import numpy as jnp

from cryodisc.statistics.composition import (
    fraction_classifier,
    fraction_no_pool,
    fraction_one_pool,
    fraction_two_pool,
)
from reproduction.analysis.purity import Purity, fit_purity


def mixed_pool(rng, fraction, squared_distance=6.0, n=40_000):
    """Lambda for a pool that is `fraction` state A, under `Lambda|A ~ N(+d^2/2, d^2)`."""
    in_a = rng.random(n) < fraction
    center = np.where(in_a, 0.5 * squared_distance, -0.5 * squared_distance)
    return jnp.asarray(center + np.sqrt(squared_distance) * rng.normal(size=n))


def overdispersed_pool(rng, fraction, gap=4.0, variance=12.0, n=200_000):
    """A mixture whose within-component variance far exceeds the honest `gap` (coherence gap/var < 1),
    with a pure state-B REFERENCE pool of that within variance and no between spread. `(mixture, ref)`.
    """
    in_a = rng.random(n) < fraction
    mixture = np.where(in_a, 0.5 * gap, -0.5 * gap) + np.sqrt(variance) * rng.normal(size=n)
    reference = -0.5 * gap + np.sqrt(variance) * rng.normal(size=n)
    return mixture, reference


# ---------------------------------------------------------------- 0 pools: the bare fixed point


@pytest.mark.parametrize("fraction", [0.1, 0.3, 0.5, 0.75, 0.9])
def test_no_pool_returns_the_fraction_from_a_pool_drawn_at_it(fraction):
    """No labels and no reference: this is the estimator a 3D classification runs on a calibrated score."""
    pool = mixed_pool(np.random.default_rng(0), fraction)
    assert fraction_no_pool(pool) == pytest.approx(fraction, abs=0.02)


def test_no_pool_does_not_drive_the_logit_to_infinity_on_a_split_pool():
    """The clip is load-bearing: at a large separation the estimate reaches an endpoint and the next
    iterate is a logit of zero, so an unclipped fixed point returns NaN on the easiest possible data."""
    pool = jnp.asarray(np.concatenate([np.full(500, 60.0), np.full(500, -60.0)]))
    recovered = fraction_no_pool(pool)
    assert np.isfinite(recovered)
    assert recovered == pytest.approx(0.5, abs=1e-3)


def test_no_pool_is_solved_rather_than_iterated_toward_from_one_half():
    """A fixed iteration count compresses toward the 0.5 it starts from, worst where the separation
    is weakest, so this asserts the ANSWER no longer depends on `d^2`."""
    for squared_distance in (6.0, 0.03, 0.01):
        pool = mixed_pool(np.random.default_rng(0), 0.3, squared_distance, n=200_000)
        assert fraction_no_pool(pool) == pytest.approx(0.3, abs=0.01), f"at d^2 = {squared_distance}"


def test_no_pool_is_compressed_under_overdispersion():
    """The reason the other two estimators exist: at coherence below one the bare fixed point is biased
    toward 0.5 by an amount that does not shrink with the pool."""
    mixture, _reference = overdispersed_pool(np.random.default_rng(1), 0.8)
    assert abs(float(fraction_no_pool(jnp.asarray(mixture))) - 0.8) > 0.05


def test_no_pool_refuses_an_empty_pool():
    with pytest.raises(ValueError, match="empty"):
        fraction_no_pool(jnp.zeros(0))


# ------------------------------------------------------ 1 pool: one density mirrored about the boundary


@pytest.mark.parametrize("fraction", [0.1, 0.3, 0.5, 0.75, 0.9])
def test_one_pool_recovers_the_fraction_the_bare_fixed_point_compresses(fraction):
    """One pure pool's density, reflected about the log-ratio boundary, stands in for the unseen state,
    and the reference-mixture MLE reads the fraction off it -- no second pool."""
    mixture, reference = overdispersed_pool(np.random.default_rng(0), fraction)
    assert fraction_one_pool(mixture, reference) == pytest.approx(fraction, abs=0.03)


def test_one_pool_beats_the_bare_fixed_point_under_overdispersion():
    """With one pool's measured density the reflection estimate lands closer to the truth than the
    reference-free fixed point on the same overdispersed score."""
    mixture, reference = overdispersed_pool(np.random.default_rng(1), 0.8)
    mirrored = fraction_one_pool(mixture, reference)
    bare = float(fraction_no_pool(jnp.asarray(mixture)))
    assert abs(mirrored - 0.8) < abs(bare - 0.8)


def test_one_pool_reads_a_pure_pool_at_the_endpoint():
    """A pure pool is read near its endpoint, not at an interior value, and never NaN, whichever pure
    pool supplies the reference."""
    rng = np.random.default_rng(2)
    pure = 2.0 + np.sqrt(12.0) * rng.normal(size=50_000)
    reference = -2.0 + np.sqrt(12.0) * rng.normal(size=50_000)
    recovered = fraction_one_pool(pure, reference)
    assert np.isfinite(recovered)
    assert recovered > 0.99


def test_one_pool_refuses_an_empty_pool():
    with pytest.raises(ValueError, match="empty"):
        fraction_one_pool(np.zeros(0), np.ones(10))


# ---------------------------------------------------------------- 2 pools: the reference mixture MLE


def pure_pools(rng, squared_distance=6.0, n=40_000, overdispersion=1.0):
    """Lambda for a pure state-A and a pure state-B pool under the equal-covariance model.

    `overdispersion` scales the honest score so `variance / gap = overdispersion`, the miscalibration
    a real pipeline shows -- an honest ratio has the two equal.
    """
    spread = np.sqrt(squared_distance)
    pool_a = 0.5 * squared_distance + spread * rng.normal(size=n)
    pool_b = -0.5 * squared_distance + spread * rng.normal(size=n)
    return overdispersion * pool_a, overdispersion * pool_b


def _mixture(rng, pool_a, pool_b, fraction, n=8_000):
    n_a = round(fraction * n)
    return np.concatenate([rng.choice(pool_a, n_a), rng.choice(pool_b, n - n_a)])


@pytest.mark.parametrize("fraction", [0.2, 0.5, 0.8])
def test_two_pool_recovers_a_known_fraction(fraction):
    """The reference mixture MLE reads the fraction off the two pure pools' empirical densities."""
    rng = np.random.default_rng(0)
    pool_a, pool_b = pure_pools(rng)
    mixture = _mixture(rng, pool_a, pool_b, fraction)
    assert fraction_two_pool(mixture, pool_a, pool_b) == pytest.approx(fraction, abs=0.03)


def test_two_pool_is_scale_invariant_where_the_fixed_point_is_biased():
    """The point of the reference route: an overdispersed score biases the bare fixed point but leaves
    the density MLE unmoved, since it reads against the pools' own densities."""
    rng = np.random.default_rng(1)
    pool_a, pool_b = pure_pools(rng, overdispersion=4.0)
    mixture = _mixture(rng, pool_a, pool_b, 0.25)
    assert fraction_two_pool(mixture, pool_a, pool_b) == pytest.approx(0.25, abs=0.03)
    assert abs(float(fraction_no_pool(jnp.asarray(mixture))) - 0.25) > 0.025


def test_two_pool_recovers_a_fraction_when_the_pools_share_a_mean():
    """Equal means, different variance: the fraction is still identifiable from the SHAPE of the two
    densities, which the reference MLE reads without a location gap."""
    rng = np.random.default_rng(4)
    pool_a = rng.normal(0.0, 1.0, size=40_000)  # same mean as b, narrower
    pool_b = rng.normal(0.0, 3.0, size=40_000)
    mixture = _mixture(rng, pool_a, pool_b, 0.6)
    assert fraction_two_pool(mixture, pool_a, pool_b) == pytest.approx(0.6, abs=0.06)


def test_two_pool_refuses_coincident_pools():
    """When the two pools' densities coincide there is no fraction to read, and it is refused."""
    rng = np.random.default_rng(3)
    same = rng.normal(size=10_000)
    with pytest.raises(ValueError, match="coincide"):
        fraction_two_pool(rng.normal(size=1_000), same, same.copy())


def test_two_pool_refuses_an_empty_pool():
    with pytest.raises(ValueError, match="empty"):
        fraction_two_pool(np.zeros(0), np.ones(10), np.ones(10))


# ---------------------------------------------------------- classify-and-count: the counting baseline


@pytest.mark.parametrize("fraction", [0.2, 0.5, 0.8])
def test_classifier_recovers_a_known_fraction_when_states_separate(fraction):
    """Well-separated states leave few ambiguous particles, so hard-labeling and counting reads the fraction."""
    rng = np.random.default_rng(0)
    pool_a, pool_b = pure_pools(rng, squared_distance=40.0)
    mixture = _mixture(rng, pool_a, pool_b, fraction)
    assert fraction_classifier(mixture, pool_a, pool_b) == pytest.approx(fraction, abs=0.03)


def test_classifier_is_compressed_toward_an_even_mixture_on_overlapping_states():
    """Overlapping states make counting the hard labels biased toward 0.5, the defect the two-pool MLE avoids."""
    rng = np.random.default_rng(2)
    pool_a, pool_b = pure_pools(rng, squared_distance=2.0)
    mixture = _mixture(rng, pool_a, pool_b, 0.8)
    assert fraction_classifier(mixture, pool_a, pool_b) < 0.75  # compressed toward an even mixture
    assert fraction_two_pool(mixture, pool_a, pool_b) == pytest.approx(0.8, abs=0.05)


def test_classifier_refuses_an_empty_pool():
    with pytest.raises(ValueError, match="empty"):
        fraction_classifier(np.ones(10), np.zeros(0), np.ones(10))


# ---------------------------------------------------------------- the purity line


def test_a_perfect_forward_model_puts_the_anchors_at_one_and_zero():
    known = np.linspace(0.0, 1.0, 5)
    fit = fit_purity(known, known)

    assert fit.f_a == pytest.approx(1.0)
    assert fit.f_b == pytest.approx(0.0)
    assert fit.separation == pytest.approx(1.0)
    assert fit.midpoint == pytest.approx(0.5)
    assert fit.residual_rms == pytest.approx(0.0, abs=1e-12)


def test_a_compression_and_an_offset_are_separated_rather_than_pooled():
    """The two departures enter the anchors with the same and with opposite signs, so a readout
    that reported only `|p_hat - p_true|` would call these two very different findings equal."""
    known = np.linspace(0.0, 1.0, 5)
    compressed = fit_purity(known, 0.5 + 0.6 * (known - 0.5))  # separation 0.6, midpoint 0.5
    shifted = fit_purity(known, known + 0.15)  # separation 1.0, midpoint 0.65

    assert compressed.separation == pytest.approx(0.6)
    assert compressed.midpoint == pytest.approx(0.5)
    assert shifted.separation == pytest.approx(1.0)
    assert shifted.midpoint == pytest.approx(0.65)


def test_the_anchors_are_recovered_end_to_end_from_pools_of_known_composition():
    """The estimator and the line together, which is the path a known-fraction benchmark takes."""
    rng = np.random.default_rng(1)
    known = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    recovered = [fraction_no_pool(mixed_pool(rng, p)) for p in known]
    fit = fit_purity(known, recovered)

    assert fit.f_a == pytest.approx(1.0, abs=0.03)
    assert fit.f_b == pytest.approx(0.0, abs=0.03)
    assert fit.residual_rms < 0.02


def test_a_shape_departure_splits_between_the_anchors_and_the_residual():
    """A subpopulation that is inert or contaminating changes the SHAPE of the sweep, and a least
    squares fit does not leave that outside the line: it absorbs whatever part of the departure
    projects onto `{1, p}` and reports the rest as residual. A clean line is not evidence of a clean pool.
    """
    known = np.linspace(0.0, 1.0, 7)
    design = np.stack([np.ones_like(known), known], 1)
    bump = 0.1 * np.sin(np.pi * known)
    absorbed = design @ np.linalg.lstsq(design, bump, rcond=None)[0]

    orthogonal_only = fit_purity(known, known + bump - absorbed)
    assert orthogonal_only.f_a == pytest.approx(1.0, abs=1e-9)
    assert orthogonal_only.f_b == pytest.approx(0.0, abs=1e-9)
    assert orthogonal_only.residual_rms > 0.01

    absorbed_only = fit_purity(known, known + absorbed)
    assert absorbed_only.residual_rms == pytest.approx(0.0, abs=1e-9)
    assert absorbed_only.midpoint > 0.5  # a clean line, and the axis has moved anyway


def test_the_error_bars_come_from_the_residuals_and_carry_the_covariance():
    """`f_a` is `intercept + slope`, whose variance is not the sum of theirs -- they anticorrelate,
    and ignoring the cross term would quote an anchor as worse-determined than it is."""
    rng = np.random.default_rng(2)
    known = np.linspace(0.0, 1.0, 9)
    fit = fit_purity(known, known + rng.normal(0.0, 0.01, known.size))

    assert 0.0 < fit.f_a_err < 0.03
    assert 0.0 < fit.f_b_err < 0.03
    assert abs(fit.f_a - 1.0) < 3.0 * fit.f_a_err


def test_a_sweep_that_determines_no_line_is_refused():
    with pytest.raises(ValueError, match="at least three mixtures"):
        fit_purity([0.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="determines no line"):
        fit_purity([0.5, 0.5, 0.5], [0.4, 0.5, 0.6])
    with pytest.raises(ValueError, match="must agree"):
        fit_purity([0.0, 0.5, 1.0], [0.0, 1.0])


def test_the_anchors_are_derived_and_not_stored_twice():
    """`separation` and `midpoint` are views of the same two numbers, so no run can report a set
    of four that disagree with each other."""
    fit = Purity(
        f_a=0.9, f_b=0.2, f_a_err=0.0, f_b_err=0.0, separation_err=0.0, midpoint_err=0.0,
        n_points=3, residual_rms=0.0,
    )  # fmt: skip

    assert fit.separation == pytest.approx(0.7)
    assert fit.midpoint == pytest.approx(0.55)


def test_the_separation_and_midpoint_carry_error_bars_from_the_same_fit():
    """`separation` is the slope and `midpoint` is `intercept + slope/2`, so both errors come off the
    covariance `line_fit` already returns."""
    rng = np.random.default_rng(0)
    known = np.linspace(0.05, 0.95, 9)
    recovered = 0.15 + 0.70 * known + rng.normal(0.0, 0.02, known.size)
    fit = fit_purity(known, recovered)

    assert fit.separation_err > 0.0 and fit.midpoint_err > 0.0
    assert (1.0 - fit.separation) / fit.separation_err > 5.0
    assert fit.separation_err != pytest.approx(fit.f_a_err + fit.f_b_err)


def test_the_comparison_grid_spans_the_mixture_not_just_the_pure_pools():
    """`_reference_grid` (kept for the SI estimators) must cover a mixture particle beyond the pure
    supports, or that mass is truncated."""
    from cryodisc.statistics.composition import _reference_grid

    grid = _reference_grid(np.array([0.0, 1.0]), np.array([2.0, 3.0]), np.array([-5.0, 8.0]), nodes=50)
    assert grid.min() == pytest.approx(-5.0)
    assert grid.max() == pytest.approx(8.0)
