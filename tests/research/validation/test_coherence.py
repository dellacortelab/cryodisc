"""Checks on the label-free calibration readout.

The identity has a closed form under the model -- one where every particle shares a `d^2`, and
`1/(1 + Var/(4E))` where they do not -- so it is checked against that rather than against another
statistic, and against the currencies it is supposed to catch things they cannot.
"""

import numpy as np
import pytest
from conftest import DEFLECTIONS, gaussian_pools
from jax import numpy as jnp

from cryodisc.statistics.information import deflection
from reproduction.validation.coherence import (
    coherence,
    coherence_pair,
    fractional_error,
    predicted_coherence,
)

POOL = 200_000
"""Particles per pool. A coherence is a mean over a VARIANCE, whose sampling error falls more slowly
than a mean's, so this size is what the abs=0.02 tolerances asserted below buy.
"""


@pytest.mark.parametrize("squared_distance", DEFLECTIONS)
def test_a_correctly_scored_pure_pool_reads_one_at_every_separation(squared_distance):
    """`d^2` cancels out of the ratio, so this is the same one over a 33x range in separation."""
    from_a, from_b = gaussian_pools(np.random.default_rng(0), squared_distance, POOL)
    read_a, read_b = coherence_pair(from_a, from_b)

    assert read_a.value == pytest.approx(1.0, abs=0.02)
    assert read_b.value == pytest.approx(1.0, abs=0.02)
    assert read_a.points_at_its_own_bank and read_b.points_at_its_own_bank


def test_a_scale_error_is_caught_here_and_by_nothing_else():
    """The reason this module exists: halve Lambda and the discriminability reports nothing amiss.

    A mis-set amplitude, a wrong whitening or a doubled noise spectrum all reach the score as an
    affine map, and `d^2` is invariant to one by construction -- a halved score reads the same
    separation, carrying no sign of being wrong. Only the coherence reads the scale itself. It reads
    `1/s` and not `s`: the mean goes as `s` and the variance as `s^2`, so a score that is too SMALL
    reads ABOVE one, which is why the readout is not clipped there.
    """
    from_a, from_b = gaussian_pools(np.random.default_rng(1), 4.0, POOL)
    half_a, half_b = 0.5 * from_a, 0.5 * from_b

    assert float(deflection(half_a, half_b)) == pytest.approx(float(deflection(from_a, from_b)), rel=1e-6)
    assert coherence(from_a).value == pytest.approx(1.0, abs=0.02)
    assert coherence(half_a).value == pytest.approx(2.0, abs=0.05)
    assert coherence(2.0 * from_a).value == pytest.approx(0.5, abs=0.02)


def test_decoherence_grows_with_the_noise_a_wrong_template_adds():
    """Anything uncorrelated added to Lambda inflates the variance and not the mean, which is the
    single direction every decoherence pushes -- template error, an unfound pose, an impure pool."""
    rng = np.random.default_rng(2)
    from_a, _ = gaussian_pools(rng, 4.0, POOL)
    readings = [coherence(from_a + jnp.asarray(rng.normal(0.0, extra, from_a.shape))) for extra in (0.0, 2.0, 4.0)]

    assert [reading.decoherence for reading in readings] == sorted(reading.decoherence for reading in readings)
    assert readings[0].decoherence == pytest.approx(0.0, abs=0.02)
    assert readings[-1].decoherence > 0.5


def test_a_pool_whose_mean_points_at_the_other_bank_is_reported_rather_than_absorbed():
    """A swapped label survives `value` as a sign, because the ratio alone would read a clean 1.0."""
    from_a, from_b = gaussian_pools(np.random.default_rng(3), 4.0, POOL)
    swapped, _ = coherence_pair(from_b, from_a)

    assert not swapped.points_at_its_own_bank
    assert swapped.value == pytest.approx(-1.0, abs=0.02)


def test_a_pool_too_thin_for_a_variance_is_refused():
    with pytest.raises(ValueError, match="at least two particles"):
        coherence(jnp.array([1.0]))


def heterogeneous_pool(rng, squared_distances):
    """One Lambda per particle, each from its OWN `d_i^2`: `Lambda_i | A ~ N(+d_i^2/2, d_i^2)`.

    `conftest.gaussian_pools` draws a whole pool from one `d^2`, the DEGENERATE case. A real pool is
    this: `d_i^2` depends on the viewing direction and the defocus, so it varies particle to particle
    even when the model is exactly right.
    """
    squared_distances = np.asarray(squared_distances, float)
    return jnp.asarray(rng.normal(+0.5 * squared_distances, np.sqrt(squared_distances)))


@pytest.mark.parametrize("spread", [0.2, 0.5, 1.0])
def test_a_heterogeneous_pool_reads_its_predicted_baseline_and_not_one(spread):
    """The regime every real pool is in.

    A correctly scored pool whose `d_i^2` varies reads `1/(1 + Var/(4E))`, strictly below one, with no
    defect present. Against the closed form rather than a stored number, and at three spreads so a
    baseline ignoring `Var[d^2]` could not pass by fitting one of them.
    """
    rng = np.random.default_rng(7)
    d2 = rng.lognormal(np.log(4.0), spread, POOL)
    read = coherence(heterogeneous_pool(rng, d2), baseline=predicted_coherence(jnp.asarray(d2)))

    assert read.baseline == pytest.approx(predicted_coherence(jnp.asarray(d2)), rel=1e-12)
    assert read.baseline < 1.0
    # the reading lands on the baseline, so the pool is CORRECTLY scored ...
    assert read.value == pytest.approx(read.baseline, rel=0.05)
    # ... and `decoherence` says so, where measuring against one would have charged it the whole deficit
    assert read.decoherence == pytest.approx(0.0, abs=0.05 * read.baseline)
    assert 1.0 - read.value > 0.5 * (1.0 - read.baseline)


def test_reading_above_one_is_a_scale_error_and_nothing_else_can_do_it():
    """Heterogeneity only pushes the baseline DOWN, so a reading above one is a miscalibration whatever
    the baseline turns out to be -- checkable before the baseline is known, and so the one departure
    worth refusing on.
    """
    from_a, _ = gaussian_pools(np.random.default_rng(8), 4.0, POOL)

    for scale in (1.0 / 5.6, 0.5):
        read = coherence(scale * from_a)
        assert read.value == pytest.approx(1.0 / scale, rel=0.05)
        assert read.scale_error == pytest.approx(scale, rel=0.05)
        assert read.indicates_scale_error()

    # a correctly scored pool does not trip it, nor one that merely decohered ...
    rng = np.random.default_rng(9)
    assert not coherence(from_a).indicates_scale_error()
    assert not coherence(from_a + jnp.asarray(rng.normal(0.0, 4.0, from_a.shape))).indicates_scale_error()

    # ... and nor does a genuinely HETEROGENEOUS pool, which is the case the guard must not fire on:
    # its baseline is far below one for no bad reason, and the check is against one precisely so that
    # such a pool passes without its baseline having to be known first
    d2 = rng.lognormal(np.log(4.0), 1.23, POOL)
    spread = coherence(heterogeneous_pool(rng, d2), baseline=predicted_coherence(jnp.asarray(d2)))
    assert spread.baseline < 0.3
    assert not spread.indicates_scale_error()

    # ... and nor does a pool too SMALL to have measured anything, whatever it happens to read. A
    # variance over n points is itself uncertain by sqrt(2/(n-1)), which at six particles is 63%
    tiny = coherence(0.2 * from_a[:6])
    assert abs(tiny.value) > 1.25
    assert not tiny.indicates_scale_error()
    assert tiny.indicates_scale_error(min_particles=2)


def test_a_baseline_outside_its_own_range_is_refused():
    """`1/(1 + Var/4E)` cannot exceed one or reach zero, so a caller passing either has computed
    something else and would get a `decoherence` measured against it."""
    from_a, _ = gaussian_pools(np.random.default_rng(10), 4.0, 1000)

    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match=r"lies in \(0, 1\]"):
            coherence(from_a, baseline=bad)


def test_predicted_coherence_is_one_only_where_every_particle_shares_a_d2():
    """The degenerate case is the boundary of the range, approached from below as the spread shrinks."""
    assert predicted_coherence(jnp.full((500,), 7.0)) == pytest.approx(1.0, rel=1e-12)

    rng = np.random.default_rng(11)
    shrinking = [predicted_coherence(jnp.asarray(rng.normal(7.0, s, 200_000))) for s in (2.0, 1.0, 0.5)]
    assert shrinking == sorted(shrinking)
    assert shrinking[-1] < 1.0

    with pytest.raises(ValueError, match="positive mean"):
        predicted_coherence(jnp.zeros((10,)))


def test_the_predicted_baseline_moves_with_the_amplitude_it_was_computed_at():
    """The property that makes `decoherence` only as good as the calibration under it.

    `d^2` goes as `a^2`, so a mis-fitted amplitude rescales every `d_i^2`, and the baseline is
    `mean/(mean + Var/4)` -- not homogeneous in that scale. Pinned rather than merely documented,
    because a future `predicted_coherence` that normalized the spread would look tidier, would read as
    scale-free, and would silently stop tracking the quantity `score_pool` brackets against.
    """
    rng = np.random.default_rng(12)
    d2 = jnp.asarray(rng.lognormal(np.log(250.0), 0.5, 4000))

    at_fitted = predicted_coherence(d2)
    # the amplitude measured 13.5% low on a real pool, which scales every d^2 by 1.135**2
    at_truth = predicted_coherence(d2 * 1.135**2)

    assert at_truth < at_fitted
    assert abs(at_truth / at_fitted - 1.0) > 0.10
    # and the direction: a LARGER d^2 spread relative to its mean reads as a lower baseline
    assert predicted_coherence(d2 * 4.0) < predicted_coherence(d2)


def planted_pool(rng, squared_distances, *, scale, spread):
    """`heterogeneous_pool` with a known FRACTIONAL error on each particle's `d_i^2`.

    The mean of `Lambda_i` moves to `(d_i^2 / 2)(1 + delta_i)` -- the templates being wrong about that
    particle -- while the VARIANCE stays `d_i^2`, because the noise projects onto the rendered template
    difference and not onto the truth. That asymmetry is the whole reason a fractional error reads as
    decoherence rather than as a scale error.

    Returns:
        `(log_ratio, d2, delta)`, where `d2` is the closed form an estimator is handed and `delta` the
        planted error it has to recover without seeing it.
    """
    squared_distances = np.asarray(squared_distances, float)
    delta = rng.normal(scale, spread, squared_distances.shape)
    mean = 0.5 * squared_distances * (1.0 + delta)
    return jnp.asarray(rng.normal(mean, np.sqrt(squared_distances))), jnp.asarray(squared_distances), delta


@pytest.mark.parametrize("squared_distance", [30.0, 300.0])
def test_the_fractional_error_recovers_a_planted_one_at_the_same_size_over_a_10x_in_d2(squared_distance):
    """The claim that makes `fractional_error_sd` the comparable form.

    A fraction is dimensionless, so the SAME planted error must come back at both `d^2` -- which is
    what says the CryoBench excess going as `E[d^2]^2` is one fractional error and not two coincidences.
    An additive variance would read `sd / sqrt(10)` smaller at the ten-times-larger separation.
    """
    rng = np.random.default_rng(11)
    d2 = rng.lognormal(np.log(squared_distance), 0.5, POOL)
    log_ratio, closed_form, delta = planted_pool(rng, d2, scale=0.025, spread=0.175)

    scale, spread = fractional_error(log_ratio, closed_form)

    assert scale == pytest.approx(float(delta.mean()), abs=0.005)
    assert spread == pytest.approx(float(delta.std()), rel=0.02)


def test_the_score_s_own_noise_is_subtracted_and_not_reported_as_a_fractional_error():
    """The correction is load-bearing, not cosmetic.

    `Var[Lambda_i] = d_i^2` makes `2 Lambda_i / d_i^2` scatter by `2 / d_i` with the templates exactly
    right, which at `d^2` of 30 is 37% -- larger than the effect being measured. Uncorrected, a perfect
    pool would report a third of its `d^2` as model error.
    """
    rng = np.random.default_rng(12)
    d2 = rng.lognormal(np.log(30.0), 0.5, POOL)
    log_ratio, closed_form, _ = planted_pool(rng, d2, scale=0.0, spread=0.0)

    scale, spread = fractional_error(log_ratio, closed_form)
    naive = float(jnp.std(2.0 * log_ratio / closed_form - 1.0))

    assert scale == pytest.approx(0.0, abs=0.005)
    # nothing was planted, so the whole reading is sampling and the corrected number is zero ...
    assert spread == pytest.approx(0.0, abs=0.02)
    # ... where the uncorrected one would have charged the pool a large error it does not have
    assert naive > 0.3


def test_a_fractional_error_needs_one_positive_d2_per_particle():
    """A particle carrying no separation has no fractional error, and dividing by it would report one."""
    log_ratio = jnp.asarray([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="one d.2 per particle"):
        fractional_error(log_ratio, jnp.asarray([1.0, 2.0]))
    with pytest.raises(ValueError, match="must be positive"):
        fractional_error(log_ratio, jnp.asarray([1.0, 0.0, 3.0]))


def test_the_power_asymmetry_correction_recovers_an_amplitude_scale_error_reads_wrong():
    """`scale_error` is a LOWER bound when the two templates differ in power; this is the exact one.

    The construction makes the answer known. With `L = (P_A - P_B) / d^2` the correction is
    `r = 1 + (c - 1) / (1 + L)`, so a reading `c` at a chosen `L` has one right answer, and the
    equal-power reading `scale_error` gives is a different number whenever `L` is not zero.
    """
    rng = np.random.default_rng(0)
    scale = 1.3
    from_a, _ = gaussian_pools(rng, DEFLECTIONS[2], POOL)
    read = coherence(jnp.asarray(from_a) * scale)

    # equal power is the one case where the two agree, L being zero there
    assert read.use_power_asymmetry(4.0, 4.0, 10.0) == pytest.approx(1.0 + (abs(read.value) - 1.0))

    # and they part company as soon as the templates differ, by exactly the lever
    lever = 1.0 + (9.0 - 4.0) / 10.0
    assert read.use_power_asymmetry(9.0, 4.0, 10.0) == pytest.approx(1.0 + (abs(read.value) - 1.0) / lever)
    assert read.use_power_asymmetry(9.0, 4.0, 10.0) != pytest.approx(read.scale_error, rel=1e-6)


def test_the_power_asymmetry_refuses_a_d2_it_cannot_form_a_lever_from():
    """`L` is undefined without a positive `d^2`, and NaN where the lever vanishes.

    The singular case is not an edge to paper over: at `L = -1` the reading carries no information
    about the amplitude at all, so a number there would be invented.
    """
    read = coherence(jnp.asarray(gaussian_pools(np.random.default_rng(1), DEFLECTIONS[2], 4096)[0]))

    with pytest.raises(ValueError, match="d2 must be positive"):
        read.use_power_asymmetry(1.0, 1.0, 0.0)
    assert np.isnan(read.use_power_asymmetry(0.0, 10.0, 10.0))
