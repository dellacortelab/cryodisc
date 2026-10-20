"""Checks on the K -> infinity extrapolation of the marginal rung.

The fit returns a number for any input, so what is worth testing is that it REFUSES: two grids, a
negative intercept, a limit on the wrong side of its own points, a limit above the rung that bounds
it -- each a failed fit that would otherwise read as a rung.
"""

from functools import partial

import numpy as np
import pytest

from cryodisc.statistics.convergence import CROSSOVER_EXPONENT
from cryodisc.statistics.convergence import extrapolate_in_k as _extrapolate_in_k

extrapolate_in_k = partial(_extrapolate_in_k, exponent=2.0 / 3.0)
"""These test the fitting machinery at a fixed exponent, whatever the physics constant is; a test that
passes its own `exponent=` still overrides it."""

TRUE_LIMIT = 10.0
"""The intercept the synthetic points below are built from, so the expectation is DERIVED."""

TRUE_SLOPE = 5.0
"""`c` in `d^2(K) = limit - c K^(-2/3)`."""


def exact_ladder(counts):
    """The rung a perfect `K^(-2/3)` convergence would produce at each node count.

    Args:
        counts: Node counts.

    Returns:
        `TRUE_LIMIT - TRUE_SLOPE * K^(-2/3)`, which the fit must invert exactly.
    """
    return TRUE_LIMIT - TRUE_SLOPE * np.asarray(counts, float) ** (-2.0 / 3.0)


def test_a_perfect_k_to_the_minus_two_thirds_ladder_inverts_exactly():
    """Cube counts, so `K^(-2/3)` is 1/4, 1/16, 1/64 and the arithmetic is checkable by hand."""
    counts = [8, 64, 512]
    values = exact_ladder(counts)
    np.testing.assert_allclose(values, [8.75, 9.6875, 9.921875])

    fit = extrapolate_in_k(counts, values)
    assert fit.limit == pytest.approx(TRUE_LIMIT, rel=1e-9)
    assert fit.slope == pytest.approx(TRUE_SLOPE, rel=1e-9)
    assert fit.chi2_per_dof == pytest.approx(0.0, abs=1e-18)
    assert fit.counts == (8, 64, 512)
    fit.check(ceiling=TRUE_LIMIT + 1.0)


def test_the_points_are_sorted_before_they_are_fitted():
    """`check` reads `values[-1]` as the finest grid, so an unsorted input would compare the limit
    against whichever count happened to be last."""
    counts = [512, 8, 64]
    fit = extrapolate_in_k(counts, exact_ladder(counts))
    assert fit.counts == (8, 64, 512)
    assert fit.values[-1] == pytest.approx(9.921875)
    assert fit.limit == pytest.approx(TRUE_LIMIT, rel=1e-9)


def test_two_grids_are_refused_because_the_fit_could_not_fail():
    """Two points determine a line exactly: the residual is zero and the error would be reported as
    0 by construction rather than as something small."""
    with pytest.raises(ValueError, match="three distinct"):
        extrapolate_in_k([8, 64], [8.75, 9.6875])
    # three values but only two distinct counts is the same degeneracy wearing a third point
    with pytest.raises(ValueError, match="three distinct"):
        extrapolate_in_k([8, 64, 64], [8.75, 9.6875, 9.69])


def test_error_bars_widen_the_intercept_when_the_points_scatter():
    """A fit whose points sit worse on a line than their bars claim must not report the bars. The
    same points with a deliberate kink have to come out with a larger intercept error."""
    counts = [8, 64, 512, 4096]
    clean = exact_ladder(counts)
    errors = [0.1] * 4
    tight = extrapolate_in_k(counts, clean, errors)
    kinked = extrapolate_in_k(counts, clean + np.array([0.0, 0.6, -0.6, 0.0]), errors)
    assert kinked.chi2_per_dof > tight.chi2_per_dof
    assert kinked.limit_err > tight.limit_err


def test_a_zero_error_bar_cannot_pin_the_line_through_one_point():
    """A model-only rung carries error 0, which as `1/err^2` is infinite. The floor makes such a
    point dominant rather than absolute, so the fit stays finite."""
    counts = [8, 64, 512]
    fit = extrapolate_in_k(counts, exact_ladder(counts), [0.0, 0.2, 0.2])
    assert np.isfinite(fit.limit)
    assert np.isfinite(fit.limit_err)


def test_a_negative_intercept_is_not_a_squared_distance():
    """A ladder falling steeply enough extrapolates below zero, which is arithmetic, not a rung."""
    fit = extrapolate_in_k([8, 64, 512], [-1.0, -3.0, -3.5])
    with pytest.raises(ValueError, match="not a squared distance"):
        fit.check()


def test_a_limit_on_the_wrong_side_of_its_own_points_is_refused():
    """The direction is read from the data, a ladder that falls in K being real. What is never
    allowed is a limit BETWEEN the measured points, which no convergence produces."""
    rising = extrapolate_in_k([8, 64, 512], exact_ladder([8, 64, 512]))
    assert rising.slope > 0.0
    # the same fit judged against a fabricated finest point above the limit
    fabricated = type(rising)(
        limit=rising.limit,
        limit_err=rising.limit_err,
        slope=rising.slope,
        counts=rising.counts,
        values=(8.75, 9.6875, 11.0),
        chi2_per_dof=rising.chi2_per_dof,
    )
    with pytest.raises(ValueError, match="BELOW its finest grid"):
        fabricated.check()


def test_a_non_monotone_ladder_is_refused_however_well_a_line_fits_it():
    """A non-monotone ladder must be refused even when every other gate passes.

    Both side checks compare the limit against the FINEST point alone, so a ladder that falls and then
    rises satisfies whichever branch its fitted slope selects while straddling the rest. Then
    `limit - y` changes sign along the ladder, its logarithm is undefined, and the exponents come back
    NaN -- the one statistic that sees it. Here chi2/dof, span, and a positive limit all pass.
    """
    fit = extrapolate_in_k([16, 54, 1024], [0.3228, 0.0193, 0.0787], [0.1109, 0.0876, 0.0529])

    assert fit.chi2_per_dof < 4.0 and fit.span == 64.0 and fit.limit > 0.0
    assert not np.all(np.isfinite(fit.local_exponents))
    # and the limit really does sit between its own points, which is what the NaN encodes
    assert min(fit.values) < fit.limit < max(fit.values)

    with pytest.raises(ValueError, match="sits between the points it was fitted to"):
        fit.check(min_span=0.0)


def test_the_limit_may_not_exceed_the_rung_that_bounds_it():
    """Marginalizing a pose against its prior cannot beat scoring at the TRUE pose, so a limit above
    `known_pose` means one of the two is wrong.

    The bound must be `known_pose` and no bank-dependent rung, since a ceiling measured on one bank
    understates the true one and so is not a bound on a rung extrapolated to K -> infinity.
    """
    fit = extrapolate_in_k([8, 64, 512], exact_ladder([8, 64, 512]))
    with pytest.raises(ValueError, match="exceeds the known-pose rung"):
        fit.check(ceiling=TRUE_LIMIT - 1.0)


def test_the_inputs_are_checked_before_anything_is_fitted():
    """Each of these would otherwise produce a plausible number."""
    with pytest.raises(ValueError, match="must agree"):
        extrapolate_in_k([8, 64, 512], [1.0, 2.0])
    with pytest.raises(ValueError, match="must be positive"):
        extrapolate_in_k([8, 64, 0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="non-negative"):
        extrapolate_in_k([8, 64, 512], [1.0, 2.0, 3.0], [0.1, -0.1, 0.1])


def shallow_ladder(counts, exponent):
    """A ladder converging at a DIFFERENT rate than the module assumes.

    The regime a coarse bank on a large particle is measured in: the asymptotic `K^(-2/3)` has not
    been reached, so the deficit falls more slowly and the points curve away from a straight line in
    `K^(-2/3)`.
    """
    return TRUE_LIMIT - TRUE_SLOPE * np.asarray(counts, float) ** (-exponent)


WIDE = [8, 64, 512]
"""Counts spanning 64x, where a wrong exponent is unmissable."""

NARROW = [16, 54, 128]
"""Counts spanning 8x, where a wrong exponent is not identifiable."""


def test_a_wrong_exponent_over_a_wide_span_is_caught_by_the_goodness_of_fit():
    """`chi2_per_dof` is the test of whether the assumed law holds, so `check` has to refuse on it --
    otherwise an extrapolation of the wrong curve comes back as a rung."""
    values = shallow_ladder(WIDE, 0.47)
    fit = extrapolate_in_k(WIDE, values, [0.002] * 3)

    assert fit.weighted and fit.exponent == pytest.approx(2.0 / 3.0)
    assert fit.chi2_per_dof > 4.0
    with pytest.raises(ValueError, match=r"do not follow K\^\(-0\.6667\)"):
        fit.check(ceiling=TRUE_LIMIT + 1.0)


def test_the_measured_exponent_recovers_a_limit_the_assumed_one_misses():
    """The escape hatch the refusal has to come with: a caller who has measured the rate gets the
    right answer from the same three points that the default rejects."""
    values = shallow_ladder(WIDE, 0.47)

    assumed = extrapolate_in_k(WIDE, values, [0.002] * 3)
    measured = extrapolate_in_k(WIDE, values, [0.002] * 3, exponent=0.47)

    assert measured.limit == pytest.approx(TRUE_LIMIT, rel=1e-9)
    assert measured.chi2_per_dof == pytest.approx(0.0, abs=1e-12)
    measured.check(ceiling=TRUE_LIMIT + 1.0)
    assert abs(assumed.limit - TRUE_LIMIT) > 2.0 * assumed.limit_err


def test_closely_spaced_grids_are_refused_because_the_exponent_is_not_identifiable():
    """The failure no residual can see, and the reason `span` is checked before anything else: on a
    narrow span a wrong exponent biases the intercept while the fit still looks good."""
    values = shallow_ladder(NARROW, 0.55)
    fit = extrapolate_in_k(NARROW, values, [0.01] * 3)

    assert fit.span == pytest.approx(8.0)
    assert fit.chi2_per_dof < 4.0
    assert abs(fit.limit - TRUE_LIMIT) > 5.0 * fit.limit_err

    with pytest.raises(ValueError, match="NOT IDENTIFIABLE"):
        fit.check(ceiling=TRUE_LIMIT + 1.0)
    # and deliberately, having read why, it can still be quoted
    fit.check(ceiling=TRUE_LIMIT + 1.0, min_span=0.0)


def test_the_drift_is_reported_but_is_noisier_than_the_signal_where_it_would_be_needed():
    """Why `local_exponents` is a diagnostic and not a guard. On a wide span it is clean enough to read;
    on a narrow one the scatter it shows under the TRUE law exceeds what a false one produces."""
    flat = extrapolate_in_k(WIDE, exact_ladder(WIDE))
    np.testing.assert_allclose(flat.local_exponents, [2.0 / 3.0] * 2, rtol=1e-9)

    false_signal = np.ptp(extrapolate_in_k(NARROW, shallow_ladder(NARROW, 0.47), [0.05] * 3).local_exponents)

    rng = np.random.default_rng(0)
    exact = exact_ladder(NARROW)
    true_noise = []
    for _ in range(600):
        drifting = extrapolate_in_k(NARROW, exact + rng.normal(0.0, 0.05, 3), [0.05] * 3).local_exponents
        if np.all(np.isfinite(drifting)):
            true_noise.append(np.ptp(drifting))

    assert np.median(true_noise) > false_signal


def test_the_goodness_of_fit_refusal_needs_real_error_bars():
    """Unweighted, `chi2/dof` IS the variance estimate rather than a test of the law -- it has no scale
    to be judged against -- so that refusal stays off. The span check does not care either way."""
    fit = extrapolate_in_k(WIDE, shallow_ladder(WIDE, 0.47))

    assert not fit.weighted
    assert fit.limit_err > 0.0
    fit.check(ceiling=TRUE_LIMIT + 1.0)


def test_a_non_positive_exponent_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        extrapolate_in_k(WIDE, [1.0, 2.0, 3.0], exponent=0.0)


def test_a_ladder_that_falls_in_k_is_extrapolated_rather_than_refused():
    """A finer bank can move a particle's posterior off a node it landed on by luck, so a rung that
    FALLS in K is real -- and `check` allows it, refusing only when the limit sits on the wrong side
    of the finest grid."""
    falling = extrapolate_in_k([8, 64, 512], [10.0, 9.0, 8.75])

    assert falling.limit < min(falling.values)
    assert falling.slope < 0.0
    assert np.all(np.isfinite(falling.local_exponents))
    falling.check(min_span=0.0)


def test_a_ladder_that_straddles_its_own_limit_is_still_refused():
    """A non-monotone ladder makes the deficit change sign across its points, so the local exponents
    go NaN and the fit is refused -- the same signal that must keep catching this case."""
    straddling = extrapolate_in_k([8, 64, 512], [9.0, 11.0, 9.5])

    assert not np.all(np.isfinite(straddling.local_exponents))
    with pytest.raises(ValueError, match="sits between the points"):
        straddling.check(min_span=0.0)


# --- converge_marginal: the adaptive stepping loop (lifted from kc_survey into the library) ---

from cryodisc.statistics.convergence import (
    DEFAULT_LADDER,
    converge_caching,
    converge_marginal,
    top_octave_gain,
)


def _synthetic_scorer(limit, slope, exponent=CROSSOVER_EXPONENT):
    """A scorer whose d2(K) = limit - slope*K^(-p): a perfect converging ladder, no realization drift."""

    def score(k):
        return k, float(limit - slope * k ** (-exponent))

    return score


def test_converge_marginal_reaches_the_plateau():
    """A clean K^(-2/3) ladder converges and the extrapolated plateau recovers the true limit."""
    out = converge_marginal(_synthetic_scorer(TRUE_LIMIT, TRUE_SLOPE), ladder=DEFAULT_LADDER)
    assert out.status == "converged"
    assert out.extrapolation is not None
    assert out.plateau == pytest.approx(TRUE_LIMIT, rel=1e-3)
    # it must stop before exhausting the ladder once the top octave has flattened
    assert len(out.curve) < len(DEFAULT_LADDER)


def test_converge_marginal_censors_a_rising_ladder():
    """A ladder that keeps climbing (unbounded power, never flattens) is censored at the cap."""
    # d2 = K^0.35 gains ~21% per top octave (1 - 0.5**0.35), above RISE forever, so it never converges
    out = converge_marginal(lambda k: (k, float(k) ** 0.35), ladder=DEFAULT_LADDER)
    assert out.status == "censored"
    assert out.plateau == pytest.approx(max(out.curve.values()))
    assert len(out.curve) == len(DEFAULT_LADDER)  # it grew all the way to the cap


def test_converge_marginal_respects_the_cap():
    """No bank above the cap is ever scored."""
    seen = []

    def score(k):
        seen.append(k)
        return k, float(TRUE_LIMIT - TRUE_SLOPE * k ** (-CROSSOVER_EXPONENT))

    converge_marginal(score, ladder=DEFAULT_LADDER, cap=64128)
    assert max(seen) <= 64128


def test_top_octave_gain_flattens():
    """The gain falls toward zero as the curve flattens, and one point reads fully rising."""
    assert top_octave_gain({504: 5.0}) == 1.0
    flat = {504: 9.9, 2016: 9.99, 8064: 9.999}
    assert top_octave_gain(flat) < 0.15


def test_converge_caching_returns_the_artifact_at_the_converged_bank():
    """converge_caching converges the scalar AND hands back the finest bank's full artifact -- the
    per-bank object (Lambda arrays, in a real driver) the scalar Convergence cannot carry."""

    def simulate(k):
        return k, {"bank": k, "d2": float(TRUE_LIMIT - TRUE_SLOPE * k ** (-CROSSOVER_EXPONENT))}

    conv, artifact = converge_caching(simulate, lambda a: a["d2"], ladder=DEFAULT_LADDER)
    assert conv.status == "converged"
    assert conv.plateau == pytest.approx(TRUE_LIMIT, rel=1e-3)
    # the returned object is the one from the finest bank actually scored, matched to the curve
    assert artifact["bank"] == max(conv.curve)
    assert artifact["d2"] == pytest.approx(conv.curve[max(conv.curve)])


def test_converge_caching_keys_the_artifact_on_the_realized_count():
    """The artifact is cached by the REALIZED count simulate reports, not the requested one, so a
    quantizing bank still returns the artifact matching the curve's finest key."""

    def simulate(k):
        realized = k + 1  # stands in for pose_bank quantizing the request
        return realized, {"realized": realized}

    conv, artifact = converge_caching(simulate, lambda a: float(a["realized"]), ladder=[8, 64, 512], cap=512)
    assert artifact["realized"] == max(conv.curve)
