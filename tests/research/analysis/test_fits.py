"""`statistics.fits`: the straight-line fit and the log-log power-law wrapper the panel drivers read."""

import numpy as np
import pytest

from reproduction.analysis.fits import fit_power_law, line_fit


def test_a_planted_exponent_comes_back_with_a_credible_error_bar():
    """The estimator end. Noise is added in the LOG, which is where the fit lives -- adding it
    linearly would make the scatter grow with `y` and the error bar would be reporting that."""
    rng = np.random.default_rng(0)
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    truth = -0.65
    y = np.exp(np.log(3.0) + truth * np.log(x) + rng.normal(0.0, 0.02, x.size))

    fit = fit_power_law(x, y)

    assert fit["n"] == x.size
    assert fit["exponent"] == pytest.approx(truth, abs=3.0 * fit["error"])
    assert 0.0 < fit["error"] < 0.05
    assert fit["r"] < -0.99
    assert fit["intercept"] == pytest.approx(np.log(3.0), abs=0.1)


def test_an_exact_power_law_is_recovered_exactly():
    """No noise, so the residual is zero and the exponent is the plant to machine precision. The
    error is then zero rather than a nan, which is `line_fit`'s stated behavior at zero residual."""
    x = np.array([1.0, 2.0, 4.0, 8.0])
    fit = fit_power_law(x, 5.0 * x**1.5)

    assert fit["exponent"] == pytest.approx(1.5, rel=1e-12)
    assert fit["error"] == pytest.approx(0.0, abs=1e-12)


def test_non_positive_rows_are_dropped_rather_than_logged():
    """A log of zero is not a number, and keeping one would poison the fit rather than fail it. The
    count comes back so a caller can see how much of its data survived."""
    x = [1.0, 2.0, 0.0, 4.0, 8.0, -1.0]
    y = [1.0, 4.0, 9.0, 16.0, 64.0, 1.0]

    fit = fit_power_law(x, y)

    assert fit["n"] == 4, "the zero abscissa and the negative one must both go"
    assert np.isfinite(fit["exponent"])


def test_a_fit_with_no_residual_to_read_or_no_lever_arm_returns_only_a_count():
    """Two refusals, both silent failures if they were fits instead.

    Two points determine a line exactly and leave nothing to estimate a scatter from, so an error
    bar off them would be zero for the wrong reason. A constant abscissa gives a slope that is a
    division by zero -- which is exactly what a consecutive-step reading of an ordered ladder takes,
    so this is reached in practice and not only in principle.
    """
    assert fit_power_law([1.0, 2.0], [1.0, 4.0]) == {"n": 2}
    assert fit_power_law([3.0, 3.0, 3.0, 3.0], [1.0, 2.0, 3.0, 4.0]) == {"n": 4}


def test_the_power_law_is_the_line_fit_taken_in_logs_and_not_a_second_estimator():
    """The reason this lives beside `line_fit` rather than in a driver: the two must agree by
    construction, so a change to how the degrees of freedom are counted reaches both."""
    x = np.array([1.0, 2.0, 5.0, 11.0, 20.0])
    y = np.array([2.0, 3.1, 6.2, 11.5, 19.0])

    fit = fit_power_law(x, y)
    (intercept, slope), covariance, _ = line_fit(np.log(x), np.log(y))

    assert fit["exponent"] == pytest.approx(float(slope), rel=1e-12)
    assert fit["intercept"] == pytest.approx(float(intercept), rel=1e-12)
    assert fit["error"] == pytest.approx(float(np.sqrt(covariance[1, 1])), rel=1e-12)
