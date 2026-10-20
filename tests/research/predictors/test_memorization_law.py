"""The fit recovers a planted law, and refuses the two ways the ladder can be unfittable.

The driver's job is to say whether the exponent is -1, so a fit that quietly returns -1 whatever the
data would be worse than no fit at all. These plant a known amplitude and exponent and check both
come back, then check the refusals.
"""

import numpy as np
import pytest

from reproduction.experiments.benchmark.memorization_law import (
    fit_power_law,
    from_the_arms,
    independent_coefficients,
)
from reproduction.predictors.memorization import (
    _shell_cumulative,
    band_voxels,
    implied_band,
)


def test_a_planted_inverse_law_comes_back():
    """Amplitude 5000 at slope -1, read back through the weighted log fit."""
    n = np.array([2000.0, 4000.0, 8000.0, 16000.0, 32000.0])
    signal = 5000.0 / n
    sigma = signal * 0.01
    free = fit_power_law(n, signal, sigma)
    assert free["slope"] == pytest.approx(-1.0, abs=1e-6)
    assert free["amplitude"] == pytest.approx(5000.0, rel=1e-6)
    fixed = fit_power_law(n, signal, sigma, slope=-1.0)
    assert fixed["amplitude"] == pytest.approx(5000.0, rel=1e-6)


def test_a_different_exponent_is_not_forced_to_minus_one():
    """The free fit is the test, so a ladder that falls as N^-0.5 must report -0.5."""
    n = np.array([2000.0, 4000.0, 8000.0, 16000.0, 32000.0])
    signal = 300.0 * n**-0.5
    free = fit_power_law(n, signal, signal * 0.01)
    assert free["slope"] == pytest.approx(-0.5, abs=1e-6)


def test_the_slope_error_bar_widens_with_scatter():
    """A noisy ladder must not report a confident exponent."""
    n = np.array([2000.0, 4000.0, 8000.0, 16000.0, 32000.0])
    signal = 5000.0 / n
    tight = fit_power_law(n, signal, signal * 0.01)
    loose = fit_power_law(n, signal, signal * 0.30)
    assert loose["slope_sem"] > tight["slope_sem"] * 10


def test_a_sign_change_refuses_rather_than_fits():
    """A rung whose memorization went negative is a finding; a log fit would simply crash on it."""
    n = np.array([2000.0, 4000.0, 8000.0])
    signal = np.array([2.5, -0.2, 0.6])
    with pytest.raises(SystemExit, match="non-positive"):
        fit_power_law(n, signal, np.full(3, 0.05))


def test_two_rungs_cannot_support_a_free_slope():
    """Two points define a line exactly, so the exponent would carry no error bar at all."""
    n = np.array([2000.0, 4000.0])
    signal = 5000.0 / n
    with pytest.raises(SystemExit, match="three rungs"):
        fit_power_law(n, signal, signal * 0.01)
    # but the fixed-slope reading is still well posed
    assert fit_power_law(n, signal, signal * 0.01, slope=-1.0)["amplitude"] == pytest.approx(5000.0, rel=1e-6)


def test_the_geometric_count_is_the_band_limited_half_disc():
    """M_eff is quoted beside the fitted amplitude, so its own definition has to be pinned."""
    assert independent_coefficients(192, 0.125) == pytest.approx(np.pi * 24.0**2 / 2)
    # it scales as the square of the band, which is what makes the comparison informative
    assert independent_coefficients(192, 0.25) == pytest.approx(4 * independent_coefficients(192, 0.125))


def _rungs(**arm):
    """Two rungs whose four arms all carry the same recorded fields."""
    return [{"self": dict(arm), "null": dict(arm)} for _ in range(2)]


def test_an_amplitude_of_exactly_2v_implies_the_band_it_was_scored_to():
    """The zero point of the whole implied-against-scored comparison: an amplitude that is exactly
    `2V` at a band must imply that same band back. `band_voxels` steps at each shell, so a naive
    bisection lands on the step's lower edge and reports a band coarser than the one scored to.
    """
    for box in (64, 192, 256):
        exact = 2 * band_voxels(box, 0.125)
        assert implied_band(box, exact) == pytest.approx(0.125, rel=1e-12)


def test_the_implied_band_resolves_finer_than_one_shell():
    """Two amplitudes inside one shell must not report one band.

    A shell is 1/box of the band, against an amplitude known far more precisely, so rounding to the
    shell throws away most of the precision the measurement has.
    """
    low = implied_band(192, 2 * 61564.0)
    high = implied_band(192, 2 * 62700.0)
    assert high > low
    assert not (round(low * 192) < round(high * 192)), "the two sit in one shell, which is the point"


def test_an_amplitude_past_the_whole_grid_saturates_at_the_corner():
    """Rather than run off the end of the shell table, or return a band inside the box."""
    assert implied_band(32, 1e12) == pytest.approx((len(_shell_cumulative(32)) - 1) / 32)
    assert implied_band(192, 0.0) == 0.0


def test_the_arms_are_read_rather_than_the_flag():
    assert from_the_arms(_rungs(box_size=256), "box_size", None, "--box-size") == 256


def test_arms_that_disagree_refuse_rather_than_average():
    rows = [{"self": {"box_size": 192}, "null": {"box_size": 256}}]
    with pytest.raises(SystemExit, match="one ladder cannot carry two"):
        from_the_arms(rows, "box_size", None, "--box-size")


def test_a_flag_contradicting_the_arms_refuses():
    with pytest.raises(SystemExit, match="contradicts the arms"):
        from_the_arms(_rungs(box_size=256), "box_size", 192, "--box-size")


def test_a_required_field_no_arm_carries_refuses_and_an_optional_one_is_none():
    with pytest.raises(SystemExit, match="no arm record carries a box_size"):
        from_the_arms(_rungs(), "box_size", None, "--box-size")
    assert from_the_arms(_rungs(), "pixel_size", None, "--pixel-size", required=False) is None


def test_a_float_field_tolerates_its_own_float32_round_trip():
    """`--pixel-size 1.3` against a recorded 1.2999999523162842 is not a contradiction.

    The scorer stores cryoSPARC's `blob/psize_A` as float32, so the arms record 1.3 as
    1.2999999523162842 and an exact comparison refused the only value anyone would type.
    """
    recorded = float(np.float32(1.3))
    assert from_the_arms(_rungs(pixel_size=recorded), "pixel_size", 1.3, "--pixel-size") == 1.3
    with pytest.raises(SystemExit, match="contradicts the arms"):
        from_the_arms(_rungs(pixel_size=recorded), "pixel_size", 1.077, "--pixel-size")
