"""Checks on the operating point and the ratio it is allowed to gate.

A ratio whose numerator and denominator come from different exposures is meaningless. Within one
`run_chain` both arms share a spectrum and the ratio is safe by construction; these cover the
ACROSS-run case, where the operating points must match or the ratio is refused.

`rung_ratio` reads only `operating_point` and one rung, so these are duck-typed rather than built
through a full `run_chain` -- the arithmetic is what is under test, not the ladder.
"""

from types import SimpleNamespace

import pytest

from cryodisc.pipeline.chain import Rung
from cryodisc.pipeline.operating_point import OperatingPoint, rung_ratio

BASE = OperatingPoint(dose=40.0, pixel_size=1.5, box_size=128, rmsf=0.5, contrast=1.0, template_power=1000.0)


def result(template_power=1000.0, d2=2.0, **overrides):
    """A stand-in carrying the two attributes `rung_ratio` reads."""
    point = OperatingPoint(
        dose=overrides.get("dose", BASE.dose),
        pixel_size=overrides.get("pixel_size", BASE.pixel_size),
        box_size=overrides.get("box_size", BASE.box_size),
        rmsf=overrides.get("rmsf", BASE.rmsf),
        contrast=overrides.get("contrast", BASE.contrast),
        template_power=template_power,
    )
    return SimpleNamespace(marginal=Rung(d2=d2, d2_err=0.1, information=0.4, n_for_error=5000.0), operating_point=point)


def test_two_runs_at_one_exposure_match():
    """Identical settings and a template power within the re-draw scatter still match."""
    assert BASE.matches(BASE)
    assert BASE.matches(OperatingPoint(40.0, 1.5, 128, 0.5, 1.0, 1005.0))
    assert BASE.matches(OperatingPoint(40.0, 1.5, 128, 0.5, 1.0, 1005.0), tolerance=0.01)


def test_the_17_percent_that_caused_the_2_point_85x_does_not_match():
    """A template power gap wide enough to signal a different exposure fails to match."""
    assert not BASE.matches(OperatingPoint(40.0, 1.5, 128, 0.5, 1.0, 1175.0))


@pytest.mark.parametrize(
    "other",
    [
        OperatingPoint(80.0, 1.5, 128, 0.5, 1.0, 1000.0),
        OperatingPoint(40.0, 1.2, 128, 0.5, 1.0, 1000.0),
        OperatingPoint(40.0, 1.5, 64, 0.5, 1.0, 1000.0),
        OperatingPoint(40.0, 1.5, 128, 1.5, 1.0, 1000.0),
        OperatingPoint(40.0, 1.5, 128, 0.5, 2.0, 1000.0),
    ],
)
def test_any_discrete_setting_differing_is_a_different_exposure(other):
    """Even at an identical template power: two runs that reached the same signal by different means
    are not interchangeable, and the ratio would be measuring the difference."""
    assert not BASE.matches(other)


def test_a_ratio_across_matching_runs_is_the_ratio_of_the_rungs():
    numerator, denominator = result(d2=6.0), result(d2=2.0)
    assert rung_ratio(numerator, denominator) == pytest.approx(3.0)


def test_a_ratio_across_operating_points_is_refused_rather_than_returned():
    """The whole point: it has to raise, not warn and not silently rescale."""
    with pytest.raises(ValueError, match="refusing a ratio across operating points"):
        rung_ratio(result(template_power=1175.0), result(template_power=1000.0))
    with pytest.raises(ValueError, match="refusing a ratio"):
        rung_ratio(result(), result(box_size=64))


def test_an_unknown_rung_name_is_refused():
    """`getattr` on a typo would otherwise raise an AttributeError from somewhere unhelpful."""
    with pytest.raises(ValueError, match="is not a rung"):
        rung_ratio(result(), result(), "margnial")
