"""What a ladder was measured at, and the rule for when two of them may be divided.

Below `chain` rather than inside it, because two modules enforce the same rule and neither owns it:
`rung_ratio` here refuses a cross-run ratio, and `validation.floor.twin_floor` refuses a twin whose
exposure disagrees with the measurement's.

The rule guards against dividing rungs from different exposures: a rung moves with the exposure while
the achieved rung does not follow, so the ratio would measure the exposure difference rather than the
pipeline. Within one `run_chain` both arms share one spectrum; ACROSS runs nothing else enforces it.
"""

from dataclasses import dataclass
from typing import Protocol

RUNG_NAMES = ("known_pose", "local_pose", "refined_pose", "marginal", "best_match", "achieved")
"""The ladder in the order it is reported, which `chain.ChainResult.summary` prints and `rung_ratio` accepts."""


@dataclass(frozen=True)
class OperatingPoint:
    """What the ladder was measured at, recorded so a ratio across two runs can be refused.

    Compare with `matches` before forming any cross-run ratio; see the module docstring for what
    happens when nothing does.

    Args:
        dose: Total exposure in e/A^2.
        pixel_size: Pixel size in A.
        box_size: Image size in pixels.
        rmsf: Intra-state RMS fluctuation in A, which damps the signal.
        contrast: Template scale `a` the stack was rendered at.
        template_power: `mean_pose sum_k a^2 |M_A(k)|^2 / sigma^2(k)`, the whitened power of the
            state-A bank. The scalar the rungs actually live on: every one of the parameters above
            moves the ladder only through this, so two runs agreeing here agree on the exposure
            even if they reached it differently.
    """

    dose: float
    pixel_size: float
    box_size: int
    rmsf: float
    contrast: float
    template_power: float

    def matches(self, other: "OperatingPoint", tolerance: float = 0.01) -> bool:
        """Whether two runs are close enough in exposure for a ratio between them to mean anything.

        Args:
            other: The operating point to compare against.
            tolerance: Fractional agreement required of `template_power`. The default admits the
                sampling scatter of a re-drawn stack and refuses an exposure difference.

        Returns:
            True when the discrete settings are identical and `template_power` agrees to
            `tolerance`.
        """
        discrete = (self.box_size, self.pixel_size, self.dose, self.rmsf, self.contrast) == (
            other.box_size,
            other.pixel_size,
            other.dose,
            other.rmsf,
            other.contrast,
        )
        scale = max(abs(self.template_power), abs(other.template_power), 1e-30)
        return discrete and abs(self.template_power - other.template_power) / scale <= tolerance


class Ladder(Protocol):
    """A ladder `rung_ratio` can divide: an `OperatingPoint` and a `Rung` per `RUNG_NAMES` entry."""

    # Structural, so src/ names the shape it needs without importing the research tier that defines it.
    operating_point: "OperatingPoint"


def rung_ratio(numerator: Ladder, denominator: Ladder, rung: str = "marginal") -> float:
    """One run's rung over another's, refused unless both were measured at the same exposure.

    Args:
        numerator: The run on top.
        denominator: The run underneath.
        rung: Which rung to compare. Any field of `chain.ChainResult` that is a `Rung`.

    Returns:
        The ratio of the two `d2` values.

    Raises:
        ValueError: If `rung` is not a rung, or if the two operating points disagree.
    """
    if rung not in RUNG_NAMES:
        raise ValueError(f"{rung!r} is not a rung; expected one of {RUNG_NAMES}")
    if not numerator.operating_point.matches(denominator.operating_point):
        raise ValueError(
            f"refusing a ratio across operating points: template power "
            f"{numerator.operating_point.template_power:.4g} against "
            f"{denominator.operating_point.template_power:.4g}. A rung moves with the exposure and "
            "the achieved rung does not follow it, so the ratio measures the exposure difference "
            "rather than the pipeline. Re-run both arms at one plan, box, rmsf and contrast."
        )
    return float(getattr(numerator, rung).d2 / getattr(denominator, rung).d2)
