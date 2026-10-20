"""The one operating point every experiment inherits, as a single importable source.

Both the drivers (their argparse defaults) and the `reproduction/experiments/reproduce.py` runner read this, so each
knob is declared ONCE rather than every driver hardcoding its own. Hardcoded per-driver defaults are
what drifted (pixel 1.5 vs 2.0, contrast 1 vs 0.1) and made replaying stored args reproduce a divergent
operating point -- the lesson of the rebuild. A departure is `replace(SHARED, pixel_size=1.30)`.
"""

from dataclasses import dataclass

OP_FLAGS = ("dose", "pixel_size", "rmsf", "contrast")
"""The operating-point knobs, in the order they are emitted; a driver takes the subset it accepts."""

_FLAG_NAME = {"dose": "--dose", "pixel_size": "--pixel-size", "rmsf": "--rmsf", "contrast": "--contrast"}


@dataclass(frozen=True)
class RunSettings:
    """The one operating point every experiment inherits, as values and as CLI flags.

    Distinct from `pipeline.OperatingPoint`, which records what a run WAS measured at (its per-pair box
    and measured template power); this is the settings a run is CONFIGURED with. A driver reads the
    fields for its argparse defaults; the runner emits them as flags with `to_flags`.
    """

    dose: float = 40.0
    pixel_size: float = 2.0
    rmsf: float = 0.5
    contrast: float = 1.0

    def to_flags(self, knobs: tuple[str, ...] = OP_FLAGS) -> list[str]:
        """The requested knobs as `--flag value` argv, in `knobs` order.

        Args:
            knobs: Which fields to emit; a driver lacking a flag (e.g. no `--contrast`) omits it.
        """
        return [x for k in knobs for x in (_FLAG_NAME[k], f"{getattr(self, k):g}")]


SHARED = RunSettings()
"""The single operating point. Import this for a driver's operating-point defaults, or emit it through
the reproduce.py runner. A departure is `replace(SHARED, pixel_size=1.30)`, never a second default."""
