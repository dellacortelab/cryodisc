"""Named, pre-formatted numbers a figure driver emits for the manuscript to fill by `[[val:name]]`.

A driver records each number it is cited for under a name, formatting it once; the returned string is
what the driver prints, so the same rounding reaches the console, `NUMBERS.txt` and the manuscript.
`build.py` merges every driver's dump and substitutes the tokens, so a value is written in one place.
"""

import json
from pathlib import Path


class Ledger:
    """A driver's named numbers, each formatted once and reused wherever it is shown."""

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def __call__(self, name: str, value: float | str, spec: str = "") -> str:
        """Record `value` under `name`, formatted by `spec`, and return the formatted string.

        Recording the same name with a different formatted string is a bug, not a last-write-wins, so
        it raises: two call sites that disagree would fill the manuscript with whichever ran last.
        """
        text = format(value, spec)
        if self._values.setdefault(name, text) != text:
            raise ValueError(f"{name!r} already recorded as {self._values[name]!r}, now {text!r}")
        return text

    def dump(self, path: Path) -> None:
        """Write the `{name: formatted}` map for `build.py` to merge."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._values, indent=1, sort_keys=True) + "\n")
