"""Figure S3: discriminability accumulates at shells coarser than the declared resolution.

    .venv/bin/python .../resolution_offset.py --records results/fsc_bridge_cas9.json results/fsc_bridge_ribosome_wide.json --out docs/manuscript/figures/resolution_offset
Resolution is set by the finest shell that still reproduces; discriminability by wherever the two
states differ. Draws the cumulative `d^2` of (F1), truncated at each map's own gold-standard
`FSC = 0.143` shell, against shell resolution in units of that declared resolution. Scaling the
abscissa by each pair's declared resolution is the point: in angstroms the curves look unrelated,
scaled they overlap, making the offset a statement about the FSC criterion. The quartile table is
printed under one convention (`searchsorted`), so a hand-copied one cannot drift. Rationale: docs/rationale/figures.md
"""

import argparse
import json
from pathlib import Path

import numpy as np

from reproduction.record import payload_of
from reproduction.reporting.figures import PAD, RULE, SERIES, SINGLE_COLUMN, save, use_house_style
from reproduction.reporting.numbers import Ledger

DRIVER = "reproduction/experiments/figures/resolution_offset.py"

LABELS = {
    "fsc_bridge_cas9": "SpCas9 6nt/12nt",
    "fsc_bridge_ribosome_wide": "80S ratchet",
}
"""Readable names for the record stems."""


def read_curve(path: Path) -> dict:
    """The cumulative `d^2` of one pair, truncated at its gold-standard FSC shell.

    Truncation is at `FSC = 0.143` rather than at the scoring band, because the comparison is against
    the resolution a depositor would declare, and that is where the criterion places it.

    Args:
        path: An `fsc_bridge.py` record.

    Returns:
        The label, the declared resolution, the shell resolutions scaled by it, and the cumulative
        fraction of `d^2` at each.

    Raises:
        SystemExit: If the record carries no extended cumulative or no FSC crossing.
    """
    record = json.loads(path.read_text())
    result = payload_of(record)
    try:
        extended = np.asarray(result["predictions"]["global"]["cumulative_a_beyond_band"], float)
        crossing = int(result["fsc_crossing_shell"])
        declared = float(result["fsc_crossing_resolution"])
        box, pixel = int(result["box_size"]), float(result["pixel_size"])
    except (KeyError, TypeError) as missing:
        raise SystemExit(f"{path} is not an fsc_bridge record with a crossing: {missing}") from missing
    if crossing > extended.size:
        raise SystemExit(f"{path}: FSC crosses at shell {crossing}, past the {extended.size} computed")
    cumulative = extended[:crossing]
    shells = np.arange(1, cumulative.size + 1)
    return {
        "label": LABELS.get(path.stem, path.stem),
        "declared": declared,
        # shell 1 is the box, so the abscissa runs from very coarse down to the declared resolution
        "scaled": (box * pixel / shells) / declared,
        "fraction": cumulative / cumulative[-1],
    }


QUARTILES = (0.25, 0.50, 0.75, 0.90)
"""The fractions Note 6's table reports. The half-point is the one the panel marks."""


def reaches(curve: dict, fraction: float) -> float:
    """Where the cumulative first reaches `fraction`, in units of the declared resolution.

    The first shell at or above the threshold, never the last one below it; the two can differ by a
    large step, so the choice is made once and applied everywhere.

    Args:
        curve: As returned by `read_curve`.
        fraction: Share of `d^2` accumulated.

    Returns:
        The multiple of the declared resolution at which that share is reached.
    """
    return float(curve["scaled"][int(np.searchsorted(curve["fraction"], fraction))])


def half_point(curve: dict) -> float:
    """Where the cumulative reaches half, in units of the declared resolution.

    Args:
        curve: As returned by `read_curve`.

    Returns:
        The multiple of the declared resolution at which half of `d^2` has accumulated.
    """
    return reaches(curve, 0.5)


def draw(curves: list[dict]):
    """The panel: cumulative fraction of `d^2` against scaled shell resolution.

    Args:
        curves: As returned by `read_curve`, one per pair.

    Returns:
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.82))
    axes.axhline(0.5, color=RULE, linewidth=0.9, zorder=1)
    # the declared resolution is 1 on this axis; everything left of it is coarser
    axes.axvline(1.0, color=RULE, linewidth=0.9, linestyle=(0, (4, 2)), zorder=1)

    for index, curve in enumerate(curves):
        axes.plot(
            curve["scaled"],
            curve["fraction"],
            color=SERIES[index],
            linewidth=1.4,
            label=f"{curve['label']} ({curve['declared']:.1f} Å)",
            zorder=3,
        )
        half = half_point(curve)
        # the two half-points nearly coincide, which is the message; a white edge keeps both readable
        axes.plot(
            [half],
            [0.5],
            marker="o",
            color=SERIES[index],
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=4 + index,
        )

    axes.set_xscale("log")
    axes.invert_xaxis()
    axes.set_xlabel("shell resolution / declared resolution")
    axes.set_ylabel("cumulative share of $d^2$")
    axes.set_ylim(0, 1.02)
    axes.legend(frameon=False, fontsize=6, loc="upper left")
    figure.tight_layout(pad=PAD)
    return figure


def main() -> int:
    """Draw the panel and print every number on it.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        nargs="+",
        type=Path,
        default=[Path("results/fsc_bridge_cas9.json"), Path("results/fsc_bridge_ribosome_wide.json")],
    )
    parser.add_argument("--out", type=Path, default=Path("docs/manuscript/figures/resolution_offset"))
    args = parser.parse_args()

    use_house_style()
    curves = [read_curve(path) for path in args.records]
    written = save(draw(curves), args.out)

    print(f"{len(curves)} pairs, cumulative d^2 truncated at each map's own FSC = 0.143 shell")
    print("  " + "pair".ljust(18) + "declared" + "".join(f"{int(f * 100):>9d}%" for f in QUARTILES))
    for curve in curves:
        row = "".join(f"{reaches(curve, f):>9.2f}x" for f in QUARTILES)
        print(f"  {curve['label']:18s}{curve['declared']:6.2f} A{row}")
    for curve in curves:
        half = half_point(curve)
        print(
            f"  {curve['label']:18s} declared {curve['declared']:5.2f} A   "
            f"half of d^2 by {half:.2f}x declared = {half * curve['declared']:.1f} A"
        )
    halves = [half_point(curve) for curve in curves]
    print(
        f"  the half-points agree at {min(halves):.2f}x to {max(halves):.2f}x of declared resolution, "
        f"where in angstroms they are {min(h * c['declared'] for h, c in zip(halves, curves, strict=True)):.1f} "
        f"and {max(h * c['declared'] for h, c in zip(halves, curves, strict=True)):.1f} A"
    )
    val = Ledger()
    val("res_offset_min", min(halves), ".1f")  # the two pairs' half-d^2 multiples of declared resolution
    val("res_offset_max", max(halves), ".1f")
    val.dump(args.out.parent / "values" / f"{args.out.name}.json")
    print("  wrote " + ", ".join(str(path) for path in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
