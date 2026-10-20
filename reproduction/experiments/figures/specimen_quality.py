"""Figure 7: a better specimen buys little for the differences people actually study.
A B-factor damps `d^2` by `exp(-B k^2 / 2)`, so evaluated at the spacing the state difference occupies
(not the map resolution) the raw factor a grid improvement is worth is `exp(Delta_B / 2 d^2)`. Two
things take that back: the raw factor collapses toward one at coarse spacing, and the counting floor of
`sigma_p^2 = [p(1-p) + m/d^2]/N` absorbs most of what survives, so the realized gain in microscope
hours is far smaller. Panel (b) places the survey pairs on the same axis: every one sits at coarse
spacing, where both levers buy almost nothing.
    .venv/bin/python .../specimen_quality.py --record results/decision_curve.json \\
        --spectrum results/difference_spectrum.json --survey results/eq1_mechanistic.json \\
        --out docs/manuscript/figures/specimen_quality
Delta_B is taken from the literature (a grid that lowers B, thinner ice that lowers it less); the figure
fits nothing. Rationale: docs/rationale/figures.md
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from reproduction.record import payload_of
from reproduction.reporting.figures import (
    FINE_PT,
    PAD,
    RULE,
    SERIES,
    TICK_PT,
    label_panels,
    save,
    use_house_style,
)
from reproduction.reporting.numbers import Ledger

_AT = (3.0, 6.0, 10.0, 20.0)

DRIVER = "reproduction/experiments/figures/specimen_quality.py"

GRID = "HexAuFoil"
"""The grid improvement, drawn as both its raw effect on `d^2` and its realized gain in hours."""

ICE = "thinner ice (filtered)"
"""The second lever, drawn once as its raw effect on `d^2`."""

DRAWN = (GRID, ICE)
"""The two improvements drawn; the record carries others."""


def read_record(path: Path) -> list[dict]:
    """The purchasing-power curves for the drawn qualities.

    Args:
        path: A `decision_curve.py` record.

    Returns:
        One curve per drawn quality, each carrying its name, its `Delta_B`, the spacing axis, the raw
        multiplier on `d^2` and the realized gain in microscope hours.

    Raises:
        SystemExit: If the record carries no purchasing power, or a drawn quality is absent.
    """
    record = json.loads(path.read_text())
    result = payload_of(record)
    for key in ("purchasing_power", "qualities"):
        if key not in result:
            raise SystemExit(f"{path} carries no {key}")
    deltas = {entry["name"]: float(entry["delta_b"]) for entry in result["qualities"]}

    curves: list[dict[str, Any]] = []
    for name in DRAWN:
        rows = sorted(
            (row for row in result["purchasing_power"] if row["quality"] == name), key=lambda row: row["spacing"]
        )
        if not rows:
            raise SystemExit(f"{path} carries no purchasing power for {name!r}")
        curves.append(
            {
                "name": name,
                "delta_b": deltas.get(name, float("nan")),
                "d2": float(rows[0]["d2"]),
                "spacing": np.array([row["spacing"] for row in rows], float),
                "raw": np.array([row["raw_gain_on_d2"] for row in rows], float),
                "gain": np.array([row["gain"] for row in rows], float),
            }
        )
    return curves


def read_survey_integral(path: Path) -> dict[str, dict]:
    """Each improvement's multiplier integrated over the survey pairs' whitened difference spectra.

    The headline numbers come from this block, not from evaluating the curve at the median spacing: the
    multiplier is convex in spatial frequency, so the spectrum-weighted integral exceeds the value at a
    pair's median spacing and a single-spacing reading understates the gain.

    Args:
        path: A `decision_curve.py` record carrying a `survey_integral` block.

    Returns:
        One entry per drawn quality, keyed by name, with its integrated and median-spacing multipliers.

    Raises:
        SystemExit: If the record carries no integral, or a drawn quality is absent from it.
    """
    result = payload_of(json.loads(path.read_text()))
    if "survey_integral" not in result:
        raise SystemExit(f"{path} carries no survey_integral; re-run decision_curve.py")
    by_name = {entry["name"]: entry for entry in result["survey_integral"]["per_quality"]}
    for name in DRAWN:
        if name not in by_name:
            raise SystemExit(f"{path} carries no survey integral for {name!r}")
    return by_name


def read_survey_spacings(spectrum: Path, survey: Path) -> np.ndarray:
    """Where each surveyed pair's state difference sits, in angstroms.

    The spacing is `half_power_spacing_a`, the shell at which half the difference power has
    accumulated, read from the difference spectrum and restricted to the pairs admitted to the
    discriminability law so the overlay is the same population the paper surveys.

    Args:
        spectrum: A `difference_spectrum.py` record, one row per pair.
        survey: An `eq1_mechanistic.py` record, whose rows are the admitted survey pairs.

    Returns:
        The finite half-power spacings of the admitted pairs, one per pair.

    Raises:
        SystemExit: If fewer than half the admitted pairs carry a spacing, so the overlay would
            quietly describe a different population than the law.
    """
    admitted = {row["name"] for row in payload_of(json.loads(survey.read_text()))["rows"]}
    spacings = []
    for row in payload_of(json.loads(spectrum.read_text()))["rows"]:
        if row["name"] not in admitted:
            continue
        value = row["defocus"][0].get("half_power_spacing_a")
        if value is not None and np.isfinite(value):
            spacings.append(float(value))
    if len(spacings) < len(admitted) / 2:
        raise SystemExit(
            f"{len(spacings)} of {len(admitted)} admitted pairs carry a spacing in {spectrum}; the "
            "overlay would not represent the survey"
        )
    return np.array(spacings, float)


def draw(curves: list[dict], spacings: np.ndarray):
    """Raw versus realized multiplier against spacing, over the survey's spacing distribution.

    Color is the lever (blue the grid, orange the ice) and line style the quantity: a solid line is the
    improvement on `d'^2`, a dashed line the realized reduction in microscope hours.

    Args:
        curves: As returned by `read_record`.
        spacings: The surveyed pairs' half-power spacings, from `read_survey_spacings`.

    Returns:
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(4.0, 5.3))
    rows = figure.add_gridspec(2, 1, height_ratios=(2.8, 0.75))
    top = figure.add_subplot(rows[0])
    bottom = figure.add_subplot(rows[1], sharex=top)

    grid = next(c for c in curves if c["name"] == GRID)
    ice = next(c for c in curves if c["name"] == ICE)

    # (a) each lever's improvement on d'^2 (solid), and what it is worth once the counting floor of
    # eq:variance has taken its share (dashed): the realized reduction in hours runs far below it
    dashed = (0, (4, 2))
    top.axhline(1.0, color=RULE, linewidth=0.9, zorder=1)
    top.plot(grid["spacing"], grid["raw"], color=SERIES[0], linewidth=1.6, zorder=4, label="Better grid ($d'^2$)")
    top.plot(
        grid["spacing"],
        grid["gain"],
        color=SERIES[0],
        linewidth=1.6,
        linestyle=dashed,
        zorder=4,
        label="Better grid (hours)",
    )
    top.plot(ice["spacing"], ice["raw"], color=SERIES[1], linewidth=1.6, zorder=3, label="Thinner ice ($d'^2$)")
    top.plot(
        ice["spacing"],
        ice["gain"],
        color=SERIES[1],
        linewidth=1.6,
        linestyle=dashed,
        zorder=3,
        label="Thinner ice (hours)",
    )

    top.set_yscale("log")
    top.set_ylabel("Multiplier  ($\\times$)")
    top.legend(frameon=False, fontsize=TICK_PT, loc="upper right")
    top.tick_params(axis="x", labelbottom=False)

    # (b) where the surveyed pairs' differences sit, on the same spacing axis so the vertical alignment
    # with (a) reads: every pair lands where the curves have already collapsed toward one
    edges = np.geomspace(spacings.min(), spacings.max(), 26)
    bottom.hist(spacings, bins=edges, color=RULE, linewidth=0.0)
    median = float(np.median(spacings))
    bottom.axvline(median, color="0.2", linewidth=0.9, zorder=3)
    bottom.annotate(
        "median",
        xy=(median, 0.96),
        xytext=(3, 0),
        textcoords="offset points",
        xycoords=("data", "axes fraction"),
        fontsize=TICK_PT,
        color="0.2",
        va="top",
        ha="left",
    )
    bottom.annotate(
        f"{len(spacings):,} pairs", xy=(0.02, 0.9), xycoords="axes fraction", fontsize=FINE_PT, color="0.45", va="top"
    )
    bottom.set_ylabel("Surveyed pairs")

    # a log axis defaults to 2x10^0 style labels here, which collide and read as exponents rather than
    # as the angstrom spacings they are
    top.set_xscale("log")
    ticks = [2, 3, 5, 10, 20, 40]
    bottom.set_xticks(ticks)
    bottom.set_xticklabels([str(value) for value in ticks])
    bottom.minorticks_off()
    bottom.set_xlim(2.0, 42.0)
    bottom.set_xlabel("Spacing where the two states differ (Å)")

    top.set_box_aspect(1.0)  # square main panel, matching fig:memorization
    label_panels(top, bottom)
    figure.tight_layout(pad=PAD, h_pad=0.6)
    return figure


def main() -> int:
    """Draw the panel and print every number on it.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=Path("results/decision_curve.json"))
    parser.add_argument("--spectrum", type=Path, default=Path("results/difference_spectrum.json"))
    parser.add_argument("--survey", type=Path, default=Path("results/eq1_mechanistic.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/manuscript/figures/specimen_quality"))
    args = parser.parse_args()

    use_house_style()
    curves = read_record(args.record)
    integral = read_survey_integral(args.record)
    spacings = read_survey_spacings(args.spectrum, args.survey)
    written = save(draw(curves, spacings), args.out)

    grid = next(c for c in curves if c["name"] == GRID)
    median = float(np.median(spacings))
    grid_raw = float(integral[GRID]["raw_integrated_median"])
    grid_hours = float(integral[GRID]["gain_integrated_median"])
    ice_raw = float(integral[ICE]["raw_integrated_median"])
    ice_hours = float(integral[ICE]["gain_integrated_median"])

    val = Ledger()
    val("specimen_survey_n", len(spacings), ",d")
    val("specimen_survey_spacing_med", median, ".0f")
    val("specimen_survey_spacing_min", float(spacings.min()), ".1f")
    val("specimen_grid_raw_at_med", grid_raw, ".2f")
    val("specimen_grid_hours_at_med", grid_hours, ".2f")
    val("specimen_ice_raw_at_med", ice_raw, ".2f")
    val("specimen_ice_hours_at_med", ice_hours, ".2f")
    val.dump(args.out.parent / "values" / f"{args.out.name}.json")

    print("multiplier on d'^2 by where the state difference lives:")
    header = "  " + "spacing".ljust(24) + "".join(f"{value:>9.0f} A" for value in _AT)
    print(header)
    for curve in curves:
        row = "".join(f"{np.interp(v, curve['spacing'], curve['raw']):>11.2f}" for v in _AT)
        print(f"  {curve['name']:24s}{row}")
    print(
        f"\n{len(spacings):,} survey pairs on the same axis: spacings {spacings.min():.1f}-{spacings.max():.1f} A, "
        f"median {median:.1f} A"
    )
    print(
        f"  integrated over each pair's spectrum the grid is worth {grid_raw:.2f}x on d'^2 ({grid_hours:.2f}x in "
        f"hours), thinner ice {ice_raw:.2f}x on d'^2 ({ice_hours:.2f}x in hours) (d^2 = {grid['d2']:.2f}); "
        f"at the median spacing alone they were {integral[GRID]['raw_at_median_spacing']:.2f}x and "
        f"{integral[ICE]['raw_at_median_spacing']:.2f}x on d'^2"
    )
    print("  wrote " + ", ".join(str(path) for path in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
