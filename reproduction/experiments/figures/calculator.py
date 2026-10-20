"""Figure 4: how many particles a stated precision costs, and the check that the number is right.

    .venv/bin/python .../calculator.py --deflections results/eq1_mechanistic.json --check results/calculator_check.json --out .../calculator
Left, the design chart: pick a precision, find your pair on the horizontal axis, read the count off the
vertical. Right, `sigma_p^2 = [p(1-p) + v/Delta^2]/N` run backwards on the real SpCas9 particles of
Figure 1 (take an `N`, measure the error by resampling), with a residual strip. The surveyed pairs are
deposited MODELS (so the panel says "PDB conformational pairs", never "real"). Which regime a pair is in
DEPENDS ON THE MIXTURE, so the driver prints the counting-dominated fraction at each mixture. The worked
particle-count example is the real SpCas9 pair of Figure 1 (`EXAMPLE`). Rationale: docs/rationale/figures.md
"""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cryodisc.statistics.information import fisher_from_deflection
from cryodisc.statistics.power import n_for_error_from_information
from reproduction.record import payload_of
from reproduction.reporting.figures import (
    BAND,
    DOUBLE_COLUMN,
    FINE_PT,
    GAP_IN,
    RULE,
    SERIES,
    SINGLE_COLUMN,
    TICK_PT,
    label_panels,
    save,
    use_house_style,
)
from reproduction.reporting.numbers import Ledger

_TARGETS = [0.05, 0.01]
"""Precisions drawn as curves in panel (a), told apart by line style."""

_REPORTED = (0.05, 0.02, 0.01)
"""Precisions the body text cites a particle cost for, a superset of the drawn curves."""

DRIVER = "reproduction/experiments/figures/calculator.py"

EXAMPLE = ("SpCas9 pair", 2.906)
"""The worked particle-count example: the SpCas9 6nt/12nt pools of Figure 1, keyed by `(label, d'^2_obs)`."""


def read_converged(path: Path) -> list[dict]:
    """The converged pose-marginal `d^2` of each surveyed pair, at the single reference dose.

    The survey is collected at one reference dose, so the plateau needs no rescaling; it is the
    distribution the law is fit to. Mapped onto the `d2_marginal` key the downstream count code
    already reads.

    Args:
        path: The `eq1_mechanistic.py` record.

    Returns:
        Rows carrying `name`, `d2_marginal` and `ca_rmsd`.

    Raises:
        SystemExit: If the record carries no rows.
    """
    result = payload_of(json.loads(path.read_text()))
    rows = [
        {"name": row["name"], "d2_marginal": row["d2_plateau"], "ca_rmsd": row["ca_rmsd"]}
        for row in result.get("rows", [])
    ]
    if not rows:
        raise SystemExit(f"{path} carries no converged rows")
    return rows


def read_survey_deflections(path: Path) -> list[dict]:
    """Every converged pair's plateau d'^2, mass or not -- the full survey distribution the counting
    statistics span. `read_converged` is the narrower mass-bearing set equation (1) fits."""
    result = payload_of(json.loads(path.read_text()))
    survey = [{"name": d["name"], "d2_marginal": d["d2_plateau"]} for d in result.get("converged_deflections", [])]
    return survey or read_converged(path)  # fall back to the fit set for a record written before this field


NAMES = {
    "msba": "MsbA",
    "fanac": "FaNaC",
    "nkcc1": "NKCC1",
    "slc19a3": "SLC19A3",
    "slco2a1": "SLCO2A1",
    "ste2": "Ste2",
    "hk-atpase": "HK-ATPase",
}
"""Display names of the seven recovery systems, keyed as `calculator_check.py` records them."""


def read_check(path: Path) -> dict:
    """The `calculator_check.py` record: SpCas9's sweep per fraction and the seven systems' ratios.

    Args:
        path: The result JSON.

    Returns:
        `sweep`, `{fraction: arrays of n, predicted, measured, err, floor}`; `systems`, as recorded;
        and `trials`, the simulated experiments behind each point.
    """
    result = payload_of(json.loads(path.read_text()))
    sweep = {}
    for key, rows in result["cas9_sweep"].items():
        sweep[float(key)] = {
            name: np.array([row[field] for row in rows], dtype=float)
            for name, field in (("n", "n"), ("predicted", "predicted"), ("measured", "measured"), ("floor", "floor"))
        }
        # a sample SD over `trials` experiments carries relative error 1/sqrt(2(trials-1))
        sweep[float(key)]["err"] = sweep[float(key)]["measured"] * np.array([row["measured_rel_err"] for row in rows])
    return {"sweep": sweep, "systems": result["systems"], "trials": int(result["cas9_sweep"][key][0]["trials"])}


def counting_share(rows: list[dict], fraction: float) -> tuple[float, float]:
    """How much of the variance on the fraction is counting rather than the states being alike.

    Mirrors `benchmark/counting_floor.predicted`: `sigma_p^2 = [p(1-p) + v/Delta^2]/N`, and the
    perfect-label floor is `p(1-p)/N`, so the floor's share is `p(1-p) / (p(1-p) + v/Delta^2)`. The
    second-moment term IS the reciprocal of the deflection, `v/Delta^2 = 1/d^2`, which is what lets a
    survey row stand in for a measured pool. Reproduced rather than imported because `experiments`
    modules do not import each other.

    NOT `fisher_from_deflection`, which is the exact-likelihood information and gives a different
    number. The expression here reproduces `floor_share_at_half` and `floor_share_at_min_fraction`.

    Args:
        rows: Survey rows carrying `d2_marginal`.
        fraction: The population fraction to evaluate at.

    Returns:
        `(share of pairs above 0.5, median share)`. A pair above 0.5 is counting-dominated.
    """
    binomial = fraction * (1.0 - fraction)
    shares = np.array([binomial / (binomial + 1.0 / float(row["d2_marginal"])) for row in rows])
    return float((shares > 0.5).mean()), float(np.median(shares))


def route_gap(deflections: np.ndarray, fraction: float) -> float:
    """Median fraction by which the Fisher count falls below the variance-inversion count.

    The exact-likelihood information `fisher_from_deflection` against the equal-covariance value
    `1/[p(1-p)+1/d'^2]`; the count is `1/I`, so the Fisher route needs the fewer particles wherever the
    exact likelihood is the more informative.
    """
    counting = 1.0 / (fraction * (1.0 - fraction) + 1.0 / deflections)
    fisher = np.asarray(jax.vmap(lambda value: fisher_from_deflection(value, float(fraction)))(deflections))
    return float(np.median(1.0 - counting / fisher))


def counts(deflections: np.ndarray, target: float, fraction: float) -> np.ndarray:
    """`N` for each deflection at a stated precision and expected fraction.

    Args:
        deflections: `d^2` values.
        target: Wanted standard error on the fraction.
        fraction: Expected population fraction.

    Returns:
        Particle counts.
    """
    # float() on `fraction`/`target`, since a numpy scalar reaches the jitted guard as a traced array
    # and raises; the DEFLECTIONS stay an array through `vmap`, else each distinct d^2 recompiles
    information = jax.vmap(lambda value: fisher_from_deflection(value, float(fraction)))(
        jnp.asarray(deflections, dtype=float)
    )
    return np.asarray(n_for_error_from_information(information, error_target=float(target), fraction=float(fraction)))


RARE = 0.05
"""The minority population drawn beside the even mixture: where counting costs least and the states
being alike costs most."""


def draw(rows: list[dict], targets: list[float], fractions: tuple[float, float], check: dict):
    """The design chart, its test on one real pair over collection size, and its test on seven real systems.

    Nothing on either panel is prose. Where the surveyed pairs come from, what the dashed floors are,
    what the +/- mean and what the gap to the floor costs are all four sentences, and four sentences
    are a caption -- on the axes they compete with the curves for the same space and get read first.

    Args:
        rows: Survey rows, for the rug.
        targets: Precisions to draw one curve each for.
        fractions: The even and the minority fraction, drawn in two colors.
        check: The `calculator_check.py` record, as `read_check` returns it.

    Returns:
        The figure.
    """
    import matplotlib.pyplot as plt

    surveyed = np.array([row["d2_marginal"] for row in rows], dtype=float)
    # a residual strip under (b): formula and measurement coincide, and a panel where two things
    # overlap hides the size of their departure, which the strip makes readable
    # constrained_layout, not tight_layout: panel (c) spans both rows and tight_layout silently falls
    # back to loose default margins for a spanning axes. With (c)'s names now INSIDE its axes there is no
    # outside decoration to reserve for, so constrained_layout gives the three columns one even gap.
    figure = plt.figure(figsize=(DOUBLE_COLUMN, SINGLE_COLUMN * 1.24), layout="constrained")
    # drive the gap from the same GAP_IN as the tight_layout figures (its pads are in inches, so the
    # inter-panel gap is 2 * w_pad = GAP_IN); wspace/hspace stay 0 so no extra fraction is added on top
    figure.get_layout_engine().set(w_pad=GAP_IN / 2, h_pad=GAP_IN / 2, wspace=0.0, hspace=0.0)
    # shorter main row + taller strip row, so the main panels read roughly square and the strips are
    # tall enough for a single-line y-label
    outer = figure.add_gridspec(2, 3, height_ratios=[2.1, 1.1], width_ratios=[1.1, 1.1, 1.0])
    left = figure.add_subplot(outer[0, 0])
    # the survey in its own strip under (a), as (b) has its residual strip: inside the log axis its bars
    # would read as particle counts
    survey = figure.add_subplot(outer[1, 0], sharex=left)
    right = figure.add_subplot(outer[0, 1])
    departure = figure.add_subplot(outer[1, 1], sharex=right)
    breadth = figure.add_subplot(outer[:, 2])
    # down to 0.3, below the survey minimum: the rising part of each curve, where the states being
    # alike sets the cost, is the regime the paper's own low-d^2 specimens sit in
    grid = np.geomspace(0.3, surveyed.max() * 1.3, 240)

    # color is the population fraction throughout the figure; in (a) line style is the precision target,
    # and each curve flattens onto its own counting floor, so the plateau draws the floor and needs no line
    prec_styles = ("-", (0, (3.5, 2.0)))
    for index, fraction in enumerate(fractions):
        color = SERIES[index]
        for target, style in zip(targets, prec_styles):
            left.plot(grid, counts(grid, target, fraction), color=color, lw=1.4, ls=style, zorder=4)
    # label the precision on the even-mixture curves, so the style-to-precision map is read off the plot
    for target, style in zip(targets, prec_styles):
        curve = counts(grid, target, fractions[0])
        left.annotate(
            rf"$\pm${100 * target:g}%",
            xy=(grid[0], curve[0]),
            xytext=(0, 5),
            textcoords="offset points",
            color="0.3",
            fontsize=TICK_PT,
            ha="left",
            va="bottom",
            annotation_clip=False,
        )
    for index, fraction in enumerate(fractions):
        left.plot([], [], color=SERIES[index], lw=1.4, label=f"$p = {fraction:g}$")
    left.legend(loc="upper right", handlelength=1.6, borderpad=0.2, labelspacing=0.3, fontsize=TICK_PT)
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_ylabel("Particles needed")
    left.margins(x=0.14)

    left.tick_params(axis="x", labelbottom=False)
    survey.hist(surveyed, bins=np.geomspace(surveyed.min(), surveyed.max(), 36), color=RULE, lw=0)
    survey.set_xlabel("Discriminability  $d'^2$")
    survey.set_ylabel("Surveyed pairs")
    survey.annotate(
        f"{len(surveyed):,} pairs", xy=(0.02, 0.92), xycoords="axes fraction", fontsize=FINE_PT, color="0.45", va="top"
    )
    # (b) the calculator's prediction against simulated experiments on real SpCas9 particles, each with
    # its own finite references; fraction is the color, predicted a line, measured the markers, floor dotted
    for index, fraction in enumerate(fractions):
        color = SERIES[index]
        run = check["sweep"][fraction]
        right.plot(run["n"], run["predicted"], color=color, lw=1.3, zorder=2)
        right.plot(run["n"], run["floor"], color=color, lw=0.9, ls=":", zorder=1)
        right.errorbar(
            run["n"],
            run["measured"],
            yerr=run["err"],
            fmt="o",
            color=color,
            mec=color,
            ms=4.0,
            mew=0.8,
            elinewidth=0.8,
            capsize=1.6,
            zorder=4,
        )
        departure.errorbar(
            run["n"],
            100.0 * (run["measured"] / run["predicted"] - 1.0),
            yerr=100.0 * run["err"] / run["predicted"],
            fmt="o",
            color=color,
            mec=color,
            ms=3.6,
            mew=0.8,
            elinewidth=0.8,
            capsize=1.6,
            zorder=3,
        )
    # the fraction color key lives once, in (a); here the legend names only the element types
    right.plot([], [], color="0.3", lw=1.3, label="Predicted")
    right.plot([], [], "o", color="0.3", ms=4.0, label="Measured")
    right.plot([], [], color="0.3", lw=0.9, ls=":", label="Counting floor")
    right.set_xscale("log")
    right.set_yscale("log")
    for axis in (right.xaxis, right.yaxis):
        axis.set_minor_formatter(plt.NullFormatter())
    right.set_ylabel("Error,  $\\sigma_p$")
    right.tick_params(axis="x", labelbottom=False)
    right.legend(loc="lower left", handlelength=1.6, borderpad=0.2, labelspacing=0.3, fontsize=TICK_PT)

    # the strip's band is one experiment count's own precision, the scale a departure is read against
    precision = 100.0 / np.sqrt(2.0 * (check["trials"] - 1))
    departure.axhspan(-precision, precision, color=BAND, lw=0, zorder=1)
    departure.axhline(0.0, color=RULE, lw=0.8, zorder=2)
    departure.set_xscale("log")
    departure.set_ylim(-32, 38)
    departure.set_ylabel("Rel. diff. (%)")
    departure.set_xlabel("Particles collected")
    # no "%" on the ticks: the narrower labels keep the y-axis title clear of the survey histogram
    departure.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))

    # (c) every recovery system, read as one ratio each: sigma_p scales as 1/sqrt(N) in both, so their
    # ratio does not depend on the collection size it was read at. Rows order by d'^2, which the labels carry
    systems = sorted(check["systems"], key=lambda entry: entry["d2_obs"])
    rows_y = np.arange(len(systems))
    breadth.axvline(1.0, color=RULE, lw=0.8, zorder=2)
    for index, (fraction, nudge) in enumerate(zip(fractions, (0.12, -0.12))):
        color = SERIES[index]
        ratio = np.array([entry["by_fraction"][f"{fraction:g}"]["ratio"] for entry in systems])
        spread = ratio / np.sqrt(2.0 * (check["trials"] - 1))
        breadth.errorbar(
            ratio,
            rows_y + nudge,
            xerr=spread,
            fmt="o",
            color=color,
            mec=color,
            ms=4.0,
            mew=0.8,
            elinewidth=0.8,
            capsize=1.4,
            zorder=3,
        )
    # each system named inside the panel, above its own pair and just right of the x=1 line, so the
    # names need no margin outside the axes and the three columns can share one even gap
    breadth.set_yticks([])
    for i, entry in enumerate(systems):
        breadth.annotate(
            f"{NAMES.get(entry['system'], entry['system'])}  ({entry['d2_obs']:.1f})",
            xy=(1.0, rows_y[i] + 0.24),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=FINE_PT,
            ha="left",
            va="bottom",
            color="0.25",
        )
    breadth.set_xlim(0.85, 1.25)
    breadth.set_ylim(-0.3, len(systems) - 0.4)
    breadth.set_xlabel("Measured / predicted")
    # no key here: the two colors mean what (a) already says they mean

    # `departure` is the difference strip under (b) and shares its axis, so it is not its own panel
    label_panels(left, right, breadth)
    return figure


def main(argv: Iterable[str] | None = None) -> int:
    """Draw the calculator and print what a reader would take off it."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deflections", type=Path, default=Path("results/eq1_mechanistic.json"))
    parser.add_argument("--check", type=Path, default=Path("results/calculator_check.json"))
    parser.add_argument("--fraction", type=float, default=0.5, help="expected fraction the chart assumes")
    parser.add_argument("--out", type=Path, default=Path("docs/manuscript/figures/calculator"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    if len(_TARGETS) > 2:
        raise SystemExit(f"panel (a) tells precision apart by two line styles; {len(_TARGETS)} targets need another")
    if len((args.fraction, RARE)) > len(SERIES):
        raise SystemExit(f"two fractions against {len(SERIES)} validated hues; drop one or facet")

    use_house_style()
    rows = read_converged(args.deflections)
    check = read_check(args.check)
    figure = draw(rows, _TARGETS, (args.fraction, RARE), check)
    written = save(figure, args.out)

    survey = read_survey_deflections(args.deflections)  # all converged pairs; the counting stats span these
    pivot = EXAMPLE
    deflections = [row["d2_marginal"] for row in survey]
    print(f"{len(survey)} surveyed pairs, d^2 {min(deflections):.2f} to {max(deflections):.1f}")
    print(f"  worked example: {pivot[0]}  d'^2 {pivot[1]:.3f}")
    val = Ledger()
    # panel (c) labels each system with its d'^2: print it so the drawn label lands in NUMBERS.txt
    # (the panel-text check's only corpus), and emit a token so prose can cite the same value
    for entry in check["systems"]:
        print(f"  {entry['system']:10s} d'^2 {entry['d2_obs']:.2f}")
        val(f"calc_sys_d2_{entry['system']}", entry["d2_obs"], ".1f")
    val("calc_example_d2", pivot[1], ".2f")
    val("calc_survey_n", len(survey), ",d")  # the full converged survey the distribution stats span
    val("calc_d2_min", min(deflections), ".2f")
    val("calc_d2_max", max(deflections), ".0f")
    for target in _REPORTED:
        even = counts(np.array([pivot[1]]), target, 0.5)[0]
        rare = counts(np.array([pivot[1]]), target, RARE)[0]
        tag = f"{100 * target:g}"
        val(f"calc_even_{tag}pct", even, ",.0f")
        val(f"calc_rare_{tag}pct", rare, ",.0f")
        print(f"    +/-{100 * target:g}%  {even:8.0f} particles at 50/50   {rare:8.0f} at 5%   (the {pivot[0]})")
    for fraction in (args.fraction, RARE):
        tag = "even" if fraction == args.fraction else "rare"
        run = check["sweep"][fraction]
        ratio = run["measured"] / run["predicted"]
        print(
            f"  SpCas9 at p = {fraction:g}: measured/predicted "
            f"{val(f'calc_ratio_small_{tag}', ratio[0], '.2f')} at N = {run['n'][0]:.0f}, "
            f"{val(f'calc_ratio_large_{tag}', ratio[-1], '.2f')} at N = {run['n'][-1]:.0f}"
        )
        across = [entry["by_fraction"][f"{fraction:g}"]["ratio"] for entry in check["systems"]]
        print(
            f"  seven systems at p = {fraction:g}: measured/predicted "
            f"{val(f'calc_systems_lo_{tag}', min(across), '.2f')} to {val(f'calc_systems_hi_{tag}', max(across), '.2f')}"
        )
    even = check["sweep"][args.fraction]
    print(
        f"  sigma_p sits {val('calc_floor_ratio', float(np.median(even['predicted'] / even['floor'])), '.2f')}x above the counting floor"
    )
    print(
        f"  each point pinned to {val('calc_pinned', 100 / np.sqrt(2.0 * (check['trials'] - 1)), '.1f')}% at "
        f"{val('calc_trials', check['trials'], ',d')} experiments"
    )
    for fraction in (args.fraction, RARE):
        above, median = counting_share(survey, fraction)
        tag = "even" if fraction == args.fraction else "rare"
        val(f"calc_counting_dom_{tag}", 100 * above, ".0f")
        val(f"calc_counting_med_{tag}", median, ".3f")
        # flipped framing: a pair is discriminability-limited when counting is the minority of its
        # variance, i.e. d'^2 below 1/[p(1-p)], the deflection where the two terms cross
        threshold = 1.0 / (fraction * (1.0 - fraction))
        disc_limited = float(np.mean([row["d2_marginal"] < threshold for row in survey]))
        val(f"calc_disc_lim_{tag}", 100 * disc_limited, ".0f")
        val(f"calc_disc_threshold_{tag}", threshold, ".0f")
        print(
            f"  at p = {fraction:g}: {100 * above:.0f}% of the {len(survey)} pairs are counting-dominated, "
            f"median share {median:.3f}; {100 * disc_limited:.0f}% discriminability-limited (d'^2 < {threshold:.0f})"
        )
    for fraction in (args.fraction, RARE):  # how far the Fisher and variance-inversion counts diverge
        tag = "even" if fraction == args.fraction else "rare"
        gap = route_gap(np.array(deflections), fraction)
        val(f"calc_route_gap_{tag}", 100 * gap, ".0f")
        print(f"  at p = {fraction:g}: Fisher count is {100 * gap:.0f}% below the variance-inversion count")
    even_2pp = counts(np.array(deflections), 0.02, 0.5)
    val("calc_median_2pp", float(np.median(even_2pp)), ",.0f")
    print(f"  the median surveyed pair takes {np.median(even_2pp):.0f} particles for +/-2% at 50/50")
    val.dump(args.out.parent / "values" / f"{args.out.name}.json")
    print("  wrote " + ", ".join(str(path) for path in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
