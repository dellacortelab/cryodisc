"""Microscope hours against achievable `sigma_p`, and the two things that flatten the curve.

    .venv/bin/python reproduction/experiments/benchmark/decision_curve.py --out results/decision_curve.json
With `sigma_p^2 = [ p(1-p) + m/d^2 ] / N`, the counting floor `p(1-p)` is immune to specimen quality:
a saturated row cannot be helped by a better grid. The penalty `m = exp(Delta_B / (2 d^2))` depends on
WHERE the state difference lives, NOT the map resolution, and `check_published` asserts it reproduces
the published multipliers. Uncertainty bands are quoted rather than derived, with three spans on the
same chain printed side by side; the driver never prints hours as a single number.
Rationale: docs/rationale/reproduction/experiments/decision_curve.md
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from reproduction.record import payload_of, write_result

DRIVER = "reproduction/experiments/benchmark/decision_curve.py"

THROUGHPUT = (
    ("micrographs/hour (Krios, AFIS)", 400.0, 250.0, 750.0),
    ("usable micrograph fraction", 0.75, 0.65, 0.85),
    ("particles picked per micrograph", 200.0, 50.0, 600.0),
    ("usable particle fraction", 0.33, 0.15, 0.55),
)
"""The throughput chain, `(link, central, low, high)`. Sources in the program's Section 8."""

QUOTED_RANGE = (7.0e3, 6.0e4)
"""The plan's assembled usable-particles-per-hour range, which is what the hours band is taken from.

Narrower than the product of the links' ranges because the links are not independent: a grid that
yields 600 particles per micrograph is not also the grid that fails 35% of its micrographs. Kept as a
constant rather than derived so that the disagreement with the product is visible instead of resolved
by whichever combination rule the code happened to use.
"""

QUALITIES = (
    ("HexAuFoil", -54.0, 10.0, "Naydenova, Jia & Russo, Science 370:223 (2020); Delta-B = 54 +/- 10 A^2"),
    ("baseline", 0.0, 0.0, "whatever grid the measurement was made on"),
    (
        "thinner ice (filtered)",
        -12.8,
        0.0,
        "Neselu et al., JSB:X 7:100085 (2023): 0.128 A^2/nm filtered, R^2 = 0.988; 100 nm thinner, e.g. 150 -> 50 nm",
    ),
    ("+100 nm ice (unfiltered)", 21.0, 0.0, "the same source's unfiltered slope, 0.21 A^2/nm, R^2 = 0.94"),
)
"""Named specimen qualities as `(name, Delta_B, Delta_B error, source)`. Negative is an improvement.

The third field is the published one-sigma uncertainty on Delta_B, which `m = exp(Delta_B / 2 d^2)`
amplifies exponentially, so a figure drawn at the central value alone would overstate the precision.
The ice rows carry 0.0 because their sources quote a slope and R^2 rather than an interval on Delta_B;
that is a gap in what can be propagated, not a claim that they are exact."""

PUBLISHED = (
    (54.0, 4.0, 5.4),
    (54.0, 3.0, 20.0),
    (12.8, 3.0, 2.0),
    (12.8, 2.0, 4.9),
)
"""`(Delta_B, spacing, published multiplier)` for `check_published`."""


def particles_per_hour(links=THROUGHPUT, quoted=QUOTED_RANGE) -> dict:
    """The throughput chain multiplied out, with all three readings of its spread.

    Args:
        links: `(name, central, low, high)` per link.
        quoted: The plan's assembled `(low, high)`, used for `low`/`high`.

    Returns:
        `central`; `low`/`high` from `quoted` and `span` their ratio, which is what the hours band
        uses; `extremes_low`/`extremes_high`/`extremes_span` from the product of the links' ranges;
        and `quadrature_span`, the log-spans added as if the links were independent.
    """
    central, low, high, log_variance = 1.0, 1.0, 1.0, 0.0
    for _, mid, small, large in links:
        central *= mid
        low *= small
        high *= large
        log_variance += math.log(large / small) ** 2
    return {
        "central": central,
        "low": quoted[0],
        "high": quoted[1],
        "span": quoted[1] / quoted[0],
        "extremes_low": low,
        "extremes_high": high,
        "extremes_span": high / low,
        "quadrature_span": math.exp(math.sqrt(log_variance)),
        "links": {name: {"central": mid, "low": small, "high": large} for name, mid, small, large in links},
    }


def damping_multiplier(delta_b: float, spacing: float) -> float:
    """What a B-factor change costs `d^2` for a difference living near one spatial frequency.

    `exp(-B k^2 / 4)` on amplitude is `exp(-B k^2 / 2)` on `d^2`, evaluated at `k = 1/spacing`.

    Args:
        delta_b: Change in B relative to baseline, in A^2. Negative improves.
        spacing: The spacing, in A, at which the state difference's signal sits. NOT the resolution
            of the reconstruction -- a 15 degree quaternary rotation is a low-resolution difference
            in a 3 A map, and using the map's resolution here overstates the penalty enormously.

    Returns:
        The factor `d^2` is divided by, so above 1 is a penalty.
    """
    return math.exp(delta_b / (2.0 * spacing**2))


def check_published(published=PUBLISHED, tolerance: float = 0.03) -> list[dict]:
    """Verify `damping_multiplier` against the multipliers the literature quotes.

    Args:
        published: `(delta_b, spacing, expected)` triples.
        tolerance: Allowed fractional disagreement.

    Returns:
        One row per check with `derived`, `published` and `ratio`.

    Raises:
        SystemExit: On any disagreement beyond `tolerance`. The expression is the whole basis of the
            specimen-quality family, so a mismatch means the family is wrong and printing it would be
            worse than not producing it.
    """
    rows = []
    for delta_b, spacing, expected in published:
        derived = damping_multiplier(delta_b, spacing)
        ratio = derived / expected
        rows.append({"delta_b": delta_b, "spacing": spacing, "published": expected, "derived": derived, "ratio": ratio})
        if abs(ratio - 1.0) > tolerance:
            raise SystemExit(
                f"damping_multiplier({delta_b}, {spacing}) = {derived:.3f} against a published "
                f"{expected:.3f}: the exp(Delta_B / 2 d^2) reading of the quoted gains is wrong"
            )
    return rows


def variance_terms(fraction: float, d2: float, multiplier: float = 1.0) -> dict:
    """The two terms of `N sigma_p^2`, kept apart because they respond to different interventions.

    Args:
        fraction: Population fraction `p`.
        d2: Per-particle discriminability, so the discriminability term is `multiplier / d2`.
        multiplier: The specimen-quality penalty `m` from `damping_multiplier`.

    Returns:
        `counting` = `p(1-p)`, `discriminability` = `m/d2`, their `total`, and the `floor_share`.
        `I_p / cap` is that same ratio, so it is not returned again under the name `saturation`.
    """
    counting = fraction * (1.0 - fraction)
    discriminability = multiplier / d2
    total = counting + discriminability
    return {
        "counting": counting,
        "discriminability": discriminability,
        "total": total,
        "floor_share": counting / total,
    }


def purchasing_power(*, fraction: float, d2: float, delta_b: float, spacing: float) -> dict:
    """What a specimen-quality change is worth in hours, and how much of it each damper takes.

    Args:
        fraction: Population fraction `p`.
        d2: Per-particle discriminability at baseline.
        delta_b: The B-factor change on offer, negative for an improvement.
        spacing: Where the state difference's signal sits, in A.

    Returns:
        `multiplier` (the raw effect on `d^2`), `gain` (the ratio of required hours, which is what the
        change actually buys), and `lost_to_counting`, the fraction of the raw effect the counting
        floor absorbs.
    """
    multiplier = damping_multiplier(delta_b, spacing)
    baseline = variance_terms(fraction, d2, 1.0)["total"]
    improved = variance_terms(fraction, d2, multiplier)["total"]
    gain = baseline / improved
    raw = 1.0 / multiplier
    return {
        "multiplier": multiplier,
        "gain": gain,
        "raw_gain_on_d2": raw,
        "lost_to_counting": 1.0 - (gain - 1.0) / (raw - 1.0) if abs(raw - 1.0) > 1e-12 else 0.0,
    }


def purchasing_band(*, fraction: float, d2: float, delta_b: float, delta_b_err: float, spacing: float) -> dict:
    """`purchasing_power` at the central `Delta_B` and at one sigma either side.

    The band is NOT symmetric and cannot be summarized by a single +/-: `m = exp(Delta_B / 2 d^2)` is
    exponential in `Delta_B`, so both edges are returned.

    Args:
        fraction: Population fraction `p`.
        d2: Per-particle discriminability at baseline.
        delta_b: The central B-factor change, negative for an improvement.
        delta_b_err: Its published one-sigma uncertainty. Zero yields a zero-width band.
        spacing: Where the state difference's signal sits, in A.

    Returns:
        The central row, plus `gain_low`/`gain_high` and `raw_low`/`raw_high` ordered so low <= high.
    """
    central = purchasing_power(fraction=fraction, d2=d2, delta_b=delta_b, spacing=spacing)
    if delta_b_err <= 0.0:
        return {
            **central,
            "gain_low": central["gain"],
            "gain_high": central["gain"],
            "raw_low": central["raw_gain_on_d2"],
            "raw_high": central["raw_gain_on_d2"],
        }
    # a LARGER |Delta_B| in the improving direction is a bigger gain, so the sign decides which edge is
    # which; sorting rather than assuming keeps this right for the ice rows too
    edges = [
        purchasing_power(fraction=fraction, d2=d2, delta_b=delta_b + sign * delta_b_err, spacing=spacing)
        for sign in (-1.0, 1.0)
    ]
    gains = sorted(edge["gain"] for edge in edges)
    raws = sorted(edge["raw_gain_on_d2"] for edge in edges)
    return {**central, "gain_low": gains[0], "gain_high": gains[1], "raw_low": raws[0], "raw_high": raws[1]}


def survey_spectrum_integral(
    spectrum_path: Path, survey_path: Path, *, fraction: float, d2: float, qualities=QUALITIES
) -> dict:
    """Each improvement's multiplier integrated over the survey pairs' whitened difference spectra.

    Replaces evaluating the multiplier at a single representative spacing. The raw effect on `d^2` is
    `exp(-Delta_B k^2 / 2)` per shell, convex in `k`, so a pair's gain is the spectrum-weighted average
    over its shells and exceeds the value at the pair's median spacing. The weight is `power / sigma^2`,
    the same per-shell contribution `d^2` is the sum of, so the integral is exactly
    `d^2_improved / d^2_baseline` under the reference plan the spectra were rendered on.

    Args:
        spectrum_path: A `difference_spectrum.py` record, per-pair `power`, the shell `1/sigma^2`
            weight, the shell frequencies and the pixel size.
        survey_path: An `eq1_mechanistic.py` record whose rows name the admitted survey pairs.
        fraction: Population fraction `p`.
        d2: Per-particle discriminability at baseline, held fixed so only the spacing-to-spectrum change
            moves the number.
        qualities: `(name, Delta_B, err, source)` rows; the baseline row is skipped.

    Returns:
        `n_pairs`, the admitted pairs' `spacing_median_a`, the input provenance, and one `per_quality`
        row per improvement carrying the integrated multiplier and realized hours gain (median and mean
        over pairs) beside the single-median-spacing reading they replace.

    Raises:
        SystemExit: If no admitted pair carries a usable spectrum.
    """
    spectrum = json.loads(Path(spectrum_path).read_text())
    survey = json.loads(Path(survey_path).read_text())
    pixel_size = float(spectrum["arguments"]["pixel_size"])
    body = payload_of(spectrum)
    weight = np.asarray(body["shell_weight"], float)
    wavenumber = np.asarray(body["frequency_cyc_px"], float) / pixel_size
    admitted = {row["name"] for row in payload_of(survey)["rows"]}

    contributions, spacings = [], []
    for row in body["rows"]:
        if row["name"] not in admitted:
            continue
        defocus = row["defocus"][0]
        contribution = np.asarray(defocus["power"], float) * weight
        spacing = defocus.get("half_power_spacing_a")
        if contribution.sum() <= 0 or spacing is None or not np.isfinite(spacing):
            continue
        contributions.append(contribution)
        spacings.append(float(spacing))
    if not contributions:
        raise SystemExit(f"{spectrum_path} carries no admitted pair with a usable spectrum")
    spacing_median = float(np.median(spacings))
    baseline_total = variance_terms(fraction, d2, 1.0)["total"]

    per_quality = []
    for name, delta_b, _, _ in qualities:
        if name == "baseline":
            continue
        raw_shell = np.exp(-delta_b * wavenumber**2 / 2.0)  # factor on d^2 per shell, >1 for improvement
        raw = np.array([float((c * raw_shell).sum() / c.sum()) for c in contributions])
        gain = np.array([baseline_total / variance_terms(fraction, d2, 1.0 / r)["total"] for r in raw])
        at_median = purchasing_power(fraction=fraction, d2=d2, delta_b=delta_b, spacing=spacing_median)
        per_quality.append(
            {
                "name": name,
                "delta_b": delta_b,
                "raw_integrated_median": float(np.median(raw)),
                "raw_integrated_mean": float(np.mean(raw)),
                "gain_integrated_median": float(np.median(gain)),
                "gain_integrated_mean": float(np.mean(gain)),
                "raw_at_median_spacing": at_median["raw_gain_on_d2"],
                "gain_at_median_spacing": at_median["gain"],
            }
        )
    return {
        "n_pairs": len(contributions),
        "spacing_median_a": spacing_median,
        "pixel_size_a": pixel_size,
        "d2": d2,
        "fraction": fraction,
        "spectrum_source": str(spectrum_path),
        "spectrum_commit": spectrum.get("commit"),
        "survey_source": str(survey_path),
        "survey_commit": survey.get("commit"),
        "per_quality": per_quality,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line. Every path is an argument; nothing is read from a constant."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("results/decision_curve.json"))
    parser.add_argument(
        "--spectrum",
        type=Path,
        default=Path("results/difference_spectrum.json"),
        help="per-pair whitened difference spectra, over which each improvement's multiplier is "
        "integrated instead of read off at a single representative spacing",
    )
    parser.add_argument(
        "--survey",
        type=Path,
        default=Path("results/eq1_mechanistic.json"),
        help="the admitted survey pairs, so the integral describes the same population the paper surveys",
    )
    parser.add_argument(
        "--d2",
        type=float,
        nargs="+",
        default=(2.906,),
        help="per-particle discriminability to draw the hours curves for. The default is the real "
        "SpCas9 6nt/12nt pair's d'^2_obs, a low-discriminability pair where a specimen helps most",
    )
    parser.add_argument("--fraction", type=float, default=0.5, help="population fraction p")
    parser.add_argument(
        "--spacings",
        type=float,
        nargs="+",
        default=(2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 20.0, 30.0, 40.0),
        help="spacings, in A, at which a state difference might sit. A quaternary rotation is at the "
        "large end and a side-chain rearrangement at the small end; this axis is what decides whether "
        "grid technology is worth anything for discrimination",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    throughput = particles_per_hour()
    checks = check_published()

    print("throughput chain")
    for name, mid, low, high in THROUGHPUT:
        print(f"  {name:<38} {mid:>8.3g}   [{low:g}, {high:g}]")
    print(
        f"  {'= usable particles per Krios hour':<38} {throughput['central']:>8.3g}   "
        f"[{throughput['low']:.3g}, {throughput['high']:.3g}]  span {throughput['span']:.1f}x (quoted)"
    )
    print(
        f"  spans: {throughput['span']:.1f}x quoted, {throughput['quadrature_span']:.1f}x in quadrature, "
        f"{throughput['extremes_span']:.1f}x with every link at its extreme "
        f"[{throughput['extremes_low']:.3g}, {throughput['extremes_high']:.3g}]"
    )

    print("\nexp(Delta_B / 2 d^2) against the published multipliers")
    print(f"  {'Delta_B':>8} {'spacing':>8} {'published':>10} {'derived':>9} {'ratio':>7}")
    for row in checks:
        print(
            f"  {row['delta_b']:8.1f} {row['spacing']:8.1f} {row['published']:10.2f} "
            f"{row['derived']:9.3f} {row['ratio']:7.3f}"
        )

    print("\nwhat a grid improvement buys, against where the state difference sits")
    print(f"  {'spacing':>8} " + " ".join(f"{name:>13}" for name, _, _, _ in QUALITIES if name != "baseline"))
    power = []
    for spacing in args.spacings:
        cells = []
        for name, delta_b, err, _ in QUALITIES:
            if name == "baseline":
                continue
            got = purchasing_band(
                fraction=args.fraction, d2=args.d2[0], delta_b=delta_b, delta_b_err=err, spacing=spacing
            )
            power.append({"spacing": spacing, "quality": name, "d2": args.d2[0], **got})
            cells.append(f"{got['gain']:>13.2f}")
        print(f"  {spacing:8.1f} " + " ".join(cells))
    print(
        f"  gain = ratio of required hours at d^2 = {args.d2[0]:g}, p = {args.fraction:g}. Above 1 the "
        "quality change helps; HexAuFoil is negative Delta_B and the ice rows are positive, so their "
        "columns run opposite ways"
    )

    integral = survey_spectrum_integral(args.spectrum, args.survey, fraction=args.fraction, d2=args.d2[0])
    print(
        f"\nmultiplier integrated over {integral['n_pairs']:,} survey pairs' whitened difference spectra "
        f"(median spacing {integral['spacing_median_a']:.1f} A)"
    )
    print(f"  {'quality':>24} {'raw@med-spc':>12} {'hrs@med-spc':>12} {'raw integ.':>12} {'hrs integ.':>12}")
    for entry in integral["per_quality"]:
        print(
            f"  {entry['name']:>24} {entry['raw_at_median_spacing']:>12.3f} {entry['gain_at_median_spacing']:>12.3f} "
            f"{entry['raw_integrated_median']:>12.3f} {entry['gain_integrated_median']:>12.3f}"
        )

    write_result(
        args.out,
        {
            "throughput": throughput,
            "published_checks": checks,
            "purchasing_power": power,
            "qualities": [{"name": n, "delta_b": b, "delta_b_err": e, "source": s} for n, b, e, s in QUALITIES],
            "survey_integral": integral,
        },
        driver=DRIVER,
        arguments=vars(args),
    )
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
