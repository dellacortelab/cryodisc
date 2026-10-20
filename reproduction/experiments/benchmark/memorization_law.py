"""Does the self-`Lambda` of a map's own particles fall as `M_eff / N`?
A particle recognizes its 1/N contribution to a map; scored against the OTHER half (disjoint
particles, no true difference) that recognition is the whole signal, so `M_eff / N` means a
reconstruction-template measurement carries a predictable memorization term.
    .venv/bin/python .../memorization_law.py --manifest results/e19_ladder.json --results results --out out.json
The null arm is subtracted, not assumed zero. The exponent is fitted (free slope) before imposed
(fixed at -1). The scorer's fitted amplitude is divided out, since `Lambda_self` is linear in it. The
predicted amplitude `M_eff` is refuted; the geometry that replaces it is the 3D voxel count, from the
`2k/N` per-particle weight (see `band_voxels`). Rationale: docs/rationale/reproduction/experiments/benchmark-and-fetch.md
"""

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from reproduction.analysis.fits import log_curvature as quadratic_term
from reproduction.analysis.fits import weighted_power_law as fit_power_law
from reproduction.predictors.memorization import band_voxels, implied_band
from reproduction.record import payload_of, write_result

DRIVER = "reproduction/experiments/benchmark/memorization_law.py"


def arm(results: Path, tag: str, prefix: str = "e19") -> dict:
    """Mean per-particle `Lambda` and its standard error, from the stored array.

    The array rather than the JSON summary: the fit weights by the standard error.

    Args:
        results: The results directory.
        tag: The run tag, as `_e19_score_arm.sh` wrote it.
        prefix: The ladder's file prefix. A second ladder on a different specimen writes its arms
            under its own prefix, and mixing two specimens' arms into one fit would read as scatter.

    Returns:
        Mean, standard error and count.

    Raises:
        SystemExit: If the array is missing or empty.
    """
    path = results / f"{prefix}_{tag}.npy"
    if not path.exists():
        raise SystemExit(f"{path} is missing; that arm did not score, so the rung cannot be fitted")
    values = np.asarray(np.load(path), dtype=float).ravel()
    if values.size < 2:
        raise SystemExit(f"{path} holds {values.size} particles, too few for a standard error")
    # `Lambda_self` is LINEAR in the scorer's fitted amplitude, but the derivation assumes an
    # unattenuated template, so the amplitude is recorded here as a nuisance factor to divide out
    record = path.with_suffix(".json")
    amplitude, box_size, pixel_size, band_limit = None, None, None, None
    if record.exists():
        document = json.loads(record.read_text())
        payload = payload_of(document, record)
        amplitude = payload.get("amplitude")
        # THE BOX COMES FROM THE ARM, not a flag: both geometry ratios below are cubic-ish in it, so a
        # default right for one specimen is silently wrong for the next
        box_size = payload.get("box_size")
        # the pixel size makes two ladders comparable: `--band-limit` is in cycles per PIXEL, so one
        # nominal band is a different resolution -- and a different `2V` volume -- at each pixel size
        pixel_size = payload.get("pixel_size")
        # the band too, from `arguments` (what the scorer integrated to), since every ratio below is
        # computed at it and a wrong band gives a wrong 2V with nothing to say so
        band_limit = document.get("arguments", {}).get("band_limit")
    return {
        "mean": float(values.mean()),
        "sem": float(values.std(ddof=1) / math.sqrt(values.size)),
        "n": int(values.size),
        "amplitude": amplitude,
        "box_size": None if box_size is None else int(box_size),
        "pixel_size": None if pixel_size is None else float(pixel_size),
        "band_limit": None if band_limit is None else float(band_limit),
    }


def independent_coefficients(box_size: int, band_limit: float) -> float:
    """How many independent Fourier coefficients one band-limited projection carries.

    The disc of radius `band_limit * box_size`, halved for Friedel symmetry. This is the amplitude
    the prediction gave, and it is REFUTED; `band_voxels` is the quantity that fits. It is kept
    because the ratio between the predicted and the fitted amplitude is the finding.

    Args:
        box_size: Image size in pixels.
        band_limit: Cutoff in cycles per pixel.

    Returns:
        The coefficient count.
    """
    radius = band_limit * box_size
    return math.pi * radius**2 / 2.0


def from_the_arms(
    rows: list[dict], field: str, override: float | None, flag: str, required: bool = True
) -> float | None:
    """One scalar the arms recorded, refusing rather than averaging if they disagree.

    Two boxes or two pixel sizes in one ladder is two specimens or two extractions, and either way
    one geometry cannot describe both. BOTH arms are read, not just `self`: they are two halves of
    one particle set through one scorer, so a disagreement is a mispaired pool -- exactly the error
    this exists to catch, and the one it would miss looking at half the records.

    Args:
        rows: The rungs, each with a `self` and a `null` arm.
        field: The key on an arm, e.g. `box_size`.
        override: An explicit value from the command line, or None.
        flag: The flag `override` came from, for the message.
        required: Whether a ladder with no record of the field is an error.

    Returns:
        The resolved value, or None when it is optional and nothing carries it.

    Raises:
        SystemExit: If the arms disagree, if an override contradicts them, or if a required field is
            recorded nowhere.
    """
    found = sorted({arm[field] for row in rows for arm in (row["self"], row["null"]) if arm.get(field)})
    if len(found) > 1:
        raise SystemExit(f"the arms recorded {field} as {found}; one ladder cannot carry two")
    if override is not None:
        # EXACT for the box (an int), TOLERANT for the floats: they round-trip through float32, so an
        # exact `!=` would refuse the very pixel size anyone would type
        exact = isinstance(override, int) and not isinstance(override, bool)
        differs = found and (found[0] != override if exact else not math.isclose(found[0], override, rel_tol=1e-6))
        if differs:
            raise SystemExit(
                f"{flag} {override} contradicts the arms, which recorded {found[0]}. The geometry "
                "ratios are cubic-ish in these, so this is not a rounding difference"
            )
        return override
    if found:
        return found[0]
    if required:
        raise SystemExit(
            f"no arm record carries a {field}, so the geometry ratios cannot be computed. Pass "
            f"{flag} explicitly, having checked what the arms actually scored at"
        )
    return None


def collect(manifest: dict, results: Path, prefix: str = "e19") -> list[dict]:
    """Read both arms of every rung the manifest lists.

    Args:
        manifest: The ladder manifest's result block.
        results: The results directory.
        prefix: The ladder's file prefix, so two specimens' ladders cannot be read into one fit.

    Returns:
        One row per rung, self and null and their difference.
    """
    rows = []
    for row in manifest["rungs"]:
        n = int(row["n_particles"])
        self_arm, null_arm = (arm(results, f"self_n{n}", prefix), arm(results, f"null_n{n}", prefix))
        # signed, so the labeling check below can read it; the law is fitted to the magnitude
        difference = self_arm["mean"] - null_arm["mean"]
        rows.append(
            {
                "n_particles": n,
                "n_half_a": row.get("n_half_a"),
                "self": self_arm,
                "null": null_arm,
                "difference": difference,
                # independent arms, so their errors add in quadrature
                "difference_sem": math.hypot(self_arm["sem"], null_arm["sem"]),
            }
        )
    return sorted(rows, key=lambda r: r["n_particles"])


def main(argv: Iterable[str] | None = None) -> int:
    """Fit the law and print the ladder."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path, help="results/e19_ladder.json")
    parser.add_argument("--results", default=Path("results"), type=Path, help="where the score arrays are")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--box-size",
        type=int,
        default=None,
        help="the extraction box the rungs scored at. Defaults to what the ARM RECORDS say, which is "
        "the only source that cannot disagree with the measurement; pass it only to override, and a "
        "value contradicting the records is refused",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="A/px of the extraction, defaulted from the arm records the same way as --box-size. It "
        "converts the band from cycles/px to Angstroms, which is the only form two ladders on "
        "different extractions can be compared in",
    )
    parser.add_argument(
        "--band-limit",
        type=float,
        default=None,
        help="cycles/px the score integrated to, defaulted from the arm records like --box-size",
    )
    parser.add_argument("--prefix", default="e19", help="file prefix the ladder's arms were written under")
    # THE FIT RANGE, since -1 is ASYMPTOTIC: a single power law over every rung absorbs the low-N
    # curvature into the slope, so fitting the asymptote earns the prediction. 0 keeps every rung.
    # DRAW STRUCTURE is a fit diagnostic: `choice(replace=False)` returns PREFIXES above `pop_size/50`,
    # so same-seed rungs nest; fitting the two sides apart exposes it (the nested side's chi2/dof is low)
    parser.add_argument(
        "--split-at",
        type=float,
        default=None,
        help="also fit the rungs below and at-or-above this particle count as separate blocks, and "
        "report both. Pass the draw's nesting threshold (numpy's is `pop_size / 50`) to test whether "
        "rungs that share particles are holding the fit together",
    )
    parser.add_argument(
        "--fit-min",
        type=float,
        default=0.0,
        help="fit the law on rungs of at least this many particles; 0 uses every rung",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifest = json.loads(args.manifest.read_text())
    manifest = payload_of(manifest)
    rows = collect(manifest, args.results, args.prefix)
    n = np.array([row["n_particles"] for row in rows], dtype=float)

    print(f"{'N':>8}{'self':>12}{'null':>12}{'self-null':>12}{'+/-':>10}")
    for row in rows:
        print(
            f"{row['n_particles']:>8}{row['self']['mean']:>12.4f}{row['null']['mean']:>12.4f}"
            f"{row['difference']:>12.4f}{row['difference_sem']:>10.4f}"
        )

    differences = np.array([row["difference"] for row in rows])
    # THE SIGN IS A CHECK: half A's particles must prefer map A, but the fit is on the magnitude, so
    # an inverted split-to-half-map labeling would fit the law perfectly and hide itself
    signs = {"positive": int((differences > 0).sum()), "negative": int((differences < 0).sum())}
    if signs["negative"] == len(rows):
        print(
            f"\n!! all {len(rows)} rungs are NEGATIVE. Half A's particles should prefer map A, so "
            "this is the split-to-half-map labeling inverted, not a negative memorization term. "
            "The magnitudes below are still the law; the pairing needs fixing before they are quoted."
        )
    elif signs["negative"]:
        print(f"\n!! {signs['negative']} of {len(rows)} rungs are negative while the rest are not")

    signal = np.abs(differences)
    sigma = np.array([row["difference_sem"] for row in rows])
    # EVERY rung is kept in the record and drawn; `--fit-min` restricts only what the LAW is fitted
    # to, so a reader can distinguish a restricted fit from a short ladder
    fit_on = n >= args.fit_min
    if int(fit_on.sum()) < 3:
        raise SystemExit(
            f"--fit-min {args.fit_min:,.0f} leaves {int(fit_on.sum())} of {n.size} rungs; a free-slope fit needs three"
        )
    if args.fit_min > 0:
        print(
            f"\n== fitting the law on {int(fit_on.sum())} of {n.size} rungs, N >= {args.fit_min:,.0f}"
            f" (drawn: all {n.size})"
        )
    free = fit_power_law(n[fit_on], signal[fit_on], sigma[fit_on])
    fixed = fit_power_law(n[fit_on], signal[fit_on], sigma[fit_on], slope=-1.0)
    curvature = quadratic_term(n[fit_on], signal[fit_on], sigma[fit_on])
    blocks = None
    if args.split_at is not None:
        below, above = n < args.split_at, n >= args.split_at
        if below.sum() < 3 or above.sum() < 3:
            raise SystemExit(
                f"--split-at {args.split_at:,.0f} leaves {int(below.sum())} rung(s) below and "
                f"{int(above.sum())} at or above; a free-slope fit on each side needs three"
            )
        blocks = {
            "threshold": float(args.split_at),
            "below": fit_power_law(n[below], signal[below], sigma[below]),
            "at_or_above": fit_power_law(n[above], signal[above], sigma[above]),
        }
        gap = blocks["below"]["slope"] - blocks["at_or_above"]["slope"]
        spread = math.hypot(blocks["below"]["slope_sem"], blocks["at_or_above"]["slope_sem"])
        blocks["slope_gap"] = float(gap)
        blocks["slope_gap_sigma"] = float(abs(gap) / spread) if spread > 0 else None

    amplitudes = [row["self"]["amplitude"] for row in rows]
    deattenuated = None
    if all(a for a in amplitudes):
        scale = np.array(amplitudes, dtype=float)
        deattenuated = {
            "fitted_amplitude": scale.tolist(),
            "free": fit_power_law(n[fit_on], (signal / scale)[fit_on], (sigma / scale)[fit_on]),
            "fixed": fit_power_law(n[fit_on], (signal / scale)[fit_on], (sigma / scale)[fit_on], slope=-1.0),
        }
    box_size = from_the_arms(rows, "box_size", args.box_size, "--box-size")
    # Optional, unlike the box: only the Angstrom conversion needs it, so a ladder whose arms predate
    # the field still fits, and reports its band in cycles per pixel alone.
    pixel_size = from_the_arms(rows, "pixel_size", args.pixel_size, "--pixel-size", required=False)
    # Required, unlike the pixel size: every geometry number below is computed at this band, so a
    # ladder whose arms do not record one has to be told explicitly rather than given a default.
    band_limit = from_the_arms(rows, "band_limit", args.band_limit, "--band-limit")

    m_eff = independent_coefficients(box_size, band_limit)
    # The rungs are fitted against the TOTAL particle count and the law is per half, so the
    # predicted amplitude is 2V, not V.
    voxels = band_voxels(box_size, band_limit)

    span = math.log10(n[fit_on].max() / n[fit_on].min())
    print(
        f"\nfitted on {int(fit_on.sum())} of {len(rows)} rungs, {n[fit_on].min():,.0f} to "
        f"{n[fit_on].max():,.0f} particles = {span:.2f} decades"
    )
    print(f"free slope      {free['slope']:+.4f} +/- {free['slope_sem']:.4f}   (M_eff/N predicts -1)")
    print(
        f"  chi2/dof {free['chi_square_per_dof']:.2f} on {free['dof']} dof, "
        f"{abs(free['slope'] + 1.0) / free['slope_sem']:.1f} sigma from -1"
    )
    if blocks is not None:
        for name, label in (("below", "below"), ("at_or_above", "at or above")):
            block = blocks[name]
            print(
                f"  {label:>11} {args.split_at:,.0f}: slope {block['slope']:+.4f} +/-"
                f" {block['slope_sem']:.4f} on {block['dof'] + 2} rungs, chi2/dof"
                f" {block['chi_square_per_dof']:.2f}"
            )
        print(
            f"  the two blocks differ by {blocks['slope_gap']:+.4f} = {blocks['slope_gap_sigma']:.1f}"
            " sigma  <- a chi2/dof far below 1 on one side means its rungs share particles"
        )
    if curvature is not None:
        print(
            f"  curvature in log n {curvature['curvature']:+.4f} +/- {curvature['curvature_sem']:.4f}"
            f" = {curvature['sigma']:.1f} sigma  <- a straight line is the right shape below ~2 sigma"
        )
    print(f"amplitude at -1 {fixed['amplitude']:.1f} +/- {fixed['amplitude_sem']:.1f} nats x particles")
    print(f"M_eff from geometry, box {box_size} band {band_limit} cyc/px: {m_eff:.1f}")
    print(f"ratio amplitude / M_eff  {fixed['amplitude'] / m_eff:.4f}   <- predicted, refuted")
    print(f"2 x band voxels          {2 * voxels:.1f}")
    print(f"ratio amplitude / 2V     {fixed['amplitude'] / (2 * voxels):.4f}   <- slice sampling")
    # inverting the ratio says what band the amplitude corresponds to, the form that travels between
    # ladders at different pixel sizes (the ratio itself only compares ladders sharing a pixel size)
    band = implied_band(box_size, fixed["amplitude"])
    # carried through from the amplitude's own error, since the band is what gets quoted and a quoted
    # number without an error bar is not a measurement
    edges = sorted(
        implied_band(box_size, fixed["amplitude"] + s) for s in (-fixed["amplitude_sem"], fixed["amplitude_sem"])
    )
    band_sem = (edges[1] - edges[0]) / 2.0
    scored_a = None if pixel_size is None else pixel_size / band_limit
    implied_a = None if pixel_size is None else pixel_size / band
    implied_a_sem = None if pixel_size is None else pixel_size * band_sem / band**2
    print(
        f"band the amplitude implies {band:.4f} +/- {band_sem:.4f} cyc/px against the {band_limit:.4f} scored"
        + (
            ""
            if implied_a is None
            else f" -- {implied_a:.2f} +/- {implied_a_sem:.2f} A against {scored_a:.2f} A at {pixel_size:.3f} A/px"
        )
    )
    if deattenuated:
        # the self arm calibrates on the pool it measures, so its amplitude carries the memorization;
        # rescaling to the held-out fit is NOT the repair (it attenuates an unattenuated template)
        print(
            f"\nfitted amplitude {', '.join(f'{a:.3f}' for a in deattenuated['fitted_amplitude'])}"
            " (self arm, so memorization-contaminated)"
        )
        print(
            f"with it divided out: free slope {deattenuated['free']['slope']:+.4f}"
            f" +/- {deattenuated['free']['slope_sem']:.4f}, "
            f"amplitude at -1 {deattenuated['fixed']['amplitude']:,.0f}"
            f" = {deattenuated['fixed']['amplitude'] / (2 * voxels):.4f} x 2V"
        )
    else:
        print("\n!! no fitted amplitude beside the arrays, so the scorer's attenuation is left in")

    write_result(
        args.out,
        {
            "rungs": rows,
            # WHICH RUNGS THE FIT SAW, since every downstream number is conditional on the window and
            # `rungs` holds every rung while the fit may hold fewer
            "fit_min_particles": float(args.fit_min),
            "fit_rungs_used": int(fit_on.sum()),
            "fit_rungs_total": int(n.size),
            "fit_span_decades": span,
            "fit_curvature": curvature,
            "fit_blocks": blocks,
            "fit_free_slope": free,
            "fit_slope_fixed_at_minus_one": fixed,
            "m_eff_geometric": m_eff,
            "amplitude_over_m_eff": fixed["amplitude"] / m_eff,
            "band_voxels": voxels,
            "amplitude_over_two_band_voxels": fixed["amplitude"] / (2 * voxels),
            "implied_band_cycles_per_pixel": band,
            "implied_band_cycles_per_pixel_sem": band_sem,
            "implied_band_angstrom": implied_a,
            "implied_band_angstrom_sem": implied_a_sem,
            "scored_band_angstrom": scored_a,
            "pixel_size": pixel_size,
            "fit_with_scorer_amplitude_removed": deattenuated,
            "rung_signs": signs,
        },
        driver=DRIVER,
        # `box_size` RESOLVED, not as parsed: it defaults to None and is read off the arms, and a
        # recorded None would leave the geometry ratios above unattributable to any box.
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        | {"box_size": box_size, "pixel_size": pixel_size, "band_limit": band_limit},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
