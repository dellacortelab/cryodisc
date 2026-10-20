"""Combine replicate memorization ladders into one fit with a MEASURED error bar.

    .venv/bin/python .../memorization_replicates.py --laws 'results/e20r*_memorization_law.json' --out out.json
A single ladder's per-particle error treats the map as fixed, omitting the between-draw term: a
different draw of `N` particles builds a different map. This replaces it with the empirical spread of
`R` independent draws per rung, and records `inflation`, the empirical error over the per-particle
one. Rungs with one draw get the per-particle error scaled by the median inflation, or
`--drop-singletons` leaves them out. `mean_overlap` is what a reader discounts the top rungs' bars by.
Rationale: docs/rationale/figures.md
"""

import argparse
import glob
import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from reproduction.analysis.fits import log_curvature, weighted_power_law
from reproduction.record import payload_of, write_result

DRIVER = "reproduction/experiments/benchmark/memorization_replicates.py"


def rungs_of(path: Path) -> dict[int, dict]:
    """Read one ladder's rungs, keyed by particle count.

    Args:
        path: A `memorization_law.py` record.

    Returns:
        Rung size to its `difference`, `difference_sem`, held-out `null` arm, and source.

    Raises:
        SystemExit: If the record carries no rungs.
    """
    record = json.loads(Path(path).read_text())
    result = payload_of(record)
    rows = result.get("rungs", [])
    if not rows:
        raise SystemExit(f"{path} carries no rungs")
    return {
        int(row["n_particles"]): {
            "difference": float(row["difference"]),
            "difference_sem": float(row["difference_sem"]),
            "null": float(row["null"]["mean"]),
            "null_sem": float(row["null"]["sem"]),
            "source": str(path),
        }
        for row in rows
    }


def combine(ladders: list[dict[int, dict]]) -> list[dict]:
    """Average the replicates at each rung and measure their spread.

    Args:
        ladders: One mapping per replicate, from `rungs_of`.

    Returns:
        One row per rung, ordered by particle count.
    """
    sizes = sorted({n for ladder in ladders for n in ladder})
    rows = []
    for n in sizes:
        draws = [ladder[n] for ladder in ladders if n in ladder]
        values = np.array([d["difference"] for d in draws], dtype=float)
        # THE PER-PARTICLE ERRORS COMBINE AS A MEAN'S, not as a pooled sample: each replicate's own
        # sem describes its own mean, so the mean of R of them has sqrt(sum of squares) / R.
        per_particle = float(np.hypot.reduce([d["difference_sem"] for d in draws]) / len(draws))
        empirical = float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else None
        rows.append(
            {
                "n_particles": n,
                "n_draws": int(values.size),
                "mean": float(values.mean()),
                "draws": values.tolist(),
                "sem_empirical": empirical,
                "sem_per_particle": per_particle,
                "inflation": None if empirical is None else float(empirical / per_particle),
                # the null arm, averaged the same way, so the control panel can show the same rungs
                # the law panel does
                "null_mean": float(np.mean([d["null"] for d in draws])),
                # across draws where there is more than one, else the per-particle error as a mean's:
                # the reserve is the SAME particles every draw, so only the across-draw spread sees it
                "null_sem": (
                    float(np.std([d["null"] for d in draws], ddof=1) / math.sqrt(len(draws)))
                    if len(draws) > 1
                    else float(np.hypot.reduce([d["null_sem"] for d in draws]) / len(draws))
                ),
                "sources": [d["source"] for d in draws],
            }
        )
    return rows


def main(argv: Iterable[str] | None = None) -> int:
    """Combine the replicates, fit, and report the inflation over the single-ladder error."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--laws", nargs="+", required=True, help="the replicate law records, or a glob")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--pool",
        type=float,
        default=1_539_017.0,
        help="particles available to draw from, for the expected overlap between two replicates of one rung",
    )
    parser.add_argument(
        "--fit-min",
        type=float,
        default=0.0,
        help="fit only rungs at or above this particle count. The -1 is asymptotic, so the window is "
        "part of the measurement, and it is DECLARED rather than selected: the manuscript fits the "
        "range real reconstructions are built in, the bottom of `memorization.TYPICAL_DATASET`. "
        "Selecting it instead by scanning for the smallest window whose quadratic term falls below "
        "2 sigma picks the window where curvature is hardest to see, since that term's error grows "
        "as the window narrows",
    )
    parser.add_argument(
        "--drop-singletons",
        action="store_true",
        help="leave rungs with one draw out of the fit rather than giving them the per-particle "
        "error scaled by the inflation measured elsewhere",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    # a glob that the shell did not expand is still a glob, and one unexpanded pattern silently
    # fitting a single ladder is the failure this campaign exists to prevent
    paths = sorted({p for pattern in args.laws for p in glob.glob(pattern)} or set(args.laws))
    if len(paths) < 2:
        raise SystemExit(f"--laws matched {len(paths)} record(s); combining replicates needs at least two")
    rows = combine([rungs_of(Path(p)) for p in paths])

    measured = [r["inflation"] for r in rows if r["inflation"] is not None]
    inflation = float(np.median(measured)) if measured else 1.0

    print(f"{len(paths)} replicate ladders: {', '.join(Path(p).name for p in paths)}")
    print(f"\n{'N':>10}{'R':>4}{'mean':>11}{'empirical':>12}{'per-particle':>14}{'inflation':>11}{'overlap':>9}")
    for row in rows:
        overlap = row["n_particles"] / args.pool
        empirical = "--" if row["sem_empirical"] is None else f"{row['sem_empirical']:.4f}"
        ratio = "--" if row["inflation"] is None else f"{row['inflation']:.2f}x"
        row["mean_overlap"] = float(overlap) if row["n_draws"] > 1 else None
        print(
            f"{row['n_particles']:>10,}{row['n_draws']:>4}{row['mean']:>11.4f}{empirical:>12}"
            f"{row['sem_per_particle']:>14.4f}{ratio:>11}{overlap:>9.1%}"
        )
    print(f"\nmedian inflation over the {len(measured)} rung(s) with more than one draw: {inflation:.2f}x")

    usable = [
        r
        for r in rows
        if (r["sem_empirical"] is not None or not args.drop_singletons) and r["n_particles"] >= args.fit_min
    ]
    if len(usable) < 3:
        raise SystemExit(f"--fit-min {args.fit_min:,.0f} leaves {len(usable)} rung(s); a slope needs three")
    n = np.array([r["n_particles"] for r in usable], dtype=float)
    y = np.array([r["mean"] for r in usable], dtype=float)
    if np.any(y <= 0):
        raise SystemExit(f"{int((y <= 0).sum())} rung(s) average non-positive and the fit is in logs")
    sigma = np.array(
        [r["sem_empirical"] if r["sem_empirical"] is not None else r["sem_per_particle"] * inflation for r in usable],
        dtype=float,
    )
    free = weighted_power_law(n, y, sigma)
    fixed = weighted_power_law(n, y, sigma, slope=-1.0)
    span = math.log10(n.max() / n.min())
    # the same fit with NO floor, always: the law is asymptotic so the slope moves with the window,
    # and carrying both lets the caption state the dependence rather than assert a single exponent
    everything = [r for r in rows if r["sem_empirical"] is not None or not args.drop_singletons]
    n_all = np.array([r["n_particles"] for r in everything], dtype=float)
    y_all = np.array([r["mean"] for r in everything], dtype=float)
    sigma_all = np.array(
        [
            r["sem_empirical"] if r["sem_empirical"] is not None else r["sem_per_particle"] * inflation
            for r in everything
        ],
        dtype=float,
    )
    free_all = weighted_power_law(n_all, y_all, sigma_all)
    span_all = math.log10(n_all.max() / n_all.min())
    print(f"\nfitted on {n.size} rungs, {n.min():,.0f} to {n.max():,.0f} = {span:.2f} decades")
    print(f"free slope      {free['slope']:+.4f} +/- {free['slope_sem']:.4f}")
    print(
        f"  chi2/dof {free['chi_square_per_dof']:.2f} on {free['dof']} dof, "
        f"{abs(free['slope'] + 1.0) / free['slope_sem']:.1f} sigma from -1"
    )
    print(f"amplitude at -1 {fixed['amplitude']:,.0f}")
    print(
        f"over every rung ({n_all.size}, {span_all:.2f} decades) the same fit gives "
        f"{free_all['slope']:+.4f} +/- {free_all['slope_sem']:.4f}"
    )
    # THE CURVATURE THAT CHOOSES THE WINDOW, reported at whatever window was asked for, so a reader
    # can see whether this one passes the test it was selected on rather than take it on trust.
    curvature = log_curvature(n, y, sigma)
    if curvature is not None:
        print(
            f"quadratic in log N  {curvature['curvature']:+.5f} +/- {curvature['curvature_sem']:.5f}"
            f"  ({curvature['sigma']:.1f} sigma)"
        )

    write_result(
        args.out,
        {
            "rungs": rows,
            "n_replicates": len(paths),
            "median_inflation": inflation,
            "fit_span_decades": span,
            "fit_free_slope_all_rungs": free_all,
            "span_decades_all_rungs": span_all,
            "n_rungs_all": int(n_all.size),
            "fit_free_slope": free,
            "fit_slope_fixed_at_minus_one": fixed,
            "dropped_singletons": bool(args.drop_singletons),
            "fit_min_particles": float(args.fit_min),
            "log_quadratic": curvature,
        },
        driver=DRIVER,
        arguments={
            "laws": paths,
            "pool": args.pool,
            "fit_min": args.fit_min,
            "drop_singletons": bool(args.drop_singletons),
        },
    )
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
