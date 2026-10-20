"""Where every surveyed pair's state difference sits in spatial frequency, from the structures alone.

    .venv/bin/python .../difference_spectrum.py --panels 'data/survey_population_*.json' --out out.json
Stores the shell-resolved difference power under fixed optics, so the whole noise side of the
collection (dose, DQE, radiation envelope, Debye-Waller B, band limit) can be swept afterwards
without re-rendering. The CTF does NOT re-weight (see `rungs.spectrum`), so `--defocus` renders a pass
per value and takes a list. No particles, pose bank or fitting: the cheap arm's arithmetic with the
shell sum left undone, its defaults matching the survey's plan so it reproduces the recorded `d2_known_pose`.
Rationale: docs/rationale/reproduction/experiments/spectrum.md
"""

import argparse
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from cryodisc.forward.optics import CTFParams
from cryodisc.grids.fourier import shell_indices
from cryodisc.grids.poses import pose_bank
from cryodisc.microscope import CollectionPlan
from reproduction.operating_point import SHARED
from reproduction.predictors.spectrum import d2_from_spectrum, difference_spectrum, half_power_spacing, shell_weight
from reproduction.record import write_result
from reproduction.targets import CandidateRejected, load_panels

DRIVER = "reproduction/experiments/benchmark/difference_spectrum.py"


def measure(candidate, plan, args, angles) -> dict:
    """One pair's spectrum at every requested defocus, with the geometry to regress it on.

    Args:
        candidate: The pair, as `load_panels` returns it.
        plan: The reference collection.
        args: Parsed arguments, for the box, the rmsf and the defocus list.
        angles: Poses to average over.

    Returns:
        The row.

    Raises:
        CandidateRejected: If the correspondence or the box refuses the pair, exactly as the
            survey's own arm would refuse it.
    """
    pair = candidate.align(min_shared_fraction=args.min_shared_fraction)
    minimum = pair.alignment.minimum_box_size(plan.pixel_size)
    if args.box_size < minimum:
        raise CandidateRejected(
            f"{candidate.name}: box {args.box_size} is below the {minimum} its "
            f"{pair.alignment.radius:.0f} A radius needs, so the templates would be clipped"
        )
    model_a, model_b = pair.models(plan, box_size=args.box_size, form_factor="peng")

    # R_g on the pair's own superposition, the length scale the bank law and equation (1) are
    # written in, recorded here so the regression on `s` needs no second pass over the structures
    shared = pair.alignment.n_atoms
    state_a, state_b = pair.state_a.iloc[:shared], pair.state_b.iloc[:shared]
    middle = 0.5 * (state_a[["x", "y", "z"]].to_numpy() + state_b[["x", "y", "z"]].to_numpy())
    mass = state_a["atomic_weight"].to_numpy()
    spread = np.linalg.norm(middle - np.average(middle, axis=0, weights=mass), axis=1)

    n_shells = args.box_size // 2
    weight = shell_weight(plan, args.box_size, n_shells, rmsf=args.rmsf)
    row: dict = {
        "name": candidate.name,
        "motion": candidate.motion,
        "mass_kda": float(mass.sum() / 1000.0),
        "ca_rmsd": float(pair.alignment.rmsd),
        "rg": float(np.sqrt(np.average(spread**2, weights=mass))),
        "radius": float(spread.max()),
        "n_atoms": int(shared),
        "n_shells": n_shells,
        "defocus": [],
    }
    for defocus in args.defocus:
        spectrum = difference_spectrum(
            model_a,
            model_b,
            angles,
            box_size=args.box_size,
            n_shells=n_shells,
            ctf_params=CTFParams.from_plan(plan, defocus),
        )
        row["defocus"].append(
            {
                "defocus_in_angstroms": float(defocus),
                # the whole point of the record: everything else here is derivable from it
                "power": [float(value) for value in np.asarray(spectrum.power)],
                "d2_known_pose": d2_from_spectrum(spectrum, weight),
                "half_power_spacing_a": half_power_spacing(spectrum, weight) * plan.pixel_size,
            }
        )
    return row


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panels", nargs="+", required=True, help="globs over survey panel files")
    parser.add_argument("--data-dir", default="data/pdb", help="where the depositions live")
    parser.add_argument("--out", required=True, help="record to write")
    parser.add_argument("--pixel-size", type=float, default=SHARED.pixel_size)
    parser.add_argument("--box-size", type=int, default=128)
    parser.add_argument("--dose", type=float, default=SHARED.dose)
    parser.add_argument("--rmsf", type=float, default=SHARED.rmsf, help="the reference Debye-Waller, B = 8 pi^2 rmsf^2")
    parser.add_argument("--voltage", type=float, default=300.0)
    parser.add_argument(
        "--defocus",
        type=float,
        nargs="+",
        default=[15000.0],
        help="underfocus in A, one render pass each. The default is the survey's `mid_defocus`, so "
        "the shell sum reproduces the d2_known_pose already in results/population_*.json",
    )
    parser.add_argument("--poses", type=int, default=96, help="bank whose nodes the spectrum averages over")
    parser.add_argument("--min-shared-fraction", type=float, default=0.0)
    parser.add_argument("--only", nargs="+", default=None, help="score only these rows, named as the panel does")
    args = parser.parse_args(list(argv) if argv is not None else None)

    plan = CollectionPlan(
        dose=args.dose,
        pixel_size=args.pixel_size,
        defocus_range=(min(args.defocus), max(args.defocus)),
        voltage_in_kv=args.voltage,
    )
    angles = pose_bank(args.poses).angles
    candidates, unresolved = load_panels(args.panels, Path(args.data_dir), fetch=False)
    if args.only:
        wanted = set(args.only)
        candidates = [candidate for candidate in candidates if candidate.name in wanted]
    print(f"{len(candidates)} pair(s), {len(unresolved)} unresolved, {angles.shape[0]} poses", flush=True)

    rows, refusals = [], list(unresolved)
    for position, candidate in enumerate(candidates, start=1):
        started = time.monotonic()
        try:
            row = measure(candidate, plan, args, angles)
        except (CandidateRejected, ValueError) as refusal:
            refusals.append(f"{candidate.name}: {refusal}")
            print(f"[{position}/{len(candidates)}] dropped: {str(refusal)[:90]}", flush=True)
            continue
        row["seconds"] = time.monotonic() - started
        rows.append(row)
        first = row["defocus"][0]
        print(
            f"[{position}/{len(candidates)}] {candidate.name[:44]:46s} "
            f"d2 {first['d2_known_pose']:9.4g}  half power {first['half_power_spacing_a']:6.1f} A  "
            f"{row['seconds']:.1f}s",
            flush=True,
        )
        # written every row, not once at the end, so a run that dies partway keeps its scored rows
        write_result(
            args.out,
            {
                "rows": rows,
                "refusals": refusals,
                "n_scored": len(rows),
                "n_refused": len(refusals),
                "frequency_cyc_px": [
                    float(value) for value in np.asarray(shell_indices(args.box_size, row["n_shells"])[1])
                ],
                "shell_weight": [
                    float(value) for value in shell_weight(plan, args.box_size, row["n_shells"], rmsf=args.rmsf)
                ],
            },
            driver=DRIVER,
            arguments=vars(args),
        )
    print(f"\n{len(rows)} scored, {len(refusals)} refused -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
