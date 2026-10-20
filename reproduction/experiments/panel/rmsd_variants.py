"""Is C-alpha RMSD the right size variable, or only the conventional one?
    .venv/bin/python reproduction/experiments/panel/rmsd_variants.py --out results/rmsd_variants.json
Equation (1) predicts `d^2` from `ca_rmsd * mass_kda`, but `d^2` counts DISPLACED SCATTERING MASS and
C-alpha ignores side chains. This recomputes every scored pair under four other size definitions and
refits the law under each. Every variant is over HEAVY atoms and under the SAME C-alpha superposition,
so only the choice of atoms varies. The mass-weighted variant is the physically motivated one, closest
to what the score integrates. The correspondence is over atoms in BOTH structures, the matched-atom
count recorded, since a variant that wins only on the most complete models found deposition quality.
Rationale: docs/rationale/targets.md
"""

import argparse
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from cryodisc.specimen.structures import HYDROGEN
from reproduction.record import write_result
from reproduction.survey import (
    POPULATION_ARMS,
    POPULATION_PANELS,
    join_key,
    read_survey_rows,
)
from reproduction.targets import load_panels

DRIVER = "reproduction/experiments/panel/rmsd_variants.py"

BACKBONE = frozenset({"N", "CA", "C", "O"})
"""The peptide backbone, as a middle ground between C-alpha and everything."""


def variants(pair) -> dict[str, float]:
    """Every definition of the size of the change, in one C-alpha frame.

    Args:
        pair: A `ConformerPair` from `Candidate.align`.

    Returns:
        `{name: RMSD in A}` plus the atom count each was taken over.

    Raises:
        ValueError: If the two states' shared atoms do not correspond row for row, if any hydrogen
            survived the alignment, or if the C-alpha variant does not reproduce the alignment's own
            RMSD. All are cheap and all catch a failure that would otherwise be silent.
    """
    n = pair.alignment.n_atoms
    a, b = pair.state_a.head(n), pair.state_b.head(n)
    key = ["chain", "residue_id", "atom"]
    if not (a[key].to_numpy() == b[key].to_numpy()).all():
        raise ValueError("the two states' shared atoms are not in correspondence")

    names = a["atom"].to_numpy()
    # every definition below is over heavy atoms, and that rests on `align()`'s default; check it
    # here rather than trust it, since `hydrogens=True` is a live ablation knob elsewhere
    if a["element"].str.strip().str.upper().isin(HYDROGEN).any():
        raise ValueError("hydrogens reached the variants; the pair was aligned with hydrogens kept")
    square = ((b[["x", "y", "z"]].to_numpy() - a[["x", "y", "z"]].to_numpy()) ** 2).sum(1)
    weight = a["atomic_weight"].to_numpy()

    masks = {
        "ca": names == "CA",
        "cb": names == "CB",
        "backbone": np.isin(names, list(BACKBONE)),
        "all": np.ones(len(names), bool),
    }
    out: dict[str, float] = {}
    for name, mask in masks.items():
        out[f"rmsd_{name}"] = float(np.sqrt(square[mask].mean())) if mask.any() else float("nan")
        out[f"n_{name}"] = int(mask.sum())
    out["rmsd_mass_weighted"] = float(np.sqrt((square * weight).sum() / weight.sum()))

    if abs(out["rmsd_ca"] - pair.alignment.rmsd) > 1e-6:
        raise ValueError(
            f"recomputed C-alpha RMSD {out['rmsd_ca']:.6f} against the alignment's "
            f"{pair.alignment.rmsd:.6f}; the atom slice is wrong"
        )
    return out


def fit(size: np.ndarray, mass: np.ndarray, d2: np.ndarray) -> dict:
    """The product law `d^2 = A (size * mass)^p`, by least squares in logs.

    Args:
        size: One RMSD variant, in A.
        mass: Molecular mass in kDa.
        d2: Marginalized discriminability.

    Returns:
        The coefficient, exponent, `R^2` and the geometric scatter of measured against predicted.
    """
    x = np.log(size * mass)
    y = np.log(d2)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = y - predicted
    r2 = 1.0 - residual.var() / y.var()
    return {
        "coefficient": float(math.exp(intercept)),
        "exponent": float(slope),
        "r2": float(r2),
        "residual_factor": float(math.exp(residual.std())),
        "n": int(x.size),
    }


def scored(patterns: Iterable[str], *, supersede: bool = False) -> dict[str, tuple[float, float]]:
    """Each scored pair's marginalized `d^2` and mass, keyed by name.

    Args:
        patterns: Globs over survey result files, one per survey arm.

    Returns:
        `{name: (d2_marginal, mass_kda)}`, skipping rows that scored no marginal rung. A pair scored
        in two arms is refused by `read_survey_rows` rather than silently overwritten.
    """
    found: dict[str, tuple[float, float]] = {}
    for row in read_survey_rows(list(patterns), supersede=supersede):
        value = row.get("d2_marginal")
        if value is None or not math.isfinite(float(value)):
            continue
        found[join_key(row)] = (float(value), float(row["mass_kda"]))
    return found


def main() -> int:
    """Recompute every size variant and refit the law under each.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    # POPULATION_PANELS, matching POPULATION_ARMS: pairing `--rows` with a different panel set would
    # report a corpus that is an artifact of which structures happen to be on disk, not a choice
    parser.add_argument("--panels", nargs="+", default=list(POPULATION_PANELS))
    parser.add_argument("--rows", nargs="+", default=list(POPULATION_ARMS))
    parser.add_argument(
        "--min-shared-fraction",
        type=float,
        default=None,
        help="overlap below which a pair is refused, as `Candidate.align` takes it. Must MATCH "
        "the arm being read: the module default re-refuses here what the survey was told to "
        "admit, and the driver then reports a corpus it did not measure",
    )
    parser.add_argument(
        "--supersede",
        action="store_true",
        help="let a LATER --rows glob replace an earlier one's row of the same pair, "
        "instead of refusing it as a double count. For an arm that deliberately "
        "re-measures rows of another, as results/populationfix_* does for results/population_*; the "
        "order of --rows then carries the precedence",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="download any structure `--data-dir` does not already hold. WITHOUT this the corpus is "
        "silently whatever happens to be cached, rather than the full population of scored pairs",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/pdb"))
    parser.add_argument("--out", type=Path, default=Path("results/rmsd_variants.json"))
    args = parser.parse_args()

    targets = scored(args.rows, supersede=True)
    if not targets:
        raise SystemExit(f"{' '.join(args.rows)} matched no scored row")

    rows = []
    # `load_panels` refuses a pair that two panels both carry, which is what the relax superset does
    candidates, failures = load_panels(args.panels, data_dir=args.data_dir, fetch=args.fetch)
    for candidate in candidates:
        key = join_key(candidate)
        if key not in targets:
            continue
        try:
            measured = variants(candidate.align(min_shared_fraction=args.min_shared_fraction))
        # align_conformers raises ValueError on a correspondence failure and OSError on an
        # unreadable file; either way one bad pair must not take the sweep down
        except (ValueError, KeyError, OSError, RuntimeError) as error:
            failures.append(f"{candidate.name}: {error}")
            continue
        d2, mass = targets[key]
        rows.append({"name": candidate.name, "d2_marginal": d2, "mass_kda": mass, **measured})

    if not rows:
        raise SystemExit("no pair both scored and aligned")

    masses = np.array([r["mass_kda"] for r in rows], float)
    d2_values = np.array([r["d2_marginal"] for r in rows], float)
    fits = {}
    for name in ("ca", "cb", "backbone", "all", "mass_weighted"):
        size = np.array([r[f"rmsd_{name}"] for r in rows], float)
        good = np.isfinite(size) & (size > 0)
        fits[name] = fit(size[good], masses[good], d2_values[good])

    print(f"{len(rows)} pairs aligned and scored, of {len(targets)} scored rows")
    if failures:
        print(f"  {len(failures)} pair(s) could not be built; first: {failures[0][:90]}")
    print(f"\n  {'variant':14s}{'n':>6s}{'coeff':>9s}{'exponent':>10s}{'R^2':>8s}{'residual':>10s}")
    for name, result in fits.items():
        print(
            f"  {name:14s}{result['n']:6d}{result['coefficient']:9.4f}{result['exponent']:10.3f}"
            f"{result['r2']:8.4f}{result['residual_factor']:10.3f}x"
        )
    best = min(fits.items(), key=lambda kv: kv[1]["residual_factor"])
    print(
        f"\n  tightest: {best[0]} at {best[1]['residual_factor']:.3f}x "
        f"against ca at {fits['ca']['residual_factor']:.3f}x"
    )
    print(
        f"  wrote {write_result(args.out, {'rows': rows, 'fits': fits, 'failures': failures}, driver=DRIVER, arguments=vars(args))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
