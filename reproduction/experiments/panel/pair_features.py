"""Every structural feature a two-state pair carries, computed once so a regression can rank them.
    .venv/bin/python reproduction/experiments/panel/pair_features.py --out results/pair_features.json
Equation (1) is a power law in mass and Ca RMSD; this computes the other structural numbers so
`feature_regression.py` can ask whether that is the best pair. Grouped by what they could explain:
size and shape, motion magnitude, motion character. The one with a mechanism is `coherence`: `d^2`
integrates `|dRho(k)|^2`, whose cross terms cancel for a rigid displacement and add for an incoherent
one, so two pairs at one `RMSD x M` need not share a `d^2`. The superposition is the survey's own
(`Candidate.align()`), so each feature is on the correspondence that produced the pair's `d^2`; a
re-derived one returns a believable but wrong RMSD. Rationale: docs/rationale/reproduction/experiments/preprocess-and-panel.md
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from reproduction.record import write_result
from reproduction.survey import (
    POPULATION_ARMS,
    POPULATION_PANELS,
    join_key,
    read_survey_rows,
)
from reproduction.targets import load_panels

DRIVER = "reproduction/experiments/panel/pair_features.py"

XYZ = ["x", "y", "z"]
NEIGHBOR_A = 8.0
"""Contact radius in angstroms for the coherence sum.

8 A is the usual Ca-Ca contact cutoff: far enough to reach the next strand or helix turn, close
enough that a pair inside it is in the same rigid element if there is one. The coherence measure is a
RATIO of sums over the same neighbor set, so the cutoff sets which scale of rigidity is being asked
about rather than the units of the answer."""


def gyration(coords: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Radius of gyration and the shape descriptors of the gyration tensor.

    Args:
        coords: `(n, 3)` positions.
        weights: `(n,)` masses.

    Returns:
        `rg`, and the three shape scalars `asphericity`, `acylindricity` and `anisotropy`, each
        dimensionless and each computed from the tensor's sorted eigenvalues.
    """
    center = np.average(coords, axis=0, weights=weights)
    delta = coords - center
    tensor = (weights[:, None, None] * delta[:, :, None] * delta[:, None, :]).sum(0) / weights.sum()
    lam = np.sort(np.linalg.eigvalsh(tensor))[::-1]
    total = float(lam.sum())
    return {
        "rg": float(np.sqrt(total)),
        "asphericity": float(lam[0] - 0.5 * (lam[1] + lam[2])) / total if total else float("nan"),
        "acylindricity": float(lam[1] - lam[2]) / total if total else float("nan"),
        # Kappa^2, 0 for a sphere and 1 for a rod, so it is comparable across sizes
        "anisotropy": float(1.5 * (lam**2).sum() / total**2 - 0.5) if total else float("nan"),
    }


def coherence(coords: np.ndarray, shift: np.ndarray, cutoff: float = NEIGHBOR_A) -> float:
    """How much neighboring atoms move TOGETHER rather than independently.

    `sum_{i~j} dr_i . dr_j / sum_{i~j} |dr_i||dr_j|` over pairs within `cutoff`, so it is +1 for a
    displacement field that is locally rigid, 0 for one that is locally random, and negative for one
    that is locally opposed.

    This is the quantity equation (1) cannot see. `d^2` integrates the band-limited `|dRho(k)|^2`,
    which for a displacement field carries the cross terms `dr_i . dr_j`; a coherent field cancels
    those at low `k` while an incoherent one accumulates them. Two pairs at one `RMSD x M` therefore
    need not have one `d^2`.

    Args:
        coords: `(n, 3)` reference positions.
        shift: `(n, 3)` displacements.
        cutoff: Neighbor radius in angstroms.

    Returns:
        The ratio, or nan if no atom has a neighbor or nothing moves.
    """
    from scipy.spatial import cKDTree

    pairs = cKDTree(coords).query_pairs(cutoff, output_type="ndarray")
    if not len(pairs):
        return float("nan")
    left, right = shift[pairs[:, 0]], shift[pairs[:, 1]]
    norms = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    total = float(norms.sum())
    return float((left * right).sum() / total) if total > 0 else float("nan")


def features(state_a: pd.DataFrame, state_b: pd.DataFrame, n_shared: int) -> dict[str, float]:
    """Every feature this driver computes, for one superposed pair.

    Args:
        state_a: State A's atoms, the first `n_shared` corresponding.
        state_b: The same for state B.
        n_shared: How many leading rows correspond.

    Returns:
        The features, keyed by name.
    """
    a = state_a.iloc[:n_shared]
    b = state_b.iloc[:n_shared]
    xa, xb = a[XYZ].to_numpy(), b[XYZ].to_numpy()
    # `atomic_weight` is the mmdf schema's name; refused rather than defaulted to unit weights, since a
    # weight of one is a legitimate-looking number that would silently make every mass-weighted feature wrong
    if "atomic_weight" not in a:
        raise KeyError("the atom table carries no `atomic_weight`, and unit weights are not a mass")
    mass = a["atomic_weight"].to_numpy()
    shift = xb - xa
    disp = np.linalg.norm(shift, axis=1)
    mid = 0.5 * (xa + xb)

    ga, gb = gyration(xa, mass), gyration(xb, mass)
    out: dict[str, float] = {
        # --- size and shape, from the midpoint structure ---------------------------------------
        "n_atoms": float(len(a)),
        "rg": float(0.5 * (ga["rg"] + gb["rg"])),
        "dmax": float(np.linalg.norm(mid - mid.mean(0), axis=1).max() * 2.0),
        "asphericity": float(0.5 * (ga["asphericity"] + gb["asphericity"])),
        "acylindricity": float(0.5 * (ga["acylindricity"] + gb["acylindricity"])),
        "anisotropy": float(0.5 * (ga["anisotropy"] + gb["anisotropy"])),
        # --- motion magnitude --------------------------------------------------------------------
        "rmsd_all": float(np.sqrt((disp**2).mean())),
        "max_disp": float(disp.max()),
        "median_disp": float(np.median(disp)),
        "sum_sq_disp": float((disp**2).sum()),
        "mass_weighted_disp": float((mass * disp**2).sum()),
        "frac_moved_1a": float((disp > 1.0).mean()),
        "frac_moved_2a": float((disp > 2.0).mean()),
        # --- motion character --------------------------------------------------------------------
        "d_rg": float(abs(ga["rg"] - gb["rg"])),
        "d_asphericity": float(abs(ga["asphericity"] - gb["asphericity"])),
        "com_shift": float(np.linalg.norm(np.average(shift, axis=0, weights=mass))),
        "coherence": coherence(mid, shift),
    }
    # how localized the motion is: 1 when every atom moves equally, 1/n when one atom carries it all
    fourth = float((disp**4).sum())
    out["participation_ratio"] = float(out["sum_sq_disp"] ** 2 / (len(a) * fourth)) if fourth > 0 else float("nan")
    # the share of the displacement that is a net translation of the whole molecule
    out["translation_fraction"] = (
        float(len(a) * np.linalg.norm(shift.mean(0)) ** 2 / (disp**2).sum()) if (disp**2).sum() > 0 else float("nan")
    )
    return out


def main(argv: Iterable[str] | None = None) -> int:
    """Compute the features for every survey pair whose structures are on disk."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    # POPULATION_PANELS, matching POPULATION_ARMS: pairing `--rows` with a different panel set would
    # report a corpus that is an artifact of which structures happen to be on disk, not a choice
    parser.add_argument("--panels", nargs="+", default=list(POPULATION_PANELS))
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="download any structure `--data-dir` does not already hold, so the corpus is the "
        "panel set and not whatever happens to be cached",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/pdb"))
    parser.add_argument("--out", type=Path, default=Path("results/pair_features.json"))
    parser.add_argument("--limit", type=int, default=None, help="stop after this many pairs, for a smoke test")
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = read_survey_rows(args.rows, supersede=True)
    scored = {join_key(row): row for row in rows}

    computed: list[dict] = []
    refusals: dict[str, int] = {}
    # the panel's own correspondence via `Candidate.align()`, not one re-derived from the row's `note`
    # (which lacks the labels and assembly): a guessed correspondence returns a believable but wrong RMSD
    candidates, unresolved = load_panels(args.panels, data_dir=args.data_dir, fetch=args.fetch)
    if unresolved:
        refusals["unresolved_deposition"] = len(unresolved)
    for candidate in candidates:
        row = scored.get(join_key(candidate))
        if row is None:
            continue
        try:
            pair = candidate.align(min_shared_fraction=args.min_shared_fraction)
            values = features(pair.state_a, pair.state_b, pair.alignment.n_atoms)
        except (ValueError, KeyError, IndexError, OSError) as error:
            kind = "correspondence" if "share only" in str(error) else type(error).__name__
            refusals[kind] = refusals.get(kind, 0) + 1
            continue
        computed.append(
            {
                "name": candidate.name,
                "motion": row.get("motion"),
                "mass_kda": row.get("mass_kda"),
                "ca_rmsd": row.get("ca_rmsd"),
                "radius": row.get("radius"),
                "d2_marginal": row.get("d2_marginal"),
                "d2_known_pose": row.get("d2_known_pose"),
                **values,
            }
        )
        if len(computed) % 50 == 0:
            print(f"  {len(computed)} pairs computed", flush=True)
        if args.limit is not None and len(computed) >= args.limit:
            break

    payload = {"rows": computed, "n_rows": len(computed), "refusals": refusals, "neighbor_cutoff_a": NEIGHBOR_A}
    write_result(
        args.out,
        payload,
        driver=DRIVER,
        arguments={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    )
    print(f"{len(computed)} pairs featured, {sum(refusals.values())} refused {refusals}")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
