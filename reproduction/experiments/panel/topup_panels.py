"""The survey rows whose pose bank was too coarse, written out as panels to re-measure.
    .venv/bin/python reproduction/experiments/panel/topup_panels.py --cut 5.0 --per-panel 8
The selection borrows Fig 5's abscissa `x = N (Ca RMSD/R_g)^(3/2) / D`, which orders how coarse a bank
is against the conformational change. A published-survey row is selected when its `x` at the survey's
own top rung `SURVEY_TOP_RUNG` falls below `--cut`, those being the rows the finer ladders show were
short of convergence. Written as PANELS, not one `--only` list, because `score_survey.py` records after
every marginal row: a panel that dies keeps its finished rows. Members are dealt round-robin after
sorting by `x`, so each panel is a cross-section of the cut. Rationale: docs/rationale/reproduction/experiments/panel.md
"""

import argparse
import glob
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from reproduction.survey import POPULATION_ARMS, read_survey_rows

DRIVER = "reproduction/experiments/panel/topup_panels.py"

BANK_CONSTANT = 29.8
"""`D` of equation (3), from `results/floor_*` and the survey's own ladders. See the module docstring."""

SURVEY_TOP_RUNG = 8_064
"""Realized node count of the published survey's finest bank, which every row of it shares.

Read off `results/population_*.json`, where every row carries the same quantized `--extrapolate-nodes`
triple whose top rung this is.
"""


def radii_of_gyration(names: Iterable[str], features: str) -> dict[str, float]:
    """`R_g` per pair, from the feature table where it has one and from the structures where not.

    Args:
        names: Pair names needing a length scale.
        features: Path to `pair_features.py`'s record.

    Returns:
        `{name: R_g}` in angstroms, over every name that could be measured.

    Raises:
        SystemExit: If a name has no `R_g` from either route, since dropping it silently would
            shrink the cut without saying so.
    """
    from reproduction.record import payload_of
    from reproduction.targets import load_panels

    wanted = set(names)
    found = {
        row["name"]: float(row["rg"])
        for row in payload_of(json.loads(Path(features).read_text()))["rows"]
        if row.get("rg") and np.isfinite(row["rg"]) and row["name"] in wanted
    }
    remaining = wanted - set(found)
    if remaining:
        # the same superposition `score_survey.py` scores on, so the ratio is the pair's own
        candidates, _ = load_panels("data/survey_population_*.json", Path("data/pdb"), fetch=False)
        for candidate in candidates:
            if candidate.name not in remaining:
                continue
            pair = candidate.align(min_shared_fraction=0.0)
            shared = pair.alignment.n_atoms
            state_a, state_b = pair.state_a.iloc[:shared], pair.state_b.iloc[:shared]
            middle = 0.5 * (state_a[["x", "y", "z"]].to_numpy() + state_b[["x", "y", "z"]].to_numpy())
            weight = state_a["atomic_weight"].to_numpy()
            spread = np.linalg.norm(middle - np.average(middle, axis=0, weights=weight), axis=1)
            found[candidate.name] = float(np.sqrt(np.average(spread**2, weights=weight)))
    still = wanted - set(found)
    if still:
        raise SystemExit(f"{len(still)} row(s) have no R_g, so their x cannot be read: {sorted(still)[:5]}")
    return found


def select(rows: list[dict], scales: dict[str, float], cut: float) -> list[tuple[float, dict]]:
    """The rows below the cut, with the `x` each was judged on, coarsest first.

    Args:
        rows: Survey rows carrying `ca_rmsd`.
        scales: `R_g` per pair name.
        cut: Keep rows whose `x` at `SURVEY_TOP_RUNG` is below this.

    Returns:
        `(x, row)` ascending in `x`.
    """
    picked = [(SURVEY_TOP_RUNG * (row["ca_rmsd"] / scales[row["name"]]) ** 1.5 / BANK_CONSTANT, row) for row in rows]
    return sorted((entry for entry in picked if entry[0] < cut), key=lambda entry: entry[0])


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", nargs="+", default=list(POPULATION_ARMS), help="globs over the survey records")
    parser.add_argument("--panels", default="data/survey_population_*.json", help="glob over the source panels")
    parser.add_argument("--features", default="results/pair_features.json", help="where R_g comes from")
    parser.add_argument("--cut", type=float, default=5.0, help="keep rows with x below this at 8,064 nodes")
    parser.add_argument("--per-panel", type=int, default=8, help="rows per written panel")
    parser.add_argument("--out", default="data/survey_topup_{}.json", help="output template, one `{}` for the index")
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = [
        row
        for row in read_survey_rows(args.rows, supersede=True)
        if isinstance(row.get("d2_marginal"), (int, float)) and row["d2_marginal"] > 0
    ]
    scales = radii_of_gyration((row["name"] for row in rows), args.features)
    picked = select(rows, scales, args.cut)

    source: dict[str, dict] = {}
    for path in sorted(glob.glob(args.panels)):
        for candidate in json.loads(Path(path).read_text())["candidates"]:
            source.setdefault(candidate["name"], candidate)
    absent = [row["name"] for _, row in picked if row["name"] not in source]
    if absent:
        raise SystemExit(f"{len(absent)} selected row(s) are in no panel, so they cannot be re-run: {absent[:5]}")

    n_panels = -(-len(picked) // args.per_panel)
    for index in range(n_panels):
        members = picked[index::n_panels]
        body = {
            "_note": (
                f"Top-up panel {index + 1} of {n_panels}: rows of the published survey whose pose bank was too "
                f"coarse, selected as x = N (Ca RMSD/R_g)^(3/2) / D < {args.cut:g} at the survey's own top rung "
                f"N = {SURVEY_TOP_RUNG}, D = {BANK_CONSTANT}. Every row is already in "
                f"results/population_*.json; nothing new is admitted to the survey. Written by {DRIVER}. "
                f"x here runs {members[0][0]:.3f} to {members[-1][0]:.3f}."
            ),
            "ligands": {},
            "candidates": [source[row["name"]] for _, row in members],
        }
        Path(args.out.format(index + 1)).write_text(json.dumps(body, indent=1))

    sizes = [len(picked[index::n_panels]) for index in range(n_panels)]
    print(
        f"{len(rows)} survey rows -> {len(picked)} at x < {args.cut:g} "
        f"(x from {picked[0][0]:.3f} to {picked[-1][0]:.3f})\n"
        f"wrote {n_panels} panel(s) of {min(sizes)}-{max(sizes)} rows to {args.out.format('*')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
