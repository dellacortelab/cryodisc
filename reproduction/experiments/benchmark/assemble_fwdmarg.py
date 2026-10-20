"""Assemble per-K ab-initio forward-marginal runs into one convergence curve for a system.

Reads the results/<prefix>_K*.json a fanned-out set of cas9_forward_marginal.py jobs wrote and records
the d'^2_marginal vs pose-bank curve, the plateau (the two largest banks within 5%), the pose
efficiency at the plateau, and the comparison to d'^2_obs -- the headline the issue-1 ladder quotes.
"""

import argparse
import glob
import json
from pathlib import Path

from reproduction.record import write_result

DRIVER = "reproduction/experiments/benchmark/assemble_fwdmarg.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", default="results/cas9_fwdmarg_K*.json", help="per-K result files")
    parser.add_argument("--out", type=Path, default=Path("results/cas9_forward_marginal.json"))
    args = parser.parse_args(argv)

    rows = []
    for path in sorted(glob.glob(args.glob)):
        r = json.loads(Path(path).read_text())["result"]
        rows.append(r)
    if not rows:
        raise SystemExit(f"no files matched {args.glob}")
    rows.sort(key=lambda r: r["nodes"])

    curve = {str(r["nodes"]): r["d2_marginal"] for r in rows}
    known_pose = rows[-1]["d2_known_pose"]
    obs = rows[-1]["d2_obs"]
    plateau = rows[-1]["d2_marginal"]
    # converged if the top two banks agree within 5%, else report the largest as a lower bound
    converged = len(rows) >= 2 and abs(rows[-1]["d2_marginal"] - rows[-2]["d2_marginal"]) <= 0.05 * plateau
    rising = len(rows) >= 2 and rows[-1]["d2_marginal"] > rows[-2]["d2_marginal"]
    # the inverse-crime signature: a perfect-model marginal climbs toward the known-pose ceiling as the
    # bank densifies, so a marginal well above d'^2_obs and still rising means pose marginalization in
    # the forward model does NOT reproduce obs -- the obs<known-pose gap is a real-data effect
    # (model mismatch + envelope) the FSC route carries and this inverse-crime sim does not.
    if plateau > 1.5 * obs and rising:
        finding = (
            "inverse-crime marginal climbs toward the known-pose ceiling with bank size and sits well "
            "above d'^2_obs; pose marginalization in the perfect forward model does NOT account for the "
            "known-pose->obs gap, which is therefore a real-data effect (model mismatch + envelope)"
        )
    elif converged and abs(plateau - obs) <= 0.3 * obs:
        finding = "forward-sim marginal plateaus near d'^2_obs: pose marginalization reproduces obs"
    else:
        finding = "forward-sim marginal is a largest-bank lower bound; add a larger K to resolve the plateau"

    record = {
        "system": rows[-1]["system"],
        "d2_marginal_plateau": plateau,
        "converged": bool(converged),
        "status": "plateau" if converged else "largest-bank lower bound (add a larger K to confirm)",
        "d2_known_pose": known_pose,
        "pose_efficiency_plateau": plateau / known_pose if known_pose else None,
        "d2_obs": obs,
        "marginal_over_obs": plateau / obs if obs else None,
        "finding": finding,
        "curve_nodes_to_d2": curve,
        "n_banks": len(rows),
        "per_bank": [
            {"nodes": r["nodes"], "d2_marginal": r["d2_marginal"], "pose_efficiency": r["pose_efficiency"]}
            for r in rows
        ],
        "convention": rows[-1]["convention"],
    }
    for r in rows:
        print(f"  K={r['nodes']:>8}   d2_marginal {r['d2_marginal']:.4g}   pose_eff {r['pose_efficiency']:.4g}")
    print(
        f"plateau {plateau:.4g} ({'converged' if converged else 'LOWER BOUND'})   "
        f"known-pose {known_pose:.4g}   obs {obs}   marginal/obs {plateau / obs:.3g}"
    )

    write_result(args.out, record, driver=DRIVER, arguments=vars(args))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
