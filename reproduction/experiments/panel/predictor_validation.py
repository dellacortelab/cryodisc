"""Validate the pool-free predictor: does closed-form I_local predict the converged marginal I(p)?

    .venv/bin/python reproduction/experiments/panel/predictor_validation.py \
        --converge results/fork_deflection --local results/ip_local_p1.json results/ip_local_p2.json

Reads ip_converge ladders (top-bank `information` = measured marginal I(p)) against ip_local records
(closed-form `i_local`, the Haar-averaged CRLB rung), both at one contrast, and fits
log I_measured ~ log I_local. rho_global = I_measured / I_local is the marginalization loss. A pair still
gaining > RISE over its top octave has not plateaued, so its I_measured is a lower bound and it is
excluded from the headline fit and flagged. No new compute; a check on the retained deflection-currency
data before the dose sweep. Companion: docs/planning/figure5_pose_marginalization.md.
"""

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

RISE = 0.15  # top-octave d^2 gain above this => still rising at the largest realized bank


def load_converge(result_dirs: list[Path], contrast: float) -> dict:
    """slug -> top-bank info for the ladder at `contrast`: name, I_measured, d2, convergence gain."""
    out = {}
    for result_dir in result_dirs:
        pattern = str(result_dir / f"*_a{contrast}.json")
        for f in sorted(glob.glob(pattern)):
            row = json.loads(Path(f).read_text())["rows"][0]
            slug = Path(f).name.split("ip_converge_")[-1].split("deflection_")[-1].rsplit("_a", 1)[0]
            rungs = row["rungs"]
            prev = rungs[-2]["d2"] if len(rungs) > 1 else 0.0  # single-bank ladder: unassessable, read as rising
            gain = (rungs[-1]["d2"] - prev) / max(rungs[-1]["d2"], 1e-12)
            out[slug] = {
                "name": row["name"],
                "rmsd": row["rmsd"],
                "i_measured": rungs[-1]["information"],
                "d2_top": rungs[-1]["d2"],
                "top_bank": rungs[-1]["k"],
                "top_octave_gain": float(gain),
                "converged": abs(gain) < RISE,
            }
    return out


def load_local(local_files: list[Path], contrast: float) -> dict:
    """pair name -> i_local at `contrast`, merged across ip_local records."""
    out = {}
    for f in local_files:
        for row in json.loads(Path(f).read_text())["rows"]:
            match = [a for a in row["amplitudes"] if a["amplitude"] == contrast]
            if match:
                out[row["name"]] = match[0]["i_local"]
    return out


def powfit(x: list[float], y: list[float]) -> tuple[float, float, float, float]:
    """Power-law fit y = a * x^b in log space; returns (b, a, R^2, corr)."""
    lx, ly = np.log(np.asarray(x)), np.log(np.asarray(y))
    b, c = np.polyfit(lx, ly, 1)
    resid = ly - (c + b * lx)
    r2 = 1 - (resid**2).sum() / ((ly - ly.mean()) ** 2).sum()
    return float(b), math.exp(float(c)), float(r2), float(np.corrcoef(lx, ly)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--converge", type=Path, nargs="+", default=[Path("results/fork_deflection")])
    parser.add_argument(
        "--local", type=Path, nargs="+", default=[Path("results/ip_local_p1.json"), Path("results/ip_local_p2.json")]
    )
    parser.add_argument("--contrast", type=float, default=1.0)
    args = parser.parse_args()

    converge = load_converge(args.converge, args.contrast)
    local = load_local(args.local, args.contrast)

    print(f"contrast {args.contrast}; I_measured at each pair's top realized bank")
    print(f"{'pair':<12}{'rmsd':>6}{'I_local':>9}{'I_meas':>9}{'rho_g':>7}{'top_oct':>9}{'top_bank':>9}  status")
    rows = []
    for slug, info in sorted(converge.items()):
        il = local.get(info["name"])
        if il is None:
            continue
        rho = info["i_measured"] / il
        status = "converged" if info["converged"] else "RISING"
        rows.append({"slug": slug, "i_local": il, "i_measured": info["i_measured"], **info})
        print(
            f"{slug:<12}{info['rmsd']:>6.2f}{il:>9.3f}{info['i_measured']:>9.3f}{rho:>7.3f}"
            f"{info['top_octave_gain'] * 100:>+8.1f}%{info['top_bank']:>9}  {status}"
        )

    conv = [r for r in rows if r["converged"]]
    print(
        f"\nrho_global (converged {len(conv)}/{len(rows)}): {[round(r['i_measured'] / r['i_local'], 3) for r in conv]}"
    )
    for label, subset in (("all", rows), ("converged", conv)):
        if len(subset) >= 3:
            b, a, r2, corr = powfit([r["i_local"] for r in subset], [r["i_measured"] for r in subset])
            print(
                f"  fit [{label:>9}, n={len(subset)}]: I_meas = {a:.3f} I_local^{b:.3f}   R^2={r2:.3f}  corr={corr:.3f}"
            )


if __name__ == "__main__":
    main()
