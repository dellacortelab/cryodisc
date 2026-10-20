"""Compute the closed-form local predictor I_local(a) for the marginal Fisher information.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python reproduction/experiments/panel/ip_local.py \
        --panel data/survey_topup_1.json --rows NAME_A NAME_B \
        --contrasts 1.0 0.1 --nodes 8064 --out results/ip_local_1.json

`I_local = E_R[I_p(d2_crlb(R))]` and `I_known = E_R[I_p(d2_known(R))]`, the Haar average over a pose
bank of the capped Fisher map applied to the per-pose CRLB and pose-known squared distances. It is the
pool-free predictor of the measured marginal I(p): free at any contrast from one render, since d^2
scales as a^2. No stack is simulated and no pool is scored, so it reports what the marginal WOULD carry
if the pose were locally resolved -- the ceiling the marginalization loss `rho_global` sits below.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from cryodisc import CollectionPlan
from cryodisc.forward.optics import CTFParams
from cryodisc.grids.poses import pose_bank
from cryodisc.resolution import noise_spectrum
from cryodisc.rungs import local_information
from cryodisc.statistics.information import fisher_from_deflection
from reproduction.operating_point import SHARED
from reproduction.targets import load_panel

DRIVER = "reproduction/experiments/panel/ip_local.py"


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


def measure_row(candidate, plan, args) -> dict:
    """I_local(a) and I_known(a) for one candidate over the requested contrasts, one render each."""
    pair = candidate.align(min_shared_fraction=args.min_shared_fraction)
    box_size = max(args.box_size, pair.alignment.minimum_box_size(plan.pixel_size))
    model_a, model_b = pair.models(plan, box_size=box_size, form_factor="peng")
    spectrum = noise_spectrum(plan, box_size, rmsf=args.rmsf)
    reference_optics = CTFParams.from_plan(plan, plan.mid_defocus)
    grid = pose_bank(args.nodes)
    fisher = jax.vmap(lambda d2: fisher_from_deflection(d2, args.fraction))

    contrasts = []
    for a in args.contrasts:
        t0 = time.time()
        rung = local_information(model_a, model_b, grid.angles, spectrum, contrast=a, ctf_params=reference_optics)
        i_local = float(jnp.mean(fisher(rung.crlb)))
        i_known = float(jnp.mean(fisher(rung.known_pose)))
        contrasts.append(
            {
                "amplitude": a,
                "i_local": i_local,
                "i_known": i_known,
                "mean_d2_crlb": float(jnp.mean(rung.crlb)),
                "mean_d2_known": float(jnp.mean(rung.known_pose)),
                "seconds": round(time.time() - t0, 1),
            }
        )
        print(
            f"    a={a:<5} I_local={i_local:7.4f}  I_known={i_known:7.4f}  ({contrasts[-1]['seconds']:.0f}s)",
            flush=True,
        )
    return {
        "name": candidate.name,
        "box_size": box_size,
        "rmsd": float(pair.alignment.rmsd),
        "nodes": int(grid.angles.shape[0]),
        "amplitudes": contrasts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--rows", nargs="*", default=None, help="candidate names; default: all in panel")
    parser.add_argument("--data-dir", type=Path, default=Path("data/pdb"))
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--contrasts", type=float, nargs="+", default=[1.0, 0.1])
    parser.add_argument("--nodes", type=int, default=8064, help="Haar bank size for the E_R average")
    parser.add_argument("--box-size", type=int, default=128)
    parser.add_argument("--rmsf", type=float, default=SHARED.rmsf)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--min-shared-fraction", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--pixel-size", type=float, default=SHARED.pixel_size)
    parser.add_argument("--dose", type=float, default=SHARED.dose)
    args = parser.parse_args()

    plan = CollectionPlan(dose=args.dose, pixel_size=args.pixel_size)
    candidates, _ = load_panel(args.panel, data_dir=args.data_dir, fetch=args.fetch)
    by_name = {c.name: c for c in candidates}
    names = args.rows if args.rows else list(by_name)

    def payload(rows: list) -> dict:
        return {
            "driver": DRIVER,
            "commit": _commit(),
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "rows": rows,
        }

    rows = []
    for name in names:
        if name not in by_name:
            print(f"  {name}: not in panel, skipping", flush=True)
            continue
        print(f"  {name}:", flush=True)
        try:
            rows.append(measure_row(by_name[name], plan, args))
        except Exception as error:  # noqa: BLE001 - one row must fail alone
            print(f"    failed: {error}", flush=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload(rows), indent=2))

    if args.out:
        print(f"wrote {args.out} ({len(rows)} rows)", flush=True)
    else:
        print(json.dumps(payload(rows), indent=2), flush=True)


if __name__ == "__main__":
    main()
