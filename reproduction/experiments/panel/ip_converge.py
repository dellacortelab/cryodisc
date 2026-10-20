"""Measure where the per-particle Fisher information I(p) plateaus against bank size.

    CUDA_VISIBLE_DEVICES=6 .venv/bin/python reproduction/experiments/panel/ip_converge.py \
        --panel data/survey_panel_1.json --rows NAME_A NAME_B --data-dir data/pdb \
        --banks 504 2016 8064 33384 64128 --out results/ip_converge_1.json

The count quantity for the marginal rung is the Fisher information about the fraction, mapped from the
discriminability `d^2` (deflection) through the Gaussian model. Capped at `1 / (p (1 - p))`, it cannot
be Richardson-extrapolated and must be READ at a bank large enough to have plateaued. One stack per
row is scored against every bank (`--poses haar` makes it bank-independent), using the same scoring
primitives as `run_chain`, so the survey's I(p) run can be sized on evidence.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from cryodisc import CollectionPlan
from cryodisc.grids.poses import pose_bank
from cryodisc.microscope import Transfer
from cryodisc.particles.sim import simulate_stack
from cryodisc.particles.snr import streamed_whitened_power
from cryodisc.pipeline.chain import _pool_d2, _shift_lattice
from cryodisc.resolution import noise_spectrum
from cryodisc.rungs import streamed_pose_log_ratio
from cryodisc.statistics.information import fisher_from_deflection
from cryodisc.statistics.power import n_for_error_from_information
from reproduction.operating_point import SHARED
from reproduction.targets import load_panel

DRIVER = "reproduction/experiments/panel/ip_converge.py"


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


def measure_row(candidate, plan, args) -> dict:
    """I(p)(K) and d2(K) for one candidate, one stack scored against every bank."""
    pair = candidate.align(min_shared_fraction=args.min_shared_fraction)
    box_size = max(args.box_size, pair.alignment.minimum_box_size(plan.pixel_size))

    # run_chain's setup: under poses="haar" the stack is drawn from the true measure (angles=None), so
    # it does not depend on the bank and every rung below scores the SAME particles
    shifts = _shift_lattice(box_size, args.shift_sigma)
    model_a, model_b = pair.models(plan, box_size=box_size, form_factor="peng")
    spectrum = noise_spectrum(plan, box_size, rmsf=args.rmsf)

    stack_key, _ = jax.random.split(jax.random.key(args.seed))
    ref_angles = pose_bank(max(args.banks)).angles  # a converged bank: SNR is a mean over poses

    def simulate(contrast: float):
        return simulate_stack(
            models=(model_a, model_b),
            key=stack_key,
            n_particles=args.n_particles,
            noise_spectrum=spectrum,
            contrast=contrast,
            occupancy=(args.fraction, 1.0 - args.fraction),
            angles=None,  # haar: the true measure, bank-independent
            plan=plan,
            shift_sigma=args.shift_sigma,
        )

    def geometry_snr(transfer) -> float:
        """G: the per-particle matched-filter SNR at unit contrast; snr = contrast * G."""
        return float(
            jnp.sqrt(
                streamed_whitened_power(model_a.bare_contrast, ref_angles, spectrum, transfer=transfer, contrast=1.0)
            )
        )

    # SNR is the experiment-facing knob: solve contrast = snr / G from a unit-contrast calibration
    # draw (defocus, hence G, does not depend on contrast, so the same key reuses the same transfer).
    if args.snr is not None:
        calib = simulate(1.0)
        contrast = args.snr / geometry_snr(Transfer.from_defocus(calib.defocus, plan, box_size))
    else:
        contrast = args.contrast

    stack = simulate(contrast)
    transfer = Transfer.from_defocus(stack.defocus, plan, box_size)
    snr = contrast * geometry_snr(transfer)  # achieved per-particle SNR; == args.snr when set
    print(f"    contrast={contrast:.4g}  SNR={snr:.4g}", flush=True)
    is_a = stack.state == 0
    drawn = float(stack.fraction(0))

    rungs = []
    for k in sorted(args.banks):
        grid = pose_bank(k)
        actual_k = int(grid.angles.shape[0])
        t0 = time.time()
        ratios = streamed_pose_log_ratio(
            stack.images,
            model_a.bare_contrast,
            model_b.bare_contrast,
            grid.angles,
            grid.log_weights,
            spectrum,
            amplitude=contrast,  # MUST match the simulated stack, or the LLR is miscalibrated
            shifts=shifts,
            transfer=transfer,
            expansion=None,  # no expansion plan on the shift route (chain.py:533)
        )
        d2, d2_err = _pool_d2(ratios.marginal, is_a)
        information = float(fisher_from_deflection(d2, drawn))
        n = float(n_for_error_from_information(information, error_target=args.sigma, fraction=drawn))
        rungs.append(
            {
                "k_requested": int(k),
                "k": actual_k,
                "information": information,
                "d2": float(d2),
                "d2_err": float(d2_err),
                "n_for_error": n,
                "seconds": round(time.time() - t0, 1),
            }
        )
        print(
            f"    K={actual_k:>6d}  I(p)={information:6.3f}  d2={d2:8.2f}  n={n:9.0f}  ({rungs[-1]['seconds']:.0f}s)",
            flush=True,
        )
    return {
        "name": candidate.name,
        "box_size": box_size,
        "rmsd": float(pair.alignment.rmsd),
        "fraction": drawn,
        "amplitude": float(contrast),  # per-pair when --snr sets it; scores d^2 as a^2
        "snr": float(snr),  # achieved per-particle matched-filter SNR
        "rungs": rungs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--rows", nargs="*", default=None, help="candidate names; default: all in panel")
    parser.add_argument("--data-dir", type=Path, default=Path("data/pdb"))
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--banks", type=int, nargs="+", default=[504, 2016, 8064, 33384, 64128])
    parser.add_argument("--box-size", type=int, default=128)
    parser.add_argument(
        "--rmsf",
        type=float,
        default=SHARED.rmsf,
        help="specimen Debye-Waller floor in A; enters sigma^2 as a signal-damping envelope. Default 0.5 "
        "(B~20) is the ordered-protein floor. 0 = rigid atoms with unphysical high-k difference power.",
    )
    parser.add_argument("--shift-sigma", type=float, default=1.0)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--n-particles", type=int, default=4000)
    parser.add_argument(
        "--contrast",
        type=float,
        default=SHARED.contrast,
        help="template scale a; scales d^2 as a^2. Applied to BOTH the simulated stack and the scorer "
        "so the LLR stays calibrated. Default 1.0 matches the original ladders. Ignored when --snr is set.",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=None,
        help="target per-particle matched-filter SNR; sets contrast = snr / G per pair, so every pair "
        "sits at the same SNR. Overrides --contrast. This is the experiment-facing knob.",
    )
    parser.add_argument("--sigma", type=float, default=0.02, help="error target n_for_error is quoted at")
    parser.add_argument("--min-shared-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
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
        # write after EVERY row so a crash keeps the pairs already measured, and progress is visible
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload(rows), indent=2))

    if args.out:
        print(f"wrote {args.out} ({len(rows)} rows)", flush=True)
    else:
        print(json.dumps(payload(rows), indent=2), flush=True)


if __name__ == "__main__":
    main()
