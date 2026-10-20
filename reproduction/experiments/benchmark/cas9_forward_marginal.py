"""Ab-initio pose-marginal forward-simulation d'^2 for a two-state system (SpCas9).

Simulate pure pools from the two reconstructed state MAPS at the dataset's REAL recorded viewing
angles, then score them pose-marginalized over a bank -- the forward model's own prediction of the
attainable d'^2_obs that the known-pose closed form (upper bound) and the FSC route bracket. The
closed-form known-pose d'^2 is computed from the same map templates as a deterministic anchor: the
noise is scaled so it equals a stated target (the real MAP known-pose), which pins the absolute SNR
via the physical dose-model noise shape. Pose efficiency = marginal / known-pose is the transferable
readout. Inverse-crime by construction (particles come from the maps that score them), so this is the
forward model's marginal free of real-data model mismatch -- an upper bound on real marginalization.

One bank per run (--fixed-bank) so a convergence ladder runs as independent jobs and is assembled
offline, the pattern score_pool.py's marginal arm uses for a large real pool.
"""

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from cryodisc import CollectionPlan
from cryodisc.forward.cryojax_model import CryoJAXModel
from cryodisc.grids.poses import (
    euler_from_relion,
    expansion_plan_or_none,
    pose_bank,
    realized_nodes,
)
from cryodisc.microscope.transfer import Transfer
from cryodisc.particles.metadata import band_limit_envelope
from cryodisc.particles.metadata import load as load_meta
from cryodisc.particles.sim import simulate_stack
from cryodisc.resolution import noise_spectrum
from cryodisc.rungs.marginal import known_pose_d2, streamed_pose_log_ratio
from cryodisc.specimen.density import read_map
from cryodisc.statistics.information import deflection
from reproduction.record import write_result

DRIVER = "reproduction/experiments/benchmark/cas9_forward_marginal.py"


def _model_from_map(path: Path, pixel_size: float, box_size: int) -> CryoJAXModel:
    """A forward model whose templates are projections of a reconstructed map."""
    volume, _, _ = read_map(path)
    return CryoJAXModel.from_voxel_grid(jnp.asarray(volume, float), pixel_size=pixel_size, box_size=box_size)


def _real_angles(meta_path: Path, limit: int, seed: int) -> np.ndarray:
    """Recorded viewing angles for a pool, in the renderer's convention, capped at `limit`."""
    meta = load_meta(meta_path)
    if not meta.has_euler:
        raise ValueError(f"{meta_path} records no viewing direction (a 3D refinement writes alignments3D/pose)")
    angles = np.asarray(euler_from_relion(meta.euler))
    if limit and angles.shape[0] > limit:
        angles = angles[np.random.default_rng(seed).permutation(angles.shape[0])[:limit]]
    return angles


def _band_limited_transfer(defocus, plan: CollectionPlan, box_size: int, band) -> Transfer:
    """The per-group CTF transfer for a simulated stack, with the hard band mask folded in."""
    transfer = Transfer.from_defocus(jnp.asarray(defocus), plan, box_size)
    if band is None:
        return transfer
    return eqx.tree_at(lambda t: t.values, transfer, transfer.values * band)


def _simulate_pool(model_a, model_b, angles, *, state, plan, noise, contrast, box_size, band, seed):
    """One pure pool of `state` (0 or 1) rendered at the given angles, with its scoring transfer."""
    stack = simulate_stack(
        (model_a, model_b),
        key=jax.random.key(seed),
        n_particles=angles.shape[0],
        noise_spectrum=noise,
        contrast=contrast,
        occupancy=jnp.asarray([1.0, 0.0] if state == 0 else [0.0, 1.0]),
        plan=plan,
        angles=jnp.asarray(angles),
    )
    return stack.images, _band_limited_transfer(stack.defocus, plan, box_size, band)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    angles_a = _real_angles(args.meta_a, args.limit, args.seed)
    angles_b = _real_angles(args.meta_b, args.limit, args.seed + 1)
    box_size = args.box_size or int(read_map(args.map_a)[0].shape[-1])
    model_a = _model_from_map(args.map_a, args.pixel_size, box_size)
    model_b = _model_from_map(args.map_b, args.pixel_size, box_size)

    plan = CollectionPlan(dose=args.dose, pixel_size=args.pixel_size)
    band = band_limit_envelope(box_size, args.pixel_size, args.band_limit) if args.band_limit else None
    noise_base = noise_spectrum(plan, box_size, rmsf=args.rmsf)

    # ANCHOR: the closed-form known-pose d'^2 is deterministic in the templates, noise and transfer.
    # Scale the noise so it equals the stated target (the real MAP known-pose), which d'^2 ~ 1/noise
    # makes a single factor -- this is what pins the absolute SNR the marginal is then measured at.
    templates_a = model_a.bare_contrast(jnp.asarray(angles_a))
    templates_b = model_b.bare_contrast(jnp.asarray(angles_a))
    anchor_transfer = _band_limited_transfer(
        jnp.full((angles_a.shape[0],), plan.defocus_range[0]), plan, box_size, band
    )
    known_unit = float(jnp.mean(known_pose_d2(templates_a, templates_b, noise_base, transfer=anchor_transfer)))
    noise_scale = known_unit / args.target_known_pose if args.target_known_pose else 1.0
    noise = noise_base * noise_scale
    known_pose = known_unit / noise_scale  # == target when anchored, else the physical-noise value

    images_a, transfer_a = _simulate_pool(
        model_a,
        model_b,
        angles_a,
        state=0,
        plan=plan,
        noise=noise,
        contrast=args.contrast,
        box_size=box_size,
        band=band,
        seed=args.seed + 10,
    )
    images_b, transfer_b = _simulate_pool(
        model_a,
        model_b,
        angles_b,
        state=1,
        plan=plan,
        noise=noise,
        contrast=args.contrast,
        box_size=box_size,
        band=band,
        seed=args.seed + 20,
    )

    grid = pose_bank(args.fixed_bank)
    nodes = realized_nodes(args.fixed_bank)
    expansion = expansion_plan_or_none(grid, box_size)

    def score(images, transfer):
        return streamed_pose_log_ratio(
            images,
            model_a.bare_contrast,
            model_b.bare_contrast,
            grid.angles,
            grid.log_weights,
            noise,
            amplitude=args.contrast,
            transfer=transfer,
            expansion=expansion,
        ).marginal

    marginal_a = score(images_a, transfer_a)
    marginal_b = score(images_b, transfer_b)
    d2_marginal = float(deflection(marginal_a, marginal_b))
    pose_efficiency = d2_marginal / known_pose if known_pose > 0 else None

    print(f"nodes {nodes}   box {box_size}   pixel {args.pixel_size} A   band {args.band_limit} /A")
    print(f"known-pose d'^2 {known_pose:.4g} (anchor target {args.target_known_pose})   noise scale {noise_scale:.4g}")
    print(f"marginal  d'^2 {d2_marginal:.4g}   pose efficiency {pose_efficiency:.4g}   obs {args.d2_obs}")

    write_result(
        args.out,
        {
            "system": args.system,
            "nodes": int(nodes),
            "fixed_bank_requested": int(args.fixed_bank),
            "box_size": box_size,
            "pixel_size": args.pixel_size,
            "band_limit": args.band_limit,
            "dose": args.dose,
            "rmsf": args.rmsf,
            "contrast": args.contrast,
            "n_a": int(angles_a.shape[0]),
            "n_b": int(angles_b.shape[0]),
            "d2_known_pose": known_pose,
            "d2_known_pose_unit_noise": known_unit,
            "noise_scale": noise_scale,
            "target_known_pose": args.target_known_pose,
            "d2_marginal": d2_marginal,
            "pose_efficiency": pose_efficiency,
            "d2_obs": args.d2_obs,
            "marginal_over_obs": d2_marginal / args.d2_obs if args.d2_obs else None,
            "convention": "inverse-crime map forward sim at REAL recorded angles; centered (no shift "
            "marginalization); noise anchored so closed-form known-pose == target_known_pose",
        },
        driver=DRIVER,
        arguments=vars(args),
    )
    print(f"\n-> {args.out}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line. Every path and operating-point value is an argument."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", default="cas9")
    parser.add_argument("--map-a", type=Path, required=True, help="state-A full map (build_full_maps)")
    parser.add_argument("--map-b", type=Path, required=True, help="state-B full map")
    parser.add_argument("--meta-a", type=Path, required=True, help="state-A pose file (cryoSPARC .cs)")
    parser.add_argument("--meta-b", type=Path, required=True, help="state-B pose file")
    parser.add_argument("--fixed-bank", type=int, required=True, metavar="K", help="pose bank size for this run")
    parser.add_argument("--limit", type=int, default=20000, help="cap each pool at N recorded angles")
    parser.add_argument("--pixel-size", type=float, default=1.3)
    parser.add_argument("--box-size", type=int, default=None, help="defaults to the map dimension")
    parser.add_argument("--dose", type=float, default=40.0)
    parser.add_argument("--band-limit", type=float, default=0.125, help="hard score cutoff, inverse A")
    parser.add_argument("--rmsf", type=float, default=0.0, help="intra-state fluctuation; 0 for a map state")
    parser.add_argument("--contrast", type=float, default=1.0, help="template scale vs noise")
    parser.add_argument(
        "--target-known-pose",
        type=float,
        default=None,
        help="scale noise so the closed-form known-pose d'^2 equals this (the real MAP known-pose), "
        "pinning absolute SNR. Omit to run at the physical dose-model noise unscaled",
    )
    parser.add_argument("--d2-obs", type=float, default=2.906, help="the measured d'^2_obs, for the printed ratio")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
