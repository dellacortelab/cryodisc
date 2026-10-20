"""Does our projection model reproduce a multislice one? The two-engine gate.
A bank cannot certify itself, so the gate compares the projection bank against a Parakeet multislice
one shell by shell and pose by pose (`validation.bank`). Two subcommands: `render` writes the GPU-only
multislice reference once, and `compare` is a cheap read that re-gates a cryoJAX change against it.
    .venv/bin/python reproduction/experiments/forward/gate.py render --structure a.pdb --out banks/a_multislice.npz
    .venv/bin/python reproduction/experiments/forward/gate.py compare --structure a.pdb --reference banks/a_multislice.npz
The poses are written with the bank and read back, not regenerated (differing poses would read as a
forward-model deficit); a median NCC is not a gate, so the verdict reads per-shell level and shape.
Rationale: docs/rationale/reproduction/experiments/gate.md
"""

import argparse
import dataclasses
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from cryodisc.forward.cryojax_model import CryoJAXModel
from cryodisc.forward.optics import CTFParams
from cryodisc.grids.poses import pose_bank
from reproduction.operating_point import SHARED
from reproduction.record import write_result

# the `research` imports below cross into a repo-root package (only `src/` is installed), so they
# come after the `cryodisc` imports above rather than beside them.
from reproduction.validation.bank import bank_verdict, compare_engines, ctf_sampling_limit
from reproduction.validation.parakeet import MultisliceConfig, laptop_config, render_bank
from reproduction.validation.template_bank import CTF_FREE, TemplateBank

_N_SHELLS = 8

DRIVER = "reproduction/experiments/forward/gate.py"


def save_bank(bank: TemplateBank, path: Path) -> Path:
    """Write a bank and everything needed to compare against it later.

    The provenance travels WITH the images. A bank on disk whose box, pixel, voltage, defocus and
    poses have to be remembered separately is a bank that will eventually be gated against something
    it does not match, and the gate would report that as a forward-model deficit.

    Args:
        bank: The bank.
        path: Where to write, as `.npz`.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images=np.asarray(bank.images),
        angles=np.asarray(bank.angles),
        pixel_size=bank.pixel_size,
        voltage_in_kv=bank.voltage_in_kv,
        # `None` does not survive npz, so CTF_FREE is written as a NaN and read back as None
        defocus_in_angstroms=np.nan if bank.defocus_in_angstroms is CTF_FREE else bank.defocus_in_angstroms,
        engine=bank.engine,
    )
    return path


def load_bank(path: Path) -> TemplateBank:
    """Read a bank written by `save_bank`, provenance and all.

    Args:
        path: The `.npz`.

    Returns:
        The bank, with `defocus_in_angstroms` back to `CTF_FREE` where it was written NaN.
    """
    stored = np.load(path, allow_pickle=False)
    defocus = float(stored["defocus_in_angstroms"])
    return TemplateBank(
        images=jnp.asarray(stored["images"]),
        angles=jnp.asarray(stored["angles"]),
        pixel_size=float(stored["pixel_size"]),
        voltage_in_kv=float(stored["voltage_in_kv"]),
        defocus_in_angstroms=CTF_FREE if np.isnan(defocus) else defocus,
        engine=str(stored["engine"]),
    )


def cryojax_bank(structure: Path, angles, config: MultisliceConfig) -> TemplateBank:
    """The projection bank, rendered to match a multislice one exactly.

    Args:
        structure: Atomic model.
        angles: `(p, 3)` Euler angles in degrees, from the reference bank when there is one.
        config: The multislice settings, which supply the box, pixel, voltage and defocus so the two
            engines cannot be given different ones.

    Returns:
        The bank, at the same defocus the reference used.
    """
    model = CryoJAXModel.from_structure(
        structure,
        box_size=config.box_size,
        pixel_size=config.pixel_size,
        voltage_in_kilovolts=config.voltage_in_kv,
    )
    optics = CTFParams(
        defocus_in_angstroms=config.defocus_in_angstroms,
        spherical_aberration_in_mm=config.spherical_aberration_in_mm,
        b_factor=0.0,
    )
    return TemplateBank.from_model(model, angles, ctf_params=optics, defocus_in_angstroms=config.defocus_in_angstroms)


_OPTICS = {
    "voltage": ("voltage_in_kv", 300.0),
    "defocus": ("defocus_in_angstroms", 5000.0),
    "solvent": ("solvent", False),
}
"""Optics arguments a caller may override, as `argument -> (MultisliceConfig field, default)`.

The default lives HERE and not in `add_argument`, which declares `None`: only that makes a stated
argument distinguishable from a defaulted one, which under `--laptop` is the difference between
rendering at the stated 5000 A and at the preset's 12000 A.
"""


def config_from(args: argparse.Namespace) -> MultisliceConfig:
    """The render settings, from the laptop preset or the arguments, which override it field by field."""
    stated = {field: value for name, (field, _) in _OPTICS.items() if (value := getattr(args, name)) is not None}
    if args.laptop:
        return dataclasses.replace(laptop_config(box_size=args.box_size, pixel_size=args.pixel_size), **stated)
    return MultisliceConfig(
        box_size=args.box_size,
        pixel_size=args.pixel_size,
        dose=None,
        slice_thickness_in_angstroms=5.0,  # the gate reads the CTF at 5 A steps, not the class 3 A default
        **(dict(_OPTICS.values()) | stated),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line. Every path is an argument; nothing is read from a constant."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "mode", choices=("render", "compare"), help="render the multislice reference, or gate against it"
    )
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="the .npz to write (render) or the result JSON (compare)"
    )
    parser.add_argument("--reference", type=Path, help="the multislice bank to gate against, required by compare")
    parser.add_argument("--box-size", type=int, default=64)
    parser.add_argument("--pixel-size", type=float, default=SHARED.pixel_size)
    parser.add_argument("--voltage", type=float, default=None, help="kV; default 300.0")
    # 5000 rather than 12000: a gate is judged only where the CTF is SAMPLED, and a higher defocus
    # pushes the sampling limit coarser than Nyquist, hiding the band the gate should read
    parser.add_argument("--defocus", type=float, default=None, help="underfocus in A; default 5000.0")
    parser.add_argument(
        "--nodes", type=int, default=16, help="poses to render the comparison over; `poses.pose_bank` quantizes it"
    )
    parser.add_argument("--laptop", action="store_true", help="coarse, unconverged settings for exercising the wiring")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="where the intermediate .yaml and .h5 land -- they run to gigabytes, so a cluster wants "
        "node-local scratch. Default is the system temporary directory",
    )
    # OFF by default, since the other engine has no solvent model: rendering the reference in ice adds a
    # field one side cannot, so the gate cannot pass. On asks what a solvent slab costs; off gates the integrator.
    parser.add_argument(
        "--solvent",
        action="store_true",
        default=None,
        help="render the reference in amorphous ice, which cryoJAX does not model. Default off",
    )
    args = parser.parse_args(argv)
    if args.mode == "compare" and args.reference is None:
        parser.error("compare needs --reference: a gate has nothing to measure against without one")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from(args)

    if args.mode == "render":
        angles = pose_bank(args.nodes).angles
        bank = render_bank(args.structure, angles, config, work_dir=args.work_dir)
        save_bank(bank, args.out)
        print(f"rendered {len(bank)} multislice templates -> {args.out}")
        return 0

    reference = load_bank(args.reference)
    # the poses come off the reference rather than being regenerated: a gate whose banks sit at
    # different poses measures the pose convention, and no spectral statistic can see that
    under_test = cryojax_bank(args.structure, reference.angles, config)
    comparison = compare_engines(under_test, reference, n_shells=_N_SHELLS)
    verdict = bank_verdict(comparison)

    # the band is stated BEFORE the verdict, because a gate judged past the CTF sampling limit is
    # measuring the sampling rather than the physics
    sampled = (
        1.0
        / ctf_sampling_limit(
            reference.box_size, reference.pixel_size, config.defocus_in_angstroms, reference.voltage_in_kv
        )
        * reference.pixel_size
    )
    print(f"{under_test.engine} against {reference.engine}, {len(reference)} templates")
    print(
        f"  CTF sampled to {sampled:.1f} A   (Nyquist {2 * reference.pixel_size:.1f} A) -- shells past this are excluded"
    )
    print(f"  median NCC     {float(np.median(np.asarray(comparison.correlation))):.4f}")
    print(f"  worst level    {verdict.level_worst:.4f}   worst shape {verdict.shape_worst:.4f}")
    # `inf` is what `certified_resolution` carries when NOTHING is certified (the band ends at DC), so
    # it is printed as "NOTHING" rather than as a number reading "certified to infinite resolution"
    certified = f"{verdict.certified_resolution:.1f} A" if np.isfinite(verdict.certified_resolution) else "NOTHING"
    # the band's COARSE end too: it is not DC whenever `min_transfer` excluded the lowest shells
    low = verdict.certified_low_resolution
    span = f"   (from {low:.1f} A; coarser shells were not judged)" if np.isfinite(low) else ""
    print(f"  certified to   {certified}{span}")
    print(f"  verdict        {'PASS' if verdict.passes else 'FAIL'}")

    write_result(
        args.out,
        {
            "passes": verdict.passes,
            "passes_correlation": verdict.passes_correlation,
            "passes_band": verdict.passes_band,
            "level_worst": verdict.level_worst,
            "shape_worst": verdict.shape_worst,
            "certified_from_k": verdict.certified_from_k,
            "certified_k": verdict.certified_k,
            "certified_low_resolution": verdict.certified_low_resolution,
            "certified_resolution": verdict.certified_resolution,
            "failing_level": verdict.failing_level.tolist(),
            "failing_shape": verdict.failing_shape.tolist(),
            "median_ncc": float(np.median(np.asarray(comparison.correlation))),
            "amp_ratio": np.asarray(comparison.amp_ratio).tolist(),
            "frc": np.asarray(comparison.frc).tolist(),
            "n_templates": len(reference),
            "engines": [under_test.engine, reference.engine],
            "provenance": list(reference.provenance()),
            "sampled_resolution": sampled,
        },
        driver=DRIVER,
        arguments=vars(args),
    )
    print(f"\n-> {args.out}")
    return 0 if verdict.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
