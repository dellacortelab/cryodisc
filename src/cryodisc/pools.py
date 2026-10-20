"""A pool of real particles and the two states to score it against, from paths and nothing else.

Every real-data driver starts the same way: read a particle table, keep the states asked for, load
and recenter the images, and build the two forward models the log-ratio is taken between. Takes paths
and values, never a `Namespace`: the drivers keep their own CLIs and hand this layer what it needs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Int

from cryodisc.forward.cryojax_model import CryoJAXModel
from cryodisc.microscope import CollectionPlan, Transfer
from cryodisc.particles.images import load_stack
from cryodisc.particles.metadata import (
    ParticleMeta,
    load,
    recenter,
    select_states,
)
from cryodisc.particles.metadata import (
    check_shift_sign as measure_shift_sign,
)
from cryodisc.specimen.density import center_of_mass_offset, read_map, refuse_a_corner_origin
from cryodisc.specimen.structures import align_conformers


@dataclass(frozen=True)
class Pool:
    """A real particle set ready to score: the images and the CTF each was taken under.

    The clean input to `pipeline.fraction.estimate_fraction`, after a driver has done its own
    ingestion policy -- reading the table, refitting or grouping the CTF, recentering, inverting
    contrast. This layer knows none of that; it holds what the scorer needs and nothing else.

    Args:
        images: The particle images on the fft grid, `n` of them at `box_size`x`box_size`.
        transfer: The per-particle transfer, grouped by defocus (`particles.transfer_from_meta`).
        box_size: Image size in pixels; the templates are rendered to match.
        pixel_size: Angstroms per pixel, the scale the templates are built at.
        labels: True state per particle (`0` for A, `1` for B) when known, for validation only --
            the estimator never reads them. `None` on a real pool, whose composition is the unknown.
    """

    images: Complex[Array, "n y x"]
    transfer: Transfer
    box_size: int
    pixel_size: float
    labels: Int[Array, " n"] | None = None

    @property
    def n_particles(self) -> int:
        """How many particles the pool holds."""
        return int(self.images.shape[0])


def subsample_rows(n_rows: int, subsample: int, seed: int) -> np.ndarray | None:
    """Which rows a `--subsample` draw keeps, or `None` for all of them.

    Drawn UNIFORMLY at random, not from the front: a table prefix is one acquisition session's
    defocus, ice and drift. Returned SORTED so kept rows stay in table order, keeping `meta` aligned
    with any per-particle array written beside it.

    Args:
        n_rows: Rows available after `states` and `limit`.
        subsample: Cap. Zero, or a value at or above `n_rows`, keeps everything.
        seed: Seed for the draw.

    Returns:
        Ascending row indices, or `None` when nothing is dropped (so the caller skips indexing
        entirely).
    """
    if subsample <= 0 or subsample >= n_rows:
        return None
    return np.sort(np.random.default_rng(seed).choice(n_rows, size=subsample, replace=False))


def read_pool(
    meta_path: str | Path,
    *,
    root: str | Path | None = None,
    states: list[str] | None = None,
    limit: int = 0,
    subsample: int = 0,
    subsample_seed: int = 0,
    box_size: int | None = None,
    pixel_size: float | None = None,
    cs_pose_transpose: bool | None = None,
    invert_contrast: bool = False,
    recenter_images: bool = True,
    check_shift_sign: bool = False,
) -> tuple[ParticleMeta, jnp.ndarray, float, dict, np.ndarray | None]:
    """Metadata and images, with the pixel size and the recentering settled before anything is scored.

    Args:
        meta_path: The particle `.cs` or `.star`.
        root: Prefix for the stack paths the table records.
        states: Stack-name prefixes to keep, in the order that fixes which is state A.
        limit: Particles to keep per state. Zero keeps all of them.
        subsample: Keep at most this many particles, drawn uniformly at random over the whole
            selection after `limit` and `states`. Zero keeps all of them.
        subsample_seed: Seed for that draw.
        box_size: Expected box, checked against the stacks.
        pixel_size: Override the table's own A/px. `None` takes the table's median.
        cs_pose_transpose: Override `CS_POSE_TRANSPOSE` for this file, chosen at the reader alone.
        invert_contrast: Negate the stack after loading. Needed for a star-sourced stack in the other
            contrast convention, since `load_star` defaults the sign to +1.
        recenter_images: Apply the table's own shifts. False scores the images as stored.
        check_shift_sign: Measure the recentering sign from the images rather than inheriting it. A sign
            error is invisible in every summary statistic and DOUBLES the centering error -- but the
            measurement is NOT free of that failure either, naming a winner on any pool however close
            the field. Read the margin it records; see `particles.metadata.check_shift_sign`.

    Returns:
        `(meta, images, pixel_size, shift_report, labels)` with `images` as unshifted `fft2`.
        `labels` is the per-particle state index when more than one state was named and `None`
        otherwise -- the pool's truth, which a real collection never gives away.

    Rationale: docs/rationale/core-modules.md
    """
    reader_options = {} if cs_pose_transpose is None else {"pose_transpose": cs_pose_transpose}
    meta, labels = select_states(load(meta_path, **reader_options), states, limit=limit)

    # subsample BEFORE `load_stack`: the images are what cost, so drop rows before reading the stack
    keep = subsample_rows(len(meta), subsample, subsample_seed)
    if keep is not None:
        meta = meta[keep]
        if labels is not None:
            labels = labels[keep]

    settled = pixel_size if pixel_size else float(np.median(meta.pixel_size))
    stack = load_stack(meta, root, expected_box_size=box_size)
    if invert_contrast:
        stack = -stack
    images = jnp.fft.fft2(jnp.asarray(stack))

    shift_report: dict = {"applied": False}
    if meta.has_shift and recenter_images:
        offsets = jnp.asarray(meta.shift_in_pixels(settled))
        sign = 1.0
        if check_shift_sign:
            # record the margin: a thin win is what a reader of the result file cannot reconstruct
            sign, swap, margin, scores = measure_shift_sign(images, offsets, pixel_size=settled)
            shift_report |= {
                "measured_sign": sign,
                "swap_axes": swap,
                "margin": margin,
                "scores": {name: float(value) for name, value in scores.items()},
            }
            if swap:
                offsets = offsets[:, ::-1]
        images = recenter(images, offsets, sign)
        shift_report["applied"] = sign != 0.0
    return meta, images, settled, shift_report, labels


_AMPLITUDE_TOLERANCE = 1.2
"""How far a map pair's amplitudes may differ before the pair is refused, as a ratio.

Two states of one molecule hold the same total scattering, so their maps carry equal power and the
ratio sits at 1. The 1.2 margin admits the ordinary postprocessing differences between two deposited
maps while refusing a deposition-time normalization -- a flat scale across the band rather than a
B-factor -- that would otherwise bias every particle's log-ratio.
"""


def _amplitude_ratio(volume_a: np.ndarray, volume_b: np.ndarray) -> float:
    """`sqrt(power_a / power_b)`, each about its OWN mean.

    About its own mean because a deposited map's constant is solvent level, which says nothing about
    how much molecule is in the box; leaving it in makes the ratio depend on the masking rather than
    the density.

    Args:
        volume_a: State A's density.
        volume_b: State B's, on the same grid.

    Returns:
        The amplitude ratio, or `inf` where B carries no power at all.
    """
    power_a = float(np.sum((volume_a - volume_a.mean()) ** 2))
    power_b = float(np.sum((volume_b - volume_b.mean()) ** 2))
    return np.inf if power_b == 0.0 else float(np.sqrt(power_a / power_b))


def build_states(
    plan: CollectionPlan,
    *,
    box_size: int,
    pixel_size: float,
    structure_a: str | Path | None = None,
    structure_b: str | Path | None = None,
    map_a: str | Path | None = None,
    map_b: str | Path | None = None,
    assembly: str | None = "1",
    chains: list[str] | None = None,
    ligands: tuple[str, ...] = (),
    superpose: bool = True,
    offset_from_maps: list[str | Path] | None = None,
    ignore_origin: bool = False,
) -> tuple[list[CryoJAXModel], dict[str, Any]]:
    """The two states to score against, from either a structure pair or a pair of maps.

    Two routes because two kinds of ground truth exist. A deposited PAIR is two atomic models of one
    molecule, which `align_conformers` superposes and refuses where they do not correspond. A pair of
    MAPS is already in a common frame and has no atoms to correspond, and is deliberately NOT
    superposed: guessing a rotation between densities is a different measurement with its own failure
    mode.

    Args:
        plan: The collection, supplying the voltage the templates are rendered at.
        box_size: Image size the particles were extracted at.
        pixel_size: Angstroms per pixel, which the maps must agree with.
        structure_a: State A as a deposited model.
        structure_b: State B as a deposited model.
        map_a: State A as a map. Given, this takes the map route; passing a structure pair as well
            is refused rather than silently preferred.
        map_b: State B as a map.
        assembly: Biological assembly to build; `None` for the deposit.
        chains: Author chains to keep, applied before the assembly is expanded.
        ligands: Residue names to keep even where the two entries differ, so a bound-ligand
            difference reaches the templates. Empty keeps only shared atoms.
        superpose: Whether to superpose the structure pair.
        ignore_origin: Accept a map whose header origin sits away from the box center. Off; see
            `density.refuse_a_corner_origin` for why the header alone cannot always settle it.
        offset_from_maps: Move the rendered pair to these TWO maps' center-of-mass MIDPOINT, because
            `align_conformers` centers the pair on its two states averaged. State A's map alone
            under-corrects by half the inter-state shift -- a partial fix that looks like a wrong
            convention; see docs/rationale/core-modules.md.

    Returns:
        `(models, report)` -- two `CryoJAXModel`, and what they were built from, for the record.

    Raises:
        ValueError: If the arguments name no two states, or name them twice, or name only half of a
            pair. If a map's own voxel size disagrees with the particles' pixel size, which would
            rescale every spatial frequency invisibly, if a map is not the particles' box, if the two
            maps' amplitudes differ by more than `_AMPLITUDE_TOLERANCE`, if `offset_from_maps` does
            not name exactly two maps, or if one of those records no voxel size of its own.
    """
    # the route refusal lives here too, not only in a driver's CLI check: a library caller with a
    # bad combination would otherwise pick a route silently
    structures = structure_a is not None or structure_b is not None
    maps = map_a is not None or map_b is not None
    if structures and maps:
        raise ValueError("pass either structure_a/_b or map_a/_b, not both: they name the same two states")
    if not structures and not maps:
        raise ValueError("need two states: structure_a/_b for atomic models, or map_a/_b for voxel maps")
    if structures and not (structure_a and structure_b):
        raise ValueError("structure_a and structure_b are both required: a log-ratio takes two states")
    if maps and not (map_a and map_b):
        raise ValueError("map_a and map_b are both required: a log-ratio takes two states")
    # `offset_from_maps` is read in the structure branch alone; a map pair is already in the
    # particles' frame and has nothing to offset, so naming one on the map route is a wrong call
    if maps and offset_from_maps:
        raise ValueError(
            "offset_from_maps moves a rendered ATOMIC pair onto the maps' center of mass, and the map "
            "route has no atoms to move: the maps ARE the states and already carry their own frame"
        )

    # `or map_b is None` is dead after the route guard, but narrows both to `str | Path` for mypy
    if map_a is None or map_b is None:
        offset = None
        if offset_from_maps:
            if len(offset_from_maps) != 2:
                raise ValueError(
                    f"the offset is the MIDPOINT of the two states' maps, so it needs exactly two, got "
                    f"{len(offset_from_maps)}. One map alone under-corrects by half the inter-state "
                    "shift, which is a partial fix that reads as a wrong convention rather than failing."
                )
            # each map's OWN voxel size, not the particles' pixel size: `center_of_mass_offset`
            # converts a voxel centroid to angstroms, so a wrong scale silently rescales the offset.
            # The offset is in angstroms and applied to ATOMS, so these maps need not be on the
            # particles' grid -- only to record their own grid.
            offsets = []
            for path in offset_from_maps:
                volume, voxel, origin = read_map(path)
                if voxel is None:
                    raise ValueError(
                        f"{path} records no voxel size, and the center of mass is measured in VOXELS -- "
                        "defaulting one scales the offset by an unknown factor"
                    )
                refuse_a_corner_origin(path, origin, voxel, volume.shape, ignore_origin)
                offsets.append(center_of_mass_offset(volume, voxel))
            offset = sum(offsets) / len(offsets)
        pair = align_conformers(
            structure_a,
            structure_b,
            assembly=assembly,
            chains=chains,
            ligands=ligands,
            superpose=superpose,
            offset=offset,
        )
        return list(pair.models(plan, box_size=box_size)), {
            "from": "structures",
            "state_a": str(structure_a),
            "state_b": str(structure_b),
            "ca_rmsd": pair.alignment.rmsd,
            "n_matched": pair.alignment.n_matched,
            "superposed": superpose,
            "offset_in_angstroms": None if offset is None else [float(value) for value in offset],
        }

    models, volumes = [], []
    for path in (map_a, map_b):
        volume, voxel, origin = read_map(path)
        if voxel is None:
            raise ValueError(f"{path} records no voxel size, and defaulting one rescales every frequency")
        refuse_a_corner_origin(path, origin, voxel, volume.shape, ignore_origin)
        if abs(voxel - pixel_size) > 1e-3:
            raise ValueError(
                f"{path} is {voxel:.4g} A/voxel against the particles' {pixel_size:.4g} A/pixel. A "
                "template bank is not transferable across pixel sizes; resample the map first, or "
                "pass --pixel-size if the particles' own value is the wrong one."
            )
        if volume.shape[-1] != box_size:
            raise ValueError(f"{path} is {volume.shape[-1]} voxels against the particles' box of {box_size}")
        volumes.append(np.asarray(volume, float))
        models.append(
            CryoJAXModel.from_voxel_grid(
                volume, pixel_size=pixel_size, box_size=box_size, voltage_in_kilovolts=plan.voltage_in_kv
            )
        )
    ratio = _amplitude_ratio(*volumes)
    if not 1 / _AMPLITUDE_TOLERANCE <= ratio <= _AMPLITUDE_TOLERANCE:
        raise ValueError(
            f"{map_a} carries {ratio:.3g}x the amplitude of {map_b}. Two states of one molecule hold "
            "the same total scattering, so this is a deposition-time normalization and the "
            "calibration cannot absorb it: one amplitude is fitted for the PAIR, so an imbalance "
            "adds a constant to every particle's log-ratio whatever its state, and the pool that "
            "should read 0 reads 1 instead. Rescale one map onto the other's power first. The ratio "
            "is measured on the whole box about each map's own mean."
        )
    return models, {
        "from": "maps",
        "state_a": str(map_a),
        "state_b": str(map_b),
        "amplitude_ratio": float(ratio),
    }
