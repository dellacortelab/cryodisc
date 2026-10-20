"""The consensus template production picks against, and the check that its two states were superposed.

Class averaging favors high-contrast particles; if that selection is even weakly state-correlated
it carries fraction information, and a recovered fraction is then partly reading the picker. Picking
against the consensus `(V_A + V_B) / 2` cannot prefer a state by construction. Building the template
lives here too so the two drivers that build it cannot drift.

Rationale: docs/rationale/preprocess.md
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Float

from cryodisc.specimen.density import voxel_centroid

if TYPE_CHECKING:
    from cryodisc.specimen.structures import ConformerPair

RENDER_OFFSET_TOLERANCE = 1.0
"""Centroid disagreement in ANGSTROMS, above which the render is called broken rather than the input.

An assertion on the RENDERER: `consensus_template` builds both states from one `align_conformers`
superposition, so anything above about a voxel is a bug in the render path or in the (x, y, z) to
(z, y, x) conjugation `density` applies, not a frame the user got wrong. Worth checking because a
template that is a blur of two poses is shared by every pool, so no comparison between pools shows it.

Angstroms and not voxels, because it is a statement about a MOLECULE: a conformational change
conserves mass and moves a compact particle's centroid by well under one. In voxels the same pair's
threshold would depend on the pixel size, which the caller often does not choose.
"""


def consensus(volume_a: np.ndarray, volume_b: np.ndarray) -> np.ndarray:
    """The state-neutral picking template, `(V_A + V_B) / 2`.

    Args:
        volume_a: State A map.
        volume_b: State B map, on the same grid.

    Returns:
        The consensus map.

    Raises:
        ValueError: If the two maps are not on the same grid.
    """
    a, b = np.asarray(volume_a, float), np.asarray(volume_b, float)
    if a.shape != b.shape:
        raise ValueError(f"the two maps are on different grids: {a.shape} against {b.shape}")
    return 0.5 * (a + b)


def center_of_mass_separation(volume_a: np.ndarray, volume_b: np.ndarray) -> float:
    """Distance in VOXELS between the two maps' centers of mass.

    The check the consensus cannot make on itself: two states that were never superposed give a
    consensus that is a blur of two poses, which a picker uses and simply picks worse. A pure
    conformational change moves the center of mass by well under a voxel, so anything above about one
    is a frame error, not a motion.

    Args:
        volume_a: State A map.
        volume_b: State B map, on the same grid.

    Returns:
        The distance in voxels.

    Raises:
        ValueError: If the two maps are not on the same grid, or either carries no positive mass.
    """
    a, b = np.asarray(volume_a, float), np.asarray(volume_b, float)
    if a.shape != b.shape:
        raise ValueError(f"the two maps are on different grids: {a.shape} against {b.shape}")
    centers = []
    for volume in (a, b):
        # clipped at zero: a banded map carries small negative lobes from the raised-cosine taper,
        # and a signed "mass" has no centroid worth the name. Clipped here, not in `voxel_centroid`,
        # which serves a caller that wants the map it was given
        mass = np.maximum(volume, 0.0)
        if mass.sum() <= 0.0:
            raise ValueError("a map carries no positive density, so it has no center of mass to compare")
        centers.append(voxel_centroid(mass))
    # a DIFFERENCE of two centroids, so the box center cancels and no origin convention enters
    return float(np.linalg.norm(centers[0] - centers[1]))


@dataclass(frozen=True)
class ConsensusTemplate:
    """Both states rendered on one grid and banded, with the consensus between them.

    Args:
        state_a: State A, low-passed to the template band. A CONTROL template, never production.
        state_b: State B, on the same grid and the same band. Likewise.
        middle: `(V_A + V_B) / 2` of the two banded maps -- the production picker and the production
            starting reference.
        offset: Distance in ANGSTROMS between the two RAW renders' centers of mass, measured before
            the band. Returned as well as checked, because it is the one number that says the render
            path is still sane and no caller can recover it from the maps afterwards.
        minimum_box_size: Voxels this molecule needs at this pixel size, from
            `alignment.minimum_box_size`. Carried on the result so a reader of the template has the
            number without the `Alignment` in hand; what to DO about a box below it is the caller's.
    """

    state_a: Float[np.ndarray, "z y x"]
    state_b: Float[np.ndarray, "z y x"]
    middle: Float[np.ndarray, "z y x"]
    offset: float
    minimum_box_size: int


def consensus_template(
    pair: "ConformerPair",
    *,
    box_size: int,
    pixel_size: float,
    resolution: float,
) -> ConsensusTemplate:
    """Render both states on one grid, band them, and build the state-neutral map between them.

    Rendering here (not reading maps) makes superposition, the scored assembly and a shared grid hold
    by construction. Both states come back band-limited, and the consensus is the mean of the two
    BANDED maps, not the band of their mean. A box below the molecule's floor is not refused here (the
    floor comes back on the result); that is the caller's policy.

    Args:
        pair: The two conformers in one frame, from `structures.align_conformers`.
        box_size: Edge of the render grid, in voxels.
        pixel_size: A/voxel of that grid.
        resolution: Low-pass applied to both states, in A.

    Returns:
        The banded states, the consensus, the centroid offset and the box floor.

    Raises:
        ValueError: If the two rendered states disagree about their centroid by more than
            `RENDER_OFFSET_TOLERANCE`.
    """
    from cryodisc.specimen.density import from_model, lowpass

    model_a, model_b = pair.models_on_grid(box_size=int(box_size), pixel_size=float(pixel_size))
    volume_a, volume_b = from_model(model_a), from_model(model_b)

    # Measured on the RAW renders, before the band: the raised-cosine taper puts small negative
    # lobes into the banded maps that would corrupt the centroid.
    offset = center_of_mass_separation(volume_a, volume_b) * pixel_size
    if offset > RENDER_OFFSET_TOLERANCE:
        raise ValueError(
            f"the two RENDERED states' centers of mass are {offset:.3f} A apart, above "
            f"{RENDER_OFFSET_TOLERANCE:g}. Both were built from one superposition on one image "
            "config, so this is a bug in the render path and not a frame that needs fixing."
        )

    banded_a = lowpass(volume_a, pixel_size, resolution)
    banded_b = lowpass(volume_b, pixel_size, resolution)
    return ConsensusTemplate(
        state_a=banded_a,
        state_b=banded_b,
        middle=consensus(banded_a, banded_b),
        offset=float(offset),
        minimum_box_size=int(pair.alignment.minimum_box_size(pixel_size)),
    )
