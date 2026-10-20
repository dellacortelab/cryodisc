"""The consensus template production picks against, and the check that its two states were superposed.

Picking against `(V_A + V_B) / 2` cannot prefer a state by construction, and `consensus_template` is
the one pipeline that builds it. The check the consensus cannot make on itself is that the two states
were superposed at all, which `center_of_mass_separation` supplies.
"""

import numpy as np
import pytest
from conftest import hinged_pair

from reproduction.datasets.picking import (
    RENDER_OFFSET_TOLERANCE,
    center_of_mass_separation,
    consensus,
    consensus_template,
)


def blob(box, center, radius=6.0):
    """A ball of density centered at `center`, as a cubic map."""
    z, y, x = np.indices((box, box, box))
    return (np.sqrt((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2) <= radius).astype(float)


@pytest.fixture(scope="module")
def bent_pair(tmp_path_factory):
    """Two conformers of one molecule, differing by a hinge rather than by a rigid displacement.

    A rigid shift is the failure `center_of_mass_separation` exists to catch, so the pair must not be
    one: a real conformational change conserves mass and moves the centroid by well under a voxel,
    which is what makes the sub-voxel offset asserted below an assertion on the renderer.
    """
    return hinged_pair(tmp_path_factory.mktemp("bent"))


def test_the_consensus_sits_exactly_between_the_two_states():
    a, b = blob(32, (16, 16, 14)), blob(32, (16, 16, 18))
    middle = consensus(a, b)
    assert np.linalg.norm(middle - a) == pytest.approx(np.linalg.norm(middle - b))
    with pytest.raises(ValueError, match="different grids"):
        consensus(a, blob(24, (12, 12, 12)))


def test_two_unsuperposed_maps_are_visible_in_their_centers_of_mass():
    """Every distance between a consensus and its two states is equal by construction, so the usual
    symmetry report says nothing. What can actually be wrong is that the maps were never superposed,
    and the consensus of two unaligned maps is a blur of two poses."""
    assert center_of_mass_separation(blob(32, (16, 16, 16)), blob(32, (16, 16, 16))) == pytest.approx(0.0)
    assert center_of_mass_separation(blob(32, (16, 16, 12)), blob(32, (16, 16, 20))) == pytest.approx(8.0, abs=0.2)
    with pytest.raises(ValueError, match="no positive density"):
        center_of_mass_separation(np.zeros((16, 16, 16)), blob(16, (8, 8, 8), radius=3.0))


def test_the_centers_of_mass_are_compared_axis_by_axis_in_the_arrays_own_order():
    """The distance is a norm over three axes, so a centroid returned in the wrong axis order would
    still be a plausible number on a symmetric test map and wrong on every real one."""
    displaced = center_of_mass_separation(blob(32, (16, 16, 16)), blob(32, (16, 22, 16)))
    assert displaced == pytest.approx(6.0, abs=0.2)
    # the same displacement moved to another axis has to give the same distance, and a map whose
    # negative lobes were counted as mass would not
    assert center_of_mass_separation(blob(32, (10, 16, 16)) - 0.5, blob(32, (16, 16, 16)) - 0.5) == pytest.approx(
        6.0, abs=0.2
    )


def test_the_template_pipeline_renders_both_states_bands_them_and_averages(bent_pair):
    """One function so the same map is built the same way wherever it is needed: `cryosparc_chain.py
    refine` renders the map every pool starts from. Two copies of this could drift into rendering one
    molecule two different ways, and the refinement would not fail on it -- it would simply do worse."""
    template = consensus_template(bent_pair, box_size=32, pixel_size=2.0, resolution=20.0)

    assert template.middle.shape == (32, 32, 32)
    assert template.state_a.shape == template.state_b.shape == template.middle.shape
    assert template.minimum_box_size == bent_pair.alignment.minimum_box_size(2.0)
    # equidistant from the two states, which is the property the consensus is chosen FOR
    assert np.linalg.norm(template.middle - template.state_a) == pytest.approx(
        np.linalg.norm(template.middle - template.state_b), rel=1e-6
    )
    # rendered from one superposition on one image config, so this is an assertion on the RENDERER
    assert template.offset < RENDER_OFFSET_TOLERANCE
    # and what comes back is the BAND, not the raw render: a Peng potential is nearly point-like, so
    # an unbanded map is a sampling pattern rather than a molecule
    assert (template.state_a > 0.01 * template.state_a.max()).mean() > 0.02


def test_a_box_below_the_floor_is_rendered_and_reported_rather_than_judged(bent_pair):
    """The box floor is the CALLER's policy and this function holds no opinion about it, which is why
    the floor comes back on the result. A picker that chooses its own box can refuse a clipped
    template outright; `cryosparc_chain.py refine` is handed the box the particles were already extracted at, where
    refusing would abort a refinement over a grid the invocation cannot change, and warns."""
    clipped = consensus_template(bent_pair, box_size=16, pixel_size=2.0, resolution=20.0)
    assert clipped.minimum_box_size > 16
    assert clipped.middle.shape == (16, 16, 16)
