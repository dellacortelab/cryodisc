"""`forward.expansion`: seven of every eight templates, built from the eighth.

The identities are pinned in `tests/test_poses.py` against a real render, one at a time, in REAL
space. What is checked here is the FOURIER form of them, which is where the rungs actually score --
a separate claim, because a real-space roll is a linear phase and the Fourier forms use no phase at all.
"""

import numpy as np
import pytest
from conftest import ALPHA_CARBON, conformer_pair
from jax import numpy as jnp

from cryodisc.forward.expansion import expand_templates
from cryodisc.grids.poses import expansion_plan, hopf_grid
from cryodisc.specimen.structures import align_conformers


@pytest.fixture(scope="module")
def model():
    """A real pair, coarse enough to render a whole bank in the suite's budget."""
    pair = align_conformers("data/pdb/8WJ1.pdb", "data/pdb/8WJ2.pdb", assembly="1")
    return pair.models_on_grid(box_size=48, pixel_size=4.0)[0]


@pytest.fixture(scope="module")
def asymmetric_model(tmp_path_factory):
    """A structure with no symmetry at all, which is the only kind that can see a pole error.

    8WJ1 carries a two-fold axis along z, so its projection down the beam at a pole is invariant
    under a half turn in the image plane -- the transform `expansion_plan` applies at a pole -- and so
    cannot see a pole error. A random blob has no such axis, so it reads the pole nodes honestly.
    """
    coords = np.random.default_rng(7).uniform(-30.0, 30.0, (400, 3))
    bent = coords + np.where(coords[:, :1] > 0.0, np.array([[4.0, 0.0, 0.0]]), 0.0)
    pair = conformer_pair(tmp_path_factory.mktemp("asymmetric"), coords, bent, atoms=ALPHA_CARBON)
    return pair.models_on_grid(box_size=48, pixel_size=4.0)[0]


@pytest.mark.parametrize("structure", ["model", "asymmetric_model"])
@pytest.mark.parametrize("triangulation", [1, 3, 4])
def test_the_expansion_reproduces_a_direct_render(request, structure, triangulation):
    """Every node of the bank, built from the eighth, against rendering that node outright.

    The acceptance test for the whole 8x. Both classes are covered -- `T` = 3 is Class II, whose
    sphere is a different lattice, so its mirror pairs vertices the Class I construction never
    produces. `T` = 4 is the smallest with vertices AT the poles, where phi is free and the plan's
    pole handling is exercised. BOTH structures are required: an asymmetric one is needed to certify
    an identity that must hold for every structure. Held at 1e-5, the render's own float32 noise.
    """
    model = request.getfixturevalue(structure)
    grid = hopf_grid(triangulation)
    angles = np.asarray(grid.angles)
    rendered, source, turns, mirror = expansion_plan(grid)
    assert len(rendered) * 8 == len(angles)

    direct = np.asarray(model.bare_contrast(jnp.asarray(angles)))
    built = np.asarray(expand_templates(model.bare_contrast(jnp.asarray(angles[rendered])), source, turns, mirror))
    assert np.abs(built - direct).max() / np.abs(direct).max() < 1e-5


def test_the_expansion_is_a_permutation_and_not_a_resampling(model):
    """Every value in the expanded bank appears in the rendered eighth, to the bit.

    The property that makes the 8x free: an interpolation would produce values between the ones it was
    given, and this must produce exactly the ones it was given, rearranged. That is what separates this
    route from rotating by an arbitrary angle.
    """
    grid = hopf_grid(1)
    angles = np.asarray(grid.angles)
    rendered, source, turns, mirror = expansion_plan(grid)
    base = model.bare_contrast(jnp.asarray(angles[rendered]))
    built = np.asarray(expand_templates(base, source, turns, mirror))

    for node in (0, 5, 37, 61, 95):
        # the multiset of values in an expanded template is EXACTLY that of the template it came
        # from -- same numbers, different places. `atol=0, rtol=0` because "exactly" is the claim
        built_values = np.sort_complex(built[node].ravel())
        source_values = np.sort_complex(np.asarray(base)[source[node]].ravel())
        assert np.allclose(built_values, source_values, rtol=0, atol=0), f"node {node} is not a permutation"


def test_a_plan_that_disagrees_with_itself_is_refused(model):
    """Three arrays of different lengths cannot describe one bank, and indexing past the rendered set
    is the shape a plan built for a different grid has."""
    grid = hopf_grid(1)
    rendered, source, turns, mirror = expansion_plan(grid)
    base = model.bare_contrast(jnp.asarray(np.asarray(grid.angles)[rendered]))
    with pytest.raises(ValueError, match="disagrees with itself"):
        expand_templates(base, source[:-1], turns, mirror)
    with pytest.raises(ValueError, match="outside"):
        expand_templates(base, source + len(rendered), turns, mirror)


def test_a_turn_that_matches_no_quarter_turn_is_refused_rather_than_left_unwritten(model):
    """The four turns must PARTITION the nodes, because what they miss is never written.

    `expand_templates` applies each quarter turn to the nodes that ask for it and writes the results
    into an `empty_like` buffer. A turn matching none of the four -- 0.5 is inside the legal range and
    matches nothing -- would leave that node at whatever was allocated and return an all-zero template
    with no error, so a non-integer or out-of-range turn is refused explicitly.
    """
    grid = hopf_grid(1)
    angles = np.asarray(grid.angles)
    rendered, source, turns, mirror = expansion_plan(grid)
    base = model.bare_contrast(jnp.asarray(angles[rendered]))

    with pytest.raises(ValueError, match="must be integers"):
        expand_templates(base, source, turns.astype(float) + 0.5, mirror)
    with pytest.raises(ValueError, match="0..3"):
        expand_templates(base, source, turns + 4, mirror)
