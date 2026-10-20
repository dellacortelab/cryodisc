"""Checks on the streaming marginal rung.

It exists to lift a memory ceiling, so the only thing that matters is that it returns EXACTLY what
`pose_log_ratio` returns on the same bank. A streaming reduction that is subtly wrong -- a prior
renormalized per chunk, a ragged last chunk mishandled -- produces a perfectly plausible rung, so
every check here is against the whole-bank answer rather than against a tolerance on physics.
"""

import numpy as np
import pytest
from conftest import gaussian_blob_models
from jax import numpy as jnp
from scipy.spatial.transform import Rotation

from cryodisc.forward.expansion import expand_templates
from cryodisc.grids.poses import expansion_plan, hopf_grid, pose_bank, shift_grid
from cryodisc.microscope import Transfer
from cryodisc.rungs.marginal import known_pose_d2, pose_log_ratio, streamed_pose_log_ratio
from cryodisc.specimen.structures import align_conformers

BOX_SIZE = 24
NOISE_LEVEL = 0.6
TARGET_D2 = 10.0

RELATIVE = 1e-4
"""Agreement demanded, relative to the scale of the log-ratio itself.

The two routes sum the SAME terms in a different order, so in float32 they cannot agree to the last
bit: the scores entering the `logsumexp` run orders of magnitude larger than the log-ratio they
difference to, so a few ulp on the terms is a few times 1e-5 on the output, while single-chunk cases
agree to the bit. Structural failures are orders of magnitude above this bar.
"""


def _close(got, want):
    """Max absolute gap against `RELATIVE` times the scale of `want`."""
    gap = np.abs(np.asarray(got - want)).max()
    scale = float(np.abs(np.asarray(want)).max())
    return gap <= RELATIVE * max(scale, 1.0), gap, scale


@pytest.fixture(scope="module")
def setup():
    """Two states, a bank, and a noisy pool -- the smallest thing that exercises every axis."""
    centers, build = gaussian_blob_models(BOX_SIZE, 5, seed=0, spread=4.0, sigma_range=(1.5, 2.5))
    moved = centers.copy()
    moved[0] += np.array([1.3, 0.0, 0.0])
    model_a, model_b = build(centers), build(moved)
    spectrum = jnp.asarray(np.full((BOX_SIZE, BOX_SIZE), NOISE_LEVEL, np.float32))
    # scaled to an ordinary separation: at amplitude 1 these templates put the log-ratio near 1e3,
    # where float32 rounding alone is a whole unit and the comparison stops meaning anything
    unit = float(
        known_pose_d2(
            jnp.fft.fft2(model_a.forward_batch(jnp.zeros((1, 3)))),
            jnp.fft.fft2(model_b.forward_batch(jnp.zeros((1, 3)))),
            spectrum,
        )[0]
    )
    amplitude = float(np.sqrt(TARGET_D2 / unit))
    grid = pose_bank(45)  # deliberately not a round number
    angles = jnp.asarray(Rotation.random(11, random_state=2).as_euler("zyz", degrees=True).astype(np.float32))
    rng = np.random.default_rng(1)
    clean = jnp.fft.fft2(model_a.forward_batch(angles))
    noise = jnp.fft.fft2(jnp.asarray(rng.standard_normal((11, BOX_SIZE, BOX_SIZE)).astype(np.float32)))
    images = amplitude * clean + noise * jnp.sqrt(spectrum / (BOX_SIZE * BOX_SIZE))
    return model_a, model_b, grid, images, spectrum, amplitude


@pytest.mark.parametrize("chunk", [1, 7, 45, 1000])
def test_it_matches_the_whole_bank_at_every_chunk_size(setup, chunk):
    """Including chunk sizes that do not divide 45, and one larger than the grid."""
    model_a, model_b, grid, images, spectrum, amplitude = setup
    reference = pose_log_ratio(
        images,
        model_a.bare_contrast(grid.angles),
        model_b.bare_contrast(grid.angles),
        spectrum,
        amplitude=amplitude,
        pose_log_weights=grid.log_weights,
    )
    streamed = streamed_pose_log_ratio(
        images,
        model_a.bare_contrast,
        model_b.bare_contrast,
        grid.angles,
        grid.log_weights,
        spectrum,
        amplitude=amplitude,
        pose_chunk=chunk,
    )
    ok, gap, scale = _close(streamed.marginal, reference.marginal)
    assert ok, f"marginal differs by {gap:.3g} on a scale of {scale:.3g}"
    ok, gap, scale = _close(streamed.best_match, reference.best_match)
    assert ok, f"best_match differs by {gap:.3g} on a scale of {scale:.3g}"


def test_the_prior_is_normalized_once_and_not_per_chunk(setup):
    """The failure a per-chunk call to `pose_log_ratio` would introduce, and it is silent.

    A grid whose weights vary strongly across the pose axis is what exposes it: renormalizing inside
    each chunk would tilt the posterior toward whichever poses happened to share a chunk. The looser
    tolerance is float32 accumulation order, not the property under test: the gap does not grow with
    the number of chunks, and in float64 the same comparison is exact to many digits.
    """
    model_a, model_b, grid, images, spectrum, _amplitude = setup
    tilted = jnp.asarray(np.linspace(-4.0, 4.0, grid.angles.shape[0]).astype(np.float32))
    reference = pose_log_ratio(
        images,
        model_a.bare_contrast(grid.angles),
        model_b.bare_contrast(grid.angles),
        spectrum,
        pose_log_weights=tilted,
    )
    streamed = streamed_pose_log_ratio(
        images, model_a.bare_contrast, model_b.bare_contrast, grid.angles, tilted, spectrum, pose_chunk=4
    )
    gap = float(np.abs(np.asarray(streamed.marginal - reference.marginal)).max())
    scale = float(np.abs(np.asarray(reference.marginal)).max())
    assert gap <= 1e-3 * max(scale, 1.0), f"differs by {gap:.3g} on a scale of {scale:.3g}"


def test_it_matches_with_a_per_group_transfer(setup):
    """The transfer folds onto the particle, so it has to survive being applied outside the loop."""
    model_a, model_b, grid, images, spectrum, _amplitude = setup
    rng = np.random.default_rng(3)
    values = jnp.asarray(rng.uniform(0.4, 1.2, (2, BOX_SIZE, BOX_SIZE)).astype(np.float32))
    transfer = Transfer(values=values, index=jnp.asarray(np.arange(images.shape[0]) % 2))
    reference = pose_log_ratio(
        images,
        model_a.bare_contrast(grid.angles),
        model_b.bare_contrast(grid.angles),
        spectrum,
        pose_log_weights=grid.log_weights,
        transfer=transfer,
    )
    streamed = streamed_pose_log_ratio(
        images,
        model_a.bare_contrast,
        model_b.bare_contrast,
        grid.angles,
        grid.log_weights,
        spectrum,
        transfer=transfer,
        pose_chunk=8,
    )
    ok, gap, scale = _close(streamed.marginal, reference.marginal)
    assert ok, f"differs by {gap:.3g} on a scale of {scale:.3g}"


def test_mismatched_weights_are_refused(setup):
    _, _, grid, images, spectrum, _ = setup
    with pytest.raises(ValueError, match="one weight per pose"):
        streamed_pose_log_ratio(images, lambda a: None, lambda a: None, grid.angles, grid.log_weights[:-1], spectrum)


@pytest.mark.parametrize("chunk", [8, 45, 1000])
def test_the_eightfold_expansion_scores_the_same_bank_to_the_same_number(setup, chunk):
    """Rendering an eighth and permuting the rest must not move the answer at all.

    The whole case for the expansion is that it is free: the seven built templates are the rendered
    one rearranged, so the only thing that changes is how many times the renderer runs. If the two
    routes disagreed by more than accumulation order, the saving would be buying a different `d^2`.

    Chunk sizes below eight are included on purpose. The plan chunks over the RENDERED EIGHTH rather
    than the bank, so `_pose_chunks` divides the pose chunk by eight, and a chunk of 8 is the
    smallest that does not floor to zero slots.
    """
    model_a, model_b, grid, images, spectrum, amplitude = setup
    plan = expansion_plan(grid)
    assert len(plan[0]) * 8 == grid.angles.shape[0]

    common = {"amplitude": amplitude, "pose_chunk": chunk}
    whole = streamed_pose_log_ratio(
        images,
        model_a.bare_contrast,
        model_b.bare_contrast,
        grid.angles,
        grid.log_weights,
        spectrum,
        **common,
    )
    expanded = streamed_pose_log_ratio(
        images,
        model_a.bare_contrast,
        model_b.bare_contrast,
        grid.angles,
        grid.log_weights,
        spectrum,
        expansion=plan,
        **common,
    )
    # A LOOSER bar than `RELATIVE`, because the fixture's `from_voxel_grid` templates interpolate a
    # sampled volume and so are not exactly equivariant under a quarter turn; atomic models hold to
    # 1e-7. See `forward/expansion.py`.
    for name, got, want in (
        ("marginal", expanded.marginal, whole.marginal),
        ("best_match", expanded.best_match, whole.best_match),
    ):
        gap = np.abs(np.asarray(got - want)).max()
        assert gap <= 1e-3, f"{name} differs by {gap:.3g}, too much for the voxel renderer alone"


@pytest.mark.parametrize("structure", ["real", "asymmetric"])
def test_the_expansion_is_exact_on_atomic_templates(structure, asymmetric_pair):
    """The claim the voxel bound above is measured against, so that a loosened bar cannot hide a bug.

    `from_voxel_grid` interpolates and so cannot be exactly equivariant; a Gaussian mixture projected
    analytically can, and is. The grid must carry pole nodes and the structure a genuine asymmetry, or
    the expansion's pole handling goes unexercised -- so `hopf_grid(4)` and both structures.
    """
    grid = hopf_grid(4)
    if structure == "real":
        pair = align_conformers("data/pdb/8WJ1.pdb", "data/pdb/8WJ2.pdb", assembly="1")
    else:
        pair = asymmetric_pair
    model = pair.models_on_grid(box_size=BOX_SIZE, pixel_size=6.0)[0]
    rendered, source, turns, mirror = expansion_plan(grid)

    direct = np.asarray(model.bare_contrast(grid.angles))
    built = np.asarray(expand_templates(model.bare_contrast(grid.angles[np.asarray(rendered)]), source, turns, mirror))
    assert np.abs(built - direct).max() / np.abs(direct).max() < 1e-5


def test_an_odd_box_is_refused_rather_than_quietly_rotated_wrong(setup):
    """The quarter turn is an even-box identity, and on an odd box it is wrong by a fifth of the image.

    An even box puts the image center half a pixel off the sample lattice and an odd box puts it on a
    sample, so the two need different index permutations. Applying the even table to an odd box is not
    a rounding, and it would read downstream as a smaller signal, which nothing else catches.
    """
    with pytest.raises(ValueError, match="even-box identity"):
        expand_templates(jnp.zeros((1, 33, 33), jnp.complex64), np.array([0]), np.array([0]), np.array([False]))


def test_a_plan_that_does_not_describe_this_bank_is_refused(setup):
    """A plan silently scoring the wrong nodes would read as a smaller signal, which nothing catches.

    The dangerous shape is a plan built for a DIFFERENT bank: its indices are all in range and its
    arrays are all self-consistent, so the arithmetic completes and returns a number for poses the
    particles were never near.
    """
    model_a, model_b, grid, images, spectrum, amplitude = setup
    rendered, source, turns, mirror = expansion_plan(grid)

    with pytest.raises(ValueError, match="describes"):
        streamed_pose_log_ratio(
            images,
            model_a.bare_contrast,
            model_b.bare_contrast,
            grid.angles,
            grid.log_weights,
            spectrum,
            amplitude=amplitude,
            expansion=(rendered, source[:-1], turns[:-1], mirror[:-1]),
        )
    with pytest.raises(ValueError, match="outside the rendered set"):
        streamed_pose_log_ratio(
            images,
            model_a.bare_contrast,
            model_b.bare_contrast,
            grid.angles,
            grid.log_weights,
            spectrum,
            amplitude=amplitude,
            expansion=(rendered[:-1], source, turns, mirror),
        )


@pytest.mark.parametrize("chunk", [1, 7, 45, 1000])
def test_it_matches_the_whole_bank_with_shifts_at_every_chunk_size(setup, chunk):
    """The shift route streamed, which is the one the survey's ladder runs on.

    Without streaming this path has no way past a whole bank, which does not fit in memory at
    production box sizes. The reduction composes for the same reason it does without shifts, the joint
    (pose, shift) `logsumexp` being a sum over a product set; what is new is that BOTH states have to
    be reduced in one call, since they share the inverse transform.
    """
    model_a, model_b, grid, images, spectrum, amplitude = setup
    shifts = shift_grid(BOX_SIZE, 2.0, sigma=1.5)
    reference = pose_log_ratio(
        images,
        model_a.bare_contrast(grid.angles),
        model_b.bare_contrast(grid.angles),
        spectrum,
        amplitude=amplitude,
        pose_log_weights=grid.log_weights,
        shifts=shifts,
    )
    streamed = streamed_pose_log_ratio(
        images,
        model_a.bare_contrast,
        model_b.bare_contrast,
        grid.angles,
        grid.log_weights,
        spectrum,
        amplitude=amplitude,
        shifts=shifts,
        pose_chunk=chunk,
    )
    ok, gap, scale = _close(streamed.marginal, reference.marginal)
    assert ok, f"marginal differs by {gap:.3g} on a scale of {scale:.3g}"
    ok, gap, scale = _close(streamed.best_match, reference.best_match)
    assert ok, f"best_match differs by {gap:.3g} on a scale of {scale:.3g}"


def test_the_shift_prior_is_not_renormalized_per_chunk(setup):
    """A tilted POSE prior under shifts, which is where a per-chunk normalization would show."""
    model_a, model_b, grid, images, spectrum, _amplitude = setup
    shifts = shift_grid(BOX_SIZE, 2.0, sigma=1.5)
    tilted = jnp.asarray(np.linspace(-4.0, 4.0, grid.angles.shape[0]).astype(np.float32))
    reference = pose_log_ratio(
        images,
        model_a.bare_contrast(grid.angles),
        model_b.bare_contrast(grid.angles),
        spectrum,
        pose_log_weights=tilted,
        shifts=shifts,
    )
    streamed = streamed_pose_log_ratio(
        images, model_a.bare_contrast, model_b.bare_contrast, grid.angles, tilted, spectrum, shifts=shifts, pose_chunk=4
    )
    gap = float(np.abs(np.asarray(streamed.marginal - reference.marginal)).max())
    scale = float(np.abs(np.asarray(reference.marginal)).max())
    assert gap <= 1e-3 * max(scale, 1.0), f"differs by {gap:.3g} on a scale of {scale:.3g}"


def test_streamed_shifts_match_with_a_per_group_transfer(setup):
    """Both chunkings at once: poses in the loop, particles inside `_shift_reduce`, groups folded."""
    model_a, model_b, grid, images, spectrum, _amplitude = setup
    shifts = shift_grid(BOX_SIZE, 2.0, sigma=1.5)
    rng = np.random.default_rng(3)
    values = jnp.asarray(rng.uniform(0.4, 1.2, (2, BOX_SIZE, BOX_SIZE)).astype(np.float32))
    transfer = Transfer(values=values, index=jnp.asarray(np.arange(images.shape[0]) % 2))
    reference = pose_log_ratio(
        images,
        model_a.bare_contrast(grid.angles),
        model_b.bare_contrast(grid.angles),
        spectrum,
        pose_log_weights=grid.log_weights,
        shifts=shifts,
        transfer=transfer,
    )
    streamed = streamed_pose_log_ratio(
        images,
        model_a.bare_contrast,
        model_b.bare_contrast,
        grid.angles,
        grid.log_weights,
        spectrum,
        shifts=shifts,
        transfer=transfer,
        pose_chunk=8,
    )
    ok, gap, scale = _close(streamed.marginal, reference.marginal)
    assert ok, f"differs by {gap:.3g} on a scale of {scale:.3g}"


@pytest.mark.parametrize("chunk", [8, 45, 1000])
def test_the_expansion_under_shifts_carries_the_plan_s_own_error_and_no_more(setup, chunk):
    """The eightfold plan composes with the shift lattice, to the accuracy the PLAN has.

    The plan is not exact: on a voxel renderer the seven built templates reproduce the true renders
    only to about 1e-3, since the renderer is not exactly equivariant under the quarter turns and the
    mirror, and this holds identically in float32 and float64 -- so the bar below is the plan's error,
    not the streaming's. `pipeline.chain` does not pass a plan on the shift route for this reason.
    """
    model_a, model_b, grid, images, spectrum, amplitude = setup
    shifts = shift_grid(BOX_SIZE, 2.0, sigma=1.5)
    plan = expansion_plan(grid)
    common = {"amplitude": amplitude, "shifts": shifts, "pose_chunk": chunk}
    whole = streamed_pose_log_ratio(
        images, model_a.bare_contrast, model_b.bare_contrast, grid.angles, grid.log_weights, spectrum, **common
    )
    expanded = streamed_pose_log_ratio(
        images,
        model_a.bare_contrast,
        model_b.bare_contrast,
        grid.angles,
        grid.log_weights,
        spectrum,
        expansion=plan,
        **common,
    )
    gap = float(np.abs(np.asarray(expanded.marginal - whole.marginal)).max())
    assert gap <= 2e-3, f"the plan moves the shift-marginalized answer by {gap:.3g}, past its own template error"


def test_a_streamed_shift_grid_on_another_box_is_refused(setup):
    """The guard `pose_log_ratio` applies, applied on the route that skips it."""
    model_a, model_b, grid, images, spectrum, _amplitude = setup
    with pytest.raises(ValueError, match="built on box"):
        streamed_pose_log_ratio(
            images,
            model_a.bare_contrast,
            model_b.bare_contrast,
            grid.angles,
            grid.log_weights,
            spectrum,
            shifts=shift_grid(BOX_SIZE * 2, 2.0),
        )
