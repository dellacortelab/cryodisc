"""What the pose search has to get right before a recovered pose can be believed.

The sign of the recovered in-plane angle and the payoff for freeing it are both pinned on noiseless
particles, where the answer is exact. Alignability is pinned on noisy ones, being a property of the
exposure rather than of the search.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cryodisc.grids.fourier import whitening_weight
from cryodisc.grids.poses import geodesic_error, pose_bank
from cryodisc.microscope import CollectionPlan, Transfer
from cryodisc.particles.sim import simulate_stack, uniform_angles
from cryodisc.resolution import noise_spectrum
from cryodisc.rungs.refine import refine_poses, view_angles

BOX_SIZE = 48
PLAN = CollectionPlan(dose=40.0, pixel_size=1.5, defocus_range=(8000.0, 20000.0))
# counts are quantized to 10 * frequency^2 + 2; 42 keeps the search grid SMALLER than the 96-node
# bank the in-plane comparison below prices it against
VIEWS = view_angles(42)


@pytest.fixture(scope="module")
def model(bent_pair_factory):
    """A particle large enough to align. Below the limit the search returns noise -- see the ~10 kDa test."""
    return bent_pair_factory(n_atoms=6000, spread=11.0, bend=2.5).models(PLAN, box_size=BOX_SIZE)[0]


@pytest.fixture(scope="module")
def spectrum():
    return noise_spectrum(PLAN, BOX_SIZE, rmsf=0.5)


@pytest.fixture(scope="module")
def bank(model):
    return model.bare_contrast(VIEWS)


def test_the_in_plane_angle_comes_back_with_the_sign_it_went_in_with(model, spectrum, bank):
    """A flipped sign still returns a plausible pose, so it is pinned here.

    Queries sit on bank viewing directions, leaving the in-plane quantum as the only error.
    """
    quantum = 360.0 / 128
    nodes = jnp.asarray([3, 37, 60])
    truth = jnp.stack([VIEWS[nodes, 0], VIEWS[nodes, 1], jnp.asarray([0.0, quantum, 200.0])], axis=-1)
    fit = refine_poses(model.bare_contrast(truth), bank, VIEWS, spectrum)
    np.testing.assert_allclose(fit.angles[:, :2], truth[:, :2], atol=1e-4)
    assert float(jnp.max(geodesic_error(fit.angles, truth))) < quantum


def test_a_free_in_plane_angle_beats_spending_templates_on_it(model, spectrum, bank):
    """A smaller template budget, and still 2.4x the pose accuracy.

    The bank is held to its nearest node by geodesic, which no search over it can beat, so the
    comparison does not depend on how a bank is searched.
    """
    truth = uniform_angles(jax.random.PRNGKey(1), 120)
    grid = pose_bank(96)
    assert grid.angles.shape[0] >= VIEWS.shape[0]  # the bank gets MORE templates, not equal

    nearest = jnp.min(
        jax.vmap(lambda node: geodesic_error(truth, jnp.broadcast_to(node, truth.shape)))(grid.angles), axis=0
    )

    searched = geodesic_error(refine_poses(model.bare_contrast(truth), bank, VIEWS, spectrum).angles, truth)
    assert float(jnp.median(searched)) < 0.5 * float(jnp.median(nearest))


def test_a_declared_ctf_is_the_difference_between_a_pose_and_a_coin_flip(model, spectrum, bank):
    """The CTF zeros invert the template, so omitting `transfer` loses the search rather than accuracy."""
    truth = uniform_angles(jax.random.PRNGKey(2), 60)
    defocus = jnp.linspace(*PLAN.defocus_range, truth.shape[0])
    transfer = Transfer.from_defocus(defocus, PLAN, BOX_SIZE, 6)
    images = model.bare_contrast(truth) * transfer.values[transfer.index]

    declared = geodesic_error(refine_poses(images, bank, VIEWS, spectrum, transfer=transfer).angles, truth)
    ignored = geodesic_error(refine_poses(images, bank, VIEWS, spectrum).angles, truth)
    assert float(jnp.median(declared)) < 15.0
    assert float(jnp.median(ignored)) > 60.0


def test_a_particle_too_small_to_align_is_not_aligned(bent_pair_factory, spectrum):
    """Alignability is `||C T||^2_W`, which makes it a statement about molecular weight.

    A ~10 kDa particle at this exposure cannot be posed. Confident poses here would hand `chain` an
    efficiency built on nothing.
    """
    small = bent_pair_factory(n_atoms=700, spread=7.0, bend=2.5).models(PLAN, box_size=BOX_SIZE)[0]
    truth = uniform_angles(jax.random.PRNGKey(4), 60)
    stack = simulate_stack(
        models=(small,), key=jax.random.PRNGKey(5), n_particles=60, noise_spectrum=spectrum, plan=PLAN, angles=truth
    )
    transfer = Transfer.from_defocus(stack.defocus, PLAN, BOX_SIZE, 6)
    fit = refine_poses(stack.images, small.bare_contrast(VIEWS), VIEWS, spectrum, transfer=transfer)
    assert float(jnp.median(geodesic_error(fit.angles, truth))) > 45.0


def test_the_score_is_the_whitened_matched_filter_it_claims_to_be(model, spectrum, bank):
    """`C^2 / 2||T||^2` at the true pose, against the same sum taken exactly on the fft grid.

    Bracketed, not approximated: the polar disc misses the box corners, so `C` runs a few percent
    low and the score, quadratic in it, about twice that. The bracket catches a stray factor of
    `n_angles` or `n_pixels`, which is how this normalization goes wrong.
    """
    truth = VIEWS[jnp.asarray([21])]
    fit = refine_poses(model.bare_contrast(truth), bank, VIEWS, spectrum)

    exact = 0.5 * float(jnp.sum(jnp.abs(bank[21]) ** 2 * whitening_weight(spectrum)))
    assert 0.7 * exact < float(fit.score[0]) <= exact


def test_geodesic_error_reads_the_rotation_and_not_the_angles():
    """At theta = 0, phi and psi are the same rotation, and differencing the triples reports 90
    degrees of error where there is none."""
    gimbal = jnp.asarray([[40.0, 0.0, 50.0]])
    same = jnp.asarray([[0.0, 0.0, 90.0]])
    assert float(geodesic_error(gimbal, same)[0]) == pytest.approx(0.0, abs=1e-3)
    assert float(geodesic_error(gimbal, gimbal)[0]) == pytest.approx(0.0, abs=1e-4)


def test_geodesic_error_resolves_a_pose_recovered_exactly():
    """`arccos((tr-1)/2)` has infinite derivative at the identity and in float32 reads exactly
    recovered poses as wrong. The chordal form is conditioned there, the only regime `pose_error`
    occupies.
    """
    poses = uniform_angles(jax.random.PRNGKey(0), 4000)
    assert float(jnp.max(geodesic_error(poses, poses))) < 1e-4
    small = poses.at[:, 2].add(0.01)
    assert float(jnp.median(geodesic_error(small, poses))) == pytest.approx(0.01, abs=2e-3)


def test_a_bank_with_a_spun_template_is_refused(model, spectrum, bank):
    """The in-plane angle is measured from the template, so a bank at psi != 0 offsets every pose."""
    spun = VIEWS.at[0, 2].set(10.0)
    with pytest.raises(ValueError, match="psi = 0"):
        refine_poses(model.bare_contrast(VIEWS[:1]), bank, spun, spectrum)


def test_view_angles_is_a_viewing_grid_and_nothing_more():
    views = view_angles(42)
    assert views.shape == (42, 3)
    np.testing.assert_array_equal(np.asarray(views[:, 2]), 0.0)
    with pytest.raises(ValueError, match="must be positive"):
        view_angles(0)


@pytest.fixture(scope="module")
def off_center(model, spectrum, bank):
    """A stack with real centering error, and the one expensive search over it, run once."""
    truth = uniform_angles(jax.random.PRNGKey(7), 80)
    stack = simulate_stack(
        models=(model,),
        key=jax.random.PRNGKey(8),
        n_particles=80,
        noise_spectrum=spectrum,
        plan=PLAN,
        angles=truth,
        shift_sigma=3.0,
    )
    transfer = Transfer.from_defocus(stack.defocus, PLAN, BOX_SIZE, 6)
    fit = refine_poses(stack.images, bank, VIEWS, spectrum, transfer=transfer, shift_iterations=3)
    return truth, stack, transfer, fit


def test_an_off_center_particle_needs_the_translation_search(off_center, spectrum, bank):
    """Centering error does not degrade the pose search, it destroys it.

    Three pixels of offset on a 48-pixel box takes the pose error from usable to worse than chance,
    because a wrong viewing direction fits the displaced particle better than the right one does.
    """
    truth, stack, transfer, fit = off_center
    centered = geodesic_error(refine_poses(stack.images, bank, VIEWS, spectrum, transfer=transfer).angles, truth)
    assert float(jnp.median(centered)) > 90.0
    assert float(jnp.median(geodesic_error(fit.angles, truth))) < 25.0


def test_the_recovered_shift_undoes_the_one_that_was_applied(off_center, spectrum, bank):
    """`-shift` is the offset the particle was rendered with, to within the peak's one-pixel grid."""
    _, stack, _, fit = off_center
    residual = jnp.linalg.norm(-fit.shift - stack.shift, axis=1)
    assert float(jnp.median(jnp.linalg.norm(stack.shift, axis=1))) > 2.0
    assert float(jnp.median(residual)) < 1.0


def test_searching_shifts_on_centered_particles_finds_none(model, spectrum, bank):
    """The search has to be safe to leave on: it must not invent an offset that is not there."""
    truth = uniform_angles(jax.random.PRNGKey(9), 40)
    stack = simulate_stack(
        models=(model,), key=jax.random.PRNGKey(10), n_particles=40, noise_spectrum=spectrum, plan=PLAN, angles=truth
    )
    transfer = Transfer.from_defocus(stack.defocus, PLAN, BOX_SIZE, 6)
    still = refine_poses(stack.images, bank, VIEWS, spectrum, transfer=transfer)
    moved = refine_poses(stack.images, bank, VIEWS, spectrum, transfer=transfer, shift_iterations=3)
    np.testing.assert_array_equal(np.asarray(still.shift), 0.0)
    assert float(jnp.max(jnp.abs(moved.shift))) <= 1.0
    assert float(jnp.median(geodesic_error(moved.angles, truth))) < 1.5 * float(
        jnp.median(geodesic_error(still.angles, truth))
    )


@pytest.mark.parametrize("box_size", [46, 47])
def test_an_odd_box_is_searched_about_the_same_center_as_an_even_one(bent_pair_factory, box_size):
    """cryoJAX rotates about `box // 2` at both parities; `box / 2` is half a pixel out on an odd box.

    That breaks the polar-shift identity the module rests on, and the search then returns worse than
    chance with a plausible score and no error.
    """
    model = bent_pair_factory(n_atoms=6000, spread=11.0, bend=2.5).models(PLAN, box_size=box_size)[0]
    spectrum = noise_spectrum(PLAN, box_size, rmsf=0.5)
    truth = uniform_angles(jax.random.PRNGKey(1), 40)
    fit = refine_poses(model.bare_contrast(truth), model.bare_contrast(VIEWS), VIEWS, spectrum)
    assert float(jnp.median(geodesic_error(fit.angles, truth))) < 20.0


def test_a_polar_grid_too_small_to_have_a_spacing_is_refused(model, spectrum, bank):
    """`radii[1]` at `n_radii=1` is an out-of-bounds gather, and JAX CLAMPS it: the quadrature weight
    comes out zero, the bank is zeroed, and every particle is returned at view 0 with score 0."""
    for kwargs in ({"n_radii": 1}, {"n_angles": 1}):
        with pytest.raises(ValueError, match="n_radii >= 2 and n_angles >= 2"):
            refine_poses(model.bare_contrast(VIEWS[:4]), bank, VIEWS, spectrum, **kwargs)


def test_a_spectrum_with_a_dead_mode_is_refused_rather_than_silently_collapsing(model, spectrum, bank):
    """`nan > 0` is False, so the profiled score masks a NaN correlation to zero and the search comes
    back at view 0 for every particle, looking confident. Every other rung propagates the NaN."""
    images = model.bare_contrast(VIEWS[:4])
    for broken in (spectrum.at[5, 5].set(jnp.nan), spectrum.at[5, 5].set(0.0), spectrum.at[5, 5].set(-1.0)):
        with pytest.raises(ValueError, match="finite and strictly positive"):
            refine_poses(images, bank, VIEWS, broken)


def test_an_empty_bank_says_so(model, spectrum):
    """It used to fail inside the psi guard, reading as a convention error rather than an empty bank."""
    empty = model.bare_contrast(VIEWS[:0])
    with pytest.raises(ValueError, match="bank is empty"):
        refine_poses(model.bare_contrast(VIEWS[:2]), empty, VIEWS[:0], spectrum)
