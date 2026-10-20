"""Checks on the local rung.

The Schur complement should delete exactly the part of a difference map that a small pose change
could fake. Volumes built from isotropic Gaussians make that testable to high precision: rotating or
translating the centers transforms the map exactly, so a state B built that way differs from A by a
pure pose change and the rung must return almost nothing.
"""

import cryojax.simulator as cxs
import numpy as np
import pytest
from conftest import gaussian_blob_models
from cryojax.rotations import SO3
from jax import numpy as jnp
from scipy.spatial.transform import Rotation

from cryodisc.grids.fourier import frequency_vectors, whitening_weight
from cryodisc.grids.poses import rotation_of
from cryodisc.rungs.local import _POSE_CHUNK, local_information

BOX_SIZE = 48
ANGLES = jnp.asarray(np.array([[20.0, 80.0, -10.0], [137.0, 62.0, 88.0], [45.0, 110.0, 200.0]], np.float32))
AXIS = np.array([0.3, 0.5, 0.8]) / np.linalg.norm([0.3, 0.5, 0.8])


@pytest.fixture(scope="module")
def blobs():
    """Centers and a builder, from `conftest.gaussian_blobs`: the maps are sums of ISOTROPIC
    Gaussians, so a map built from rotated or translated centers is an exact transform of the
    original -- which is what lets a pure pose change be planted and the residual read as error."""
    return gaussian_blob_models(BOX_SIZE, 8, seed=1, spread=7.0, sigma_range=(2.5, 3.5))


@pytest.fixture(scope="module")
def white_spectrum():
    return jnp.asarray(np.ones((BOX_SIZE, BOX_SIZE), np.float32))


def retained(model_a, model_b, spectrum, **kwargs):
    """Fraction of known-pose information surviving pose orthogonalization, checking the ordering."""
    rung = local_information(model_a, model_b, ANGLES, spectrum, **kwargs)
    assert np.all(np.asarray(rung.crlb) <= np.asarray(rung.known_pose) * (1 + 1e-4))
    return float(jnp.mean(rung.crlb / rung.known_pose)), rung


@pytest.mark.parametrize("degrees", [0.25, 1.0, 5.0])
def test_a_pure_rotation_is_almost_entirely_removed(blobs, white_spectrum, degrees):
    """The rung's reason to exist: a rotated copy of A carries no conformational information.

    The residual is the second-order part of a finite rotation, which a linearization cannot see.
    """
    centers, build = blobs
    rotation = Rotation.from_rotvec(np.deg2rad(degrees) * AXIS).as_matrix()
    fraction, rung = retained(build(centers), build(centers @ rotation.T), white_spectrum)

    assert float(jnp.mean(rung.known_pose)) > 100.0
    assert fraction < 0.01


@pytest.mark.parametrize("pixels", [0.25, 1.0])
def test_a_pure_translation_is_removed_more_exactly_still(blobs, white_spectrum, pixels):
    """Translation comes out two orders cleaner than rotation, since its sensitivities are exact
    phase ramps rather than differences. Replacing them with differences shows up here."""
    centers, build = blobs
    fraction, rung = retained(build(centers), build(centers + np.array([pixels, 0.0, 0.0])), white_spectrum)

    assert float(jnp.mean(rung.known_pose)) > 100.0
    assert fraction < 1e-3


@pytest.mark.parametrize("pixels", [0.25, 1.0])
def test_a_genuine_conformational_change_survives(blobs, white_spectrum, pixels):
    """The negative control: an orthogonalization that removed everything would pass the two above."""
    centers, build = blobs
    moved = centers.copy()
    moved[0] += np.array([pixels, 0.0, 0.0])
    moved[3] -= np.array([0.0, pixels, 0.0])

    fraction, _ = retained(build(centers), build(moved), white_spectrum)
    assert fraction > 0.5


def test_identical_states_carry_no_information(blobs, white_spectrum):
    """A state against itself gives exactly zero, not a small positive number."""
    centers, build = blobs
    model = build(centers)
    rung = local_information(model, model, ANGLES, white_spectrum)

    assert float(jnp.max(rung.known_pose)) == 0.0
    assert float(jnp.max(rung.crlb)) == 0.0


def test_the_dc_mode_is_ignored(blobs, white_spectrum):
    """sigma^2 at DC has no effect at all, because particles are mean-subtracted.

    Without dropping it, the voxel path's imperfect mass conservation would enter d^2 as signal, and
    a spectrum whose DC bin is near zero would divide by it.
    """
    centers, build = blobs
    moved = centers.copy()
    moved[0] += np.array([0.7, 0.0, 0.0])
    model_a, model_b = build(centers), build(moved)

    base = local_information(model_a, model_b, ANGLES, white_spectrum)
    for dc_value in (1e-8, 1e8):
        spectrum = np.array(white_spectrum)
        spectrum[0, 0] = dc_value
        altered = local_information(model_a, model_b, ANGLES, jnp.asarray(spectrum))
        assert np.array_equal(np.asarray(altered.known_pose), np.asarray(base.known_pose))
        assert np.array_equal(np.asarray(altered.crlb), np.asarray(base.crlb))


def test_the_default_step_is_converged(blobs, white_spectrum):
    """The default agrees with a ten-times finer step, so the answer is not a function of the knob.

    Two percent, because the step is a shallow optimum and not a plateau: truncation grows above it
    and float32 noise on the differenced residual grows below it.
    """
    centers, build = blobs
    moved = centers.copy()
    moved[0] += np.array([0.7, 0.0, 0.0])
    model_a, model_b = build(centers), build(moved)

    default = local_information(model_a, model_b, ANGLES, white_spectrum)
    finer = local_information(model_a, model_b, ANGLES, white_spectrum, step=1e-4)
    assert np.allclose(np.asarray(default.crlb), np.asarray(finer.crlb), rtol=0.02)


def test_scaling_in_amplitude_and_noise(blobs, white_spectrum):
    """d^2 goes as contrast squared and inversely with noise level, pinning both scale conventions."""
    centers, build = blobs
    moved = centers.copy()
    moved[0] += np.array([0.7, 0.0, 0.0])
    model_a, model_b = build(centers), build(moved)

    base = local_information(model_a, model_b, ANGLES, white_spectrum)
    louder = local_information(model_a, model_b, ANGLES, white_spectrum, contrast=3.0)
    noisier = local_information(model_a, model_b, ANGLES, 4.0 * white_spectrum)

    assert np.allclose(np.asarray(louder.known_pose / base.known_pose), 9.0, rtol=1e-4)
    assert np.allclose(np.asarray(louder.crlb / base.crlb), 9.0, rtol=1e-4)
    assert np.allclose(np.asarray(noisier.known_pose / base.known_pose), 0.25, rtol=1e-4)


def _whitened_pose_basis(model_a, model_b, pose_angles, spectrum, *, step=1e-3):
    """(A10)'s `G_phi` and the transferred difference, whitened, rebuilt OUTSIDE the module.

    An independent re-derivation, so the checks below are on the appendix's formula rather than on
    the module agreeing with itself. Same five directions and same whitening as `rungs.local`, but the
    arrays come back raw, for the explicit Gram inverse the module replaces with a QR.

    Args:
        model_a: Forward model of state A.
        model_b: Forward model of state B.
        pose_angles: One Euler triple in degrees.
        spectrum: sigma^2(k) on the unshifted fft grid.
        step: Rotation half-step in radians, matching the module's default.

    Returns:
        `(pose_images, difference)`, real arrays of shape `(2 * box^2, 5)` and `(2 * box^2,)`. The
        Re/Im stacking is what makes a Euclidean norm on them the `Sigma^-1` inner product.
    """
    weight = np.sqrt(np.asarray(whitening_weight(spectrum)))
    frequency_y, frequency_x = (np.asarray(k) for k in frequency_vectors(spectrum.shape[-1]))
    base = rotation_of(pose_angles)

    def render(model, rotation):
        pose = cxs.QuaternionPose(wxyz=rotation.wxyz)
        return np.asarray(jnp.fft.fft2(model.render_at_pose(pose, None)))

    def whiten(image):
        weighted = image * weight
        return np.concatenate([weighted.real.ravel(), weighted.imag.ravel()])

    def operating(rotation):
        return 0.5 * (render(model_a, rotation) + render(model_b, rotation))

    # both combinations at the reference pose come from ONE pair of renders; asking `operating` for
    # it and then re-rendering for the difference cost two of the sixteen renders per call
    image_a, image_b = render(model_a, base), render(model_b, base)
    reference = 0.5 * (image_a + image_b)

    columns = [(-2j * np.pi * frequency_x) * reference, (-2j * np.pi * frequency_y) * reference]
    for axis in np.eye(3):
        plus = operating(SO3.exp(step * jnp.asarray(axis)).compose(base))
        minus = operating(SO3.exp(-step * jnp.asarray(axis)).compose(base))
        columns.append((plus - minus) / (2.0 * step))

    return np.stack([whiten(column) for column in columns], axis=1), whiten(image_a - image_b)


@pytest.fixture(scope="module")
def appendix_basis(blobs, white_spectrum):
    """The pair both (A10) checks run on, with its whitened pose basis at every reference pose.

    Module-scoped because the two tests below want the same pair at the same poses, and building it
    twice is two model builds and a set of renders for nothing.

    Returns:
        `(model_a, model_b, [(pose_images, difference) per angle])`.
    """
    centers, build = blobs
    moved = centers.copy()
    moved[0] += np.array([0.7, 0.0, 0.0])
    model_a, model_b = build(centers), build(moved)
    bases = [_whitened_pose_basis(model_a, model_b, angles, white_spectrum) for angles in ANGLES]
    return model_a, model_b, bases


def test_the_whitened_stack_is_the_noise_inner_product(blobs):
    """The whitening is an ISOMETRY of `Sigma^-1`, which is what makes (A10) a projector at all.

    (A10) asks for the `Sigma^-1`-orthogonal residual, and `rungs.local` gets it by whitening first
    and then taking an ordinary least-squares residual. That substitution is only valid if the
    whitened real stack reproduces the complex noise-weighted inner product exactly:

        <whiten(u), whiten(v)> == Re sum_k conj(u(k)) v(k) / sigma^2(k)

    which is checked here against a direct complex evaluation, on a NON-WHITE spectrum so a wrong
    power of `sigma^2` cannot hide behind `sigma^2 == 1`. This is the claim that carries the
    conventions -- the contrast square root, the Re/Im stacking, and DC being dropped.
    """
    rng = np.random.default_rng(0)
    # a steep, strictly positive spectrum, so every mode is weighted differently
    spectrum = jnp.asarray((0.05 + rng.random((BOX_SIZE, BOX_SIZE)) ** 2 * 20.0).astype(np.float32))
    inverse = np.asarray(whitening_weight(spectrum))
    weight = np.sqrt(inverse)

    def whiten(image):
        weighted = image * weight
        return np.concatenate([weighted.real.ravel(), weighted.imag.ravel()])

    def complex_image():
        return rng.standard_normal((BOX_SIZE, BOX_SIZE)) + 1j * rng.standard_normal((BOX_SIZE, BOX_SIZE))

    u, v = complex_image(), complex_image()

    # `inverse` already zeroes DC, so the direct sum drops the same mode the stack does. rel=1e-6
    # rather than exact because the float32 spectrum makes `sqrt(inverse) ** 2` not bit-identical to it
    assert whiten(u) @ whiten(v) == pytest.approx(float(np.sum(np.conj(u) * v * inverse).real), rel=1e-6)
    assert whiten(u) @ whiten(u) == pytest.approx(float(np.sum(np.abs(u) ** 2 * inverse)), rel=1e-6)

    # and the test HAS power: whitening by the weight instead of its square root breaks the identity,
    # which is the specific slip that would silently rescale every rung on this ladder
    wrong = np.concatenate([(u * inverse).real.ravel(), (u * inverse).imag.ravel()])
    assert not np.isclose(wrong @ wrong, float(np.sum(np.abs(u) ** 2 * inverse)), rtol=0.1)


def test_the_pose_directions_stay_independent_over_the_grid(appendix_basis):
    """(A10) removes exactly FIVE directions, so its Gram must have rank five at every pose.

    A pose where two of the five collapse would make the Schur complement remove fewer directions
    than the appendix says it does, and the rung would read high with nothing to show it. Rank, not
    conditioning: the numbers say the Gram is comfortably conditioned (see `rungs.local`), and this
    is the assertion that would fail if that stopped being true.
    """
    for index, (pose_images, _) in enumerate(appendix_basis[2]):
        rank = np.linalg.matrix_rank(pose_images.T @ pose_images)
        assert rank == 5, f"pose {index} spans only {rank} of the five pose directions"


def test_the_rung_is_the_explicit_schur_complement_of_the_appendix(appendix_basis, white_spectrum):
    """`LocalRung.crlb` equals (A9)-(A11) built with the explicit 5x5 inverse of (A10).

    The module takes a QR of the same columns instead, which is the same projector by a better
    conditioned route; this is the check that the shortcut did not change the quantity. It also pins
    the identity the naming rests on -- `crlb` IS `d^2_CRLB` -- so a later rename cannot quietly
    leave the appendix describing something the code no longer computes.
    """
    model_a, model_b, bases = appendix_basis
    rung = local_information(model_a, model_b, ANGLES, white_spectrum)

    for index, (pose_images, difference) in enumerate(bases):
        residual = difference - pose_images @ np.linalg.solve(pose_images.T @ pose_images, pose_images.T @ difference)

        assert float(difference @ difference) == pytest.approx(float(rung.known_pose[index]), rel=1e-4)
        assert float(residual @ residual) == pytest.approx(float(rung.crlb[index]), rel=1e-3)


def test_the_independent_basis_annihilates_a_pure_pose_change(blobs, white_spectrum):
    """`_whitened_pose_basis` is right about ground truth, not merely consistent with `rungs.local`.

    Without this the equivalence test above could pass VACUOUSLY: a helper carrying the same
    convention error as the module it checks would agree with it perfectly. The ground truth here
    needs neither implementation -- a state B that is state A moved by a pure pose change has its
    whole difference inside the span of the pose columns, so the projected residual must vanish --
    and the negative control is what stops a helper that annihilates everything from passing.

    Only the SPAN of the columns can matter, which is why a per-column sign or scale slip is not what
    this is guarding: flipping the phase ramp's sign, rescaling a rotation column by `1/(2 h)`, or
    swapping `k_x` for `k_y` all leave the column space -- and so the projector -- unchanged. What is
    actually at risk is the Re/Im stacking and whether the rotation columns are derivatives at all.
    Asserted loosely because the point is the ORDERS of magnitude between the two regimes.
    """
    centers, build = blobs

    def residual_fraction(moved_centers):
        pose_images, difference = _whitened_pose_basis(build(centers), build(moved_centers), ANGLES[0], white_spectrum)
        gram = pose_images.T @ pose_images
        residual = difference - pose_images @ np.linalg.solve(gram, pose_images.T @ difference)
        return float(residual @ residual) / float(difference @ difference)

    assert residual_fraction(centers + np.array([1.0, 0.0, 0.0])) < 1e-3

    # the negative control: two blobs moved in OPPOSITE directions is no rigid motion, and most of it
    # must survive -- a basis that spanned the whole image would pass the assertion above
    moved = centers.copy()
    moved[0] += np.array([0.7, 0.0, 0.0])
    moved[3] -= np.array([0.0, 0.7, 0.0])
    assert residual_fraction(moved) > 0.4


@pytest.mark.parametrize("chunk", [3, _POSE_CHUNK, 32])
def test_chunking_the_pose_axis_does_not_move_the_rung(blobs, white_spectrum, chunk):
    """Chunking exists to bound memory, so it must be invisible in the answer.

    The bank length is deliberately not a multiple of any chunk tried, so `lax.map`'s remainder
    path is exercised rather than only its scan. `known_pose` carries no cross-batch reduction, so
    chunking moves it by no more than the compiler's last bit; `crlb` additionally absorbs the QR's.
    """
    centers, build = blobs
    rotation = Rotation.from_rotvec(np.deg2rad(2.0) * AXIS).as_matrix()
    model_a, model_b = build(centers), build(centers @ rotation.T + 0.4)
    angles = jnp.asarray(np.random.default_rng(0).uniform(-180.0, 180.0, (11, 3)).astype(np.float32))

    reference = local_information(model_a, model_b, angles, white_spectrum, chunk=angles.shape[0])
    chunked = local_information(model_a, model_b, angles, white_spectrum, chunk=chunk)

    known = np.asarray(reference.known_pose)
    assert np.max(np.abs(np.asarray(chunked.known_pose) - known)) < 1e-6 * known.mean()
    assert np.max(np.abs(np.asarray(chunked.crlb) - np.asarray(reference.crlb))) < 1e-5 * known.mean()
