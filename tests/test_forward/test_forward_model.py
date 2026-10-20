"""Self-consistency checks on the cryoJAX forward model.

These need no second engine. Each one pins an assumption the model already rests on, chosen
because its failure mode is silent: wrong units, a flipped sign, or a library convention that a
cryoJAX upgrade could change without erroring.
"""

import numpy as np
import pytest
from conftest import ALPHA_CARBON, atom_records, gaussian_blobs, write_pdb
from cryojax.constants import wavelength_from_kilovolts
from cryojax.simulator import AstigmaticCTF, EulerAnglePose, FourierVoxelGridVolume
from jax import numpy as jnp
from scipy.spatial.transform import Rotation

from cryodisc.forward.cryojax_model import CryoJAXModel, _image_config
from cryodisc.forward.optics import CTFParams
from cryodisc.grids.fourier import shell_metrics

VOLTAGE_IN_KILOVOLTS = 300.0
ANGLES = (20.0, 80.0, -10.0)


@pytest.fixture(scope="module")
def pdb_path(tmp_path_factory):
    """A small synthetic structure, so the tests need no network access."""
    coords = np.random.default_rng(0).uniform(-12.0, 12.0, (60, 3))
    directory = tmp_path_factory.mktemp("pdb")
    return write_pdb(directory / "synthetic.pdb", atom_records(coords, atoms=ALPHA_CARBON))


@pytest.fixture(scope="module")
def model(pdb_path):
    return CryoJAXModel.from_structure(pdb_path, box_size=128, pixel_size=1.0)


@pytest.fixture(scope="module")
def gaussian_map():
    """A sum of isotropic 3D Gaussians, whose projection is analytic at every pose.

    Isotropy is the point, and `conftest.gaussian_blobs` states why: the line integral stays a sum of
    2D Gaussians and gives an exact reference at any orientation. Widths are generous here so the map
    is band-limited and interpolation is the only error left.

    Centers and widths are kept well inside the box, since the analytic reference has infinite tails
    and the sampled map does not: tails truncated at the edge would swamp the interpolation error
    being measured.
    """
    centers, sigmas, amplitudes, build = gaussian_blobs(64, 10, seed=1, spread=9.0, sigma_range=(2.5, 4.0))
    return build(centers), centers, sigmas, amplitudes


def test_ctf_zeros_match_analytic():
    """CTF zero positions agree with the analytic prediction from defocus and wavelength.

    This is the one check that validates units end to end -- defocus in A, Cs in mm, the
    relativistic wavelength, and the amplitude-contrast phase all enter the zero positions, so an
    A-vs-nm or factor-of-10 slip moves them.
    """
    defocus, cs_in_mm, amplitude_contrast = 12000.0, 2.7, 0.07
    wavelength = float(wavelength_from_kilovolts(VOLTAGE_IN_KILOVOLTS))
    cs = cs_in_mm * 1e7

    k = np.linspace(1e-4, 0.5, 200_001)
    grid = jnp.asarray(np.stack([np.zeros_like(k), k], axis=-1)[None])
    values = np.asarray(
        AstigmaticCTF(defocus, 0.0, 0.0, spherical_aberration_in_mm=cs_in_mm)(
            grid, wavelength, amplitude_contrast_ratio=amplitude_contrast
        )[0]
    )
    sign = np.sign(values)
    numeric = k[:-1][sign[:-1] != sign[1:]][:4]

    # sin(chi - phase_ac) = 0 at chi - phase_ac = -m pi, with
    # chi = 2 pi (-defocus lambda k^2 / 2 + Cs lambda^3 k^4 / 4)
    phase_ac = np.arctan(amplitude_contrast / np.sqrt(1.0 - amplitude_contrast**2))
    analytic = []
    for m in range(1, 5):
        roots = np.roots(
            [
                (np.pi / 2) * cs * wavelength**3,
                -np.pi * defocus * wavelength,
                -(phase_ac - m * np.pi),
            ]
        )
        positive = roots[np.isreal(roots)].real
        analytic.append(np.sqrt(positive[positive > 0].min()))

    assert np.allclose(numeric, analytic, atol=1e-4)


def test_projection_integrates_to_scattering_mass(model):
    """A bare projection integrates to the volume's total scattering amplitude.

    `sum * pixel_size^2` is the invariant, not `sum`, so this also pins the pixel-size scaling.
    """
    image = np.asarray(model.forward(*ANGLES))
    pixel_size = float(np.asarray(model.image_config.pixel_size))
    mass = float(np.asarray(model.volume.amplitudes).sum())
    assert image.sum() * pixel_size**2 == pytest.approx(mass, rel=1e-4)


@pytest.mark.parametrize("defocus", [5000.0, 12000.0, 20000.0, 30000.0])
def test_contrast_sign_is_negative(model, defocus):
    """Protein is dark in contrast mode, at every defocus.

    A flipped sign turns every downstream cross-correlation negative, which is far cheaper to
    catch here than inside a matched filter.
    """
    image = np.asarray(model.forward(*ANGLES, CTFParams(defocus)))
    assert image.flat[np.argmax(np.abs(image))] < 0.0


def test_pose_convention_is_intrinsic_zyz():
    """`EulerAnglePose` is scipy's INTRINSIC zyz, and is not the extrinsic convention.

    Checked at a generic angle triple on purpose: at phi = 0 or psi = 0 the two conventions
    coincide, so a check at those angles passes vacuously.
    """
    angles = (17.0, 41.0, -63.0)
    actual = np.asarray(
        EulerAnglePose(phi_angle=angles[0], theta_angle=angles[1], psi_angle=angles[2]).rotation.as_matrix()
    )
    intrinsic = Rotation.from_euler("zyz", angles, degrees=True).as_matrix()
    extrinsic = Rotation.from_euler("ZYZ", angles, degrees=True).as_matrix()
    assert np.allclose(actual, intrinsic, atol=1e-5)
    assert not np.allclose(actual, extrinsic, atol=1e-2)


def test_amp_ratio_catches_level_error_that_ncc_misses(model):
    """A pure level error is invisible to NCC and FRC, and visible to `amp_ratio`.

    This is the reason `shell_metrics` reports a level statistic at all, so it is worth pinning:
    if a future refactor normalizes `amp_ratio`, every gate built on it goes blind.
    """
    angles = jnp.asarray(np.random.default_rng(0).uniform(0.0, 180.0, (8, 3)))
    bank = model.forward_batch(angles, CTFParams(12000.0))

    box_size = bank.shape[-1]
    k = np.fft.fftfreq(box_size)
    ky, kx = np.meshgrid(k, k, indexing="ij")
    damping = np.where(np.sqrt(ky**2 + kx**2) > 0.25, 0.7, 1.0)
    damaged = jnp.asarray(np.real(np.fft.ifft2(np.fft.fft2(np.asarray(bank), axes=(1, 2)) * damping, axes=(1, 2))))

    def ncc(image_a, image_b):
        a = np.asarray(image_a) - np.asarray(image_a).mean()
        b = np.asarray(image_b) - np.asarray(image_b).mean()
        return (a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b))

    assert np.median([ncc(a, b) for a, b in zip(damaged, bank)]) > 0.99  # NCC misses it

    metrics = shell_metrics(damaged, bank, n_shells=8)
    high = np.asarray(metrics["k_cyc_px"]) > 0.3
    assert np.allclose(np.asarray(metrics["frc"])[high], 1.0, atol=1e-3)  # nor does FRC
    assert np.allclose(np.asarray(metrics["amp_ratio"])[high], 0.7, atol=1e-2)


def test_padding_invariance(pdb_path):
    """A larger box, center-cropped, reproduces the smaller box.

    Guards the padding and centering path. Bare projections only: with a CTF the two boxes sample
    the transfer function on different k grids, so they legitimately differ.
    """
    small = np.asarray(CryoJAXModel.from_structure(pdb_path, box_size=128, pixel_size=1.0).forward(*ANGLES))
    large = np.asarray(CryoJAXModel.from_structure(pdb_path, box_size=192, pixel_size=1.0).forward(*ANGLES))
    lo, hi = (192 - 128) // 2, (192 + 128) // 2
    cropped = large[lo:hi, lo:hi]
    assert np.linalg.norm(cropped - small) / np.linalg.norm(small) < 1e-2


def _exact_projection(box_size, centers, sigmas, amplitudes, angles):
    """Analytic line integral along z of the rotated Gaussian sum, on the (y, x) image grid."""
    rotated = centers @ Rotation.from_euler("zyz", angles, degrees=True).as_matrix().T
    axis = np.arange(box_size) - box_size / 2.0
    y, x = np.meshgrid(axis, axis, indexing="ij")
    image = np.zeros((box_size, box_size))
    for center, sigma, amplitude in zip(rotated, sigmas, amplitudes):
        squared = (y - center[1]) ** 2 + (x - center[0]) ** 2
        image += amplitude * sigma * np.sqrt(2 * np.pi) * np.exp(-squared / (2 * sigma**2))
    return image


def test_voxel_projection_matches_analytic(gaussian_map):
    """The voxel path reproduces an exact projection, in both level and shape.

    This pins the pose convention for voxel maps too: the rotation is applied as `R c`, and the
    transpose would leave a 40% error.

    Mass gets the looser tolerance on purpose. The atom path conserves it to 1e-4, but sinc
    correction rescales amplitudes at every frequency including DC, so the voxel path trades exact
    mass for an accurate spectrum. That is the right trade here, since a discriminability is a
    weighted sum over all k rather than a total.
    """
    grid, *gaussians = gaussian_map
    model = CryoJAXModel.from_voxel_grid(grid, pixel_size=1.0)
    image = np.asarray(model.forward(*ANGLES))
    exact = _exact_projection(grid.shape[-1], *gaussians, ANGLES)

    assert np.linalg.norm(image - exact) / np.linalg.norm(exact) < 1e-2
    assert image.sum() == pytest.approx(exact.sum(), rel=2e-2)


def test_voxel_defaults_beat_cryojax_defaults(gaussian_map):
    """`sinc_correction` and `pad_scale` are corrections, not preferences.

    cryoJAX's own defaults leave a several-percent error in the shells that carry nearly all the
    power, which is a level error of exactly the kind `shell_metrics` exists to catch and which
    would propagate squared into a discriminability. Pinned so a refactor cannot quietly drop them.
    """
    grid, *gaussians = gaussian_map
    box_size = grid.shape[-1]
    exact = _exact_projection(box_size, *gaussians, ANGLES)

    def error(volume):
        model = CryoJAXModel(volume=volume, image_config=_image_config(box_size, 1.0, VOLTAGE_IN_KILOVOLTS))
        image = np.asarray(model.forward(*ANGLES))
        return np.linalg.norm(image - exact) / np.linalg.norm(exact)

    library = error(FourierVoxelGridVolume.from_real_voxel_grid(grid))
    ours = error(CryoJAXModel.from_voxel_grid(grid, pixel_size=1.0).volume)
    assert library > 0.05
    assert ours < library / 5.0
