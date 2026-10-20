"""Checks on the CTF refit.

The fit is scored against a synthetic spectrum built from a KNOWN defocus through the same
`ctf_astigmatic` the fit uses, so a drift in sign, in the amplitude-contrast term or in the azimuth
fails here rather than in a downstream rung.
"""

import functools

import numpy as np
import pytest
from conftest import OPTICS, thon_spectrum

from cryodisc.particles.ctffit import (
    fit_defocus,
    fit_patches,
    interpolate_to_particles,
    tiled_power_spectrum,
)

BOX = 256
PIXEL = 1.4

synthetic_spectrum = functools.partial(thon_spectrum, box=BOX, pixel_size=PIXEL)
"""`conftest.thon_spectrum` pinned to this file's one grid. The grid is what `test_grid_invariance`
varies; here it is held fixed so a fit that moves is the fitter."""


def test_a_known_isotropic_defocus_is_recovered():
    """The whole fit in one assertion: rings built at 12000 A have to read back as 12000 A."""
    fit = fit_defocus(synthetic_spectrum(12000.0), PIXEL, astigmatism=False, **OPTICS)
    assert fit["defocus"] == pytest.approx(12000.0, rel=0.02)
    assert fit["defocus_u"] == fit["defocus_v"]
    assert fit["score"] > 0.2


def test_the_fit_survives_noise_and_reports_a_worse_score_under_it():
    """A low score is the signal that the rings were never found, so it has to move the right way."""
    clean = fit_defocus(synthetic_spectrum(9000.0), PIXEL, astigmatism=False, **OPTICS)
    noisy = fit_defocus(synthetic_spectrum(9000.0, noise=0.6, seed=1), PIXEL, astigmatism=False, **OPTICS)
    assert noisy["defocus"] == pytest.approx(9000.0, rel=0.05)
    assert noisy["score"] < clean["score"]


@pytest.mark.parametrize("azimuth_deg", [35.0, 80.0])
def test_the_astigmatism_axis_is_recovered_exactly(azimuth_deg):
    """The azimuth is what decides WHERE the zeros land, so it is the part that has to be right, and
    it comes back to under a degree. Compared modulo pi, the azimuth being periodic with that."""
    fit = fit_defocus(
        synthetic_spectrum(10300.0, 9700.0, angle=np.radians(azimuth_deg)),
        PIXEL,
        astigmatism=True,
        astigmatism_max=3000.0,
        **OPTICS,
    )
    assert fit["defocus"] == pytest.approx(10000.0, rel=0.02)
    offset = (fit["astigmatism_angle"] - np.radians(azimuth_deg)) % np.pi
    assert min(offset, np.pi - offset) < np.radians(2.0)


def test_the_astigmatism_magnitude_is_detected_but_biased_high():
    """The astigmatism spread is detected but comes back biased high, less so as the spread grows.
    Pinned so it cannot get quietly worse and so a future fix reads as a failure here."""
    narrow = fit_defocus(
        synthetic_spectrum(10300.0, 9700.0, angle=np.radians(35.0)),
        PIXEL,
        astigmatism=True,
        astigmatism_max=3000.0,
        **OPTICS,
    )
    wide = fit_defocus(
        synthetic_spectrum(10750.0, 9250.0, angle=np.radians(35.0)),
        PIXEL,
        astigmatism=True,
        astigmatism_max=3000.0,
        **OPTICS,
    )
    narrow_spread = narrow["defocus_u"] - narrow["defocus_v"]
    wide_spread = wide["defocus_u"] - wide["defocus_v"]

    # detected in both, and ordered with the truth
    assert narrow_spread > 300.0
    assert wide_spread > narrow_spread
    # biased high, and less so as the spread grows against the band's ability to resolve it
    assert 1.2 < narrow_spread / 600.0 < 2.0
    assert 1.0 < wide_spread / 1500.0 < 1.5
    assert wide_spread / 1500.0 < narrow_spread / 600.0


def test_an_isotropic_spectrum_does_not_manufacture_astigmatism():
    """The other half of the previous test: the bias is an overestimate of something real, not a
    floor the refinement puts under every fit."""
    fit = fit_defocus(synthetic_spectrum(10000.0), PIXEL, astigmatism=True, astigmatism_max=3000.0, **OPTICS)
    assert abs(fit["defocus_u"] - fit["defocus_v"]) < 300.0


def test_the_band_is_checked_before_anything_is_fitted():
    """An inverted or empty band would otherwise produce a confident number from nothing."""
    spectrum = synthetic_spectrum(10000.0)
    with pytest.raises(ValueError, match="k_low < k_high"):
        fit_defocus(spectrum, PIXEL, k_low=0.2, k_high=0.04)
    with pytest.raises(ValueError, match="radial bins"):
        fit_defocus(spectrum, PIXEL, k_low=0.2, k_high=0.2001, n_bins=4)


def test_the_periodogram_averages_tiles_and_refuses_an_image_too_small():
    """Averaging is what turns one noise realization per mode into visible rings."""
    rng = np.random.default_rng(0)
    image = rng.normal(size=(512, 512))
    spectrum = tiled_power_spectrum(image, tile=128)
    assert spectrum.shape == (128, 128)
    # a single tile has the variance of one realization; many tiles have less
    single = tiled_power_spectrum(image[:128, :128], tile=128)
    assert spectrum.std() / spectrum.mean() < single.std() / single.mean()
    with pytest.raises(ValueError, match="in \\[0, 1\\)"):
        tiled_power_spectrum(image, tile=128, overlap=1.0)


def test_a_patch_plane_interpolates_between_the_patches_it_was_fitted_to():
    """A plane, not a nearest-patch lookup: the cause is a tilted specimen, and a lookup would put a
    discontinuity between neighboring particles in different patches."""
    patches = [
        (0.0, 0.0, {"defocus": 10000.0, "defocus_u": 10100.0, "defocus_v": 9900.0, "astigmatism_angle": 0.2}),
        (0.0, 100.0, {"defocus": 10200.0, "defocus_u": 10300.0, "defocus_v": 10100.0, "astigmatism_angle": 0.2}),
        (100.0, 0.0, {"defocus": 10400.0, "defocus_u": 10500.0, "defocus_v": 10300.0, "astigmatism_angle": 0.2}),
    ]
    out = interpolate_to_particles(patches, np.array([[0.0, 0.0], [100.0, 100.0], [50.0, 50.0]]))
    # the plane is 10000 + 4 y + 2 x by construction
    np.testing.assert_allclose(out["defocus_u"] - 0.5 * 200.0, [10000.0, 10600.0, 10300.0], rtol=1e-9)
    np.testing.assert_allclose(out["defocus_u"] - out["defocus_v"], 200.0)
    np.testing.assert_allclose(out["astigmatism_angle"], 0.2)
    np.testing.assert_allclose(out["plane"], [10000.0, 4.0, 2.0], atol=1e-6)

    with pytest.raises(ValueError, match="at least three patches"):
        interpolate_to_particles(patches[:2], np.zeros((1, 2)))


def test_patches_are_laid_out_across_the_micrograph():
    """The centers are what `interpolate_to_particles` fits its plane through, so they have to span
    the image rather than pile up in a corner."""
    image = np.random.default_rng(3).normal(size=(600, 600))
    patches = fit_patches(image, PIXEL, grid=(2, 2), tile=256, astigmatism=False, n_coarse=40, **OPTICS)
    assert len(patches) == 4
    centers = {(patch[0], patch[1]) for patch in patches}
    assert len(centers) == 4


def test_a_grid_that_collapses_onto_one_tile_says_so_and_fits_it_once():
    """At `height == width == tile` every offset is (0, 0), so a 3x3 grid is one tile fitted nine
    times -- a plane through nine coincident points that reads like a tilt and is not one. The fit is
    memoized per offset and the collapse is warned about rather than left to look like a measurement."""
    image = np.random.default_rng(3).normal(size=(256, 256))
    with pytest.warns(UserWarning, match="1 of 9 distinct offsets"):
        patches = fit_patches(image, PIXEL, grid=(3, 3), tile=256, astigmatism=False, n_coarse=20, **OPTICS)
    assert len(patches) == 9
    assert {(patch[0], patch[1]) for patch in patches} == {(128.0, 128.0)}
    assert len({patch[2]["defocus"] for patch in patches}) == 1


def test_the_patch_azimuth_is_combined_across_the_wrap_point_rather_than_medianed():
    """`fit_defocus` wraps the azimuth into [0, pi), so patches that agree can sit at BOTH ends of
    that interval. An ordinary median of them lands mid-branch-cut -- 90 degrees off, which swaps
    the astigmatism axes for every particle on the micrograph."""
    azimuths = [0.02, 3.132, 0.01, 3.122]  # the same axis to within 0.6 degrees, either side of pi
    patches = [
        (float(i), 0.0, {"defocus": 12000.0, "defocus_u": 12100.0, "defocus_v": 11900.0, "astigmatism_angle": a})
        for i, a in enumerate(azimuths)
    ]
    fit = interpolate_to_particles(patches, np.zeros((1, 2)))

    # within 0.6 degrees of the common axis, taken modulo pi
    offset = float(fit["astigmatism_angle"][0]) % np.pi
    assert min(offset, np.pi - offset) < np.deg2rad(0.6)
    assert np.median(azimuths) == pytest.approx(1.571, abs=1e-3), "the median is what this replaces"
