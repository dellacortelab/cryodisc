"""Checks on the shared Fourier grid and the conventions riding on it.

These are the conventions that fail silently rather than loudly: a whitening that keeps DC still
runs, a grid built at the wrong Nyquist still has the right shape. So each one is pinned against
something outside the module -- a closed form, a real-space roll, an explicit sum.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from cryodisc.grids.fourier import (
    frequency_grid,
    phase_ramp,
    shell_indices,
    shell_metrics,
    whitening_weight,
)


def test_the_frequency_grid_is_the_one_radial_layout():
    """Shells, phase ramps and the resolution model all index this, so it is built in one place."""
    box_size = 64
    cyc_px = frequency_grid(box_size)
    assert float(cyc_px[0, 0]) == 0.0
    assert float(cyc_px[box_size // 2, 0]) == pytest.approx(0.5)
    np.testing.assert_allclose(np.asarray(cyc_px), np.asarray(cyc_px).T, atol=1e-7)
    # the pixel size is the only thing between cyc/px and inverse A, and it is an argument, not a
    # per-module habit
    np.testing.assert_allclose(np.asarray(frequency_grid(box_size, 1.5)), np.asarray(cyc_px) / 1.5, rtol=1e-6)
    # shell_indices reads the same grid, so its band edges land where this says they do
    index, centers = shell_indices(box_size, 8)
    assert int(index[0, 0]) == 0
    assert bool(jnp.all(index[cyc_px >= 0.5] == 8))
    assert float(centers[-1]) == pytest.approx(0.5 - 0.5 / 16)


def test_the_whitening_drops_dc_and_inverts_everything_else():
    """The convention a rung cannot be allowed to re-derive. DC is dropped rather than inverted:
    keeping it would weight the one mode that carries no signal, since particles are mean-subtracted,
    and everything else is `1 / spectrum`."""
    spectrum = jnp.asarray(np.random.default_rng(0).uniform(0.5, 2.0, size=(8, 8)))
    weight = whitening_weight(spectrum)

    assert float(weight[0, 0]) == 0.0
    off_dc = np.ones((8, 8), dtype=bool)
    off_dc[0, 0] = False
    np.testing.assert_allclose(np.asarray(weight)[off_dc], 1.0 / np.asarray(spectrum)[off_dc], rtol=1e-6)


def test_a_non_square_spectrum_is_refused():
    """DC is at (0, 0) of a square grid; a rectangular one would index somewhere else entirely."""
    with pytest.raises(ValueError, match="square"):
        whitening_weight(jnp.ones((8, 16)))


def test_the_ramp_sign_matches_a_real_space_roll():
    """Pinned against `np.roll` at an asymmetric shift, so a transposed or negated convention fails.
    A sign error here is invisible on centered particles and wrong on every real one."""
    box_size = 16
    image = np.random.default_rng(0).normal(size=(box_size, box_size))
    ramp = np.asarray(phase_ramp(jnp.asarray([[2.0, -1.0]]), box_size))[0]

    shifted = np.real(np.fft.ifft2(np.fft.fft2(image) * ramp))
    assert np.allclose(shifted, np.roll(image, (2, -1), axis=(0, 1)), atol=1e-5)


def test_the_ramp_leaves_every_mode_at_unit_modulus():
    """`rungs.marginal` factors a shift out of the model power on exactly this grounds: `|ramp| = 1`,
    so a translation enters the cross term alone. A ramp that damped anything would break that."""
    ramp = np.asarray(phase_ramp(jnp.asarray([[3.0, -2.5], [0.0, 0.0]]), 16))
    np.testing.assert_allclose(np.abs(ramp), 1.0, rtol=1e-6)


def test_the_signed_axes_agree_with_the_radial_grid():
    """`frequency_vectors` is the only place a per-axis frequency comes from -- an azimuth for
    astigmatism, a translation tangent for `rungs.local` -- and it must be the SAME grid the radial
    one is built on, since two `fftfreq` calls at different pixel sizes would put a CTF's
    astigmatism on one Nyquist and its zeros on another. This is also what lets a caller wanting
    only `|k|` take `frequency_grid` instead of re-norming these by hand."""
    from cryodisc.grids.fourier import frequency_vectors

    box_size = 32
    k_y, k_x = frequency_vectors(box_size, 1.5)
    np.testing.assert_allclose(
        np.asarray(jnp.sqrt(k_y**2 + k_x**2)), np.asarray(frequency_grid(box_size, 1.5)), rtol=1e-6
    )
    assert k_y.shape == k_x.shape == (box_size, box_size)
    # row index is y, so k_y varies down the rows and k_x across them
    np.testing.assert_allclose(np.asarray(k_y), np.asarray(k_x).T, rtol=1e-6)
    assert float(k_y[1, 0]) == pytest.approx(1.0 / (box_size * 1.5))
    assert float(k_x[0, 1]) == pytest.approx(1.0 / (box_size * 1.5))
    # and it is signed, which is the whole reason it exists
    assert float(k_y[box_size - 1, 0]) < 0.0


# --- shell_metrics: the shell-resolved bank comparison. The statistics are pinned by
# construction, so what is worth testing is the shells with nothing in them.

SHELL_BOX = 16


def shell_banks(rng, n=4, box_size=SHELL_BOX):
    return jnp.asarray(rng.normal(size=(n, box_size, box_size)))


def test_a_uniformly_scaled_bank_reads_its_scale_and_nothing_else():
    """Shell 0 is skipped: at `box_size // 2` shells it holds the DC mode alone, which the images
    are mean-subtracted against, so both powers are float noise and their ratio is meaningless."""
    reference = shell_banks(np.random.default_rng(0))
    metrics = shell_metrics(0.7 * reference, reference, n_shells=SHELL_BOX // 2)
    assert np.allclose(metrics["amp_ratio"][1:], 0.7, rtol=1e-4)
    assert np.allclose(metrics["frc"][1:], 1.0, rtol=1e-4)


def test_an_empty_shell_reports_nan_and_leaves_the_filled_ones_alone():
    """Past about `box_size` shells some hold no modes at all, and 0/0 would take the whole
    dictionary with them -- including shells that do have data, once a caller averages."""
    reference = shell_banks(np.random.default_rng(1))
    metrics = shell_metrics(0.7 * reference, reference, n_shells=SHELL_BOX)

    empty = np.asarray(metrics["n_modes"]) == 0
    assert empty.any(), "the case under test did not occur"
    for key in ("amp_ratio", "frc", "rel_sq_err"):
        values = np.asarray(metrics[key])
        assert np.isnan(values[empty]).all()
        assert np.isfinite(values[~empty]).all()


def test_the_per_image_ratio_is_what_a_one_image_bank_reports():
    """`validation.bank` reads its per-template spread off `amp_ratio_per_image`, which must match a
    `shell_metrics` call over one-image banks, image by image.

    The tolerance is tight rather than zero because the fusion XLA picks may differ on the GPU this
    gate runs on."""
    n_shells = SHELL_BOX // 4
    reference = shell_banks(np.random.default_rng(2), n=5)
    # a per-image level difference, so the spread across images is something rather than nothing
    under_test = reference * jnp.asarray([0.4, 0.7, 1.0, 1.3, 2.0])[:, None, None]

    per_image = np.asarray(shell_metrics(under_test, reference, n_shells=n_shells)["amp_ratio_per_image"])
    assert per_image.shape == (5, n_shells)
    for i in range(5):
        one_image = shell_metrics(under_test[i][None], reference[i][None], n_shells=n_shells)["amp_ratio"]
        np.testing.assert_allclose(per_image[i], np.asarray(one_image), rtol=1e-6)
    # each image carries its own scale, and the bank statistic is a power-weighted reduction over
    # them rather than any one of them -- which is the whole reason both are reported
    scales = np.broadcast_to(np.array([0.4, 0.7, 1.0, 1.3, 2.0])[:, None], per_image.shape)
    np.testing.assert_allclose(per_image, scales, rtol=1e-4)


def test_an_empty_shell_is_nan_in_the_per_image_ratio_too():
    """The convention lives in one place, so the per-image ratio inherits it. A caller taking a
    percentile down the bank axis reads NaN there and drops the shell, exactly as it does for the
    bank figure; a 0/0 that happened to land on a finite number would be silently averaged in."""
    reference = shell_banks(np.random.default_rng(3))
    metrics = shell_metrics(0.7 * reference, reference, n_shells=SHELL_BOX)

    empty = np.asarray(metrics["n_modes"]) == 0
    assert empty.any(), "the case under test did not occur"
    per_image = np.asarray(metrics["amp_ratio_per_image"])
    assert np.isnan(per_image[:, empty]).all()
    assert np.isfinite(per_image[:, ~empty]).all()


def test_the_shell_sum_keeps_the_out_of_band_bin_it_scatters_into():
    """`particles.em` gathers its shell averages back through the same `shell_indices` output, so
    index `n_shells` has to hold the corners' own total. Dropping it would clamp every corner mode
    onto the outermost real shell, silently -- jax indexes out of bounds without complaining."""
    from cryodisc.grids.fourier import shell_sum

    box_size, n_shells = 16, 8
    index, _ = shell_indices(box_size, n_shells)
    counts = shell_sum(jnp.ones((box_size, box_size)), index, n_shells)

    assert counts.shape == (n_shells + 1,)
    assert float(counts.sum()) == box_size**2
    assert float(counts[-1]) > 0.0, "the corners lie outside every shell, so the last bin is not empty"
    # flat or gridded, the same answer: `em` holds both the values and the index already flattened
    np.testing.assert_allclose(
        np.asarray(shell_sum(jnp.ones(box_size**2), index.reshape(-1), n_shells)), np.asarray(counts)
    )
    # and it is the scatter-add it claims to be, against an explicit per-shell sum
    values = jnp.asarray(np.random.default_rng(0).normal(size=(box_size, box_size)))
    totals = np.asarray(shell_sum(values, index, n_shells))
    for shell in range(n_shells + 1):
        assert totals[shell] == pytest.approx(float(np.asarray(values)[np.asarray(index) == shell].sum()), abs=1e-4)
