"""`fourier`, `rungs.marginal` and `fourier.shell_metrics` against an explicitly summed transform.

The conventions on the Fourier grid are the ones that fail silently: a shifted array indexed with
unshifted rings swaps low and high `k`, a negated ramp agrees only on centered particles, a whitening
that keeps DC still runs. So the reference sums the transform mode by mode, writes the frequency
wrap out by hand, and builds the shells from its own radii.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from cryodisc.grids.fourier import (
    frequency_grid,
    frequency_vectors,
    phase_ramp,
    shell_indices,
    shell_metrics,
    whitening_weight,
)
from cryodisc.rungs.marginal import known_pose_log_ratio

from . import ref_estimators as estimators
from . import ref_images as ref

BOX = 16
"""Small, because the reference transform is an explicit sum and its cost grows fast."""


def test_the_explicit_transform_reproduces_the_fft():
    """The reference's own foundation. If the two transforms disagree, every comparison below is
    about the reference rather than about production."""
    rng = np.random.default_rng(0)
    image = rng.normal(size=(BOX, BOX))
    np.testing.assert_allclose(ref.explicit_dft2(image), np.fft.fft2(image), atol=1e-9)


@pytest.mark.parametrize("box_size", [15, 16])
def test_the_frequency_grid_matches_a_hand_written_wrap(box_size):
    """Where Nyquist lands and which modes are negative, at both parities -- an odd box has no
    unpaired Nyquist mode and an even one does, and production must agree at both."""
    pixel_size = 1.3
    expected = ref.explicit_frequencies(box_size, pixel_size)
    k_y, k_x = frequency_vectors(box_size, pixel_size)
    # production is float32, the reference float64, so they differ in the eighth digit and no closer
    np.testing.assert_allclose(np.asarray(k_y)[:, 0], expected, atol=1e-7)
    np.testing.assert_allclose(np.asarray(k_x)[0, :], expected, atol=1e-7)

    radius = np.sqrt(expected[:, None] ** 2 + expected[None, :] ** 2)
    np.testing.assert_allclose(np.asarray(frequency_grid(box_size, pixel_size)), radius, atol=1e-7)


def test_the_whitening_drops_dc_and_matches_the_reference_elsewhere():
    """Particles are mean-subtracted, so DC carries no signal; weighting it would weight a pure
    offset, and every rung would still run -- just no longer on a common scale."""
    rng = np.random.default_rng(1)
    spectrum = rng.uniform(0.5, 2.0, (BOX, BOX))
    np.testing.assert_allclose(
        np.asarray(whitening_weight(jnp.asarray(spectrum))), ref.whitening_weight(spectrum), atol=1e-9
    )


@pytest.mark.parametrize("shift", [(2.0, -1.0), (0.5, 0.25), (-3.0, 4.0)])
def test_the_phase_ramp_matches_an_independently_built_one(shift):
    """Built here from an outer product of two hand-written 1D ramps, so the sign and the axis order
    are both under test rather than inherited."""
    produced = np.asarray(phase_ramp(jnp.asarray([list(shift)]), BOX))[0]
    # the ramp argument reaches 2 pi * 4 * 0.5 = 12.6 rad at the largest shift here, and float32
    # carries that with a relative error of ~1e-7, so the sine and cosine land within ~2e-6
    np.testing.assert_allclose(produced, ref.explicit_phase_ramp(np.asarray(shift), BOX), atol=1e-5)


def test_an_integer_ramp_is_a_real_space_roll():
    """The strongest statement of the sign convention, and it involves no phases on the other side:
    multiplying by the ramp must move the image the way `numpy.roll` does."""
    rng = np.random.default_rng(2)
    image = rng.normal(size=(BOX, BOX))
    ramp = np.asarray(phase_ramp(jnp.asarray([[3.0, -2.0]]), BOX))[0]
    moved = np.real(np.fft.ifft2(ref.explicit_dft2(image) * ramp))
    np.testing.assert_allclose(moved, np.roll(image, (3, -2), axis=(0, 1)), atol=1e-6)


def test_the_shell_assignment_matches_the_reference_layout():
    """Equal-width in `|k|` out to the AXIS Nyquist, so the grid's corners fall outside every shell.
    A reference that binned to the corner radius instead would put modes in shells production drops."""
    n_shells = 6
    index, centers = shell_indices(BOX, n_shells)
    expected_index, expected_centers = ref.shell_assignment(BOX, n_shells)
    np.testing.assert_array_equal(np.asarray(index), expected_index)
    np.testing.assert_allclose(np.asarray(centers), expected_centers, atol=1e-9)
    assert int(np.asarray(index)[0, 0]) == 0
    assert (np.asarray(index) == n_shells).any()  # the corners really are excluded


def test_the_known_pose_log_ratio_matches_the_whitened_definition():
    """`Lambda = <mu_A - mu_B, Y - (mu_A + mu_B)/2>_w`, written out on images. Production computes it
    as a cross term minus a power difference, which is the same thing rearranged, so this checks the
    rearrangement."""
    rng = np.random.default_rng(3)
    templates_a = rng.normal(size=(BOX, BOX)) + 1j * rng.normal(size=(BOX, BOX))
    templates_b = templates_a + 0.3 * (rng.normal(size=(BOX, BOX)) + 1j * rng.normal(size=(BOX, BOX)))
    images = rng.normal(size=(5, BOX, BOX)) + 1j * rng.normal(size=(5, BOX, BOX))
    spectrum = rng.uniform(0.5, 2.0, (BOX, BOX))

    # one template per PARTICLE, which is what the known-pose rung scores: each particle is read at
    # its own true pose, and here that pose happens to be the same one
    produced = np.asarray(
        known_pose_log_ratio(
            jnp.asarray(images),
            jnp.asarray(np.broadcast_to(templates_a, images.shape)),
            jnp.asarray(np.broadcast_to(templates_b, images.shape)),
            jnp.asarray(spectrum),
        )
    )
    expected = ref.whitened_log_ratio(images, templates_a, templates_b, spectrum)
    np.testing.assert_allclose(produced, expected, rtol=2e-4, atol=1e-5)


def test_the_log_ratio_has_the_gaussian_law_its_index_predicts():
    """The end-to-end statement the whole ladder rests on: for a common-covariance Gaussian pair the
    log-ratio is `N(+-d^2/2, d^2)`, with the SAME `d^2` the whitened separation gives. Checked by
    simulating from the model rather than by rearranging it."""
    rng = np.random.default_rng(4)
    n_modes = 96
    mean_a = rng.normal(size=n_modes)
    mean_b = mean_a + 0.15 * rng.normal(size=n_modes)
    variance = rng.uniform(0.5, 2.0, n_modes)
    d2 = estimators.true_d2_gaussian(mean_a, mean_b, variance)

    n = 40_000
    noise = rng.normal(size=(n, n_modes)) * np.sqrt(variance)
    from_a = estimators.llr_gaussian(mean_a + noise, mean_a, mean_b, variance)
    from_b = estimators.llr_gaussian(mean_b + noise, mean_a, mean_b, variance)

    assert from_a.mean() == pytest.approx(+0.5 * d2, rel=0.03)
    assert from_b.mean() == pytest.approx(-0.5 * d2, rel=0.03)
    assert from_a.var() == pytest.approx(d2, rel=0.03)
    assert estimators.ref_mean_gap(np.concatenate([from_a, from_b]), np.arange(2 * n) < n) == pytest.approx(
        d2, rel=0.03
    )


def test_the_shell_statistics_match_an_explicit_loop_over_shells():
    """`amp_ratio` carries the level and `frc` the shape. Both are summed here shell by shell in a
    plain loop, where production scatters into shells with an indexed add."""
    rng = np.random.default_rng(5)
    n_shells = 5
    reference_bank = rng.normal(size=(4, BOX, BOX))
    # a deliberate level difference, so `amp_ratio` has something to see
    test_bank = reference_bank * 0.8 + 0.05 * rng.normal(size=(4, BOX, BOX))

    produced = shell_metrics(jnp.asarray(test_bank), jnp.asarray(reference_bank), n_shells=n_shells)
    amplitude, correlation = ref.shell_amplitude_and_correlation(test_bank, reference_bank, n_shells)

    np.testing.assert_allclose(np.asarray(produced["amp_ratio"]), amplitude, rtol=2e-4)
    np.testing.assert_allclose(np.asarray(produced["frc"]), correlation, rtol=2e-4)
    # and the identity that ties them together, which neither side computed the other way
    np.testing.assert_allclose(
        np.asarray(produced["rel_sq_err"]), amplitude**2 - 2.0 * correlation * amplitude + 1.0, rtol=2e-4
    )


def test_a_pure_level_change_moves_the_amplitude_and_not_the_correlation():
    """The blind spot that motivates having both: scaling a bank leaves every normalized statistic
    at 1.0 while the level halves. A gate reading only the correlation could not see it."""
    rng = np.random.default_rng(6)
    n_shells = 5
    bank = rng.normal(size=(3, BOX, BOX))
    produced = shell_metrics(jnp.asarray(0.5 * bank), jnp.asarray(bank), n_shells=n_shells)

    np.testing.assert_allclose(np.asarray(produced["amp_ratio"]), 0.5, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(produced["frc"]), 1.0, rtol=1e-4)
