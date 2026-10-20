"""Checks on the whitened-power scalar and the SNR that is its square root.

The point of these is not that the arithmetic is right -- it is four lines -- but that the scalings
are the ones an `OperatingPoint` and `resolution.predict_resolution` assume. Each test moves one
input and asserts the exponent it enters with, because a power that went in linearly where the rest
of the package expects a square is invisible in any single number.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from cryodisc.grids.fourier import whitening_weight
from cryodisc.microscope import Transfer
from cryodisc.particles.snr import matched_filter_snr, whitened_power

BOX = 16


def bank(rng, n_poses=5):
    """fft2 of a few real templates, so the conjugate symmetry is the real one."""
    return jnp.asarray(np.fft.fft2(rng.normal(size=(n_poses, BOX, BOX))))


def flat_spectrum(level=1.0):
    return jnp.full((BOX, BOX), level)


def test_the_snr_is_the_root_of_the_power_at_any_amplitude():
    templates = bank(np.random.default_rng(0))
    power = whitened_power(templates, flat_spectrum(), contrast=1.7)

    assert matched_filter_snr(templates, flat_spectrum(), contrast=1.7) == pytest.approx(np.sqrt(power))


def test_the_power_is_the_whitened_bank_power_with_dc_dropped():
    """The scalar two runs must agree on before a ratio between them means anything. Computed here
    from the definition rather than read off the function."""
    box_size, n_poses, n_groups = 8, 3, 2
    rng = np.random.default_rng(0)
    templates = jnp.asarray(
        rng.normal(size=(n_poses, box_size, box_size)) + 1j * rng.normal(size=(n_poses, box_size, box_size))
    )
    spectrum = jnp.asarray(rng.uniform(0.5, 2.0, size=(box_size, box_size)))
    # UNEQUAL groups -- three particles in group 0 and one in group 1 -- which is what
    # `transfer_from_meta` produces on a real pool and what tells a particle-weighted average apart
    # from a per-group one. An unweighted `mean` over the group axis gives the single particle in
    # group 1 the same say as the three in group 0
    transfer = Transfer(
        values=jnp.asarray(rng.uniform(-1.0, 1.0, size=(n_groups, box_size, box_size))),
        index=jnp.asarray([0, 0, 0, 1], jnp.int32),
    )
    contrast = 1.5

    share = np.array([0.75, 0.25])
    weight = np.asarray(whitening_weight(spectrum)) * (share[:, None, None] * np.asarray(transfer.values**2)).sum(0)
    expected = contrast**2 * np.mean((np.abs(np.asarray(templates)) ** 2 * weight).sum(axis=(-2, -1)))
    assert whitened_power(templates, spectrum, transfer=transfer, contrast=contrast) == pytest.approx(
        float(expected), rel=1e-5
    )


def test_the_amplitude_enters_squared_and_the_spectrum_inversely():
    """`a` and the spectrum are one statement about signal-to-noise made twice, so the scalar has to
    move with either -- and the particle count is quadratic in the SNR, so a wrong exponent here is
    amplified downstream."""
    box_size = 8
    templates = jnp.ones((2, box_size, box_size), jnp.complex64)
    spectrum = jnp.ones((box_size, box_size))
    transfer = Transfer(values=jnp.ones((1, box_size, box_size)), index=jnp.zeros(2, jnp.int32))

    unit = whitened_power(templates, spectrum, transfer=transfer)
    # DC is dropped, so a flat bank on a flat spectrum carries one unit per off-DC mode
    assert unit == pytest.approx(box_size**2 - 1)
    assert whitened_power(templates, spectrum, transfer=transfer, contrast=3.0) == pytest.approx(9.0 * unit)
    assert whitened_power(templates, 4.0 * spectrum, transfer=transfer) == pytest.approx(unit / 4.0)
    assert matched_filter_snr(templates, spectrum, transfer=transfer, contrast=3.0) == pytest.approx(
        3.0 * matched_filter_snr(templates, spectrum, transfer=transfer)
    )


def test_the_transfer_is_averaged_over_its_groups_rather_than_taken_at_one():
    """A pool is scored against a per-particle transfer, so no single group represents it. The
    average is exact because the whitened power is linear in `T^2`, which is what this pins."""
    templates = bank(np.random.default_rng(2))
    values = jnp.stack([jnp.full((BOX, BOX), 0.5), jnp.full((BOX, BOX), 1.0)])
    transfer = Transfer(values=values, index=jnp.array([0, 1]))

    pooled = whitened_power(templates, flat_spectrum(), transfer=transfer)
    each = [
        whitened_power(templates, flat_spectrum(), transfer=Transfer(values=values[i : i + 1], index=jnp.array([0])))
        for i in (0, 1)
    ]
    assert pooled == pytest.approx(0.5 * (each[0] + each[1]), rel=1e-5)


def test_no_transfer_is_the_same_as_a_transfer_of_one():
    templates = bank(np.random.default_rng(3))
    unity = Transfer(values=jnp.ones((1, BOX, BOX)), index=jnp.zeros(1, int))

    assert whitened_power(templates, flat_spectrum(), transfer=unity) == pytest.approx(
        whitened_power(templates, flat_spectrum()), rel=1e-6
    )
