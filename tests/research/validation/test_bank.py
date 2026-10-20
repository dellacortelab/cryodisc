"""Checks on the bank-certification gate.

Every test here plants a KNOWN defect in a bank and asks whether the gate sees it. The defects are
the ones this gate exists for and the ones the old median-NCC criterion could not see: a level loss
that leaves every normalized statistic at 1.0, incoherent power that leaves the level at 1.0, a pose
permutation that leaves the shell power untouched, and a pure translation that must be forgiven.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from cryodisc.grids.fourier import frequency_grid, phase_ramp, shell_indices
from reproduction.validation.bank import (
    D2_TOL,
    bank_verdict,
    compare_banks,
    ctf_sampling_limit,
    matched_correlations,
    register_subpixel,
    shell_transfer_rms,
)

BOX = 32
N_SHELLS = 6


def smooth_bank(n_templates=6, seed=0):
    """A bank of band-limited blobs, which is what a template actually looks like."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n_templates, BOX, BOX))
    envelope = np.exp(-40.0 * np.asarray(frequency_grid(BOX)) ** 2)
    return np.real(np.fft.ifft2(np.fft.fft2(noise) * envelope))


def damp_above(bank, k_cut, factor):
    """Multiply a bank's amplitude by `factor` above `k_cut` cycles per pixel.

    The interpolation defect this gate was built for, in its purest form: a level loss confined to
    the high shells, invisible to any normalized statistic.
    """
    radius = np.asarray(frequency_grid(BOX))
    weight = np.where(radius > k_cut, factor, 1.0)
    return np.real(np.fft.ifft2(np.fft.fft2(bank) * weight))


def test_an_identical_bank_certifies_to_nyquist():
    """The null case. Anything short of a full pass here means the gate has a floor of its own, and
    every verdict below would inherit it."""
    bank = smooth_bank()
    verdict = bank_verdict(compare_banks(jnp.asarray(bank), jnp.asarray(bank), n_shells=N_SHELLS))
    assert verdict.passes
    assert verdict.passes_correlation and verdict.passes_band
    assert verdict.level_worst < 1e-4
    assert verdict.shape_worst < 1e-4
    assert verdict.certified_k == pytest.approx(0.5)


def test_a_high_frequency_level_loss_fails_the_band_while_passing_the_correlation():
    """The defect this replaces: a median NCC that passes comfortably while a quarter of the amplitude
    is gone from the band where the signal lives."""
    bank = smooth_bank()
    damaged = damp_above(bank, 0.25, 0.70)
    comparison = compare_banks(jnp.asarray(damaged), jnp.asarray(bank), n_shells=N_SHELLS, fit_scale=False)
    verdict = bank_verdict(comparison)

    assert verdict.passes_correlation  # the old gate would have stopped here
    assert not verdict.passes_band
    assert not verdict.passes
    assert verdict.failing_level.size > 0
    # the level is what saw it. The shape barely moves -- a scaled bank still correlates perfectly --
    # and the residue is the one shell straddling the cut, where part of the modes are damped and
    # part are not. Compared AGAINST the level rather than against an absolute number, since the
    # claim is that one statistic saw the defect and the other did not.
    assert verdict.shape_worst < 0.05 * verdict.level_worst
    # and the certified band stops at the damage rather than running to Nyquist
    assert 0.0 < verdict.certified_k <= 0.30


def test_incoherent_power_fails_the_shape_while_passing_the_level():
    """The symmetric blind spot. Replacing coherent amplitude with noise of the same power leaves
    the level at 1.0, so the level criterion alone would certify a bank carrying no shared signal."""
    rng = np.random.default_rng(1)
    bank = smooth_bank()
    scrambled = np.real(np.fft.ifft2(np.fft.fft2(bank) * np.exp(2j * np.pi * rng.uniform(size=bank.shape))))
    comparison = compare_banks(jnp.asarray(scrambled), jnp.asarray(bank), n_shells=N_SHELLS, fit_scale=False)
    verdict = bank_verdict(comparison)

    assert not verdict.passes_band
    assert verdict.failing_shape.size > 0
    assert verdict.shape_worst > D2_TOL


def test_a_pure_translation_is_forgiven():
    """A centering convention is not an optics failure. The level is translation-invariant by
    construction and the shape is registered, so a shifted bank must certify."""
    bank = smooth_bank()
    shifted = np.real(np.fft.ifft2(np.fft.fft2(bank) * np.asarray(phase_ramp(jnp.asarray([[2.0, -3.0]]), BOX))[0]))
    verdict = bank_verdict(compare_banks(jnp.asarray(shifted), jnp.asarray(bank), n_shells=N_SHELLS))
    assert verdict.passes


def test_a_sub_pixel_translation_is_forgiven_too():
    """The case integer registration cannot fix, and the reason `register_subpixel` exists: a
    residual half-pixel offset costs `cos(2 pi k.t)`, already zero at 0.7 Nyquist, so an unregistered
    shape statistic would report a catastrophic mismatch for a pure convention difference."""
    bank = smooth_bank()
    offset = np.array([0.5, -0.25])
    shifted = np.real(np.fft.ifft2(np.fft.fft2(bank) * np.asarray(phase_ramp(jnp.asarray([offset]), BOX))[0]))
    verdict = bank_verdict(compare_banks(jnp.asarray(shifted), jnp.asarray(bank), n_shells=N_SHELLS))
    assert verdict.passes


def test_the_registration_recovers_a_planted_sub_pixel_offset():
    """Measured directly, since everything above depends on it. The residual bound is
    `1/(2 upsample)` per axis, which at the default 32 is 0.016 px."""
    bank = smooth_bank(n_templates=1)
    offset = np.array([0.375, -1.125])
    shifted = np.real(np.fft.ifft2(np.fft.fft2(bank) * np.asarray(phase_ramp(jnp.asarray([offset]), BOX))[0]))
    # the offset that brings the SHIFTED image back onto the original is its negative
    found = np.asarray(register_subpixel(bank[0], shifted[0]))
    np.testing.assert_allclose(found, -offset, atol=1.0 / 32.0)


def test_a_permuted_bank_is_caught_by_the_correlation_and_not_by_the_shells():
    """The blind spot the correlation covers: a permutation barely moves the bank-summed shell
    power, so every spectral statistic certifies a mispaired bank."""
    bank = smooth_bank(n_templates=6)
    permuted = bank[[1, 0, 3, 2, 5, 4]]
    comparison = compare_banks(jnp.asarray(permuted), jnp.asarray(bank), n_shells=N_SHELLS, fit_scale=False)
    verdict = bank_verdict(comparison)

    assert not verdict.passes_correlation
    assert not verdict.passes
    # the shells barely notice, which is precisely why the correlation is not redundant
    assert verdict.level_worst < 0.2


def test_the_fitted_scale_absorbs_a_flat_level_error_and_is_reported():
    """Deliberate: two engines are in different units, so a gate that failed on units would be
    useless. The cost is that a frequency-flat error is invisible, which is why `scale` is on the
    comparison rather than hidden inside it."""
    bank = smooth_bank()
    comparison = compare_banks(jnp.asarray(3.0 * bank), jnp.asarray(bank), n_shells=N_SHELLS, fit_scale=True)
    assert comparison.scale == pytest.approx(3.0, rel=1e-3)
    assert bank_verdict(comparison).passes

    # without the fit, the same bank fails on level alone
    unfitted = compare_banks(jnp.asarray(3.0 * bank), jnp.asarray(bank), n_shells=N_SHELLS, fit_scale=False)
    assert unfitted.scale == 1.0
    assert not bank_verdict(unfitted).passes_band


def test_the_vectorized_correlation_keeps_the_constant_image_guard():
    """The one branch a `vmap` cannot take: `normalized_correlation` returns 0 for a constant image
    by an early return on a host-side `float(norm)`, and under the vectorized form that has to
    become a `where` -- including on the denominator, or the selected-away branch leaves a NaN."""
    bank = smooth_bank(n_templates=2)
    flat = np.zeros_like(bank)
    correlations = np.asarray(matched_correlations(jnp.asarray(bank), jnp.asarray(flat)))
    assert np.all(correlations == 0.0)


def test_the_certified_band_stops_at_the_first_failing_shell():
    """The useful output when the gate fails, since it says what the bank may still be used for."""
    bank = smooth_bank()
    verdict = bank_verdict(
        compare_banks(jnp.asarray(damp_above(bank, 0.35, 0.5)), jnp.asarray(bank), n_shells=N_SHELLS, fit_scale=False)
    )
    assert not verdict.passes
    # damage above 0.35 leaves the shells below it certifiable
    assert 0.30 <= verdict.certified_k <= 0.42


def test_a_resolution_is_quoted_when_a_pixel_size_is_given():
    """A shell frequency means nothing without one, and a table that omitted it would be the
    'projector accuracy without its grid' failure this repo already records."""
    bank = smooth_bank()
    comparison = compare_banks(
        jnp.asarray(damp_above(bank, 0.25, 0.5)), jnp.asarray(bank), n_shells=N_SHELLS, pixel_size=1.5, fit_scale=False
    )
    verdict = bank_verdict(comparison)
    assert np.isfinite(verdict.certified_resolution)
    assert verdict.certified_resolution == pytest.approx(1.5 / verdict.certified_k, rel=1e-6)
    assert np.isfinite(comparison.resolution[1:]).all()


def test_the_per_template_spread_sees_a_defect_the_bank_sum_hides():
    """The level statistic sums power over the bank, so one bad template in a large bank does not
    move it. That is reported rather than folded into the verdict, which the docstring states."""
    bank = smooth_bank(n_templates=8)
    damaged = bank.copy()
    damaged[0] = damp_above(bank[[0]], 0.15, 0.2)[0]
    comparison = compare_banks(jnp.asarray(damaged), jnp.asarray(bank), n_shells=N_SHELLS, fit_scale=False)

    assert comparison.amp_low[-1] < 0.9 * comparison.amp_ratio[-1]
    assert comparison.correlation.min() < np.median(comparison.correlation)


def test_the_per_template_spread_is_an_explicit_percentile_of_an_explicit_ratio():
    """`amp_low`/`amp_high` come out of the single `shell_metrics` call that produces the bank
    statistic. The percentiles must match what that call gives -- pinned here against a plain numpy
    ratio and percentile, which share no code with the production path.

    The pair is chosen so registration is a no-op: the defect is a zero-phase low-pass, so the
    cross-correlation peak stays at the origin. The test therefore pins the statistic it is about
    rather than the alignment underneath it."""
    bank = smooth_bank(n_templates=8)
    under_test = bank * np.array([0.5, 0.7, 0.9, 1.0, 1.0, 1.1, 1.3, 2.0])[:, None, None]
    # and two templates additionally damped at high k, so the ratio depends on the shell and not
    # only on which template it came from
    under_test[[0, 5]] = damp_above(under_test[[0, 5]], 0.25, 0.6)
    comparison = compare_banks(jnp.asarray(under_test), jnp.asarray(bank), n_shells=N_SHELLS, fit_scale=False)

    membership = np.asarray(shell_indices(BOX, N_SHELLS)[0])

    def shell_power(images):
        """Per-template power in each shell, summed by an explicit mask instead of a scatter-add."""
        spectrum = np.fft.fft2(images - images.mean(axis=(-2, -1), keepdims=True))
        return np.stack([(np.abs(spectrum[:, membership == s]) ** 2).sum(axis=1) for s in range(N_SHELLS)], axis=1)

    ratio = np.sqrt(shell_power(under_test) / shell_power(bank))
    np.testing.assert_allclose(comparison.amp_low, np.percentile(ratio, 5, axis=0), rtol=1e-4)
    np.testing.assert_allclose(comparison.amp_high, np.percentile(ratio, 95, axis=0), rtol=1e-4)


def test_the_inputs_are_checked():
    """Each of these would otherwise index or broadcast into a plausible number."""
    bank = smooth_bank()
    with pytest.raises(ValueError, match="bank shapes differ"):
        compare_banks(jnp.asarray(bank), jnp.asarray(bank[:3]))
    with pytest.raises(ValueError, match="square box"):
        compare_banks(jnp.zeros((2, 8, 16)), jnp.zeros((2, 8, 16)))
    with pytest.raises(ValueError, match="empty"):
        compare_banks(jnp.zeros((0, BOX, BOX)), jnp.zeros((0, BOX, BOX)))
    with pytest.raises(ValueError, match="no shell holds enough modes"):
        # a mode floor no shell can meet, which is the guard's own condition rather than a proxy
        bank_verdict(compare_banks(jnp.asarray(bank), jnp.asarray(bank), n_shells=N_SHELLS, min_modes=10**6))


def test_the_sampling_limit_is_the_defocus_that_sets_it_and_never_exceeds_nyquist():
    """A gate judged past where the CTF is sampled measures the box, not the engines."""
    # box 64 at 1.5 A/px and 12000 A: chi cycles faster than the grid can follow above 4.9 A
    limit = ctf_sampling_limit(64, 1.5, 12000.0, 300.0)
    assert 1.5 / limit == pytest.approx(4.9, abs=0.15)
    # a LARGER defocus oscillates faster and pushes the usable band coarser
    assert ctf_sampling_limit(64, 1.5, 24000.0, 300.0) < limit
    # and a slow enough CTF is sampled everywhere, where the cap has to hold: without it the limit
    # would claim frequencies the grid does not hold at all
    assert ctf_sampling_limit(64, 1.5, 500.0, 300.0) == 0.5


def test_transfer_is_weakest_where_the_ctf_is_and_is_read_off_the_optics_alone():
    """The exclusion that makes the level criterion stable, and why it cannot hide a defect."""
    rms = shell_transfer_rms(64, 1.5, 5000.0, 300.0, n_shells=8)
    assert rms.shape == (8,)
    assert np.all((rms >= 0.0) & (rms <= 1.0))
    # the lowest shell is the weak one: phase contrast vanishes toward DC and only Q is left
    assert rms[0] < rms[1:].min()
    # it depends on the OPTICS and on nothing measured, so a different defocus moves it
    assert not np.allclose(rms, shell_transfer_rms(64, 1.5, 20000.0, 300.0, n_shells=8))


def test_a_shell_the_microscope_barely_transfers_is_dropped_from_the_verdict():
    """An amplitude RATIO near a CTF zero divides two small numbers, so it swings with the binning
    rather than with the forward model, and such a shell must be dropped from the verdict."""
    rng = np.random.default_rng(17)
    reference = rng.normal(size=(4, 64, 64))
    under_test = reference.copy()
    # break the bank ONLY in the shell the optics barely transfers, which for 5000 A is the lowest
    spectrum = np.fft.fft2(under_test)
    index, _ = shell_indices(64, 8)
    spectrum[:, np.asarray(index) == 0] *= 3.0
    under_test = np.real(np.fft.ifft2(spectrum))

    transfer = shell_transfer_rms(64, 1.5, 5000.0, 300.0, n_shells=8)
    judged = compare_banks(jnp.asarray(under_test), jnp.asarray(reference), n_shells=8, min_transfer=0.0)
    excluded = compare_banks(
        jnp.asarray(under_test),
        jnp.asarray(reference),
        n_shells=8,
        transfer_rms=transfer,
        min_transfer=0.5,
    )
    assert judged.usable[0]
    assert not excluded.usable[0], "the weakly transferred shell should leave the verdict"
    # and the shells the microscope DOES transfer are still judged, so this is not a blanket loosening
    assert excluded.usable[1:].sum() >= 4


def test_an_unjudged_shell_does_not_extend_the_certified_band():
    """`certified_k` walked past any shell that did not FAIL, and an unusable shell cannot fail -- it
    was never judged. So the walk must stop at the last shell actually examined rather than run to
    Nyquist through the top shells `k_max_cyc_px` leaves unusable."""
    bank = jnp.asarray(smooth_bank())
    limit = 0.25
    comparison = compare_banks(bank, bank, n_shells=N_SHELLS, k_max_cyc_px=limit, pixel_size=1.0)

    assert not comparison.usable.all(), "the point of the test is that some shells go unjudged"
    verdict = bank_verdict(comparison)
    # certified only through the band that was actually examined, not to 0.5
    assert verdict.certified_k <= limit
    assert verdict.certified_resolution >= 1.0 / limit


def test_a_shell_skipped_mid_band_does_not_stop_the_walk():
    """`min_transfer` exists to skip the CTF's zeros and the low shells where phase contrast
    vanishes. Those gaps are expected, so an unusable shell must not TERMINATE the certification
    either -- stopping at the first one would certify nothing at all on a real comparison."""
    bank = jnp.asarray(smooth_bank())
    transfer = np.ones(N_SHELLS)
    transfer[1] = 0.0  # a zero of the CTF, in the middle of an otherwise perfect comparison
    comparison = compare_banks(bank, bank, n_shells=N_SHELLS, transfer_rms=transfer, min_transfer=0.5, pixel_size=1.0)

    assert not comparison.usable[1]
    assert bank_verdict(comparison).certified_k == pytest.approx(0.5)


def test_an_unjudged_shell_at_the_LOW_end_does_not_certify_the_band_from_dc():
    """The mirror of the fix above. `min_transfer` excludes the lowest shells because phase contrast
    vanishes as k goes to zero, so the walk skipped one and reported a band starting at DC -- and a
    level error in that unjudged low shell then passed. The band must start where the judging did."""
    bank = np.asarray(smooth_bank())
    transfer = np.ones(N_SHELLS)
    transfer[0] = 0.327  # what 5000 A actually transfers in the lowest shell
    comparison = compare_banks(
        jnp.asarray(bank), jnp.asarray(bank), n_shells=N_SHELLS, transfer_rms=transfer, min_transfer=0.5, pixel_size=1.0
    )

    assert not comparison.usable[0], "the fixture must leave the lowest shell unjudged"
    verdict = bank_verdict(comparison)
    # the band starts where the judging did, and says so
    assert verdict.certified_from_k > 0.0
    assert np.isfinite(verdict.certified_low_resolution)
    assert verdict.certified_k == pytest.approx(0.5)


def test_the_certified_band_stops_at_the_sampling_limit_not_the_shell_edge():
    """`usable` is decided on a shell's CENTER, so the outermost admitted shell straddles the limit
    and its outer edge would claim a band that was never sampled. The certified band must stop at the
    sampling limit, not at the shell edge past it."""
    bank = jnp.asarray(smooth_bank())
    limit = 0.30477  # deliberately mid-shell, unlike the round number the sibling test uses
    verdict = bank_verdict(compare_banks(bank, bank, n_shells=N_SHELLS, k_max_cyc_px=limit, pixel_size=1.0))

    assert verdict.certified_k <= limit + 1e-12
