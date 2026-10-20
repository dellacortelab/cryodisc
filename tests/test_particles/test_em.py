"""Recovery checks on the EM fit.

Each test generates particles from known parameters and asks whether the fit returns them. The
templates are arbitrary smooth images rather than cryoJAX renders, because the fit does not care
where a template came from and synthetic ones keep these fast.
"""

import numpy as np
import pytest
from jax import numpy as jnp

from cryodisc.grids.fourier import phase_ramp, shell_indices
from cryodisc.grids.poses import shift_grid
from cryodisc.microscope import CollectionPlan, Transfer, defocus_groups_for
from cryodisc.particles.em import calibrate_at_known_poses, fit_noise_model

BOX_SIZE = 32
N_PIXELS = BOX_SIZE**2


def shell_average(values, n_shells=BOX_SIZE // 2):
    """Average `values` within the same shells the fit smooths over.

    The fit returns a shell-averaged spectrum, so a per-mode truth is the wrong reference: where the
    true spectrum swings inside one shell, no shell-averaged estimator can follow it, and comparing
    against the unaveraged truth would charge the fit for that.
    """
    index = np.asarray(shell_indices(BOX_SIZE, n_shells)[0]).reshape(-1)
    totals = np.bincount(index, weights=values.reshape(-1), minlength=n_shells + 1)
    counts = np.maximum(np.bincount(index, minlength=n_shells + 1), 1)
    return (totals / counts)[index].reshape(values.shape)


def make_templates(rng, n_states, n_poses, box_size=BOX_SIZE):
    """Smooth real images, returned as their fft2.

    Smooth on purpose: templates whose power is flat to Nyquist make every mode equally informative,
    which is not the regime the fit runs in and would hide a shell-dependent error.
    """
    k = np.fft.fftfreq(box_size)
    falloff = np.exp(-(k[:, None] ** 2 + k[None, :] ** 2) / (2 * 0.12**2))
    noise = rng.normal(size=(n_states, n_poses, box_size, box_size))
    return np.fft.fft2(np.real(np.fft.ifft2(np.fft.fft2(noise) * falloff)))


def make_particles(rng, templates, noise_spectrum, contrast, state_of, pose_of):
    """Particles as fft2 of real images, so the conjugate symmetry is the real one."""
    chosen = templates[state_of, pose_of]
    real_noise = np.fft.ifft2(np.fft.fft2(rng.normal(size=chosen.shape)) * np.sqrt(noise_spectrum / N_PIXELS)).real
    return contrast * chosen + np.fft.fft2(real_noise)


def colored_spectrum(level, box_size=BOX_SIZE):
    """A noise spectrum that rises toward low k, as a real one does."""
    k = np.fft.fftfreq(box_size)
    radius = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    return level * (1.0 + 8.0 * np.exp(-(radius**2) / (2 * 0.1**2)))


def test_recovers_amplitude_occupancy_and_spectrum():
    """The three parameters come back from data generated at known values.

    Occupancy is set to 0.3 rather than 0.5 so a fit that merely reported its own initialization
    would fail.
    """
    rng = np.random.default_rng(0)
    n_particles, n_states, n_poses = 3000, 2, 6
    contrast, level = 2.5, 4.0
    templates = make_templates(rng, n_states, n_poses)
    noise_spectrum = colored_spectrum(level)
    state_of = (rng.random(n_particles) > 0.3).astype(int)
    pose_of = rng.integers(0, n_poses, n_particles)
    images = make_particles(rng, templates, noise_spectrum, contrast, state_of, pose_of)

    fit = fit_noise_model(jnp.asarray(images), jnp.asarray(templates), n_iterations=200)

    assert float(fit.contrast) == pytest.approx(contrast, rel=0.05)
    assert float(fit.state_weights[0]) == pytest.approx(0.3, abs=0.03)

    fitted = np.asarray(fit.noise_spectrum)
    target = shell_average(noise_spectrum)
    assert np.allclose(fitted, target, rtol=0.1)
    # and the shape, not just the level: the spectrum spans 9x and the fit has to span it too
    assert fitted.max() / fitted.min() == pytest.approx(target.max() / target.min(), rel=0.1)


def test_the_occupancy_converges_when_the_particles_are_NOT_separable():
    """The stopping rule has to watch the occupancy, not only sigma^2.

    Random templates are near-orthogonal, so occupancy is right after one iteration whatever the
    stopping rule watches. Ambiguity has to come from the two states being SIMILAR, so state B here
    is state A plus a 3% perturbation. sigma^2 settles almost immediately; the occupancy is a
    separate fixed-point iteration whose contraction rate goes to 1 as the states approach each
    other, which is the regime every interesting specimen sits in.
    """
    rng = np.random.default_rng(7)
    n_particles, n_poses = 4000, 4
    base = make_templates(rng, 1, n_poses)
    perturbation = make_templates(np.random.default_rng(11), 1, n_poses)
    templates = np.concatenate([base, base + 0.03 * perturbation], axis=0)

    state_of = (rng.random(n_particles) > 0.25).astype(int)
    pose_of = rng.integers(0, n_poses, n_particles)
    images = jnp.asarray(make_particles(rng, templates, colored_spectrum(4.0), 1.0, state_of, pose_of))

    fit = fit_noise_model(images, jnp.asarray(templates), n_iterations=400)

    # The pool must actually BE ambiguous or this test has no more power than the one above. Sum
    # the POSE axis out first: `responsibilities` is (n, state, pose), and the max over both axes
    # is ~1 whenever the pose posterior is peaked, however ambiguous the STATE is.
    by_state = np.asarray(fit.responsibilities).sum(axis=2)
    assert np.median(by_state.max(axis=1)) < 0.95, "particles are too separable for this test to bite"

    drawn = float((state_of == 0).mean())
    assert float(fit.state_weights[0]) == pytest.approx(drawn, abs=0.05)
    # and specifically not stuck near the 1/2 initialization, which is what the old rule returned
    assert abs(float(fit.state_weights[0]) - 0.5) > 0.15


def test_log_likelihood_increases_monotonically():
    """EM cannot decrease the observed-data likelihood, which is the one structural guarantee here.

    Checked by running the fit to successively higher iteration caps, since the loop is a
    `while_loop` and keeps no history.
    """
    rng = np.random.default_rng(1)
    templates = make_templates(rng, 2, 4)
    state_of = (rng.random(400) > 0.4).astype(int)
    pose_of = rng.integers(0, 4, 400)
    images = jnp.asarray(make_particles(rng, templates, colored_spectrum(3.0), 2.0, state_of, pose_of))

    values = [
        float(fit_noise_model(images, jnp.asarray(templates), n_iterations=n, tolerance=0.0).log_likelihood)
        for n in range(1, 9)
    ]
    assert np.all(np.diff(values) > -1e-3 * np.abs(values[0]))
    assert values[-1] > values[0]


def test_per_particle_transfer_is_needed_when_present():
    """With per-particle transfer, supplying it recovers the contrast and ignoring it does not.

    The negative half is the point. Folding one averaged transfer into the templates leaves a
    per-particle model error, which the fit absorbs into `a` and sigma^2 rather than reporting.
    """
    rng = np.random.default_rng(2)
    n_particles, n_states, n_poses, n_groups = 2000, 2, 4, 5
    contrast = 2.0
    templates = make_templates(rng, n_states, n_poses)

    k = np.fft.fftfreq(BOX_SIZE)
    radius_squared = k[:, None] ** 2 + k[None, :] ** 2
    defocus = np.linspace(8000.0, 20000.0, n_groups)
    transfer = np.sin(2 * np.pi * 0.5 * defocus[:, None, None] * 0.0197 * radius_squared[None])
    group_of = rng.integers(0, n_groups, n_particles)

    state_of = (rng.random(n_particles) > 0.35).astype(int)
    pose_of = rng.integers(0, n_poses, n_particles)
    noise_spectrum = colored_spectrum(2.0)
    images = make_particles(rng, templates * transfer[group_of[0]], noise_spectrum, contrast, state_of, pose_of)
    # regenerate with the true per-particle transfer rather than one group's
    chosen = templates[state_of, pose_of] * transfer[group_of]
    real_noise = np.fft.ifft2(np.fft.fft2(rng.normal(size=chosen.shape)) * np.sqrt(noise_spectrum / N_PIXELS)).real
    images = contrast * chosen + np.fft.fft2(real_noise)

    with_transfer = fit_noise_model(
        jnp.asarray(images),
        jnp.asarray(templates),
        transfer=Transfer(values=jnp.asarray(transfer), index=jnp.asarray(group_of)),
        n_iterations=200,
    )
    without = fit_noise_model(
        jnp.asarray(images),
        jnp.asarray(templates * transfer.mean(0)),
        n_iterations=200,
    )

    assert float(with_transfer.contrast) == pytest.approx(contrast, rel=0.05)
    assert abs(float(without.contrast) - contrast) > 5 * abs(float(with_transfer.contrast) - contrast)


def test_pose_weights_correct_a_nonuniform_grid():
    """Quadrature weights matter: an unweighted prior over a clustered pose grid tilts the posterior.

    Duplicated poses stand in for a dense patch of an SO(3) grid. Under a uniform prior the three
    duplicates carry 3/5 of the prior mass where the truth is 1/3, and passing the multiplicity as a
    log weight is what removes the tilt.

    Run at low SNR on purpose. When the likelihood is sharp it swamps the prior and both answers are
    right to 1e-6, so a test at workable SNR would pass while measuring nothing. Real particles are
    in the noisy regime, which is exactly where the prior bites.
    """
    rng = np.random.default_rng(3)
    n_particles = 1500
    base = make_templates(rng, 1, 3)[0]
    templates = np.stack([base[[0, 0, 0, 1, 2]]])  # pose 0 appears three times
    multiplicity = np.array([3.0, 3.0, 3.0, 1.0, 1.0])

    pose_of = rng.integers(0, 3, n_particles)
    images = make_particles(rng, np.stack([base]), colored_spectrum(2.0), 0.03, np.zeros(n_particles, int), pose_of)

    def mass_on_pose_zero(log_weights):
        fit = fit_noise_model(
            jnp.asarray(images),
            jnp.asarray(templates),
            pose_log_weights=None if log_weights is None else jnp.asarray(log_weights),
            n_iterations=200,
        )
        return float(np.asarray(fit.responsibilities)[:, 0, :3].sum(1).mean())

    truth = float((pose_of == 0).mean())
    uniform = mass_on_pose_zero(None)
    weighted = mass_on_pose_zero(-np.log(multiplicity))
    assert abs(weighted - truth) < abs(uniform - truth)


def test_a_supplied_spectrum_with_a_zero_mode_does_not_poison_the_fit():
    """The documented default init, mean particle power, is zero at DC for mean-subtracted particles.

    One zero mode reaches every parameter through 1/sigma^2, so the floor the M step applies has to
    cover the initial spectrum too rather than only the ones the iteration produces.
    """
    rng = np.random.default_rng(0)
    templates = make_templates(rng, n_states=2, n_poses=3)
    particles = make_particles(
        rng, templates, colored_spectrum(0.5), 1.0, rng.integers(0, 2, 64), rng.integers(0, 3, 64)
    )
    particles[:, 0, 0] = 0.0
    fit = fit_noise_model(
        jnp.asarray(particles), jnp.asarray(templates), noise_spectrum_init=jnp.abs(particles).mean(0) ** 2
    )

    assert float(jnp.abs(particles).mean(0)[0, 0]) == 0.0
    assert bool(jnp.isfinite(fit.contrast) & jnp.isfinite(fit.log_likelihood))
    assert bool(jnp.isfinite(fit.responsibilities).all())


def test_one_based_group_ids_are_rejected_rather_than_clamped():
    """JAX clamps an out-of-bounds gather instead of raising, so without the check every particle
    lands on a neighboring group's transfer and the fit converges to a confidently wrong answer."""
    rng = np.random.default_rng(3)
    templates = make_templates(rng, n_states=2, n_poses=3)
    particles = make_particles(
        rng, templates, colored_spectrum(0.5), 1.0, rng.integers(0, 2, 32), rng.integers(0, 3, 32)
    )
    transfer = Transfer(values=jnp.ones((4, BOX_SIZE, BOX_SIZE)), index=jnp.asarray(rng.integers(1, 5, 32)))
    with pytest.raises(RuntimeError, match="transfer.index out of range"):
        fit_noise_model(jnp.asarray(particles), jnp.asarray(templates), transfer=transfer, n_iterations=2)


def shifted_particles(rng, templates, shifts, level=0.05):
    """Particles whose true poses are translated, so the shift latent has something to recover."""
    n = shifts.shape[0]
    clean = templates[rng.integers(0, templates.shape[0], n), rng.integers(0, templates.shape[1], n)]
    clean = clean * np.asarray(phase_ramp(jnp.asarray(shifts, float), BOX_SIZE))
    spectrum = colored_spectrum(level)
    noise = np.fft.fft2(np.fft.ifft2(np.fft.fft2(rng.normal(size=clean.shape)) * np.sqrt(spectrum / N_PIXELS)).real)
    return jnp.asarray(clean + noise)


def test_the_shift_prior_width_comes_back_from_shifted_particles():
    """The second moment of the shift posterior is what identifies sigma, so it is compared against
    the width of the shifts actually used rather than the one they were drawn from: rounding onto
    the grid and clipping at `max_shift` both move the truth, and neither is the estimator's fault."""
    rng = np.random.default_rng(7)
    templates = make_templates(rng, n_states=2, n_poses=2)
    for sigma in (1.5, 3.0):
        shifts = np.clip(np.round(rng.normal(0.0, sigma, (800, 2))), -6.0, 6.0)
        fit = fit_noise_model(
            shifted_particles(rng, templates, shifts),
            jnp.asarray(templates),
            shifts=shift_grid(BOX_SIZE, max_shift=6.0),
            n_iterations=60,
        )
        empirical = np.sqrt((shifts**2).sum(-1).mean() / 2.0)
        assert float(fit.shift_sigma) == pytest.approx(empirical, rel=0.05)


def test_the_width_is_fitted_rather_than_read_off_the_grid():
    """The grid supplies candidates; its own `log_weights` are replaced by the prior being fitted.
    Two grids that differ only in their stated prior therefore have to agree."""
    rng = np.random.default_rng(8)
    templates = make_templates(rng, n_states=2, n_poses=2)
    shifts = np.clip(np.round(rng.normal(0.0, 2.0, (600, 2))), -6.0, 6.0)
    images = shifted_particles(rng, templates, shifts)

    fits = [
        fit_noise_model(
            images, jnp.asarray(templates), shifts=shift_grid(BOX_SIZE, max_shift=6.0, sigma=sigma), n_iterations=60
        )
        for sigma in (None, 0.25)
    ]
    assert float(fits[0].shift_sigma) == pytest.approx(float(fits[1].shift_sigma), rel=1e-4)
    assert float(fits[0].shift_sigma) == pytest.approx(np.sqrt((shifts**2).sum(-1).mean() / 2.0), rel=0.05)


def test_no_shift_grid_leaves_the_fit_as_it_was():
    """One candidate at the origin, so the shift latent is present but carries no width."""
    rng = np.random.default_rng(9)
    templates = make_templates(rng, n_states=2, n_poses=3)
    particles = make_particles(
        rng, templates, colored_spectrum(0.5), 1.2, rng.integers(0, 2, 64), rng.integers(0, 3, 64)
    )
    fit = fit_noise_model(jnp.asarray(particles), jnp.asarray(templates))

    assert float(fit.shift_sigma) == 0.0
    assert fit.responsibilities.shape == (64, 2, 3)
    assert float(fit.contrast) == pytest.approx(1.2, rel=0.05)


def test_fewer_particles_than_groups_does_not_poison_the_fit():
    """An empty defocus bin averages 0/0 into an all-NaN CTF row, and NOTHING downstream raises.

    No particle gathers that row, so a pose search over the same transfer looks clean; but the
    M-step reduces over every group and `0 * nan` takes the whole spectrum to NaN. Capping the
    group count at the particle count is what makes "no group is empty" true rather than intended.
    """
    plan = CollectionPlan(dose=40.0, pixel_size=1.5, defocus_range=(8000.0, 20000.0))
    transfer = Transfer.from_defocus(jnp.asarray([9000.0, 14000.0, 19000.0]), plan, BOX_SIZE, n_groups=8)
    assert transfer.values.shape[0] == 3
    assert bool(jnp.all(jnp.isfinite(transfer.values)))
    np.testing.assert_array_equal(np.sort(np.asarray(transfer.index)), [0, 1, 2])

    rng = np.random.default_rng(0)
    templates = jnp.asarray(make_templates(rng, n_states=2, n_poses=2))
    spectrum = colored_spectrum(0.5)
    particles = jnp.asarray(
        make_particles(rng, np.asarray(templates), spectrum, 1.0, np.array([0, 1, 0]), np.array([0, 1, 1]))
    )
    fit = fit_noise_model(particles, templates, transfer=transfer, n_iterations=20)
    assert bool(jnp.all(jnp.isfinite(fit.noise_spectrum)))
    assert bool(jnp.all(jnp.isfinite(fit.responsibilities)))


def test_an_empty_stack_or_group_count_is_refused():
    plan = CollectionPlan(dose=40.0, pixel_size=1.5)
    with pytest.raises(ValueError, match="at least one particle and one group"):
        Transfer.from_defocus(jnp.zeros(0), plan, BOX_SIZE)
    with pytest.raises(ValueError, match="at least one particle and one group"):
        Transfer.from_defocus(jnp.asarray([1e4]), plan, BOX_SIZE, n_groups=0)


def test_the_default_group_count_keeps_the_ctf_phase_inside_a_group_small():
    """Too few defocus bins is not a coarser answer but a wrong one.

    A group spanning `w` of defocus spans `pi lambda w k^2` of CTF phase at Nyquist; past about half
    a turn its templates anti-correlate with its own particles, so the count must keep the group's
    phase below half a turn.
    """
    plan = CollectionPlan(dose=40.0, pixel_size=1.5, defocus_range=(8000.0, 20000.0))
    n_groups = defocus_groups_for(plan)
    lam = 0.0197  # 300 kV, A
    span = (20000.0 - 8000.0) / n_groups
    assert np.pi * lam * span * plan.nyquist**2 < 0.5 * np.pi
    # the count has to grow with the range and with the SQUARE of the resolution
    assert (
        defocus_groups_for(CollectionPlan(dose=40.0, pixel_size=0.75, defocus_range=(8000.0, 20000.0))) > 3 * n_groups
    )
    assert defocus_groups_for(CollectionPlan(dose=40.0, pixel_size=1.5, defocus_range=(8000.0, 8000.0))) == 1


def test_a_huge_stack_does_not_wrap_the_group_index():
    """`arange(n) * n_groups` overflows int32 past 2^31, wrapping the index negative and putting
    NaN CTF rows back -- the same failure the group cap was added to remove."""
    n = 50_000
    defocus = jnp.asarray(np.random.default_rng(0).uniform(8000.0, 20000.0, n))
    transfer = Transfer.from_defocus(defocus, CollectionPlan(dose=40.0, pixel_size=1.5), 4, n_groups=n)
    index = np.asarray(transfer.index)
    assert index.min() == 0 and index.max() == n - 1
    assert bool(jnp.all(jnp.isfinite(transfer.values)))


def make_own_pose_templates(rng, bank, pose_of, mix):
    """Each particle's true template: its bank node tilted `mix` of the way toward an unrelated one.

    The stand-in for a bank that does not quite contain the particle's pose. A real bank misses by an
    angle; here it misses by a fixed overlap, which is the same thing for a least-squares contrast
    and is exactly computable -- a fit against the bank can recover at most `sqrt(1 - mix^2)` of the
    truth, so the bias has a predicted size rather than merely a sign.
    """
    extra = make_templates(rng, 1, len(pose_of))[0]
    chosen = bank[0, pose_of]
    # matched in power before mixing, so `mix` is an overlap and not also a rescaling
    scaled = extra * np.sqrt(
        (np.abs(chosen) ** 2).sum(axis=(1, 2), keepdims=True) / (np.abs(extra) ** 2).sum(axis=(1, 2), keepdims=True)
    )
    return np.sqrt(1.0 - mix**2) * chosen + mix * scaled


def test_the_known_pose_calibration_recovers_a_planted_amplitude_and_spectrum():
    """The parameters come back when every particle is handed its own template.

    No latent is left, so this is the easy case and it should be exact rather than approximate: it
    is the reference the harder test below reads the bank's bias against.
    """
    rng = np.random.default_rng(0)
    n_particles = 2000
    contrast, level = 2.5, 4.0
    templates = make_templates(rng, 1, n_particles)[0]
    noise_spectrum = colored_spectrum(level)
    images = make_particles(
        rng, templates[None, :], noise_spectrum, contrast, np.zeros(n_particles, int), np.arange(n_particles)
    )

    fit = calibrate_at_known_poses(jnp.asarray(images), jnp.asarray(templates)[None], n_iterations=200)

    assert float(fit.contrast) == pytest.approx(contrast, rel=0.02)
    assert np.allclose(np.asarray(fit.noise_spectrum), shell_average(noise_spectrum), rtol=0.1)
    # the assignment was given, not inferred, so nothing here is a posterior to be read as confidence
    assert np.asarray(fit.state_weights).tolist() == [1.0]
    assert float(fit.shift_sigma) == 0.0


def test_an_uncentered_pool_fits_the_amplitude_low_until_the_shift_is_marginalized():
    """The same bias as the coarse bank below, from the OTHER latent, and much larger.

    A picked particle is not centered, and an uncorrected offset attenuates the cross term, so the
    least-squares `a` shrinks toward zero. `d^2` goes as `a^2`, so what reads here as a 10x error in
    the contrast is a 100x error in the number the calibration exists to produce.
    """
    rng = np.random.default_rng(11)
    n_particles = 800
    contrast, level = 2.5, 4.0
    templates = make_templates(rng, 1, n_particles)[0]
    noise_spectrum = colored_spectrum(level)
    images = make_particles(
        rng, templates[None, :], noise_spectrum, contrast, np.zeros(n_particles, int), np.arange(n_particles)
    )
    offsets = rng.integers(-3, 4, (n_particles, 2)).astype(float)
    uncentered = jnp.asarray(images) * phase_ramp(jnp.asarray(offsets), BOX_SIZE)
    stacked = jnp.asarray(templates)[None]

    assumed = calibrate_at_known_poses(uncentered, stacked, n_iterations=200)
    fitted = calibrate_at_known_poses(uncentered, stacked, shifts=shift_grid(BOX_SIZE, 4.0), n_iterations=200)

    assert float(assumed.contrast) < 0.5 * contrast
    assert float(fitted.contrast) == pytest.approx(contrast, rel=0.02)
    # sqrt(E|t|^2 / 2) for offsets uniform on -3..3 per axis, the convention `fit_noise_model` reports
    assert float(fitted.shift_sigma) == pytest.approx(2.0, rel=0.1)


def test_a_bank_that_misses_the_pose_fits_the_amplitude_low_and_the_known_pose_route_does_not():
    """The bias that makes an absolute `d^2` off a real stack unquotable, and the fix, side by side.

    Particles are generated at poses the bank does NOT contain -- each is its nearest node tilted
    `mix` of the way toward an unrelated template -- so the best a fit against that bank can do is
    `sqrt(1 - mix^2)` of the true contrast. That is the same mechanism as a coarse SO(3) grid
    missing a particle's true rotation, and `d^2` goes as `a^2`, so a 0.8 here is a `d^2` 1.6x low.

    The known-pose calibration sees each particle's own template and so has no bank to be coarse.
    """
    rng = np.random.default_rng(3)
    n_particles, n_poses, mix = 1500, 8, 0.6
    contrast, level = 2.5, 2.0
    bank = make_templates(rng, 1, n_poses)
    pose_of = rng.integers(0, n_poses, n_particles)
    own = make_own_pose_templates(rng, bank, pose_of, mix)
    images = jnp.asarray(
        make_particles(
            rng, own[None, :], colored_spectrum(level), contrast, np.zeros(n_particles, int), np.arange(n_particles)
        )
    )

    against_bank = fit_noise_model(images, jnp.asarray(bank), n_iterations=200)
    at_own_poses = calibrate_at_known_poses(images, jnp.asarray(own)[None], n_iterations=200)

    # the bank fit lands ON the overlap it is capped by, to under a percent -- the cap is analytic,
    # so this pins the MECHANISM and not merely the direction of the error
    recoverable = np.sqrt(1.0 - mix**2)
    assert float(against_bank.contrast) == pytest.approx(contrast * recoverable, rel=0.02)
    assert float(against_bank.contrast) < contrast
    # and the known-pose route is unbiased on the same particles, which is the whole claim
    assert float(at_own_poses.contrast) == pytest.approx(contrast, rel=0.05)


def test_the_known_pose_calibration_uses_a_per_particle_transfer_when_given():
    """Ignoring a per-particle transfer biases the contrast, and passing it removes the bias.

    The negative half is the point, as in the bank fit's own transfer test: without it the fit sees
    templates systematically larger than the particles they generated and shrinks `a` to compensate.
    """
    rng = np.random.default_rng(5)
    n_particles = 1200
    contrast = 3.0
    templates = make_templates(rng, 1, n_particles)[0]
    plan = CollectionPlan(dose=40.0, pixel_size=1.5, defocus_range=(8000.0, 20000.0))
    defocus = jnp.asarray(rng.uniform(8000.0, 20000.0, n_particles))
    transfer = Transfer.from_defocus(defocus, plan, BOX_SIZE)

    values = np.asarray(transfer.values)[np.asarray(transfer.index)]
    noise = np.fft.fft2(
        np.fft.ifft2(
            np.fft.fft2(rng.normal(size=(n_particles, BOX_SIZE, BOX_SIZE))) * np.sqrt(colored_spectrum(1.0) / N_PIXELS)
        ).real
    )
    images = jnp.asarray(contrast * values * templates + noise)

    with_it = calibrate_at_known_poses(images, jnp.asarray(templates)[None], transfer=transfer, n_iterations=200)
    without_it = calibrate_at_known_poses(images, jnp.asarray(templates)[None], n_iterations=200)

    assert float(with_it.contrast) == pytest.approx(contrast, rel=0.05)
    assert float(without_it.contrast) < 0.8 * contrast


def test_the_known_pose_calibration_refuses_a_stack_of_templates_it_cannot_pair_off():
    """One template per particle, not one per (state, pose) -- the two shapes are both plausible.

    A (s, p, y, x) stack passed here would broadcast into something with a meaning nobody intended,
    so the shape is checked rather than trusted.
    """
    images = jnp.zeros((4, BOX_SIZE, BOX_SIZE), dtype=complex)
    with pytest.raises(ValueError, match="one template per particle per state"):
        calibrate_at_known_poses(images, jnp.zeros((2, 3, BOX_SIZE, BOX_SIZE), dtype=complex))
    with pytest.raises(ValueError, match="one template per particle per state"):
        calibrate_at_known_poses(images, jnp.zeros((1, 5, BOX_SIZE, BOX_SIZE), dtype=complex))
    with pytest.raises(ValueError, match="must be square"):
        calibrate_at_known_poses(
            jnp.zeros((4, BOX_SIZE, BOX_SIZE + 2), dtype=complex),
            jnp.zeros((1, 4, BOX_SIZE, BOX_SIZE + 2), dtype=complex),
        )


def test_a_pure_pool_of_the_second_state_is_not_calibrated_against_the_first():
    """Collapsing the STATE latent along with the pose latent reads the contrast low.

    A single-state pool still has to discover which state it is. Handed only state A's templates it
    fits the contrast to a systematic template mismatch, exactly as a coarse bank does; handed both
    it puts the occupancy on B and recovers the truth.
    """
    rng = np.random.default_rng(7)
    n_particles = 1500
    contrast = 2.5
    both = make_templates(rng, 2, n_particles)
    images = jnp.asarray(
        make_particles(rng, both, colored_spectrum(2.0), contrast, np.ones(n_particles, int), np.arange(n_particles))
    )

    against_a_alone = calibrate_at_known_poses(images, jnp.asarray(both[:1]), n_iterations=200)
    over_both = calibrate_at_known_poses(images, jnp.asarray(both), n_iterations=200)
    told_the_truth = calibrate_at_known_poses(
        images, jnp.asarray(both), state_of=jnp.ones(n_particles, int), n_iterations=200
    )

    # fitting the occupancy finds the right state and the right contrast
    assert float(over_both.state_weights[1]) > 0.99
    assert float(over_both.contrast) == pytest.approx(contrast, rel=0.05)
    # being told is at least as good, and is what a labeled pool should pass
    assert float(told_the_truth.contrast) == pytest.approx(contrast, rel=0.05)
    # and the wrong single state is biased LOW, which is the failure this guards
    assert float(against_a_alone.contrast) < 0.7 * contrast


def planted_envelope(shape, n_shells=4, box_size=BOX_SIZE):
    """`a(k)` rising over the shells, as an envelope or B-factor mismatch would make it.

    Returned on the fft grid so it multiplies a template stack directly. The out-of-band corner bin
    takes the outermost shell's value, there being no shell beyond it to interpolate toward.
    """
    index = np.asarray(shell_indices(box_size, n_shells)[0])
    return np.asarray([*shape, shape[-1]])[index]


def test_the_amplitude_can_be_fitted_per_shell_and_recovers_a_planted_envelope():
    """The test the option exists for: one scalar `a` cannot represent a frequency-dependent error.

    A scalar assumes the templates are wrong by the same factor everywhere. Planting a factor that
    RISES with `k` -- what an envelope or B-factor mismatch against the renderer looks like -- the
    scalar fit returns a single number that is wrong at both ends, while `a(k)` returns the shape.
    """
    rng = np.random.default_rng(11)
    n_particles, n_shells = 4000, 4
    shape = np.array([0.6, 1.0, 1.6, 2.4])
    templates = make_templates(rng, 2, n_particles)
    state_of = np.arange(n_particles) % 2
    images = jnp.asarray(
        make_particles(
            rng,
            templates,
            colored_spectrum(1.0),
            planted_envelope(shape),
            state_of,
            np.arange(n_particles),
        )
    )
    stack, labels = jnp.asarray(templates), jnp.asarray(state_of)

    scalar = calibrate_at_known_poses(images, stack, state_of=labels, n_iterations=200)
    shells = calibrate_at_known_poses(images, stack, state_of=labels, n_amplitude_shells=n_shells, n_iterations=200)

    assert scalar.amplitude_spectrum is None, "a scalar fit must not invent a spectrum"
    assert shells.amplitude_spectrum is not None
    index = np.asarray(shell_indices(BOX_SIZE, n_shells)[0]).reshape(-1)
    fitted = np.asarray(shells.amplitude_spectrum).reshape(-1)
    for shell, truth in enumerate(shape):
        assert fitted[index == shell].mean() == pytest.approx(truth, rel=0.05), f"shell {shell}"

    # and the scalar it replaces cannot be right at both ends AT ONCE, which is the point: whatever
    # single number least squares picks, one end of the band is off by tens of percent
    one_number = float(scalar.contrast)
    assert not (0.9 * shape[0] < one_number < 1.1 * shape[0] and 0.9 * shape[-1] < one_number < 1.1 * shape[-1])


def test_a_flat_envelope_makes_the_shell_fit_reproduce_the_scalar_exactly():
    """The reduction that makes `a(k)` safe to turn on: it must find nothing where nothing is there.

    A fitted `a(k)` has more freedom than a scalar, so it will always fit the data at least as well,
    and a rising `a(k)` is only evidence of an envelope if a FLAT truth comes back flat. The reported
    scalar is the whitened-power-weighted mean of the shells, so on constant truth it must land on the
    scalar answer -- not merely near it -- and that is what makes the two runs comparable.
    """
    rng = np.random.default_rng(12)
    n_particles, contrast = 3000, 1.8
    templates = make_templates(rng, 2, n_particles)
    state_of = np.arange(n_particles) % 2
    images = jnp.asarray(
        make_particles(rng, templates, colored_spectrum(1.0), contrast, state_of, np.arange(n_particles))
    )
    stack, labels = jnp.asarray(templates), jnp.asarray(state_of)

    scalar = calibrate_at_known_poses(images, stack, state_of=labels, n_iterations=200)
    shells = calibrate_at_known_poses(images, stack, state_of=labels, n_amplitude_shells=6, n_iterations=200)

    assert float(shells.contrast) == pytest.approx(float(scalar.contrast), rel=1e-4)
    assert float(scalar.contrast) == pytest.approx(contrast, rel=0.05)
    # flat to a few percent: this is sampling noise on six independent ratios, not structure
    fitted = np.asarray(shells.amplitude_spectrum).reshape(-1)
    assert fitted.std() / fitted.mean() < 0.05
    # the noise spectrum is not paid for out of the contrast's new freedom
    assert np.allclose(np.asarray(shells.noise_spectrum), np.asarray(scalar.noise_spectrum), rtol=0.05)


def test_a_shell_with_no_template_power_falls_back_instead_of_dividing_by_nothing():
    """Where the templates say nothing about a gain, `a(k)` must not report one.

    A shell the templates put no power in has a zero denominator in the least squares, and dividing
    by it would hand back an unbounded number that then multiplies the templates in every rung
    downstream. Those shells take the whole-band answer, which is the honest statement that there is
    no evidence there to move off the average.
    """
    rng = np.random.default_rng(13)
    n_particles, n_shells = 600, 8
    templates = make_templates(rng, 1, n_particles)
    # emptied above the third shell, so the outer five carry exactly zero template power
    index = np.asarray(shell_indices(BOX_SIZE, n_shells)[0])
    templates = np.where(index[None, None] < 3, templates, 0.0)
    images = jnp.asarray(
        make_particles(rng, templates, colored_spectrum(1.0), 2.0, np.zeros(n_particles, int), np.arange(n_particles))
    )

    shells = calibrate_at_known_poses(images, jnp.asarray(templates), n_amplitude_shells=n_shells, n_iterations=200)

    fitted = np.asarray(shells.amplitude_spectrum).reshape(-1)
    assert np.all(np.isfinite(fitted)), "an empty shell divided by its own zero denominator"
    # the empty shells all agree, being the one whole-band fallback rather than five noisy ratios
    empty = fitted[index.reshape(-1) >= 3]
    assert empty.std() == pytest.approx(0.0, abs=1e-6)
    assert 0.0 < empty[0] < 10.0


def test_the_shell_count_must_be_at_least_one_shell():
    """Zero or negative shells is a caller who means `None`, and would index an empty reduction."""
    images = jnp.zeros((4, BOX_SIZE, BOX_SIZE), dtype=complex)
    templates = jnp.zeros((1, 4, BOX_SIZE, BOX_SIZE), dtype=complex)
    with pytest.raises(ValueError, match="at least one shell"):
        calibrate_at_known_poses(images, templates, n_amplitude_shells=0)
    with pytest.raises(ValueError, match="at least one shell"):
        calibrate_at_known_poses(images, templates, n_amplitude_shells=-3)
