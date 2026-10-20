"""Checks on the marginalized rung.

These run a simulated pool, since a marginal index has no closed form. Where a closed form does
exist -- the pose-known oracle -- it is checked against `fisher`, which computes the same quantity
by a completely different route.
"""

import numpy as np
import pytest
from conftest import gaussian_blob_models
from jax import numpy as jnp
from scipy.spatial.transform import Rotation

from cryodisc.grids.fourier import phase_ramp, whitening_weight
from cryodisc.grids.poses import ShiftGrid, pose_bank, shift_grid
from cryodisc.microscope import Transfer
from cryodisc.rungs.local import local_information
from cryodisc.rungs.marginal import (
    known_pose_d2,
    known_pose_log_ratio,
    pose_log_ratio,
)
from cryodisc.statistics.information import deflection, fisher_from_deflection

BOX_SIZE = 32
NOISE_LEVEL = 0.6
N_PARTICLES = 1000
"""Pool size for the tests that read the Fisher information about the fraction.

Those are ratios of one pool's information to another's and are flat in the pool size, so a modest
pool keeps a wide margin. The tests that compare a scored gap against a closed form do NOT draw on
this: a shared noise draw makes them exact per particle on whatever pool they are handed.
"""


def _information(from_a, from_b, fraction=0.5):
    """I(p) about the fraction, mapped from the pools' discriminability, as the production chain reads it."""
    return fisher_from_deflection(deflection(from_a, from_b), fraction)


@pytest.fixture(scope="module")
def pair():
    """Two states differing by a genuine conformational change, and their shared noise spectrum."""
    centers, build = gaussian_blob_models(BOX_SIZE, 6, seed=0, spread=5.0, sigma_range=(2.0, 3.0))
    shifted = centers.copy()
    shifted[0] += np.array([1.6, 0.0, 0.0])
    shifted[3] -= np.array([0.0, 1.4, 0.0])
    spectrum = jnp.asarray(np.full((BOX_SIZE, BOX_SIZE), NOISE_LEVEL, np.float32))
    return build(centers), build(shifted), spectrum


@pytest.fixture(scope="module")
def grid():
    return pose_bank(288)


@pytest.fixture(scope="module")
def banks(pair, grid):
    model_a, model_b, _ = pair
    return jnp.fft.fft2(model_a.forward_batch(grid.angles)), jnp.fft.fft2(model_b.forward_batch(grid.angles))


@pytest.fixture(scope="module")
def truth(pair):
    """Each particle's own templates at a continuous, off-grid pose."""
    model_a, model_b, _ = pair
    angles = jnp.asarray(Rotation.random(N_PARTICLES, random_state=3).as_euler("zyz", degrees=True).astype(np.float32))
    return jnp.fft.fft2(model_a.forward_batch(angles)), jnp.fft.fft2(model_b.forward_batch(angles))


def numpy_colored_noise(rng, spectrum, n=N_PARTICLES):
    """Noise whose per-mode power is exactly the spectrum, drawn from a NUMPY generator.

    Separate from `particles.sim.colored_noise` only because these tests share one draw across two
    pools from a `np.random.Generator`, where the real one takes a PRNG key. The scaling must track
    it: `sqrt(sigma^2 / n_pixels)`, because `fft2` of unit-variance real noise carries power `n_pixels`.
    """
    white = jnp.asarray(rng.standard_normal((n, BOX_SIZE, BOX_SIZE)).astype(np.float32))
    return jnp.fft.fft2(white) * jnp.sqrt(spectrum / (BOX_SIZE * BOX_SIZE))


def pools(rng, truth, spectrum, amplitude, ramp=1.0):
    """Two labeled pools sharing one noise draw, so a difference of means is near-deterministic.

    Sized from `truth`, so a test that only needs a ratio can pass a slice of it.
    """
    template_a, template_b = truth
    noise = numpy_colored_noise(rng, spectrum, n=template_a.shape[0])
    return amplitude * template_a * ramp + noise, amplitude * template_b * ramp + noise


def test_the_oracle_matches_the_deterministic_fisher_rung(pair, grid):
    """Two independent routes to the same d^2: a whitened norm, and a scored gap over a sampled pool.

    Nothing else pins the amplitude and whitening conventions across the two modules. A shared noise
    draw makes the agreement algebraic rather than statistical, so a few particles per pose suffice.
    The oracle's per-particle log-ratio is calibrated, so its mean gap is the discriminability directly.
    """
    model_a, model_b, spectrum = pair
    angles = jnp.asarray(np.array([[20.0, 80.0, -10.0], [137.0, 62.0, 88.0]], np.float32))
    amplitude = 0.02
    n_particles = 8

    deterministic = local_information(model_a, model_b, angles, spectrum, contrast=amplitude)
    template_a = jnp.fft.fft2(model_a.forward_batch(angles))
    template_b = jnp.fft.fft2(model_b.forward_batch(angles))

    rng = np.random.default_rng(11)
    for index in range(2):
        stacked_a = jnp.broadcast_to(template_a[index], (n_particles, BOX_SIZE, BOX_SIZE))
        stacked_b = jnp.broadcast_to(template_b[index], (n_particles, BOX_SIZE, BOX_SIZE))
        from_a, from_b = pools(rng, (stacked_a, stacked_b), spectrum, amplitude)
        gap_a = known_pose_log_ratio(from_a, stacked_a, stacked_b, spectrum, amplitude=amplitude)
        gap_b = known_pose_log_ratio(from_b, stacked_a, stacked_b, spectrum, amplitude=amplitude)
        measured = gap_a.mean() - gap_b.mean()
        assert float(measured) == pytest.approx(float(deterministic.known_pose[index]), rel=1e-3)


def test_the_closed_form_d2_is_the_mean_gap_the_oracle_scores(pair, truth):
    """`known_pose_d2` needs no stack, so it is the prediction `known_pose_log_ratio` must land on.

    Both together, because either alone is satisfiable by the wrong convention: the gap pins the
    scale (a factor on the amplitude or the whitening moves it) and the per-particle SPREAD pins that
    `d_i^2` is per particle at all, which is what `validation.coherence`'s baseline is computed from.

    Asserted PER PARTICLE: `pools` shares one noise draw between the two states, so
    `Lambda(a m_A + n) - Lambda(a m_B + n)` cancels the noise and leaves `a^2 ||m_A - m_B||^2_w = d_i^2`
    identically. A pool is still needed for the SPREAD assertion, since one pose carries no spread.
    """
    _, _, spectrum = pair
    amplitude = 0.02
    d2 = known_pose_d2(truth[0], truth[1], spectrum, amplitude=amplitude)

    from_a, from_b = pools(np.random.default_rng(21), truth, spectrum, amplitude)
    scored_a = known_pose_log_ratio(from_a, truth[0], truth[1], spectrum, amplitude=amplitude)
    scored_b = known_pose_log_ratio(from_b, truth[0], truth[1], spectrum, amplitude=amplitude)

    # Lambda|A - Lambda|B = d_i^2, particle by particle, so the pool mean is not what carries this
    np.testing.assert_allclose(np.asarray(scored_a - scored_b), np.asarray(d2), rtol=1e-3)
    # per particle and not a pool constant: these poses carry a real spread and the closed form has it
    assert float(jnp.std(d2)) > 0.05 * float(jnp.mean(d2))
    assert d2.shape == (truth[0].shape[0],)


def test_the_closed_form_d2_refuses_mismatched_template_stacks():
    spectrum = jnp.ones((8, 8))
    with pytest.raises(ValueError, match="template stacks differ"):
        known_pose_d2(jnp.zeros((3, 8, 8), complex), jnp.zeros((4, 8, 8), complex), spectrum)


def test_knowing_the_pose_beats_marginalizing_it(pair, grid, banks, truth):
    """The oracle is a ceiling: no pose treatment can do better than being told the answer."""
    _, _, spectrum = pair
    bank_a, bank_b = banks
    amplitude = 0.004
    from_a, from_b = pools(np.random.default_rng(12), truth, spectrum, amplitude)

    def marginal(images):
        return pose_log_ratio(images, bank_a, bank_b, spectrum, amplitude=amplitude, pose_log_weights=grid.log_weights)

    oracle = _information(
        known_pose_log_ratio(from_a, truth[0], truth[1], spectrum, amplitude=amplitude),
        known_pose_log_ratio(from_b, truth[0], truth[1], spectrum, amplitude=amplitude),
    )
    integrated = _information(marginal(from_a).marginal, marginal(from_b).marginal)
    assert float(oracle) > float(integrated)


def test_marginalizing_pays_off_only_when_the_posterior_is_diffuse(pair, grid, banks, truth):
    """Integrating the pose beats point-estimating it at low SNR, and ties as the posterior sharpens.

    The payoff is the reason the rung exists, and its collapse at high SNR is the reason a result
    measured at a comfortable SNR says nothing about a real dataset.
    """
    _, _, spectrum = pair
    bank_a, bank_b = banks

    def payoff(amplitude):
        from_a, from_b = pools(np.random.default_rng(13), truth, spectrum, amplitude)

        def ratios(images):
            return pose_log_ratio(
                images, bank_a, bank_b, spectrum, amplitude=amplitude, pose_log_weights=grid.log_weights
            )

        out_a, out_b = ratios(from_a), ratios(from_b)
        integrated = float(_information(out_a.marginal, out_b.marginal))
        point = float(_information(out_a.best_match, out_b.best_match))
        return integrated / point

    assert payoff(0.002) > 1.15
    assert payoff(0.016) < 1.01


def _reference_shift_log_ratio(images, bank_a, bank_b, spectrum, amplitude, pose_log_weights, shifts):
    """The joint (pose, shift) log-ratio written the slow, obvious way, in numpy.

    Independent of the implementation where it counts: it applies each phase ramp to the templates and
    accumulates one score at a time, where `pose_log_ratio` gets every shift from one inverse
    transform of the whitened cross-spectrum. Two routes to one number check the transform's
    normalization, its wrap-around indexing and the sign of its ramp at once.

    `whitening_weight` is shared rather than reimplemented, because the whitening convention -- DC
    zeroed, since particles are mean-subtracted -- is a decision the whole ladder makes together and
    not the thing under test here.
    """
    images = np.asarray(images)
    inverse = np.asarray(whitening_weight(spectrum))
    ramps = np.asarray(shifts.ramps())
    log_pose = np.asarray(pose_log_weights)
    log_pose = log_pose - np.log(np.sum(np.exp(log_pose)))
    log_shift = np.asarray(shifts.log_weights)

    marginal, best_match = [], []
    for image in images:
        free_energy, peak = [], []
        for bank in (bank_a, bank_b):
            models = amplitude * np.asarray(bank)
            scores = np.empty((len(models), len(ramps)))
            for c, model in enumerate(models):
                power = float(np.sum(np.abs(model) ** 2 * inverse))
                for t, ramp in enumerate(ramps):
                    cross = float(np.sum(np.real(image * np.conj(model * ramp)) * inverse))
                    scores[c, t] = log_pose[c] + log_shift[t] + cross - 0.5 * power
            free_energy.append(float(np.log(np.sum(np.exp(scores - scores.max()))) + scores.max()))
            peak.append(float(scores.max()))
        marginal.append(free_energy[0] - free_energy[1])
        best_match.append(peak[0] - peak[1])
    return np.asarray(marginal), np.asarray(best_match)


def test_the_transform_route_matches_the_obvious_one_shift_at_a_time(pair, grid, banks):
    """The transform gives every integer shift at once; this checks it against applying each ramp.

    A small case, because the reference is a quadruple loop. It is the test that would catch a wrong
    transform normalization, the wrong sign on the ramp, or negative shifts not wrapping -- none of
    which the information-level tests below could distinguish from a slightly worse rung.
    """
    _, _, spectrum = pair
    bank_a, bank_b = banks[0][:5], banks[1][:5]
    shifts = shift_grid(BOX_SIZE, 2.0, sigma=1.5)
    amplitude = 0.004

    rng = np.random.default_rng(21)
    images = jnp.asarray(
        (rng.normal(size=(3, BOX_SIZE, BOX_SIZE)) + 1j * rng.normal(size=(3, BOX_SIZE, BOX_SIZE))).astype(np.complex64)
    )
    pose_log_weights = grid.log_weights[:5]

    measured = pose_log_ratio(
        images, bank_a, bank_b, spectrum, amplitude=amplitude, pose_log_weights=pose_log_weights, shifts=shifts
    )
    expected_marginal, expected_best = _reference_shift_log_ratio(
        images, bank_a, bank_b, spectrum, amplitude, pose_log_weights, shifts
    )

    np.testing.assert_allclose(np.asarray(measured.marginal), expected_marginal, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(np.asarray(measured.best_match), expected_best, rtol=2e-3, atol=2e-3)


def test_a_sub_pixel_shift_grid_is_refused_rather_than_rounded(pair, grid, banks):
    """The transform reaches the pixel lattice and nothing between it, so a finer grid is an error.

    Rounding it silently would return a coarser quadrature than the caller asked for and read as a
    slightly worse rung.

    Checked on the SHIFTS, so a hand-built grid is caught too. A field saying what the spacing is
    would default to something valid on any grid the caller assembled itself, which is the one case
    where the check has to work.
    """
    _, _, spectrum = pair
    bank_a, bank_b = banks[0][:3], banks[1][:3]
    images = jnp.zeros((2, BOX_SIZE, BOX_SIZE), dtype=jnp.complex64)

    def score(shifts):
        return pose_log_ratio(images, bank_a, bank_b, spectrum, pose_log_weights=grid.log_weights[:3], shifts=shifts)

    with pytest.raises(ValueError, match="whole number of pixels"):
        score(shift_grid(BOX_SIZE, 2.0, step=0.5))
    hand_built = ShiftGrid(
        shifts=jnp.asarray([[0.0, 0.0], [0.4, -0.6]]),
        log_weights=jnp.full(2, -jnp.log(2.0)),
        box_size=BOX_SIZE,
    )
    with pytest.raises(ValueError, match="whole number of pixels"):
        score(hand_built)


def test_a_coarse_integer_lattice_is_scored_rather_than_refused(pair, grid, banks):
    """A step of two is every bit as much a pixel lattice as a step of one, and indexes exactly.

    The guard is about whether the shifts are whole pixels, not about which spacing they came from,
    so a caller who wants a wide window cheaply is not forced onto the fine grid to get it.
    """
    _, _, spectrum = pair
    bank_a, bank_b = banks[0][:3], banks[1][:3]
    images = jnp.zeros((2, BOX_SIZE, BOX_SIZE), dtype=jnp.complex64)

    coarse = shift_grid(BOX_SIZE, 4.0, step=2.0)
    assert coarse.shifts.shape[0] == 25
    measured = pose_log_ratio(images, bank_a, bank_b, spectrum, pose_log_weights=grid.log_weights[:3], shifts=coarse)
    assert bool(jnp.all(jnp.isfinite(measured.marginal)))


def test_a_shift_grid_built_on_another_box_is_refused(pair, grid, banks):
    """The scan caught this by failing to broadcast its ramps; indexing a lattice cannot, so the
    box agreement is checked outright. A mismatched grid otherwise aliases into a different
    quadrature and returns a number."""
    _, _, spectrum = pair
    bank_a, bank_b = banks[0][:3], banks[1][:3]
    images = jnp.zeros((2, BOX_SIZE, BOX_SIZE), dtype=jnp.complex64)

    with pytest.raises(ValueError, match="built on box"):
        pose_log_ratio(
            images,
            bank_a,
            bank_b,
            spectrum,
            pose_log_weights=grid.log_weights[:3],
            shifts=shift_grid(BOX_SIZE * 2, 2.0),
        )


def test_the_shift_latent_recovers_off_center_particles(pair, grid, banks, truth):
    """Uncentered particles collapse the rung, and marginalizing the shift brings it back.

    Also the honest price: the extra latent costs information even on particles that are centered. The
    three claims are ratios of one pool's information to another's, and hold across pool sizes; what
    may NOT be cut is the BANK, which below 96 nodes stops representing the off-grid truth poses.
    """
    _, _, spectrum = pair
    bank_a, bank_b = banks
    amplitude = 0.004
    shifts = shift_grid(BOX_SIZE, 2.0)

    rng = np.random.default_rng(14)
    offsets = rng.integers(-2, 3, (2, N_PARTICLES))
    # through `phase_ramp`, so a flipped convention breaks this test instead of canceling with itself
    ramp = phase_ramp(jnp.asarray(offsets.T, float), BOX_SIZE).astype(jnp.complex64)

    def information(ramp_used, shift_grid_used):
        from_a, from_b = pools(np.random.default_rng(15), truth, spectrum, amplitude, ramp=ramp_used)

        def ratios(images):
            return pose_log_ratio(
                images,
                bank_a,
                bank_b,
                spectrum,
                amplitude=amplitude,
                pose_log_weights=grid.log_weights,
                shifts=shift_grid_used,
            ).marginal

        return float(_information(ratios(from_a), ratios(from_b)))

    centered = information(1.0, None)
    uncorrected = information(ramp, None)
    corrected = information(ramp, shifts)
    cost_of_the_latent = information(1.0, shifts)

    assert uncorrected < 0.2 * centered
    assert corrected > 5.0 * uncorrected
    assert cost_of_the_latent < centered


def test_the_oracle_needs_the_shift_latent_too(pair, truth):
    """Knowing the POSE does not center the particle, so the oracle needs the shift latent too.

    Without it `--known-pose` reads a near-zero coherence on a pool where the bank route, given a
    shift window, reads a strongly positive value on the same particles.
    """
    _, _, spectrum = pair
    amplitude = 0.004
    shifts = shift_grid(BOX_SIZE, 2.0)

    rng = np.random.default_rng(21)
    offsets = rng.integers(-2, 3, (2, N_PARTICLES))
    ramp = phase_ramp(jnp.asarray(offsets.T, float), BOX_SIZE).astype(jnp.complex64)
    from_a, from_b = pools(np.random.default_rng(22), truth, spectrum, amplitude, ramp=ramp)

    def information(shift_grid_used):
        scored = [
            known_pose_log_ratio(images, truth[0], truth[1], spectrum, amplitude=amplitude, shifts=shift_grid_used)
            for images in (from_a, from_b)
        ]
        return float(_information(*scored))

    assert information(shifts) > 5.0 * information(None)


def test_a_single_zero_shift_is_the_centered_oracle(pair, truth):
    """The two code paths are separate arithmetic -- a sum over modes against an inverse transform
    read at one offset -- so they are pinned to each other rather than assumed to agree.

    The tolerance is a fraction of the spread and not machine epsilon: the shift route reaches the
    same number through a float32 inverse transform of the whole box, so a tighter bound would measure
    the FFT rather than the agreement.
    """
    _, _, spectrum = pair
    from_a, _ = pools(np.random.default_rng(23), truth, spectrum, 0.004)
    zero = ShiftGrid(shifts=jnp.zeros((1, 2)), log_weights=jnp.zeros(1), box_size=BOX_SIZE)

    centered = known_pose_log_ratio(from_a, truth[0], truth[1], spectrum, amplitude=0.004)
    scored = known_pose_log_ratio(from_a, truth[0], truth[1], spectrum, amplitude=0.004, shifts=zero)

    assert float(jnp.max(jnp.abs(scored - centered))) < 1e-4 * float(jnp.std(centered))


def test_the_oracle_refuses_a_sub_pixel_shift_grid(pair, truth):
    """The same lattice guard the bank route carries, now that both reach the offsets the same way."""
    _, _, spectrum = pair
    from_a, _ = pools(np.random.default_rng(24), truth, spectrum, 0.004)
    half = ShiftGrid(shifts=jnp.full((1, 2), 0.5), log_weights=jnp.zeros(1), box_size=BOX_SIZE)

    with pytest.raises(ValueError, match="whole number of pixels"):
        known_pose_log_ratio(from_a, truth[0], truth[1], spectrum, shifts=half)


def test_transfer_folds_onto_the_data(pair, grid, banks, truth):
    """Folding a group transfer onto the data equals applying it to the templates.

    The factorization is what makes a per-particle signed CTF affordable: `C` is real, so it never
    forces the shared pose bank to be rebuilt per particle.
    """
    _, _, spectrum = pair
    bank_a, bank_b = banks
    amplitude = 0.004

    frequency = np.fft.fftfreq(BOX_SIZE)
    radius_squared = frequency[:, None] ** 2 + frequency[None, :] ** 2
    defocus = np.linspace(8000.0, 20000.0, 4)
    transfer = jnp.asarray(
        np.sin(2 * np.pi * 0.5 * defocus[:, None, None] * 0.0197 * radius_squared).astype(np.float32)
    )
    group = np.random.default_rng(16).integers(0, 4, N_PARTICLES)

    images = amplitude * truth[0] * transfer[group] + numpy_colored_noise(np.random.default_rng(17), spectrum)
    folded = pose_log_ratio(
        images,
        bank_a,
        bank_b,
        spectrum,
        amplitude=amplitude,
        pose_log_weights=grid.log_weights,
        transfer=Transfer(values=transfer, index=jnp.asarray(group)),
    ).marginal
    applied = pose_log_ratio(
        images[group == 0],
        bank_a * transfer[0],
        bank_b * transfer[0],
        spectrum,
        amplitude=amplitude,
        pose_log_weights=grid.log_weights,
    ).marginal

    reference = np.asarray(folded)[group == 0]
    assert np.abs(reference - np.asarray(applied)).max() < 1e-3 * np.abs(reference).mean()


def test_identical_states_carry_no_information(pair, grid, banks, truth):
    """Comparing a state against itself gives Lambda exactly zero, at every pose treatment."""
    _, _, spectrum = pair
    bank_a, _ = banks
    images = 0.004 * truth[0] + numpy_colored_noise(np.random.default_rng(18), spectrum)

    ratios = pose_log_ratio(images, bank_a, bank_a, spectrum, amplitude=0.004, pose_log_weights=grid.log_weights)
    assert float(jnp.max(jnp.abs(ratios.marginal))) == 0.0
    assert float(jnp.max(jnp.abs(ratios.best_match))) == 0.0
    assert float(jnp.max(jnp.abs(known_pose_log_ratio(images, truth[0], truth[0], spectrum)))) == 0.0


def test_a_wrong_box_transfer_is_rejected_rather_than_reinterpreted(pair, grid, banks, truth):
    """`values.reshape(-1, n_pixels)` accepts any total size, so an unchecked wrong box would not
    fail: it would repack the transfer into bogus rows and return finite, plausible log-ratios."""
    _, _, spectrum = pair
    bank_a, bank_b = banks
    images = truth[0] + numpy_colored_noise(np.random.default_rng(18), spectrum)
    half = BOX_SIZE // 2
    with pytest.raises(ValueError, match="transfer shape"):
        pose_log_ratio(
            images,
            bank_a,
            bank_b,
            spectrum,
            pose_log_weights=grid.log_weights,
            transfer=Transfer(
                values=jnp.ones((4, half, half)),
                index=jnp.asarray(np.random.default_rng(19).integers(0, 4, N_PARTICLES)),
            ),
        )


def test_one_based_group_ids_are_rejected_rather_than_clamped(pair, grid, banks, truth):
    """The rung gathers the same way the fit does, so it needs the same guard: a clamped gather
    returns finite log-ratios computed against the wrong group's transfer."""
    _, _, spectrum = pair
    bank_a, bank_b = banks
    images = truth[0] + numpy_colored_noise(np.random.default_rng(20), spectrum)
    transfer = Transfer(
        values=jnp.ones((4, BOX_SIZE, BOX_SIZE)),
        index=jnp.asarray(np.random.default_rng(21).integers(1, 5, N_PARTICLES)),
    )
    with pytest.raises(RuntimeError, match="transfer.index out of range"):
        pose_log_ratio(images, bank_a, bank_b, spectrum, pose_log_weights=grid.log_weights, transfer=transfer)
