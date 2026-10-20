"""What the predicted sigma^2(k) has to be for `contrast = 1` to be the right pairing.

The level is a closed form at DC, so it is pinned against the arithmetic rather than against itself.
The rest is the closed loop: a stack simulated at the predicted spectrum, fitted by `particles.em`, and the
spectrum coming back -- with nothing anywhere in the chain chosen by hand.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cryodisc.forward.optics import CTFParams
from cryodisc.grids.fourier import frequency_grid
from cryodisc.grids.poses import pose_bank
from cryodisc.microscope import CollectionPlan
from cryodisc.particles.em import fit_noise_model
from cryodisc.particles.sim import simulate_stack
from cryodisc.resolution import dqe, noise_spectrum

BOX_SIZE = 64
PIXEL_SIZE = 1.5
PLAN = CollectionPlan(dose=40.0, pixel_size=PIXEL_SIZE)
DEFOCUS = 12000.0
# one defocus, so the templates a fit scores against carry the same CTF the particles do
FIXED = CollectionPlan(dose=40.0, pixel_size=PIXEL_SIZE, defocus_range=(DEFOCUS, DEFOCUS))
OPTICS = CTFParams.from_plan(FIXED, DEFOCUS)


@pytest.fixture(scope="module")
def models(bent_pair_factory):
    """Two conformers of a compact blob, the second bent in half."""
    return bent_pair_factory(n_atoms=2400, spread=9.0, bend=3.0).models(FIXED, box_size=BOX_SIZE)


def test_the_shot_noise_level_is_the_closed_form_at_dc():
    """At k = 0 both envelopes are one, so only the exposure and the DQE plateau are left.

    The 4 is the square of the 2 in cryoJAX's `|psi|^2 = 1 + 2C`, and the box^2 is the unnormalized
    fft2 the package uses throughout. Neither is fitted, which is why `contrast = 1` is the pairing.
    """
    expected = BOX_SIZE**2 / (4.0 * PLAN.dose * PLAN.pixel_size**2 * float(dqe(jnp.zeros(()), nyquist=PLAN.nyquist)))
    assert float(noise_spectrum(PLAN, BOX_SIZE)[0, 0]) == pytest.approx(expected, rel=1e-6)


def test_the_level_tracks_the_exposure_and_the_pixel():
    """Twice the dose halves the noise; a bigger pixel collects more electrons into each one."""
    base = float(noise_spectrum(PLAN, BOX_SIZE)[0, 0])
    doubled = CollectionPlan(dose=80.0, pixel_size=PIXEL_SIZE)
    assert float(noise_spectrum(doubled, BOX_SIZE)[0, 0]) == pytest.approx(base / 2.0, rel=1e-6)
    coarse = CollectionPlan(dose=40.0, pixel_size=3.0)
    # 4x the area per pixel, but the DQE is read at a Nyquist half as high, so only the ratio is exact
    assert float(noise_spectrum(coarse, BOX_SIZE)[0, 0]) < base / 3.0


def test_the_noise_rises_with_frequency_because_the_signal_is_damped_there():
    spectrum = np.asarray(noise_spectrum(PLAN, BOX_SIZE))
    k = np.asarray(frequency_grid(BOX_SIZE, PIXEL_SIZE))
    order = np.argsort(k.ravel())
    radial = spectrum.ravel()[order]
    assert radial[-1] > 20.0 * radial[0]
    assert np.all(np.diff(radial) >= -1e-6)


def test_flexibility_inflates_the_noise_only_where_it_damps_the_signal():
    """E_conf is one at k = 0 and falls as exp(-B k^2 / 4), so the DC level cannot move."""
    rigid = noise_spectrum(PLAN, BOX_SIZE)
    floppy = noise_spectrum(PLAN, BOX_SIZE, rmsf=1.5)
    assert float(floppy[0, 0]) == pytest.approx(float(rigid[0, 0]), rel=1e-6)
    assert bool(jnp.all(floppy >= rigid - 1e-6))
    assert float(floppy[BOX_SIZE // 2, 0] / rigid[BOX_SIZE // 2, 0]) > 10.0


def test_a_stack_simulated_at_the_predicted_noise_fits_back_to_it(models):
    """The loop with nothing chosen by hand: a plan in, and `particles.em` returns the plan's own numbers.

    `contrast = 1` is the claim being tested, not a convenience. cryoJAX renders contrast against
    the incident beam and the spectrum is in those same units, so a fitted scale anywhere but one
    would mean the two disagree about what an electron is. The outer-band ratio is a small systematic
    inside the tolerance rather than a sampling estimate a bigger pool would tighten.
    """
    grid = pose_bank(12)
    n_particles = 600
    spectrum = noise_spectrum(PLAN, BOX_SIZE, rmsf=0.5)

    key, pose_key = jax.random.split(jax.random.PRNGKey(0))
    index = jax.random.randint(pose_key, (n_particles,), 0, grid.angles.shape[0])
    stack = simulate_stack(
        models,
        key=key,
        n_particles=n_particles,
        noise_spectrum=spectrum,
        contrast=1.0,
        occupancy=(0.6, 0.4),
        angles=grid.angles[index],
        plan=FIXED,
    )

    templates = jnp.stack([jnp.fft.fft2(model.forward_batch(grid.angles, OPTICS)) for model in models])
    fit = fit_noise_model(stack.images, templates, pose_log_weights=grid.log_weights, n_iterations=200)

    assert float(fit.contrast) == pytest.approx(1.0, rel=0.15)
    outer = np.asarray(frequency_grid(BOX_SIZE, PIXEL_SIZE)) > 0.1
    ratio = float(jnp.mean(fit.noise_spectrum[outer] / spectrum[outer]))
    assert ratio == pytest.approx(1.0, rel=0.05)


def test_the_particle_images_look_like_noise_at_a_real_exposure(models):
    """The regime check: one particle is far below the noise, which is why cryo-EM needs a stack."""
    spectrum = noise_spectrum(PLAN, BOX_SIZE)
    stack = simulate_stack(models, key=jax.random.PRNGKey(1), n_particles=8, noise_spectrum=spectrum, plan=FIXED)
    signal = jnp.abs(jnp.fft.fft2(models[0].forward_batch(stack.angles, OPTICS))) ** 2
    per_mode = float(jnp.mean((signal / spectrum).at[:, 0, 0].set(0.0)))
    assert 1e-3 < per_mode < 1.0
