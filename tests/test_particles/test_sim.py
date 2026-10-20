"""What a simulated stack has to be before a fit against it means anything.

The sampling and the noise are checked against their own distributions, and then the stack is run
through `particles.em`, which is the only test in the package that connects two real conformers to a fitted
noise model. Everything `particles.em` returns was chosen here, so the assertion is against the truth rather
than against a second fit.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cryodisc.grids.fourier import frequency_grid
from cryodisc.grids.poses import ShiftGrid, pose_bank
from cryodisc.microscope import CollectionPlan
from cryodisc.particles.em import fit_noise_model
from cryodisc.particles.sim import ParticleStack, colored_noise, simulate_stack, uniform_angles

BOX_SIZE = 48
PIXEL_SIZE = 2.0
PLAN = CollectionPlan(dose=40.0, pixel_size=PIXEL_SIZE, defocus_range=(8000.0, 20000.0))


@pytest.fixture(scope="module")
def models(bent_pair_factory):
    """Two conformers of one molecule, the second bent in its second half."""
    return bent_pair_factory(n_atoms=240, spread=14.0, bend=5.0).models(PLAN, box_size=BOX_SIZE)


def _colored(level):
    """A spectrum that rises toward low k, as a real one does."""
    return level * (1.0 + 4.0 * jnp.exp(-frequency_grid(BOX_SIZE) / 0.08))


def test_poses_are_uniform_on_the_sphere_not_on_the_polar_angle():
    """Sampling theta uniformly would pile particles onto the poles. cos(theta) is the Haar measure."""
    angles = np.asarray(uniform_angles(jax.random.PRNGKey(0), 40_000))
    cos_theta = np.cos(np.radians(angles[:, 1]))
    assert cos_theta.mean() == pytest.approx(0.0, abs=0.02)
    # uniform in cos(theta) has variance 1/3; uniform in theta would give ~0.19
    assert cos_theta.var() == pytest.approx(1.0 / 3.0, abs=0.02)
    assert np.histogram(cos_theta, bins=8, range=(-1.0, 1.0))[0].std() / 5000.0 < 0.03


def test_the_noise_has_the_spectrum_it_was_asked_for_and_is_real():
    spectrum = _colored(1.0)
    noise = colored_noise(jax.random.PRNGKey(1), spectrum, 3000)
    measured = jnp.mean(jnp.abs(noise) ** 2, axis=0)
    np.testing.assert_allclose(np.asarray(measured), np.asarray(spectrum), rtol=0.12)
    # colored in Fourier from real white noise, so the inverse transform has no imaginary part
    assert float(jnp.max(jnp.abs(jnp.imag(jnp.fft.ifft2(noise[:20]))))) < 1e-6


def test_the_stack_carries_its_ground_truth(models):
    stack = simulate_stack(
        models,
        key=jax.random.PRNGKey(2),
        n_particles=64,
        noise_spectrum=_colored(600.0),
        occupancy=(0.75, 0.25),
        plan=PLAN,
        shift_sigma=1.5,
    )
    assert isinstance(stack, ParticleStack)
    assert stack.images.shape == (64, BOX_SIZE, BOX_SIZE)
    assert stack.angles.shape == (64, 3)
    assert set(np.unique(np.asarray(stack.state))) <= {0, 1}
    assert 0.55 < float(stack.fraction(0)) < 0.95
    low, high = PLAN.defocus_range
    assert bool(jnp.all((stack.defocus >= low) & (stack.defocus <= high)))
    # not mean-subtracted: DC is each consumer's own convention, and dropping it here against
    # templates that keep it would read as a template scale of zero
    assert float(jnp.min(jnp.abs(stack.images[:, 0, 0]))) > 0.0


def test_one_key_is_one_dataset(models):
    def stack(seed):
        return simulate_stack(
            models, key=jax.random.PRNGKey(seed), n_particles=8, noise_spectrum=_colored(600.0), plan=PLAN
        )

    np.testing.assert_array_equal(np.asarray(stack(3).images), np.asarray(stack(3).images))
    assert not np.allclose(np.asarray(stack(3).images), np.asarray(stack(4).images))


def test_the_shift_uses_the_one_ramp_convention(models):
    """Simulating at one sign and marginalizing at the other would agree only on centered particles."""
    angles = jnp.zeros((1, 3))
    common = {
        "models": models[:1],
        "n_particles": 1,
        "noise_spectrum": jnp.zeros((BOX_SIZE, BOX_SIZE)),
        "angles": angles,
        "plan": None,
    }
    centered = simulate_stack(key=jax.random.PRNGKey(5), shift_sigma=0.0, **common)
    shifted = simulate_stack(key=jax.random.PRNGKey(5), shift_sigma=3.0, **common)
    ramp = ShiftGrid(shifts=shifted.shift, log_weights=jnp.zeros(1), box_size=BOX_SIZE).ramps()
    undone = shifted.images / ramp
    np.testing.assert_allclose(np.asarray(undone), np.asarray(centered.images), atol=1e-3)


def test_em_recovers_the_spectrum_amplitude_and_occupancy_it_was_given(models):
    """The loop closing: two deposited-style conformers in, and the numbers chosen here come back.

    Poses are drawn from the bank `particles.em` marginalizes over, since a stack sitting between the nodes is
    a pose-search failure rather than a fitting one, and that is a different rung's question.
    """
    grid = pose_bank(12)
    n_particles, contrast, occupancy = 480, 2.5, (0.7, 0.3)
    # a per-mode SNR near 0.13, which is the regime real data sits in and the regime the float32
    # sigma^2 M-step is stable in; a near-noiseless stack takes that subtraction to nan
    spectrum = _colored(600.0)

    key, pose_key = jax.random.split(jax.random.PRNGKey(6))
    index = jax.random.randint(pose_key, (n_particles,), 0, grid.angles.shape[0])
    stack = simulate_stack(
        models,
        key=key,
        n_particles=n_particles,
        noise_spectrum=spectrum,
        contrast=contrast,
        occupancy=occupancy,
        angles=grid.angles[index],
        # no optics on either side: this pins the fit against an arbitrary spectrum, where the
        # template units cancel. `test_resolution/test_noise.py` is the absolute, CTF-rendered case.
        plan=None,
    )

    templates = jnp.stack([jnp.fft.fft2(model.forward_batch(grid.angles)) for model in models])
    fit = fit_noise_model(stack.images, templates, pose_log_weights=grid.log_weights, n_iterations=200)

    assert float(fit.contrast) == pytest.approx(contrast, rel=0.05)
    assert float(fit.state_weights[0]) == pytest.approx(float(stack.fraction(0)), abs=0.03)
    # sigma^2 is shell-smoothed by the fit, so compare where the spectrum is flat rather than at DC
    outer = frequency_grid(BOX_SIZE) > 0.2
    ratio = float(jnp.mean(fit.noise_spectrum[outer] / spectrum[outer]))
    assert ratio == pytest.approx(1.0, rel=0.05)
