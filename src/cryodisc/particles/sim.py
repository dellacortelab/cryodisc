"""Particle stacks with a known answer.

The only module that makes a stack of noisy particles: the spectrum, contrast, occupancy, pose,
shift and defocus are all CHOSEN here and returned beside the images, which lets a fit be checked
against truth rather than against another fit. Images come out as `fft2`, unshifted, and NOT
mean-subtracted; DC is a convention each consumer applies for itself. The state is selected on the
volume, so a stack costs one projection per particle rather than one per state per particle.
"""

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Int, PRNGKeyArray

from cryodisc.forward.cryojax_model import CryoJAXModel
from cryodisc.forward.optics import CTFParams
from cryodisc.grids.fourier import phase_ramp
from cryodisc.microscope import CollectionPlan


class ParticleStack(eqx.Module):
    """Simulated particles and everything that was true about them.

    Args:
        images: fft2 of the stack, unshifted, not mean-subtracted, (n, y, x).
        state: Which state each particle is, (n,), each in `[0, n_states)`.
        angles: Euler angles in degrees, cryoJAX intrinsic zyz, (n, 3). Feeds `forward_batch` directly.
        shift: Translation applied to each particle as (y, x) in pixels, (n, 2).
        defocus: Underfocus of each particle in A, (n,), continuous -- what a `microscope.Transfer` is
            built from after `em`'s binning.
        noise_spectrum: The sigma^2(k) the noise was drawn from, (y, x).
        contrast: The `a` the templates were scaled by. Only `a^2 / sigma^2` is observable, so this
            and `noise_spectrum` are one SNR statement made in two places.
    """

    images: Complex[Array, "n y x"]
    state: Int[Array, " n"]
    angles: Float[Array, "n 3"]
    shift: Float[Array, "n 2"]
    defocus: Float[Array, " n"]
    noise_spectrum: Float[Array, "y x"]
    contrast: Float[Array, ""]

    def fraction(self, state: int = 0) -> Float[Array, ""]:
        """The occupancy actually drawn, which is not the occupancy asked for.

        Args:
            state: Which state to count.

        Returns:
            Its share of the stack. `rungs.achieved.fraction_bias` wants this rather than the nominal one.
        """
        return jnp.mean(self.state == state)


def uniform_angles(key: PRNGKeyArray, n_particles: int) -> Float[Array, "n 3"]:
    """Euler angles drawn from the Haar measure on SO(3).

    Uniform in `phi` and `psi` but in `cos(theta)`, not `theta`: sampling the polar angle uniformly
    piles particles onto the poles of the viewing sphere, which is the same Jacobian a quadrature rule
    absorbs into the bank's own nodes.

    Args:
        key: PRNG key.
        n_particles: Number of poses to draw.

    Returns:
        (n, 3) angles in degrees, cryoJAX intrinsic zyz.
    """
    phi_key, theta_key, psi_key = jax.random.split(key, 3)
    phi = jax.random.uniform(phi_key, (n_particles,), maxval=360.0)
    theta = jnp.degrees(jnp.arccos(jax.random.uniform(theta_key, (n_particles,), minval=-1.0, maxval=1.0)))
    psi = jax.random.uniform(psi_key, (n_particles,), maxval=360.0)
    return jnp.stack([phi, theta, psi], axis=-1)


def colored_noise(key: PRNGKeyArray, noise_spectrum: Float[Array, "y x"], n_particles: int) -> Complex[Array, "n y x"]:
    """Gaussian noise whose fft has per-mode power `noise_spectrum`.

    Drawn white in real space and colored in Fourier, so it is real-valued by construction rather
    than by a symmetrization that would halve the variance of the modes it pairs up.

    Args:
        key: PRNG key.
        noise_spectrum: sigma^2(k) on the unshifted fft grid, (y, x). Must be EVEN in k -- true of
            every radial spectrum here, and what keeps `fft2` of real noise Hermitian.
        n_particles: Number of realizations.

    Returns:
        (n, y, x) on the unshifted fft grid, with `E |noise(k)|^2 = sigma^2(k)`.
    """
    box_size = noise_spectrum.shape[-1]
    white = jax.random.normal(key, (n_particles, box_size, box_size))
    # sqrt(sigma^2 / n_pixels) because fft2 of unit-variance real noise carries power n_pixels
    return jnp.fft.fft2(white) * jnp.sqrt(noise_spectrum / noise_spectrum.size)


def simulate_stack(
    models: Sequence[CryoJAXModel],
    *,
    key: PRNGKeyArray,
    n_particles: int,
    noise_spectrum: Float[Array, "y x"],
    contrast: float = 1.0,
    occupancy: Sequence[float] | None = None,
    plan: CollectionPlan | None,
    angles: Float[Array, "n 3"] | None = None,
    shift_sigma: float = 0.0,
) -> ParticleStack:
    """Render a stack of noisy particles from two or more conformational states.

    Args:
        models: One forward model per state, as `ConformerPair.models` returns, sharing an image
            config and atom set.
        key: PRNG key, split internally over state, pose, shift, defocus and noise.
        n_particles: Stack size.
        noise_spectrum: sigma^2(k) on the unshifted fft grid, (y, x). Color it -- a white spectrum
            over-weights low frequencies against any real dataset.
        contrast: `a`, scaling templates against noise. Aim at real data's per-mode SNR ~ 0.1; a
            near-noiseless stack is HARDER for `em` (its float32 sigma^2 M-step degrades at high SNR).
        occupancy: Share of the stack in each state, normalized internally. `None` is uniform.
        angles: Poses to render at, (n, 3) degrees, instead of the Haar measure -- what puts particles
            ON a pose bank, which `em` and `rungs.pose_log_ratio` cannot recover between.
        plan: Collection supplying the per-particle defocus and optics. Must be the plan the models
            were built from (checked). Undefaulted because `None` renders the bare potential, far
            larger than contrast with nothing to pair a noise level with.
        shift_sigma: Width in pixels of the isotropic Gaussian the translation is drawn from. Zero
            gives centered particles.

    Returns:
        The stack and its ground truth.
    """
    if n_particles < 1:
        raise ValueError(f"need at least one particle, got {n_particles}")
    box_size = noise_spectrum.shape[-1]
    if noise_spectrum.shape != (box_size, box_size):
        raise ValueError(f"noise_spectrum must be square, got {noise_spectrum.shape}")
    if angles is not None and angles.shape != (n_particles, 3):
        raise ValueError(f"angles must be ({n_particles}, 3), got {angles.shape}")
    config = models[0].image_config
    if box_size != config.shape[-1]:
        raise ValueError(f"noise_spectrum is built on box {box_size}, the models render at {config.shape[-1]}")
    if plan is not None:
        # voltage as well as pixel size: CTFParams carries no voltage, so a rendered CTF and
        # `resolution` can silently take their wavelength from different sources
        for name, planned, built in (
            ("pixel_size", plan.pixel_size, float(config.pixel_size)),
            ("voltage_in_kv", plan.voltage_in_kv, float(config.voltage_in_kilovolts)),
        ):
            # RELATIVE tolerance: an absolute one would refuse a pixel size not exactly representable
            # in float32, failing on values that only round-trip approximately through the image config
            if abs(planned - built) > 1e-6 * max(abs(planned), 1.0):
                raise ValueError(
                    f"plan.{name} is {planned} but the models were built at {built}. "
                    "`ConformerPair.models(plan, box_size)` builds them from the plan, which is the "
                    "way to keep the rendered stack and the radial engine on one microscope."
                )
    return _simulate(
        models,
        key=key,
        n_particles=n_particles,
        noise_spectrum=noise_spectrum,
        contrast=jnp.asarray(contrast, float),
        occupancy=jnp.ones(len(models)) if occupancy is None else jnp.asarray(occupancy, float),
        angles=angles,
        plan=plan,
        shift_sigma=jnp.asarray(shift_sigma, float),
    )


@eqx.filter_jit
def _simulate(
    models: Sequence[CryoJAXModel],
    *,
    key: PRNGKeyArray,
    n_particles: int,
    noise_spectrum: Float[Array, "y x"],
    contrast: Float[Array, ""],
    occupancy: Float[Array, " s"],
    angles: Float[Array, "n 3"] | None,
    plan: CollectionPlan | None,
    shift_sigma: Float[Array, ""],
) -> ParticleStack:
    """`simulate_stack` once its arguments are checked.

    The swept knobs arrive as arrays because `filter_jit` makes every non-array leaf static, so a
    float would recompile the render per value.
    """
    box_size = noise_spectrum.shape[-1]
    state_key, pose_key, shift_key, defocus_key, noise_key = jax.random.split(key, 5)
    state = jax.random.choice(state_key, len(models), (n_particles,), p=occupancy / occupancy.sum())
    if angles is None:
        angles = uniform_angles(pose_key, n_particles)
    shift = shift_sigma * jax.random.normal(shift_key, (n_particles, 2))

    low, high = (0.0, 0.0) if plan is None else plan.defocus_range
    defocus = jax.random.uniform(defocus_key, (n_particles,), minval=low, maxval=high)

    clean = jnp.fft.fft2(_render_states(models, state, angles, defocus, plan))
    # through phase_ramp, which owns the ramp sign: a second copy would let a stack be simulated at one
    # sign and marginalized at the other. Not ShiftGrid -- these are continuous per-particle draws
    ramps = phase_ramp(shift, box_size)

    images = contrast * clean * ramps + colored_noise(noise_key, noise_spectrum, n_particles)
    return ParticleStack(
        images=images,
        state=state,
        angles=angles,
        shift=shift,
        defocus=defocus,
        noise_spectrum=noise_spectrum,
        contrast=contrast,
    )


_RENDER_CHUNK = 100
"""Particles per render batch. cryoJAX builds a `(batch, box, n_atoms, 5)` temporary, so chunking
bounds peak memory to the chunk rather than the whole stack."""


def _render_states(
    models: Sequence[CryoJAXModel],
    state: Int[Array, " n"],
    angles: Float[Array, "n 3"],
    defocus: Float[Array, " n"],
    plan: CollectionPlan | None,
) -> Float[Array, "n y x"]:
    """Each particle rendered from its own state's volume, at one projection per particle.

    Gathering the volume rather than selecting a rendered image keeps the cost flat in `n_states`
    instead of linear. `jnp.stack` is also what enforces the shared atom set.
    """
    banked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *[model.volume for model in models])

    def one(particle: tuple[Int[Array, ""], Float[Array, " 3"], Float[Array, ""]]) -> Float[Array, "y x"]:
        index, row, dz = particle
        volume = jax.tree.map(lambda leaf: leaf[index], banked)
        model = eqx.tree_at(lambda m: m.volume, models[0], volume)
        return model._render(row[0], row[1], row[2], None if plan is None else CTFParams.from_plan(plan, dz))

    # lax.map rather than a bare vmap: it vmaps within a batch and scans across them, bounding the
    # temporary to the chunk, and handles a ragged tail itself
    return jax.lax.map(one, (state, angles, defocus), batch_size=_RENDER_CHUNK)
