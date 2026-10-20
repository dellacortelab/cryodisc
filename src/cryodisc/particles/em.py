"""Expectation-maximization fit of the noise spectrum, template contrast, and state occupancies.

Particles are a Gaussian mixture over (state, pose, shift) components, all latent,

    Y_i(k) = a C_i(k) M_{s,p}(k) T_t(k) + noise,    noise ~ N(0, sigma^2(k)).

E-step a softmax, M-steps closed form, on the UNSHIFTED fft grid; the shift width is FITTED, sigma^2(k)
shell-smoothed and `a` scalar. In float32 the sigma^2 M-step has relative error ~ `2^-24 (1 + SNR(k)^2)`, fixed by `jax_enable_x64`.
"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Int

from cryodisc.grids.fourier import shell_indices, shell_sum
from cryodisc.grids.poses import ShiftGrid
from cryodisc.microscope import Transfer, fold_transfer


class NoiseModel(eqx.Module):
    """A converged fit.

    Args:
        noise_spectrum: sigma^2(k) on the unshifted fft grid, radially smoothed.
        contrast: Scalar template scale `a`. Where `amplitude_spectrum` is set this is its
            whitened-power-weighted mean, equal to `a` when the shells agree.
        amplitude_spectrum: `a(k)` on the unshifted fft grid, or `None` for a scalar fit. A rising
            `a(k)` is an envelope mismatch a scalar would hide.
        state_weights: Occupancy per state, summing to one.
        shift_sigma: Fitted width of the isotropic shift prior in pixels, zero when no shift grid was
            given. Below the grid `step` it reports the grid rather than the data.
        responsibilities: Posterior over (state, pose) per particle, shift marginalized out.
        log_likelihood: Observed-data log likelihood at the returned parameters.
        n_iterations: Iterations actually run.
        delta: Final mean |log| change in sigma^2, for comparison against the tolerance.
    """

    noise_spectrum: Float[Array, "y x"]
    contrast: Float[Array, ""]
    state_weights: Float[Array, " s"]
    shift_sigma: Float[Array, ""]
    responsibilities: Float[Array, "n s p"]
    log_likelihood: Float[Array, ""]
    n_iterations: Int[Array, ""]
    delta: Float[Array, ""]
    amplitude_spectrum: Float[Array, "y x"] | None = None


class _ShiftAxis(eqx.Module):
    """The two arrays the shift latent needs, which exist or do not exist together.

    One optional bundle rather than two `| None` locals, so a reader (and a checker) sees at each use
    site that a non-`None` shift axis is what makes the joint posterior non-`None`. No `log_weights`:
    the Gaussian width is fitted by EM, so the prior is `radius_squared` and the current `shift_sigma`.
    """

    ramp: Complex[Array, "t k"]
    radius_squared: Float[Array, " t"]


class _TransferPower(eqx.Module):
    """C(k)^2 per group, flat over pixels, plus each particle's group.

    The steps need C^2 but never C, since the term linear in C folded onto the data in `_prepare`.
    Never `None`: no transfer is one group of ones, which both steps reduce to the uniform case on
    their own, so neither carries a second code path that could drift from this one.
    """

    squared: Float[Array, "g k"]
    index: Int[Array, " n"]


class _Params(eqx.Module):
    """What EM updates. sigma^2 is flat over pixels here and reshaped only on the way out.

    `contrast` is scalar for a single fitted `a` and flat over pixels for a fitted `a(k)`. Every
    update below is written so the two differ only in that shape, since the alternative is a second
    arithmetic path for the shell-resolved case that could drift from the scalar one.
    """

    noise_spectrum: Float[Array, " k"]
    contrast: Float[Array, "..."]
    state_weights: Float[Array, " s"]
    shift_sigma: Float[Array, ""]


class _State(eqx.Module):
    """The `while_loop` carry: parameters plus what the stopping rule reads."""

    params: _Params
    delta: Float[Array, ""]
    iteration: Int[Array, ""]


class _Expectation(eqx.Module):
    """E-step output. `cross` and `power` are whitened inner products the M-step reuses.

    `power` carries no shift axis: `|T_t| = 1`, so the model power is the same at every candidate.
    """

    responsibilities: Float[Array, "n ct"]
    cross: Float[Array, "n ct"]
    power: Float[Array, "n c"]
    log_evidence: Float[Array, ""]


class _Problem(eqx.Module):
    """Everything derived once from the inputs and reused every iteration.

    Images and templates are flattened over pixels, so each step is a matmul over the pixel axis and
    the component axis stays intact. `latent_shape` is `(n_states, n_poses, n_shifts)`, the three
    axes the flat component index unpacks into.
    """

    model_real: Float[Array, "ct k"]
    model_imag: Float[Array, "ct k"]
    model_power: Float[Array, "c k"]
    weighted_real: Float[Array, "n k"]
    weighted_imag: Float[Array, "n k"]
    data_power: Float[Array, "n k"]
    data_power_total: Float[Array, " k"]
    transfer_power: _TransferPower
    log_pose: Float[Array, " p"]
    shift_radius_squared: Float[Array, " t"]
    shell_index: Int[Array, " k"]
    noise_floor: Float[Array, ""]
    n_shells: int
    latent_shape: tuple[int, int, int]

    @property
    def n_particles(self) -> int:
        return self.data_power.shape[0]

    def smooth_spectrum(self, spectrum: Float[Array, " k"]) -> Float[Array, " k"]:
        """Shell-average a candidate sigma^2 and floor it, keeping the flat pixel layout.

        Together, because one zero mode reaches every parameter through 1/sigma^2, so a caller who
        smoothed without flooring would take the whole fit to nan.
        """
        return _smooth_to_shells(spectrum, self.shell_index, self.n_shells, self.noise_floor)


def _running(state: "_State", *, n_iterations: int, tolerance: float) -> Array:
    """Whether the EM loop has another iteration to run, for both routes' `while_loop`.

    Args:
        state: The loop state.
        n_iterations: Hard iteration cap.
        tolerance: Stop when BOTH the mean |log| change in sigma^2 and the largest change in any
            state weight fall below this. The occupancy is watched too because it converges much more
            slowly than the spectrum; see `fit_noise_model`'s `step`.

    Returns:
        The scalar predicate `while_loop` wants.
    """
    return (state.iteration < n_iterations) & (state.delta > tolerance)


def _require_square(images: Complex[Array, "n y x"]) -> tuple[int, int]:
    """The box, refusing a non-square or odd-sized stack.

    Args:
        images: The particle stack.

    Returns:
        `(box_size, n_pixels)`.

    Raises:
        ValueError: If the images are not square. Every shell map, ramp and transfer here is built
            on one edge length, so a rectangular stack would be indexed as if it were square.
    """
    _, box_y, box_x = images.shape
    if box_y != box_x:
        raise ValueError(f"images must be square, got {(box_y, box_x)}")
    if box_y % 2 != 0:
        raise ValueError(f"box size must be even so DC is its own shell (n_shells = box // 2); got {box_y}")
    return box_y, box_y * box_x


def _smooth_to_shells(
    spectrum: Float[Array, " k"], shell_index: Array, n_shells: int, noise_floor: Float[Array, ""]
) -> Float[Array, " k"]:
    """Shell-average a candidate sigma^2 and floor it, keeping the flat pixel layout.

    Together, because one zero mode reaches every parameter through `1/sigma^2`, so a caller who
    smoothed without flooring would take the whole fit to nan.

    Gathered back through the SAME index, out-of-band bin and all, which is why `shell_sum` keeps
    that bin: the corners are a shell of their own here rather than being discarded.

    Args:
        spectrum: The candidate, flattened.
        shell_index: Shell each mode belongs to.
        n_shells: How many shells.
        noise_floor: Smallest value any mode may take.

    Returns:
        The smoothed, floored spectrum, flat.
    """
    totals = shell_sum(spectrum, shell_index, n_shells)
    counts = shell_sum(jnp.ones_like(spectrum), shell_index, n_shells)
    return jnp.maximum((totals / jnp.maximum(counts, 1.0))[shell_index], noise_floor)


def _log_normalizer(n_particles: int, spectrum: Float[Array, " k"]) -> Float[Array, ""]:
    """The Gaussian normalizing constant both routes subtract off their evidence.

    Args:
        n_particles: Pool size.
        spectrum: sigma^2 per mode, flat.

    Returns:
        The constant.
    """
    return 0.5 * n_particles * jnp.log(2.0 * jnp.pi * spectrum).sum()


def _prepare(
    images: Complex[Array, "n y x"],
    templates: Complex[Array, "s p y x"],
    transfer: Transfer | None,
    shifts: ShiftGrid | None,
    pose_log_weights: Float[Array, " p"] | None,
) -> _Problem:
    """Validate the shapes, flatten over pixels, and fold the transfer and the ramps onto the data."""
    n_particles = images.shape[0]
    n_states, n_poses = templates.shape[:2]
    box_y, n_pixels = _require_square(images)
    box_x = box_y
    if templates.shape[2:] != (box_y, box_x):
        raise ValueError(f"template shape {templates.shape[2:]} does not match images {(box_y, box_x)}")
    n_shells = box_y // 2

    if shifts is None:
        # the single candidate at the origin, so neither step needs a shift-free branch
        ramps = jnp.ones((1, n_pixels), dtype=templates.dtype)
        shift_radius_squared = jnp.zeros(1)
    else:
        if shifts.box_size != box_y:
            raise ValueError(f"shift grid is built on box {shifts.box_size}, images are {box_y}")
        ramps = shifts.ramps().reshape(-1, n_pixels)
        shift_radius_squared = (shifts.shifts**2).sum(-1)

    model = templates.reshape(n_states * n_poses, n_pixels)
    # shifts join the component axis, ordered (state, pose, shift) so a C-order reshape splits them
    shifted_model = (model[:, None, :] * ramps[None, :, :]).reshape(-1, n_pixels)
    data = images.reshape(n_particles, n_pixels)

    weighted, squared, index = fold_transfer(data, transfer, (box_y, box_x), shared_as_ones=True)
    transfer_power = _TransferPower(squared=squared, index=index)

    log_pose = jnp.zeros(n_poses) if pose_log_weights is None else pose_log_weights
    data_power = jnp.abs(data) ** 2
    return _Problem(
        model_real=shifted_model.real,
        model_imag=shifted_model.imag,
        # unexpanded: |T_t| = 1, so one row per (state, pose) serves every candidate
        model_power=jnp.abs(model) ** 2,
        weighted_real=weighted.real,
        weighted_imag=weighted.imag,
        data_power=data_power,
        data_power_total=data_power.sum(0),
        transfer_power=transfer_power,
        log_pose=log_pose - jax.scipy.special.logsumexp(log_pose),
        shift_radius_squared=shift_radius_squared,
        shell_index=shell_indices(box_y, n_shells)[0].reshape(-1),
        noise_floor=1e-10 * data_power.mean(),
        n_shells=n_shells,
        latent_shape=(n_states, n_poses, ramps.shape[0]),
    )


def _shift_log_prior(radius_squared: Float[Array, " t"], sigma: Float[Array, ""]) -> Float[Array, " t"]:
    """Log of an isotropic Gaussian over the candidates, normalized over the grid.

    Normalized over the grid rather than analytically, since the candidates are a truncated, discrete
    subset of the plane and the continuous constant would not sum to one over them.
    """
    log_weights = -0.5 * radius_squared / jnp.maximum(sigma**2, jnp.finfo(radius_squared.dtype).tiny)
    return log_weights - jax.scipy.special.logsumexp(log_weights)


def _e_step(problem: _Problem, params: _Params) -> _Expectation:
    """Posterior over (state, pose, shift) for every particle, from a softmax over components."""
    inverse = 1.0 / params.noise_spectrum
    cross = (problem.weighted_real * inverse) @ problem.model_real.T + (
        problem.weighted_imag * inverse
    ) @ problem.model_imag.T
    whitened_power = (problem.model_power * inverse).T
    power = (problem.transfer_power.squared @ whitened_power)[problem.transfer_power.index]

    n_shifts = problem.latent_shape[2]
    log_prior = (
        jnp.log(params.state_weights)[:, None, None]
        + problem.log_pose[None, :, None]
        + _shift_log_prior(problem.shift_radius_squared, params.shift_sigma)[None, None, :]
    ).reshape(-1, n_shifts)
    # the shift axis is kept separate only so the shift-invariant power broadcasts over it
    log_joint = (
        -0.5 * (problem.data_power @ inverse)[:, None, None]
        + params.contrast * cross.reshape(problem.n_particles, -1, n_shifts)
        - 0.5 * params.contrast**2 * power[:, :, None]
        + log_prior[None]
    ).reshape(problem.n_particles, -1)
    log_evidence = jax.scipy.special.logsumexp(log_joint, axis=1)
    return _Expectation(
        responsibilities=jnp.exp(log_joint - log_evidence[:, None]),
        cross=cross,
        power=power,
        log_evidence=log_evidence.sum(),
    )


def _m_step(problem: _Problem, expectation: _Expectation) -> _Params:
    """Closed-form updates: `a` first, then sigma^2 at the new `a`."""
    responsibilities = expectation.responsibilities
    n_particles = responsibilities.shape[0]
    by_latent = responsibilities.reshape(n_particles, *problem.latent_shape)
    # every term but the cross one is shift-invariant, so it wants the shift marginalized away first
    over_shift = by_latent.sum(3).reshape(n_particles, -1)
    n_components = over_shift.shape[1]

    contrast = (responsibilities * expectation.cross).sum() / (over_shift * expectation.power).sum()
    state_weights = by_latent.sum((0, 2, 3)) / n_particles
    # E|t|^2 = 2 sigma^2 in the plane, so the second moment of the shift posterior halves into sigma^2
    shift_second_moment = by_latent.sum((0, 1, 2)) @ problem.shift_radius_squared
    shift_sigma = jnp.sqrt(shift_second_moment / (2.0 * n_particles))

    cross_mode = (
        problem.model_real * (responsibilities.T @ problem.weighted_real)
        + problem.model_imag * (responsibilities.T @ problem.weighted_imag)
    ).sum(0)
    # sum_i r_ic C_i(k)^2 == sum_g C_g(k)^2 sum_{i in g} r_ic, so this reduces over transfer groups
    # and never forms an (n, n_pixels) array
    squared, index = problem.transfer_power.squared, problem.transfer_power.index
    group_resp = jnp.zeros((squared.shape[0], n_components)).at[index].add(over_shift)
    power_mode = (problem.model_power * (group_resp.T @ squared)).sum(0)

    residual = (problem.data_power_total - 2.0 * contrast * cross_mode + contrast**2 * power_mode) / n_particles
    return _Params(
        noise_spectrum=problem.smooth_spectrum(residual),
        contrast=contrast,
        state_weights=state_weights,
        shift_sigma=shift_sigma,
    )


@eqx.filter_jit
def fit_noise_model(
    images: Complex[Array, "n y x"],
    templates: Complex[Array, "s p y x"],
    *,
    transfer: Transfer | None = None,
    shifts: ShiftGrid | None = None,
    noise_spectrum_init: Float[Array, "y x"] | None = None,
    pose_log_weights: Float[Array, " p"] | None = None,
    n_iterations: int = 40,
    tolerance: float = 1e-4,
) -> NoiseModel:
    """Fit sigma^2(k), `a`, the state occupancies, and the shift prior width by EM.

    Args:
        images: fft2 of the particle stack, unshifted.
        templates: fft2 of the noiseless templates, one per (state, pose).
        transfer: Per-group transfer. `None` means it is already folded into `templates`.
        shifts: Candidate translations to marginalize. Only the positions and the box size are read:
            the grid's own `log_weights` are replaced by the isotropic Gaussian being fitted. `None`
            assumes centered particles. Cost is multiplicative in the candidate count.
        noise_spectrum_init: Starting sigma^2(k). Defaults to the mean particle power, which
            overestimates it by the signal power and so starts with soft responsibilities.
        pose_log_weights: Log prior over poses, normalized internally. Use the quadrature weights
            when poses come from an SO(3) grid, since an unweighted sum over a non-uniform grid
            tilts the pose posterior toward wherever the grid is dense.
        n_iterations: Hard iteration cap.
        tolerance: Stop when the mean |log| change in sigma^2 falls below this.

    Returns:
        The fit.
    """
    problem = _prepare(images, templates, transfer, shifts, pose_log_weights)

    def step(state: _State) -> _State:
        params = _m_step(problem, _e_step(problem, state.params))
        # watch the OCCUPANCY as well as the spectrum: sigma^2 is dominated by data power and settles
        # in a couple of iterations, while state_weights converges far more slowly toward its answer
        spectrum_delta = jnp.mean(jnp.abs(jnp.log(params.noise_spectrum / state.params.noise_spectrum)))
        weight_delta = jnp.max(jnp.abs(params.state_weights - state.params.state_weights))
        return _State(params=params, delta=jnp.maximum(spectrum_delta, weight_delta), iteration=state.iteration + 1)

    n_states = problem.latent_shape[0]
    initial = _State(
        params=_Params(
            # a supplied spectrum is floored but not smoothed (the caller's to shape); the default
            # mean particle power is zero at DC for mean-subtracted particles, which 1/sigma^2 sends to nan
            noise_spectrum=(
                problem.smooth_spectrum(problem.data_power_total / problem.n_particles)
                if noise_spectrum_init is None
                else jnp.maximum(noise_spectrum_init.reshape(-1), problem.noise_floor)
            ),
            contrast=jnp.asarray(1.0),
            state_weights=jnp.full(n_states, 1.0 / n_states),
            # the second moment of a uniform prior over the grid, so the shift starts diffuse for the
            # same reason the spectrum starts overestimated
            shift_sigma=jnp.sqrt(problem.shift_radius_squared.mean() / 2.0),
        ),
        delta=jnp.asarray(jnp.inf),
        iteration=jnp.asarray(0),
    )
    state = jax.lax.while_loop(partial(_running, n_iterations=n_iterations, tolerance=tolerance), step, initial)
    expectation = _e_step(problem, state.params)

    box_size = images.shape[-1]
    noise_spectrum = state.params.noise_spectrum
    log_normalizer = _log_normalizer(problem.n_particles, noise_spectrum)
    return NoiseModel(
        noise_spectrum=noise_spectrum.reshape(box_size, box_size),
        contrast=state.params.contrast,
        state_weights=state.params.state_weights,
        shift_sigma=state.params.shift_sigma,
        responsibilities=expectation.responsibilities.reshape(problem.n_particles, *problem.latent_shape).sum(3),
        log_likelihood=expectation.log_evidence - log_normalizer,
        n_iterations=state.iteration,
        delta=state.delta,
    )


@eqx.filter_jit
def calibrate_at_known_poses(
    images: Complex[Array, "n y x"],
    templates: Complex[Array, "s n y x"],
    *,
    transfer: Transfer | None = None,
    state_of: Int[Array, " n"] | None = None,
    shifts: ShiftGrid | None = None,
    noise_spectrum_init: Float[Array, "y x"] | None = None,
    n_amplitude_shells: int | None = None,
    n_iterations: int = 40,
    tolerance: float = 1e-4,
) -> NoiseModel:
    """Fit `a`, sigma^2(k) and the occupancies with each particle at its OWN pose and no bank.

    `fit_noise_model` marginalizes the pose over a bank, and a bank too coarse to match a particle
    shrinks the least-squares contrast, making an absolute `d^2` off a real stack unquotable. Where
    the poses are known the pose latent collapses and both continuous updates stay closed form,

        a          = sum_k cross(k) / sigma^2(k)  /  sum_k power(k) / sigma^2(k)
        sigma^2(k) = shell-average over particles of |Y - a C T|^2

    The pose collapsing does NOT collapse the SHIFT (pass `shifts`; `|ramp| = 1` keeps it exact) nor
    the STATE (occupancies are still fitted). `n_amplitude_shells` restricts the contrast ratio to a
    shell, giving `a(k)`, the test for an envelope or B-factor mismatch a scalar `a` hides.

    Args:
        images: fft2 of the particle stack, unshifted.
        templates: fft2 of the noiseless template for each (state, particle) -- each particle rendered
            at its OWN recorded pose in every state. Not the (state, pose) grid `fit_noise_model`
            takes. One state is allowed and means the assignment is not in question.
        transfer: Per-group transfer. `None` means it is already folded into `templates`.
        state_of: Known state of each particle, for a labeled pool. `None` fits the occupancies; passing
            them where they ARE known avoids a diffuse state posterior biasing `a`.
        shifts: Candidate translations to marginalize, at any spacing. Only the positions and the box
            size are read: the grid's own `log_weights` are replaced by the isotropic Gaussian being
            fitted, exactly as `fit_noise_model` does, so `shift_sigma` comes back fitted by EM. `None`
            says the templates already sit at each particle's true shift, a claim about the extraction.
        noise_spectrum_init: Starting sigma^2(k). Defaults to the mean particle power, as
            `fit_noise_model` does and for the same reason.
        n_amplitude_shells: Fit `a(k)` in this many radial shells instead of one scalar `a`. `None` keeps
            the scalar. Coarse is the point -- one shell per sigma^2 bin is a per-mode gain that would
            fit the difference signal itself.
        n_iterations: Hard iteration cap.
        tolerance: Stop when the mean |log| change in sigma^2 falls below this.

    Returns:
        The fit. `responsibilities` is `(n, s, 1)`, the pose axis being length one. `amplitude_spectrum`
        is `None` unless `n_amplitude_shells` was given, and `contrast` is a scalar either way.
        `shift_sigma` is zero unless `shifts` was given. `log_likelihood` is the observed-data likelihood
        of this mixture and is NOT comparable to `fit_noise_model`'s, which marginalizes another latent.

    Raises:
        ValueError: If the images are not square, if there is not one template per particle per state,
            if `state_of` does not name one state per particle, if `n_amplitude_shells` is not
            positive, or if `transfer` or `shifts` does not match the box.

    Rationale: docs/rationale/particles/em.md
    """
    n_particles = images.shape[0]
    box_y, n_pixels = _require_square(images)
    box_x = box_y
    if n_amplitude_shells is not None and n_amplitude_shells < 1:
        raise ValueError(f"n_amplitude_shells must be at least one shell, got {n_amplitude_shells}")
    if templates.ndim != 4 or templates.shape[1:] != images.shape:
        raise ValueError(
            f"need one template per particle per state, (s, n, y, x), got {templates.shape} against "
            f"{images.shape} of images. This is not the (state, pose) grid `fit_noise_model` takes."
        )
    n_states = templates.shape[0]
    if state_of is not None and state_of.shape != (n_particles,):
        raise ValueError(f"state_of must name one state per particle, got {state_of.shape}")
    n_shells = box_y // 2
    shift_axis: _ShiftAxis | None = None
    if shifts is not None:
        if shifts.box_size != box_y:
            raise ValueError(f"shifts are built on box {shifts.box_size}, the images are box {box_y}")
        # any translation, whole pixel or not: this route builds the ramps rather than reading an
        # inverse transform off the pixel lattice, so it has no reason to inherit that guard
        shift_axis = _ShiftAxis(
            ramp=shifts.ramps().reshape(-1, n_pixels),
            radius_squared=jnp.sum(jnp.asarray(shifts.shifts) ** 2, axis=-1),
        )

    data = images.reshape(n_particles, n_pixels)
    model = templates.reshape(n_states, n_particles, n_pixels)
    weighted, squared, index = fold_transfer(data, transfer, (box_y, box_x), shared_as_ones=True)

    # per-(state, particle), not materialized as one (s, n, k) array: states are walked in Python so
    # the largest temporary stays (n, k) instead of growing with the state count
    cross_raw = [weighted.real * model[s].real + weighted.imag * model[s].imag for s in range(n_states)]
    power_raw = [squared[index] * jnp.abs(model[s]) ** 2 for s in range(n_states)]

    def product(s: int) -> Complex[Array, "n k"]:
        """`Y_w conj(T_s)` per mode: the complex form of `cross_raw[s]`, which the shift latent needs.

        Recomputed rather than stored: a complex copy is twice the largest array here, and this is
        two multiplies XLA can fuse into its consumer.
        """
        return weighted * jnp.conj(model[s])

    data_power = jnp.abs(data) ** 2
    data_power_total = data_power.sum(0)

    shell_index = shell_indices(box_y, n_shells)[0].reshape(-1)
    noise_floor = 1e-10 * data_power.mean()
    fixed = None if state_of is None else jax.nn.one_hot(state_of, n_states)
    # a SECOND shell map at its own resolution: sigma^2 wants many narrow bins and a(k) few wide ones,
    # and tying them would make the envelope as noisy as the spectrum
    amplitude_shell = None if n_amplitude_shells is None else shell_indices(box_y, n_amplitude_shells)[0].reshape(-1)

    def smooth(spectrum: Float[Array, " k"]) -> Float[Array, " k"]:
        """`_smooth_to_shells` at this route's shell map, which is the same rule the pose-grid fit uses."""
        return _smooth_to_shells(spectrum, shell_index, n_shells, noise_floor)

    def fit_contrast(
        cross_mode: Float[Array, " k"], power_mode: Float[Array, " k"], inverse: Float[Array, " k"]
    ) -> Float[Array, "..."]:
        """Weighted least squares for `a`, over the whole band or within each amplitude shell.

        The scalar is the ratio of the two whitened sums; the shell-resolved version is the same ratio
        restricted to a shell, so `n_amplitude_shells=1` does NOT quite reproduce the scalar -- the
        corner modes outside every shell get a bin of their own rather than joining the only one.
        """
        numerator, denominator = cross_mode * inverse, power_mode * inverse
        scalar = numerator.sum() / denominator.sum()
        if amplitude_shell is None:
            return scalar
        per_shell_cross = shell_sum(numerator, amplitude_shell, n_amplitude_shells)
        per_shell_power = shell_sum(denominator, amplitude_shell, n_amplitude_shells)
        # a shell with no template power says nothing about a gain, so it takes the whole-band answer
        # rather than dividing by ~0 and amplifying noise without bound
        usable = per_shell_power > 1e-6 * jnp.maximum(per_shell_power.mean(), noise_floor)
        per_shell = jnp.where(usable, per_shell_cross / jnp.where(usable, per_shell_power, 1.0), scalar)
        return per_shell[amplitude_shell]

    def responsibility(
        params: _Params,
    ) -> tuple[Float[Array, "n s"], Float[Array, "n s t"] | None, Float[Array, ""]]:
        """State posterior, the joint `(state, shift)` posterior where there is one, and the evidence.

        One-hot in the state where the labels are given -- but never in the shift, which no caller
        knows.
        """
        inverse = 1.0 / params.noise_spectrum
        # contrast folded into the WEIGHT rather than multiplying the reduced sums, so a scalar and
        # an a(k) share this arithmetic -- both are just a per-mode factor
        linear, quadratic = params.contrast * inverse, params.contrast**2 * inverse
        power = jnp.stack([raw @ quadratic for raw in power_raw], axis=1)
        floor = -0.5 * (data_power @ inverse)
        if shift_axis is None:
            cross = jnp.stack([raw @ linear for raw in cross_raw], axis=1)
            log_joint = floor[:, None] + cross - 0.5 * power + jnp.log(params.state_weights)[None, :]
            evidence = jax.scipy.special.logsumexp(log_joint, axis=1)
            if fixed is not None:
                return fixed, None, jnp.sum(jnp.sum(fixed * log_joint, axis=1))
            return jnp.exp(log_joint - evidence[:, None]), None, evidence.sum()

        # cross term at every candidate shift, from the phase ramps directly -- not the inverse
        # transform rungs.marginal uses, which is confined to the pixel lattice
        cross = jnp.stack([((product(s) * linear) @ jnp.conj(shift_axis.ramp).T).real for s in range(n_states)], axis=1)
        # the fitted Gaussian, not the grid's stated weights, so the width is identified by EM
        log_shift = _shift_log_prior(shift_axis.radius_squared, params.shift_sigma)
        log_joint = (
            floor[:, None, None]
            + cross
            - 0.5 * power[:, :, None]
            + jnp.log(params.state_weights)[None, :, None]
            + log_shift[None, None, :]
        )
        if fixed is not None:
            # the LABEL is known and the shift still is not, so the state axis is masked and the shift
            # posterior is renormalized inside it rather than the whole thing being one-hot
            log_joint = log_joint + jnp.log(fixed)[:, :, None]
        evidence = jax.scipy.special.logsumexp(log_joint, axis=(1, 2))
        full = jnp.exp(log_joint - evidence[:, None, None])
        return full.sum(2), full, evidence.sum()

    def step(state: _State) -> _State:
        posterior, full, _ = responsibility(state.params)
        # summed at the CURRENT posterior, so these cannot be hoisted out of the loop
        # stacked, not sum(): the builtin starts from int 0 and fit_contrast takes arrays
        power_mode = jnp.stack([posterior[:, s] @ power_raw[s] for s in range(n_states)]).sum(axis=0)
        if shift_axis is None or full is None:  # the first disjunct is what narrows `shift_axis` below
            cross_mode = jnp.stack([posterior[:, s] @ cross_raw[s] for s in range(n_states)]).sum(axis=0)
        else:
            # |ramp| = 1, so averaging over the shift posterior moves only the CROSS term: it becomes
            # Re(Y conj(T) conj(R)) with R = sum_t r_t ramp_t, one effective ramp, keeping arrays (n, k)
            cross_mode = jnp.stack(
                [(product(s) * jnp.conj(full[:, s, :] @ shift_axis.ramp)).real.sum(0) for s in range(n_states)]
            ).sum(axis=0)
        inverse = 1.0 / state.params.noise_spectrum
        contrast = fit_contrast(cross_mode, power_mode, inverse)
        residual = (data_power_total - 2.0 * contrast * cross_mode + contrast**2 * power_mode) / n_particles
        spectrum = smooth(residual)
        # occupancy in the stopping rule as in fit_noise_model: sigma^2 settles fast and state_weights
        # does not, so a spectrum-only test would exit with the occupancy still at 1/n_states
        delta = jnp.maximum(
            jnp.mean(jnp.abs(jnp.log(spectrum / state.params.noise_spectrum))),
            jnp.max(jnp.abs(posterior.mean(0) - state.params.state_weights)),
        )
        second_moment = (
            state.params.shift_sigma
            if shift_axis is None or full is None
            else jnp.sqrt(jnp.sum(full * shift_axis.radius_squared[None, None, :]) / (2.0 * n_particles))
        )
        return _State(
            params=_Params(
                noise_spectrum=spectrum,
                contrast=contrast,
                state_weights=posterior.mean(0),
                shift_sigma=second_moment,
            ),
            delta=delta,
            iteration=state.iteration + 1,
        )

    initial = _State(
        params=_Params(
            noise_spectrum=(
                smooth(data_power_total / n_particles)
                if noise_spectrum_init is None
                else jnp.maximum(noise_spectrum_init.reshape(-1), noise_floor)
            ),
            # shaped as whatever the loop will carry, since `while_loop` requires the two to match
            contrast=jnp.asarray(1.0) if amplitude_shell is None else jnp.ones(n_pixels),
            state_weights=jnp.full(n_states, 1.0 / n_states),
            # starts diffuse and the EM tightens it, as `fit_noise_model`; zero with no shift grid to fit
            shift_sigma=jnp.zeros(()) if shift_axis is None else jnp.sqrt(shift_axis.radius_squared.mean() / 2.0),
        ),
        delta=jnp.asarray(jnp.inf),
        iteration=jnp.asarray(0),
    )
    state = jax.lax.while_loop(partial(_running, n_iterations=n_iterations, tolerance=tolerance), step, initial)

    posterior, _, evidence = responsibility(state.params)
    spectrum = state.params.noise_spectrum
    log_normalizer = _log_normalizer(n_particles, spectrum)
    contrast, amplitude_spectrum = state.params.contrast, None
    if amplitude_shell is not None:
        # reduced to a scalar by the same whitened power that weighted the fit, so `contrast` is the
        # one number to report and equals the scalar answer whenever the shells agree
        weight = sum(posterior[:, s] @ power_raw[s] for s in range(n_states)) / spectrum
        amplitude_spectrum = contrast.reshape(box_y, box_x)
        contrast = (contrast * weight).sum() / weight.sum()
    return NoiseModel(
        noise_spectrum=spectrum.reshape(box_y, box_x),
        contrast=contrast,
        state_weights=state.params.state_weights,
        shift_sigma=state.params.shift_sigma,
        responsibilities=posterior[:, :, None],
        log_likelihood=evidence - log_normalizer,
        n_iterations=state.iteration,
        delta=state.delta,
        amplitude_spectrum=amplitude_spectrum,
    )
