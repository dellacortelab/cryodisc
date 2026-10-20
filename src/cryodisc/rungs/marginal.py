"""The marginalized rung: integrating over the pose rather than profiling it to a point.

    F_s(i) = logsumexp_c [ log w_c + Re <Y_i, a M_{s,c}>_w - ||a M_{s,c}||^2_w / 2 ]

is the free energy of state s for particle i, `Lambda_i = F_A(i) - F_B(i)`; c runs over poses and, given
a shift grid, shifts. `|ramp| = 1`, so a shift enters only the cross term, which ONE inverse transform
gives at every integer shift at once -- exact quadrature on the PIXEL LATTICE, so a sub-pixel grid is
rejected. `marginal` is the logsumexp; `best_match` the max; `known_pose_log_ratio` the oracle.
Rationale: docs/rationale/rungs/marginal.md
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float

from cryodisc.forward.expansion import expand_templates, pose_chunks
from cryodisc.grids.fourier import whitening_weight
from cryodisc.grids.poses import ShiftGrid
from cryodisc.microscope import Transfer, fold_transfer

_SHIFT_SCRATCH_BUDGET = 900 * 1024 * 1024
"""Device scratch the shift batch is allowed, in bytes, which sets the batch size.

The live cost is the complex product and transform output, scaling with `n_poses * box^2`, so a
batch fixed in PARTICLES would mean nothing. `_shift_chunk` converts this budget into a count
instead, so the bound travels with the problem. Not a throughput knob: the counts it spans perform
within a few percent of each other.
"""


def _shift_chunk(n_poses: int, n_pixels: int) -> int:
    """Particles per batch that keep the live `(chunk, poses, y, x)` surfaces inside the budget."""
    # product and transform output, one of each for the packed pair, plus their hermitian temporaries
    per_particle = 4 * n_poses * n_pixels * jnp.dtype(jnp.complex64).itemsize
    return max(1, _SHIFT_SCRATCH_BUDGET // per_particle)


class LogRatio(eqx.Module):
    """Per-particle log likelihood ratio of state A over state B.

    Args:
        marginal: Pose integrated against its prior.
        best_match: Pose point-estimated on the noisy image.
    """

    marginal: Float[Array, " n"]
    best_match: Float[Array, " n"]


def pose_log_ratio(
    images: Complex[Array, "n y x"],
    templates_a: Complex[Array, "p y x"],
    templates_b: Complex[Array, "p y x"],
    noise_spectrum: Float[Array, "y x"],
    *,
    amplitude: float = 1.0,
    pose_log_weights: Float[Array, " p"] | None = None,
    shifts: ShiftGrid | None = None,
    transfer: Transfer | None = None,
) -> LogRatio:
    """Per-particle log-ratio with the pose marginalized, and with it point-estimated.

    Args:
        images: fft2 of the particle stack, unshifted.
        templates_a: fft2 of the noiseless state-A templates, one per pose.
        templates_b: fft2 of the noiseless state-B templates, on the same poses.
        noise_spectrum: sigma^2(k) on the unshifted fft grid.
        amplitude: Template scale `a`.
        pose_log_weights: Log pose prior, normalized internally. Use the quadrature weights from
            `poses.pose_bank`, which are uniform -- still passed rather than assumed, since an
            unweighted sum over a non-uniform bank tilts the posterior.
        shifts: Translations to marginalize as well. Every candidate must be a whole number of
            pixels, at any spacing -- a transform reaches the pixel lattice and nothing between it.
            `None` assumes centered particles, which skips the transforms and scores by one matmul.
        transfer: Per-group transfer. `None` means it is already folded into the templates.

    Returns:
        Both readings, one entry per particle.

    Raises:
        ValueError: If the template banks disagree, if they do not match the images, if the bank is
            empty, if the noise spectrum is not finite and positive, if the transfer is misshapen, or
            if `shifts` leaves the pixel lattice or is built on another box.
    """
    if templates_a.shape != templates_b.shape:
        raise ValueError(f"template banks differ: {templates_a.shape} vs {templates_b.shape}")
    if templates_a.shape[1:] != images.shape[1:]:
        raise ValueError(f"template shape {templates_a.shape[1:]} does not match images {images.shape[1:]}")
    if templates_a.shape[0] < 1:
        raise ValueError("the pose bank is empty; there is nothing to marginalize over")
    if not bool(jnp.all(jnp.isfinite(noise_spectrum) & (noise_spectrum > 0.0))):
        raise ValueError("noise_spectrum must be finite and strictly positive at every mode")
    # transfer shape guards are not repeated here: `fold_transfer` raises them, with these exact
    # messages, on this path
    if shifts is not None:
        _check_shift_lattice(shifts, images.shape[-1])
    # `jnp.asarray`, not the bare float: `eqx.filter_jit` treats a Python scalar as STATIC, so a
    # caller sweeping the amplitude would retrace per value and miss the XLA cache
    return _pose_log_ratio(
        images, templates_a, templates_b, noise_spectrum, jnp.asarray(amplitude), pose_log_weights, shifts, transfer
    )


def _check_shift_lattice(shifts: ShiftGrid, box_size: int) -> None:
    """Refuse a shift grid this module cannot score exactly.

    Shared by the bank route and the oracle, which reach the lattice the same way -- one inverse
    transform gives the cross term at every INTEGER offset at once, and at nothing between them.

    Args:
        shifts: The grid to check.
        box_size: Box the images are on.

    Raises:
        ValueError: If the grid was built on another box, or leaves the pixel lattice.
    """
    if shifts.box_size != box_size:
        # reading the lattice by index cannot fail on a mismatched box, so it is checked here
        raise ValueError(f"shifts are built on box {shifts.box_size}, the images are box {box_size}")
    # the VALUES, not a `step` field describing them: a hand-built grid carries no such field,
    # and a step of 2 is a perfectly good lattice this route scores exactly
    lattice = np.asarray(shifts.shifts)
    if not np.array_equal(lattice, np.round(lattice)):
        off = lattice[np.any(lattice != np.round(lattice), axis=-1)][0]
        raise ValueError(
            f"every shift must be a whole number of pixels, got {tuple(off)}. A transform gives "
            "the cross term at the integer offsets and nothing between them; see this module's "
            "docstring for why a fractional one needs a transform of its own."
        )


@eqx.filter_jit
def _pose_log_ratio(
    images: Complex[Array, "n y x"],
    templates_a: Complex[Array, "p y x"],
    templates_b: Complex[Array, "p y x"],
    noise_spectrum: Float[Array, "y x"],
    amplitude: Float[Array, ""],
    pose_log_weights: Float[Array, " p"] | None,
    shifts: ShiftGrid | None,
    transfer: Transfer | None,
) -> LogRatio:
    """`pose_log_ratio` once its arguments are checked.

    Split off so the whole thing can be jitted: the lattice guard reads the shift VALUES, which
    would both trace to nothing and force a device sync. `rungs.refine` splits for the same reason.
    """
    box_size = images.shape[-1]
    n_particles, n_poses = images.shape[0], templates_a.shape[0]
    n_pixels = box_size * box_size

    inverse = whitening_weight(noise_spectrum).reshape(-1)
    model_a = templates_a.reshape(n_poses, n_pixels)
    model_b = templates_b.reshape(n_poses, n_pixels)
    data = images.reshape(n_particles, n_pixels)

    weighted_data, squared, index = fold_transfer(data, transfer, (box_size, box_size))
    power_a = _bank_power(model_a, inverse, amplitude, squared, index)
    power_b = _bank_power(model_b, inverse, amplitude, squared, index)

    log_pose = jnp.zeros(n_poses) if pose_log_weights is None else pose_log_weights
    log_pose = log_pose - jax.scipy.special.logsumexp(log_pose)

    if shifts is None:
        whitened = weighted_data * inverse[None]
        score_a = _bank_scores(whitened, model_a, log_pose, amplitude, power_a)
        score_b = _bank_scores(whitened, model_b, log_pose, amplitude, power_b)
        return LogRatio(
            marginal=jax.scipy.special.logsumexp(score_a, axis=1) - jax.scipy.special.logsumexp(score_b, axis=1),
            best_match=score_a.max(1) - score_b.max(1),
        )

    total_a, peak_a, total_b, peak_b = _shift_reduce(
        weighted_data, model_a, model_b, inverse, amplitude, log_pose, shifts, power_a, power_b, box_size
    )
    return LogRatio(marginal=total_a - total_b, best_match=peak_a - peak_b)


def _shift_reduce(
    weighted_data: Complex[Array, "n k"],
    model_a: Complex[Array, "p k"],
    model_b: Complex[Array, "p k"],
    inverse: Float[Array, " k"],
    amplitude: Float[Array, ""],
    log_pose: Float[Array, " p"],
    shifts: ShiftGrid,
    power_a: Float[Array, "n p"],
    power_b: Float[Array, "n p"],
    box_size: int,
) -> tuple[Array, Array, Array, Array]:
    """Both states integrated and peaked over the joint (pose, shift) latent, per particle.

    Split out of `_pose_log_ratio` so `streamed_pose_log_ratio` can call it on ONE pose chunk at a
    time: `logsumexp` and `max` compose over chunks only if the two states are reduced SEPARATELY,
    the difference of two log-sums being no log-sum.

    Args:
        weighted_data: `C x` per particle, flattened, from `fold_transfer`.
        model_a: State-A templates for these poses, flattened to (p, k).
        model_b: State B on the same poses.
        inverse: `1 / sigma^2(k)`, flattened.
        amplitude: Template scale `a`.
        log_pose: Log pose prior for these poses, normalized over the WHOLE bank by the caller.
        shifts: The translation lattice, already checked against the box.
        power_a: `_bank_power` for state A on these poses.
        power_b: The same for state B.
        box_size: Side of the box, which the flattened shapes no longer carry and the lattice indexes against.

    Returns:
        `(logsumexp_a, max_a, logsumexp_b, max_b)`, one entry per particle.
    """
    n_particles, n_pixels = weighted_data.shape
    n_poses = model_a.shape[0]
    # negative shifts index from the end, which is the fft's own wrap-around and needs no branch
    index = jnp.mod(jnp.round(shifts.shifts).astype(int), box_size)
    prior = log_pose[:, None] + shifts.log_weights[None]
    # broadcast so one particle's row is one element of the map, whether the transfer is shared or not
    power_a = jnp.broadcast_to(power_a, (n_particles, n_poses))
    power_b = jnp.broadcast_to(power_b, (n_particles, n_poses))

    def at_particle(row: tuple[Array, Array, Array]) -> tuple[Array, Array, Array, Array]:
        """One particle's two readings per state, over the joint (pose, shift) latent."""
        datum, particle_power_a, particle_power_b = row
        whitened = (datum * inverse).reshape(box_size, box_size)

        def surfaces(row_a: Array, row_b: Array) -> tuple[Array, Array]:
            """BOTH states' cross terms at every integer shift, from ONE transform. See the docstring."""
            packed = _hermitian(whitened * jnp.conj(row_a).reshape(box_size, box_size)) + 1j * _hermitian(
                whitened * jnp.conj(row_b).reshape(box_size, box_size)
            )
            surface = jnp.fft.ifft2(packed) * n_pixels
            return surface.real, surface.imag

        cross_a, cross_b = jax.vmap(surfaces)(model_a, model_b)
        gather = (slice(None), index[:, 0], index[:, 1])
        score_a = prior + amplitude * cross_a[gather] - 0.5 * particle_power_a[:, None]
        score_b = prior + amplitude * cross_b[gather] - 0.5 * particle_power_b[:, None]
        logsumexp = jax.scipy.special.logsumexp
        return logsumexp(score_a), score_a.max(), logsumexp(score_b), score_b.max()

    return jax.lax.map(at_particle, (weighted_data, power_a, power_b), batch_size=_shift_chunk(n_poses, n_pixels))


_POSE_SCRATCH_BUDGET = 512 * 1024 * 1024
"""Scratch one pose chunk may use, in bytes, which sets the chunk size.

The ceiling this lifts is the BANK, not the reduction: `pose_log_ratio` is handed both banks whole,
so a large grid exceeds device memory before any scoring. Set deliberately well below what the
chunk's arrays need, because the RENDER is the larger cost and is not counted here -- a
Gaussian-mixture projection holds intermediates several times its output. Raise it only with a
render that has been measured, not one that merely looks affordable.
"""


def _pose_chunk(n_particles: int, n_pixels: int) -> int:
    """Poses per chunk that keep one chunk's bank and scores inside the budget."""
    # one complex template and its whitened power per pose, plus a real score per particle per pose
    per_pose = 2 * n_pixels * jnp.dtype(jnp.complex64).itemsize + n_particles * jnp.dtype(jnp.float32).itemsize
    return max(1, _POSE_SCRATCH_BUDGET // per_pose)


def streamed_pose_log_ratio(
    images: Complex[Array, "n y x"],
    render_a: "Callable[[Float[Array, 'c 3']], Complex[Array, 'c y x']]",
    render_b: "Callable[[Float[Array, 'c 3']], Complex[Array, 'c y x']]",
    angles: Float[Array, "p 3"],
    pose_log_weights: Float[Array, " p"],
    noise_spectrum: Float[Array, "y x"],
    *,
    amplitude: float = 1.0,
    shifts: ShiftGrid | None = None,
    transfer: Transfer | None = None,
    pose_chunk: int | None = None,
    expansion: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> LogRatio:
    """`pose_log_ratio` with the bank rendered and reduced in chunks, so `K` is not memory-bound.

    Both `logsumexp` and the max compose over chunks, so one streaming pass gives both readings; the
    pose prior is normalized once over the whole grid first. Takes render callables rather than
    materialized banks (a bank argument would already have hit the ceiling); `CryoJAXModel.bare_contrast`
    is one.

    Args:
        images: fft2 of the particle stack, unshifted.
        render_a: Maps `(c, 3)` angles to `(c, y, x)` fft2 state-A templates.
        render_b: The same for state B.
        angles: `(p, 3)` Euler angles of the bank, in degrees.
        pose_log_weights: Log pose prior over the WHOLE grid, normalized internally exactly once.
        noise_spectrum: sigma^2(k) on the unshifted fft grid.
        amplitude: Template scale `a`.
        shifts: Translations to marginalize as well, on the pixel lattice, exactly as
            `pose_log_ratio` takes them. `None` assumes centered particles. Streaming the pose loop
            lifts the top rung above what a whole bank fits in; the shift surfaces are already
            chunked over particles inside `_shift_reduce`, so the two chunkings compose.
        transfer: Per-group transfer. `None` means it is already folded into the templates.
        pose_chunk: Poses per chunk, or `None` to size it from `_POSE_SCRATCH_BUDGET`.
        expansion: `poses.expansion_plan` for this bank, which cuts the RENDER eightfold by building
            seven of every eight templates from the eighth by index permutation. The answer is
            unchanged -- the expansion resamples nothing. `None` renders every node. Only a bank
            whose directions are antipodally closed has a plan; see `expansion_plan`.

    Returns:
        Both readings, one entry per particle, identical to `pose_log_ratio` on the same bank.

    Raises:
        ValueError: If the angles and weights disagree, the bank is empty, the noise spectrum is not
            finite and positive, the transfer is misshapen, or the expansion plan does not describe
            this bank.
    """
    if angles.shape[0] != pose_log_weights.shape[0]:
        raise ValueError(f"one weight per pose: {angles.shape[0]} angles, {pose_log_weights.shape[0]} weights")
    if angles.shape[0] < 1:
        raise ValueError("the pose bank is empty; there is nothing to marginalize over")
    if not bool(jnp.all(jnp.isfinite(noise_spectrum) & (noise_spectrum > 0.0))):
        raise ValueError("noise_spectrum must be finite and strictly positive at every mode")
    n_particles, box_size = images.shape[0], images.shape[-1]
    n_pixels = box_size * box_size
    if shifts is not None:
        _check_shift_lattice(shifts, box_size)
    inverse = whitening_weight(noise_spectrum).reshape(-1)
    data = images.reshape(n_particles, n_pixels)
    weighted_data, squared, index = fold_transfer(data, transfer, (box_size, box_size))
    whitened = weighted_data * inverse[None] if shifts is None else None
    # normalized ONCE, over every pose, before the grid is broken up
    log_pose = pose_log_weights - jax.scipy.special.logsumexp(pose_log_weights)

    step = pose_chunk if pose_chunk is not None else _pose_chunk(n_particles, n_pixels)
    running = [(jnp.full((n_particles,), -jnp.inf), jnp.full((n_particles,), -jnp.inf)) for _ in range(2)]
    scale = jnp.asarray(amplitude)
    for piece, nodes, plan in pose_chunks(angles, step, expansion):
        weights = log_pose[nodes]
        # both states at once on the shift route (they share one inverse transform in
        # `_shift_reduce.surfaces`); the centered route has none to share and renders them in turn
        if shifts is None:
            for slot, render in enumerate((render_a, render_b)):
                templates = render(piece)
                if plan is not None:
                    templates = expand_templates(templates, *plan)
                model = templates.reshape(-1, n_pixels)
                power = _bank_power(model, inverse, scale, squared, index)
                score = _bank_scores(whitened, model, weights, scale, power)
                total, peak = running[slot]
                running[slot] = (
                    jnp.logaddexp(total, jax.scipy.special.logsumexp(score, axis=1)),
                    jnp.maximum(peak, score.max(1)),
                )
            continue
        models = []
        for render in (render_a, render_b):
            templates = render(piece)
            if plan is not None:
                templates = expand_templates(templates, *plan)
            models.append(templates.reshape(-1, n_pixels))
        chunk = _shift_reduce(
            weighted_data,
            models[0],
            models[1],
            inverse,
            scale,
            weights,
            shifts,
            _bank_power(models[0], inverse, scale, squared, index),
            _bank_power(models[1], inverse, scale, squared, index),
            box_size,
        )
        for slot in range(2):
            total, peak = running[slot]
            running[slot] = (jnp.logaddexp(total, chunk[2 * slot]), jnp.maximum(peak, chunk[2 * slot + 1]))
    return LogRatio(
        marginal=running[0][0] - running[1][0],
        best_match=running[0][1] - running[1][1],
    )


def _hermitian(spectrum: Complex[Array, "y x"]) -> Complex[Array, "y x"]:
    """The conjugate-symmetric part of a spectrum, whose inverse transform is exactly real.

    `Re ifft2(P) == ifft2(hermitian(P))` for ANY `P`, because the antisymmetric half is precisely
    what the inverse transform sends to the imaginary axis. Taking it here is what lets the two
    states share one transform: the imaginary output slot is free only if it carries no leakage from
    the real one, and hermitizing guarantees that rather than assuming it of the caller's input.

    Args:
        spectrum: Any array on the unshifted fft grid.

    Returns:
        `(P(k) + conj(P(-k))) / 2`, where `-k` is the fft's own wrap-around, so this holds on odd
        boxes as well as even ones.
    """
    return 0.5 * (spectrum + jnp.conj(jnp.roll(spectrum[::-1, ::-1], (1, 1), axis=(0, 1))))


def known_pose_log_ratio(
    images: Complex[Array, "n y x"],
    templates_a: Complex[Array, "n y x"],
    templates_b: Complex[Array, "n y x"],
    noise_spectrum: Float[Array, "y x"],
    *,
    amplitude: float = 1.0,
    transfer: Transfer | None = None,
    shifts: ShiftGrid | None = None,
) -> Float[Array, " n"]:
    """The oracle: each particle scored against its own true template in both states.

    No grid, so a continuous true pose is scored exactly rather than rounded onto a node. The pose
    being known does NOT make the shift known -- pass `shifts` for anything off a picker; see
    docs/rationale/rungs/marginal.md.

    Args:
        images: fft2 of the particle stack, unshifted.
        templates_a: fft2 of each particle's own state-A template, at its true pose.
        templates_b: fft2 of each particle's own state-B template, likewise.
        noise_spectrum: sigma^2(k) on the unshifted fft grid.
        amplitude: Template scale `a`.
        transfer: Per-group transfer. `None` means it is already folded into the templates.
        shifts: Translations to marginalize, on the pixel lattice. `None` says the templates already
            sit at each particle's true shift, which is a claim about the extraction.

    Returns:
        Lambda, one entry per particle.

    Raises:
        ValueError: If a template bank does not match the images, if the noise spectrum is not finite
            and positive, or if `shifts` leaves the pixel lattice or is built on another box.
    """
    if templates_a.shape != images.shape or templates_b.shape != images.shape:
        raise ValueError("the oracle needs one template per particle per state, matching `images`")
    if not bool(jnp.all(jnp.isfinite(noise_spectrum) & (noise_spectrum > 0.0))):
        raise ValueError("noise_spectrum must be finite and strictly positive at every mode")
    if shifts is not None:
        _check_shift_lattice(shifts, images.shape[-1])
    # `jnp.asarray` and not the bare float; see `pose_log_ratio` for why and what it costs.
    return _known_pose_log_ratio(
        images, templates_a, templates_b, noise_spectrum, jnp.asarray(amplitude), transfer, shifts
    )


@eqx.filter_jit
def _known_pose_log_ratio(
    images: Complex[Array, "n y x"],
    templates_a: Complex[Array, "n y x"],
    templates_b: Complex[Array, "n y x"],
    noise_spectrum: Float[Array, "y x"],
    amplitude: Float[Array, ""],
    transfer: Transfer | None,
    shifts: ShiftGrid | None,
) -> Float[Array, " n"]:
    """The jitted body; see `known_pose_log_ratio`, which holds the guards `jit` cannot.

    Args:
        images: fft2 of the particle stack, unshifted.
        templates_a: fft2 of each particle's own state-A template.
        templates_b: The state-B template for the same particle.
        noise_spectrum: sigma^2(k) on the unshifted fft grid.
        amplitude: Template scale `a`.
        transfer: Per-group transfer, or `None`.
        shifts: Translations to marginalize, or `None` for centered particles.

    Returns:
        Lambda, one entry per particle.
    """
    inverse, model_a, model_b = _oracle_models(templates_a, templates_b, noise_spectrum, amplitude, transfer)
    n_particles = images.shape[0]
    if shifts is None:
        difference = model_a - model_b
        cross = (jnp.real(images * jnp.conj(difference)) * inverse).reshape(n_particles, -1).sum(1)
        power_gap = ((jnp.abs(model_a) ** 2 - jnp.abs(model_b) ** 2) * inverse).reshape(n_particles, -1).sum(1)
        return cross - 0.5 * power_gap

    # `|ramp| = 1`, so a shift moves the cross term and leaves the model power alone -- one transform
    # per particle, not one per candidate. Same identity as the bank route's.
    box_size = images.shape[-1]
    n_pixels = box_size * box_size
    index = jnp.mod(jnp.round(shifts.shifts).astype(int), box_size)
    log_shift = shifts.log_weights - jax.scipy.special.logsumexp(shifts.log_weights)
    power_a = ((jnp.abs(model_a) ** 2) * inverse).reshape(n_particles, -1).sum(1)
    power_b = ((jnp.abs(model_b) ** 2) * inverse).reshape(n_particles, -1).sum(1)

    def at_particle(row: tuple[Array, Array, Array, Array, Array]) -> Array:
        """One particle's log-ratio, with its own shift integrated out."""
        image, own_a, own_b, particle_power_a, particle_power_b = row
        whitened = image * inverse[0]
        # BOTH states from ONE transform: the Hermitian part of each product transforms to a real
        # surface, so they ride the real and imaginary halves of a single complex ifft2
        packed = _hermitian(whitened * jnp.conj(own_a)) + 1j * _hermitian(whitened * jnp.conj(own_b))
        surface = jnp.fft.ifft2(packed) * n_pixels
        gather = (index[:, 0], index[:, 1])
        score_a = log_shift + surface.real[gather] - 0.5 * particle_power_a
        score_b = log_shift + surface.imag[gather] - 0.5 * particle_power_b
        logsumexp = jax.scipy.special.logsumexp
        return logsumexp(score_a) - logsumexp(score_b)

    return jax.lax.map(at_particle, (images, model_a, model_b, power_a, power_b), batch_size=_shift_chunk(1, n_pixels))


def _oracle_models(
    templates_a: Complex[Array, "n y x"],
    templates_b: Complex[Array, "n y x"],
    noise_spectrum: Float[Array, "y x"],
    amplitude: Float[Array, ""],
    transfer: Transfer | None,
) -> tuple[Float[Array, "1 y x"], Complex[Array, "n y x"], Complex[Array, "n y x"]]:
    """The whitening weight and the two scaled models, shared by the oracle score and its closed form.

    One code path on purpose: `known_pose_log_ratio` and `known_pose_d2` have to agree about the
    amplitude, the per-particle transfer and the dropped DC mode, and a second implementation of that
    convention is how a score and the `d^2` checking it drift apart while each looks self-consistent.

    Args:
        templates_a: fft2 of each particle's own state-A template.
        templates_b: The state-B template for the same particle.
        noise_spectrum: sigma^2(k) on the unshifted fft grid.
        amplitude: Template scale `a`.
        transfer: Per-group transfer, or `None` when it is already folded in.

    Returns:
        `(inverse, model_a, model_b)`, the weight carrying a leading axis so it broadcasts over particles.
    """
    inverse = whitening_weight(noise_spectrum)[None]
    model_a, model_b = amplitude * templates_a, amplitude * templates_b
    if transfer is not None:
        per_particle = transfer.values[transfer.checked_index()]
        model_a, model_b = model_a * per_particle, model_b * per_particle
    return inverse, model_a, model_b


def known_pose_d2(
    templates_a: Complex[Array, "n y x"],
    templates_b: Complex[Array, "n y x"],
    noise_spectrum: Float[Array, "y x"],
    *,
    amplitude: float = 1.0,
    transfer: Transfer | None = None,
) -> Float[Array, " n"]:
    """`d_i^2` per particle at its own true pose, from the templates and NO STACK.

    The oracle rung's separation is a property of the templates, the optics and `sigma^2(k)` alone; the
    images only add noise around it. So this is the number `known_pose_log_ratio` must reproduce in the
    mean, and `validation.coherence.predicted_coherence` turns its spread into that pool's baseline.

    Args:
        templates_a: fft2 of each particle's own state-A template, at its true pose and shift.
        templates_b: The state-B template for the same particle.
        noise_spectrum: sigma^2(k) on the unshifted fft grid.
        amplitude: Template scale `a`.
        transfer: Per-group transfer. `None` means it is already folded into the templates.

    Returns:
        `d_i^2`, one entry per particle.

    Raises:
        ValueError: If the two template stacks do not match.
    """
    if templates_a.shape != templates_b.shape:
        raise ValueError(f"the two template stacks differ: {templates_a.shape} vs {templates_b.shape}")

    inverse, model_a, model_b = _oracle_models(
        templates_a, templates_b, noise_spectrum, jnp.asarray(amplitude), transfer
    )
    difference = model_a - model_b
    return ((jnp.abs(difference) ** 2) * inverse).reshape(difference.shape[0], -1).sum(1)


def _bank_power(
    model: Complex[Array, "p k"],
    inverse: Float[Array, " k"],
    amplitude: Float[Array, ""],
    squared: Float[Array, "g k"] | None,
    index: Array | None,
) -> Float[Array, "n p"]:
    """`||a C M||^2_W` per (particle, pose), as one row broadcast when the transfer is shared.

    Args:
        model: The pose bank, flattened to (p, k).
        inverse: `1 / sigma^2(k)`, flattened.
        amplitude: Template scale `a`.
        squared: `C^2` per group from `fold_transfer`, or `None`.
        index: Group per particle from `fold_transfer`, or `None`.

    Returns:
        (1, p) when the transfer is shared, (n, p) otherwise.
    """
    mode_power = _mode_power(model, inverse, amplitude)
    return mode_power.sum(1)[None] if squared is None else (squared @ mode_power.T)[index]


def _bank_scores(
    whitened: Complex[Array, "n k"],
    model: Complex[Array, "p k"],
    log_prior: Float[Array, " p"],
    amplitude: Float[Array, ""],
    power,
) -> Float[Array, "n p"]:
    """The joint log-likelihood of every (particle, pose), up to a constant shared by both states.

    `log p(theta) + a Re<W x, M> - ||a C M||^2_W / 2`. The cancelling constant is the data's own
    power, which is why the two states can be differenced without it.

    Args:
        whitened: `W C x` per particle, flattened.
        model: The pose bank, flattened.
        log_prior: Normalized log weight per pose.
        amplitude: Template scale `a`.
        power: `_bank_power`'s output, broadcasting over particles when the transfer is shared.

    Returns:
        (n, p) scores.
    """
    return log_prior[None] + amplitude * (whitened @ jnp.conj(model).T).real - 0.5 * power


def _mode_power(
    model: Complex[Array, "p k"], inverse: Float[Array, " k"], amplitude: Float[Array, ""]
) -> Float[Array, "p k"]:
    """Whitened model power per mode, left unsummed so a per-group transfer can weight each mode."""
    return amplitude**2 * jnp.abs(model) ** 2 * inverse[None]
