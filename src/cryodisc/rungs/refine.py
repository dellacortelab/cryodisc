"""Each particle's pose, recovered by matched filter with the in-plane angle off the grid.

Rotating a template in-plane by `alpha` is a circular shift of its polar-Fourier modes, so all
`n_angles` cost one inverse FFT:

    C(v, alpha) = dr dtheta sum_r r sum_m Y_m(r) conj(T_m(v, r)) e^{i m alpha}

A bank of `V` directions thus searches `V * n_angles` poses at the cost of `V`, resolving `psi` to `360 / n_angles` deg; bank rendered at `psi = 0`, real radial CTF folds onto the particle, score `C^2 / (2 ||T||^2_W)`.
Rationale: docs/rationale/forward-and-rungs.md
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from cryodisc.grids.fourier import phase_ramp, whitening_weight
from cryodisc.grids.poses import sphere_angles
from cryodisc.microscope import Transfer

_SEARCH_CHUNK = 100
"""Particles per batch when translations are searched, bounding the `(chunk, views, y, x)` rotated
bank -- the largest array here by a factor of `y x / n_angles`."""

_ROTATION_CHUNK = 1000
"""Particles per batch without them, where the largest array is the `(chunk, y, x)` inverse
transform inside `_polar_modes` -- no template axis, so ten times the particles still fits."""


BANK_RADII = 48
"""Radial samples of the polar template grid, the bank quadrature `refine_poses` defaults to.

Converged for an index, which integrates over the band and so cancels the per-frequency error."""


def _CENTER(box_size: int) -> int:
    """The pixel cryoJAX rotates a projection about, which is `box // 2` at BOTH parities.

    `box / 2` agrees on an even box and is half a pixel out on an odd one, breaking the identity the
    module rests on: rotating a template is then no longer a circular shift of its polar modes, and
    the search degrades to worse than chance while still returning a plausible score. Pinned by a test.
    """
    return box_size // 2


class PoseFit(eqx.Module):
    """Recovered poses and their scores.

    Args:
        angles: (n, 3) Euler angles in degrees, cryoJAX intrinsic zyz, ready for `forward_batch`.
        score: `C^2 / (2 ||T||^2_W)` at the fit. Comparable across particles, but off the ladder's
            scale: the contrast is profiled out, and the quadrature deficit below enters squared,
            leaving it low.
        shift: (n, 2) translation as (y, x) in pixels, in `ShiftGrid`'s sense -- the ramp that
            recenters the particle, so `-shift` estimates the offset it was rendered with. Zero
            throughout unless `shift_iterations` was set.
    """

    angles: Float[Array, "n 3"]
    score: Float[Array, " n"]
    shift: Float[Array, "n 2"]


def view_angles(n_nodes: int) -> Float[Array, "v 3"]:
    """Viewing directions nearest `n_nodes` in count, for this rung's search.

    Exactly `poses.sphere_angles`. The count is quantized to the geodesic's `10T + 2` (12, 32, 42, 92,
    122, 162, 252), nearest in log, so this does NOT return `n_nodes` directions -- read the returned
    length rather than assuming the request.

    Args:
        n_nodes: Roughly how many directions to search.

    Returns:
        `(v, 3)` Euler triples in degrees, every one with `psi = 0`, where `v` is the nearest
        available `10T + 2` to `n_nodes` and not `n_nodes` itself.

    Raises:
        ValueError: If `n_nodes` is not positive.
    """
    return sphere_angles(n_nodes)


def refine_poses(
    images: Complex[Array, "n y x"],
    templates: Complex[Array, "v y x"],
    bank_angles: Float[Array, "v 3"],
    noise_spectrum: Float[Array, "y x"],
    *,
    transfer: Transfer | None = None,
    shift_iterations: int = 0,
    n_radii: int = BANK_RADII,
    n_angles: int = 128,
) -> PoseFit:
    """Search every viewing direction and every in-plane angle, for each particle.

    Args:
        images: fft2 of the particle stack, unshifted.
        templates: fft2 of the noiseless templates, one per viewing direction, at `psi = 0` and
            CTF-free -- `CryoJAXModel.bare_contrast` produces exactly this. A CTF folded in here would tie
            the bank to one defocus group.
        bank_angles: `(v, 3)` angles the templates were rendered at, from `view_angles`. `psi` must
            be zero, the recovered in-plane angle being measured from the template.
        noise_spectrum: sigma^2(k) on the unshifted fft grid.
        transfer: Per-group transfer, applied to the particles. Omitting it for a CTF'd stack costs
            the whole search: those particles align no better than chance.
        shift_iterations: Rounds of joint pose-and-translation search. Zero assumes centered particles,
            which is what `particles.sim.simulate_stack` produces at its default `shift_sigma=0`. The
            expensive axis here; three rounds settle a few pixels.
        n_radii: Radial samples of the polar grid, out to the inscribed circle.
        n_angles: Angular samples. Sets the quadrature accuracy and the `360 / n_angles` degree
            resolution of the recovered `psi`.

    Returns:
        One pose and one score per particle.
    """
    if templates.shape[0] < 1:
        raise ValueError("the bank is empty; there is nothing to search over")
    if templates.shape[1:] != images.shape[1:]:
        raise ValueError(f"template shape {templates.shape[1:]} does not match images {images.shape[1:]}")
    if bank_angles.shape != (templates.shape[0], 3):
        raise ValueError(f"bank_angles must be ({templates.shape[0]}, 3), got {bank_angles.shape}")
    if float(jnp.max(jnp.abs(bank_angles[:, 2]))) != 0.0:
        raise ValueError("the bank must be rendered at psi = 0; the in-plane angle is what is searched for")
    if transfer is not None and transfer.index.shape != (images.shape[0],):
        raise ValueError(f"transfer.index must be one index per particle, got {transfer.index.shape}")
    # n_radii = 1 makes `radii[1]` an out-of-bounds gather, which JAX CLAMPS: the quadrature weight
    # is zero, the bank zeroed, every particle returned at `bank_angles[0]`. Two is the minimum.
    if n_radii < 2 or n_angles < 2:
        raise ValueError(f"need n_radii >= 2 and n_angles >= 2, got {n_radii} and {n_angles}")
    # a non-finite or non-positive mode makes every correlation NaN; `nan > 0` is False, so the
    # profiled score masks it to zero and the search returns `bank_angles[0]` looking confident
    if not bool(jnp.all(jnp.isfinite(noise_spectrum) & (noise_spectrum > 0.0))):
        raise ValueError("noise_spectrum must be finite and strictly positive at every mode")
    return _refine(images, templates, bank_angles, noise_spectrum, transfer, shift_iterations, n_radii, n_angles)


@eqx.filter_jit
def _refine(
    images: Complex[Array, "n y x"],
    templates: Complex[Array, "v y x"],
    bank_angles: Float[Array, "v 3"],
    noise_spectrum: Float[Array, "y x"],
    transfer: Transfer | None,
    shift_iterations: int,
    n_radii: int,
    n_angles: int,
) -> PoseFit:
    """`refine_poses` once its arguments are checked.

    Split off so the whole thing can be jitted: the `psi = 0` guard reads a value, which would both
    trace to nothing and force a device sync. `particles.sim.simulate_stack` splits for the same reason.
    """
    box_size = images.shape[-1]
    weight = whitening_weight(noise_spectrum)
    radii, coordinates = _polar_grid(box_size, n_radii, n_angles)
    whitened = templates * jnp.sqrt(weight)
    modes = jax.lax.map(lambda spectrum: _polar_modes(spectrum, coordinates), whitened, batch_size=_ROTATION_CHUNK)
    # every correlation constant folded in once, not per particle per iteration: polar quadrature
    # weight, radial Jacobian, and the n_pixels of Parseval between real-space and Fourier sums
    quadrature = (radii[1] - radii[0]) * (2.0 * jnp.pi / n_angles) * box_size * box_size
    bank = quadrature * radii[:, None] * jnp.conj(modes)

    # ||C T||^2_W per (group, view), summed exactly on the fft grid. C is squared: it belongs to the
    # model once, and the correlation has already spent its copy on the data.
    model_power = (jnp.abs(whitened) ** 2).reshape(templates.shape[0], -1)
    if transfer is None:
        weighted = images * jnp.sqrt(weight)
        power = jnp.sum(model_power, axis=1)[None]
        group = jnp.zeros(images.shape[0], dtype=jnp.int32)
    else:
        group = transfer.checked_index()
        weighted = images * transfer.values[group] * jnp.sqrt(weight)
        power = (transfer.values.reshape(transfer.values.shape[0], -1) ** 2) @ model_power.T

    view, in_plane, score, shift = _search(
        weighted, group, bank, jnp.fft.ifft2(whitened).real, power, coordinates, shift_iterations
    )
    return PoseFit(
        angles=bank_angles[view].at[:, 2].set(360.0 * in_plane / n_angles),
        score=score,
        shift=shift,
    )


def _polar_grid(box_size: int, n_radii: int, n_angles: int) -> tuple[Float[Array, " r"], Float[Array, "2 r a"]]:
    """Radii of the polar sampling ring, and the `(y, x)` pixel coordinates of every sample.

    The correlation lands below the exact Fourier sum, mostly from the order-1 `map_coordinates`
    interpolation (where JAX caps) rather than the inscribed-circle aperture; band-limiting does not
    close it, a finer kernel (zero-pad the spectrum before resampling) does.

    The deficit is the same factor for every template, which leaves the argmax untouched and is why
    this rung tolerates it. A rung that subtracts one template's score from another's does not: see
    `PoseFit.score`, and note that `marginal` scores `a*cross - 0.5*power`, where shrinking `cross`
    alone would bias `Lambda` rather than cancel.
    """
    radii = jnp.linspace(0.0, 0.5 * box_size - 1.0, n_radii)
    angle = 2.0 * jnp.pi * jnp.arange(n_angles) / n_angles
    center = _CENTER(box_size)
    return radii, jnp.stack([center + radii[:, None] * jnp.sin(angle), center + radii[:, None] * jnp.cos(angle)])


def _polar_modes(spectrum: Complex[Array, "y x"], coordinates: Float[Array, "2 r a"]) -> Complex[Array, "r a"]:
    """Angular Fourier modes of one image's polar resampling, where `alpha` becomes an index."""
    image = jnp.fft.ifft2(spectrum).real
    return jnp.fft.fft(jax.scipy.ndimage.map_coordinates(image, list(coordinates), order=1), axis=-1)


def _rotate(images: Float[Array, "v y x"], degrees: Float[Array, " v"]) -> Float[Array, "v y x"]:
    """Rotate each image about the box center by its own angle, matching the polar index's sign.

    Bilinear: it feeds only the translation peak-finder, where a uniform contrast error moves nothing.
    """
    center = _CENTER(images.shape[-1])
    axis = jnp.arange(images.shape[-1]) - center
    y, x = jnp.meshgrid(axis, axis, indexing="ij")

    def one(image: Float[Array, "y x"], angle: Float[Array, ""]) -> Float[Array, "y x"]:
        cos, sin = jnp.cos(jnp.radians(angle)), jnp.sin(jnp.radians(angle))
        return jax.scipy.ndimage.map_coordinates(
            image, [center + cos * y - sin * x, center + sin * y + cos * x], order=1
        )

    return jax.vmap(one)(images, degrees)


def _peak_shift(spectrum: Complex[Array, "y x"], templates: Float[Array, "v y x"]) -> tuple[Array, Array]:
    """Each template's best translation of the particle onto it, and the correlation there.

    One 2D FFT gives the whole translation surface, so every shift is searched at once with no
    optimizer and no grid. The peak sits at MINUS the shift that was applied to the particle, so the
    returned value is what `ShiftGrid` must be handed to undo it.
    """
    box_size = spectrum.shape[-1]
    surface = jnp.fft.ifft2(jnp.conj(spectrum)[None] * jnp.fft.fft2(templates)).real
    flat = surface.reshape(templates.shape[0], -1)
    location = jnp.argmax(flat, axis=1)
    offset = jnp.stack([location // box_size, location % box_size], axis=-1)
    return jnp.where(offset > box_size // 2, offset - box_size, offset).astype(float), jnp.max(flat, axis=1)


def _correlate(
    spectrum: Complex[Array, "y x"], coordinates: Float[Array, "2 r a"], bank: Complex[Array, "v r a"]
) -> Float[Array, "v a"]:
    """Whitened correlation of one particle against every template at every in-plane angle.

    `bank` arrives conjugated and carrying the quadrature weights, so this is the whole of the
    per-particle work: one polar resampling, one contraction and one inverse FFT.
    """
    return jnp.fft.ifft(jnp.einsum("rm,vrm->vm", _polar_modes(spectrum, coordinates), bank), axis=-1).real


def _profile(correlation: Float[Array, "v a"], power: Float[Array, " v"]) -> Float[Array, "v a"]:
    """The matched-filter score with the contrast profiled out.

    A negative correlation matches the particle's inverse; squaring it would let a strong anti-match
    win the argmax, and the norm is what stops a heavy template winning on mass alone.
    """
    return jnp.where(correlation > 0.0, correlation**2 / (2.0 * power[:, None]), 0.0)


def _search(
    weighted: Complex[Array, "n y x"],
    group: Float[Array, " n"],
    bank: Complex[Array, "v r a"],
    real_bank: Float[Array, "v y x"],
    power: Float[Array, "g v"],
    coordinates: Float[Array, "2 r a"],
    shift_iterations: int,
) -> tuple[Array, Array, Array, Array]:
    """Best `(view, in_plane, score, shift)` per particle, batched to bound the correlation cube."""
    box_size, n_angles = weighted.shape[-1], coordinates.shape[-1]

    def one(particle: tuple[Complex[Array, "y x"], Array]) -> tuple[Array, Array, Array, Array]:
        spectrum, index = particle
        model_power = power[index]

        def step(shift: Float[Array, " 2"], _: None) -> tuple[Float[Array, " 2"], None]:
            """Score EVERY viewing direction at its own best shift, then recenter on the winner.

            Scoring only the centered cube's winner is what fails: an off-center particle lets a wrong
            direction out-score the right one, and the recentering then chases it.
            """
            centered = spectrum * phase_ramp(shift[None], box_size)[0]
            in_plane = jnp.argmax(_profile(_correlate(centered, coordinates, bank), model_power), axis=-1)
            offset, peak = _peak_shift(centered, _rotate(real_bank, 360.0 * in_plane / n_angles))
            return shift + offset[jnp.argmax(_profile(peak[:, None], model_power))], None

        shift, _ = jax.lax.scan(step, jnp.zeros(2), None, length=shift_iterations)
        # a zero shift is a ramp of ones, and `shift_iterations` is static, so skip building it
        centered = spectrum if shift_iterations == 0 else spectrum * phase_ramp(shift[None], box_size)[0]
        score = _profile(_correlate(centered, coordinates, bank), model_power)
        view, in_plane = jnp.unravel_index(jnp.argmax(score), score.shape)
        return view, in_plane, score[view, in_plane], shift

    batch = _SEARCH_CHUNK if shift_iterations > 0 else _ROTATION_CHUNK
    return jax.lax.map(one, (weighted, group), batch_size=batch)
