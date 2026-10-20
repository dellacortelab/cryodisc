"""The Fourier grid every module shares, and the conventions that ride on it.

Images, templates, spectra and ramps all live on the UNSHIFTED `fft2` grid, DC at index (0, 0) and
Nyquist on the axes. Nothing here fftshifts, and mixing a shifted array into an unshifted index
silently swaps low and high `k`. Three conventions decided once:
    - `whitening_weight` zeroes DC, particles being mean-subtracted.
    - `phase_ramp` fixes the sign of a translation.
    - `frequency_grid` fixes where Nyquist lands and whether `k` is in cyc/px or inverse A.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Int


def frequency_grid(box_size: int, pixel_size: float = 1.0) -> Float[Array, "y x"]:
    """Radial spatial frequency of every fft mode.

    The one radial-grid builder, so the unshifted layout and where Nyquist lands are decided once.

    Args:
        box_size: Image size in pixels.
        pixel_size: Pixel size in A. The default leaves the grid in cyc/px, which is what the shells
            and the pose code use; pass the real pixel size for inverse A, which is what the
            resolution model uses.

    Returns:
        `|k|` on the unshifted fft grid, zero at index (0, 0) and reaching `0.5 / pixel_size` on the
        axes.
    """
    k_y, k_x = frequency_vectors(box_size, pixel_size)
    return jnp.sqrt(k_y**2 + k_x**2)


def frequency_vectors(box_size: int, pixel_size: float = 1.0) -> tuple[Float[Array, "y x"], Float[Array, "y x"]]:
    """The signed `(k_y, k_x)` components of every fft mode, which `frequency_grid` takes the norm of.

    Separate because the two axes must sometimes be told APART: `microscope.ctf_astigmatic` reads the
    azimuth `arctan2(k_y, k_x)` and `rungs.local` builds per-axis translation tangents. Same layout,
    units and `fftfreq` as `frequency_grid`.

    Args:
        box_size: Image size in pixels.
        pixel_size: Pixel size in A, defaulting to cyc/px.

    Returns:
        `(k_y, k_x)`, each `(box_size, box_size)`. Row index is `y`, so `arctan2(k_y, k_x)` is the
        azimuth measured from the `x` axis.
    """
    k = jnp.fft.fftfreq(box_size) / pixel_size
    return jnp.broadcast_to(k[:, None], (box_size, box_size)), jnp.broadcast_to(k[None, :], (box_size, box_size))


def shell_indices(box_size: int, n_shells: int) -> tuple[Int[Array, "y x"], Float[Array, " n_shells"]]:
    """Radial shell index per fft mode, plus each shell's center frequency.

    Shells are equal-width in |k| out to 0.5 cyc/px, the axis Nyquist, so the grid corners lie
    outside every shell and are assigned `n_shells` for later discarding.

    Args:
        box_size: Image size in pixels.
        n_shells: Number of shells.

    Returns:
        `(index, k_cyc_px)`.
    """
    kr = frequency_grid(box_size)
    index = jnp.where(kr >= 0.5, n_shells, jnp.floor(kr * 2.0 * n_shells).astype(int))
    edges = jnp.linspace(0.0, 0.5, n_shells + 1)
    return index, 0.5 * (edges[:-1] + edges[1:])


def whitening_weight(noise_spectrum: Float[Array, "y x"]) -> Float[Array, "y x"]:
    """`1 / sigma^2(k)`, with DC zeroed because particles are mean-subtracted.

    Lives in one place so every rung weights by the same thing.

    Args:
        noise_spectrum: sigma^2(k) on the unshifted fft grid, DC at index (0, 0).

    Returns:
        The weight. Take its square root for an amplitude whitening.
    """
    box_size = noise_spectrum.shape[-1]
    if noise_spectrum.shape != (box_size, box_size):
        raise ValueError(f"noise_spectrum must be square, got {noise_spectrum.shape}")
    return (1.0 / noise_spectrum).at[0, 0].set(0.0)


def phase_ramp(shifts: Float[Array, "n 2"], box_size: int) -> Complex[Array, "n y x"]:
    """Phase ramps translating a Fourier image, carrying the sign convention for a shift.

    Available as a function, not only `ShiftGrid.ramps`, for a single continuous translation with no
    quadrature rule attached. Separable, so it costs `2 * box_size` complex exponentials.

    Args:
        shifts: (n, 2) translations as (y, x) in pixels.
        box_size: Image size in pixels.

    Returns:
        (n, y, x) on the unshifted fft grid. `fft2(image) * ramp[j]` shifts the image by `shifts[j]`.
    """
    k = jnp.fft.fftfreq(box_size)
    rows = jnp.exp(-2j * jnp.pi * shifts[:, 0, None] * k)
    columns = jnp.exp(-2j * jnp.pi * shifts[:, 1, None] * k)
    return rows[:, :, None] * columns[:, None, :]


def shell_sum(values: Float[Array, "..."], index: Int[Array, "..."], n_shells: int) -> Float[Array, " n_shells+1"]:
    """Sum `values` into the radial bins `shell_indices` assigned, out-of-band bin included.

    The out-of-band bin (index `n_shells`) is RETURNED, not dropped: `particles.em` gathers shell
    averages back through it, and dropping it would clamp every corner mode onto the outermost shell.

    Args:
        values: Anything shaped like the index; both are flattened.
        index: Shell index per mode, from `shell_indices`, with `n_shells` marking out-of-band.
        n_shells: Number of shells.

    Returns:
        `(n_shells + 1,)` sums, the last entry being everything outside the band.
    """
    return jnp.zeros(n_shells + 1).at[index.reshape(-1)].add(values.reshape(-1))


@eqx.filter_jit
def shell_metrics(
    bank_a: Float[Array, "n y x"],
    bank_b: Float[Array, "n y x"],
    *,
    n_shells: int = 8,
) -> dict[str, Array]:
    """Level and shape of `bank_a` relative to `bank_b`, per radial shell.

    A normalized statistic cannot see a level error, so `amp_ratio` carries the level, `frc` the
    shape, and the two combine as `rel_sq_err = A^2 - 2 c A + 1`. `amp_ratio` is translation-invariant;
    `frc` assumes a shared centering, so a sub-pixel offset reads as a shape mismatch.

    Args:
        bank_a: Images under test, (n, box, box).
        bank_b: Reference images, same shape.
        n_shells: Number of radial shells.

    Returns:
        Per-shell `k_cyc_px`, `n_modes`, `amp_ratio`, `frc`, `rel_sq_err` and `power_frac`, each
        `(n_shells,)`; plus `amp_ratio_per_image`, the same level statistic image by image, `(n,
        n_shells)`. A shell holding no modes reports NaN for the three ratios; `n_modes` lets those
        shells be dropped. Everything is in cycles per pixel.

    Raises:
        ValueError: If the two banks have different shapes.
    """
    if bank_a.shape != bank_b.shape:
        raise ValueError(f"bank shapes differ: {bank_a.shape} vs {bank_b.shape}")
    box_size = bank_a.shape[-1]
    index, k_cyc_px = shell_indices(box_size, n_shells)

    def in_band(values: Array) -> Array:
        """Shell sums with the out-of-band bin dropped: nothing here reports on the box corners."""
        return shell_sum(values, index, n_shells)[:n_shells]

    def one(image_a: Array, image_b: Array) -> tuple[Array, Array, Array]:
        spectrum_a = jnp.fft.fft2(image_a - image_a.mean())
        spectrum_b = jnp.fft.fft2(image_b - image_b.mean())
        return (
            in_band(jnp.abs(spectrum_a) ** 2),
            in_band(jnp.abs(spectrum_b) ** 2),
            in_band(jnp.real(spectrum_a * jnp.conj(spectrum_b))),
        )

    # the per-image sums are kept, not just their bank totals: `amp_ratio_per_image` is read off
    # them, and the reduction below is the only thing separating the two
    image_power_a, image_power_b, image_cross = jax.vmap(one)(bank_a, bank_b)
    power_a, power_b, cross = image_power_a.sum(0), image_power_b.sum(0), image_cross.sum(0)
    n_modes = in_band(jnp.ones((box_size, box_size)))
    # an empty shell has nothing to average, so it reports nan deliberately rather than arriving
    # there through 0/0; the floor covers a filled shell whose reference happens to be silent
    filled = n_modes > 0
    floor = jnp.finfo(power_b.dtype).tiny

    def amplitude_ratio(power_test: Array, power_reference: Array) -> Array:
        """`sqrt(P_a / P_b)` per shell, under the empty-shell and silent-reference convention above.

        A function so the bank statistic and the per-image one are the SAME ratio over different sums,
        never differing by a floor or NaN rule applied to one and not the other.
        """
        return jnp.where(filled, jnp.sqrt(power_test / jnp.maximum(power_reference, floor)), jnp.nan)

    amp_ratio = amplitude_ratio(power_a, power_b)
    frc = jnp.where(filled, cross / jnp.sqrt(jnp.maximum(power_a * power_b, floor)), jnp.nan)
    return {
        "k_cyc_px": k_cyc_px,
        "n_modes": n_modes,
        "amp_ratio": amp_ratio,
        "amp_ratio_per_image": amplitude_ratio(image_power_a, image_power_b),
        "frc": frc,
        "rel_sq_err": amp_ratio**2 - 2.0 * frc * amp_ratio + 1.0,
        "power_frac": power_b / power_b.sum(),
    }
