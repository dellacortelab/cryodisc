"""Independent reference for the image layer: the transform, the grid, the ramp, the shells.

Nothing here imports `cryodisc`; each quantity is built by a different route than
production, so the Fourier conventions are checked, not inherited. DC drops from `Lambda` (mean-subtracted).

    F[u, v]              = sum_y sum_x f[y, x] exp(-2 pi i (u y + v x) / n)   (unnormalized, numpy's)
    freq of mode u       = u / n for u <= n/2, else (u - n) / n
    F[shift by t](k)     = F(k) exp(-2 pi i k . t)   (sign: multiplying by the ramp moves by +t)
    Lambda               = sum_k [Y conj(M_A - M_B) - |M_A|^2/2 + |M_B|^2/2] / sigma^2(k)
"""

import numpy as np

__all__ = [
    "explicit_dft2",
    "explicit_frequencies",
    "explicit_phase_ramp",
    "shell_amplitude_and_correlation",
    "shell_assignment",
    "whitened_log_ratio",
    "whitening_weight",
]


def explicit_dft2(image: np.ndarray) -> np.ndarray:
    """The 2D discrete Fourier transform, summed mode by mode.

    Args:
        image: `(n, n)` real or complex image.

    Returns:
        `(n, n)` spectrum, unshifted, unnormalized. Same convention as `numpy.fft.fft2`, reached by
        a different route.
    """
    data = np.asarray(image, dtype=complex)
    n = data.shape[0]
    if data.shape != (n, n):
        raise ValueError(f"expected a square image, got {data.shape}")
    index = np.arange(n)
    # a 1D kernel applied along each axis in turn: still an explicit sum, but O(n^3) not O(n^4)
    kernel = np.exp(-2j * np.pi * np.outer(index, index) / n)
    return kernel @ data @ kernel


def explicit_frequencies(n: int, pixel_size: float = 1.0) -> np.ndarray:
    """The signed frequency of each mode along one axis, with the wrap written out.

    Args:
        n: Box size in pixels.
        pixel_size: Pixel size in A. The default leaves the result in cycles per pixel.

    Returns:
        `(n,)` frequencies. Modes above `n/2` are NEGATIVE, which is the convention a shifted array
        would silently break.
    """
    index = np.arange(n)
    wrapped = np.where(index <= n // 2, index, index - n)
    # an even box has one unpaired mode at Nyquist; numpy signs it negative, reproduced here
    if n % 2 == 0:
        wrapped[n // 2] = -(n // 2)
    return wrapped / (n * pixel_size)


def explicit_phase_ramp(shift: np.ndarray, n: int) -> np.ndarray:
    """The ramp that translates an image by `shift`, built from the outer product of two 1D ramps.

    Args:
        shift: `(2,)` translation as (y, x) in pixels.
        n: Box size.

    Returns:
        `(n, n)` unit-modulus ramp. `fft2(image) * ramp` is the transform of the image moved by
        `+shift`.
    """
    frequency = explicit_frequencies(n)
    rows = np.exp(-2j * np.pi * shift[0] * frequency)
    columns = np.exp(-2j * np.pi * shift[1] * frequency)
    return np.outer(rows, columns)


def whitening_weight(noise_variance: np.ndarray) -> np.ndarray:
    """`1 / sigma^2(k)` with DC zeroed.

    Args:
        noise_variance: `sigma^2` on the unshifted grid.

    Returns:
        The weight. DC is zero because particles are mean-subtracted, so that mode carries no signal
        and weighting it would weight a pure offset.
    """
    weight = 1.0 / np.asarray(noise_variance, dtype=float)
    weight[0, 0] = 0.0
    return weight


def whitened_log_ratio(
    images: np.ndarray,
    template_a: np.ndarray,
    template_b: np.ndarray,
    noise_variance: np.ndarray,
    amplitude: float = 1.0,
) -> np.ndarray:
    """The known-pose log-ratio for a stack of Fourier images, from the definition.

    Args:
        images: `(n, y, x)` fft of the particles, unshifted, not mean-subtracted.
        template_a: `(y, x)` fft of the state-A template at the same pose.
        template_b: `(y, x)` fft of the state-B template.
        noise_variance: `sigma^2` on the unshifted grid.
        amplitude: Template scale `a`.

    Returns:
        `(n,)` log-ratios.
    """
    weight = whitening_weight(noise_variance)
    model_a = amplitude * np.asarray(template_a)
    model_b = amplitude * np.asarray(template_b)
    cross = np.sum(np.real(np.asarray(images) * np.conj(model_a - model_b)) * weight, axis=(-2, -1))
    power = np.sum((np.abs(model_a) ** 2 - np.abs(model_b) ** 2) * weight)
    return cross - 0.5 * power


def shell_assignment(n: int, n_shells: int) -> tuple[np.ndarray, np.ndarray]:
    """Which radial shell each mode falls in, and the shell centers.

    Shells are equal-width in `|k|` out to 0.5 cycles per pixel, the AXIS Nyquist, so the grid's
    corners lie outside every shell and are assigned `n_shells` for discarding.

    Args:
        n: Box size.
        n_shells: Number of shells.

    Returns:
        `(index, centers)`.
    """
    frequency = explicit_frequencies(n)
    radius = np.sqrt(frequency[:, None] ** 2 + frequency[None, :] ** 2)
    index = np.where(radius >= 0.5, n_shells, np.floor(radius * 2.0 * n_shells).astype(int))
    edges = np.linspace(0.0, 0.5, n_shells + 1)
    return index, 0.5 * (edges[:-1] + edges[1:])


def shell_amplitude_and_correlation(
    bank_a: np.ndarray, bank_b: np.ndarray, n_shells: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-shell level and shape of one bank against another, by an explicit loop over shells.

    Args:
        bank_a: `(n, y, x)` real images under test.
        bank_b: `(n, y, x)` real reference images.
        n_shells: Number of shells.

    Returns:
        `(amplitude_ratio, correlation)`, each `(n_shells,)`. The amplitude ratio is
        `sqrt(P_a / P_b)` and the correlation is `Re<F_a, F_b> / sqrt(P_a P_b)`, both from
        bank-summed spectra.
    """
    under_test = np.asarray(bank_a, dtype=float)
    reference = np.asarray(bank_b, dtype=float)
    n = under_test.shape[-1]
    index, _ = shell_assignment(n, n_shells)

    power_a = np.zeros(n_shells)
    power_b = np.zeros(n_shells)
    cross = np.zeros(n_shells)
    for image_a, image_b in zip(under_test, reference, strict=True):
        spectrum_a = np.fft.fft2(image_a - image_a.mean())
        spectrum_b = np.fft.fft2(image_b - image_b.mean())
        for shell in range(n_shells):
            mask = index == shell
            power_a[shell] += np.sum(np.abs(spectrum_a[mask]) ** 2)
            power_b[shell] += np.sum(np.abs(spectrum_b[mask]) ** 2)
            cross[shell] += np.sum(np.real(spectrum_a[mask] * np.conj(spectrum_b[mask])))

    with np.errstate(divide="ignore", invalid="ignore"):
        amplitude = np.sqrt(power_a / power_b)
        correlation = cross / np.sqrt(power_a * power_b)
    return amplitude, correlation
