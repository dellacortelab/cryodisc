r"""Per-micrograph and per-patch CTF fitting, so the defocus need not be taken on trust.

A deposited defocus can be wrong in value or units, and refitting is the only check. Fitting through
`microscope.ctf_astigmatic` -- the function `metadata.transfer_from_meta` scores with -- rules out a
sign, contrast or azimuth mismatch by construction. The method is CTFFIND-style: a tiled periodogram,
median-filter background removal, a coarse 1D defocus search, then a local `(mean, Ac, As)` refinement.

Rationale: docs/rationale/particles/ctffit.md
"""

import warnings
from typing import Any

import numpy as np
from jaxtyping import Float
from scipy.ndimage import median_filter
from scipy.optimize import minimize

from cryodisc.microscope import ctf_astigmatic


def tiled_power_spectrum(image: Float[np.ndarray, "y x"], tile: int = 512, overlap: float = 0.5) -> np.ndarray:
    r"""Average `|fft2|^2` over overlapping cosine-windowed tiles.

    Args:
        image: The micrograph.
        tile: Tile edge in pixels, reduced to the largest power of two that fits when the image is
            smaller.
        overlap: Fractional overlap between neighboring tiles.

    Returns:
        `(tile, tile)` real, fft-ordered.

    Raises:
        ValueError: If not even one tile fits, or if `overlap` is not in `[0, 1)`.
    """
    picture = np.asarray(image, float)
    height, width = picture.shape
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if min(height, width) < tile:
        tile = 1 << int(np.floor(np.log2(min(height, width))))
    step = max(1, int(tile * (1.0 - overlap)))
    window = np.outer(np.hanning(tile), np.hanning(tile))

    total, count = np.zeros((tile, tile)), 0
    for y in range(0, height - tile + 1, step):
        for x in range(0, width - tile + 1, step):
            patch = picture[y : y + tile, x : x + tile]
            total += np.abs(np.fft.fft2((patch - patch.mean()) * window)) ** 2
            count += 1
    if count == 0:
        raise ValueError(f"image {picture.shape} is smaller than one {tile} tile")
    return total / count


def _median_filter(values: np.ndarray, width: int) -> np.ndarray:
    """Median filter with edge replication, the width forced odd and clipped to the array.

    Args:
        values: The profile.
        width: Window in samples.

    Returns:
        The filtered profile, the same length.
    """
    # scipy.ndimage.median_filter centers an EVEN window off by half a sample, which would shift the
    # estimated envelope against the profile it is subtracted from; forcing the width odd aligns them
    width = max(3, int(width) | 1)
    width = min(width, len(values) if len(values) % 2 else len(values) - 1)
    if width < 3:
        return values.copy()
    return median_filter(values, size=width, mode="nearest")


def _oscillation(values: np.ndarray, width: int) -> np.ndarray:
    """The profile with its smooth envelope removed and its scale normalized.

    Args:
        values: The profile.
        width: Median-filter window in samples.

    Returns:
        The standardized residual, which is what both the data and the model are scored as.
    """
    residual = values - _median_filter(values, width)
    spread = residual.std()
    return residual / spread if spread > 0 else residual


def _radial_profile(
    spectrum: np.ndarray, pixel_size: float, k_low: float, k_high: float, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Radially average an fft-ordered power spectrum over a band.

    Args:
        spectrum: `(n, n)` fft-ordered power.
        pixel_size: Pixel size in A.
        k_low: Low edge of the band in inverse A.
        k_high: High edge.
        n_bins: Bins across the band.

    Returns:
        `(k_centers, mean_power)`, restricted to bins that got at least one mode.
    """
    n = spectrum.shape[0]
    frequency = np.fft.fftfreq(n, d=pixel_size)
    radius = np.hypot(frequency[None, :], frequency[:, None]).ravel()
    inside = (radius >= k_low) & (radius <= k_high)
    index = np.clip(((radius[inside] - k_low) / (k_high - k_low) * n_bins).astype(int), 0, n_bins - 1)
    total = np.bincount(index, weights=spectrum.ravel()[inside], minlength=n_bins)
    count = np.bincount(index, minlength=n_bins)
    occupied = count > 0
    centers = k_low + (np.arange(n_bins) + 0.5) / n_bins * (k_high - k_low)
    return centers[occupied], total[occupied] / count[occupied]


def _radial_ctf(k: np.ndarray, defocus: float, optics: dict[str, float]) -> np.ndarray:
    """The isotropic CTF through the SAME function that builds a scoring transfer.

    Args:
        k: Spatial frequency in inverse A.
        defocus: Mean underfocus in A.
        optics: `voltage_in_kv`, `spherical_aberration_in_mm`, `amplitude_contrast_ratio`.

    Returns:
        The CTF, unsquared. Evaluated with `k_y = 0` and equal defoci, which makes the astigmatic
        form collapse to the radial one exactly rather than approximately.
    """
    return np.asarray(
        ctf_astigmatic(np.zeros_like(k), k, defocus_u=defocus, defocus_v=defocus, **optics)  # type: ignore[arg-type]
    )


_BACKGROUND_FRACTION = 0.12
"""Share of the radial profile used as the running-background window, as a fraction of its length.

A constant rather than a parameter: unlike the band and the defocus bounds it is not per-dataset.
"""


def fit_defocus(
    spectrum: np.ndarray,
    pixel_size: float,
    *,
    voltage_in_kv: float = 300.0,
    spherical_aberration_in_mm: float = 2.7,
    amplitude_contrast_ratio: float = 0.1,
    k_low: float = 1.0 / 25.0,
    k_high: float = 1.0 / 5.0,
    defocus_low: float = 2000.0,
    defocus_high: float = 40000.0,
    n_coarse: int = 400,
    n_bins: int = 180,
    astigmatism: bool = True,
    astigmatism_max: float = 2000.0,
) -> dict[str, Any]:
    """Fit `(defocus_u, defocus_v, astigmatism_angle)` to a power spectrum.

    Args:
        spectrum: fft-ordered power spectrum, from `tiled_power_spectrum`.
        pixel_size: Pixel size in A.
        voltage_in_kv: Accelerating voltage.
        spherical_aberration_in_mm: Cs.
        amplitude_contrast_ratio: Q.
        k_low: Low edge of the fit band in inverse A. The default 25-5 A is where Thon rings localize
            a defocus, NOT where the specimen carries signal.
        k_high: High edge of the fit band.
        defocus_low: Lowest mean underfocus searched, in A.
        defocus_high: Highest.
        n_coarse: Nodes in the coarse 1D scan.
        n_bins: Radial bins across the band.
        astigmatism: Refine it, on the 2D spectrum since it is invisible in a radial average.
        astigmatism_max: Bound on `|defocus_u - defocus_v|` in A.

    Returns:
        The fit in `ParticleMeta`'s conventions -- angstroms and RADIANS -- plus the band, the optics
        assumed, and the correlation `score`. A low score means the rings were never found.

    Raises:
        ValueError: If the band is empty or inverted.
    """
    if not 0.0 < k_low < k_high:
        raise ValueError(f"need 0 < k_low < k_high, got {k_low} and {k_high}")
    optics = {
        "voltage_in_kv": voltage_in_kv,
        "spherical_aberration_in_mm": spherical_aberration_in_mm,
        "amplitude_contrast_ratio": amplitude_contrast_ratio,
    }
    centers, profile = _radial_profile(spectrum, pixel_size, k_low, k_high, n_bins)
    if centers.size < 8:
        raise ValueError(f"only {centers.size} radial bins fall in [{k_low}, {k_high}]; widen the band")
    width = max(3, int(_BACKGROUND_FRACTION * len(centers)))
    observed = _oscillation(profile, width)

    def radial_score(defocus: float) -> float:
        model = _oscillation(_radial_ctf(centers, defocus, optics) ** 2, width)
        return float(observed @ model) / len(centers)

    coarse = np.linspace(defocus_low, defocus_high, n_coarse)
    scores = np.array([radial_score(dz) for dz in coarse])
    best_defocus = float(coarse[int(np.argmax(scores))])
    # a local polish over one coarse step either side, since the coarse grid quantizes the answer
    step = coarse[1] - coarse[0]
    fine = np.linspace(best_defocus - step, best_defocus + step, 41)
    fine_scores = np.array([radial_score(dz) for dz in fine])
    best_defocus = float(fine[int(np.argmax(fine_scores))])
    best_score = float(fine_scores.max())

    defocus_u = defocus_v = best_defocus
    angle = 0.0
    if astigmatism:
        n = spectrum.shape[0]
        frequency = np.fft.fftfreq(n, d=pixel_size)
        grid_x, grid_y = np.meshgrid(frequency, frequency)
        radius = np.hypot(grid_x, grid_y)
        band = (radius >= k_low) & (radius <= k_high)
        radius_band, x_band, y_band = radius[band], grid_x[band], grid_y[band]

        background = np.interp(radius_band, centers, _median_filter(profile, width))
        observed_2d = spectrum[band] - background
        observed_2d = observed_2d / (observed_2d.std() or 1.0)

        def astigmatic_score(mean_defocus: float, half_spread: float, azimuth: float) -> float:
            model = (
                np.asarray(
                    ctf_astigmatic(
                        y_band,
                        x_band,
                        defocus_u=mean_defocus + half_spread,
                        defocus_v=mean_defocus - half_spread,
                        astigmatism_angle=azimuth,
                        **optics,  # type: ignore[arg-type]
                    )
                )
                ** 2
            )
            # the model's own background, removed by the same filter and interpolated the same way
            model = model - np.interp(
                radius_band, centers, _median_filter(_radial_ctf(centers, mean_defocus, optics) ** 2, width)
            )
            spread = model.std()
            return float(observed_2d @ model / (len(observed_2d) * spread)) if spread > 0 else -1e30

        def objective(vector: np.ndarray) -> float:
            mean_defocus, cosine_part, sine_part = vector
            half_spread = float(np.hypot(cosine_part, sine_part))
            if half_spread > astigmatism_max / 2.0:
                return 1e30
            return -astigmatic_score(mean_defocus, half_spread, 0.5 * float(np.arctan2(sine_part, cosine_part)))

        # a few off-axis seeds as well as the isotropic one, to break the symmetry of a start at
        # Ac = As = 0, where the azimuth is undefined and every direction looks equally good
        seed_radius = min(astigmatism_max / 8.0, 200.0)
        seeds = [(best_defocus, 0.0, 0.0)] + [
            (best_defocus, seed_radius * np.cos(t), seed_radius * np.sin(t))
            for t in (0.0, np.pi / 3.0, 2.0 * np.pi / 3.0)
        ]
        best_score, best_vector = -1e30, seeds[0]
        for seed in seeds:
            result = minimize(
                objective, seed, method="Nelder-Mead", options={"xatol": 1.0, "fatol": 1e-8, "maxiter": 800}
            )
            if -result.fun > best_score:
                best_score, best_vector = float(-result.fun), tuple(result.x)
        mean_defocus, cosine_part, sine_part = best_vector
        half_spread = float(np.hypot(cosine_part, sine_part))
        defocus_u, defocus_v = mean_defocus + half_spread, mean_defocus - half_spread
        angle = 0.5 * float(np.arctan2(sine_part, cosine_part))

    return {
        "defocus_u": float(defocus_u),
        "defocus_v": float(defocus_v),
        "astigmatism_angle": float(np.mod(angle, np.pi)),
        "defocus": float(0.5 * (defocus_u + defocus_v)),
        "score": float(best_score),
        "voltage_in_kv": voltage_in_kv,
        "spherical_aberration_in_mm": spherical_aberration_in_mm,
        "amplitude_contrast_ratio": amplitude_contrast_ratio,
        "band": (float(k_low), float(k_high)),
        "pixel_size": float(pixel_size),
    }


def fit_patches(
    image: Float[np.ndarray, "y x"],
    pixel_size: float,
    *,
    grid: tuple[int, int] = (3, 3),
    tile: int = 512,
    **kwargs: Any,
) -> list[tuple[float, float, dict[str, Any]]]:
    """Fit a defocus per patch of a micrograph.

    A tilted specimen carries a per-particle defocus, which a per-patch fit plus
    `interpolate_to_particles` reproduces without a patch-CTF job.

    Args:
        image: The micrograph.
        pixel_size: Pixel size in A.
        grid: Patches down and across.
        tile: Patch edge in pixels.
        **kwargs: Passed to `fit_defocus`.

    Returns:
        `[(center_y, center_x, fit), ...]` with centers in PIXELS, one entry per grid position
        including positions whose offsets coincide.

    Warns:
        UserWarning: If distinct offsets are fewer than grid positions, so the fitted plane has no
            slope along at least one axis.
    """
    height, width = image.shape
    down, across = grid
    # memoized on the offset: the offsets are not always distinct (at height == tile every row offset
    # is 0), and refitting an identical tile would repeat the whole cost
    fitted: dict[tuple[int, int], dict[str, Any]] = {}
    patches = []
    for row in range(down):
        for column in range(across):
            y0 = int(row * (height - tile) / max(down - 1, 1)) if down > 1 else (height - tile) // 2
            x0 = int(column * (width - tile) / max(across - 1, 1)) if across > 1 else (width - tile) // 2
            if (y0, x0) not in fitted:
                fitted[y0, x0] = fit_defocus(
                    tiled_power_spectrum(image[y0 : y0 + tile, x0 : x0 + tile], tile), pixel_size, **kwargs
                )
            # copied so repeated offsets stay independent dicts
            patches.append((y0 + tile / 2.0, x0 + tile / 2.0, dict(fitted[y0, x0])))
    if len(fitted) < down * across:
        # a collapsed grid still fits a plane, but interpolate_to_particles gets a rank-deficient
        # design whose slope along the collapsed axis is 0 -- a false tilt, not a measured one
        warnings.warn(
            f"a {down}x{across} grid of {tile} px patches on a {height}x{width} image lands on "
            f"{len(fitted)} of {down * across} distinct offsets, so the fitted defocus plane carries "
            "no slope along at least one axis; use a smaller tile or a smaller grid",
            UserWarning,
            stacklevel=2,
        )
    return patches


def interpolate_to_particles(
    patches: list[tuple[float, float, dict[str, Any]]], positions: Float[np.ndarray, "n 2"]
) -> dict[str, np.ndarray]:
    """Per-particle defocus from a patch fit, by a plane through the patch defoci.

    A PLANE rather than a nearest-patch lookup, the cause being a tilted roughly flat specimen -- and a
    plane puts no discontinuity between neighbors falling in different patches. The astigmatism is the
    patch median, being a property of the optics and not of position.

    Args:
        patches: The output of `fit_patches`.
        positions: `(n, 2)` particle centers as (y, x) in pixels of the micrograph.

    Returns:
        `{"defocus_u", "defocus_v", "astigmatism_angle", "plane"}`, the first three per particle and
        `plane` the fitted `(constant, d/dy, d/dx)` coefficients.

    Raises:
        ValueError: If fewer than three patches are given, which cannot determine a plane.
    """
    if len(patches) < 3:
        raise ValueError(f"a plane needs at least three patches, got {len(patches)}")
    center_y = np.array([patch[0] for patch in patches])
    center_x = np.array([patch[1] for patch in patches])
    defocus = np.array([patch[2]["defocus"] for patch in patches])
    design = np.stack([np.ones_like(center_y), center_y, center_x], 1)
    coefficients, *_ = np.linalg.lstsq(design, defocus, rcond=None)

    where = np.asarray(positions, float).reshape(-1, 2)
    predicted = coefficients[0] + coefficients[1] * where[:, 0] + coefficients[2] * where[:, 1]
    spread = float(np.median([patch[2]["defocus_u"] - patch[2]["defocus_v"] for patch in patches]))
    # a DOUBLED-ANGLE circular mean, not a median: azimuths wrap into [0, pi), so a plain median of
    # angles split across both ends lands mid-branch-cut. Doubling makes the mean well defined
    azimuth = 2.0 * np.asarray([patch[2]["astigmatism_angle"] for patch in patches], float)
    angle = float(0.5 * np.arctan2(np.sin(azimuth).mean(), np.cos(azimuth).mean()) % np.pi)
    return {
        "defocus_u": predicted + 0.5 * spread,
        "defocus_v": predicted - 0.5 * spread,
        "astigmatism_angle": np.full(len(where), angle),
        "plane": np.asarray(coefficients),
    }
