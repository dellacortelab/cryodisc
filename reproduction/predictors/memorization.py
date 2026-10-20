"""The geometry the memorization law rests on: how many free parameters a band holds.

Shared by the ladder driver and the memorization law, so it lives here rather than in either.
"""

from functools import cache

import numpy as np


def band_voxels(box_size: int, band_limit: float) -> float:
    """How many 3D Fourier voxels the band holds, which is the amplitude that fits.

    A particle's noise lands on the map's central slice at its own orientation, so the count that
    matters is the 3D voxel count `sum_k 2 pi k (2k / N)`, not the 2D one -- the same `N / 2k`
    slice-sampling factor that turns image SNR into map SSNR.

    Counted on the grid rather than as `(4/3) pi R^3`, because the shell binning rounds and the
    smooth form undercounts at small boxes.

    Args:
        box_size: Image size in pixels.
        band_limit: Cutoff in cycles per pixel.

    Returns:
        The voxel count inside the band, DC excluded.
    """
    cumulative = _shell_cumulative(box_size)
    return float(cumulative[min(round(band_limit * box_size), len(cumulative) - 1)])


@cache
def _shell_cumulative(box_size: int) -> np.ndarray:
    """Voxels in shells `1..R`, indexed by `R`. DC is shell 0 and is excluded.

    Cached because `band_voxels` and `implied_band` read the same table in opposite directions, and
    each call otherwise rebuilds three `box_size**3` meshgrids.

    Args:
        box_size: Image size in pixels.

    Returns:
        The cumulative count, length `max shell + 1`, so `[-1]` is the whole grid.
    """
    axis = np.fft.fftfreq(box_size) * box_size
    kz, ky, kx = np.meshgrid(axis, axis, axis, indexing="ij")
    shell = np.round(np.sqrt(kx**2 + ky**2 + kz**2)).astype(int)
    counts = np.bincount(shell.ravel())
    counts[0] = 0
    return np.cumsum(counts)


def implied_band(box_size: int, amplitude: float) -> float:
    """The band whose voxel count the fitted amplitude actually equals.

    Inverts `band_voxels`, so a measured amplitude can be read as the band it corresponds to. That is
    the comparison two ladders can share: `--band-limit` is in cycles per pixel, so one nominal value
    is a different physical resolution on every extraction, while the implied band is in Angstroms
    once the pixel size is known.

    Interpolated within the shell rather than bisected on the band: `band_voxels` depends on the band
    only through `round(band * box_size)`, so it is a step function and bisecting converges to the
    step's lower edge. Reading the fractional position inside the shell makes a ratio of exactly 1
    return the scored band exactly.

    Args:
        box_size: Image size in pixels.
        amplitude: The fitted amplitude at slope -1, in nats x particles. Halved before inverting,
            because the prediction is `2V` and the rungs are fitted against the total count.

    Returns:
        The band in cycles per pixel, on `[0, sqrt(3)/2]` -- the grid's corner, not Nyquist, which a
        band above `0.5` says the amplitude exceeds what the whole box holds. Zero when the amplitude
        is below the innermost shell.
    """
    cumulative = _shell_cumulative(box_size)
    target = amplitude / 2.0
    if target <= 0.0:
        return 0.0
    if target >= cumulative[-1]:
        return float(len(cumulative) - 1) / box_size
    shell = int(np.searchsorted(cumulative, target, side="left"))
    below = float(cumulative[shell - 1]) if shell else 0.0
    within = (target - below) / (float(cumulative[shell]) - below)
    return (shell - 1 + within) / box_size
