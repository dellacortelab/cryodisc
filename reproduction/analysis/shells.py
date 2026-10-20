"""Fourier shells on a cubic grid, and the correlation between two maps binned on them.

Shared by the FSC bridge and the memorization amplitude, which is why it lives here. The two must
bin identically or their curves are not comparable, and `reproduction/experiments/` is a directory of scripts
with no importable names -- reaching into a sibling's file is the import the layout test refuses.

**The DC shell is carried and never used.** Every sum starts at shell 1, matching the scorer's
convention that the mean is not a discriminating coefficient.
"""

import numpy as np


def shell_index(box_size: int) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Integer shell of every Fourier voxel, and the frequency vectors that go with it.

    Args:
        box_size: Cubic box edge in voxels.

    Returns:
        The shell index, the per-shell voxel count, and the three frequency grids in voxel units.
    """
    axis = np.fft.fftfreq(box_size) * box_size
    kz, ky, kx = np.meshgrid(axis, axis, axis, indexing="ij")
    index = np.round(np.sqrt(kx**2 + ky**2 + kz**2)).astype(int)
    counts = np.bincount(index.ravel(), minlength=box_size)[: box_size // 2 + 1]
    return index, counts, (kz, ky, kx)


class Shells:
    """Shell-averaged quantities on one fixed grid, so every curve is binned the same way.

    The DC shell is carried and never used: every sum below starts at 1, matching the scorer's
    convention that the mean is not a discriminating coefficient.
    """

    def __init__(self, index: np.ndarray, counts: np.ndarray) -> None:
        self.index = index
        self.counts = counts
        self.n_shells = counts.size
        self._flat = index.ravel()
        self._inside = self._flat < self.n_shells

    def mean(self, field: np.ndarray) -> np.ndarray:
        """Shell average of a real field.

        Args:
            field: Any real array on the Fourier grid.

        Returns:
            One value per shell.
        """
        total = np.bincount(self._flat[self._inside], weights=field.ravel()[self._inside], minlength=self.n_shells)[
            : self.n_shells
        ]
        return total / np.maximum(self.counts, 1)

    def fsc(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """Fourier shell correlation of two transforms.

        Args:
            first: Transform of the first map.
            second: Transform of the second map.

        Returns:
            One correlation per shell.
        """
        cross = self.mean(np.real(first * np.conj(second)))
        norm = np.sqrt(self.mean(np.abs(first) ** 2) * self.mean(np.abs(second) ** 2))
        return cross / np.maximum(norm, 1e-30)
