"""Particle images off disk, where the real-data arm starts.

`metadata` records a `stack_path` and `stack_index` per row but never opens an image; this module
follows them, opening each stack ONCE and returning the stack in the metadata's row order. A stored
path is relative to a project root the file does not name, so `root` is REQUIRED. Images come back
REAL and unshifted in stored order and units; the `fft2`, and any recentering, normalization or crop
before it, is the caller's. `read_micrograph` sits here because a micrograph is what `ctffit` fits.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
from jaxtyping import Float

from cryodisc._optional import require_mrcfile
from cryodisc.particles.metadata import ParticleMeta


def read_micrograph(path: str | Path) -> Float[np.ndarray, "y x"]:
    """One micrograph as a 2D float array, from `.mrc` or `.npy`.

    Returns:
        The image. A movie or multi-frame stack collapses to its FIRST FRAME, the right guess for the
        motion-corrected micrographs a CTF is fitted from.

    Raises:
        ImportError: If reading an `.mrc` and `mrcfile` is not installed.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        image = np.asarray(np.load(path), np.float32)
    else:
        mrcfile = require_mrcfile("reading a micrograph")
        with mrcfile.open(path, permissive=True) as handle:
            image = np.asarray(handle.data, np.float32)
    return image[0] if image.ndim == 3 else image


def load_stack(
    meta: ParticleMeta,
    root: str | Path,
    *,
    apply_sign: bool = True,
    expected_box_size: int | None = None,
) -> Float[np.ndarray, "n y x"]:
    """Read the particle images a metadata table points at.

    Args:
        meta: The table. Its `stack_path` and `stack_index` are followed row by row.
        root: Directory the stored paths are relative to; an absolute stored path is used as-is.
            Required (see the module docstring).
        apply_sign: Multiply each image by its `ParticleMeta.sign`, so the returned stack has one
            contrast convention. Scoring at the wrong sign anti-correlates every template with its
            own data and still produces a number.
        expected_box_size: Refuse a stack whose images are not this size. `None` uses `meta.box_size`
            when recorded, and checks nothing otherwise.

    Returns:
        `(n, y, x)` float32 images in `meta`'s row order.

    Raises:
        FileNotFoundError: If a stack is missing, naming the resolved path and root, since a wrong
            `root` is the common cause.
        ValueError: If a stack's images are the wrong size, or an index is off the end of its file.
        ImportError: If `mrcfile` is not installed.
    """
    mrcfile = require_mrcfile("reading particle images")
    box_size = expected_box_size if expected_box_size is not None else (meta.box_size or None)
    rows_of: dict[str, list[int]] = defaultdict(list)
    for row, path in enumerate(meta.stack_path):
        rows_of[str(path)].append(row)

    images: Float[np.ndarray, "n y x"] | None = None
    for stored, rows in rows_of.items():
        resolved = Path(stored) if Path(stored).is_absolute() else Path(root) / stored
        if not resolved.exists():
            raise FileNotFoundError(
                f"{resolved} does not exist. It came from {stored!r} in {meta.source!r}, resolved "
                f"against root={str(root)!r}; a wrong root is the usual cause and the stored path "
                "alone does not reveal one."
            )
        with mrcfile.mmap(resolved, mode="r", permissive=True) as handle:
            # a one-image .mrc comes back 2D; atleast_3d would append the axis rather than prepend it,
            # turning a 64x64 image into 64 images of 64x1
            stack = handle.data if handle.data.ndim == 3 else handle.data[None]
            wanted = np.asarray(meta.stack_index, int)[rows]
            # fancy indexing wraps a negative index to the end of the file, silently returning the
            # LAST image and shifting every row by one, so both ends are checked explicitly
            if wanted.min(initial=0) < 0:
                raise ValueError(f"frame index {wanted.min()} is negative; image indices are 1-based in the star")
            if wanted.max(initial=-1) >= stack.shape[0]:
                raise ValueError(
                    f"{resolved} holds {stack.shape[0]} images and {meta.source!r} asks for index "
                    f"{int(wanted.max())}. The metadata and the stacks are out of step."
                )
            if box_size is not None and stack.shape[-2:] != (box_size, box_size):
                raise ValueError(
                    f"{resolved} holds {stack.shape[-2]}x{stack.shape[-1]} images against the "
                    f"expected {box_size}x{box_size}."
                )
            block = np.asarray(stack[wanted], np.float32)
        if images is None:
            images = np.empty((len(meta), *block.shape[1:]), np.float32)
        images[rows] = block

    if images is None:
        raise ValueError(f"{meta.source!r} carries no particles to load")
    if apply_sign:
        images = images * np.asarray(meta.sign, np.float32)[:, None, None]
    return images
