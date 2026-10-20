"""A molecule as a grid: rendering one, reading one, banding it, and placing it where templates are.

`structures` holds atoms and `forward` projects them; neither reads a MAP, which is what a deposition
ships and a reconstruction produces. Everything here assumes a BOX-CENTERED map, voxel index `n // 2`
per axis at the `Frame` origin (floor division, not `n / 2`, differing by half a voxel at odd `n`). A
header placing the origin at a corner needs recentering first; `read_map` returns it so a caller can tell.

Rationale: docs/rationale/density.md
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from jaxtyping import Float

from cryodisc._optional import require_mrcfile

if TYPE_CHECKING:
    from cryodisc.forward.cryojax_model import CryoJAXModel

_FLIP = np.eye(3)[::-1]
"""Reverses a 3-vector, taking (x, y, z) to (z, y, x).

A rotation derived from atoms must be conjugated by this before it can move voxels; getting it wrong
is a transpose that still produces a map and reads as a failed alignment.
"""


def from_model(model: "CryoJAXModel") -> Float[np.ndarray, "z y x"]:
    """Render a forward model's molecule as a real-space map, on the model's own grid.

    The electrostatic potential, rendered analytically by cryoJAX from the Peng (1996) scattering
    factors. Only meaningful band-limited: low-pass before comparing anything to it. The box and pixel
    size are the model's own, keeping the map on one grid with the templates.

    Args:
        model: The forward model, from `ConformerPair.models` or `CryoJAXModel.from_structure`.

    Returns:
        `(box, box, box)` real map indexed (z, y, x) and box-centered at `box // 2`.

    Raises:
        TypeError: If the model's volume is not a Gaussian mixture -- a model built from a voxel grid
            is already a map, and re-rendering would silently resample it.

    Rationale: docs/rationale/density.md
    """
    import cryojax.simulator as cxs

    if not isinstance(model.volume, cxs.GaussianMixtureVolume):
        raise TypeError(
            f"from_model renders an ATOMIC model, and this one holds {type(model.volume).__name__}. "
            "A model built from a voxel grid already has its map; use that array rather than "
            "round-tripping it through a renderer."
        )
    box = int(model.image_config.shape[0])
    pixel = float(model.image_config.pixel_size)
    # `GaussianMixtureRenderFn` rather than `volume.to_real_voxel_grid`, which is deprecated and
    # removed in cryoJAX 0.6.0. The two are bit-identical on this version, so the switch is free.
    return np.asarray(cxs.GaussianMixtureRenderFn((box, box, box), pixel)(model.volume))


def _header_axes(header: Any) -> tuple[int, int, int]:
    """Which of x, y, z the column, row and section axes each run along.

    Args:
        header: The `mrcfile` header record.

    Returns:
        `(mapc, mapr, maps)`, or `(1, 2, 3)` for a header that does not name all three.
    """
    axes = (int(header.mapc), int(header.mapr), int(header.maps))
    return axes if sorted(axes) == [1, 2, 3] else (1, 2, 3)


def _to_zyx(volume: np.ndarray, header: Any) -> np.ndarray:
    """Put an MRC data array on (z, y, x), whatever axis order its header declares.

    `mrcfile` VALIDATES `mapc`/`mapr`/`maps` and never reorders, so `handle.data` is always indexed
    (section, row, column) -- (z, y, x) only for the usual `1, 2, 3`. `read_map`'s cubic check cannot
    see a permutation, every axis being the same length.

    Args:
        volume: The array as `mrcfile` hands it over, indexed (section, row, column).
        header: The `mrcfile` header record.

    Returns:
        The same data indexed (z, y, x), contiguous.

    Rationale: docs/rationale/density.md
    """
    # array dim 0/1/2 are section/row/column, running along physical axes[2]/axes[1]/axes[0]; invert
    # that to find where each physical axis lives, then read back in z, y, x order.
    axes = _header_axes(header)
    dimension_of = {axes[2]: 0, axes[1]: 1, axes[0]: 2}
    return np.ascontiguousarray(np.transpose(volume, (dimension_of[3], dimension_of[2], dimension_of[1])))


def header_pixel_size(path: Path, voxel_size: Any) -> float | None:
    """The single voxel size an MRC header records, refusing one that differs across axes.

    All three axes, because `read_map`'s cubic check constrains the SHAPE and says nothing about
    `cella`. See docs/rationale/density.md.

    Args:
        path: The file, for the message.
        voxel_size: The `mrcfile` voxel size record, in A.

    Returns:
        The voxel size in A, or `None` where the header records none.

    Raises:
        ValueError: If the three axes disagree.
    """
    sizes = np.array([voxel_size.x, voxel_size.y, voxel_size.z], float)
    if not np.any(sizes):
        return None
    if not np.allclose(sizes, sizes[0], rtol=1e-4, atol=0.0):
        raise ValueError(
            f"{path}: expected one voxel size, got (x, y, z) = "
            f"({sizes[0]:g}, {sizes[1]:g}, {sizes[2]:g}) A. Every operation here scales the three "
            "axes together, so an anisotropic map would be resampled to the wrong shape without "
            "complaint. Resample it to a cubic voxel first."
        )
    return float(sizes[0])


def _header_origin(header: Any, voxel_size: Any) -> np.ndarray:
    """The origin an MRC header records, in A as (x, y, z), from whichever convention carries it.

    MRC2014 sets one of two places: words 50-52 (`ORIGIN`) in angstroms, or words 5-7
    (`NXSTART`/`NYSTART`/`NZSTART`) as a column/row/section index. `ORIGIN` when non-zero, the start
    words otherwise -- never their sum; see docs/rationale/density.md.

    Args:
        header: The `mrcfile` header record.
        voxel_size: The `mrcfile` voxel size record, in A.

    Returns:
        `(3,)` origin in A as (x, y, z).
    """
    origin = np.array([header.origin.x, header.origin.y, header.origin.z], float)
    if np.any(origin):
        return origin
    # the start words are indexed by column/row/section, so `_header_axes` says which of x, y, z each
    # of those runs along.
    axes = _header_axes(header)
    start = np.zeros(3)
    for axis, value in zip(axes, (header.nxstart, header.nystart, header.nzstart), strict=True):
        start[axis - 1] = float(value)
    return start * np.array([voxel_size.x, voxel_size.y, voxel_size.z], float)


def refuse_a_corner_origin(
    path: str | Path, origin: np.ndarray | None, voxel: float, shape: tuple[int, ...], ignore: bool = False
) -> None:
    """Refuse a map whose header does not put it at the box center, under EITHER MRC convention.

    Everything downstream assumes a box-centered map, so a corner origin displaces every template from
    the particles and reads as a smaller signal rather than failing. Two headers say "box-centered" and
    disagree by half a box: `_header_origin` reads `ORIGIN` when set and falls back to `NXSTART *
    voxel`, and `NXSTART = -N/2` also means box-centered, so both readings are admitted.

    Args:
        path: The map, for the message.
        origin: The header origin in angstroms, or `None` if it records none.
        voxel: The map's own voxel size, which sets what "negligible" means.
        shape: The volume's shape in voxels, for the `NXSTART = -N/2` reading.
        ignore: Accept any origin, for a header known to be wrong.

    Raises:
        ValueError: If the origin is more than half a voxel from the box center under both readings.
    """
    if origin is None or ignore:
        return
    # `shape` is (z, y, x) and `origin` is (x, y, z), so the half-widths are reversed before comparing;
    # this keeps the two orders agreeing here rather than relying on the cubic check elsewhere.
    center = 0.5 * voxel * np.asarray(shape, dtype=float)[::-1]
    displacement = min(float(np.linalg.norm(origin)), float(np.linalg.norm(np.asarray(origin) + center)))
    if displacement > 0.5 * voxel:
        raise ValueError(
            f"{path} records an origin {np.round(np.asarray(origin), 2).tolist()} A that is "
            f"{displacement:.2f} A from the box center under both MRC readings (a zero origin and "
            f"nxstart = -N/2), and everything here assumes a box-centered map. Recenter it, or pass "
            "ignore_origin=True if the header is known to be wrong; a corner origin displaces every "
            "template from the particles and reads as a smaller signal rather than failing."
        )


def read_map(path: str | Path) -> tuple[np.ndarray, float | None, np.ndarray | None]:
    """Read a density from `.mrc`, `.map` or `.npy`.

    Args:
        path: The file.

    Returns:
        `(volume, pixel_size, origin)`, the volume indexed (z, y, x) whatever axis order the header
        declares (see `_to_zyx`). Both are `None` for a `.npy`. `pixel_size` is returned rather than
        defaulted: a wrong voxel size rescales every spatial frequency invisibly.

    Raises:
        ValueError: If the map is not cubic, or its voxel size differs across axes.
        ImportError: If reading an `.mrc` and `mrcfile` is not installed.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        volume, pixel_size, origin = np.load(path), None, None
    else:
        mrcfile = require_mrcfile("reading a map")
        with mrcfile.open(path, permissive=True) as handle:
            volume = _to_zyx(np.asarray(handle.data, np.float32), handle.header)
            pixel_size = header_pixel_size(path, handle.voxel_size)
            origin = _header_origin(handle.header, handle.voxel_size)
    volume = np.asarray(volume)
    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError(f"{path}: expected a cubic map, got shape {volume.shape}")
    return volume, pixel_size, origin


def write_map(
    path: str | Path,
    volume: Float[np.ndarray, "z y x"],
    pixel_size: float,
    *,
    origin: np.ndarray | None = None,
) -> Path:
    """Write a density as MRC with its voxel size and origin recorded.

    float32 for viewer compatibility; the voxel size because a map imported at the wrong one is the
    wrong SIZE. The ORIGIN travels too: without it a round trip turns a corner-origin map into one
    claiming to be box-centered, disarming `refuse_a_corner_origin`.

    Args:
        path: Destination.
        volume: Cubic map.
        pixel_size: A/voxel.
        origin: `(3,)` position of the first voxel in A as (x, y, z), from `read_map`. `None` writes
            a zero origin, which is the box-centered convention everything here renders in.

    Returns:
        The path written.

    Raises:
        ImportError: If `mrcfile` is not installed.
    """
    mrcfile = require_mrcfile("writing a map")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(destination, overwrite=True) as handle:
        handle.set_data(np.asarray(volume, np.float32))
        handle.voxel_size = (pixel_size, pixel_size, pixel_size)
        if origin is not None:
            # `ORIGIN` and not the start words: `_header_origin` prefers it when non-zero, and
            # writing the pair would let a reader that sums them cancel a real offset to zero
            handle.header.origin.x, handle.header.origin.y, handle.header.origin.z = (
                float(value) for value in np.asarray(origin, float).ravel()[:3]
            )
    return destination


def lowpass(volume: Float[np.ndarray, "z y x"], pixel_size: float, resolution: float) -> np.ndarray:
    """Isotropic Fourier low-pass with a raised-cosine edge.

    Raised rather than sharp because a hard cut rings, and the ringing is a spatial oscillation that
    anything fitting the map afterwards will fit as density. The taper runs from 0.8 to 1.0 of cutoff.

    Args:
        volume: Cubic map.
        pixel_size: A/voxel.
        resolution: Cutoff in A.

    Returns:
        The filtered map, real.

    Raises:
        ValueError: If the cutoff is past Nyquist, which filters nothing while reading as a band.
    """
    volume = np.asarray(volume, float)
    if resolution < 2.0 * pixel_size:
        raise ValueError(
            f"a {resolution} A low-pass is past the {2 * pixel_size:.2f} A Nyquist of a {pixel_size} "
            "A/voxel map, so it would filter nothing while reading as a stated band"
        )
    frequency = np.fft.fftfreq(volume.shape[0], pixel_size)
    radius = np.sqrt(frequency[:, None, None] ** 2 + frequency[None, :, None] ** 2 + frequency[None, None, :] ** 2)
    cutoff = 1.0 / resolution
    taper = 0.5 * (1.0 + np.cos(np.pi * np.clip((radius - 0.8 * cutoff) / (0.2 * cutoff), 0.0, 1.0)))
    return np.real(np.fft.ifftn(np.fft.fftn(volume) * taper))


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Pearson correlation between two maps.

    Args:
        first: A map.
        second: Another, on the same grid.

    Returns:
        The correlation, or NaN when either map is constant.
    """
    a = np.asarray(first, float).ravel()
    b = np.asarray(second, float).ravel()
    a, b = a - a.mean(), b - b.mean()
    scale = np.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / scale) if scale > 0.0 else float("nan")


def voxel_centroid(volume: Float[np.ndarray, "z y x"]) -> np.ndarray:
    """The density-weighted centroid of a map, in voxels, in ARRAY order.

    `scipy` rather than `np.indices`, which materializes one VOLUME-sized index array per axis to
    produce three scalars. Negatives are NOT clipped; that is the caller's, and
    `center_of_mass_separation` makes it.

    Args:
        volume: Map indexed (z, y, x). Densities are the weights.

    Returns:
        (3,) centroid in voxels, ordered (z, y, x) as the array is.
    """
    from scipy.ndimage import center_of_mass

    return np.asarray(center_of_mass(np.asarray(volume, float)), float)


def center_of_mass_offset(volume: np.ndarray, pixel_size: float) -> np.ndarray:
    """Where a map's density sits relative to its box center, in A and in the atom frame's axis order.

    The axis reversal is why this is a function: maps are indexed (z, y, x) and structures (x, y, z),
    so an offset measured in index order and applied to atoms has two axes swapped -- the right
    MAGNITUDE in the wrong direction. See docs/rationale/density.md.

    Args:
        volume: Cubic map, ordered (z, y, x). Densities are weights, so a map with negative voxels
            must be inverted first.
        pixel_size: Voxel size in A.

    Returns:
        (3,) offset in A, ordered (x, y, z), from the box center to the density's center of mass.

    Raises:
        ValueError: If the map is not cubic, or its densities sum to zero (center of mass undefined).
    """
    volume = np.asarray(volume, float)
    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError(f"expected a cubic map, got shape {volume.shape}")
    if volume.sum() == 0.0:
        raise ValueError("the map's densities sum to zero, so it has no center of mass")
    # index order (z, y, x) -> atom order (x, y, z)
    centers = voxel_centroid(volume)[::-1]
    # `// 2`, this module's convention and cryoJAX's rotation center at BOTH parities. NOT `(n - 1) / 2`,
    # which is half a voxel out on every even box; see docs/rationale/density.md
    return (centers - volume.shape[-1] // 2) * pixel_size
