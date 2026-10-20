"""Per-particle imaging metadata from real cryo-EM files, and the `Transfer` it builds.

The only reader of cryoSPARC `.cs` and RELION `.star` (3.1 two-block and older single-block forms).
Unlike `sim`, a deposited pixel size or defocus unit can be wrong, so `load_star` refuses to default
one and `ctffit` refits; the optics are astigmatic (a defocus PAIR); defocus belongs to the PARTICLE,
so `transfer_from_meta` groups by a phase tolerance whose bin width scales as `k_max^-2`. The
recentering sign is MEASURED; see `check_shift_sign`. Rationale: docs/rationale/particles/metadata.md
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float

from cryodisc.grids.fourier import frequency_grid, frequency_vectors, phase_ramp
from cryodisc.microscope import Transfer, ctf_astigmatic, wavelength

SHIFT_SIGN = +1.0
"""Sign taking a stored origin offset to the translation that recenters the particle.

Measured on real cryoSPARC output by `check_shift_sign`, not assumed: the stored value is the
CORRECTION, so the image moves by PLUS it. A sign error is invisible in every summary statistic and
DOUBLES the centering error it was meant to remove, so re-run `check_shift_sign` on any new dataset.
"""


@dataclass(frozen=True)
class PoolOptics:
    """One pool's imaging optics as the four scalars a CTF fit needs.

    Carried together from the table that knows them to the fitter, which otherwise DEFAULTS the three
    optics and would fit plausibly at silently wrong values. Medians, not means.

    Args:
        pixel_size: A/px of the images being fitted, NOT the particle box's when the extraction
            downsampled.
        voltage_in_kv: Accelerating voltage.
        spherical_aberration_in_mm: Cs.
        amplitude_contrast_ratio: Q.
    """

    pixel_size: float
    voltage_in_kv: float
    spherical_aberration_in_mm: float
    amplitude_contrast_ratio: float

    @classmethod
    def from_columns(
        cls,
        pixel_size: np.ndarray | float,
        voltage_in_kv: np.ndarray | float,
        spherical_aberration_in_mm: np.ndarray | float,
        amplitude_contrast_ratio: np.ndarray | float,
    ) -> "PoolOptics":
        """The median of each column, for a caller holding raw `.cs` fields rather than a table.

        `ParticleMeta.optics` is the route to prefer; this exists for a driver that read a `.cs`
        through `read_cs_array` without the `blob/*` fields `_load_cs` needs to build a `ParticleMeta`.

        Args:
            pixel_size: Per-particle pixel size in A, or a scalar override.
            voltage_in_kv: Per-particle accelerating voltage, or a scalar.
            spherical_aberration_in_mm: Per-particle Cs, or a scalar.
            amplitude_contrast_ratio: Per-particle Q, or a scalar.

        Returns:
            The four medians.
        """
        return cls(
            pixel_size=float(np.median(pixel_size)),
            voltage_in_kv=float(np.median(voltage_in_kv)),
            spherical_aberration_in_mm=float(np.median(spherical_aberration_in_mm)),
            amplitude_contrast_ratio=float(np.median(amplitude_contrast_ratio)),
        )

    def as_kwargs(self) -> dict[str, float]:
        """The four as `ctffit.fit_patches` keyword arguments.

        Returns:
            `{"pixel_size", "voltage_in_kv", "spherical_aberration_in_mm", "amplitude_contrast_ratio"}`,
            so a call site reads `fit_patches(image, **optics.as_kwargs())` and cannot drop one.
        """
        return {
            "pixel_size": self.pixel_size,
            "voltage_in_kv": self.voltage_in_kv,
            "spherical_aberration_in_mm": self.spherical_aberration_in_mm,
            "amplitude_contrast_ratio": self.amplitude_contrast_ratio,
        }


@dataclass(frozen=True)
class ParticleMeta:
    """Per-particle imaging parameters, one entry per row of the particle stack.

    Angles in radians, lengths in angstroms, both formats converted on read. `defocus_u >= defocus_v`
    is NOT enforced; neither format guarantees it and `microscope.ctf_astigmatic` does not need it.

    Args:
        defocus_u: (n,) underfocus along the astigmatism axis in A, positive.
        defocus_v: (n,) underfocus along the perpendicular axis in A.
        astigmatism_angle: (n,) azimuth of `defocus_u` in radians.
        phase_shift: (n,) phase-plate shift in radians.
        voltage_in_kv: (n,) accelerating voltage.
        spherical_aberration_in_mm: (n,) Cs.
        amplitude_contrast_ratio: (n,) Q.
        b_factor: (n,) per-particle CTF envelope B in A^2, zero when the file carries none.
        pixel_size: (n,) A/px of the STORED particle box, not necessarily the refinement's.
        box_size: Image size in pixels, zero when the file does not say.
        shift: (n, 2) refined origin offset as (y, x) in ANGSTROMS, zeros when absent. Ordered (y, x)
            to match `fourier.phase_ramp` though both formats store (x, y) -- swapped once, at the reader.
        has_shift: Whether `shift` came from the file or is a placeholder.
        euler: (n, 3) refined orientation as RELION's `(rot, tilt, psi)` in DEGREES, None when absent.
            Stored as the file spells it, NOT converted; `poses.euler_from_relion` maps it into
            `rotation_of`'s convention via one measured negation.
        has_euler: Whether `euler` came from the file or is a placeholder.
        sign: (n,) stored-image contrast sign flag, from cryoSPARC's `blob/sign`.
        stack_path: (n,) path of the .mrcs holding each particle.
        stack_index: (n,) 0-based frame within that file.
        source: The file this was read from.
    """

    defocus_u: np.ndarray
    defocus_v: np.ndarray
    astigmatism_angle: np.ndarray
    phase_shift: np.ndarray
    voltage_in_kv: np.ndarray
    spherical_aberration_in_mm: np.ndarray
    amplitude_contrast_ratio: np.ndarray
    b_factor: np.ndarray
    pixel_size: np.ndarray
    box_size: int
    shift: np.ndarray
    has_shift: bool
    sign: np.ndarray
    stack_path: np.ndarray
    stack_index: np.ndarray
    source: str
    euler: np.ndarray | None = None
    has_euler: bool = False

    _PER_ROW = (
        "defocus_u",
        "defocus_v",
        "astigmatism_angle",
        "phase_shift",
        "voltage_in_kv",
        "spherical_aberration_in_mm",
        "amplitude_contrast_ratio",
        "b_factor",
        "pixel_size",
        "shift",
        "euler",
        "sign",
        "stack_path",
        "stack_index",
    )

    def __len__(self) -> int:
        """Number of particles."""
        return len(self.defocus_u)

    def __getitem__(self, selection: Any) -> "ParticleMeta":
        """Row selection, so a metadata table can follow a particle subset.

        Args:
            selection: Anything numpy indexes rows with.

        Returns:
            The same metadata restricted to those particles. `box_size` and `source` are carried
            through unchanged, being properties of the file rather than of a row.
        """
        # filter before subscripting: `euler` is optional, and `None[selection]` would raise on every
        # `.cs` pool and every star without rlnAngle* columns
        present = [name for name in self._PER_ROW if getattr(self, name) is not None]
        return replace(self, **{name: getattr(self, name)[selection] for name in present})

    def with_ctf(
        self,
        *,
        defocus_u: np.ndarray,
        defocus_v: np.ndarray,
        astigmatism_angle: np.ndarray | None = None,
    ) -> "ParticleMeta":
        """The same particles with their defocus replaced by a refit one.

        The seam between `ctffit` and everything that scores; `interpolate_to_particles` returns exactly
        these three fields.

        Args:
            defocus_u: (n,) refit underfocus along the astigmatism axis, in A.
            defocus_v: (n,) refit underfocus along the perpendicular axis, in A.
            astigmatism_angle: (n,) refit azimuth in RADIANS, or `None` to keep the stored one.
                Radians because that is what this table holds; `ctffit` already works in them.

        Returns:
            A new table. Everything else -- voltage, Cs, Q, phase shift, B, pixel size, shifts, stack
            paths -- is carried through, a defocus refit being a statement about the defocus alone.

        Raises:
            ValueError: If a replacement is not one value per particle. A refit of the wrong length
                would otherwise broadcast or truncate into a table that still looks well formed.
        """
        replacements: dict[str, Any] = {"defocus_u": defocus_u, "defocus_v": defocus_v}
        if astigmatism_angle is not None:
            replacements["astigmatism_angle"] = astigmatism_angle
        for name, value in replacements.items():
            array = np.asarray(value, float)
            if array.shape != (len(self),):
                raise ValueError(f"{name} must be one value per particle, got {array.shape} against {len(self)}")
            replacements[name] = array
        return replace(self, **replacements)

    @property
    def defocus(self) -> np.ndarray:
        """(n,) mean defocus `(defocus_u + defocus_v) / 2` in A, the isotropic scalar."""
        return 0.5 * (self.defocus_u + self.defocus_v)

    def optics(self, pixel_size: float | None = None) -> PoolOptics:
        """This pool's optics as the four scalars a CTF fit needs. See `PoolOptics`.

        Args:
            pixel_size: Pixel size in A of the images that will be FITTED. `None` takes the median
                stored one, which is right for the particle boxes and wrong for the micrographs they
                were extracted from when the extraction downsampled.

        Returns:
            The medians.
        """
        return PoolOptics.from_columns(
            self.pixel_size if pixel_size is None else pixel_size,
            self.voltage_in_kv,
            self.spherical_aberration_in_mm,
            self.amplitude_contrast_ratio,
        )

    def shift_in_pixels(self, pixel_size: float | None = None) -> np.ndarray:
        """(n, 2) refined origin offset as (y, x) in pixels.

        Args:
            pixel_size: Pixel size of the box the shift is wanted in. `None` uses each particle's
                own stored pixel size, which is right when the images have not been rescaled.

        Returns:
            The offset in pixels, in `fourier.phase_ramp`'s (y, x) order. See `SHIFT_SIGN` for what
            still has to be applied before it recenters anything.
        """
        scale = self.pixel_size if pixel_size is None else np.full(len(self), float(pixel_size))
        return self.shift / scale[:, None]


def _structured_field(data: np.ndarray, name: str, default: float, n: int) -> np.ndarray:
    """One field of a structured array, or a constant `(n,)` column when the file lacks it."""
    return np.asarray(data[name], float) if name in (data.dtype.names or ()) else np.full(n, default)


def read_cs_array(path: str | Path, *, passthrough: str | Path | None = None) -> np.ndarray:
    """A cryoSPARC `.cs` as the structured array it is, with any passthrough joined on.

    A cryoSPARC job splits its output: `*_particles.cs` carries the fields that job produced, the rest
    sits in a `*_passthrough_*.cs` with the same rows. The join is on `uid`, not position, so a
    reordered output cannot silently mismatch.

    Args:
        path: The `.cs` file, a `.npy` holding a structured array.
        passthrough: Optional passthrough file to join in.

    Returns:
        The structured array, with the passthrough's extra fields appended.

    Raises:
        ValueError: If the passthrough does not cover the same particles.
    """
    data = np.load(path, allow_pickle=False)
    if passthrough is None:
        return data
    extra_data = np.load(passthrough, allow_pickle=False)
    # `x[argsort(x)]` IS `sort(x)`, so the coverage check and the join reuse the same two permutations
    # rather than sorting each side twice
    extra_order, data_order = np.argsort(extra_data["uid"]), np.argsort(data["uid"])
    if len(extra_data) != len(data) or not np.array_equal(extra_data["uid"][extra_order], data["uid"][data_order]):
        raise ValueError(f"passthrough {passthrough} does not cover the same particles as {path}")
    missing = [f for f in (extra_data.dtype.names or ()) if f not in (data.dtype.names or ())]
    if not missing:
        return data
    # inverting a permutation is a scatter, not another sort: `data_order` says where each sorted row
    # came FROM, so writing positions back through it says where each unsorted row GOES
    inverse = np.empty(len(data_order), int)
    inverse[data_order] = np.arange(len(data_order))
    extra_data = extra_data[extra_order[inverse]]
    # allocated and filled rather than recfunctions.append_fields, which raises OverflowError on any
    # unsigned field (every cryoSPARC table has one, e.g. blob/idx) via its masked-array fill value
    joined = np.empty(len(data), dtype=data.dtype.descr + [f for f in extra_data.dtype.descr if f[0] in missing])
    for name in data.dtype.names or ():
        joined[name] = data[name]
    for name in missing:
        joined[name] = extra_data[name]
    return joined


CS_POSE_TRANSPOSE = True
"""Whether cryoSPARC's `alignments3D/pose` needs transposing to reach the RELION reading.

Taken from pyem, and confirmed by scoring a pool both ways against its images -- the transpose wins.
A wrong value does not raise: it renders every template at a plausible but unrelated orientation,
which collapses the fitted contrast toward zero and takes the coherence with it.
"""


def relion_euler_from_cs_pose(
    pose: Float[np.ndarray, "n 3"], *, transpose: bool = CS_POSE_TRANSPOSE
) -> Float[np.ndarray, "n 3"]:
    """cryoSPARC's axis-angle `alignments3D/pose` as RELION's `(rot, tilt, psi)` in degrees.

    Rodrigues to a matrix, matrix to intrinsic ZYZ, then the negation `euler_from_relion` undoes, so a
    `.cs` pool arrives in the same convention a star pool does. Derived, not read -- the file carries
    no RELION triple.

    Args:
        pose: `(n, 3)` rotation vectors, `alignments3D/pose`.
        transpose: Use the transpose of the Rodrigues matrix. `Rodrigues(-v) == Rodrigues(v).T`, the
            only genuine degree of freedom here. See `CS_POSE_TRANSPOSE`, which is measured.

    Returns:
        `(n, 3)` of `(rot, tilt, psi)` in degrees.
    """
    vector = np.asarray(pose, float).reshape(-1, 3)
    angle = np.linalg.norm(vector, axis=1)
    # the axis is arbitrary where the angle vanishes, so guard the division rather than let a nan
    # propagate through every downstream template
    safe = np.where(angle > 1e-12, angle, 1.0)
    axis = vector / safe[:, None]
    cos, sin = np.cos(angle), np.sin(angle)
    skew = np.zeros((len(vector), 3, 3))
    skew[:, 0, 1], skew[:, 0, 2] = -axis[:, 2], axis[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = axis[:, 2], -axis[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -axis[:, 1], axis[:, 0]
    identity = np.broadcast_to(np.eye(3), (len(vector), 3, 3))
    matrix = identity + sin[:, None, None] * skew + (1.0 - cos)[:, None, None] * (skew @ skew)
    matrix = np.where((angle > 1e-12)[:, None, None], matrix, identity)
    if transpose:
        matrix = np.swapaxes(matrix, 1, 2)

    # `rotation_of((a, b, c))` builds `Rz(c) Ry(b) Rz(a)` -- angles in REVERSE order (measured, not
    # read off the name); the textbook order would swap the first and third angles
    theta = np.arccos(np.clip(matrix[:, 2, 2], -1.0, 1.0))
    phi = np.arctan2(matrix[:, 2, 1], -matrix[:, 2, 0])
    psi = np.arctan2(matrix[:, 1, 2], matrix[:, 0, 2])
    # at the poles (theta = 0 or pi) the axis is degenerate, so put the whole rotation in phi, psi = 0.
    # The two poles need DIFFERENT algebra (residual Rz(phi) vs Ry(pi) Rz(phi)), handled separately
    near_zero = theta < 1e-7
    near_pi = theta > np.pi - 1e-7
    if near_zero.any():
        phi = np.where(near_zero, np.arctan2(matrix[:, 1, 0], matrix[:, 0, 0]), phi)
        psi = np.where(near_zero, 0.0, psi)
    if near_pi.any():
        phi = np.where(near_pi, np.arctan2(matrix[:, 0, 1], -matrix[:, 0, 0]), phi)
        psi = np.where(near_pi, 0.0, psi)
    return -np.degrees(np.stack([phi, theta, psi], axis=1))


def _load_cs(
    path: str | Path, *, passthrough: str | Path | None = None, pose_transpose: bool | None = None
) -> ParticleMeta:
    """Read a cryoSPARC `.cs` particle file.

    Args:
        path: The `.cs` file, which is a `.npy` holding a structured array.
        passthrough: Optional passthrough file to join in; see `read_cs_array`.
        pose_transpose: Override `CS_POSE_TRANSPOSE` for this file. Exposed so that
            `score_pool.py --cs-pose-transpose` can score both readings against the images, which is
            how the constant came to be measured rather than inherited.

    Returns:
        The metadata.

    Raises:
        ValueError: If the passthrough does not cover the same particles.
    """
    data = read_cs_array(path, passthrough=passthrough)
    n = len(data)

    fields = data.dtype.names or ()
    # the refinement's shift is in pixels of the box THAT JOB ran in (often downsampled), so it
    # converts to angstroms through its own psize and not through the stack's
    shift, has_shift = np.zeros((n, 2)), False
    for group in ("alignments3D", "alignments2D"):
        if f"{group}/shift" in fields:
            job_pixel = np.asarray(data[f"{group}/psize_A"], float)
            stored = np.asarray(data[f"{group}/shift"], float) * job_pixel[:, None]
            shift, has_shift = stored[:, ::-1], True  # (x, y) on disk, (y, x) here
            break

    # 3D only: alignments2D has an in-plane angle but no viewing direction, so reading it as a full
    # pose would silently fix two of the three angles at zero
    euler, has_euler = None, False
    if "alignments3D/pose" in fields:
        transpose = CS_POSE_TRANSPOSE if pose_transpose is None else pose_transpose
        euler = relion_euler_from_cs_pose(np.asarray(data["alignments3D/pose"], float), transpose=transpose)
        has_euler = True

    return ParticleMeta(
        defocus_u=np.asarray(data["ctf/df1_A"], float),
        defocus_v=np.asarray(data["ctf/df2_A"], float),
        astigmatism_angle=np.asarray(data["ctf/df_angle_rad"], float),
        phase_shift=_structured_field(data, "ctf/phase_shift_rad", 0.0, n),
        voltage_in_kv=np.asarray(data["ctf/accel_kv"], float),
        spherical_aberration_in_mm=np.asarray(data["ctf/cs_mm"], float),
        amplitude_contrast_ratio=np.asarray(data["ctf/amp_contrast"], float),
        b_factor=_structured_field(data, "ctf/bfactor", 0.0, n),
        pixel_size=np.asarray(data["blob/psize_A"], float),
        box_size=int(np.asarray(data["blob/shape"])[0][0]),
        shift=shift,
        has_shift=has_shift,
        sign=_structured_field(data, "blob/sign", 1.0, n),
        stack_path=np.asarray([p.decode() if isinstance(p, bytes) else str(p) for p in data["blob/path"]]),
        stack_index=np.asarray(data["blob/idx"], int),
        source=str(path),
        euler=euler,
        has_euler=has_euler,
    )


def _parse_star(path: str | Path) -> dict[str, tuple[list[str], list[list[str]]]]:
    """Minimal RELION star reader, loop blocks only.

    Args:
        path: The star file.

    Returns:
        `{block_name: (column_names, rows)}`, where the leading `data_` is stripped from each name
        and an unnamed `data_` block is called `images`.
    """
    blocks: dict[str, tuple[list[str], list[list[str]]]] = {}
    name: str | None = None
    columns: list[str] = []
    rows: list[list[str]] = []
    in_loop = False
    with open(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("data_"):
                if name is not None:
                    blocks[name] = (columns, rows)
                name, columns, rows, in_loop = line[5:] or "images", [], [], False
            elif line == "loop_":
                in_loop = True
            elif line.startswith("_"):
                columns.append(line.split()[0][1:])
            elif in_loop:
                rows.append(line.split())
    if name is not None:
        blocks[name] = (columns, rows)
    return blocks


def load_star(path: str | Path) -> ParticleMeta:
    """Read a RELION particle star, in either the 3.1 or the 1.x/3.0 form.

    The 3.1 form has `data_optics` + `data_particles`, with optics-group columns broadcast per
    particle through `rlnOpticsGroup`. The older single-block `data_` form carries the optics per
    particle; there the pixel size is `rlnDetectorPixelSize` (micrometres) over `rlnMagnification`
    rather than `rlnImagePixelSize`, and the origin offsets are in PIXELS rather than angstroms.

    Args:
        path: The star file.

    Returns:
        The metadata.

    Raises:
        ValueError: If there is no unambiguous particle block, if it is empty, if the pixel size
            cannot be determined, or if the optics groups disagree about the box size.
    """
    blocks = _parse_star(path)
    if "particles" in blocks:
        columns, rows = blocks["particles"]
    else:
        loops = [value for value in blocks.values() if value[1]]
        if len(loops) != 1:
            raise ValueError(
                f"{path}: no data_particles block, and {len(loops)} candidate single-block loops "
                "(the RELION 1.x/3.0 form has exactly one)"
            )
        columns, rows = loops[0]
    if not rows:
        raise ValueError(f"{path}: the particle block is empty")
    position = {name: i for i, name in enumerate(columns)}
    n = len(rows)

    def column(name: str) -> np.ndarray | None:
        """The named float column, or `None` when the file does not carry it."""
        if name not in position:
            return None
        return np.array([row[position[name]] for row in rows], float)

    def column_or(name: str, default: float) -> np.ndarray:
        """The named float column, or a constant one, so a default-supplying caller gets an array."""
        found = column(name)
        return np.full(n, default) if found is None else found

    def text_column(name: str) -> np.ndarray | None:
        if name not in position:
            return None
        return np.array([row[position[name]] for row in rows])

    optics: dict[int, dict[str, str]] = {}
    if "optics" in blocks:
        optics_columns, optics_rows = blocks["optics"]
        optics_position = {name: i for i, name in enumerate(optics_columns)}
        for row in optics_rows:
            group = int(row[optics_position["rlnOpticsGroup"]])
            optics[group] = {name: row[optics_position[name]] for name in optics_columns}
    # not `column(...) or np.ones(n)`: the truth value of an array is ambiguous and raises, and the
    # default already guarantees a column comes back
    group_of = column_or("rlnOpticsGroup", 1.0).astype(int)

    def optics_column(name: str, default: float) -> np.ndarray:
        """A per-particle column if present, else the per-group optics value, else `default`.

        A particle whose `rlnOpticsGroup` is absent from `data_optics` is REFUSED rather than left at
        `default`: a renumbered, 0-based, or dropped group would otherwise send rows to silently wrong
        optics while their neighbors read theirs from the file.
        """
        per_particle = column(name)
        if per_particle is not None:
            return per_particle
        defined = {group for group, row in optics.items() if name in row}
        if defined:
            missing = sorted(set(np.unique(group_of).tolist()) - defined)
            if missing:
                raise ValueError(
                    f"{path}: optics group(s) {missing} carry no {name}, and {len(defined)} other "
                    f"group(s) do. Defaulting one particle's optics to {default} while its neighbors "
                    "read theirs from the file is invisible downstream."
                )
        out = np.full(n, float(default))
        for group, row in optics.items():
            if name in row:
                out[group_of == group] = float(row[name])
        return out

    # a wrong pixel size rescales every spatial frequency and is invisible downstream, so there is
    # no fallback to 1.0 here
    has_explicit = "rlnImagePixelSize" in position or any("rlnImagePixelSize" in r for r in optics.values())
    if has_explicit:
        pixel_size = optics_column("rlnImagePixelSize", 1.0)
    else:
        detector = optics_column("rlnDetectorPixelSize", 0.0)  # micrometres
        magnification = optics_column("rlnMagnification", 0.0)
        if not ((detector > 0).all() and (magnification > 0).all()):
            raise ValueError(
                f"{path}: cannot determine the pixel size. Needs either rlnImagePixelSize, or both "
                "rlnDetectorPixelSize (um) and rlnMagnification."
            )
        pixel_size = detector * 1e4 / magnification

    boxes = {int(float(row["rlnImageSize"])) for row in optics.values() if "rlnImageSize" in row}
    if len(boxes) > 1:
        raise ValueError(f"{path}: mixed box sizes {sorted(boxes)}; split by optics group first")

    stack_index, stack_path = np.zeros(n, int), np.array([""] * n, dtype=object)
    names = text_column("rlnImageName")
    if names is not None:
        for i, entry in enumerate(names):
            frame, _, filename = str(entry).partition("@")
            if filename:
                stack_index[i], stack_path[i] = int(frame) - 1, filename  # star frames are 1-based
            else:
                stack_path[i] = entry

    has_shift = "rlnOriginXAngst" in position or "rlnOriginYAngst" in position
    if has_shift:
        offset_x, offset_y = column_or("rlnOriginXAngst", 0.0), column_or("rlnOriginYAngst", 0.0)
    else:
        has_shift = "rlnOriginX" in position or "rlnOriginY" in position
        offset_x = column_or("rlnOriginX", 0.0) * pixel_size
        offset_y = column_or("rlnOriginY", 0.0) * pixel_size

    has_euler = all(f"rlnAngle{name}" in position for name in ("Rot", "Tilt", "Psi"))
    euler = np.stack([column_or(f"rlnAngle{name}", 0.0) for name in ("Rot", "Tilt", "Psi")], 1) if has_euler else None

    defocus_u = column_or("rlnDefocusU", 0.0)
    defocus_v = column("rlnDefocusV")
    return ParticleMeta(
        defocus_u=defocus_u,
        defocus_v=defocus_u.copy() if defocus_v is None else defocus_v,
        astigmatism_angle=np.radians(column_or("rlnDefocusAngle", 0.0)),  # star records DEGREES
        phase_shift=np.radians(column_or("rlnPhaseShift", 0.0)),
        voltage_in_kv=optics_column("rlnVoltage", 300.0),
        spherical_aberration_in_mm=optics_column("rlnSphericalAberration", 2.7),
        amplitude_contrast_ratio=optics_column("rlnAmplitudeContrast", 0.1),
        b_factor=column_or("rlnCtfBfactor", 0.0),
        pixel_size=pixel_size,
        box_size=(boxes.pop() if boxes else 0),
        shift=np.stack([offset_y, offset_x], 1),  # (y, x), the order everything downstream uses
        has_shift=has_shift,
        euler=euler,
        has_euler=has_euler,
        sign=np.ones(n),
        stack_path=stack_path.astype(str),
        stack_index=stack_index,
        source=str(path),
    )


def load(path: str | Path, **kwargs: Any) -> ParticleMeta:
    """Read particle metadata, dispatching on the file extension.

    Args:
        path: A `.cs` or `.star` file.
        **kwargs: Passed to the reader.

    Returns:
        The metadata.

    Raises:
        ValueError: If the extension is neither.
    """
    extension = os.path.splitext(str(path))[1].lower()
    if extension == ".cs":
        return _load_cs(path, **kwargs)
    if extension == ".star":
        return load_star(path, **kwargs)
    raise ValueError(f"{path}: expected a .cs or .star particle metadata file")


def select_states(
    meta: ParticleMeta, states: Sequence[str] | None, *, limit: int = 0
) -> tuple[ParticleMeta, np.ndarray | None]:
    """The particles belonging to the named states, and which state each one is.

    A benchmark names each stack for the structure that generated it, so the state label IS the file
    name. Shared by two drivers, so the row filter is defined once here rather than spelled twice.

    Args:
        meta: The particle table.
        states: File-name prefixes, in the order that fixes which is state A. `None` or empty selects
            everything and labels nothing.
        limit: Keep at most this many particles of EACH state. Zero keeps all of them.

    Returns:
        `(meta, labels)`. `labels` is the per-particle state index when more than one state was named,
        `None` otherwise -- one state is a pool of known COMPOSITION with nothing for a label to
        distinguish.

    Raises:
        ValueError: If a named state matches no stack. A prefix matching nothing is a typo or the
            wrong file, and silently selecting zero of it reads as a pool that would not separate.
    """
    labels = None
    if states:
        labels = _state_labels(meta, list(states))
        for index, state in enumerate(states):
            if not (labels == index).any():
                stacks = sorted({Path(name).name for name in meta.stack_path})
                raise ValueError(
                    f"--state {state!r} matches none of this pool's {len(stacks)} stacks. It is a "
                    f"file-name prefix, not a label column; the first few are {stacks[:4]}"
                )
        keep = labels >= 0
        meta, labels = meta[keep], labels[keep] if len(states) > 1 else None
    if limit:
        keep = _limit_per_state(labels, limit, n_particles=len(meta), n_states=len(states) if states else 1)
        meta = meta[keep]
        if labels is not None:
            labels = labels[keep]
    return meta, labels


def _state_labels(meta: ParticleMeta, states: list[str]) -> np.ndarray:
    """Which requested state each particle's stack belongs to, or -1 for none of them.

    Args:
        meta: The particle table, read for `stack_path`.
        states: File-name prefixes, in the order that fixes which is state A.

    Returns:
        One index per particle into `states`, and -1 where the stack matches no requested state.
        Later states win a tie, which cannot arise for the fixed-width names a benchmark ships.
    """
    labels = np.full(len(meta), -1)
    names = [Path(name).name for name in meta.stack_path]
    for index, state in enumerate(states):
        labels = np.where([name.startswith(f"{state}_") for name in names], index, labels)
    return labels


def _limit_per_state(labels: np.ndarray | None, limit: int, *, n_particles: int, n_states: int) -> np.ndarray:
    """A `--limit` mask that takes the first `limit` particles of EACH state, not of the pool.

    A benchmark's star orders its stacks one after another, so a flat truncation of a two-state
    selection keeps only the first state's particles and labels them as two. `n_particles` is passed
    in rather than inferred from `labels`, which is `None` exactly where there are no labels, so the
    mask is always pool-long rather than `limit`-long.

    Args:
        labels: Per-particle state index, or `None` when no state selection was made.
        limit: Particles to keep per state.
        n_particles: Rows in the pool the mask is for.
        n_states: How many states were requested.

    Returns:
        A boolean mask over the particles, always `n_particles` long.
    """
    mask = np.zeros(n_particles, bool)
    if labels is None or n_states < 2:
        mask[:limit] = True
        return mask
    for index in range(n_states):
        mask[np.flatnonzero(labels == index)[:limit]] = True
    return mask


def defocus_bin_width(k_max: float, voltage_in_kv: float = 300.0, phase_tolerance: float = 0.1) -> float:
    """Defocus quantum in A whose CTF phase error at `k_max` stays under `phase_tolerance`.

    `chi = pi lambda dz k^2`, so `d chi / d dz = pi lambda k^2` and the width is the tolerance over
    that: two particles closer than this in defocus cannot be told apart by a score stopping at
    `k_max`, so they may share a transfer row. It scales as `k_max^-2`, which is why the band is a
    required argument rather than Nyquist.

    Args:
        k_max: Highest spatial frequency the score reaches, in inverse A.
        voltage_in_kv: Accelerating voltage.
        phase_tolerance: Allowed CTF phase departure in radians at `k_max`.

    Returns:
        The bin width in A.

    Raises:
        ValueError: If `k_max` is not positive.
    """
    if k_max <= 0.0:
        raise ValueError(f"k_max must be positive, got {k_max}")
    return float(phase_tolerance / (np.pi * float(wavelength(voltage_in_kv)) * float(k_max) ** 2))


def _defocus_basis(meta: ParticleMeta) -> tuple[np.ndarray, float]:
    """The `(mean, Ac, As)` coordinates the binning rounds, and the voltage the bin width comes from.

    `transfer_from_meta` bins in this basis. Built once here so the median voltage, which both
    `defocus_bin_width` and the phase scale need, is not recomputed per caller.

    Args:
        meta: The metadata.

    Returns:
        `((n, 3)` array of `(mean_defocus, Ac, As)` in A, median voltage in kV`)`.
    """
    astigmatism = 0.5 * (meta.defocus_u - meta.defocus_v)
    smooth = np.stack(
        [
            meta.defocus,
            astigmatism * np.cos(2.0 * meta.astigmatism_angle),
            astigmatism * np.sin(2.0 * meta.astigmatism_angle),
        ],
        1,
    )
    return smooth, float(np.median(meta.voltage_in_kv))


def band_limit_envelope(box_size: int, pixel_size: float, band_limit: float) -> Float[Array, "y x"]:
    """A hard top-hat that stops a score at `band_limit`, on the unshifted fft grid.

    This is what `k_max` is often mistaken for: `k_max` sets the DEFOCUS GROUPING only, whereas a
    score that should stop at a frequency has to be stopped HERE, by an envelope the transfer carries.
    Hard rather than tapered on purpose -- a projection onto the modes the data supports makes the
    remaining score a likelihood on that subspace, and the whitening already weights by noise.

    Args:
        box_size: Image size in pixels.
        pixel_size: A/px.
        band_limit: Highest spatial frequency to keep, in inverse A. At or above Nyquist keeps the
            whole circle and still drops the fft CORNERS, which are above Nyquist off the axes.

    Returns:
        A real `(box, box)` mask, one inside the band and zero outside.

    Raises:
        ValueError: If `band_limit` is not positive.
    """
    if band_limit <= 0.0:
        raise ValueError(f"band_limit must be positive, got {band_limit}")
    frequency_y, frequency_x = frequency_vectors(box_size, pixel_size)
    radius = jnp.sqrt(frequency_y**2 + frequency_x**2)
    return (radius <= band_limit).astype(float)


def transfer_from_meta(
    meta: ParticleMeta,
    box_size: int | None = None,
    pixel_size: float | None = None,
    *,
    k_max: float,
    phase_tolerance: float = 0.1,
    envelope: Float[Array, "y x"] | None = None,
    exact: bool = False,
) -> Transfer:
    """The signed per-particle transfer, grouped to a phase tolerance.

    Particles are binned so no member's CTF phase departs from its row's by more than
    `phase_tolerance` at `k_max`. The defocus a mode at azimuth `theta` sees is

        dz(theta) = mean + Ac cos(2 theta) + As sin(2 theta)

    LINEAR in `(mean, Ac, As)`, so rounding each to `defocus_bin_width` bounds the phase error.
    Shape-changing optics -- voltage, Cs, amplitude contrast, phase plate, B -- join the key exactly
    rather than by tolerance.

    Args:
        meta: The metadata.
        box_size: Image size in pixels. `None` takes `meta.box_size`.
        pixel_size: Pixel size in A. `None` takes the median of `meta.pixel_size`.
        k_max: Highest spatial frequency the score reaches, in inverse A. Stated rather than
            defaulted to Nyquist: it sets the group count quadratically.
        phase_tolerance: Allowed CTF phase departure in radians at `k_max`.
        envelope: Extra real envelope applied to every row (DQE, dose damage, Debye-Waller). Goes
            here because consumers expect the transfer to carry the WHOLE response and would square a
            separately-applied one.
        exact: Skip the binning and give one row per distinct parameter tuple, which on a patch-CTF
            set is one row per particle. For measuring what the binning costs, not for running.

    Returns:
        The transfer and each particle's group, ready for the ladder and for `em`.

    Raises:
        ValueError: If the box size is unknown, or if `meta` is empty.
    """
    if len(meta) == 0:
        raise ValueError("the metadata is empty; there is nothing to group")
    box = meta.box_size if box_size is None else box_size
    if box < 1:
        raise ValueError(
            "the box size is unknown -- the metadata file did not record one, so pass `box_size` "
            "explicitly rather than letting a transfer be built on a guess."
        )
    pixel = float(np.median(meta.pixel_size)) if pixel_size is None else pixel_size

    smooth, voltage = _defocus_basis(meta)
    if exact:
        quantized = smooth
    else:
        width = defocus_bin_width(k_max, voltage, phase_tolerance)
        quantized = np.rint(smooth / width) * width
    # rounded only to kill float noise, not to bin: these change the transfer's shape, and two
    # voltages that differ at all are two different optics
    shape_changing = np.stack(
        [
            meta.voltage_in_kv,
            meta.spherical_aberration_in_mm,
            meta.amplitude_contrast_ratio,
            meta.phase_shift,
            meta.b_factor,
        ],
        1,
    )
    key = np.concatenate([quantized, np.round(shape_changing, 9)], 1)
    unique, index = np.unique(key, axis=0, return_inverse=True)

    frequency_y, frequency_x = frequency_vectors(box, pixel)
    damping = 1.0 if envelope is None else jnp.asarray(envelope)
    rows = []
    for defocus_mean, cosine_part, sine_part, voltage, cs, contrast, phase, b in unique:
        half_spread = float(np.hypot(cosine_part, sine_part))
        rows.append(
            ctf_astigmatic(
                frequency_y,
                frequency_x,
                defocus_u=defocus_mean + half_spread,
                defocus_v=defocus_mean - half_spread,
                astigmatism_angle=0.5 * float(np.arctan2(sine_part, cosine_part)),
                voltage_in_kv=float(voltage),
                spherical_aberration_in_mm=float(cs),
                amplitude_contrast_ratio=float(contrast),
                phase_shift=float(phase),
                b_factor=float(b),
            )
            * damping
        )
    return Transfer(values=jnp.stack(rows), index=jnp.asarray(np.asarray(index, int).ravel(), jnp.int32))


def recenter(
    images: Complex[Array, "n y x"], shift_in_pixels: Float[Array, "n 2"], sign: float = SHIFT_SIGN
) -> Complex[Array, "n y x"]:
    """Translate Fourier particles by `sign * shift_in_pixels`.

    Sub-pixel exact, a translation being a phase ramp, so there is no interpolation and no
    resampling loss. Goes through `fourier.phase_ramp` rather than writing its own exponential: a
    second copy of the ramp sign would let a stack be recentered at one convention and scored at the
    other, and the two agree only on already-centered particles.

    Args:
        images: fft2 of the particle stack, unshifted.
        shift_in_pixels: (n, 2) offsets as (y, x), from `ParticleMeta.shift_in_pixels`.
        sign: Which way the stored offset points. See `SHIFT_SIGN`, and measure it with
            `check_shift_sign` rather than inheriting it.

    Returns:
        The recentered stack.
    """
    return images * phase_ramp(sign * jnp.asarray(shift_in_pixels), images.shape[-1])


def check_shift_sign(
    images: Complex[Array, "n y x"],
    shift_in_pixels: Float[Array, "n 2"],
    *,
    pixel_size: float = 1.0,
    band_in_angstroms: float = 30.0,
) -> tuple[float, bool, float, dict[str, float]]:
    """Decide the recentering convention FROM THE IMAGES rather than from a convention document.

    Below about 40 A a compact globular particle projects to nearly the same blob whatever its pose,
    so a CENTERED pool's low band adds coherently and a mis-centered one's does not. The statistic is
    the coherent fraction of low-band power,

        |mean_i Y_i(k)|^2  over  mean_i |Y_i(k)|^2,   summed over 0 < |k| <= 1 / band

    LARGER when the recentering is right, needing no map, poses or labels. All four (sign, axis-order)
    variants plus a do-nothing baseline are scored, since a sign error and an axis swap are separate
    mistakes; if `none` wins, NOTHING should be applied. The test assumes pose-independence of the
    low band, so it is for globular targets. The margin is RETURNED and deliberately NOT thresholded:
    a hard pool can carry a clear margin and still be wrong, so read the number rather than a verdict.

    Args:
        images: fft2 of the particle stack, unshifted.
        shift_in_pixels: (n, 2) offsets as (y, x).
        pixel_size: Pixel size in A, which sets what `band_in_angstroms` means.
        band_in_angstroms: Low-resolution band to score over.

    Returns:
        `(sign, swap_axes, margin, scores)`. `sign` is `+1`, `-1`, or `0` when doing nothing wins,
        and `margin` is the winner's score over the runner-up's -- one is a tie, large is unambiguous.
    """
    box_size = images.shape[-1]
    # the radial grid from the one builder rather than rebuilt by hand: a second spelling is a second
    # chance to disagree with shell_indices about where Nyquist lands
    radius = frequency_grid(box_size, pixel_size)
    band = (radius <= 1.0 / band_in_angstroms) & (radius > 0.0)
    offsets = jnp.asarray(shift_in_pixels)

    scores = {}
    for name, sign, swap in (
        ("none", 0.0, False),
        ("minus", -1.0, False),
        ("plus", +1.0, False),
        ("minus_swap", -1.0, True),
        ("plus_swap", +1.0, True),
    ):
        moved = recenter(images, offsets[:, ::-1] if swap else offsets, sign)
        coherent = jnp.sum(jnp.abs(moved.mean(0)) ** 2 * band)
        total = jnp.sum((jnp.abs(moved) ** 2).mean(0) * band)
        scores[name] = float(coherent / jnp.maximum(total, 1e-30))

    ranked = sorted(scores.values(), reverse=True)
    margin = ranked[0] / max(ranked[1], 1e-30)
    best = max(scores, key=lambda name: scores[name])
    sign = 0.0 if best == "none" else (-1.0 if best.startswith("minus") else 1.0)
    return sign, best.endswith("_swap"), margin, scores
