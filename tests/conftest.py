"""Shared fixtures and builders: synthetic structures the tests render, cryoSPARC/RELION files the
preprocessing arm reads, and the Gaussian-blob maps the rungs and forward model check against.

Shared so a convention (PDB layout, star column order, `box // 2` centering, blob draw order) lives
once; a second copy drifts into a test failure with no bug behind it.
"""

import importlib.util
import os
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from cryodisc.specimen.structures import align_conformers

COMPILATION_CACHE = Path(__file__).resolve().parent.parent / ".jax_cache"
"""Where XLA's compiled kernels are kept between runs, because the suite's floor is compilation.

The min-compile-time threshold below is set very low so essentially every distinct shape's kernel is
cached; the fill cost is negligible and the warm run is much faster, executing the same kernels
production runs. Test-only, and deliberately not exported to the drivers: a cluster job would write
this into a shared filesystem from many ranks for a cache it fills and discards in one run.

Each setting is skipped when its own environment variable is already set, so the cache directory and
the threshold stay overridable from outside -- `jax.config.update` would otherwise win over the
environment.
"""
if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
    jax.config.update("jax_compilation_cache_dir", str(COMPILATION_CACHE))
if "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in os.environ:
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.001)

EXPERIMENTS = Path(__file__).resolve().parent.parent / "reproduction/experiments"


def load_driver(relative_path: str):
    """Load a driver from its path, the way running it would.

    `reproduction/experiments/` is deliberately not a package -- see `test_experiments_layout` -- so a driver is
    read off disk rather than imported.

    Args:
        relative_path: Path under `reproduction/experiments/`.

    Returns:
        The loaded module.
    """
    path = EXPERIMENTS / relative_path
    spec = importlib.util.spec_from_file_location(f"driver_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # register before exec: a driver defining a dataclass looks its own module up by name while the
    # class body runs
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BACKBONE = (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"))
"""Four distinctly named atoms per residue, so none of them dedup away on the altloc key."""

ALPHA_CARBON = (("CA", "C"),)
"""One atom per residue, for tests that want a given atom count rather than a given chemistry."""


def atom_records(coords, *, atoms=BACKBONE, chain="A", start_serial=1, altloc=" ", occupancy=1.0):
    """PDB ATOM lines for `coords`, laid out as residues of `atoms`.

    Args:
        coords: (n, 3) positions in A.
        atoms: `(name, element)` per position, cycled.
        chain: Chain id.
        start_serial: First atom serial, for appending a second set to an existing file.
        altloc: Alternate-location indicator.
        occupancy: Occupancy column.

    Returns:
        The lines, without a trailing END.
    """
    lines = []
    for i, (x, y, z) in enumerate(coords):
        name, element = atoms[i % len(atoms)]
        lines.append(
            f"ATOM  {start_serial + i:5d} {name:<4s}{altloc}ALA {chain}{i // len(atoms) + 1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}  0.00          {element:>2s}"
        )
    return lines


def write_pdb(path, lines):
    """Write `lines` as a PDB file and return the path."""
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def conformer_pair(directory, coords, bent, *, atoms=BACKBONE):
    """Two conformers of one molecule, aligned into a common frame.

    Args:
        directory: Where to write the two structures.
        coords: (n, 3) positions of state A.
        bent: (n, 3) positions of state B, the same atoms moved.
        atoms: `(name, element)` per position, cycled.

    Returns:
        The `ConformerPair`.
    """
    return align_conformers(
        write_pdb(directory / "a.pdb", atom_records(coords, atoms=atoms)),
        write_pdb(directory / "b.pdb", atom_records(bent, atoms=atoms)),
        assembly=None,
    )


def hinged_coordinates(n_atoms=200, spread=14.0, hinge=4.0, seed=1):
    """A blob and the same blob with one half swung along x: the preprocessing arm's standard pair.

    A hinge and not a rigid displacement: the change conserves mass and moves the centroid by well
    under a voxel, so `center_of_mass_separation` on this pair is an assertion on the renderer.

    Args:
        n_atoms: Positions drawn.
        spread: Drawn uniformly on `[-spread, spread]` per axis, in A.
        hinge: How far the positive-x half moves along x, in A.
        seed: Seed of the `default_rng` the draw comes off.

    Returns:
        `(coords, bent)`, both (n_atoms, 3) in A.
    """
    coords = np.random.default_rng(seed).uniform(-spread, spread, (n_atoms, 3))
    bent = coords + np.where(coords[:, :1] > 0.0, np.array([[hinge, 0.0, 0.0]]), 0.0)
    return coords, bent


def hinged_pair_files(directory, **kwargs):
    """`hinged_coordinates` written into `directory` as two PDBs, returned as `(path_a, path_b)`."""
    coords, bent = hinged_coordinates(**kwargs)
    return (
        write_pdb(directory / "a.pdb", atom_records(coords, atoms=ALPHA_CARBON)),
        write_pdb(directory / "b.pdb", atom_records(bent, atoms=ALPHA_CARBON)),
    )


def hinged_pair(directory, **kwargs):
    """`hinged_coordinates` written into `directory` and aligned, returned as a `ConformerPair`."""
    return conformer_pair(directory, *hinged_coordinates(**kwargs), atoms=ALPHA_CARBON)


PARTICLE_DTYPE = [
    ("uid", "<u8"),
    ("blob/path", "S64"),
    ("blob/idx", "<u4"),
    ("blob/shape", "<u4", (2,)),
    ("blob/psize_A", "<f4"),
    ("blob/sign", "<f4"),
    ("ctf/df1_A", "<f4"),
    ("ctf/df2_A", "<f4"),
    ("ctf/df_angle_rad", "<f4"),
    ("ctf/accel_kv", "<f4"),
    ("ctf/cs_mm", "<f4"),
    ("ctf/amp_contrast", "<f4"),
    ("alignments2D/class", "<u4"),
]
"""The fields a cryoSPARC class_2D output carries once its passthrough is joined on."""

LOCATION_DTYPE = [
    ("location/micrograph_path", "S64"),
    ("location/micrograph_shape", "<u4", (2,)),
    ("location/micrograph_psize_A", "<f4"),
    ("location/center_x_frac", "<f4"),
    ("location/center_y_frac", "<f4"),
]
"""Where a particle was picked. Separate because only a pick job's own output carries it."""


def cryosparc_particles(n, *, box=48, pixel_size=2.04, classes=(0, 1), stack="J5/extract/stack.mrcs", extra=None):
    """A particle table shaped like a cryoSPARC class_2D output.

    Args:
        n: Number of particles.
        box: Stored box size in pixels.
        pixel_size: A/px of that box.
        classes: Class labels, cycled over the rows.
        stack: The stored stack path, which is relative to a project directory.
        extra: Optional `{field: array}` appended to the dtype, for the columns a job type adds --
            e.g. `alignments3D/pose` and `alignments3D/shift`, which the reader branches on.

    Returns:
        The structured array.
    """
    dtype = PARTICLE_DTYPE
    if extra:
        appended = [(name, np.asarray(value).dtype, np.asarray(value).shape[1:]) for name, value in extra.items()]
        dtype = np.dtype(list(PARTICLE_DTYPE) + appended)
    data = np.zeros(n, dtype=dtype)
    for name, value in (extra or {}).items():
        data[name] = value
    data["uid"] = np.arange(n) + 100
    data["blob/path"] = stack.encode()
    data["blob/idx"] = np.arange(n)
    data["blob/shape"] = box
    data["blob/psize_A"] = pixel_size
    data["blob/sign"] = 1.0
    # spanning enough defocus that `transfer_from_meta` has to build more than one group
    data["ctf/df1_A"] = 11000.0 + 40.0 * np.arange(n)
    data["ctf/df2_A"] = 11100.0 + 40.0 * np.arange(n)
    data["ctf/accel_kv"] = 300.0
    data["ctf/cs_mm"] = 2.7
    data["ctf/amp_contrast"] = 0.1
    data["alignments2D/class"] = [classes[i % len(classes)] for i in range(n)]
    return data


def located_particles(n, **kwargs):
    """`cryosparc_particles` widened with the `location/*` fields a pick job's own output carries.

    Args:
        n: Number of particles.
        **kwargs: Passed to `cryosparc_particles`.

    Returns:
        The structured array, with every `location/*` field left at zero for the caller to fill.
    """
    base = cryosparc_particles(n, **kwargs)
    located = np.zeros(n, dtype=base.dtype.descr + LOCATION_DTYPE)
    for name in base.dtype.names:
        located[name] = base[name]
    return located


def gaussian_blobs(box_size, n_blobs, *, seed, spread, sigma_range, amplitude_range=(0.5, 1.5)):
    """Isotropic Gaussian blobs, and a builder for the voxel map they sum to.

    Isotropy is the point: a rotation or translation of the centers transforms the map exactly, so a
    state B built from moved centers differs from state A by a pure pose change and nothing else
    (making pose removal testable), and the line integral stays a sum of 2D Gaussians (giving an
    analytic projection at every orientation).

    Args:
        box_size: Voxels per side of the cubic map.
        n_blobs: Number of blobs.
        seed: Seed of the `default_rng` the draws come off, in order: centers, widths, amplitudes.
        spread: Centers are drawn uniformly on `[-spread, spread]` per axis, in voxels.
        sigma_range: `(low, high)` of the uniform width draw, in voxels.
        amplitude_range: `(low, high)` of the uniform amplitude draw.

    Returns:
        `(centers, sigmas, amplitudes, build)`, where `build(moved_centers, offset=0.0)` renders the
        float32 map of blobs of those same widths and amplitudes at `moved_centers`.
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-spread, spread, (n_blobs, 3))
    sigmas = rng.uniform(*sigma_range, n_blobs)
    amplitudes = rng.uniform(*amplitude_range, n_blobs)
    axis = np.arange(box_size) - box_size / 2.0
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")

    def build(moved_centers, offset=0.0):
        grid = np.full((box_size,) * 3, float(offset))
        for center, sigma, amplitude in zip(moved_centers, sigmas, amplitudes):
            squared = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
            grid += amplitude * np.exp(-squared / (2 * sigma**2))
        return grid.astype(np.float32)

    return centers, sigmas, amplitudes, build


def gaussian_blob_models(box_size, n_blobs, *, seed, spread, sigma_range, pixel_size=1.0):
    """`gaussian_blobs` with each map handed to cryoJAX, which is the form the rungs consume.

    See `gaussian_blobs` for what the isotropy buys and for the draw order the maps depend on.

    Args:
        box_size: Voxels per side of the cubic map.
        n_blobs: Number of blobs.
        seed: Seed of the `default_rng` the draws come off.
        spread: Centers are drawn uniformly on `[-spread, spread]` per axis, in voxels.
        sigma_range: `(low, high)` of the uniform width draw, in voxels.
        pixel_size: A/px the map is declared at.

    Returns:
        `(centers, build)`, where `build(moved_centers, offset=0.0)` returns the `CryoJAXModel` of
        the map at those centers.
    """
    # imported here so tests that never render a map do not pay for cryoJAX at collection
    from cryodisc.forward.cryojax_model import CryoJAXModel

    centers, _, _, build_grid = gaussian_blobs(box_size, n_blobs, seed=seed, spread=spread, sigma_range=sigma_range)

    def build(moved_centers, offset=0.0):
        return CryoJAXModel.from_voxel_grid(build_grid(moved_centers, offset), pixel_size=pixel_size)

    return centers, build


DEFLECTIONS = [0.3, 1.0, 3.0, 10.0]
"""Squared distances the closed-form checks are parametrized over, spanning a wide range so a
statistic that is right at one separation and wrong at another is caught."""


def gaussian_pools(rng, squared_distance, n):
    """Lambda under the model: `Lambda|A ~ N(+d^2/2, d^2)`, `Lambda|B ~ N(-d^2/2, d^2)`.

    The equal-variance Gaussian log-ratio that `deflection` and `coherence` are validated against.
    Shared so a sign or variance edited in one copy cannot leave another suite validating a different
    model. The `+d^2/2` against `-d^2/2` is what makes `Lambda` a real log-ratio.

    Args:
        rng: A `default_rng`, passed in so several tests can draw repeats off one stream.
        squared_distance: `d^2`, the separation in units of the score's own scale.
        n: Particles per pool.

    Returns:
        `(from_a, from_b)`, each `(n,)` as a JAX array.
    """
    # imported lazily so tests that never touch JAX do not pay for it at collection
    from jax import numpy as jnp

    scale = np.sqrt(squared_distance)
    return (
        jnp.asarray(rng.normal(+0.5 * squared_distance, scale, n)),
        jnp.asarray(rng.normal(-0.5 * squared_distance, scale, n)),
    )


def save_cs(path, data):
    """Write a structured array under a name that keeps its extension.

    Through a handle because `np.save` appends `.npy` to any path that does not end in it, and a
    `.cs` that landed as `.cs.npy` would be a file nothing downstream looks for.
    """
    with open(path, "wb") as handle:
        np.save(handle, data, allow_pickle=False)
    return path


OPTICS = {"voltage_in_kv": 300.0, "spherical_aberration_in_mm": 2.7, "amplitude_contrast_ratio": 0.1}
"""The three `ctffit` defaults to, so a synthetic spectrum is fitted at its own optics."""


def write_relion_star(
    path, image_names, defocus, *, pixel_size, box_size, optics=OPTICS, defocus_angle=0.0, euler=None
):
    """A RELION 3.1 star file: an optics block, then one row per particle.

    The tag order is the format `particles.metadata` parses, so it lives once.

    Args:
        path: Where to write.
        image_names: `NNNNNN@stack.mrcs` per particle, in the table's row order.
        defocus: (n,) underfocus in A written to both astigmatic axes, or (n, 2) of `(u, v)`.
        pixel_size: A/px recorded for the images.
        box_size: Their edge in pixels.
        optics: `voltage_in_kv`, `spherical_aberration_in_mm` and `amplitude_contrast_ratio` as
            recorded. The optics a pool was RENDERED through, when there are any.
        defocus_angle: Astigmatism azimuth in degrees, written to every row.
        euler: (n, 3) of RELION's (rot, tilt, psi) in degrees, or `None` to omit the columns. Opt-in
            because a deposited pool normally lacks them, and `ParticleMeta.has_euler` then reads False.
    """
    defocus = np.asarray(defocus, float)
    if defocus.ndim == 1:
        defocus = np.repeat(defocus[:, None], 2, axis=1)
    angles = np.zeros((len(image_names), 3)) if euler is None else np.asarray(euler, float)
    rows = "\n".join(
        f"{name} {axes[0]:.3f} {axes[1]:.3f} {defocus_angle} 1 0.0 0.0"
        + ("" if euler is None else f" {rot:.4f} {tilt:.4f} {psi:.4f}")
        for name, axes, (rot, tilt, psi) in zip(image_names, defocus, angles, strict=True)
    )
    path.write_text(
        "data_optics\n\nloop_\n_rlnOpticsGroup #1\n_rlnVoltage #2\n_rlnSphericalAberration #3\n"
        "_rlnAmplitudeContrast #4\n_rlnImagePixelSize #5\n_rlnImageSize #6\n"
        f"1 {optics['voltage_in_kv']} {optics['spherical_aberration_in_mm']} "
        f"{optics['amplitude_contrast_ratio']} {pixel_size} {box_size}\n\n"
        "data_particles\n\nloop_\n_rlnImageName #1\n_rlnDefocusU #2\n_rlnDefocusV #3\n"
        "_rlnDefocusAngle #4\n_rlnOpticsGroup #5\n_rlnOriginXAngst #6\n_rlnOriginYAngst #7\n"
        + ("" if euler is None else "_rlnAngleRot #8\n_rlnAngleTilt #9\n_rlnAnglePsi #10\n")
        + rows
        + "\n"
    )
    return path


def thon_spectrum(defocus_u, defocus_v=None, angle=0.0, *, box, pixel_size, noise=0.0, seed=0):
    """A Thon-ring power spectrum at a known defocus, on a stated grid.

    Built through the same `ctf_astigmatic` the fit scores with, so what is tested is the threading
    of `pixel_size`, not the CTF formula. The envelope and background `fit_defocus`'s median filter
    is tuned against are stated once.

    Args:
        defocus_u: Underfocus along the astigmatism axis in A.
        defocus_v: The perpendicular one, defaulting to isotropic.
        angle: Astigmatism azimuth in radians.
        box: Edge of the spectrum in pixels.
        pixel_size: A/px of the micrograph it came from.
        noise: Relative multiplicative noise.
        seed: PRNG seed.

    Returns:
        `(box, box)` fft-ordered power, which is what `tiled_power_spectrum` returns.
    """
    # imported here so tests that never build a CTF do not pay for the optics stack at collection
    from cryodisc.microscope import ctf_astigmatic

    frequency = np.fft.fftfreq(box, d=pixel_size)
    grid_x, grid_y = np.meshgrid(frequency, frequency)
    transfer = np.asarray(
        ctf_astigmatic(
            grid_y,
            grid_x,
            defocus_u=defocus_u,
            defocus_v=defocus_u if defocus_v is None else defocus_v,
            astigmatism_angle=angle,
            **OPTICS,
        )
    )
    radius_squared = grid_x**2 + grid_y**2
    # a decaying envelope plus a background, which is what the median filter has to remove
    spectrum = np.exp(-120.0 * radius_squared) * (transfer**2 + 0.35)
    if noise:
        spectrum = spectrum * (1.0 + noise * np.random.default_rng(seed).normal(size=spectrum.shape))
    return spectrum


def physical_grid(box, pixel):
    """(x, y, z) in A of every voxel of a box-centered map, indexed (z, y, x).

    Centered at `box // 2`, which is where cryoJAX renders and what `density` assumes.
    """
    center = box // 2
    index_z, index_y, index_x = np.indices((box, box, box))
    return (index_x - center) * pixel, (index_y - center) * pixel, (index_z - center) * pixel


def gaussian_cloud(positions, box, pixel, width):
    """Isotropic Gaussians of one width at physical positions, summed on `physical_grid`.

    Positions in A rather than voxels, so the same molecule reaches two grids as the same thing --
    what a grid-invariance claim needs and what `ConformerPair` cannot give (it superposes its two
    states, removing any centroid difference). The `box // 2` centering is load-bearing: it must match
    cryoJAX and `density` or the two grids place the molecule at different sub-voxel offsets.

    Args:
        positions: (n, 3) centers in A.
        box: Voxels per side.
        pixel: A/voxel.
        width: The one Gaussian sigma in A, shared by every position.

    Returns:
        The `(box, box, box)` map.
    """
    x, y, z = physical_grid(box, pixel)
    volume = np.zeros((box,) * 3)
    for position in positions:
        squared = (x - position[0]) ** 2 + (y - position[1]) ** 2 + (z - position[2]) ** 2
        volume += np.exp(-squared / (2.0 * width**2))
    return volume


@pytest.fixture(scope="session")
def bent_pair_factory(tmp_path_factory):
    """Builds a `ConformerPair` of `n_atoms` backbone atoms with its second half displaced.

    Each caller wants a different blob, so this memoizes nothing; the module-scoped fixtures that
    wrap it are what stop a structure being read twice.
    """

    def build(n_atoms, spread, bend):
        coords = np.random.default_rng(0).normal(0.0, spread, (n_atoms, 3))
        bent = coords.copy()
        bent[n_atoms // 2 :] += np.array([0.0, bend, 0.0])
        return conformer_pair(tmp_path_factory.mktemp("pair"), coords, bent)

    return build


@pytest.fixture(scope="session")
def asymmetric_pair(tmp_path_factory):
    """A pair with NO point symmetry, the only kind that can see a pole error.

    A symmetric structure's projection down the beam at a pole is invariant under a half turn in the
    image plane -- exactly the transform `poses.expansion_plan` can get wrong at `theta` = 0 and 180
    -- so it cannot express that bug class. Session-scoped and shared, since several files need it.
    """
    coords = np.random.default_rng(7).uniform(-30.0, 30.0, (400, 3))
    bent = coords + np.where(coords[:, :1] > 0.0, np.array([[4.0, 0.0, 0.0]]), 0.0)
    return conformer_pair(tmp_path_factory.mktemp("asymmetric"), coords, bent, atoms=ALPHA_CARBON)
