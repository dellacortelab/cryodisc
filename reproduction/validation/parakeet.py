"""The second forward model: multislice, so a gate can measure ours against physics we did not write.

Parakeet propagates the electron wave through a solvent slab slice by slice, independent of the
projection model. It produces a BANK, not a model (no derivative, too slow inside a pose search), as
ONE micrograph of well-separated instances cut back out. Three conventions are silent when wrong:
`c_10` is NEGATIVE for underfocus; solvent is `simulation.ice` (not the atomic `sample.ice`);
orientations are ROTVECS IN RADIANS. Install `cryodisc[parakeet]`; it needs a GPU with
MULTEM (an A100+ partition, and FFTW on LD_LIBRARY_PATH), so this is a cluster path.
"""

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from reproduction.validation.template_bank import TemplateBank

PARAKEET_PADDING = 100
"""Gap between neighboring instances, in pixels, borrowed from Parakeet's `simulation.padding`
default. Instances must not overlap once the CTF spreads them: a defocused particle's rings reach
past its box, and a neighbor's rings inside a cropped template would read as a forward-model deficit.
"""


@dataclass(frozen=True)
class MultisliceConfig:
    """What a multislice render needs beyond the structure and the poses.

    Args:
        box_size: Template size in pixels. Must match the cryoJAX bank it will be gated against.
        pixel_size: A/px, likewise.
        voltage_in_kv: Accelerating voltage, in kV.
        defocus_in_angstroms: Underfocus, positive here and negated into Parakeet's `c_10` (which is
            negative for underfocus). Defaulted to match `forward.optics.CTFParams` so a gate's two
            arms agree.
        spherical_aberration_in_mm: Cs.
        slice_thickness_in_angstroms: Propagation step, in A. Its convergence must be checked before
            a bank is trusted.
        dose: Electron exposure in e/A^2, or `None` for a NOISE-FREE render. A gate wants `None`:
            shot noise hides the level error the gate exists to catch.
        ice_thickness_in_angstroms: Depth of the slab the instances sit in, in A.
        solvent: Fill the slab with amorphous ice. A gate testing the projection APPROXIMATION wants
            it on; one isolating the integrator alone wants it off.

    Raises:
        ValueError: If any size is not positive, or the slice step exceeds the slab, which would
            propagate the whole specimen in one step and silently be a projection.
    """

    box_size: int
    pixel_size: float
    voltage_in_kv: float = 300.0
    defocus_in_angstroms: float = 12000.0
    spherical_aberration_in_mm: float = 2.7
    slice_thickness_in_angstroms: float = 3.0
    dose: float | None = None
    ice_thickness_in_angstroms: float = 500.0
    solvent: bool = True

    def __post_init__(self) -> None:
        if self.box_size < 1 or self.pixel_size <= 0.0:
            raise ValueError(f"box_size and pixel_size must be positive, got {self.box_size} and {self.pixel_size}")
        if self.slice_thickness_in_angstroms <= 0.0:
            raise ValueError(f"slice_thickness_in_angstroms must be positive, got {self.slice_thickness_in_angstroms}")
        if self.slice_thickness_in_angstroms > self.ice_thickness_in_angstroms:
            raise ValueError(
                f"a slice of {self.slice_thickness_in_angstroms} A through a slab of "
                f"{self.ice_thickness_in_angstroms} A is one step, which IS a projection -- so a "
                "gate run at these settings would compare the projection model against itself."
            )
        if self.defocus_in_angstroms < 0.0:
            raise ValueError(
                f"defocus_in_angstroms is underfocus and stated POSITIVE here, got "
                f"{self.defocus_in_angstroms}; it is negated into Parakeet's c_10 at the boundary."
            )
        if self.dose is not None and self.dose <= 0.0:
            raise ValueError(f"dose must be positive or None for noise-free, got {self.dose}")


def laptop_config(box_size: int = 48, pixel_size: float = 2.0) -> MultisliceConfig:
    """A configuration small enough to exercise the wiring in seconds.

    Coarse pixels, a thin slab and a large slice step, so the render is fast and the result is not
    converged. Use it to check that a bank comes back with the right shape, poses and provenance --
    never to quote a number, and never as the reference arm of a gate.

    Args:
        box_size: Template size in pixels.
        pixel_size: A/px, deliberately coarse.

    Returns:
        The configuration.
    """
    return MultisliceConfig(
        box_size=box_size,
        pixel_size=pixel_size,
        slice_thickness_in_angstroms=20.0,
        ice_thickness_in_angstroms=100.0,
        dose=None,
        solvent=False,
    )


def micrograph_layout(n_poses: int, config: MultisliceConfig) -> tuple[np.ndarray, int]:
    """Where each pose sits in the shared micrograph, and how large that micrograph must be.

    A square-ish grid with a margin, because the cost of the render grows with the micrograph area
    and a long thin layout wastes most of it. Positions are in ANGSTROMS, which is what Parakeet's
    instance list takes, and the caller converts back to pixels to crop.

    Args:
        n_poses: Templates to place.
        config: The render settings, for the box and pixel size.

    Returns:
        `(positions, micrograph_px)` with `positions` an `(n, 2)` array of `(x, y)` centers in A.

    Raises:
        ValueError: If there are no poses to place.
    """
    if n_poses < 1:
        raise ValueError(f"need at least one pose to lay out, got {n_poses}")
    spacing = (config.box_size + PARAKEET_PADDING) * config.pixel_size
    per_row = int(np.ceil(np.sqrt(n_poses)))
    # one spacing of border on every side, so no instance's CTF rings run off the micrograph
    micrograph_px = int(np.ceil((per_row + 1) * spacing / config.pixel_size))
    index = np.arange(n_poses)
    positions = np.stack([(index % per_row + 1) * spacing, (index // per_row + 1) * spacing], axis=-1)
    return positions, micrograph_px


def atoms_only_copy(structure_path: str | Path, destination: Path) -> Path:
    """The same structure with only its atoms, so Parakeet renders what it was handed.

    Parakeet's loader expands the biological assembly (REMARK 350) unconditionally and cryoJAX never
    does, so a deposition that builds an oligomer would put a DIFFERENT MOLECULE through each engine
    and the gate would report it as a forward-model deficit. The assembly records are dropped rather
    than an `assembly_index` passed, because Parakeet has no index meaning "as deposited"; which
    assembly to build is a caller's question, answered once upstream in `structures.align_conformers`.

    Args:
        structure_path: The deposited structure.
        destination: Where to write the stripped copy.

    Returns:
        The path written.
    """
    kept = [
        line
        for line in Path(structure_path).read_text().splitlines(keepends=True)
        if line.startswith(("ATOM", "HETATM", "TER", "MODEL", "ENDMDL", "END"))
    ]
    destination.write_text("".join(kept))
    return destination


def euler_to_rotvec(euler_zyz_degrees: np.ndarray) -> np.ndarray:
    """Convert cryoJAX intrinsic-zyz poses to the rotation vectors Parakeet's instances take.

    A gate whose two engines disagree about a pose convention measures the convention and reports it
    as a forward-model deficit, which no spectral statistic can see -- so the conversion lives here,
    once, rather than in whichever driver happens to call both.

    Args:
        euler_zyz_degrees: `(n, 3)` intrinsic zyz Euler angles in degrees.

    Returns:
        `(n, 3)` rotation vectors in RADIANS, which is what the instance list wants.

    The sequence is LOWERCASE `"zyz"`: scipy reads a lowercase sequence as EXTRINSIC and an uppercase
    one as intrinsic, and `cryojax_model` passes extrinsic. The uppercase form would silently swap phi
    and psi, returning a well-formed bank at the wrong poses.
    """
    from scipy.spatial.transform import Rotation

    return np.asarray(Rotation.from_euler("zyz", np.asarray(euler_zyz_degrees, float), degrees=True).as_rotvec())


def config_overrides(
    structure_path: str | Path, positions: np.ndarray, rotvecs: np.ndarray, micrograph_px: int, config: MultisliceConfig
) -> dict[str, Any]:
    """The keys this module sets on top of Parakeet's default configuration.

    Separated from the render so it can be checked without a GPU: a wrong key here is silent -- the
    default simply stands -- and it is the commonest way a render comes back looking fine and being
    at the wrong defocus, the wrong voltage or the wrong pixel size.

    Args:
        structure_path: Atomic model.
        positions: `(n, 2)` instance centers in A, from `micrograph_layout`.
        rotvecs: `(n, 3)` orientations in radians, from `euler_to_rotvec`.
        micrograph_px: Micrograph size in pixels.
        config: The render settings.

    Returns:
        A nested dict of overrides, to merge into `parakeet.config.new().model_dump()`.
    """
    extent = micrograph_px * config.pixel_size
    depth = config.ice_thickness_in_angstroms
    instances = [
        {"position": [float(x), float(y), depth / 2.0], "orientation": [float(v) for v in rotvec]}
        for (x, y), rotvec in zip(positions, rotvecs, strict=True)
    ]
    return {
        "sample": {
            "box": [extent, extent, depth],
            "centre": [extent / 2.0, extent / 2.0, depth / 2.0],
            "shape": {"type": "cuboid", "cuboid": {"length_x": extent, "length_y": extent, "length_z": depth}},
            "molecules": {"local": [{"filename": str(structure_path), "instances": instances}]},
            # `sample.ice` is the SLOW atomic model and stays off; the solvent a gate wants is
            # `simulation.ice` below. A made-up key in either block leaves the default standing
            "ice": None,
        },
        "microscope": {
            "beam": {
                # kV directly. Parakeet's beam energy is in keV, so 300 and not 0.3
                "energy": config.voltage_in_kv,
                "electrons_per_angstrom": config.dose if config.dose is not None else 0.0,
            },
            "lens": {
                # NEGATIVE for underfocus, which is Parakeet's own convention; see the docstring
                "c_10": -config.defocus_in_angstroms,
                "c_30": config.spherical_aberration_in_mm,
            },
            "detector": {"nx": micrograph_px, "ny": micrograph_px, "pixel_size": config.pixel_size},
        },
        "simulation": {"slice_thickness": config.slice_thickness_in_angstroms, "ice": config.solvent},
        "scan": {"mode": "still", "num_images": 1},
    }


def crop_templates(micrograph: np.ndarray, positions: np.ndarray, config: MultisliceConfig) -> np.ndarray:
    """Cut each instance back out of the shared micrograph.

    Args:
        micrograph: `(y, x)` rendered image.
        positions: `(n, 2)` instance centers in A.
        config: The render settings.

    Returns:
        `(n, box, box)` templates in the order the positions were given.

    Raises:
        ValueError: If an instance's box would run off the micrograph, which means the layout and
            the render disagree about the size and every template past that point is garbage.
    """
    # box taken as a LENGTH from the start, not `center +- half`, which is a pixel short on an odd box
    half = config.box_size // 2
    templates = np.zeros((len(positions), config.box_size, config.box_size), np.float32)
    for i, (x, y) in enumerate(positions):
        column, row = round(x / config.pixel_size), round(y / config.pixel_size)
        patch = micrograph[row - half : row - half + config.box_size, column - half : column - half + config.box_size]
        if patch.shape != (config.box_size, config.box_size):
            raise ValueError(
                f"instance {i} at pixel ({row}, {column}) does not fit a {config.box_size} box inside a "
                f"{micrograph.shape} micrograph; the layout and the render disagree about the size."
            )
        templates[i] = patch
    return templates


def render_bank(
    structure_path: str | Path,
    angles: Float[Array, "p 3"],
    config: MultisliceConfig,
    *,
    work_dir: str | Path | None = None,
) -> TemplateBank:
    """Render a multislice bank at the given poses, as one micrograph.

    Args:
        structure_path: Atomic model, in a format Parakeet's sample builder reads.
        angles: `(p, 3)` Euler angles in degrees, cryoJAX intrinsic zyz. Converted here, so the two
            engines are driven from ONE pose convention.
        config: The render settings.
        work_dir: Which filesystem the intermediate `.yaml` and `.h5` files land on -- they run to
            gigabytes, so a cluster wants node-local scratch rather than a home directory. `None`
            uses the system temporary directory. A fresh subdirectory is made inside it either way
            and removed afterwards; see `_render_micrograph` for why that is not optional.

    Returns:
        A `TemplateBank` whose `engine` is `"parakeet"` and whose provenance matches a cryoJAX bank
        built at the same box, pixel, voltage and defocus.

    Raises:
        ImportError: If Parakeet is not installed.
        ValueError: If `angles` is not `(p, 3)`.
    """
    # the argument checks come BEFORE the import, so a caller developing on a laptop gets the shape
    # error rather than a message about a GPU dependency they were not going to reach anyway
    poses = np.asarray(angles, float)
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError(f"angles must be (p, 3) Euler angles in degrees, got {poses.shape}")
    positions, micrograph_px = micrograph_layout(len(poses), config)

    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="parakeet-", dir=work_dir))
    try:
        # the atoms, and only the atoms: Parakeet expands REMARK 350 and cryoJAX does not, so the
        # deposited file itself would put a different molecule through each engine
        structure = atoms_only_copy(structure_path, scratch / "atoms.pdb")
        overrides = config_overrides(structure, positions, euler_to_rotvec(poses), micrograph_px, config)
        micrograph = _render_micrograph(overrides, scratch, noise_free=config.dose is None)
    finally:
        # in a `finally` because the intermediates are gigabytes and a failed render is exactly when
        # nobody is watching the filesystem
        shutil.rmtree(scratch, ignore_errors=True)

    return TemplateBank(
        images=jnp.asarray(crop_templates(micrograph, positions, config)),
        angles=jnp.asarray(poses),
        pixel_size=config.pixel_size,
        voltage_in_kv=config.voltage_in_kv,
        defocus_in_angstroms=config.defocus_in_angstroms,
        engine="parakeet",
    )


def _render_micrograph(
    overrides: dict[str, Any], scratch: Path, *, noise_free: bool
) -> np.ndarray:  # pragma: no cover - needs MULTEM, exercised on the cluster
    """Run the Parakeet stages and read the resulting micrograph, in the directory it is given.

    The ONLY part of this module that cannot run without a GPU; everything above it is exercised by
    the suite, so a wiring mistake fails on a laptop rather than in a queue.

    The scratch directory must be fresh per call, and that is a correctness requirement: Parakeet's
    stages address each other by fixed path names, so two renders sharing a directory overwrite each
    other's sample and exit wave and can return a bank rendered from the wrong structure.

    Args:
        overrides: From `config_overrides`.
        scratch: An empty directory owned by the caller, unique to this render. `render_bank`
            creates and removes it, so the stripped structure and the intermediates share one
            lifetime and one filesystem.
        noise_free: Stop after the optics stage rather than running the detector, which is what a
            gate wants: shot noise would hide the level error the gate exists to catch.

    Returns:
        `(y, x)` micrograph.

    Raises:
        ImportError: If Parakeet is not installed.
    """
    try:
        import h5py
        import parakeet.config
        import parakeet.sample
        import parakeet.simulate
        import yaml
    except ImportError as exc:
        raise ImportError(
            "multislice rendering needs Parakeet: pip install cryodisc[parakeet]. It "
            "wants a GPU with MULTEM behind it, so this is a cluster path; `laptop_config` exists to "
            "exercise the wiring, not to avoid the dependency."
        ) from exc

    # a filename, because `new()` writes one and defaults it into the working directory
    settings = _merge(parakeet.config.new(str(scratch / "defaults.yaml")).model_dump(), overrides)

    config_path = scratch / "config.yaml"
    sample_path, exit_wave_path, optics_path, image_path = (
        scratch / name for name in ("sample.h5", "exit_wave.h5", "optics.h5", "image.h5")
    )
    with config_path.open("w") as handle:
        yaml.safe_dump(settings, handle)

    # every stage addresses the previous one by PATH, which is the API these functions have
    parakeet.sample.new(str(config_path), str(sample_path))
    parakeet.sample.add_molecules(str(config_path), str(sample_path))
    parakeet.simulate.exit_wave(str(config_path), str(sample_path), str(exit_wave_path))
    parakeet.simulate.optics(str(config_path), str(exit_wave_path), str(optics_path))
    source = optics_path
    if not noise_free:
        parakeet.simulate.image(str(config_path), str(optics_path), str(image_path))
        source = image_path

    with h5py.File(source, "r") as handle:
        data = np.asarray(handle["entry/data/data"], np.float32)
    return data[0] if data.ndim == 3 else data


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively apply `overrides` onto a copy of `base`.

    Recursive rather than a top-level update, because Parakeet's configuration is nested and
    replacing a whole block would silently drop every default in it -- a `microscope` override that
    set only the lens would discard the detector.

    Args:
        base: The defaults.
        overrides: The keys to change. A `None` value replaces rather than recurses.

    Returns:
        A new dict.
    """
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged
