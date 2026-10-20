"""cryoJAX forward model for template generation.

Atomic models and voxel maps both go through cryoJAX's projection, so there is no hand-written
projector. `CryoJAXModel` is an `equinox.Module` (a pytree); poses are Euler angles (intrinsic zyz,
degrees), so the whole render path stays inside jit. The voxel path defaults to `sinc_correction=True`
and `pad_scale=2.0` (neither a cryoJAX default) for projection accuracy. `precompute_mode="all"` is
load-bearing: a config first touched inside jit otherwise leaks a tracer through its coordinate grids.
"""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from cryojax.jax_util import FloatLike
from jaxtyping import Array, Complex, Float

from cryodisc.forward.optics import CTFParams

_FOURIER_PAD_SCALE = 2.0
"""Fourier-grid oversampling for a voxel-map projection. Below 2.0 the slice-interpolation error
grows quickly; above it, memory does. A constant because no caller ever needed another value."""


class CryoJAXModel(eqx.Module):
    """Noiseless projection templates from an atomic model or a voxel map.

    Build with `from_structure` or `from_voxel_grid` rather than directly.

    Args:
        volume: Scattering-potential representation to project.
        image_config: Image shape, pixel size, and voltage.
        volume_integrator: Projection method. The default dispatches on `volume`.
    """

    volume: cxs.AbstractVolumeRepresentation
    image_config: cxs.BasicImageConfig
    volume_integrator: cxs.AbstractVolumeIntegrator = cxs.AutoVolumeProjection()

    @classmethod
    def from_structure(
        cls,
        source: Path | str | pd.DataFrame,
        box_size: int = 128,
        pixel_size: float = 1.0,
        voltage_in_kilovolts: float = 300.0,
        center: bool = True,
        projection_shape: int | None = None,
        form_factor: str = "peng",
    ) -> Self:
        """Build from an atomic structure, projecting it as a Gaussian mixture.

        Args:
            source: A structure file in any format cryoJAX can tabulate, or an already-tabulated
                atom table in the `mmdf` schema, as `conformers` produces.
            box_size: Image size in pixels.
            pixel_size: Pixel size in A.
            voltage_in_kilovolts: Accelerating voltage.
            center: Move the structure onto its own center of mass. `conformers` passes `False`:
                centering two conformers separately undoes the superposition that put them in one frame.
            form_factor: `"peng"` (tabulated five-Gaussian elastic scattering factors) or `"z"`
                (E7 ablation: one Gaussian per atom weighted by atomic number at a fixed width).
            projection_shape: Evaluate the projection on this plane and zero-pad to `box_size`, rather
                than rasterizing the mostly-empty box. Density outside the plane is LOST, so a caller
                sizing this owns the truncation; `None` rasterizes the whole box.

        Returns:
            The model.
        """
        volume = cxs.load_tabulated_volume(
            source, output_type=cxs.GaussianMixtureVolume, pdb_options={"center": center}
        )
        if form_factor == "z":
            # form_factor='z' needs per-atom elements, which survive only on the atom-table route;
            # cryoJAX's own tabulation has already discarded them from a path.
            if isinstance(source, (str, Path)):
                raise ValueError(
                    "form_factor='z' needs an atom table carrying `atomic_number`, not a path: "
                    "the element is not recoverable once cryoJAX has tabulated the structure"
                )
            volume = _to_atomic_number_weighting(volume, source["atomic_number"].to_numpy())
        elif form_factor != "peng":
            raise ValueError(f"form_factor must be 'peng' or 'z', got {form_factor!r}")
        return cls(
            volume=volume,
            image_config=_image_config(box_size, pixel_size, voltage_in_kilovolts),
            volume_integrator=(
                cxs.AutoVolumeProjection()
                if projection_shape is None
                else cxs.GaussianMixtureProjection(shape=(int(projection_shape), int(projection_shape)))
            ),
        )

    @classmethod
    def from_voxel_grid(
        cls,
        real_voxel_grid: Float[Array, "dim dim dim"] | Float[np.ndarray, "dim dim dim"],
        pixel_size: float = 1.0,
        box_size: int | None = None,
        voltage_in_kilovolts: float = 300.0,
    ) -> Self:
        """Build from a real-space voxel map, projecting it by Fourier-slice extraction.

        Takes an array rather than a path, so reading the map is the caller's business.

        Args:
            real_voxel_grid: Cubic map, ordered (z, y, x) to match cryoJAX and `mrcfile`.
            pixel_size: Voxel size in A. Templates are not transferable across pixel sizes.
            box_size: Image size in pixels, defaulting to the map's own dimension.
            voltage_in_kilovolts: Accelerating voltage.

        Returns:
            The model.
        """
        return cls(
            volume=cxs.FourierVoxelGridVolume.from_real_voxel_grid(
                real_voxel_grid, sinc_correction=True, pad_scale=_FOURIER_PAD_SCALE
            ),
            image_config=_image_config(
                real_voxel_grid.shape[-1] if box_size is None else box_size,
                pixel_size,
                voltage_in_kilovolts,
            ),
        )

    def _render(
        self,
        phi_angle: FloatLike,
        theta_angle: FloatLike,
        psi_angle: FloatLike,
        ctf_params: CTFParams | None,
    ) -> Float[Array, "y x"]:
        """Render one template, without jit so the batched path can vmap it."""
        pose = cxs.EulerAnglePose(phi_angle=phi_angle, theta_angle=theta_angle, psi_angle=psi_angle)
        return self.render_at_pose(pose, ctf_params)

    def render_at_pose(
        self,
        pose: cxs.AbstractPose,
        ctf_params: CTFParams | None = None,
    ) -> Float[Array, "y x"]:
        """Render one template at an arbitrary cryoJAX pose.

        The escape hatch from the Euler-angle API: Euler angles cannot compose two rotations without a
        chart that degenerates at theta = 0, so `rungs.local` perturbs a `QuaternionPose`.

        Args:
            pose: Any cryoJAX pose.
            ctf_params: Optics. `None` changes the QUANTITY, not just the optics: cryoJAX returns the
                bare integrated potential rather than contrast, so it is a diagnostic and not a
                template. Anything paired with an absolute noise level needs a `CTFParams`.

        Returns:
            The image in real space, as contrast against the incident beam when optics were given.
        """
        image_model = cxs.make_image_model(
            self.volume,
            self.image_config,
            pose,
            transfer_theory=(None if ctf_params is None else ctf_params.build_transfer_theory()),
            volume_integrator=self.volume_integrator,
            quantity_mode="none" if ctf_params is None else "contrast",
        )
        return image_model.simulate()

    @eqx.filter_jit
    def forward(
        self,
        phi_angle: FloatLike,
        theta_angle: FloatLike,
        psi_angle: FloatLike,
        ctf_params: CTFParams | None = None,
    ) -> Float[Array, "y x"]:
        """Render one template.

        Args:
            phi_angle: First Euler angle in degrees, cryoJAX intrinsic zyz.
            theta_angle: Second Euler angle in degrees.
            psi_angle: Third Euler angle in degrees.
            ctf_params: Optics. `None` gives a bare projection with no CTF.

        Returns:
            The image in real space.
        """
        return self._render(phi_angle, theta_angle, psi_angle, ctf_params)

    @eqx.filter_jit
    def forward_batch(
        self,
        angles: Float[Array, "n 3"],
        ctf_params: CTFParams | None = None,
    ) -> Float[Array, "n y x"]:
        """Render a stack of templates.

        Args:
            angles: (n, 3) Euler angles in degrees, one row per template.
            ctf_params: Optics. Float fields are shared by every template and array fields vary per
                template, so one `CTFParams` with an (n,) `defocus_in_angstroms` gives a realistic
                stack. `None` gives bare projections, which `render_at_pose` warns are not contrast.

        Returns:
            The images in real space.
        """
        # in_axes is filter_vmap's default `if_array(0)`, which maps the array leaves of ctf_params
        # and broadcasts the float ones, so the shared and per-particle cases need no separate path
        return eqx.filter_vmap(lambda row, ctf: self._render(row[0], row[1], row[2], ctf))(angles, ctf_params)

    @eqx.filter_jit
    def bare_contrast(self, angles: Float[Array, "p 3"]) -> Complex[Array, "p y x"]:
        """A pose bank in contrast units with no CTF, as `fft2`, ready for a transfer to shape it.

        What `microscope.Transfer` and `rungs.refine.refine_poses` both want: the CTF varies per
        defocus group and folds onto the data, so the bank it multiplies must carry none. The sign
        comes back with the transfer, `_FLAT_OPTICS` having rendered through a CTF of exactly -1.

        Chunked rather than one `forward_batch`; `filter_jit` is required because outside jit the inner
        scan recompiles every call. See docs/rationale/forward-and-rungs.md.

        Args:
            angles: (p, 3) Euler angles in degrees, cryoJAX intrinsic zyz.

        Returns:
            `fft2` of the templates, unshifted.
        """
        render = jax.lax.map(
            lambda row: self._render(row[0], row[1], row[2], _FLAT_OPTICS), angles, batch_size=_RENDER_CHUNK
        )
        return -jnp.fft.fft2(render)


def _to_atomic_number_weighting(
    volume: "cxs.GaussianMixtureVolume", atomic_numbers: "np.ndarray"
) -> "cxs.GaussianMixtureVolume":
    """Collapse the five-Gaussian scattering factor to one Gaussian per atom weighted by Z.

    E7's form-factor ablation. Z is taken from the atom table, the amplitudes normalized so the
    comparison is of SHAPE not scale, the width fixed from the tabulated variances. See
    docs/rationale/forward-and-rungs.md.

    Args:
        volume: A tabulated `GaussianMixtureVolume`, five Gaussians per atom.
        atomic_numbers: Z per atom, in the volume's own atom order. Zero-amplitude padding atoms
            must carry zero, so they stay invisible.

    Returns:
        The same positions with one Gaussian each, amplitude proportional to Z and width fixed.

    Raises:
        ValueError: If the volume carries no per-atom Gaussians, or if the two lengths disagree.
    """
    amplitudes = jnp.asarray(volume.amplitudes)
    variances = jnp.asarray(volume.variances)
    if amplitudes.ndim != 2:
        raise ValueError(f"expected per-atom Gaussians, got amplitudes of shape {amplitudes.shape}")
    z = jnp.asarray(np.asarray(atomic_numbers, dtype=float))
    if z.shape[0] != amplitudes.shape[0]:
        raise ValueError(f"{z.shape[0]} atomic numbers against {amplitudes.shape[0]} atoms")

    # cryoJAX's Gaussians are NORMALIZED (amplitude 1 integrates to 1.0 at any variance), so the
    # amplitude is the zero-frequency weight with no (2 pi sigma^2)^(3/2) factor to carry.
    fixed_variance = float(jnp.sum(amplitudes * variances) / jnp.sum(amplitudes))

    total = jnp.sum(amplitudes)
    scale = jnp.where(jnp.sum(z) > 0, total / jnp.maximum(jnp.sum(z), 1e-30), 0.0)
    new_amplitude = z * scale

    return cxs.GaussianMixtureVolume(
        positions=volume.positions,
        amplitudes=new_amplitude[:, None],
        variances=jnp.full_like(new_amplitude[:, None], fixed_variance),
    )


def pad_to_common_atom_count(models: "Sequence[CryoJAXModel]") -> tuple["CryoJAXModel", ...]:
    """Atomic models padded to one atom count, so a caller may stack their volumes.

    Needed when two states of one molecule differ in atom count (a state-specific ligand). The padding
    is EXACTLY zero density: a padded atom takes amplitude zero and copies the first real atom's
    position and variances, contributing nothing to any projection while staying well-formed.

    Args:
        models: Models built from atomic structures. All must be on the same image config.

    Returns:
        The models in the order given, each with `volume` padded to the largest atom count. Returned
        unchanged when they already agree.

    Raises:
        TypeError: If any model holds a voxel map rather than atoms, which has no atom axis to pad.
    """
    volumes = [model.volume for model in models]
    for model, volume in zip(models, volumes, strict=True):
        if not isinstance(volume, cxs.GaussianMixtureVolume):
            raise TypeError(
                f"pad_to_common_atom_count pads an ATOMIC model and this one holds "
                f"{type(volume).__name__}, which has no atom axis. Two voxel maps of one pair are "
                "already on a common grid and need no padding."
            )
    widest = max(volume.positions.shape[0] for volume in volumes)
    padded = []
    for model, volume in zip(models, volumes, strict=True):
        missing = widest - volume.positions.shape[0]
        if missing == 0:
            padded.append(model)
            continue
        filler = jnp.zeros((missing, volume.amplitudes.shape[1]), volume.amplitudes.dtype)
        grown = eqx.tree_at(
            lambda v: (v.positions, v.amplitudes, v.variances),
            volume,
            (
                jnp.concatenate([volume.positions, jnp.broadcast_to(volume.positions[:1], (missing, 3))]),
                jnp.concatenate([volume.amplitudes, filler]),
                jnp.concatenate([volume.variances, jnp.broadcast_to(volume.variances[:1], filler.shape)]),
            ),
        )
        padded.append(eqx.tree_at(lambda m: m.volume, model, grown))
    return tuple(padded)


_FLAT_OPTICS = CTFParams(
    defocus_in_angstroms=0.0, spherical_aberration_in_mm=0.0, amplitude_contrast_ratio=1.0, b_factor=0.0
)
"""Optics whose CTF is identically -1, so a render is contrast with no CTF shaping.

`chi` vanishes with the defocus and Cs, leaving `-(sqrt(1 - Q^2) sin chi + Q cos chi) = -Q`, and
`Q = 1` makes that exactly -1. The safe way around the `ctf_params=None` hazard `render_at_pose` warns of.
"""

_RENDER_CHUNK = int(os.environ.get("CRYOEM_RENDER_CHUNK", "100"))
"""Poses per render batch, bounding cryoJAX's `(batch, box, n_atoms, 5)` temporary. See `particles.sim`.

The temporary is `chunk * box * n_atoms * 5 * 4` bytes, which the default can size beyond a large
card's memory for a big assembly; `CRYOEM_RENDER_CHUNK` lowers it. Batching is numerically inert
(`jax.lax.map` returns the same array either way), so this trades speed for memory alone.
"""


def _image_config(box_size: int, pixel_size: float, voltage_in_kilovolts: float) -> cxs.BasicImageConfig:
    """Build an image config with its coordinate grids already filled in."""
    return cxs.BasicImageConfig(
        shape=(box_size, box_size),
        pixel_size=pixel_size,
        voltage_in_kilovolts=voltage_in_kilovolts,
        precompute_mode="all",
    )
