"""A rendered pose bank and where it came from, so two engines can be compared and not confused.

Two tiers, NOT interchangeable: a MODEL (`CryoJAXModel`) is differentiable and callable at any
CONTINUOUS pose; a BANK (this) is a concrete array of templates at enumerated poses plus its
provenance. A bank reaches the pool rungs (`marginal`, `best_match`, `achieved`, `particles.em`) and
not the model-tier rungs, so a ladder on one reports the rest BLANK. Two banks may be compared only
when rendered at the same poses, box, pixel, voltage and defocus, which `assert_comparable` enforces.
Images are stored in REAL space; the sign is fixed at construction so both engines share a convention.
"""

from dataclasses import dataclass
from typing import Self

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float

from cryodisc.forward.cryojax_model import CryoJAXModel
from cryodisc.forward.optics import CTFParams

CTF_FREE = None
"""`defocus_in_angstroms` of a bank rendered with no CTF at all, which is not a defocus of zero.

Zero defocus is a real optical setting with a real transfer; CTF-free means the transfer was never
applied, so a `microscope.Transfer` still has to be. Conflating them double-counts or drops the CTF.
"""


@dataclass(frozen=True)
class TemplateBank:
    """Templates at enumerated poses, and the render they came out of.

    Build with `from_model` or, on a cluster, `parakeet.render_bank`. Construct directly only for a
    bank read back off disk, where the provenance has to be restated.

    Args:
        images: `(p, y, x)` real-space contrast, in the convention `rungs.marginal.pose_log_ratio`
            takes via `fourier`.
        angles: `(p, 3)` Euler angles in degrees, cryoJAX intrinsic zyz. Kept so a bank can be
            compared and marginalized against a quadrature weight.
        pixel_size: A/px. Templates are not transferable across pixel sizes.
        voltage_in_kv: Accelerating voltage the render assumed.
        defocus_in_angstroms: The single defocus the CTF was applied at, or `CTF_FREE` when none was.
        engine: What rendered it -- `"cryojax"`, `"parakeet"`, or whatever read it off disk.
    """

    images: Float[Array, "p y x"]
    angles: Float[Array, "p 3"]
    pixel_size: float
    voltage_in_kv: float
    defocus_in_angstroms: float | None
    engine: str

    def __post_init__(self) -> None:
        """Refuse a malformed bank at construction rather than deep inside a rung.

        Raises:
            ValueError: If the images are not `(p, box, box)`, if the pose count disagrees with the
                template count, or if the bank is empty.
        """
        if self.images.ndim != 3 or self.images.shape[1] != self.images.shape[2]:
            raise ValueError(f"images must be (p, box, box) with a square box, got {self.images.shape}")
        if self.images.shape[0] == 0:
            raise ValueError("the bank is empty")
        if self.angles.shape != (self.images.shape[0], 3):
            raise ValueError(f"angles must be ({self.images.shape[0]}, 3), got {self.angles.shape}")

    def __len__(self) -> int:
        """Number of templates."""
        return int(self.images.shape[0])

    @property
    def box_size(self) -> int:
        """Image size in pixels, derived so it cannot disagree with the images."""
        return int(self.images.shape[-1])

    @property
    def fourier(self) -> Complex[Array, "p y x"]:
        """`fft2` of the templates, unshifted -- what every pool rung consumes.

        A CTF-free bank still needs a `microscope.Transfer` applied; one rendered at a defocus
        already carries it.
        """
        return jnp.fft.fft2(self.images)

    @property
    def is_ctf_free(self) -> bool:
        """Whether a `microscope.Transfer` still has to be applied to this bank."""
        return self.defocus_in_angstroms is CTF_FREE

    def provenance(self) -> tuple[int, float, float, float | None]:
        """The render settings two banks must share, as one comparable tuple.

        Returns:
            `(box_size, pixel_size, voltage_in_kv, defocus_in_angstroms)`. The engine is NOT in it,
            two engines agreeing being the thing a comparison tests.
        """
        return (self.box_size, self.pixel_size, self.voltage_in_kv, self.defocus_in_angstroms)

    def assert_comparable(
        self, other: "TemplateBank", *, angle_tolerance: float = 1e-4, setting_tolerance: float = 1e-6
    ) -> None:
        """Refuse a comparison between banks that were not rendered the same way.

        Args:
            other: The bank to compare against.
            angle_tolerance: Degrees each pose may differ by. Not zero, since two engines reach the
                same nominal angle through different float paths; not loose, since a degree of pose
                error reads as a forward-model deficit on the gate this feeds.
            setting_tolerance: Relative agreement required of pixel size, voltage and defocus. Not
                zero, since `from_model` reads its settings back through a float32 config while
                `parakeet.render_bank` keeps the Python float it was given.

        Raises:
            ValueError: If the counts, the render settings, or the poses disagree, naming which.
        """
        if len(self) != len(other):
            raise ValueError(f"{self.engine} has {len(self)} templates and {other.engine} has {len(other)}")
        if not _settings_agree(self.provenance(), other.provenance(), setting_tolerance):
            raise ValueError(
                f"{self.engine} rendered at (box, pixel, kV, defocus) = {self.provenance()} and "
                f"{other.engine} at {other.provenance()}. A gate comparing two banks measures the "
                "engines only when everything else about the render is shared."
            )
        gap = float(np.max(np.abs(np.asarray(self.angles) - np.asarray(other.angles))))
        if gap > angle_tolerance:
            raise ValueError(
                f"{self.engine} and {other.engine} differ by up to {gap:.4g} degrees in pose. A "
                "pose mismatch reads as a forward-model deficit and no spectral statistic sees it."
            )

    @classmethod
    def from_model(
        cls,
        model: CryoJAXModel,
        angles: Float[Array, "p 3"],
        *,
        ctf_params: CTFParams | None = None,
        defocus_in_angstroms: float | None = CTF_FREE,
    ) -> Self:
        """Render a bank with cryoJAX.

        Args:
            model: The forward model. Its `image_config` supplies the box, pixel and voltage.
            angles: `(p, 3)` Euler angles in degrees.
            ctf_params: Optics to render through, or `None` for a CTF-free bank. Passing optics
                WITHOUT `defocus_in_angstroms` is refused: the provenance would claim CTF-free and
                the transfer would be applied twice.
            defocus_in_angstroms: The defocus `ctf_params` carries. Must be given exactly when
                `ctf_params` is.

        Returns:
            The bank.

        Raises:
            ValueError: If `ctf_params` and `defocus_in_angstroms` disagree about whether a CTF was
                applied.
        """
        if (ctf_params is None) != (defocus_in_angstroms is CTF_FREE):
            raise ValueError(
                "ctf_params and defocus_in_angstroms must be given together: a bank that records "
                "CTF_FREE while carrying a CTF has its transfer applied twice, and one that records "
                "a defocus while carrying none has it applied never. Neither raises downstream."
            )
        config = model.image_config
        if ctf_params is None:
            # bare_contrast already carries the sign flip that puts a particle dark on a light
            # background; ifft2 rather than a second render keeps the two paths on one convention
            images = jnp.real(jnp.fft.ifft2(model.bare_contrast(angles)))
        else:
            images = model.forward_batch(angles, ctf_params)
        return cls(
            images=images,
            angles=jnp.asarray(angles),
            pixel_size=float(np.asarray(config.pixel_size).reshape(-1)[0]),
            voltage_in_kv=float(np.asarray(config.voltage_in_kilovolts).reshape(-1)[0]),
            defocus_in_angstroms=defocus_in_angstroms,
            engine="cryojax",
        )


def _settings_agree(
    left: tuple[int, float, float, float | None], right: tuple[int, float, float, float | None], tolerance: float
) -> bool:
    """Whether two `provenance()` tuples describe the same render.

    The box size is compared exactly, being a count; the three physical settings relatively, since
    one engine reads them back through a float32 config and the other keeps the Python float.
    `CTF_FREE` is compared by identity with a defocus, a bank that never had a CTF and one at a
    defocus of zero being different things.

    Args:
        left: One bank's provenance.
        right: The other's.
        tolerance: Relative agreement required of the physical settings.

    Returns:
        Whether the two renders are the same.
    """
    if left[0] != right[0]:
        return False
    for a, b in zip(left[1:], right[1:], strict=True):
        if (a is CTF_FREE) != (b is CTF_FREE):
            return False
        if a is not CTF_FREE and b is not CTF_FREE and abs(a - b) > tolerance * max(abs(a), abs(b), 1.0):
            return False
    return True
