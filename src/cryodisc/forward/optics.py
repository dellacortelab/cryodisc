"""CTF and envelope parameters, and their mapping to a cryoJAX transfer theory.

The optics are cryoJAX's `AstigmaticCTF` times a `FourierGaussian` envelope (exp(-B k^2 / 4)),
where one effective B absorbs the aggregate high-frequency damping.

`CTFParams` is an `equinox.Module`, so passing an array for one field and floats for the rest
gives per-particle optics under `eqx.filter_vmap` (plain `jax.vmap` cannot map the float leaves).
"""

import cryojax.simulator as cxs
import equinox as eqx
from cryojax import ndimage as cxn
from jaxtyping import Array, Float

from cryodisc.microscope import CollectionPlan


class CTFParams(eqx.Module):
    """Optics for one image, defaulting to typical 300 kV values.

    Args:
        defocus_in_angstroms: Underfocus, > 0.
        astigmatism_in_angstroms: df_max - df_min.
        astigmatism_angle: Azimuth of df_max in degrees, on cryoJAX's (y, x)-ordered
            frequency grid.
        spherical_aberration_in_mm: Cs.
        amplitude_contrast_ratio: Amplitude-contrast fraction.
        b_factor: Envelope B in A^2, distinct from per-atom B-factors.
    """

    defocus_in_angstroms: float | Array = 12000.0
    astigmatism_in_angstroms: float | Array = 0.0
    astigmatism_angle: float | Array = 0.0
    spherical_aberration_in_mm: float | Array = 2.7
    amplitude_contrast_ratio: float | Array = 0.07
    b_factor: float | Array = 80.0

    @classmethod
    def from_plan(cls, plan: CollectionPlan, defocus_in_angstroms: Float[Array, " ..."] | float) -> "CTFParams":
        """The 2D optics a `CollectionPlan` describes, so a stack and its templates share them.

        Args:
            plan: The collection, for the Cs and Q. Its voltage is absent because this class has
                none: cryoJAX reads it from the image config `ConformerPair.models` built.
            defocus_in_angstroms: Underfocus, a scalar for shared optics or (n,) for per-particle.

        Returns:
            The optics, with `b_factor` zero: `resolution` already carries the envelopes on the noise
            side, and a B here would damp them twice.
        """
        return cls(
            defocus_in_angstroms=defocus_in_angstroms,
            spherical_aberration_in_mm=plan.spherical_aberration_in_mm,
            amplitude_contrast_ratio=plan.amplitude_contrast_ratio,
            b_factor=0.0,
        )

    def build_transfer_theory(self) -> cxs.ContrastTransferTheory:
        """Map these parameters onto a cryoJAX `ContrastTransferTheory`.

        Returns:
            The CTF times its envelope.
        """
        return cxs.ContrastTransferTheory(
            ctf=cxs.AstigmaticCTF(
                defocus_in_angstroms=self.defocus_in_angstroms,
                astigmatism_in_angstroms=self.astigmatism_in_angstroms,
                astigmatism_angle=self.astigmatism_angle,
                spherical_aberration_in_mm=self.spherical_aberration_in_mm,
            ),
            envelope=cxn.FourierGaussian(amplitude=1.0, b_factor=self.b_factor),
            amplitude_contrast_ratio=self.amplitude_contrast_ratio,
        )
