"""The microscope: the session as planned, and the optics as they reached each particle.

    plan       `CollectionPlan`, the configuration root, read by every engine in the package
    ctf        the relativistic electron, and the contrast it produces, radial and astigmatic
    transfer   that CTF per defocus group, gathered per particle, which the ladder scores through

Everything is re-exported here. Three paths out of a plan are NOT interchangeable: `ctf` is radial,
integrated over a defocus range by `resolution`; `Transfer` is radial per defocus GROUP, scored by
the ladder and the EM fit; `forward.optics.CTFParams.from_plan` is the per-particle 2D path.
"""

from cryodisc.microscope.ctf import (
    aberration_coefficients,
    contrast,
    ctf,
    ctf_astigmatic,
    wavelength,
)
from cryodisc.microscope.plan import CollectionPlan
from cryodisc.microscope.transfer import Transfer, defocus_groups_for, fold_transfer

__all__ = [
    "CollectionPlan",
    "Transfer",
    "aberration_coefficients",
    "contrast",
    "ctf",
    "ctf_astigmatic",
    "defocus_groups_for",
    "fold_transfer",
    "wavelength",
]
