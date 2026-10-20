"""What a collection would reach, predicted before it is collected.

The no-data engine, the only part of the package that touches no particles: `transfer` is the CTF
and the envelopes that damp it, and `noise` is the sigma^2(k) a stated exposure buys. Both read one
`microscope.CollectionPlan`; this is a consumer of the instrument, not its owner.
"""

from cryodisc.resolution.noise import noise_spectrum
from cryodisc.resolution.transfer import (
    defocus_averaged_ctf2,
    dqe,
    effective_transfer_squared,
    envelope_squared,
    flexibility_envelope,
    radiation_envelope,
)

__all__ = [
    "defocus_averaged_ctf2",
    "dqe",
    "effective_transfer_squared",
    "envelope_squared",
    "flexibility_envelope",
    "noise_spectrum",
    "radiation_envelope",
]
