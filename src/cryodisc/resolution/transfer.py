"""The effective contrast transfer: the CTF folded together with everything that damps it.

    C_eff(k) = sqrt(DQE(k)) E_rad(k) E_conf(k) CTF(k)
    <C_eff^2>(k) = <CTF^2>(k) DQE(k) E_rad(k)^2 E_conf(k)^2

with the defocus average filling the CTF zeros. `envelope_squared` is the everything-but-the-CTF
half, which the noise model divides by. `rmsf` is kept out of the plan, being the target's
flexibility rather than the session's. The RADIAL path, a function of `k` alone; nothing is jitted.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from cryodisc.microscope import CollectionPlan
from cryodisc.microscope.ctf import aberration_coefficients

# DQE vs fraction of Nyquist, digitized from McMullan, Faruqi, Clare & Henderson (2014)
# Ultramicroscopy 147:156-163, Fig 4 (K2 Summit, 300 keV, counting mode). k=0 is the low-frequency
# plateau, not the plotted DQE(0) dip. Read off a figure, so two significant digits.
_DQE_FRACTION = jnp.linspace(0.0, 1.0, 21)
_DQE_VALUE = jnp.array(
    [0.78, 0.80, 0.82, 0.81, 0.79, 0.76, 0.72, 0.68, 0.64, 0.61, 0.57,
     0.54, 0.50, 0.47, 0.43, 0.40, 0.37, 0.33, 0.30, 0.27, 0.24]
)  # fmt: skip


def defocus_averaged_ctf2(k: Float[Array, " ..."], *, plan: CollectionPlan) -> Float[Array, " ..."]:
    """`<CTF^2>(k)` over the plan's defocus spread, in closed form.

    A defocus range has no common zero, so the pooled transfer never vanishes; averaging the SQUARE
    expresses that. With `CTF = -sin(chi + phi)`, `phi = arcsin(Q)` and `chi = a dz + b`,

        <CTF^2>(k) = [1 - cos(2 (a dbar + b) + 2 phi) sinc(a dwidth / pi)] / 2

    for `a = pi lambda k^2`, `b = -pi Cs lambda^3 k^4 / 2`. Exact. At `dwidth = 0` the sinc is one
    and it reduces to `ctf(k, dbar)^2`.

    Returns:
        `<CTF^2>(k)`, in [0, 1].
    """
    low, high = plan.defocus_range
    a, b = aberration_coefficients(
        k**2, voltage_in_kv=plan.voltage_in_kv, spherical_aberration_in_mm=plan.spherical_aberration_in_mm
    )
    phase = 2.0 * (a * 0.5 * (low + high) + b) + 2.0 * jnp.arcsin(plan.amplitude_contrast_ratio)
    return 0.5 * (1.0 - jnp.cos(phase) * jnp.sinc(a * (high - low) / jnp.pi))


def dqe(k: Float[Array, " ..."], *, nyquist: float) -> Float[Array, " ..."]:
    """Detector quantum efficiency, interpolated on the measured curve.

    Args:
        k: Spatial frequency in inverse A.
        nyquist: The DETECTOR's Nyquist in inverse A, set by the physical pixel and not by the
            scoring grid. Undefaulted so a cyc/px 0.5 cannot silently rescale the curve; pass
            `CollectionPlan.nyquist`.

    Returns:
        DQE(k), a power quantity, so it enters `<C_eff^2>` unsquared. Held at the endpoint values
        beyond the measured range.
    """
    return jnp.interp(jnp.clip(k / nyquist, 0.0, 1.0), _DQE_FRACTION, _DQE_VALUE)


def radiation_envelope(k: Float[Array, " ..."], *, dose: float) -> Float[Array, " ..."]:
    """Grant-Grigorieff dose-weighting, as an amplitude.

    Integrating the per-frequency critical exposure `Ne(k)` over a total exposure gives the
    amplitude actually retained, so it falls with dose: the late frames of a movie contribute
    high-frequency noise and no high-frequency signal.

    Args:
        k: Spatial frequency in inverse A.
        dose: Total electron exposure in e/A^2.

    Returns:
        `2 Ne (1 - exp(-dose / 2 Ne)) / dose` with `Ne = 0.245 k^-1.665 + 2.81`, tending to one as
        `k -> 0`.
    """
    critical = 0.245 * jnp.maximum(k, 1e-6) ** -1.665 + 2.81
    # expm1, not `1 - exp`: Ne diverges as k -> 0, so the exponent underflows the float32 epsilon
    # and the difference cancels to exactly zero at the DC pixel, where the envelope is one
    return -2.0 * critical * jnp.expm1(-dose / (2.0 * critical)) / dose


def conformational_b_factor(rmsf: float | Float[Array, ""]) -> Float[Array, ""]:
    """The B factor a per-axis RMS fluctuation corresponds to, `B = 8 pi^2 rmsf^2`, in A^2.

    One home for the conversion, so `flexibility_envelope` and the appendix ledger cannot disagree.
    Returns a jnp scalar, not a float, so `rmsf` may be a tracer under vmap.

    Args:
        rmsf: Per-axis RMS fluctuation within one conformational state, in A.

    Returns:
        The B factor in A^2.
    """
    return 8.0 * jnp.pi**2 * rmsf**2


def flexibility_envelope(k: Float[Array, " ..."], *, rmsf: float) -> Float[Array, " ..."]:
    """Debye-Waller damping from intra-state disorder, as an amplitude.

    Args:
        k: Spatial frequency in inverse A.
        rmsf: Per-axis RMS fluctuation within one conformational state, in A.

    Returns:
        `exp(-B k^2 / 4)` with `B` from `conformational_b_factor`.
    """
    return jnp.exp(-conformational_b_factor(rmsf) / 4.0 * k**2)


def envelope_squared(k: Float[Array, " ..."], *, plan: CollectionPlan, rmsf: float) -> Float[Array, " ..."]:
    """`DQE(k) E_rad(k)^2 E_conf(k)^2` -- every loss except the CTF, applied exactly once.

    The seam between the two engines. `effective_transfer_squared` multiplies it by `<CTF^2>` to
    damp a signal; `noise.noise_spectrum` divides a shot-noise floor by it to inflate a noise, the
    template carrying the CTF. Same statement, so a factor added here reaches both.

    Args:
        k: Spatial frequency in inverse A.
        plan: The collection, which supplies the dose and the Nyquist.
        rmsf: Per-axis intra-state RMS fluctuation in A.

    Returns:
        The product, dimensionless and at most one.
    """
    return (
        dqe(k, nyquist=plan.nyquist)
        * radiation_envelope(k, dose=plan.dose) ** 2
        * flexibility_envelope(k, rmsf=rmsf) ** 2
    )


def effective_transfer_squared(k: Float[Array, " ..."], *, plan: CollectionPlan, rmsf: float) -> Float[Array, " ..."]:
    """`<C_eff^2>(k)`, every loss applied exactly once.

    Args:
        k: Spatial frequency in inverse A.
        plan: The collection, which supplies the dose and the Nyquist as well as the optics, so the
            four factors cannot disagree about them.
        rmsf: Per-axis intra-state RMS fluctuation in A.

    Returns:
        `<CTF^2> DQE E_rad^2 E_conf^2`, dimensionless and at most one.
    """
    return defocus_averaged_ctf2(k, plan=plan) * envelope_squared(k, plan=plan, rmsf=rmsf)
