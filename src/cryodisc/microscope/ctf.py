"""The electron and the contrast it produces: a wavelength, and the CTF on top of it.

Two forms of the same physics, NOT interchangeable:

    ctf              radial in `k`, integrated by `resolution` over a defocus range; zeros average away
    ctf_astigmatic   the 2D grid where astigmatism and the exact zeros matter, fit/scored by `particles`

A test pins the second reducing exactly to the first. Nothing here is jitted; a parameter sweep is what a plan is for.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from cryodisc.microscope.plan import CollectionPlan


def aberration_coefficients(
    k_squared: Float[Array, " ..."], *, voltage_in_kv: float, spherical_aberration_in_mm: float
) -> tuple[Float[Array, " ..."], Float[Array, " ..."]]:
    """Split the aberration phase into its defocus-linear and defocus-free parts.

    `chi = pi lambda dz k^2 - pi Cs lambda^3 k^4 / 2`, which is `a dz + b`. `a` and `b` are returned
    separately because `resolution.defocus_averaged_ctf2` averages over a defocus range as
    `sinc(a dwidth / pi)` and needs them apart.

    Args:
        k_squared: `|k|^2` in inverse A squared.
        voltage_in_kv: Accelerating voltage.
        spherical_aberration_in_mm: Cs.

    Returns:
        `(a, b)` with `chi = a * defocus + b`, the defocus in angstroms.
    """
    lam = wavelength(voltage_in_kv)
    cs = spherical_aberration_in_mm * 1e7  # mm -> angstroms
    return jnp.pi * lam * k_squared, -0.5 * jnp.pi * cs * lam**3 * k_squared**2


def contrast(chi: Float[Array, " ..."], amplitude_contrast_ratio: float) -> Float[Array, " ..."]:
    """The CTF at a given aberration phase, which is where its SIGN convention is decided.

    `sqrt(1-Q^2) sin(chi) + Q cos(chi)` is `sin(chi + arcsin(Q))`, so the amplitude contrast is a
    pure phase offset; the leading minus makes particles dark on a light background.

    Args:
        chi: Aberration phase in radians, from `aberration_coefficients`.
        amplitude_contrast_ratio: Q.

    Returns:
        `-(sqrt(1 - Q^2) sin(chi) + Q cos(chi))`, negative at low k.
    """
    q = amplitude_contrast_ratio
    return -(jnp.sqrt(1.0 - q**2) * jnp.sin(chi) + q * jnp.cos(chi))


def wavelength(voltage_in_kv: float = 300.0) -> Float[Array, ""]:
    """Relativistic electron wavelength.

    Args:
        voltage_in_kv: Accelerating voltage.

    Returns:
        `12.2639 / sqrt(V (1 + 0.978476e-6 V))` in angstroms, 0.01969 at 300 kV.
    """
    volts = voltage_in_kv * 1e3
    return 12.2639 / jnp.sqrt(volts * (1.0 + 0.978476e-6 * volts))


def ctf(
    k: Float[Array, " ..."], *, defocus_in_angstroms: Float[Array, ""] | float, plan: CollectionPlan
) -> Float[Array, " ..."]:
    """The isotropic CTF at one defocus, unsquared.

    Args:
        k: Spatial frequency `1/d` in inverse A.
        defocus_in_angstroms: Underfocus, positive. Given per call rather than read from
            `plan.defocus_range`, which is the spread this is evaluated across.
        plan: The collection, for the voltage, Cs and Q.

    Returns:
        `-(sqrt(1 - Q^2) sin(chi) + Q cos(chi))`, negative at low k, so dark particles on a light
        background.
    """
    a, b = aberration_coefficients(
        k**2,
        voltage_in_kv=plan.voltage_in_kv,
        spherical_aberration_in_mm=plan.spherical_aberration_in_mm,
    )
    return contrast(a * defocus_in_angstroms + b, plan.amplitude_contrast_ratio)


def ctf_astigmatic(
    frequency_y: Float[Array, "..."],
    frequency_x: Float[Array, "..."],
    *,
    defocus_u: Float[Array, "..."] | float,
    defocus_v: Float[Array, "..."] | float,
    astigmatism_angle: Float[Array, "..."] | float = 0.0,
    voltage_in_kv: float = 300.0,
    spherical_aberration_in_mm: float = 2.7,
    amplitude_contrast_ratio: float = 0.07,
    phase_shift: Float[Array, "..."] | float = 0.0,
    b_factor: Float[Array, "..."] | float = 0.0,
) -> Float[Array, "..."]:
    """The CTF on a 2D frequency grid, with astigmatism, unsquared.

    Reduces exactly to `ctf` when `defocus_u == defocus_v` and the extras are zero (a test pins it).
    Optics parameters are taken one by one rather than as a `CollectionPlan`, the case this exists for
    being real metadata where the voltage, Cs and Q come from the file.

    Args:
        frequency_y: `k_y` in inverse A, from `fourier.frequency_vectors`.
        frequency_x: `k_x` in inverse A, the same shape.
        defocus_u: Underfocus along the astigmatism axis, in A, positive.
        defocus_v: Underfocus along the perpendicular axis, in A.
        astigmatism_angle: Azimuth of `defocus_u` in RADIANS, measured from the `x` axis.
        voltage_in_kv: Accelerating voltage.
        spherical_aberration_in_mm: Cs.
        amplitude_contrast_ratio: Q.
        phase_shift: Extra phase in radians, from a phase plate. Enters `chi` additively.
        b_factor: Envelope B in A^2, applied as `exp(-B k^2 / 4)`. Zero leaves the CTF undamped.

    Returns:
        `-(sqrt(1 - Q^2) sin(chi) + Q cos(chi)) exp(-B k^2 / 4)`, same shape as the grid.
    """
    k_squared = frequency_y**2 + frequency_x**2
    azimuth = jnp.arctan2(frequency_y, frequency_x)
    # per-mode defocus varies as cos(2 theta) about the mean, making the CTF linear in (mean, Ac, As)
    defocus = 0.5 * (defocus_u + defocus_v) + 0.5 * (defocus_u - defocus_v) * jnp.cos(
        2.0 * (azimuth - astigmatism_angle)
    )
    a, b = aberration_coefficients(
        k_squared, voltage_in_kv=voltage_in_kv, spherical_aberration_in_mm=spherical_aberration_in_mm
    )
    envelope = jnp.exp(-b_factor * k_squared / 4.0)
    return contrast(a * defocus + b + phase_shift, amplitude_contrast_ratio) * envelope
