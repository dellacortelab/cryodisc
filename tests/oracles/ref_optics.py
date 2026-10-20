"""Independent reference for the microscope: the electron wavelength, the CTF, and its zeros.

Nothing here imports `cryodisc`; the wavelength comes from fundamental constants, not
the fit production uses, whose coefficients could be mistyped with no internal check.

    chi(k)  = pi lambda dz k^2 - (pi/2) Cs lambda^3 k^4 + phase_shift
    CTF(k)  = -[sqrt(1 - Q^2) sin(chi) + Q cos(chi)]
    u = k^2 = [dz -+ sqrt(dz^2 - 2 n Cs lambda)] / (Cs lambda^2)   (zeros, in closed form)
Minus root is the physical branch. Derivations: docs/rationale/tests/oracles.md
"""

import math

import numpy as np

__all__ = [
    "AMPLITUDE_CONTRAST_PHASE",
    "PLANCK",
    "ctf",
    "ctf_chi",
    "ctf_zeros",
    "electron_wavelength",
    "reference_wavelength_angstrom",
]

PLANCK = 6.62607015e-34
"""Planck's constant in J s. Exact, by the 2019 SI redefinition."""

_ELECTRON_MASS = 9.1093837015e-31
"""Electron rest mass in kg, CODATA 2018."""

_ELEMENTARY_CHARGE = 1.602176634e-19
"""Elementary charge in C. Exact, by the 2019 SI redefinition."""

_SPEED_OF_LIGHT = 299792458.0
"""Speed of light in m/s. Exact, by definition of the meter."""

reference_wavelength_angstrom = {300.0: 0.0197, 200.0: 0.0251, 120.0: 0.0335}
"""Published relativistic electron wavelengths in angstroms: an external anchor, independent of both
the derivation below and production's fit."""


def electron_wavelength(voltage_in_kv: float) -> float:
    """The relativistic electron wavelength in angstroms, from fundamental constants.

    Args:
        voltage_in_kv: Accelerating voltage.

    Returns:
        `h / sqrt(2 m e V (1 + e V / (2 m c^2)))` converted to angstroms.

    Raises:
        ValueError: If the voltage is not finite and positive.
    """
    volts = float(voltage_in_kv) * 1e3
    if not math.isfinite(volts) or volts <= 0.0:
        raise ValueError(f"voltage must be finite and positive, got {voltage_in_kv!r}")
    energy = _ELEMENTARY_CHARGE * volts
    rest_energy = _ELECTRON_MASS * _SPEED_OF_LIGHT**2
    momentum = math.sqrt(2.0 * _ELECTRON_MASS * energy * (1.0 + energy / (2.0 * rest_energy)))
    return PLANCK / momentum * 1e10  # meters -> angstroms


AMPLITUDE_CONTRAST_PHASE = np.arcsin
"""`arcsin(Q)` is the phase offset the amplitude-contrast term is equivalent to.

Exposed as the identity `sqrt(1-Q^2) sin(chi) + Q cos(chi) = sin(chi + arcsin(Q))`, which the tests
use to check that term rather than restating its algebra.
"""


def ctf_chi(
    k: np.ndarray,
    *,
    defocus: float,
    wavelength: float,
    spherical_aberration_in_mm: float = 2.7,
    phase_shift: float = 0.0,
) -> np.ndarray:
    """The CTF phase `chi(k)`.

    Args:
        k: Spatial frequency in inverse angstroms.
        defocus: Underfocus in angstroms, positive.
        wavelength: Electron wavelength in angstroms.
        spherical_aberration_in_mm: Cs.
        phase_shift: Extra phase in radians.

    Returns:
        `pi lambda dz k^2 - (pi/2) Cs lambda^3 k^4 + phase_shift`.
    """
    cs = spherical_aberration_in_mm * 1e7  # mm -> angstroms
    frequency = np.asarray(k, dtype=float)
    return np.pi * wavelength * defocus * frequency**2 - 0.5 * np.pi * cs * wavelength**3 * frequency**4 + phase_shift


def ctf(
    k: np.ndarray,
    *,
    defocus: float,
    voltage_in_kv: float = 300.0,
    spherical_aberration_in_mm: float = 2.7,
    amplitude_contrast_ratio: float = 0.07,
    phase_shift: float = 0.0,
) -> np.ndarray:
    """The isotropic CTF, unsquared, from its definition.

    Args:
        k: Spatial frequency in inverse angstroms.
        defocus: Underfocus in angstroms, positive.
        voltage_in_kv: Accelerating voltage. The wavelength comes from `electron_wavelength`, which
            is derived rather than fitted.
        spherical_aberration_in_mm: Cs.
        amplitude_contrast_ratio: Q.
        phase_shift: Extra phase in radians.

    Returns:
        `-[sqrt(1 - Q^2) sin(chi) + Q cos(chi)]`, negative at low `k`.
    """
    wavelength = electron_wavelength(voltage_in_kv)
    chi = ctf_chi(
        k,
        defocus=defocus,
        wavelength=wavelength,
        spherical_aberration_in_mm=spherical_aberration_in_mm,
        phase_shift=phase_shift,
    )
    q = amplitude_contrast_ratio
    return -(math.sqrt(1.0 - q * q) * np.sin(chi) + q * np.cos(chi))


def ctf_zeros(
    n_zeros: int,
    *,
    defocus: float,
    voltage_in_kv: float = 300.0,
    spherical_aberration_in_mm: float = 2.7,
    amplitude_contrast_ratio: float = 0.0,
) -> np.ndarray:
    """The first spatial frequencies at which the CTF vanishes, in closed form.

    Solves the quadratic in `u = k^2` derived in the module docstring, taking the physical (minus)
    branch. With `Q != 0` the amplitude-contrast term is folded in as the equivalent phase offset
    `arcsin(Q)`, so the condition becomes `chi = n pi - arcsin(Q)`.

    Args:
        n_zeros: How many zeros to return, counting outward from DC.
        defocus: Underfocus in angstroms, positive.
        voltage_in_kv: Accelerating voltage.
        spherical_aberration_in_mm: Cs.
        amplitude_contrast_ratio: Q.

    Returns:
        `(m,)` frequencies in inverse angstroms, ascending, where `m <= n_zeros` -- the aberration
        cuts the sequence off once `dz^2 < 2 n Cs lambda` leaves no real root.
    """
    wavelength = electron_wavelength(voltage_in_kv)
    cs = spherical_aberration_in_mm * 1e7
    offset = math.asin(amplitude_contrast_ratio) / np.pi
    zeros = []
    for order in range(1, n_zeros + 1):
        # chi = n pi - arcsin(Q)  =>  (pi/2) Cs lam^3 k^4 - pi lam dz k^2 + (n - offset) pi = 0
        level = order - offset
        discriminant = defocus**2 - 2.0 * level * cs * wavelength
        if discriminant < 0.0:
            break
        u = (defocus - math.sqrt(discriminant)) / (cs * wavelength**2)
        if u <= 0.0:
            break
        zeros.append(math.sqrt(u))
    return np.asarray(zeros)
