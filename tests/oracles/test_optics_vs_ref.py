"""`microscope` against optics derived from fundamental constants and closed forms.

Production's wavelength is a fit; the reference derives it from `h`, `m_e`, `e` and `c`, so a mistyped
fit coefficient no internal check could see fails here. The CTF zeros are the other independent
handle: closed-form roots of a quadratic in `k^2`, so their positions are predicted without
evaluating the CTF at all.
"""

import numpy as np
import pytest

from cryodisc.grids.fourier import frequency_grid, frequency_vectors
from cryodisc.microscope import CollectionPlan, ctf, ctf_astigmatic, wavelength

from . import ref_optics as ref

WAVELENGTH_FIT_ERROR = 3e-5
"""How far production's `12.2639 / sqrt(V (1 + 0.978476e-6 V))` fit sits from the exact derivation.

The fit's own accuracy rather than a defect. A relative wavelength error `eps` moves the CTF phase by
`chi eps`, so every CTF tolerance below is `chi_max * WAVELENGTH_FIT_ERROR` rather than a round
number, which keeps a real disagreement distinguishable from this one.
"""


def phase_tolerance(defocus, k_max=0.5, voltage_in_kv=300.0):
    """The largest CTF difference the wavelength fit alone can produce over a band.

    `|d CTF| <= |d chi| = chi_max * (d lambda / lambda)`, since `|d CTF / d chi| <= 1`.
    """
    wavelength = ref.electron_wavelength(voltage_in_kv)
    chi_max = np.pi * wavelength * defocus * k_max**2
    return chi_max * WAVELENGTH_FIT_ERROR


def zero_crossings(values, k):
    """Frequencies where a sampled function changes sign, by linear interpolation."""
    sign_change = np.flatnonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0)
    left, right = values[sign_change], values[sign_change + 1]
    return k[sign_change] + (k[sign_change + 1] - k[sign_change]) * left / (left - right)


@pytest.mark.parametrize("voltage", [80.0, 120.0, 200.0, 300.0])
def test_the_wavelength_matches_a_derivation_from_fundamental_constants(voltage):
    """Production's fit against `h / sqrt(2 m e V (1 + eV / 2mc^2))`. The two share only the
    physics, so this is the check the fit's coefficients cannot get from anywhere else."""
    assert float(wavelength(voltage)) == pytest.approx(ref.electron_wavelength(voltage), rel=WAVELENGTH_FIT_ERROR)


@pytest.mark.parametrize("voltage", [120.0, 200.0, 300.0])
def test_both_agree_with_the_published_value(voltage):
    """A third, external anchor, in case the fit and the derivation were somehow wrong together."""
    published = ref.reference_wavelength_angstrom[voltage]
    assert float(wavelength(voltage)) == pytest.approx(published, rel=3e-3)
    assert ref.electron_wavelength(voltage) == pytest.approx(published, rel=3e-3)


def test_the_relativistic_correction_is_present_and_large():
    """Without it the 300 kV wavelength would be 0.0223 A rather than 0.0197, a 13% error that moves
    every CTF zero. Checked as the RATIO to the classical form, so it cannot be passed by accident."""
    volts = 300e3
    classical = ref.PLANCK / np.sqrt(2.0 * 9.1093837015e-31 * 1.602176634e-19 * volts) * 1e10
    assert float(wavelength(300.0)) / classical == pytest.approx(0.883, rel=0.01)


@pytest.mark.parametrize("defocus", [5000.0, 12000.0, 25000.0])
def test_the_ctf_matches_the_reference_pointwise(defocus):
    """Same definition, independently written, with the reference taking its wavelength from the
    derivation rather than from production."""
    plan = CollectionPlan(dose=40.0, pixel_size=1.0, voltage_in_kv=300.0)
    k = np.linspace(0.0, 0.5, 400)
    produced = np.asarray(ctf(k, defocus_in_angstroms=defocus, plan=plan))
    expected = ref.ctf(
        k,
        defocus=defocus,
        voltage_in_kv=plan.voltage_in_kv,
        spherical_aberration_in_mm=plan.spherical_aberration_in_mm,
        amplitude_contrast_ratio=plan.amplitude_contrast_ratio,
    )
    # the two differ only through the wavelength, so the bound is derived rather than tuned; it is
    # also asserted to be SMALL, or a large tolerance would pass anything
    tolerance = phase_tolerance(defocus)
    assert tolerance < 0.02
    np.testing.assert_allclose(produced, expected, atol=tolerance)


def test_the_ctf_is_negative_at_low_frequency():
    """The sign convention, which decides whether a particle is dark on light or light on dark. A
    flipped one anti-correlates every template with its own data and still produces a number."""
    plan = CollectionPlan(dose=40.0, pixel_size=1.0)
    assert float(ctf(np.asarray(0.005), defocus_in_angstroms=10000.0, plan=plan)) < 0.0
    assert ref.ctf(np.asarray(0.005), defocus=10000.0) < 0.0


def test_the_zero_positions_match_a_closed_form():
    """The strongest available check on `chi`: the zeros are the roots of a quadratic in `k^2`, so
    their positions are predicted without evaluating the CTF anywhere. A wrong Cs term, a wrong
    factor of two in the quartic, or a wrong wavelength each moves them."""
    plan = CollectionPlan(dose=40.0, pixel_size=1.0, amplitude_contrast_ratio=0.0)
    defocus = 12000.0
    zeros = ref.ctf_zeros(
        6,
        defocus=defocus,
        voltage_in_kv=plan.voltage_in_kv,
        spherical_aberration_in_mm=plan.spherical_aberration_in_mm,
        amplitude_contrast_ratio=0.0,
    )
    assert zeros.size >= 4
    # compared as POSITIONS, not as CTF values at predicted positions: near a zero the CTF has slope
    # `d chi / dk`, which is huge at these defoci, so a value comparison would measure the slope
    # rather than the agreement. A relative wavelength error eps moves a zero by eps/2 relatively.
    k = np.linspace(1e-4, 0.5, 200_000)
    found = zero_crossings(np.asarray(ctf(k, defocus_in_angstroms=defocus, plan=plan)), k)
    np.testing.assert_allclose(found[: zeros.size], zeros, rtol=WAVELENGTH_FIT_ERROR)
    # and they must be ascending and distinct, which a repeated root would break
    assert np.all(np.diff(zeros) > 0.0)


def test_the_reference_zeros_reduce_to_the_weak_aberration_spacing():
    """A check on the ORACLE, before production is compared to it. As `Cs -> 0` the quadratic's
    physical branch tends to `k^2 = n / (lambda dz)`, and at a realistic Cs the two must still agree
    to about a percent at low order. An earlier draft carried `Cs lambda^2` in the discriminant
    where dimensions demand `Cs lambda`, which put every zero 7x too low; this is what catches it."""
    defocus, voltage = 12000.0, 300.0
    wavelength_a = ref.electron_wavelength(voltage)
    zeros = ref.ctf_zeros(6, defocus=defocus, voltage_in_kv=voltage, amplitude_contrast_ratio=0.0)
    weak = np.sqrt(np.arange(1, 7) / (wavelength_a * defocus))
    np.testing.assert_allclose(zeros, weak, rtol=0.02)


def test_the_amplitude_contrast_term_is_a_pure_phase_offset():
    """`sqrt(1-Q^2) sin(chi) + Q cos(chi) = sin(chi + arcsin(Q))`. Checked by predicting where the
    zeros move to when Q is turned on, rather than by restating the algebra."""
    plan = CollectionPlan(dose=40.0, pixel_size=1.0, amplitude_contrast_ratio=0.1)
    defocus = 12000.0
    zeros = ref.ctf_zeros(
        4,
        defocus=defocus,
        voltage_in_kv=plan.voltage_in_kv,
        spherical_aberration_in_mm=plan.spherical_aberration_in_mm,
        amplitude_contrast_ratio=0.1,
    )
    k = np.linspace(1e-4, 0.5, 200_000)
    found = zero_crossings(np.asarray(ctf(k, defocus_in_angstroms=defocus, plan=plan)), k)
    np.testing.assert_allclose(found[: zeros.size], zeros, rtol=WAVELENGTH_FIT_ERROR)

    # and the offset MOVES them, so this is not passing by symmetry. Compared RELATIVELY and against
    # the wavelength tolerance: the shift has to be something the test could actually have resolved.
    plain = CollectionPlan(dose=40.0, pixel_size=1.0, amplitude_contrast_ratio=0.0)
    without = zero_crossings(np.asarray(ctf(k, defocus_in_angstroms=defocus, plan=plain)), k)
    relative_move = np.abs(without[: zeros.size] - zeros) / zeros
    assert relative_move.min() > 100.0 * WAVELENGTH_FIT_ERROR


def test_the_astigmatic_form_reproduces_the_reference_along_its_two_axes():
    """Along the astigmatism axis the mode sees `defocus_u`, and across it `defocus_v`. Compared
    against the isotropic REFERENCE at each of those defoci, so an azimuth convention that was
    transposed or negated fails rather than merely looking different."""
    box_size = 32
    pixel_size = 1.0
    k_y, k_x = frequency_vectors(box_size, pixel_size)
    produced = np.asarray(
        ctf_astigmatic(
            k_y,
            k_x,
            defocus_u=15000.0,
            defocus_v=8000.0,
            astigmatism_angle=0.0,
            voltage_in_kv=300.0,
            spherical_aberration_in_mm=2.7,
            amplitude_contrast_ratio=0.07,
        )
    )
    axis = np.asarray(frequency_grid(box_size, pixel_size))
    # row 0 runs along +x, where the azimuth is 0 and the defocus is defocus_u
    np.testing.assert_allclose(produced[0, :], ref.ctf(axis[0, :], defocus=15000.0), atol=phase_tolerance(15000.0))
    # column 0 runs along +y, azimuth pi/2, where cos(2 theta) = -1 and the defocus is defocus_v
    np.testing.assert_allclose(produced[:, 0], ref.ctf(axis[:, 0], defocus=8000.0), atol=phase_tolerance(8000.0))


def test_the_aberration_cuts_the_zero_sequence_off():
    """A real root exists only while `dz^2 >= 2 n Cs lambda^2`, so a low defocus simply has fewer
    zeros. A reference that returned a complex or NaN frequency instead would poison every test
    above rather than reporting a shorter list."""
    few = ref.ctf_zeros(40, defocus=2000.0, spherical_aberration_in_mm=2.7)
    many = ref.ctf_zeros(40, defocus=20000.0, spherical_aberration_in_mm=2.7)
    assert 0 < few.size < many.size
    assert np.all(np.isfinite(few)) and np.all(np.isfinite(many))
