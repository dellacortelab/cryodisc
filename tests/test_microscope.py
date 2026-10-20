"""Checks on the instrument description every engine reads.

Both functions are pinned against values from outside the package -- a tabulated wavelength and the
`k = 0` limit the CTF has by construction -- since a plan that is merely self-consistent would let
every engine agree on the same wrong microscope.
"""

import jax.numpy as jnp
import pytest

from cryodisc.microscope import CollectionPlan, ctf, wavelength

K = jnp.linspace(0.01, 0.5, 200)
PLAN = CollectionPlan(dose=40.0)


@pytest.mark.parametrize(("voltage", "expected"), [(300.0, 0.01969), (200.0, 0.02508)])
def test_the_wavelength_matches_the_tabulated_value(voltage, expected):
    assert float(wavelength(voltage)) == pytest.approx(expected, rel=1e-3)


def test_the_ctf_starts_at_minus_the_amplitude_contrast():
    """chi vanishes at k = 0, so only the cosine term survives, and its sign is the dark-on-light one."""
    assert float(ctf(jnp.zeros(()), defocus_in_angstroms=12000.0, plan=PLAN)) == pytest.approx(-0.07, rel=1e-6)
    assert float(jnp.max(jnp.abs(ctf(K, defocus_in_angstroms=12000.0, plan=PLAN)))) <= 1.0


def test_the_nyquist_follows_the_pixel_and_nothing_else():
    """The one place a detector's band limit is derived, so a finer pixel cannot be paired with a
    coarser Nyquist somewhere downstream."""
    assert CollectionPlan(dose=40.0, pixel_size=0.4).nyquist == pytest.approx(1.25)
    assert PLAN.nyquist == pytest.approx(0.5)


def test_the_mid_defocus_is_the_middle_of_the_range_and_is_shared_by_both_model_only_arms():
    """The midpoint is used by both `chain` and `targets`, so this pins its value AND that both
    sites read it from the plan rather than re-deriving it."""
    import inspect

    from cryodisc.pipeline import chain
    from reproduction import targets

    assert CollectionPlan(dose=40.0, defocus_range=(5000.0, 25000.0)).mid_defocus == pytest.approx(15000.0)
    assert CollectionPlan(dose=40.0, defocus_range=(8000.0, 20000.0)).mid_defocus == pytest.approx(14000.0)
    # a single repeated defocus is its own midpoint, which is what a fixed-defocus plan must get
    assert CollectionPlan(dose=40.0, defocus_range=(12000.0, 12000.0)).mid_defocus == pytest.approx(12000.0)

    for module in (chain, targets):
        source = inspect.getsource(module)
        assert "mid_defocus" in source, f"{module.__name__} must take the midpoint from the plan"
        assert "sum(plan.defocus_range)" not in source, f"{module.__name__} re-derives the midpoint"


def test_the_astigmatic_ctf_collapses_onto_the_radial_one():
    """Two implementations of the same physics, one fitted against and one scored against. A drift
    between them moves every high-order zero, so equality at zero astigmatism is pinned."""
    import numpy as np

    from cryodisc.grids.fourier import frequency_grid, frequency_vectors
    from cryodisc.microscope import ctf_astigmatic

    plan = CollectionPlan(dose=40.0, pixel_size=1.2, voltage_in_kv=300.0)
    box_size = 32
    k_y, k_x = frequency_vectors(box_size, plan.pixel_size)
    radial = ctf(frequency_grid(box_size, plan.pixel_size), defocus_in_angstroms=11000.0, plan=plan)
    astigmatic = ctf_astigmatic(
        k_y,
        k_x,
        defocus_u=11000.0,
        defocus_v=11000.0,
        voltage_in_kv=plan.voltage_in_kv,
        spherical_aberration_in_mm=plan.spherical_aberration_in_mm,
        amplitude_contrast_ratio=plan.amplitude_contrast_ratio,
    )
    # the gap is one float32 epsilon in k^2, not a tuned tolerance: `ctf` squares |k| while this
    # builds k^2 from the components
    np.testing.assert_allclose(np.asarray(astigmatic), np.asarray(radial), atol=1e-4)


def test_astigmatism_puts_the_defocus_on_the_azimuth_it_was_given():
    """The defocus a mode sees runs from `defocus_u` along the astigmatism axis to `defocus_v` across
    it, so an azimuth convention that was transposed or negated would swap the two."""
    import numpy as np

    from cryodisc.grids.fourier import frequency_vectors
    from cryodisc.microscope import ctf_astigmatic

    box_size = 32
    k_y, k_x = frequency_vectors(box_size, 1.0)
    # along +x the azimuth is 0, so at astigmatism_angle 0 that direction sees defocus_u
    along_x = ctf_astigmatic(k_y, k_x, defocus_u=14000.0, defocus_v=6000.0, astigmatism_angle=0.0)
    isotropic_u = ctf_astigmatic(k_y, k_x, defocus_u=14000.0, defocus_v=14000.0)
    isotropic_v = ctf_astigmatic(k_y, k_x, defocus_u=6000.0, defocus_v=6000.0)

    np.testing.assert_allclose(np.asarray(along_x)[0, :], np.asarray(isotropic_u)[0, :], atol=1e-6)
    np.testing.assert_allclose(np.asarray(along_x)[:, 0], np.asarray(isotropic_v)[:, 0], atol=1e-6)


def test_the_ctf_envelope_damps_and_leaves_dc_alone():
    """`b_factor` is `exp(-B k^2 / 4)`, which is one at DC whatever B is."""

    from cryodisc.grids.fourier import frequency_vectors
    from cryodisc.microscope import ctf_astigmatic

    k_y, k_x = frequency_vectors(16, 1.0)
    bare = ctf_astigmatic(k_y, k_x, defocus_u=10000.0, defocus_v=10000.0)
    damped = ctf_astigmatic(k_y, k_x, defocus_u=10000.0, defocus_v=10000.0, b_factor=200.0)
    assert float(damped[0, 0]) == pytest.approx(float(bare[0, 0]))
    assert abs(float(damped[8, 0])) < abs(float(bare[8, 0]))
