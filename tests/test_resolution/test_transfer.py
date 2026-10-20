"""Checks on the effective transfer and its envelopes.

Each factor is pinned by the limit it is supposed to have, and the assembly by the fact that it is
their product and so cannot exceed any of them.
"""

import jax.numpy as jnp
import pytest

from cryodisc.microscope import CollectionPlan, ctf
from cryodisc.resolution.transfer import (
    defocus_averaged_ctf2,
    dqe,
    effective_transfer_squared,
    flexibility_envelope,
    radiation_envelope,
)

K = jnp.linspace(0.01, 0.5, 200)
PLAN = CollectionPlan(dose=40.0)


def test_a_single_defocus_has_zeros_that_a_defocus_range_fills():
    """At the frequency where one defocus transfers nothing, the pool still transfers half."""
    single = ctf(K, defocus_in_angstroms=12000.0, plan=PLAN) ** 2
    averaged = defocus_averaged_ctf2(K, plan=PLAN)
    at_zero = int(jnp.argmin(single))
    assert float(single[at_zero]) < 1e-3
    assert float(averaged[at_zero]) > 0.3
    assert float(averaged.max()) <= 1.0


def test_the_defocus_average_tends_to_one_half_where_the_oscillation_is_fast():
    """`<sin^2> -> 1/2` once a defocus range spans many cycles, which is the high-k limit."""
    assert float(defocus_averaged_ctf2(jnp.array(0.4), plan=PLAN)) == pytest.approx(0.5, rel=0.05)


def test_the_closed_form_average_matches_a_dense_numerical_one():
    """The exactness `defocus_averaged_ctf2` claims, against the quadrature it replaced.

    Agreement here is an identity rather than a convergence, the closed form being an analytic
    integral, and what it catches is a transcription error in `chi = a dz + b`. An `a` and a `b` that
    drifted apart would still produce a smooth curve of the right shape and range.
    """
    low, high = PLAN.defocus_range
    nodes = jnp.linspace(low, high, 4001)
    numerical = jnp.mean(ctf(K[:, None], defocus_in_angstroms=nodes[None, :], plan=PLAN) ** 2, axis=1)
    assert jnp.allclose(defocus_averaged_ctf2(K, plan=PLAN), numerical, atol=2e-3)


def test_a_zero_width_range_reduces_to_the_single_defocus_ctf():
    """`sinc(0) = 1`, so no separate path is needed -- and that path is what a single-defocus plan
    takes, whose transfer would then be wrong everywhere rather than in a limit nobody reaches.

    The tolerance is float32's and not the identity's: the large `chi` at Nyquist keeps only about
    five decimal digits, so the two routes to the same phase round differently.
    """
    plan = CollectionPlan(dose=40.0, defocus_range=(15000.0, 15000.0))
    single = ctf(K, defocus_in_angstroms=15000.0, plan=plan) ** 2
    assert jnp.allclose(defocus_averaged_ctf2(K, plan=plan), single, atol=1e-5)


def test_the_dqe_interpolates_the_measured_curve_and_holds_past_it():
    assert float(dqe(jnp.array(0.0), nyquist=0.5)) == pytest.approx(0.78, rel=1e-6)
    assert float(dqe(jnp.array(0.5), nyquist=0.5)) == pytest.approx(0.24, rel=1e-6)
    assert float(dqe(jnp.array(0.25), nyquist=0.5)) == pytest.approx(0.57, rel=1e-6)
    # the detector's Nyquist is not the scoring grid's, so a finer grid does not extrapolate
    assert float(dqe(jnp.array(0.9), nyquist=0.5)) == float(dqe(jnp.array(0.5), nyquist=0.5))


def test_the_radiation_envelope_is_one_at_zero_frequency_and_falls_with_dose():
    """The DC pixel is on every fft grid here, and `1 - exp(-dose/2Ne)` cancels to zero there in float32."""
    assert float(radiation_envelope(jnp.zeros(()), dose=40.0)) == pytest.approx(1.0, rel=1e-4)
    envelope = radiation_envelope(K, dose=40.0)
    assert float(envelope.max()) <= 1.0
    assert bool(jnp.all(jnp.diff(envelope) < 0.0))
    assert float(radiation_envelope(jnp.array(0.4), dose=80.0)) < float(radiation_envelope(jnp.array(0.4), dose=20.0))


def test_the_flexibility_envelope_is_the_debye_waller_factor():
    assert float(flexibility_envelope(K, rmsf=0.0).min()) == pytest.approx(1.0, rel=1e-6)
    b_factor = 8.0 * jnp.pi**2 * 0.5**2
    expected = jnp.exp(-b_factor * 0.3**2 / 4.0)
    assert float(flexibility_envelope(jnp.array(0.3), rmsf=0.5)) == pytest.approx(float(expected), rel=1e-6)


def test_the_assembly_is_the_product_and_so_bounded_by_every_factor():
    total = effective_transfer_squared(K, plan=PLAN, rmsf=0.5)
    assert bool(jnp.all(total <= defocus_averaged_ctf2(K, plan=PLAN) + 1e-6))
    assert bool(jnp.all(total <= dqe(K, nyquist=PLAN.nyquist) + 1e-6))
    assert bool(jnp.all(total <= 1.0))
    # each loss applied exactly once, so the product is the whole of it
    factored = (
        defocus_averaged_ctf2(K, plan=PLAN)
        * dqe(K, nyquist=PLAN.nyquist)
        * radiation_envelope(K, dose=PLAN.dose) ** 2
        * flexibility_envelope(K, rmsf=0.5) ** 2
    )
    assert float(jnp.max(jnp.abs(total - factored))) < 1e-6


def test_a_more_flexible_state_transfers_less_at_every_frequency():
    rigid = effective_transfer_squared(K, plan=PLAN, rmsf=0.3)
    floppy = effective_transfer_squared(K, plan=PLAN, rmsf=1.0)
    assert bool(jnp.all(floppy <= rigid))
    assert float(floppy[-1] / rigid[-1]) < 0.1


def test_the_plan_is_the_single_source_of_the_nyquist_and_the_optics():
    """The whole point of threading it: a finer pixel cannot disagree with the DQE's Nyquist."""
    fine = CollectionPlan(dose=40.0, pixel_size=0.4)
    assert fine.nyquist == pytest.approx(1.25)
    assert float(dqe(jnp.array(0.6), nyquist=fine.nyquist)) > float(dqe(jnp.array(0.6), nyquist=PLAN.nyquist))
    # a 200 kV plan puts the CTF zeros somewhere else, and every factor sees the same voltage
    low = CollectionPlan(dose=40.0, voltage_in_kv=200.0)
    assert not bool(jnp.allclose(defocus_averaged_ctf2(K, plan=low), defocus_averaged_ctf2(K, plan=PLAN)))
