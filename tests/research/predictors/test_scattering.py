"""What the absolute predictor has to get right before `K` transferring between targets means anything.

The scattering factors are pinned against Peng's published `f_e(0)`, since everything downstream is
proportional to them. The rest are the monotonicities a resolution model has to have, plus the two
places it is allowed to say no: below Nyquist, and infinite when the SSNR never reaches threshold.
"""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from cryodisc.microscope import CollectionPlan
from reproduction.predictors.scattering import (
    calibrate_scale,
    predict_resolution,
    scattering_factor,
    signal_power,
    ssnr,
)

PLAN = CollectionPlan(dose=40.0, pixel_size=0.5)


ATOMS = pd.DataFrame({"atomic_number": np.array([6] * 3000 + [7] * 800 + [8] * 900 + [16] * 25)})
"""An atom table the size of a small protein, carrying only what `signal_power` reads."""


@pytest.fixture(scope="module")
def scale():
    """`K` pinned to 2.5 A at 50,000 particles, which most of these tests then hold fixed."""
    return calibrate_scale(2.5, atoms=ATOMS, plan=PLAN, n_particles=50_000, rmsf=0.3)


def test_scattering_factors_are_pengs():
    """f_e(0) in A, from Peng (1996) Table 1. Everything downstream is proportional to these."""
    at_zero = scattering_factor(np.array([1, 6, 7, 8, 16, 26]), np.array(0.0))
    np.testing.assert_allclose(at_zero, [0.5288, 2.5088, 2.2135, 1.9834, 5.1604, 7.1637], atol=1e-4)


def test_the_scattering_factor_falls_with_frequency():
    k = np.linspace(0.0, 1.0, 50)
    carbon = np.asarray(scattering_factor(np.array([6]), k))[:, 0]
    assert np.all(np.diff(carbon) < 0.0)
    assert carbon[-1] < 0.2 * carbon[0]


def test_signal_power_counts_every_row_so_symmetry_needs_no_separate_factor():
    """`conformers` returns the built assembly, so a symmetric particle arrives with its copies."""
    k = np.linspace(0.05, 0.5, 10)
    one = signal_power(ATOMS, k)
    both = signal_power(pd.concat([ATOMS, ATOMS], ignore_index=True), k)
    np.testing.assert_allclose(both, 2.0 * one, rtol=1e-6)


def test_calibration_reproduces_the_resolution_it_was_pinned_to():
    """The whole point of `K`: one achieved number in, and the crossing lands back on it."""
    for resolution in (1.5, 2.5, 4.0):
        scale = calibrate_scale(resolution, atoms=ATOMS, plan=PLAN, n_particles=50_000, rmsf=0.3)
        predicted = predict_resolution(atoms=ATOMS, plan=PLAN, n_particles=50_000, rmsf=0.3, scale=scale)
        assert predicted == pytest.approx(resolution, rel=2e-3)


def test_a_calibration_that_does_not_round_trip_is_refused_rather_than_returned():
    """`calibrate_scale` pins ONE frequency; `predict_resolution` returns the HIGHEST crossing. They
    are inverses only where the SSNR curve is monotone past the stated resolution, and a
    single-defocus plan is not: its CTF zeros put crossings on both sides, so calibrating at 4 A
    reads back a much finer number. Resolution enters the particle count quadratically, so a `K`
    that silently means a different resolution is a silently wrong N."""
    single = CollectionPlan(dose=40.0, pixel_size=0.5, defocus_range=(10_000.0, 10_000.0))

    with pytest.raises(ValueError, match="defocus RANGE"):
        calibrate_scale(4.0, atoms=ATOMS, plan=single, n_particles=50_000, rmsf=0.3)

    # the shipped range-plan still calibrates, so the guard fires on the plan and not on every call
    assert calibrate_scale(4.0, atoms=ATOMS, plan=PLAN, n_particles=50_000, rmsf=0.3) > 0.0


def test_more_particles_never_hurt_and_the_returns_diminish(scale):
    """Rosenthal-Henderson: resolution improves with N, and improves more slowly the more you have."""
    counts = np.array([1e3, 1e4, 1e5, 1e6, 1e7])
    resolutions = np.array(
        [predict_resolution(atoms=ATOMS, plan=PLAN, n_particles=n, rmsf=0.3, scale=scale) for n in counts]
    )
    assert np.all(np.diff(resolutions) < 0.0)
    gains = -np.diff(resolutions)  # each step is a factor of ten in N
    assert np.all(np.diff(gains) < 0.0)


def test_flexibility_damps_the_resolution(scale):
    """The repo's subject: a state that moves within itself is damped at every particle count."""
    resolutions = [
        predict_resolution(atoms=ATOMS, plan=PLAN, n_particles=50_000, rmsf=u, scale=scale)
        for u in (0.3, 0.6, 1.0, 1.5)
    ]
    assert all(a < b for a, b in pairwise(resolutions))


def test_too_few_particles_returns_infinity_rather_than_a_large_number(scale):
    """Nothing is reconstructed, which is a different statement from a coarse reconstruction."""
    assert predict_resolution(atoms=ATOMS, plan=PLAN, n_particles=1, rmsf=0.3, scale=scale) == float("inf")


def test_a_bigger_particle_reaches_further_at_the_same_particle_count(scale):
    bigger = pd.concat([ATOMS] * 8, ignore_index=True)
    assert predict_resolution(atoms=bigger, plan=PLAN, n_particles=50_000, rmsf=0.3, scale=scale) < 2.5


def test_preferred_orientation_costs_resolution(scale):
    biased = predict_resolution(
        atoms=ATOMS, plan=PLAN, n_particles=50_000, rmsf=0.3, scale=scale, orientation_factor=0.1
    )
    assert biased > 2.5


def test_nothing_is_predicted_past_the_detectors_nyquist():
    """A coarse pixel is a hard ceiling, not a term that trades against particle count."""
    coarse = CollectionPlan(dose=40.0, pixel_size=1.6)
    scale = calibrate_scale(4.0, atoms=ATOMS, plan=coarse, n_particles=50_000, rmsf=0.3)
    assert predict_resolution(atoms=ATOMS, plan=coarse, n_particles=10**9, rmsf=0.3, scale=scale) >= 3.2


def test_the_ssnr_is_linear_in_the_calibrated_scale():
    """Which is why `calibrate_scale` is a division and not a root solve."""
    k = np.linspace(0.05, 0.9, 20)
    one = ssnr(k, atoms=ATOMS, plan=PLAN, n_particles=50_000, rmsf=0.3, scale=1e-9)
    ten = ssnr(k, atoms=ATOMS, plan=PLAN, n_particles=50_000, rmsf=0.3, scale=1e-8)
    np.testing.assert_allclose(np.asarray(ten), 10.0 * np.asarray(one), rtol=1e-5)


def test_restoring_hydrogens_coherently_is_not_the_same_as_counting_them_incoherently():
    """Defect 3: a C-H bond is 1.09 A, far below the ~8 A end of this band, so a hydrogen and its
    parent are ONE scatterer here and their amplitudes add. The two sums differ by about 10x in
    what they attribute to hydrogen, which a fitted amplitude would hide and an absolute model
    cannot. Both are computed here from `scattering_factor` directly, not from `signal_power`."""
    from reproduction.predictors.scattering import scattering_factor, signal_power

    k = 1.0 / 12.0
    carbon = pd.DataFrame({"atomic_number": np.full(100, 6)})

    f_c = float(scattering_factor(np.array([6]), k)[0])
    f_h = float(scattering_factor(np.array([1]), k)[0])

    bare = float(signal_power(carbon, k))
    assert bare == pytest.approx(100.0 * f_c**2, rel=1e-6)

    coherent = float(signal_power(carbon, k, hydrogens_per_heavy_atom=1.0))
    assert coherent == pytest.approx(100.0 * (f_c + f_h) ** 2, rel=1e-6)

    incoherent = 100.0 * (f_c**2 + f_h**2)
    # the coherent restoration gains far more than the incoherent one, which is the whole point
    assert (coherent - bare) > 5.0 * (incoherent - bare)
    # and asking for none is exactly the bare sum
    assert float(signal_power(carbon, k, hydrogens_per_heavy_atom=0.0)) == pytest.approx(bare, rel=1e-6)


def test_hydrogens_already_in_the_table_are_not_counted_twice():
    """A table that has them would otherwise carry each hydrogen once in its own row and once on its
    parent, which is a silent overcount rather than an error."""
    from reproduction.predictors.scattering import signal_power

    k = 1.0 / 12.0
    heavy_only = pd.DataFrame({"atomic_number": np.full(50, 6)})
    with_hydrogens = pd.DataFrame({"atomic_number": np.concatenate([np.full(50, 6), np.full(50, 1)])})
    restored = signal_power(heavy_only, k, hydrogens_per_heavy_atom=1.0)
    assert float(signal_power(with_hydrogens, k, hydrogens_per_heavy_atom=1.0)) == pytest.approx(
        float(restored), rel=1e-6
    )


def test_a_negative_hydrogen_count_is_refused():
    from reproduction.predictors.scattering import signal_power

    with pytest.raises(ValueError, match="non-negative"):
        signal_power(pd.DataFrame({"atomic_number": np.full(4, 6)}), 0.1, hydrogens_per_heavy_atom=-1.0)
