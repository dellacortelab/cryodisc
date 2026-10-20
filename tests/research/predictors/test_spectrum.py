"""Checks on the shell-resolved state difference.

The spectrum exists to be re-weighted, so what matters is that a weighted sum over its shells is the
`d^2` the scorer would have computed. Every check here is against `local_information`, which is the
number the survey's own rows carry -- a spectrum that summed to something else would give a plausible
half-power spacing and a wrong one.
"""

import numpy as np
import pytest
from conftest import gaussian_blob_models
from jax import numpy as jnp

from cryodisc.forward.optics import CTFParams
from cryodisc.grids.fourier import frequency_grid, shell_indices, whitening_weight
from cryodisc.grids.poses import pose_bank
from cryodisc.microscope import CollectionPlan
from cryodisc.resolution.noise import noise_spectrum
from cryodisc.rungs.local import local_information
from reproduction.predictors.spectrum import (
    d2_from_spectrum,
    difference_spectrum,
    half_power_spacing,
    spacing_of_shells,
)

BOX_SIZE = 32
PIXEL_SIZE = 2.0


@pytest.fixture(scope="module")
def setup():
    """Two states differing by one displaced blob, and a collection to score them under."""
    centers, build = gaussian_blob_models(BOX_SIZE, 5, seed=0, spread=5.0, sigma_range=(1.5, 2.5))
    moved = centers.copy()
    moved[0] += np.array([2.0, 0.0, 0.0])
    plan = CollectionPlan(dose=40.0, pixel_size=PIXEL_SIZE)
    spectrum = noise_spectrum(plan, BOX_SIZE, rmsf=0.5)
    return build(centers), build(moved), plan, spectrum, pose_bank(45)


def _shell_weight(spectrum, noise, box_size, n_shells):
    """`1 / sigma^2` averaged within each shell, which is what the shell sum can see."""
    index, _ = shell_indices(box_size, n_shells)
    weight = np.asarray(whitening_weight(noise))
    flat, bins = np.asarray(index).ravel(), np.asarray(weight).ravel()
    total = np.bincount(flat, weights=bins, minlength=n_shells + 1)[:n_shells]
    counts = np.bincount(flat, minlength=n_shells + 1)[:n_shells]
    return np.where(counts > 0, total / np.maximum(counts, 1), 0.0)


def test_the_shell_sum_reproduces_the_rung_it_is_a_decomposition_of(setup):
    """The whole contract: re-weight the spectrum and get `local_information`'s `known_pose` back.

    The residual is the shell approximation rather than a disagreement -- `sigma^2` varies across the
    modes of one shell and the sum can only see their mean.
    """
    model_a, model_b, plan, noise, grid = setup
    optics = CTFParams.from_plan(plan, plan.mid_defocus)
    rung = float(jnp.mean(local_information(model_a, model_b, grid.angles, noise, ctf_params=optics).known_pose))

    n_shells = BOX_SIZE // 2
    spectrum = difference_spectrum(
        model_a, model_b, grid.angles, box_size=BOX_SIZE, n_shells=n_shells, ctf_params=optics
    )
    got = d2_from_spectrum(spectrum, _shell_weight(spectrum, noise, BOX_SIZE, n_shells))
    assert got == pytest.approx(rung, rel=0.01), f"shell sum {got:.6g} against the rung {rung:.6g}"


def test_a_change_in_the_noise_side_needs_no_new_render(setup):
    """The re-use the census rests on: one render, then any dose or B factor by re-weighting.

    Dose, DQE, the radiation envelope and the Debye-Waller term are all inside `sigma^2(k)`, so a
    spectrum rendered once under fixed optics answers for all of them. The CTF is not, and
    `spectrum.py`'s docstring says why; sweeping defocus costs a pass.
    """
    model_a, model_b, plan, _noise, grid = setup
    optics = CTFParams.from_plan(plan, plan.mid_defocus)
    n_shells = BOX_SIZE // 2
    spectrum = difference_spectrum(
        model_a, model_b, grid.angles, box_size=BOX_SIZE, n_shells=n_shells, ctf_params=optics
    )

    for dose, rmsf in ((40.0, 0.5), (80.0, 0.5), (10.0, 1.5)):
        other = CollectionPlan(dose=dose, pixel_size=PIXEL_SIZE)
        noise = noise_spectrum(other, BOX_SIZE, rmsf=rmsf)
        # the rung this plan would produce, computed the long way, against the re-weighted shells
        rung = float(jnp.mean(local_information(model_a, model_b, grid.angles, noise, ctf_params=optics).known_pose))
        got = d2_from_spectrum(spectrum, _shell_weight(spectrum, noise, BOX_SIZE, n_shells))
        assert got == pytest.approx(rung, rel=0.01), f"dose {dose}, rmsf {rmsf}: {got:.6g} against {rung:.6g}"


def test_the_half_power_spacing_is_coarser_than_the_band(setup):
    """A displaced blob puts its difference at low frequency, which is the whole premise of Fig 7."""
    model_a, model_b, _plan, noise, grid = setup
    n_shells = BOX_SIZE // 2
    spectrum = difference_spectrum(model_a, model_b, grid.angles, box_size=BOX_SIZE, n_shells=n_shells)
    weight = _shell_weight(spectrum, noise, BOX_SIZE, n_shells)
    spacing = half_power_spacing(spectrum, weight) * PIXEL_SIZE
    assert 2 * PIXEL_SIZE < spacing, f"half power at {spacing:.2f} A, which is finer than Nyquist"
    assert spacing < BOX_SIZE * PIXEL_SIZE, f"half power at {spacing:.2f} A, which is off the box"


def test_spacings_and_frequencies_are_inverse():
    _, centers = shell_indices(BOX_SIZE, BOX_SIZE // 2)
    spacing = spacing_of_shells(centers, PIXEL_SIZE)
    assert np.allclose(spacing * np.asarray(centers) / PIXEL_SIZE, 1.0)


def test_misshapen_angles_are_refused(setup):
    model_a, model_b, _, _, _ = setup
    with pytest.raises(ValueError, match="Euler triples"):
        difference_spectrum(model_a, model_b, jnp.zeros((4, 2)), box_size=BOX_SIZE)


def test_the_shell_grid_covers_the_band(setup):
    """Every in-band mode lands in a shell, or the sum silently drops signal."""
    n_shells = BOX_SIZE // 2
    index, _ = shell_indices(BOX_SIZE, n_shells)
    inside = np.asarray(frequency_grid(BOX_SIZE)) < 0.5
    assert np.all(np.asarray(index)[inside] < n_shells)
