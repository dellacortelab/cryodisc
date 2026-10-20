"""Checks on the translation latent `run_chain` wires through.

A pure helper, tested directly rather than through a full ladder.
"""

import numpy as np
import pytest

from cryodisc.grids.poses import shift_grid
from cryodisc.pipeline.chain import _shift_lattice


def test_a_zero_shift_sigma_takes_the_centered_particle_path():
    """`None` is what every consumer treats as the fast case, so an empty lattice is not the same
    thing as no lattice."""
    assert _shift_lattice(64, 0.0) is None


def test_the_window_covers_the_draw_it_was_built_for():
    """2.5 sigma each way per axis. A draw outside the lattice is truncated rather than
    marginalized. This is the only window the chain builds, so it is the one every pool rung is
    marginalized over."""
    grid = _shift_lattice(64, 2.0)
    assert grid is not None
    assert grid.box_size == 64
    reach = int(np.ceil(2.5 * 2.0))
    assert grid.shifts.shape == ((2 * reach + 1) ** 2, 2)
    lattice = np.asarray(grid.shifts)
    assert lattice.min() == -reach and lattice.max() == reach


def test_every_candidate_lands_on_the_pixel_lattice():
    """`rungs.marginal` scores a whole lattice from one transform and reaches the integer offsets
    only, so a fractional step here would be refused downstream rather than approximated."""
    grid = _shift_lattice(64, 1.6)
    lattice = np.asarray(grid.shifts)
    np.testing.assert_array_equal(lattice, np.round(lattice))
    assert lattice.shape == (81, 2)


def test_the_prior_is_the_gaussian_the_draw_came_from_and_is_normalized():
    """The lattice is a quadrature rule, so its weights have to sum to one; and the prior has to
    fall with radius, or the marginal would weight a far candidate like a near one."""
    grid = _shift_lattice(32, 1.5)
    weights = np.exp(np.asarray(grid.log_weights))
    assert weights.sum() == pytest.approx(1.0, rel=1e-6)
    radius = np.linalg.norm(np.asarray(grid.shifts), axis=-1)
    center = int(np.argmin(radius))
    assert weights[center] == weights.max()
    far = int(np.argmax(radius))
    assert weights[far] < weights[center]


def test_a_negative_window_is_refused():
    """Asserted against `shift_grid`, which owns the guard, so it is tested where it lives."""
    with pytest.raises(ValueError, match="max_shift >= 0"):
        shift_grid(32, -1.0, sigma=1.0, step=1.0)
