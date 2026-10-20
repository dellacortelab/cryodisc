"""Checks on `pose_bank`, the Hopf-fibration bank.

The construction is a Class I icosahedral geodesic sphere crossed with a uniform `psi`. Covering is
measured in `test_poses.py`; what is here is everything else the construction promises -- the
convention it hands downstream, that its nodes are distinct, that its weights are uniform, and the
two properties a caller has to know about: the node count is quantized, and the banks nest.
"""

import math

import numpy as np
import pytest
from jax import numpy as jnp
from scipy.spatial.transform import Rotation

from cryodisc.grids.poses import (
    _USABLE_TRIANGULATIONS,
    PoseGrid,
    _choose_bank,
    balanced_psi,
    geodesic_directions,
    hopf_grid,
    mean_square_miss,
    median_miss,
    pose_bank,
    rotation_of,
    subdivision_of,
)


def _quaternions(angles):
    """Unit quaternions of a pose bank, in `rotation_of`'s convention.

    Vectorized through scipy's extrinsic "zyz", which `test_the_bank_round_trips` pins against the
    single-pose path.
    """
    quat = Rotation.from_euler("zyz", np.asarray(angles, dtype=np.float64), degrees=True).as_quat()
    return np.stack([quat[:, 3], quat[:, 0], quat[:, 1], quat[:, 2]], axis=-1)


def _covering(angles, n_probe=400, seed=0):
    """Distance from a Haar-drawn rotation to its nearest node, in degrees.

    `|q . q'|` because `q` and `-q` are the same rotation, and the quaternion carries the half-angle.
    """
    probe = Rotation.random(n_probe, random_state=seed).as_quat()
    probe = np.stack([probe[:, 3], probe[:, 0], probe[:, 1], probe[:, 2]], axis=-1)  # xyzw -> wxyz
    overlap = np.abs(probe @ _quaternions(angles).T)
    return 2.0 * np.degrees(np.arccos(np.clip(overlap.max(axis=1), 0.0, 1.0)))


def test_the_bank_round_trips():
    """The Euler triple must rebuild the rotation the construction meant.

    A wrong convention still covers SO(3) and every downstream rung would run, so nothing but this
    catches it. Checked against `geodesic_directions` and a plain `psi` cycle, i.e. the definition.
    The one place that loops `rotation_of`, and the vectorized `_quaternions` is checked against it
    here rather than trusted.
    """
    grid = hopf_grid(1, 8)
    directions = geodesic_directions(1)
    psi = np.arange(8) * 45.0
    expected = Rotation.from_euler(
        "zyz",
        np.stack([np.repeat(directions[:, 0], 8), np.repeat(directions[:, 1], 8), np.tile(psi, 12)], axis=1),
        degrees=True,
    ).as_matrix()
    built = np.stack([np.asarray(rotation_of(a).as_matrix()) for a in np.asarray(grid.angles)])
    assert np.abs(expected - built).max() < 1e-4
    # q and -q are the same rotation, so the helper agrees when every overlap has magnitude one
    reference = np.stack([np.asarray(rotation_of(a).wxyz) for a in np.asarray(grid.angles)])
    assert np.allclose(np.abs((reference * _quaternions(grid.angles)).sum(axis=1)), 1.0, atol=1e-5)


def test_the_bank_is_not_degenerate():
    """Every node distinct, and no rotation left uncovered by an implausible margin.

    A product bank can degenerate by placing a viewing direction at a pole, where `phi` and `psi`
    become the same rotation and the whole `psi` cycle collapses to one node. The icosahedron puts
    no pixel center at a pole, but nothing else in the package would notice if it did.
    """
    grid = pose_bank(576)
    quaternions = _quaternions(grid.angles)
    overlap = np.abs(quaternions @ quaternions.T)
    np.fill_diagonal(overlap, 0.0)
    assert overlap.max() < 0.9999, "two nodes coincide, so the product has degenerated"
    assert _covering(grid.angles).max() < 90.0


def test_no_bank_in_the_usable_range_places_two_nodes_on_top_of_each_other():
    """The duplicate question asked across the range rather than at one size.

    The closest pair of distinct nodes is always the `psi` quantum or the edge length, never zero.
    The subdivision reaches the same vertex from up to six faces by different arithmetic, so a
    de-duplication that rounded too finely would leave near-duplicates.
    """
    for triangulation in (1, 3, 4, 9, 12, 16):
        quaternions = _quaternions(hopf_grid(triangulation).angles)
        overlap = np.abs(quaternions @ quaternions.T)
        np.fill_diagonal(overlap, 0.0)
        separation = 2.0 * np.degrees(np.arccos(np.clip(overlap.max(axis=1), 0.0, 1.0)))
        assert separation.min() > 3.0, f"T={triangulation} has a near-duplicate at {separation.min():.3f} deg"


def test_the_subdivision_de_duplicates_to_exactly_the_right_vertex_count():
    """`10 * frequency^2 + 2`, exactly, which is what says the rounding tolerance is right.

    The same vertex is reached from up to six faces by iteration from different seeds, so copies
    agree only to within the solver's tolerance. De-duplication rounds first: too fine and
    near-duplicates survive, too coarse and distinct vertices merge, both silently. The count is the
    sharpest check, and it is swept over a range because the spacing shrinks as `1/sqrt(T)` while the
    solver tolerance does not. Both usable classes are covered.
    """
    for triangulation in (1, 3, 4, 9, 12, 16, 25, 27, 48, 64, 108):
        assert len(geodesic_directions(triangulation)) == 10 * triangulation + 2


def test_the_triangulation_number_names_a_class_or_is_refused():
    """`T = m^2 + mn + n^2`, and only the two achiral classes are buildable.

    The refusal is the load-bearing half. Class III is chiral -- it keeps the icosahedral rotations
    and drops the inversion -- so a bank built on one has no antipodal partners to mirror and the 2x
    half of the 8x silently becomes 1x, while every node still scores. Anything not Loeschian at all
    is not a triangulation number, and is refused with a different message: the two are different
    mistakes and only one is about chirality.
    """
    for triangulation, expected in ((1, (1, 0)), (4, (2, 0)), (9, (3, 0)), (16, (4, 0)), (25, (5, 0))):
        assert subdivision_of(triangulation) == expected
    for triangulation, expected in ((3, (1, 1)), (12, (2, 2)), (27, (3, 3)), (48, (4, 4))):
        assert subdivision_of(triangulation) == expected
    for chiral in (7, 13, 19, 21, 28):
        with pytest.raises(ValueError, match="Class III"):
            subdivision_of(chiral)
    # a T with no (m, n) at all is a different mistake from a chiral one, and gets its own message
    for absent in (2, 5, 6, 8, 10, 11):
        with pytest.raises(ValueError, match="not a triangulation number"):
            subdivision_of(absent)
    with pytest.raises(ValueError, match="must be positive"):
        subdivision_of(0)


def test_the_median_miss_is_zero_against_a_bank_and_positive_against_a_draw():
    """The quantity `chain._extrapolate_marginal`'s nesting guard reads, checked at both extremes.

    The miss of a pose against the node it is comes back zero to float32, not to zero, because
    `PoseGrid.angles` is float32; the guard's tolerance has to clear that floor without ordering two
    equal misses by float noise.
    """
    grid = pose_bank(504)
    assert median_miss(grid.angles, grid) < 1e-4
    # a single pose is accepted, being the shape a caller with one particle passes
    assert median_miss(np.asarray(grid.angles)[3], grid) < 1e-4

    drawn = Rotation.random(400, random_state=11).as_euler("zyz", degrees=True)
    miss = median_miss(drawn, grid)
    assert 0.0 < miss < 30.0
    # and it FALLS as the bank refines, which is the property the guard asserts across a ladder
    assert median_miss(drawn, pose_bank(3888)) < miss

    with pytest.raises(ValueError, match="non-empty"):
        median_miss(np.empty((0, 3)), grid)


def test_the_weights_are_uniform_and_normalized():
    """Uniform weights are half the point: there is no quadrature tilt left to get wrong."""
    grid = pose_bank(97)
    assert float(jnp.exp(grid.log_weights).sum()) == pytest.approx(1.0, rel=1e-5)
    assert float(jnp.ptp(grid.log_weights)) == 0.0


def test_the_node_count_is_quantized_and_the_request_is_only_a_request():
    """`(10 * frequency^2 + 2) * n_psi` is not every integer, and the coarse end is very coarse.

    The node count is a product, so the request is only a request: below a couple of hundred every
    request returns the same smallest bank. A caller that names a ladder by node count and does not
    read the sizes back can build a "three-rung" ladder that is one point.
    """
    assert pose_bank(1).angles.shape == pose_bank(16).angles.shape == (96, 3)
    assert pose_bank(200).angles.shape == (96, 3)
    assert pose_bank(512).angles.shape == (504, 3)
    # from a couple of thousand up the request is honored to within a few per cent
    for want in (2048, 8192, 32768, 65536, 131072):
        got = pose_bank(want).angles.shape[0]
        assert abs(got / want - 1.0) < 0.11, f"{want} -> {got}"
    with pytest.raises(ValueError, match="must be positive"):
        pose_bank(0)


def test_the_bank_nests_which_is_why_a_ladder_needs_haar_poses():
    """The nesting hazard of a product bank, asserted so it is not rediscovered.

    `psi` is a uniform cycle, and a geodesic sphere at frequency `f` contains every vertex of the
    sphere at any frequency dividing `f`. So a pool drawn on one bank sits exactly on the nodes of
    some rungs of a ladder and not of the rest, and a `K^(-p)` fit over it measures which node counts
    divide the pool's own grid. `chain._extrapolate_marginal` catches this by using Haar poses; both
    halves are asserted because only the contrast -- non-monotone against a bank-drawn pool, monotone
    against a Haar one -- identifies the cause.
    """
    ladder = [pose_bank(n) for n in (1024, 2048, 3456, 8192, 16384, 27648, 65536)]

    on_bank = [median_miss(pose_bank(54).angles, grid) for grid in ladder]
    assert min(on_bank) == pytest.approx(0.0, abs=1e-9)
    assert on_bank != sorted(on_bank, reverse=True), f"expected nesting, got {on_bank}"

    haar = Rotation.random(400, random_state=3).as_euler("zyz", degrees=True)
    off_bank = [median_miss(haar, grid) for grid in ladder]
    assert off_bank == sorted(off_bank, reverse=True), f"not monotone against a Haar pool: {off_bank}"


def test_the_mean_square_miss_ignores_the_quadrature_weights():
    """A nearest-node distance has nothing to do with weight, and conflating them would be silent."""
    grid = pose_bank(576)
    rng = np.random.default_rng(0)
    tilted = PoseGrid(
        angles=grid.angles,
        log_weights=jnp.asarray(rng.uniform(-4.0, 0.0, grid.angles.shape[0]).astype(np.float32)),
    )
    assert mean_square_miss(grid, n_probe=800, seed=3) == mean_square_miss(tilted, n_probe=800, seed=3)


def test_balanced_psi_is_the_nearest_multiple_of_four_to_the_isotropy_condition():
    """Nearest, not up: rounding up over-samples `psi` and costs the bank its isotropy.

    `test_poses.test_the_hopf_bank_misses_by_the_same_amount_in_all_three_directions` is where the
    consequence is measured; this pins the arithmetic that produces it.
    """
    for n_views in (12, 48, 108, 192, 300, 432, 768):
        ideal = float(np.sqrt(np.pi * n_views))
        got = balanced_psi(n_views)
        assert got % 4 == 0 and got >= 4
        assert abs(got - ideal) <= 2.0, f"{n_views}: {got} is not the nearest multiple of four to {ideal:.2f}"
    with pytest.raises(ValueError, match="must be positive"):
        balanced_psi(0)


def test_no_request_can_select_a_class_two_bank():
    """The invariant: `--nodes` never selects a Class II bank (`T = 3 f^2`).

    Class II is still constructible -- `hopf_grid(12)` builds it and `subdivision_of` names it, which
    is how Class III is told apart -- but it must be unreachable through the quantizer. Swept rather
    than spot-checked: a single admitted `3 f^2` in `_USABLE_TRIANGULATIONS` puts the sawtooth back.
    Reads `_choose_bank` rather than building each bank; `pose_bank` is a thin wrapper over it, so the
    node count computed here is the one it would return.
    """
    for triangulation in _USABLE_TRIANGULATIONS:
        m, n = subdivision_of(triangulation)
        assert n == 0, f"T={triangulation} subdivides as ({m}, {n}), which is not Class I"

    # the assertion is on the selected triangulation, not on divisibility of the node count: `10 T + 2`
    # always divides the product, so a divisibility check would pass whatever the constant holds
    sizes = {}
    for requested in (1, 16, 54, 196, 200, 512, 864, 1648, 2812, 4096, 7892, 28248, 56112, 121824, 225456):
        triangulation, n_psi = _choose_bank(requested)
        sizes[requested] = (10 * triangulation + 2) * n_psi
        assert math.isqrt(triangulation) ** 2 == triangulation, (
            f"--nodes {requested} selected T={triangulation}, which is not a perfect square"
        )
        assert subdivision_of(triangulation)[1] == 0, f"--nodes {requested} selected a Class II bank"

    # two distinct requests that collide on one node count: a curve keyed by `realized_nodes` would
    # have had one silently overwrite the other
    assert sizes[196] == sizes[54] == 96
    assert pose_bank(196).angles.shape == (96, 3), "_choose_bank and pose_bank disagree"
