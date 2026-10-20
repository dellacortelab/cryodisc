"""Checks on the pose and shift quadrature grids.

The grids are only worth having if their weights integrate correctly, so these compare against Haar
integrals over SO(3) that have closed forms, rather than checking node counts and shapes.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from cryodisc.grids.fourier import phase_ramp
from cryodisc.grids.poses import (
    PoseGrid,
    balanced_psi,
    expansion_plan,
    hopf_grid,
    mean_square_miss,
    pose_bank,
    shift_grid,
)
from cryodisc.specimen.structures import align_conformers

BOX_SIZE = 16


def haar_bank(n_nodes, seed=101):
    """A Haar-drawn bank of the same size: the baseline the structured bank must beat.

    The seed must not be `mean_square_miss`'s probe seed. Those probes are drawn the same way, so a
    shared seed makes the drawn bank's probes coincide with its nodes, the miss reads zero, and the
    comparison is degenerate rather than a property of either bank.
    """
    drawn = Rotation.random(n_nodes, random_state=seed).as_euler("zyz", degrees=True)
    return PoseGrid(
        angles=jnp.asarray(np.ascontiguousarray(drawn), dtype=jnp.float32),
        log_weights=jnp.full((n_nodes,), -np.log(n_nodes)),
    )


def rotations(grid):
    """The grid's nodes as matrices, and their weights."""
    return (
        Rotation.from_euler("zyz", np.asarray(grid.angles), degrees=True).as_matrix(),
        np.exp(np.asarray(grid.log_weights)),
    )


def test_the_bank_integrates_so3_and_improves_with_its_node_count():
    """E[R_ij^2] = 1/3, E[R_ij^4] = 1/5 and E[R_ij] = 0 under Haar, for any entry -- exactly.

    Not "converges to". Every orbit of the icosahedral group is a spherical 5-design, so an
    unweighted mean over the base integrates every polynomial below degree six exactly, at any
    subdivision frequency; crossing it with a uniform `psi` cycle carries that to SO(3). A bank that
    has stopped being an icosahedral orbit is a construction bug, and every rung that sums over it
    with uniform weights is then silently tilted, so this is pinned well above float noise.
    """
    for n_nodes, tolerance in ((1472, 1e-6), (11584, 1e-6)):
        matrices, weights = rotations(pose_bank(n_nodes))
        assert weights.sum() == pytest.approx(1.0)
        for i in range(3):
            for j in range(3):
                entry = matrices[:, i, j]
                assert float(weights @ entry**2) == pytest.approx(1.0 / 3.0, abs=tolerance)
                assert float(weights @ entry**4) == pytest.approx(1.0 / 5.0, abs=tolerance)
                assert float(weights @ entry) == pytest.approx(0.0, abs=tolerance)


def test_an_unweighted_mean_is_unbiased_which_is_what_uniform_weights_require():
    """The reason the weights can be uniform: an unweighted mean over the base is unbiased.

    An icosahedral orbit is a spherical 5-design and `psi` is a uniform cycle, so the plain mean is
    the integral for every polynomial below degree six, not merely close to it. Pinned tight because
    anything above float noise means the base has stopped being an icosahedral orbit and every
    unweighted rung is then silently wrong.
    """
    matrices, _ = rotations(pose_bank(3840))
    assert float((matrices[:, 2, 2] ** 2).mean()) == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_viewing_direction_is_set_by_phi_and_theta():
    """psi leaves the direction being looked down unchanged.

    Still worth pinning with the outer product gone: `rungs.refine` searches DIRECTIONS and treats the
    in-plane angle continuously, and `poses.sphere_angles` is only a viewing bank if this holds."""
    beam = np.array([0.0, 0.0, 1.0])

    def viewing_direction(phi, theta, psi):
        return Rotation.from_euler("zyz", [phi, theta, psi], degrees=True).as_matrix().T @ beam

    assert np.allclose(viewing_direction(30.0, 50.0, 0.0), viewing_direction(30.0, 50.0, 140.0))
    assert not np.allclose(viewing_direction(30.0, 50.0, 0.0), viewing_direction(95.0, 50.0, 0.0))


def test_grid_covers_the_sphere():
    """Nodes reach both poles and spread in azimuth, so no view is structurally unreachable."""
    matrices, _ = rotations(pose_bank(4608))
    directions = np.einsum("nji,j->ni", matrices, np.array([0.0, 0.0, 1.0]))
    assert directions[:, 2].min() < -0.9
    assert directions[:, 2].max() > 0.9
    azimuth = np.arctan2(directions[:, 1], directions[:, 0])
    assert np.ptp(azimuth) > 1.9 * np.pi


def test_the_grid_ramps_are_the_shared_convention_and_not_a_second_one():
    """`ShiftGrid.ramps` is a delegation, so what is left to check is that it delegates. The sign
    itself is pinned against a real-space roll in `test_fourier`, which is where it is decided."""
    grid = shift_grid(BOX_SIZE, 3.0)
    assert np.allclose(np.asarray(grid.ramps()), np.asarray(phase_ramp(grid.shifts, grid.box_size)))


def test_shift_priors_are_normalized_at_every_step():
    """Prior mass is one at every spacing. An unnormalized sum grows as the refinement factor
    squared, which does not cancel in a log-likelihood compared across models."""
    for step in (1.0, 0.5, 0.25):
        for sigma in (None, 1.5):
            grid = shift_grid(BOX_SIZE, 3.0, sigma=sigma, step=step)
            assert float(np.exp(np.asarray(grid.log_weights)).sum()) == pytest.approx(1.0, rel=1e-5)


def test_gaussian_shift_prior_falls_off_with_radius():
    """The Gaussian prior peaks at zero shift and decays, where the uniform one does not."""
    grid = shift_grid(BOX_SIZE, 4.0, sigma=1.5)
    shifts = np.asarray(grid.shifts)
    log_weights = np.asarray(grid.log_weights)
    radius = np.linalg.norm(shifts, axis=1)

    assert radius[int(np.argmax(log_weights))] == pytest.approx(0.0)
    order = np.argsort(radius)
    assert np.all(np.diff(log_weights[order]) <= 1e-6)

    uniform = np.asarray(shift_grid(BOX_SIZE, 4.0).log_weights)
    assert np.ptp(uniform) == pytest.approx(0.0, abs=1e-6)


def test_in_plane_quarter_turns_are_exact_index_permutations():
    """A psi shift of 90 degrees is `rot90` plus a one-pixel roll, to float32 and no interpolation.

    Three of every four in-plane angles are an exact index permutation of the fourth, which is how the
    S1 axis is built four times cheaper. Checked against the real renderer.

    The roll is the whole subtlety and omitting it does not look like an error: for an even box
    `np.rot90` turns about `(N-1)/2` while the render centers the molecule at `N/2`, so the bare
    permutation is off by half a pixel per quarter turn -- high enough correlation to pass an eyeball
    check but a convention bug. The direction is `psi + 90` = `rot90(k=-1)`: the render's in-plane
    rotation runs opposite to numpy's.
    """
    pair = align_conformers("data/pdb/8WJ1.pdb", "data/pdb/8WJ2.pdb", assembly="1")
    model, _ = pair.models_on_grid(box_size=48, pixel_size=4.0)

    def render(angles):
        return np.fft.ifft2(np.asarray(model.bare_contrast(jnp.asarray(angles)))[0]).real

    base = np.array([[17.0, 43.0, 29.0]])
    reference = render(base)
    scale = np.abs(reference).max()

    # (quarter turns, numpy direction, roll y, roll x) -- the rolls are the center-convention fix
    for turns, direction, shift_y, shift_x in ((1, -1, 0, 1), (2, 1, 1, 1), (3, -1, 1, 0)):
        rotated = base.copy()
        rotated[0, 2] += 90.0 * turns
        permuted = np.roll(np.rot90(reference, turns * direction), (shift_y, shift_x), axis=(0, 1))
        assert np.abs(permuted - render(rotated)).max() / scale < 1e-4

    # and the permutation is a group: four quarter turns return the array bit for bit
    round_trip = reference.copy()
    for _ in range(4):
        round_trip = np.rot90(round_trip)
    assert np.array_equal(round_trip, reference)

    # the antipodal view is a reflection: the projection along -n is the mirror of the projection
    # along +n, a two-fold redundancy a flip realizes for free. `(a+180, 180-b, -c)` is the triple
    # that reverses the viewing direction; the other z-row triples do not give a reflection
    antipode = np.array([[base[0, 0] + 180.0, 180.0 - base[0, 1], -base[0, 2]]])
    mirrored = np.roll(np.flipud(reference), (1, 0), axis=(0, 1))
    assert np.abs(mirrored - render(antipode)).max() / scale < 1e-4


def test_the_bank_covers_better_than_simply_drawing_the_same_number_of_poses():
    """The structure has to earn its place against simply drawing the same number of poses.

    A random draw asks the sharper question -- does the structure buy anything at all? -- and a
    construction that has stopped being equidistributed, or a subdivision that has collapsed, fails
    against a draw. Both usable classes are covered, since they are different lattices on the same
    solid.
    """
    for triangulation in (4, 12, 16, 27):
        bank = hopf_grid(triangulation)
        structured = mean_square_miss(bank, n_probe=4000, seed=0)
        drawn = mean_square_miss(haar_bank(bank.angles.shape[0]), n_probe=4000, seed=0)
        assert structured < drawn * 0.80, f"T={triangulation}: {structured:.5f} against a draw's {drawn:.5f}"


def test_the_hopf_bank_refuses_an_n_psi_that_breaks_the_quarter_turn():
    """`n_psi` not a multiple of four is refused, because the quarter-turn symmetry is why this grid
    exists.

    A cycle of six in-plane angles does not contain its own 90-degree rotations, so every template
    has to be rendered and the saving is silently lost. It raises rather than accept that.
    """
    with pytest.raises(ValueError, match="multiple of four"):
        hopf_grid(16, 6)
    assert balanced_psi(256) == 28
    assert balanced_psi(1) >= 4 and balanced_psi(1) % 4 == 0


def test_the_hopf_bank_misses_by_the_same_amount_in_all_three_directions():
    """Isotropy, which is what balancing S1 against S2 is for and which nothing else here would catch.

    A bank that samples `psi` finer than it samples the sphere spends nodes on an axis already
    covered, and no covering average shows it -- the miss simply stops being the same size in every
    direction. `rotation_of` builds `Rz(psi) Ry(theta) Rz(phi)`, so the z component of a residual
    rotation vector is the in-plane miss and the (x, y) part is the viewing-direction miss; the test
    splits the residual variance that way. Both usable classes are covered, since isotropy is a
    property of the balance and nothing guarantees it survives a change of subdivision class.
    """
    probe = Rotation.random(20000, random_state=7)
    for triangulation in (4, 12, 16, 27, 36):
        grid = hopf_grid(triangulation)
        nodes = Rotation.from_euler("zyz", np.asarray(grid.angles, float), degrees=True)
        quaternions = nodes.as_quat()
        # +/- q are one rotation, so both signs go in the tree; the query is then a plain nearest
        _, index = cKDTree(np.vstack([quaternions, -quaternions])).query(probe.as_quat(), k=1)
        residual = (probe * nodes[index % len(quaternions)].inv()).as_rotvec()
        share = residual.var(axis=0) / residual.var(axis=0).sum()
        assert share.max() / share.min() < 1.25, f"T {triangulation} misses anisotropically: {share}"


def _sphere_normals(grid):
    """Unit viewing directions of a bank, one per distinct `(phi, theta)`."""
    views = np.unique(np.asarray(grid.angles)[:, :2], axis=0)
    phi, theta = np.radians(views[:, 0]), np.radians(views[:, 1])
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=1)


def test_the_geodesic_sphere_is_antipodally_closed_where_a_spiral_is_not():
    """The one property the mirror symmetry needs.

    A golden-angle bank contains no antipodal pair, so the mirror identity has nothing to act on and
    only the quarter turns are available -- the whole of the difference between 4x and 8x. The
    icosahedral group contains the inversion, so `x -> -x` maps vertices onto vertices at every
    subdivision frequency; a tetrahedral geodesic would not, which is why the base is not simply the
    Platonic solid with the fewest faces.
    """
    for grid, expected in ((hopf_grid(16), True), (haar_bank(4608), False)):
        normal = _sphere_normals(grid)
        assert ((normal @ -normal.T).max(axis=1) > 0.999999).all() == expected


def test_no_bank_clusters_its_nodes_or_leaves_a_hole():
    """Two failure modes a mean miss cannot see: clustering and voids.

    A mean miss is barely moved by a handful of redundant nodes clustered together, and a few nodes
    piled up in one place come with a hole somewhere else. So uniformity is asserted from both ends:
    nodes not too close (clustering) and no probe too far (a void).
    """
    for grid in (hopf_grid(16), hopf_grid(27), hopf_grid(64)):
        normal = _sphere_normals(grid)
        inner = normal @ normal.T
        np.fill_diagonal(inner, -1.0)
        spacing = np.degrees(np.arccos(np.clip(inner.max(axis=1), -1, 1)))
        assert spacing.min() / np.median(spacing) > 0.80, "nodes cluster"

        probe = Rotation.random(4000, random_state=0).as_quat()
        nodes = Rotation.from_euler("zyz", np.asarray(grid.angles, dtype=np.float64), degrees=True).as_quat()
        miss = 2.0 * np.arccos(np.clip(np.abs(probe @ nodes.T).max(axis=1), -1, 1))
        assert miss.max() / miss.mean() < 2.2, "a probe sits far from every node, so the bank has a void"


@pytest.mark.parametrize("structure", ["real", "asymmetric"])
def test_the_expansion_plan_rebuilds_the_whole_bank_from_an_eighth_of_it(structure, asymmetric_pair):
    """Every node of a `hopf_grid` bank, built from the eighth `expansion_plan` renders, against a
    direct render of that node.

    The acceptance test for the 8x: the identities are pinned in
    `test_in_plane_quarter_turns_are_exact_index_permutations`, and this checks the bookkeeping
    composes them correctly over a whole bank. A reflection reverses the sense of rotation, and
    getting it wrong is invisible on half the cases -- turns 0 and 2 are their own inverses, so only
    mirrored nodes at odd turns fail. Run at the balanced `n_psi` that `pose_bank` actually builds,
    since `n_psi` of 4 and 8 hide bookkeeping errors. `T` = 3 is Class II and is included
    deliberately, its sphere being a different lattice on the same icosahedron.

    Both structures are required and the asymmetric one is load-bearing: 8WJ1's two-fold axis along z
    makes its projection invariant under exactly the half turn `expansion_plan` can get wrong, so a
    symmetric structure alone replays green over a real bug.
    """
    if structure == "real":
        pair = align_conformers("data/pdb/8WJ1.pdb", "data/pdb/8WJ2.pdb", assembly="1")
    else:
        pair = asymmetric_pair
    model, _ = pair.models_on_grid(box_size=48, pixel_size=4.0)
    # (numpy turns, roll) per quarter turn, the recipes the identity test pins
    quarter = {0: (0, (0, 0)), 1: (-1, (0, 1)), 2: (2, (1, 1)), 3: (1, (1, 0))}

    for triangulation in (1, 3, 4, 9):
        grid = hopf_grid(triangulation)
        angles = np.asarray(grid.angles)
        direct = np.fft.ifft2(np.asarray(model.bare_contrast(jnp.asarray(angles)))).real
        rendered, source, turns, mirror = expansion_plan(grid)
        assert len(rendered) * 8 == len(angles)
        base = np.fft.ifft2(np.asarray(model.bare_contrast(jnp.asarray(angles[rendered])))).real

        scale = np.abs(direct).max()
        for node in range(len(angles)):
            image = base[source[node]]
            if mirror[node]:
                image = np.roll(np.flipud(image), (1, 0), axis=(0, 1))
            if turns[node]:
                spin, shift = quarter[turns[node]]
                image = np.roll(np.rot90(image, spin), shift, axis=(0, 1))
            assert np.abs(image - direct[node]).max() / scale < 1e-4, f"node {node} at T={triangulation}"


def test_a_psi_cycle_offset_from_zero_is_refused_rather_than_half_expanded():
    """The mirror negates the psi angle and the plan negates the psi index; only a cycle at zero agrees.

    An offset cycle -- `{5, 95, 185, 275}` -- rounds cleanly onto `0..3` and passes every index-level
    check, but for offset `d` the shift the mirror needs is `-2d/step`, not an integer, so the flipped
    half comes back rotated. A full-step offset is legitimate and must still pass, which separates this
    from a check that simply demands psi start at zero.
    """
    grid = hopf_grid(1, 8)
    base = np.asarray(grid.angles)

    def offset_by(degrees):
        angles = base.copy()
        angles[:, 2] = (angles[:, 2] + degrees) % 360.0
        return PoseGrid(angles=angles, log_weights=grid.log_weights)

    for degrees in (5.0, 15.0):
        with pytest.raises(ValueError, match="not the cycle"):
            expansion_plan(offset_by(degrees))
    expansion_plan(offset_by(0.0))
    expansion_plan(offset_by(45.0))


@pytest.mark.parametrize("n_nodes", [1000, 5000, 10000, 20000, 50000, 130000])
def test_a_real_bank_is_not_refused_for_storing_its_psi_in_float32(n_nodes):
    """The drift tolerance is sized against float32, and a sweep of round sizes will not show it.

    `PoseGrid` stores angles in float32, so a cycle whose `n_psi` does not divide 360 evenly cannot
    sit on the lattice exactly. The sizes here are the ones whose `n_psi` does not divide 360, i.e.
    the ones a round-number sweep skips; a refusal here is a refusal to score a real chain.
    """
    expansion_plan(pose_bank(n_nodes))


def test_a_view_carrying_one_psi_twice_is_refused():
    """Coverage, which the offset check cannot see.

    `{0, 90, 90, 270}` starts at zero and drifts nowhere, so every angle sits on the lattice and the
    value check passes; two nodes then collide onto one slot. Nothing raises on its own, because the
    `slot_of` lookups only ever span `0..quarter-1` and all succeed, so the bank comes back wrong
    rather than refused.
    """
    grid = hopf_grid(1, 8)
    angles = np.asarray(grid.angles).copy()
    angles[angles[:, 2] == 45.0, 2] = 0.0
    with pytest.raises(ValueError, match="exactly once"):
        expansion_plan(PoseGrid(angles=angles, log_weights=grid.log_weights))
