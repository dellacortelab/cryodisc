"""What the end-to-end run has to get right for an efficiency to mean anything.

One full chain is expensive, so the module-scoped fixtures run it twice -- once with the particles
on the pose bank and once off it -- and everything else reads those. The two pieces that carry the
whole experiment arm, the CTF-free contrast template and the grouped transfer, are pinned separately
and cheaply, because a sign or a binning error there would read as an algorithmic loss.
"""

from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cryodisc.forward.optics import CTFParams
from cryodisc.grids.poses import PoseGrid, pose_bank
from cryodisc.microscope import CollectionPlan, Transfer
from cryodisc.particles.sim import simulate_stack
from cryodisc.pipeline.chain import _pool_d2
from cryodisc.resolution import noise_spectrum
from cryodisc.rungs import known_pose_log_ratio, pose_log_ratio, view_angles
from cryodisc.statistics.information import deflection
from reproduction.chain import _extrapolate_marginal, _same_grid, _search_rung, run_chain

BOX_SIZE = 48
"""This pair's box floor at 2 A/voxel. A clip is common-mode -- identical in both states, so it
shrinks `d2` without making any ratio disagree -- so anything that moves `PLAN.pixel_size` has to move
this with it; `test_a_box_that_would_clip_the_templates_is_refused` is what fails if it does not.
"""
PLAN = CollectionPlan(dose=40.0, pixel_size=2.0, defocus_range=(8000.0, 20000.0))
"""2 A/voxel, which puts the box floor at 48 and is the pixel `panel/score_survey.py` scores at.

The assertions here are orderings and ratios, so the pixel buys nothing they read; it is chosen to
keep the two fixtures cheap, every render being `box^2`.
"""
GRID = pose_bank(96)
"""The smallest bank there is, which is the floor of the pool rungs' quadrature cost.

`pose_bank` quantizes hard at the bottom: every request under 193 returns the same 96-node bank, and
the next one up is 384. A larger bank only smooths the pooled number, and every assertion here is an
ordering or a ratio, so do not raise it without a reason -- the only step up is 384, at 16x the renders.
"""
VIEWS = view_angles(98)


@pytest.fixture(scope="module")
def pair(bent_pair_factory):
    """Large enough that `refined_pose` measures a pose search rather than its failure."""
    return bent_pair_factory(n_atoms=5000, spread=10.0, bend=2.5)


def _chain(pair, poses):
    return run_chain(
        pair,
        PLAN,
        key=jax.random.PRNGKey(0),
        box_size=BOX_SIZE,
        n_particles=400,
        poses=poses,
        pose_grid=GRID,
        search_views=VIEWS,
    )


@pytest.fixture(scope="module")
def on_bank(pair):
    return _chain(pair, "bank")


@pytest.fixture(scope="module")
def off_bank(pair):
    return _chain(pair, "haar")


def test_the_flat_optics_template_times_its_transfer_is_a_rendered_template(pair):
    """The experiment arm's load-bearing trick: contrast without a CTF, which cryoJAX will not emit.

    A sign error here would put the model at minus the data and read as a total algorithmic loss.
    """
    defocus = 14000.0
    model = pair.models(PLAN, box_size=BOX_SIZE)[0]
    angles = jnp.array([[10.0, 40.0, 0.0]])
    rendered = jnp.fft.fft2(model.forward_batch(angles, CTFParams.from_plan(PLAN, defocus)))
    transfer = Transfer.from_defocus(jnp.full(1, defocus), PLAN, BOX_SIZE, 1)
    reconstructed = model.bare_contrast(angles) * transfer.values[0]
    assert float(jnp.max(jnp.abs(reconstructed - rendered)) / jnp.max(jnp.abs(rendered))) < 1e-3


def test_the_transfer_groups_are_equal_count_and_in_range():
    """Equal-count bins, so no group is empty and none carries the stack."""
    defocus = jnp.asarray(np.random.default_rng(0).uniform(8000.0, 20000.0, 400))
    transfer = Transfer.from_defocus(defocus, PLAN, BOX_SIZE, 8)
    counts = np.bincount(np.asarray(transfer.index), minlength=8)
    assert transfer.values.shape == (8, BOX_SIZE, BOX_SIZE)
    np.testing.assert_array_equal(counts, np.full(8, 50))
    # the bins are ordered in defocus, so the group means are too
    means = [float(defocus[transfer.index == g].mean()) for g in range(8)]
    assert all(a < b for a, b in pairwise(means))


def test_the_two_model_only_steps_are_ordered(on_bank):
    """The only two steps of the ladder that are exact, so the only two asserted without slack.

    `local_pose` is `known_pose` minus a non-negative projection, computed in one call on one
    difference image, so the step is arithmetic rather than a claim about physics. Anything relating a
    POOL rung to a model-only one is deliberately not asserted: they average over different pose sets,
    and neither ordering follows from the definitions closely enough to be worth a test.
    """
    assert on_bank.known_pose.d2 >= on_bank.local_pose.d2
    assert on_bank.local_pose.d2_err == 0.0
    assert on_bank.marginal.d2_err > 0.0
    # logsumexp >= max holds per state, but the RATIO of two inherits no ordering, and the two rungs
    # converge as the pose posterior sharpens, so at this SNR they tie
    assert on_bank.marginal.d2 >= 0.95 * on_bank.best_match.d2


def test_the_pool_rungs_are_read_in_the_same_currency_as_the_model_rungs(on_bank):
    """Every rung is a deflection, so with the particles on the bank `marginal` has to reach the oracle.

    Both `known_pose` and `marginal` are deflections, each depressed by the same pose-spread factor
    `1 / (1 + CV^2 E[D] / 4)`. On the bank the marginal pool collapses onto the known-pose pool, so
    the two deflections coincide and the rung sits at the oracle rather than below it. Asserted on the
    RATIO: a rung in the wrong currency would not be comparable to the ladder it is quoted against.
    """
    assert on_bank.marginal.d2 / on_bank.known_pose.d2 > 0.9
    assert on_bank.best_match.d2 / on_bank.known_pose.d2 > 0.9


def test_more_information_always_means_fewer_particles(on_bank):
    """`n_for_error` is strictly decreasing in a rung's INFORMATION, so the counts run opposite it.

    Ordered by each rung's own `information`, not `d^2` or its ladder position. The count is
    `1 / (N I_p sigma^2)`, so the law lives in the information: the reported `d^2` is the
    variance-normalized deflection, and the count comes from `I_p = fisher_from_deflection(d^2)`
    (§9's eq. 6), which SATURATES at the counting ceiling `1/[p(1-p)]` -- so two rungs of different
    `d^2` above the ceiling carry equal information and equal counts, and sorting by `d^2` would not
    order them. The property being tested belongs to `statistics.power`.
    """
    rungs = sorted([on_bank.known_pose, on_bank.local_pose, on_bank.marginal], key=lambda r: r.information)
    for rung in rungs:
        assert rung.information > 0.0
        assert rung.n_for_error > 0.0
    assert all(a.n_for_error >= b.n_for_error for a, b in pairwise(rungs))


def test_a_coarse_bank_costs_far_more_than_the_fit_does(on_bank, off_bank):
    """The finding the module exists to report: with no pose refinement, the search is the loss.

    Both runs share every other input, so the gap between them is pose placement alone. The
    model-only rungs are computed before the stack is drawn, so their agreement is a property of the
    fixtures and is asserted to catch a fixture edit, not a code change. The loss is quoted in `d^2`
    and only sanity-checked in particles -- the count runs through a nonlinear `I(p)`, so its ratio
    does not track the `d^2` ratio, and only the qualitative loss is asserted.
    """
    assert on_bank.known_pose.d2 == pytest.approx(off_bank.known_pose.d2, rel=1e-6)
    assert on_bank.poses == "bank" and off_bank.poses == "haar"
    assert off_bank.marginal.d2 < 0.2 * on_bank.marginal.d2
    assert off_bank.marginal.n_for_error > 2.5 * on_bank.marginal.n_for_error


def test_the_summary_reports_every_rung_and_its_controls(on_bank):
    """Efficiency alone cannot be read: above one it is a misplaced fraction, not a free lunch."""
    text = on_bank.summary()
    for name in ("known_pose", "local_pose", "refined_pose", "marginal", "best_match", "achieved"):
        assert name in text
    assert "efficiency" in text and "fraction bias" in text and "calibration" in text
    assert "pose error" in text
    assert f"poses='{on_bank.poses}'" in text


def test_the_pose_placement_is_named_not_guessed(pair):
    with pytest.raises(ValueError, match="poses must be"):
        run_chain(pair, PLAN, key=jax.random.PRNGKey(0), box_size=BOX_SIZE, n_particles=8, poses="uniform")


def test_a_box_that_would_clip_the_templates_is_refused(pair):
    """A clip is common-mode: it hits both states identically, so it shrinks `d2` without making any
    two rungs disagree, and no ratio this module checks moves. `models` documents the floor and does
    not enforce it, `score_target` rejects the candidate, and this was the one path that quotes a
    ladder and let it through -- which is how `BOX_SIZE` sat at 48 under a 64-voxel floor."""
    floor = pair.alignment.minimum_box_size(PLAN.pixel_size)
    assert floor <= BOX_SIZE, "the module's own box must clear the floor it is testing"
    with pytest.raises(ValueError, match=f"below the {floor} voxels"):
        run_chain(pair, PLAN, key=jax.random.PRNGKey(0), box_size=floor - 2, n_particles=8)


def test_the_pose_search_does_not_care_where_the_poses_came_from(on_bank, off_bank):
    """`refined_pose` searches from scratch, so drawing particles on the bank buys it little.

    That is what separates it from `marginal`, which collapses off-bank in the test above. The two
    runs share a key, so the stacks differ only in their poses.

    "Little" rather than "nothing", because the geodesic banks nest: the 96-node bank's viewing
    directions are among those `view_angles(98)` searches, so a bank-drawn pool is handed part of its
    search for free. That is a property of the pool, not the rung, which is why `pure_pools` defaults
    to `poses="haar"` and `bank` is documented as a diagnostic.
    """
    assert on_bank.pose_error < 20.0 and off_bank.pose_error < 20.0
    assert off_bank.refined_pose.d2 == pytest.approx(on_bank.refined_pose.d2, rel=0.8)
    # the margin grows with the box, so a threshold near the measurement would be pinning the fixture
    assert off_bank.refined_pose.d2 > 2.0 * off_bank.marginal.d2


def test_the_refined_rung_pays_for_its_pose_instead_of_being_given_it(pair):
    """`refined_pose` must be the searched pose, not the true one, and only this catches the swap.

    Scored at the truth the rung is an oracle; scored at the search it pays its pose error. Nothing in
    the ladder ordering separates those two, so the oracle is compared against directly.
    """
    model_a, model_b = pair.models(PLAN, box_size=BOX_SIZE)
    spectrum = noise_spectrum(PLAN, BOX_SIZE, rmsf=0.5)
    stack = simulate_stack(
        models=(model_a, model_b),
        key=jax.random.PRNGKey(11),
        n_particles=200,
        noise_spectrum=spectrum,
        occupancy=(0.5, 0.5),
        plan=PLAN,
    )
    transfer = Transfer.from_defocus(stack.defocus, PLAN, BOX_SIZE)
    # zero shift iterations: this stack is centered, and the rung under test is the POSE cost
    searched, pose_error, shift_error = _search_rung(stack, model_a, model_b, spectrum, transfer, VIEWS, 0)
    assert shift_error == 0.0

    is_a = stack.state == 0
    at_truth = known_pose_log_ratio(
        stack.images,
        model_a.bare_contrast(stack.angles),
        model_b.bare_contrast(stack.angles),
        spectrum,
        transfer=transfer,
    )
    oracle = float(deflection(at_truth[is_a], at_truth[~is_a]))
    assert pose_error > 2.0
    # the threshold is loose because the claim is qualitative; scored at the truth by mistake the rung
    # would BE the oracle, at a ratio of exactly 1
    assert searched[0] < 0.75 * oracle


def test_the_extrapolation_reuses_the_grid_the_chain_already_scored(pair):
    """`_extrapolate_marginal` must not re-score the bank `run_chain` has already scored.

    An extrapolation ladder whose top rung is the module's own `GRID` measures that point twice from
    identical inputs -- a full marginal pass over its poses per particle plus a re-render of both
    banks, on an arm that driver's own docstring measures in hours.

    Two claims, and the second is the one that matters: the reused number IS the re-scored number,
    and the extrapolated intercept is therefore unchanged. If `_same_grid` ever matched a grid that
    would score differently, the first assertion is what fails.
    """
    model_a, model_b = pair.models(PLAN, box_size=BOX_SIZE)
    spectrum = noise_spectrum(PLAN, BOX_SIZE, rmsf=0.5)
    stack = simulate_stack(
        models=(model_a, model_b),
        key=jax.random.PRNGKey(3),
        n_particles=100,
        noise_spectrum=spectrum,
        occupancy=(0.5, 0.5),
        plan=PLAN,
    )
    transfer = Transfer.from_defocus(stack.defocus, PLAN, BOX_SIZE)
    is_a = stack.state == 0
    # the top rung is a SEPARATE grid of the module's node count rather than GRID itself, which is
    # the case that actually arises: the driver builds its own, so object identity would never match
    # three DISTINCT banks: `pose_bank` quantizes hard at the coarse end, and everything below 234
    # nodes is the same 96-node bank, so a ladder named by node count can silently be one point
    grids = [pose_bank(GRID.angles.shape[0]), pose_bank(504), pose_bank(1840)]

    # exactly what `run_chain` computes on its way to the single-grid `marginal` rung
    banks = jnp.stack([model.bare_contrast(GRID.angles) for model in (model_a, model_b)])
    ratios = pose_log_ratio(
        stack.images, banks[0], banks[1], spectrum, pose_log_weights=GRID.log_weights, transfer=transfer
    )
    scored = _pool_d2(ratios.marginal, is_a)

    rescored = _extrapolate_marginal(stack, (model_a, model_b), spectrum, transfer, None, grids, is_a)
    reused = _extrapolate_marginal(
        stack, (model_a, model_b), spectrum, transfer, None, grids, is_a, already_scored=(GRID, scored)
    )

    assert rescored.counts == reused.counts == (GRID.angles.shape[0], 504, 1840)
    assert reused.values[0] == pytest.approx(scored[0], rel=1e-6)
    # banded, unlike the exact reuse above: re-scoring is a separate `pose_log_ratio` call and XLA's
    # CPU reduction is not bit-reproducible. What is pinned is that reuse did not substitute a rung
    assert rescored.values == pytest.approx(reused.values, rel=1e-4)
    assert reused.limit == pytest.approx(rescored.limit, rel=1e-4)
    assert reused.limit_err == pytest.approx(rescored.limit_err, rel=1e-3)


def test_a_grid_is_matched_by_what_it_would_score_and_not_by_where_it_came_from():
    """The match is on the two fields the scorer reads, not on a spec or on identity.

    `pose_bank` is a pure function of its node count, so a count now WOULD settle it for banks that
    came from there -- but a caller may pass a bank from anywhere, and the pair below is the case: same
    node count, different nodes, which a count-based match would wrongly call interchangeable. It runs
    the other way too: two REQUESTS that quantize to the same bank are interchangeable, which a match
    on the requested count would miss.
    """
    assert _same_grid(pose_bank(96), pose_bank(96))
    assert not _same_grid(pose_bank(96), pose_bank(504))
    # two requests that quantize to ONE bank, which a count-based match would call different.
    # 96 is the smallest bank and everything below 193 falls back to it
    assert _same_grid(pose_bank(16), pose_bank(150))
    # same count, different nodes: a bank a caller built itself, which is what a count cannot catch
    drawn = Rotation.random(96, random_state=1).as_euler("zyz", degrees=True).astype(np.float32)
    elsewhere = PoseGrid(angles=jnp.asarray(drawn), log_weights=pose_bank(96).log_weights)
    assert elsewhere.angles.shape == pose_bank(96).angles.shape
    assert not _same_grid(elsewhere, pose_bank(96))
