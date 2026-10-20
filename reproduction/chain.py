"""Predict the separation, attempt it in silico, and report both -- the full validation ladder.

The paper apparatus behind `pipeline.chain.predict`. Where `predict` reports what the theory permits,
this goes on to ATTEMPT the separation: it searches each pose, runs the EM fit, and measures what the
algorithm actually recovers. Both arms run on one pair under one collection, so the predictions and
the achieved number read against each other; their gap (`efficiency`) is the algorithmic loss. The
shared setup is `pipeline.chain._score_pair`; this module adds the search, the fit and the controls.

Rationale: docs/rationale/chain.md
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from cryodisc.forward.cryojax_model import CryoJAXModel
from cryodisc.grids.fourier import phase_ramp
from cryodisc.grids.poses import (
    PoseGrid,
    ShiftGrid,
    expansion_plan_or_none,
    geodesic_error,
    median_miss,
)
from cryodisc.microscope import CollectionPlan, Transfer
from cryodisc.particles.em import fit_noise_model
from cryodisc.particles.sim import ParticleStack
from cryodisc.pipeline.chain import (
    Rung,
    _closed_form_deflection,
    _known_pose_rung,
    _marginal_rung,
    _pool_d2,
    _rung,
    _score_pair,
)
from cryodisc.pipeline.operating_point import RUNG_NAMES, OperatingPoint
from cryodisc.rungs import (
    Calibration,
    achieved_information,
    calibration_curve,
    fraction_bias,
    known_pose_log_ratio,
    pipeline_efficiency,
    refine_poses,
    streamed_pose_log_ratio,
    view_angles,
)
from cryodisc.specimen.structures import ConformerPair
from cryodisc.statistics.convergence import Extrapolation, extrapolate_in_k
from cryodisc.statistics.power import n_for_error

_MISS_TOLERANCE = 1e-3
"""Degrees a ladder's median miss may rise between rungs before the nesting guard refuses.

Not zero: `PoseGrid.angles` is float32, so a pose compared against the node it IS reads a tiny nonzero
miss, and the tolerance has to sit above that rounding floor and below any real ladder's miss.
"""

_DEFAULT_VIEWS = 72
"""Directions `refined_pose` searches when a caller names none, as `view_angles` places them.

Separate from the marginal bank because the in-plane angle is continuous here, so the same template
budget buys a finer bank of DIRECTIONS.
"""

_SHIFT_ITERATIONS = 3
"""Rounds of joint pose-and-translation search inside `refined_pose`.

Ignored when `shift_sigma` is zero, where the images are already centered and there is no lattice.
"""

_EM_ITERATIONS = 200
"""Iteration cap on the `particles.em` fit.

Higher than `fit_noise_model`'s bare default, since the fit here carries a pose bank and a shift
lattice that its bare caller does not.
"""


@dataclass(frozen=True)
class ChainResult:
    """The ladder, the achieved number, and the controls that say whether to believe it.

    Args:
        known_pose: The oracle. Bounds every rung below it.
        local_pose: Pose known to first order, its five directions removed; the appendix's `d^2_CRLB`.
            See `rungs.local`.
        refined_pose: Pose recovered by searching state A's bank on the noisy image, then scored as if
            known. The only rung that pays the cost of finding a pose.
        marginal: Pose integrated against its prior, on noisy images.
        best_match: Pose point-estimated on noisy images.
        achieved: What the fit delivered. `d2` is the `marginal` rung's; `n_for_error` is at the
            achieved information.
        efficiency: Achieved over marginal. One means the fit lost nothing.
        fraction_bias: Where the fit put the occupancy, against the truth. The control on
            `efficiency`, which can exceed one when the fraction is misplaced.
        calibration: Whether the responsibilities mean what they claim as probabilities.
        pose_error: Median geodesic error of the search, in degrees. The control on `refined_pose`:
            at large error the search has failed and the rung is measuring nothing.
        shift_error: Median translation error of the search, in pixels. Zero by construction when
            `shift_sigma` is zero.
        n_particles: Stack size the experiment arm ran on.
        fraction: Occupancy of state A that was simulated.
        poses: Where the particles' poses came from, which sets what `efficiency` means.
        form_factor: Which renderer built the templates, so `reproduction.validation.floor.twin_floor` can
            match a null to it.
        shift_sigma: Width in pixels of the translation the particles were drawn with. Zero is the
            centered-particle idealization.
        operating_point: The exposure every rung here was measured at. Compare two runs' before
            forming a ratio between them, or use `rung_ratio`.
        marginal_limit: The `marginal` rung extrapolated to an infinitely fine bank, when
            `marginal_grids` was given. `None` otherwise, and `marginal` is then a lower bound.
            UNCHECKED -- call `.check(ceiling=result.known_pose.d2)` before quoting it.

    Rationale: docs/rationale/chain.md
    """

    known_pose: Rung
    local_pose: Rung
    refined_pose: Rung
    marginal: Rung
    best_match: Rung
    achieved: Rung
    efficiency: float
    fraction_bias: float
    calibration: Calibration
    pose_error: float
    shift_error: float
    n_particles: int
    fraction: float
    poses: str
    form_factor: str
    shift_sigma: float
    operating_point: OperatingPoint
    marginal_limit: Extrapolation | None

    def summary(self) -> str:
        """The ladder as a table, which is how it is meant to be read.

        Returns:
            One line per rung, plus the controls.
        """
        rows = [f"{'rung':<13}{'d^2':>10}{'+-':>8}{'I_p':>10}{'N(err)':>12}", "-" * 53]
        for name in RUNG_NAMES:
            rung: Rung = getattr(self, name)
            error = f"{rung.d2_err:>8.3f}" if rung.d2_err > 0.0 else f"{'--':>8}"
            rows.append(f"{name:<13}{rung.d2:>10.3f}{error}{rung.information:>10.4f}{rung.n_for_error:>12,.0f}")
        if self.marginal_limit is not None:
            limit = self.marginal_limit
            rows.append(
                f"{'marginal(inf)':<13}{limit.limit:>10.3f}{limit.limit_err:>8.3f}"
                f"{'':>10}{'':>12}  <- extrapolated over K = {limit.counts}"
            )
        rows += [
            "-" * 53,
            f"efficiency {self.efficiency:.3f}  (achieved / marginal, poses={self.poses!r})",
            f"fraction bias {self.fraction_bias:+.3f}   calibration error {float(self.calibration.expected_error):.3f}",
            f"pose error {self.pose_error:.1f} deg   shift error {self.shift_error:.2f} px (median)",
            (
                f"{self.n_particles:,} particles simulated at fraction {self.fraction:.2f}, "
                f"shift sigma {self.shift_sigma:.1f} px"
            ),
            (
                f"operating point: template power {self.operating_point.template_power:.4g} "
                f"(box {self.operating_point.box_size}, {self.operating_point.pixel_size} A/px, "
                f"dose {self.operating_point.dose}, rmsf {self.operating_point.rmsf})"
            ),
        ]
        return "\n".join(rows)


def run_chain(
    pair: ConformerPair,
    plan: CollectionPlan,
    *,
    key: PRNGKeyArray,
    box_size: int = 128,
    rmsf: float = 0.5,
    n_particles: int = 2000,
    fraction: float = 0.5,
    poses: str = "bank",
    form_factor: str = "peng",
    pose_grid: PoseGrid | None = None,
    marginal_grids: Sequence[PoseGrid] | None = None,
    search_views: Float[Array, "v 3"] | None = None,
    shift_sigma: float = 0.0,
    error_target: float = 0.05,
) -> ChainResult:
    """Predict the separation, attempt it in silico, and report both.

    Args:
        pair: Two conformers in one frame, from `structures.align_conformers`.
        plan: The collection. The templates are built from it too, so one microscope describes both.
        key: PRNG key for the simulated stack.
        box_size: Image size in pixels. `pair.alignment.minimum_box_size` is the floor.
        rmsf: Per-axis intra-state RMS fluctuation in A, which damps the signal in both arms.
        n_particles: Stack size for the experiment arm.
        fraction: Occupancy of state A to simulate.
        poses: `"bank"` puts particles on the quadrature nodes (fitting loss alone); `"haar"` draws
            from the true measure (adds the pose-search loss). See the module docstring.
        pose_grid: Bank the pool rungs and the fit marginalize over. `None` gives a coarse default.
        marginal_grids: Banks to re-measure the marginal rung on, so it can be extrapolated to an
            infinitely fine one; at least three of distinct size (see `statistics.convergence`).
            `None` skips it, and `marginal` is then a lower bound converging from below. A grid that
            would score identically to `pose_grid` reuses that pass rather than re-measuring.
        search_views: Viewing directions `refined_pose` searches, from `rungs.view_angles`. `None`
            gives a default at the same cost as `pose_grid`.
        shift_sigma: Width in pixels of the isotropic Gaussian each particle's translation is drawn
            from. Zero centers every particle perfectly (an inverse crime, independent of
            `poses="bank"`). See the module docstring.
        error_target: Fractional error the particle counts are quoted at.

    Returns:
        The ladder and the achieved number. `marginal_limit` comes back UNCHECKED -- call
        `marginal_limit.check(ceiling=result.known_pose.d2)` before quoting it, and the rest of the
        ladder survives a refusal.

    Raises:
        ValueError: If `poses` is neither `"bank"` nor `"haar"`, if `shift_sigma` is negative, or if
            `box_size` is below `pair.alignment.minimum_box_size`.
    """
    scored = _score_pair(
        pair,
        plan,
        key=key,
        box_size=box_size,
        rmsf=rmsf,
        n_particles=n_particles,
        fraction=fraction,
        poses=poses,
        form_factor=form_factor,
        pose_grid=pose_grid,
        shift_sigma=shift_sigma,
    )
    model_a, model_b = scored.model_a, scored.model_b
    spectrum, grid, shifts = scored.spectrum, scored.grid, scored.shifts
    stack, transfer, banks = scored.stack, scored.transfer, scored.banks
    ratios, is_a, drawn = scored.ratios, scored.is_a, scored.drawn
    marginal = scored.marginal
    best_match = _pool_d2(ratios.best_match, is_a)

    marginal_limit = None
    if marginal_grids is not None:
        marginal_limit = _extrapolate_marginal(
            stack,
            (model_a, model_b),
            spectrum,
            transfer,
            shifts,
            marginal_grids,
            is_a,
            already_scored=(grid, marginal),
        )

    views = view_angles(_DEFAULT_VIEWS) if search_views is None else search_views
    refined, pose_error, shift_error = _search_rung(
        stack, model_a, model_b, spectrum, transfer, views, 0 if shifts is None else _SHIFT_ITERATIONS
    )

    fit = fit_noise_model(
        stack.images,
        banks,
        transfer=transfer,
        pose_log_weights=grid.log_weights,
        shifts=shifts,
        n_iterations=_EM_ITERATIONS,
    )
    # summed over pose, which is the nuisance; state A is index 0 because that is how sim ordered it
    responsibilities = fit.responsibilities.sum(-1)[:, 0]
    achieved = float(achieved_information(responsibilities, fraction=drawn))

    efficiency = float(pipeline_efficiency(responsibilities, deflection=marginal[0], fraction=drawn))

    return ChainResult(
        known_pose=_known_pose_rung(scored, error_target),
        local_pose=_rung(_closed_form_deflection(scored.local.crlb), fraction=drawn, error_target=error_target),
        refined_pose=_rung(*refined, fraction=drawn, error_target=error_target),
        marginal=_marginal_rung(scored, error_target),
        best_match=_rung(*best_match, fraction=drawn, error_target=error_target),
        achieved=Rung(
            d2=marginal[0],
            d2_err=marginal[1],
            information=achieved,
            n_for_error=float(
                n_for_error(
                    jnp.asarray(marginal[0]),
                    error_target=error_target,
                    fraction=drawn,
                    efficiency=max(efficiency, 1e-6),
                )
            ),
        ),
        efficiency=efficiency,
        fraction_bias=float(fraction_bias(responsibilities, fraction=drawn)),
        calibration=calibration_curve(responsibilities, is_a),
        pose_error=pose_error,
        shift_error=shift_error,
        n_particles=n_particles,
        fraction=drawn,
        poses=poses,
        form_factor=form_factor,
        shift_sigma=shift_sigma,
        operating_point=scored.operating_point,
        marginal_limit=marginal_limit,
    )


def _extrapolate_marginal(
    stack: ParticleStack,
    models: tuple[CryoJAXModel, CryoJAXModel],
    spectrum: Float[Array, "y x"],
    transfer: Transfer,
    shifts: ShiftGrid | None,
    grids: Sequence[PoseGrid],
    is_a: Bool[Array, " n"],
    *,
    already_scored: tuple[PoseGrid, tuple[float, float]] | None = None,
) -> Extrapolation:
    """Re-measure the marginal rung on each bank and extrapolate to an infinitely fine one.

    Every grid scores the SAME particles, so the points are correlated; that inflates the fit's own
    scatter rather than biasing the intercept, and shows up in `Extrapolation.chi2_per_dof`.

    `already_scored` lets a grid `run_chain` has already scored on its way to the `marginal` rung be
    reused rather than scored twice; `_same_grid` says when it applies.

    Args:
        stack: The simulated particles. Its `angles` are read to check the ladder is admissible.
        models: State A and state B.
        spectrum: sigma^2(k) on the unshifted fft grid.
        transfer: The per-group transfer.
        shifts: The translation lattice, or `None` for centered particles.
        grids: Banks to measure on.
        is_a: True where the particle's true state is A.
        already_scored: A grid this stack has already been scored on and its `(d2, d2_err)`, reused in
            place of re-scoring whenever one of `grids` would score identically.

    Returns:
        The extrapolation, unchecked -- the caller supplies the ladder bounds.

    Raises:
        ValueError: If the pool's miss to the banks does not fall with their node count, which means
            the poses nest into the ladder and the fit would read divisibility rather than refinement.
            Draw the pool from Haar (`pure_pools(poses='haar')`) to satisfy it.
    """
    ordered = sorted(grids, key=lambda g: int(g.angles.shape[0]))
    misses = [median_miss(stack.angles, grid) for grid in ordered]
    if any(later > earlier + _MISS_TOLERANCE for earlier, later in pairwise(misses)):
        raise ValueError(
            "the pool's poses nest into this ladder: median miss "
            + ", ".join(f"{m:.3f}d at K={int(g.angles.shape[0])}" for g, m in zip(ordered, misses, strict=True))
            + ". A finer bank must sit CLOSER to the particles, or the K fit measures which node "
            "counts happen to divide the pool's own grid. Draw the pool from Haar -- "
            "`pure_pools(poses='haar')` -- rather than from a bank."
        )
    counts, values, errors = [], [], []
    for grid in grids:
        if already_scored is not None and _same_grid(grid, already_scored[0]):
            d2, d2_err = already_scored[1]
        else:
            # Streamed on both routes because this path reaches banks too large to hold whole; the
            # streamed reduction renders one chunk at a time, with arithmetic identical to whole-bank
            # (`tests/test_rungs/test_streamed.py`).
            #
            # No expansion plan on the shift route, deliberately: the eightfold expansion carries a
            # small systematic against rendered templates, and the no-shift rungs were measured with
            # it, so using it only here would bias the re-measured rows relative to their comparisons.
            ratios = streamed_pose_log_ratio(
                stack.images,
                models[0].bare_contrast,
                models[1].bare_contrast,
                grid.angles,
                grid.log_weights,
                spectrum,
                shifts=shifts,
                transfer=transfer,
                expansion=expansion_plan_or_none(grid, int(stack.images.shape[-1])) if shifts is None else None,
            )
            d2, d2_err = _pool_d2(ratios.marginal, is_a)
        counts.append(int(grid.angles.shape[0]))
        values.append(d2)
        errors.append(d2_err)
    return extrapolate_in_k(counts, values, errors)


def _same_grid(one: PoseGrid, other: PoseGrid) -> bool:
    """Whether two pose grids would score the same stack to the same number.

    `pose_log_ratio` reads a `PoseGrid` through exactly two fields -- `angles` and `log_weights` --
    so two grids agreeing on both produce a bit-identical rung, and that is the whole criterion.
    Object identity would miss a caller building its own `pose_bank(96)` alongside the default.

    Args:
        one: A grid.
        other: The grid to compare it against.

    Returns:
        True when the two are interchangeable to the scorer.
    """
    return (
        one.angles.shape == other.angles.shape
        and bool(jnp.array_equal(one.angles, other.angles))
        and bool(jnp.array_equal(one.log_weights, other.log_weights))
    )


def _search_rung(
    stack: ParticleStack,
    model_a: CryoJAXModel,
    model_b: CryoJAXModel,
    spectrum: Float[Array, "y x"],
    transfer: Transfer,
    views: Float[Array, "v 3"],
    shift_iterations: int,
) -> tuple[tuple[float, float], float, float]:
    """`refined_pose`: find each particle's pose, then score the pair there as if it were known.

    The search sees state A only, as a real pipeline sees one consensus reference. Refining each
    particle against its own state would be an oracle and would put this rung above `known_pose`.

    The translation is recovered by the same search and then DIVIDED OUT before scoring, rather than
    marginalized: this rung is the one that measures what a point estimate of the latent costs, and
    a marginal over the shift with a point estimate of the pose would be neither rung.

    Returns:
        `(d2, d2_err)` as `_pool_d2` gives it, the median geodesic pose error in degrees, and the
        median translation error in pixels. Both errors qualify the rung -- a
        search that recovered nothing still returns a confident-looking number.
    """
    found = refine_poses(
        stack.images,
        model_a.bare_contrast(views),
        views,
        spectrum,
        transfer=transfer,
        shift_iterations=shift_iterations,
    )
    # `found.shift` is the ramp that recenters, so applying it puts the particle back on the origin
    # the templates are rendered about; `-found.shift` is its estimate of the offset that was drawn
    centered = stack.images * phase_ramp(found.shift, stack.images.shape[-1])
    at_found = known_pose_log_ratio(
        centered,
        model_a.bare_contrast(found.angles),
        model_b.bare_contrast(found.angles),
        spectrum,
        transfer=transfer,
    )
    is_a = stack.state == 0
    return (
        _pool_d2(at_found, is_a),
        float(jnp.median(geodesic_error(found.angles, stack.angles))),
        float(jnp.median(jnp.linalg.norm(found.shift + stack.shift, axis=-1))),
    )
