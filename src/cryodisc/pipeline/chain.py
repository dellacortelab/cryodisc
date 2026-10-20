"""What the theory permits: how separable two conformations are, and how many particles it takes.

`predict` is the prospective half of the chain -- from two atomic models and a collection plan, it
renders the templates, simulates a stack, and reads two rungs off it: the known-pose oracle and the
pose-marginalized rung whose Fisher information sets the count. Every rung's d^2 is the deflection
of its log-ratio, converted to a count through the Gaussian Fisher map.
The in-silico separation (pose search, EM fit, achieved-vs-predicted gap) is paper apparatus in
`reproduction.chain.run_chain`, which reuses `_score_pair` here so the two never re-simulate.

Rationale: docs/rationale/chain.md
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from cryodisc.forward.cryojax_model import CryoJAXModel
from cryodisc.forward.optics import CTFParams
from cryodisc.grids.poses import PoseGrid, ShiftGrid, pose_bank, shift_grid
from cryodisc.microscope import CollectionPlan, Transfer
from cryodisc.particles.sim import ParticleStack, simulate_stack
from cryodisc.particles.snr import whitened_power
from cryodisc.pipeline.operating_point import OperatingPoint
from cryodisc.resolution import noise_spectrum
from cryodisc.rungs import LocalRung, LogRatio, local_information, pose_log_ratio
from cryodisc.specimen.structures import ConformerPair
from cryodisc.statistics.information import deflection, deflection_standard_error
from cryodisc.statistics.power import information_per_particle, n_for_error

_DEFAULT_NODES = 96
"""Nodes in the bank the pool rungs marginalize over when a caller names none.

Deliberately coarse: the marginal rung it produces is a LOWER bound converging from below. This is the
smallest bank `pose_bank` builds (12 icosahedral directions by eight in-plane angles).
"""


@dataclass(frozen=True)
class Rung:
    """One rung of the ladder, in the three units it gets read in.

    Args:
        d2: `d^2` per particle, the deflection of the log-ratio on every rung.
        d2_err: Standard error of `d2`, zero on the rungs computed from the model alone.
        information: Fisher information about the population fraction, per particle.
        n_for_error: Particles needed to pin the fraction to the target error.
    """

    d2: float
    d2_err: float
    information: float
    n_for_error: float


@dataclass(frozen=True)
class Prediction:
    """How separable two conformations are under a plan, and how many particles it would take.

    The prospective half of the chain: what the theory permits, from the structures and the plan
    alone, with no pipeline run and no bounding chain. `run_chain` computes this and then goes on to
    attempt the separation in silico; a user who only wants the prediction calls `predict`.

    Args:
        known_pose: The oracle upper bound -- `d^2` with the pose known exactly. Bounds `marginal`.
        marginal: The honest, pose-marginalized rung. Its Fisher information `I(p)` sets the count,
            because the pose is a nuisance a real experiment must integrate over rather than know.
        operating_point: The exposure the rungs were measured at. Compare two predictions' before
            forming a ratio between them.
        n_particles: Stack size the `marginal` rung's information was estimated on.
        fraction: Occupancy of state A that was simulated to estimate `marginal`.

    Rationale: docs/rationale/chain.md
    """

    known_pose: Rung
    marginal: Rung
    operating_point: OperatingPoint
    n_particles: int
    fraction: float

    def summary(self) -> str:
        """The two rungs as a short table.

        Returns:
            One line each for `known_pose` and `marginal`, plus the count the marginal information buys.
        """
        rows = [f"{'rung':<13}{'d^2':>10}{'+-':>8}{'I_p':>10}{'N(err)':>12}", "-" * 53]
        for name in ("known_pose", "marginal"):
            rung: Rung = getattr(self, name)
            error = f"{rung.d2_err:>8.3f}" if rung.d2_err > 0.0 else f"{'--':>8}"
            rows.append(f"{name:<13}{rung.d2:>10.3f}{error}{rung.information:>10.4f}{rung.n_for_error:>12,.0f}")
        rows += [
            "-" * 53,
            f"{self.n_particles:,} particles simulated at fraction {self.fraction:.2f}",
            (
                f"operating point: template power {self.operating_point.template_power:.4g} "
                f"(box {self.operating_point.box_size}, {self.operating_point.pixel_size} A/px, "
                f"dose {self.operating_point.dose}, rmsf {self.operating_point.rmsf})"
            ),
        ]
        return "\n".join(rows)


@dataclass(frozen=True)
class _Scored:
    """The spine `predict` and `run_chain` share, computed once so neither re-simulates.

    Everything before the bounding chain: the two models, the simulated stack, the pose-marginalized
    log-ratio split by state, the closed-form local rung, and the operating point they were read at.
    """

    model_a: CryoJAXModel
    model_b: CryoJAXModel
    spectrum: Float[Array, "y x"]
    grid: PoseGrid
    shifts: ShiftGrid | None
    stack: ParticleStack
    transfer: Transfer
    banks: Float[Array, "two k y x"]
    ratios: LogRatio
    local: LocalRung
    is_a: Bool[Array, " n"]
    drawn: float
    marginal: tuple[float, float]
    operating_point: OperatingPoint


def _score_pair(
    pair: ConformerPair,
    plan: CollectionPlan,
    *,
    key: PRNGKeyArray,
    box_size: int,
    rmsf: float,
    n_particles: int,
    fraction: float,
    poses: str,
    form_factor: str,
    pose_grid: PoseGrid | None,
    shift_sigma: float,
) -> _Scored:
    """Build the templates, simulate a stack, and score the pose-marginalized rung on it.

    The shared setup of both entry points: `predict` reads `known_pose` and `marginal` off it and
    stops; `run_chain` goes on to the bounding chain and the fit. Validates its inputs, so both
    entry points inherit the same guards.

    Raises:
        ValueError: If `poses` is neither `"bank"` nor `"haar"`, if `shift_sigma` is negative, or if
            `box_size` is below `pair.alignment.minimum_box_size`.
    """
    if poses not in ("bank", "haar"):
        raise ValueError(f"poses must be 'bank' or 'haar', got {poses!r}")
    if shift_sigma < 0.0:
        raise ValueError(f"shift_sigma must be non-negative, got {shift_sigma}")
    # `models` documents the box floor but does not enforce it, so check here: a clipped template is
    # clipped identically in both states, shrinking `d2` invisibly with no other downstream sign
    minimum = pair.alignment.minimum_box_size(plan.pixel_size)
    if box_size < minimum:
        raise ValueError(
            f"box_size {box_size} is below the {minimum} voxels this molecule needs at "
            f"{plan.pixel_size:g} A/voxel for its {pair.alignment.radius:.0f} A radius, so both "
            "states' templates would be clipped at the box edge"
        )
    grid = pose_bank(_DEFAULT_NODES) if pose_grid is None else pose_grid
    # one lattice for all three consumers (pool rungs, search, fit): they must marginalize the same
    # translation latent, or `efficiency` compares two different models
    shifts = _shift_lattice(box_size, shift_sigma)
    model_a, model_b = pair.models(plan, box_size=box_size, form_factor=form_factor)
    spectrum = noise_spectrum(plan, box_size, rmsf=rmsf)
    # the model-only rungs need one optics, so they get the middle of the range; the pool rungs and
    # the fit see the real spread. Via `plan.mid_defocus` to match `targets.score_target`'s floor.
    reference_optics = CTFParams.from_plan(plan, plan.mid_defocus)

    local = local_information(model_a, model_b, grid.angles, spectrum, ctf_params=reference_optics)

    stack_key, pose_key = jax.random.split(key)
    on_bank = grid.angles[jax.random.randint(pose_key, (n_particles,), 0, grid.angles.shape[0])]
    stack = simulate_stack(
        models=(model_a, model_b),
        key=stack_key,
        n_particles=n_particles,
        noise_spectrum=spectrum,
        occupancy=(fraction, 1.0 - fraction),
        angles=on_bank if poses == "bank" else None,
        plan=plan,
        shift_sigma=shift_sigma,
    )
    transfer = Transfer.from_defocus(stack.defocus, plan, box_size)
    banks = jnp.stack([model.bare_contrast(grid.angles) for model in (model_a, model_b)])

    ratios = pose_log_ratio(
        stack.images,
        banks[0],
        banks[1],
        spectrum,
        pose_log_weights=grid.log_weights,
        shifts=shifts,
        transfer=transfer,
    )
    is_a = stack.state == 0
    return _Scored(
        model_a=model_a,
        model_b=model_b,
        spectrum=spectrum,
        grid=grid,
        shifts=shifts,
        stack=stack,
        transfer=transfer,
        banks=banks,
        ratios=ratios,
        local=local,
        is_a=is_a,
        drawn=float(stack.fraction(0)),
        marginal=_pool_d2(ratios.marginal, is_a),
        operating_point=OperatingPoint(
            dose=plan.dose,
            pixel_size=plan.pixel_size,
            box_size=box_size,
            rmsf=rmsf,
            contrast=float(stack.contrast),
            template_power=whitened_power(banks[0], spectrum, transfer=transfer, contrast=float(stack.contrast)),
        ),
    )


def _rung(
    value: float,
    error: float = 0.0,
    *,
    fraction: float,
    error_target: float,
) -> Rung:
    """One rung in its three units: `d^2`, its Fisher information about the fraction, and the count.

    `I(p)` always comes through the Gaussian Fisher map of `d^2` (`information_per_particle`), so a
    model-only rung and a measured pool rung sit on one scale.
    """
    info = float(information_per_particle(jnp.asarray(value), fraction=fraction))
    count = float(n_for_error(jnp.asarray(value), error_target=error_target, fraction=fraction))
    return Rung(d2=value, d2_err=error, information=info, n_for_error=count)


def _closed_form_deflection(per_particle_d2: Float[Array, " n"]) -> float:
    """The deflection a calibrated closed-form rung would read, from its per-particle `d^2` array.

    Puts a model-only rung in the same currency as the pooled rungs. The known-pose log-ratio is
    calibrated (`Lambda|A ~ N(+d_i^2/2, d_i^2)`), so the pool it would produce has deflection
    `E[D]^2 / (E[D] + Var[D]/4)` -- the same pose-spread depression the marginal deflection carries.
    """
    mean = jnp.mean(per_particle_d2)
    variance = jnp.var(per_particle_d2)
    return float(mean**2 / jnp.maximum(mean + 0.25 * variance, 1e-12))


def _known_pose_rung(scored: _Scored, error_target: float) -> Rung:
    """The oracle rung, the closed-form deflection of the known-pose pool the model implies."""
    return _rung(_closed_form_deflection(scored.local.known_pose), fraction=scored.drawn, error_target=error_target)


def _marginal_rung(scored: _Scored, error_target: float) -> Rung:
    """The pose-marginalized rung, whose count comes from the Fisher map of its measured `d^2`.

    The pose is a nuisance a real experiment integrates over, so this rung's `d^2` is the deflection
    of the marginalized log-ratio -- below the known-pose oracle, and the honest input to the
    count. At experimental SNR the marginalized score is calibrated, so its Gaussian Fisher map is the
    direct `I(p)`; strong-signal decalibration, where the two part, is not the experimental regime.
    """
    return _rung(*scored.marginal, fraction=scored.drawn, error_target=error_target)


def predict(
    pair: ConformerPair,
    plan: CollectionPlan,
    *,
    key: PRNGKeyArray,
    box_size: int = 128,
    rmsf: float = 0.5,
    n_particles: int = 2000,
    fraction: float = 0.5,
    form_factor: str = "peng",
    pose_grid: PoseGrid | None = None,
    error_target: float = 0.05,
) -> Prediction:
    """How separable two conformations are under a plan, and how many particles it would take.

    The prospective prediction, from the structures and the plan alone -- the honest half of
    `run_chain` with no in-silico separation attempted and no bounding chain. It renders the two
    templates, simulates a stack drawn from the true pose measure, and reads two rungs off it: the
    known-pose oracle and the pose-marginalized rung whose Fisher information sets the count.

    Args:
        pair: Two conformers in one frame, from `structures.align_conformers`.
        plan: The collection. The templates are built from it too, so one microscope describes both.
        key: PRNG key for the simulated stack.
        box_size: Image size in pixels. `pair.alignment.minimum_box_size` is the floor.
        rmsf: Per-axis intra-state RMS fluctuation in A, which damps the signal.
        n_particles: Stack size the marginal rung's information is estimated on. More particles pin
            `I(p)` tighter; they do not change what it converges to.
        fraction: Occupancy of state A to simulate.
        form_factor: Which renderer builds the templates.
        pose_grid: Bank the marginal rung integrates the pose over. `None` gives a coarse default,
            which makes `marginal` a lower bound converging from below (see `run_chain`'s
            `marginal_grids` for the converged quantity).
        error_target: Fractional error the particle count is quoted at.

    Returns:
        The two rungs, the operating point, and the stack size they were read on.

    Raises:
        ValueError: If `box_size` is below `pair.alignment.minimum_box_size`.
    """
    scored = _score_pair(
        pair,
        plan,
        key=key,
        box_size=box_size,
        rmsf=rmsf,
        n_particles=n_particles,
        fraction=fraction,
        poses="haar",
        form_factor=form_factor,
        pose_grid=pose_grid,
        shift_sigma=0.0,
    )
    return Prediction(
        known_pose=_known_pose_rung(scored, error_target),
        marginal=_marginal_rung(scored, error_target),
        operating_point=scored.operating_point,
        n_particles=n_particles,
        fraction=scored.drawn,
    )


def _pool_d2(log_ratio: Float[Array, " n"], is_a: Bool[Array, " n"]) -> tuple[float, float]:
    """The discriminability `d^2` and its sampling error for a rung measured on a pool.

    Args:
        log_ratio: Per-particle Lambda from any pool rung.
        is_a: True where the particle's state is A.

    Returns:
        `(d2, d2_err)`: the deflection of the split log-ratio and its standard error.
    """
    return (
        float(deflection(log_ratio[is_a], log_ratio[~is_a])),
        float(deflection_standard_error(log_ratio[is_a], log_ratio[~is_a])),
    )


def _shift_lattice(box_size: int, shift_sigma: float) -> ShiftGrid | None:
    """The translation lattice the pool rungs, the search and the fit all marginalize over.

    The half-width is `ceil(2.5 * shift_sigma)`, which covers 99.6% of the draw in each axis. Read
    it against `shift_error` before believing a rung: a draw that lands outside the lattice is not
    marginalized, it is truncated.

    Args:
        box_size: Image size in pixels.
        shift_sigma: Width of the Gaussian the true translations are drawn from. Zero returns
            `None`, which is the centered-particle path every consumer takes as a fast case.

    Returns:
        The grid, whose prior is the same Gaussian the draw came from, or `None`.
    """
    if shift_sigma <= 0.0:
        return None
    reach = np.ceil(2.5 * shift_sigma)
    # step 1.0 because `rungs.marginal` scores a whole lattice from one transform and reaches the
    # integer offsets only; a finer step there costs a transform per sub-pixel offset
    return shift_grid(box_size, float(reach), sigma=shift_sigma, step=1.0)
