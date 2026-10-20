"""Two pure pools from a structure pair, scored -- what every composition benchmark starts from.

Used by `known_fraction.py`, and here rather than in the driver: two drivers drawing pools by
slightly different routes would compare two forward models while appearing to compare two
estimators.

**The INVERSE-CRIME arm.** Particles come from the model that scores them, so a departure here is
the estimator's and not the physics'. Wrong arm for any claim about a real specimen.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from cryodisc.grids.poses import expansion_plan_or_none, pose_bank
from cryodisc.microscope import CollectionPlan, Transfer
from cryodisc.particles.sim import simulate_stack
from cryodisc.particles.snr import matched_filter_snr, streamed_whitened_power
from cryodisc.resolution import noise_spectrum
from cryodisc.rungs.marginal import (
    known_pose_d2,
    known_pose_log_ratio,
    pose_log_ratio,
    streamed_pose_log_ratio,
)
from cryodisc.specimen.structures import ConformerPair


@dataclass(frozen=True)
class PurePools:
    """Two pools believed pure in one state each, with what is needed to read their nulls.

    Args:
        log_ratio_a: Marginalized `Lambda` on the state-A pool.
        log_ratio_b: The same on the state-B pool.
        banks: The two states' CTF-free template banks, or `None` when the pools were scored
            with `stream=True` and no whole bank was ever built. Only the null arms want them.
        spectrum: The noise spectrum both pools ran at.
        pose_log_weights: Quadrature weights of the bank the score marginalized over.
        transfer: State A's transfer, standing in for both: one defocus range drew them.
        snr: Matched-filter SNR the pools ran at.
        oracle_log_ratio_a: The SAME images scored at their OWN poses. The coherence identity is
            exact for a known-pose `Lambda` and has no closed form for a marginalized one.
        oracle_log_ratio_b: The same for state B.
        oracle_d2_a: `d_i^2` from the templates and NO stack, so a check on the score rather than a
            restatement of it.
        oracle_d2_b: The same for state B.
        amplitude: The template scale `a` the pools were SCORED at. Carried so a null (e.g.
            `validation.floor.noise_floor`) runs at the measurement's own exposure, which a caller
            forwards rather than has to match by hand.
    """

    log_ratio_a: Float[Array, " n"]
    log_ratio_b: Float[Array, " n"]
    banks: tuple | None
    spectrum: Float[Array, "y x"]
    pose_log_weights: Float[Array, " k"]
    transfer: Transfer
    snr: float
    oracle_log_ratio_a: Float[Array, " n"]
    oracle_log_ratio_b: Float[Array, " n"]
    oracle_d2_a: Float[Array, " n"]
    oracle_d2_b: Float[Array, " n"]
    amplitude: float = 1.0


def _concatenate_transfers(parts: list[Transfer]) -> Transfer:
    """One `Transfer` over several pools' particles, each keeping its own groups.

    Args:
        parts: One transfer per pool, in the order their particles are concatenated.

    Returns:
        A transfer whose `values` are every pool's groups end to end and whose `index` points each
        particle at the group it had in its own pool.
    """
    offsets, total = [], 0
    for part in parts:
        offsets.append(total)
        total += part.values.shape[0]
    return Transfer(
        values=jnp.concatenate([part.values for part in parts]),
        index=jnp.concatenate([part.index + offset for part, offset in zip(parts, offsets, strict=True)]),
    )


def simulate_pure_pools(
    pair: ConformerPair,
    *,
    n_particles: int,
    box_size: int = 64,
    pixel_size: float = 1.5,
    dose: float = 40.0,
    rmsf: float = 0.5,
    nodes: int = 96,
    poses: str = "haar",
    defocus_error: float = 0.0,
    stream: bool = False,
    pose_chunk: int | None = None,
    seed: int = 0,
) -> PurePools:
    """Draw and score one pure pool per state.

    Args:
        pair: The aligned two-state pair.
        n_particles: Particles in EACH pool.
        box_size: Image size in pixels.
        pixel_size: A/px.
        dose: Total electron dose, e/A^2.
        rmsf: Per-axis intra-state RMS fluctuation in A.
        nodes: Roughly how many poses the score marginalizes over. `poses.pose_bank` quantizes it,
            coarsely below about 2000 -- read the count back rather than assuming this one.
        defocus_error: Per-particle rms RELATIVE error on the defocus the SCORER is told; the images
            keep the true one. Relative, because a fitter's error scales with defocus.
        poses: `"haar"` draws from the true measure (default, and what a real pool is); the bank must
            then cover SO(3) well enough to represent the draw. `"bank"` puts particles ON the
            quadrature nodes, isolating the COMPOSITION estimator -- a DIAGNOSTIC, not a measurement,
            and one that nests: a pool on one bank sits exactly on the nodes of others, so a `K` ladder
            built on it reads divisibility rather than refinement (`chain._extrapolate_marginal`
            refuses one).
        stream: Score through `streamed_pose_log_ratio`, which renders and reduces the bank in chunks
            instead of materializing it -- same answer, a memory knob not a modeling one. Turn it on
            for large banks, where the whole-bank path's `2 * p * box^2` peak exceeds the card.
            `banks` comes back `None` when set.
        pose_chunk: Poses per chunk when `stream` is set, or `None` to let the scorer size it.
            Ignored otherwise.
        seed: PRNG seed.

    Returns:
        The pools and their nulls.
    """
    if defocus_error < 0.0:
        raise ValueError(f"defocus_error is an rms and cannot be negative, got {defocus_error}")

    # carried out on PurePools.amplitude so a null runs at the measurement's exposure
    amplitude = 1.0
    plan = CollectionPlan(dose=dose, pixel_size=pixel_size)
    # the same floor run_chain and score_target enforce: without it a pool clips a molecule wider
    # than the box silently, so refuse rather than truncate
    minimum = pair.alignment.minimum_box_size(pixel_size)
    if box_size < minimum:
        raise ValueError(
            f"box_size {box_size} is below the {minimum} voxels this molecule needs at "
            f"{pixel_size:g} A/voxel for its {pair.alignment.radius:.0f} A radius, so both "
            "states' templates would be clipped at the box edge"
        )
    model_a, model_b = pair.models(plan, box_size=box_size)
    grid = pose_bank(nodes)
    spectrum = noise_spectrum(plan, box_size, rmsf=rmsf)
    # the whole bank is what does not fit at large K, so on the streamed path it is never built
    banks = None if stream else [model.bare_contrast(grid.angles) for model in (model_a, model_b)]
    expansion = expansion_plan_or_none(grid, box_size) if stream else None

    log_ratios, transfers, oracle_log_ratios, oracle_d2s = [], [], [], []
    # Each pool keeps its own defocus grouping, which is what makes the fusion exact: rebuilding one
    # Transfer.from_defocus over the concatenated defocus would bin the two pools together and hand
    # each particle a CTF from the wrong mean defocus.
    # The streamed path scores both pools in one fused sweep (the whole-bank path reuses one `banks`,
    # but a streamed call re-renders the bank per state). `pose_log_ratio` is per particle, so 2n in
    # one call equals n in each of two; `_pose_chunk` halves the chunk so peak memory is unchanged.
    streamed_images, streamed_transfers = [], []
    for state, key in enumerate(jax.random.split(jax.random.key(seed), 2)):
        # occupancy puts the whole pool in one state, which is what "pure by construction" means
        occupancy = [1.0, 0.0] if state == 0 else [0.0, 1.0]
        pose_key, stack_key = jax.random.split(key)
        angles = (
            grid.angles[jax.random.randint(pose_key, (n_particles,), 0, grid.angles.shape[0])]
            if poses == "bank"
            else None
        )
        stack = simulate_stack(
            (model_a, model_b),
            key=stack_key,
            n_particles=n_particles,
            noise_spectrum=spectrum,
            occupancy=jnp.asarray(occupancy),
            plan=plan,
            angles=angles,
        )
        # banks are CTF-free and the stack is rendered through the plan's defocus, so the transfer
        # must reach the scorer -- without it a pool scores against the wrong bank outright
        transfer = Transfer.from_defocus(stack.defocus, plan, box_size)

        # `fold_in`, not a wider split: `split(key, 3)` returns three DIFFERENT keys and would
        # redraw every pose and image, moving every result already recorded at these seeds.
        if defocus_error:
            wrong_defocus = stack.defocus * (
                1.0 + defocus_error * jax.random.normal(jax.random.fold_in(key, 17), stack.defocus.shape)
            )
            scoring_defocus = wrong_defocus
        else:
            scoring_defocus = stack.defocus
        scoring_transfer = Transfer.from_defocus(scoring_defocus, plan, box_size)

        images = stack.images

        if stream:
            # deferred to one fused call below; the true defocus stays in `transfers`
            streamed_images.append(images)
            streamed_transfers.append(scoring_transfer)
        else:
            scored = pose_log_ratio(
                images,
                banks[0],
                banks[1],
                spectrum,
                amplitude=amplitude,
                pose_log_weights=grid.log_weights,
                transfer=scoring_transfer,
            )
            log_ratios.append(scored.marginal)
        # the TRUE transfer travels in the result: `snr` is the pool's, not the scorer's belief
        transfers.append(transfer)

        oracle_a, oracle_b = (model.bare_contrast(stack.angles) for model in (model_a, model_b))
        # `images`, not `stack.images`: the oracle must describe the pool actually scored, or the
        # coherence and scale-error checks read a pool that does not exist
        oracle_log_ratios.append(known_pose_log_ratio(images, oracle_a, oracle_b, spectrum, transfer=scoring_transfer))
        # the TRUE transfer: perturbing the target too would hide the drop this arm exists to see
        oracle_d2s.append(known_pose_d2(oracle_a, oracle_b, spectrum, transfer=transfer))

    if stream:
        fused = streamed_pose_log_ratio(
            jnp.concatenate(streamed_images),
            model_a.bare_contrast,
            model_b.bare_contrast,
            grid.angles,
            grid.log_weights,
            spectrum,
            amplitude=amplitude,
            transfer=_concatenate_transfers(streamed_transfers),
            pose_chunk=pose_chunk,
            expansion=expansion,
        )
        log_ratios = list(jnp.split(fused.marginal, 2))

    return PurePools(
        log_ratio_a=log_ratios[0],
        log_ratio_b=log_ratios[1],
        banks=None if banks is None else tuple(banks),
        spectrum=spectrum,
        pose_log_weights=grid.log_weights,
        transfer=transfers[0],
        # the same quantity on both paths, not an approximation: `whitened_power` is a mean over
        # poses and a mean composes over chunks, so the streamed `snr` matches the whole-bank one
        snr=(
            matched_filter_snr(banks[0], spectrum, transfer=transfers[0])
            if banks is not None
            else float(
                jnp.sqrt(
                    streamed_whitened_power(
                        model_a.bare_contrast,
                        grid.angles,
                        spectrum,
                        transfer=transfers[0],
                        expansion=expansion,
                    )
                )
            )
        ),
        oracle_log_ratio_a=oracle_log_ratios[0],
        oracle_log_ratio_b=oracle_log_ratios[1],
        oracle_d2_a=oracle_d2s[0],
        oracle_d2_b=oracle_d2s[1],
        amplitude=amplitude,
    )
