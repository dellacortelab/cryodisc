"""Does `p_hat` track a `p_true` that is exact by construction, and does the axis behave?
Two pools believed PURE are mixed so `p_true = n_A/(n_A + n_B)` is exact, then recovered.
Target-agnostic: `--pair` simulates from two structures (laptop arm), `--log-ratio-a/-b` reads scored
pools from disk (real-data arm).
    .venv/bin/python .../known_fraction.py --pair a.pdb b.pdb --out results/kf.json
Coherence is reported first, the only readout that sees a scale error, but only at the ORACLE rung
(the identity has no closed form marginalized, so marginal coherence reads against a ceiling of one).
A pure pool that reads impure is the result, not a bug; `f_a`/`f_b` are measured, and `_derive_purity`
also predicts them from each pool's two moments. Rationale: docs/rationale/reproduction/experiments/benchmark-and-fetch.md
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cryodisc.grids.poses import realized_nodes
from cryodisc.specimen.structures import align_conformers
from cryodisc.statistics.convergence import converge_caching
from reproduction.analysis.fits import recovery_error
from reproduction.analysis.pure_pools import simulate_pure_pools
from reproduction.analysis.purity import (
    ESTIMATORS,
    Purity,
    fit_purity,
    fraction_estimator,
    holdout_split,
    sweep_mixtures,
)
from reproduction.analysis.recovery import gap_from_pools

# `research` is not installed -- only `src/` is -- so running this file the way its docstring says to
# puts `reproduction/experiments/benchmark/` on the path and not the repository root, and `research` would not import.
from reproduction.analysis.si_estimators import anchor_from_moments
from reproduction.operating_point import SHARED
from reproduction.record import write_result
from reproduction.targets import add_assembly_argument
from reproduction.validation.coherence import (
    Coherence,
    coherence,
    coherence_pair,
    predicted_coherence,
)

DRIVER = "reproduction/experiments/benchmark/known_fraction.py"


def simulate_pools(args: argparse.Namespace) -> dict:
    """Two pure pools from a structure pair.

    A thin adapter over `pure_pools.simulate_pure_pools`, which is where the drawing and scoring
    actually live, so any other driver needing the same pools imports that rather than this one;
    a driver importing a sibling is what `tests/test_experiments_layout.py` forbids -- two drivers drawing pools by
    slightly different routes would compare two forward models while appearing to compare two
    estimators.

    Args:
        args: Parsed arguments.

    Returns:
        The two log-ratio arrays, the banks, the spectrum, the SNR the pools ran at, and the oracle
        arm: the same images scored at their OWN poses, with the `d_i^2` those poses imply.
    """
    # keys are OMITTED, not passed as None: `align_conformers` uses an UNSET sentinel and refuses a
    # shared `chains` beside a per-entry one, so spelling all three as None would always raise
    selection = {
        name: value
        for name, value in (
            ("chains", args.chains),
            ("chains_a", args.chains_a),
            ("chains_b", args.chains_b),
            ("labels_a", None if args.labels_a is None else tuple(args.labels_a)),
            ("labels_b", None if args.labels_b is None else tuple(args.labels_b)),
        )
        if value is not None
    }
    pair = align_conformers(
        args.pair[0],
        args.pair[1],
        assembly=args.assembly,
        chain_id_kind=args.chain_id_kind,
        min_shared_fraction=args.min_shared_fraction,
        **selection,
    )

    def simulate(nodes: int):
        pools = simulate_pure_pools(
            pair,
            n_particles=args.n_particles,
            box_size=args.box_size,
            pixel_size=args.pixel_size,
            dose=args.dose,
            rmsf=args.rmsf,
            nodes=nodes,
            poses=args.poses,
            defocus_error=args.defocus_error,
            stream=args.stream,
            pose_chunk=args.pose_chunk,
            seed=args.seed,
        )
        return realized_nodes(nodes), pools

    # grow the pose bank until pool d^2 converges rather than fixing a node count; the converged bank's
    # pools carry the Lambda arrays every readout below (coherence, purity, sweep, oracle) consumes
    conv, pools = converge_caching(
        simulate,
        lambda p: gap_from_pools(np.asarray(p.log_ratio_a), np.asarray(p.log_ratio_b)),
        cap=args.cap_bank,
    )
    return {
        "log_ratio_a": pools.log_ratio_a,
        "log_ratio_b": pools.log_ratio_b,
        # None under --stream, where no whole bank was ever built. `load_pools` already returns
        # None here for the real-data arm.
        "banks": None if pools.banks is None else list(pools.banks),
        "spectrum": pools.spectrum,
        "pose_log_weights": pools.pose_log_weights,
        "transfer": pools.transfer,
        "amplitude": pools.amplitude,
        "snr": pools.snr,
        "oracle": {
            "log_ratio_a": pools.oracle_log_ratio_a,
            "log_ratio_b": pools.oracle_log_ratio_b,
            "d2_a": pools.oracle_d2_a,
            "d2_b": pools.oracle_d2_b,
        },
        "convergence": conv,
    }


def load_pools(args: argparse.Namespace) -> dict:
    """Two pools of measured log-ratios, read from disk.

    The real-data arm: the log-ratios are already scored, so no banks or simulation come with them.

    Args:
        args: Parsed arguments.

    Returns:
        The two log-ratio arrays, with no banks and no spectrum. No oracle either: a measured pool's
        pose is what a search had to find, so there is nothing to score at the true pose and the
        coherence readout on real data has no derived baseline. Recorded as absent rather than
        defaulted, a baseline of one being the number that made a heterogeneous pool look decohered.
    """
    return {
        "log_ratio_a": jnp.asarray(np.load(args.log_ratio_a)),
        "log_ratio_b": jnp.asarray(np.load(args.log_ratio_b)),
        "banks": None,
        "spectrum": None,
        "pose_log_weights": None,
        "transfer": None,
        "amplitude": 1.0,
        "snr": None,
        "oracle": None,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line. Every path is an argument; nothing is read from a constant."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pair", type=Path, nargs=2, metavar=("STATE_A", "STATE_B"), help="simulate two pure pools")
    source.add_argument("--log-ratio-a", type=Path, help="measured Lambda for the state-A pool, as .npy")
    parser.add_argument("--log-ratio-b", type=Path, help="measured Lambda for the state-B pool, required with -a")
    # default "1", matching `structures.align_conformers`: at None it can superpose non-corresponding
    # deposited copies and return a believable but wrong RMSD, measuring packing as conformation
    add_assembly_argument(parser)
    parser.add_argument(
        "--chains",
        nargs="+",
        default=None,
        metavar="CHAIN",
        help="author chains to keep, applied before the assembly is expanded. Needed wherever two "
        "entries split the same molecule differently -- panel.json records which pairs those are",
    )
    # SEPARATE per entry, because a shared `--chains` cannot express a pair whose two entries name the
    # corresponding chain differently -- scoring chain A of both can silently match the wrong copies
    parser.add_argument("--chains-a", nargs="+", default=None, metavar="CHAIN", help="author chains for state A alone")
    parser.add_argument("--chains-b", nargs="+", default=None, metavar="CHAIN", help="author chains for state B alone")
    parser.add_argument(
        "--chain-id-kind",
        default="auth_asym_id",
        choices=("auth_asym_id", "label_asym_id"),
        help="which chain identifier the --chains/--labels values name; the widened survey uses label_asym_id",
    )
    parser.add_argument(
        "--min-shared-fraction",
        type=float,
        default=0.75,
        help="overlap below which the pair is refused; lower it to admit a partial-correspondence "
        "pair, as the widened survey does",
    )
    # RENAMING makes two differently-named chains correspond: the match key is `(copy, chain,
    # residue_id, atom)`, so relabeling both entries onto one name is the only way to share atoms
    parser.add_argument(
        "--labels-a", nargs="+", default=None, metavar="CHAIN", help="rename state A's (copy, chain) pairs onto these"
    )
    parser.add_argument(
        "--labels-b", nargs="+", default=None, metavar="CHAIN", help="rename state B's (copy, chain) pairs onto these"
    )
    parser.add_argument("--out", type=Path, default=Path("results/known_fraction.json"))
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--pixel-size", type=float, default=SHARED.pixel_size)
    parser.add_argument("--box-size", type=int, default=64)
    parser.add_argument("--dose", type=float, default=SHARED.dose)
    parser.add_argument("--rmsf", type=float, default=SHARED.rmsf)
    parser.add_argument(
        "--defocus-error",
        type=float,
        default=0.0,
        help="per-particle RMS RELATIVE error on the defocus the SCORER is told, the images keeping "
        "the true one. 0 is the inverse crime and the default. A nonzero value models the mismatch "
        "between the defocus the scorer is told and the truth -- of order the spread between two "
        "independent CTF fits of the same micrographs; see docs/planning/marginalization_floor.md part 3",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="render and reduce the pose bank in chunks instead of materializing it, needed once the "
        "pose bank is too large to fit in memory. The answer is the same either way -- tests/"
        "test_rungs/test_streamed.py pins the two scorers against each other, and the reported snr "
        "composes exactly over the chunks -- so this is a memory knob only, off by default",
    )
    parser.add_argument(
        "--pose-chunk",
        type=int,
        default=None,
        help="poses per chunk under --stream; the scorer sizes it when this is unset. Affects peak "
        "memory only, never the number",
    )
    parser.add_argument("--n-particles", type=int, default=2000)
    parser.add_argument(
        "--cap-bank",
        type=int,
        default=1_000_000,
        help="--pair only: grow the pose bank to convergence up to this cap rather than fixing a node "
        "count. A low cap keeps a smoke test to one cheap bank",
    )
    parser.add_argument(
        "--poses",
        choices=("bank", "haar"),
        default="haar",
        help="draw particle poses on the quadrature nodes, or from the true measure. See --help notes",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--estimator",
        choices=ESTIMATORS,
        default="two_pool",
        help="fraction estimator for the purity sweep: 0 pools (biased/compressed when miscalibrated), "
        "1 pure pool's within-scale (A or B; they diverge when the states' scales differ), or both "
        "pools' densities (scale-consistent). Default two_pool",
    )
    parser.add_argument(
        "--holdout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="partition each pure pool into a reference half and a draw half so no particle mixed into "
        "the sweep is also an estimator reference (removes the train-on-test leak). --no-holdout uses "
        "the full pool as both, the old behavior",
    )
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=0,
        help="permutation-null draws: permute the A/B labels of the combined pool and re-run the same "
        "sweep, reading the recovered slope against the slope when there is no real A/B difference. 0 "
        "skips it. Tests separation vs chance, NOT batch source (identical accessions would need twin_floor)",
    )
    args = parser.parse_args(argv)
    if args.log_ratio_a is not None and args.log_ratio_b is None:
        parser.error("--log-ratio-a needs --log-ratio-b: a purity line takes two pools")
    return args


def _shuffle_null(log_ratio_a, log_ratio_b, fractions, *, estimator, holdout, n_shuffles, key) -> dict:
    """Permute the combined A/B labels and re-run the same sweep, giving the slope's null distribution.

    A recovered slope means something only against the slope when the two pools carry no real
    difference; this tests separation vs chance, not whether that difference is structural or batch.
    """
    combined = np.concatenate([np.asarray(log_ratio_a), np.asarray(log_ratio_b)])
    n_a = int(np.asarray(log_ratio_a).shape[0])
    slopes = []
    for subkey in jax.random.split(key, n_shuffles):
        perm_key, split_key, sweep_key = jax.random.split(subkey, 3)
        permuted = combined[jax.random.permutation(perm_key, combined.shape[0])]
        pseudo_a, pseudo_b = permuted[:n_a], permuted[n_a:]
        if holdout:
            draw_a, draw_b, ref_a, ref_b = holdout_split(pseudo_a, pseudo_b, split_key)
        else:
            draw_a, draw_b, ref_a, ref_b = pseudo_a, pseudo_b, pseudo_a, pseudo_b
        try:
            known, recovered = sweep_mixtures(
                draw_a, draw_b, fractions, key=sweep_key, estimator=fraction_estimator(estimator, ref_a, ref_b)
            )
            slopes.append(float(fit_purity(known, recovered).separation))
        except ValueError:
            slopes.append(0.0)  # coincident references: no fraction is identifiable -- the null's point
    slopes = np.asarray(slopes)
    return {
        "n_shuffles": int(n_shuffles),
        "mean": float(slopes.mean()),
        "sd": float(slopes.std()),
        "upper_95": float(np.percentile(slopes, 95)),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pools = simulate_pools(args) if args.pair else load_pools(args)
    log_ratio_a, log_ratio_b = pools["log_ratio_a"], pools["log_ratio_b"]

    read_a, read_b = coherence_pair(log_ratio_a, log_ratio_b)
    # the purity line reads p_hat off each mixed pool with the chosen estimator; with --holdout the
    # reference half is disjoint from the draw half, so no mixed particle is also one of its references
    if args.holdout:
        draw_a, draw_b, ref_a, ref_b = holdout_split(log_ratio_a, log_ratio_b, jax.random.key(args.seed + 2))
    else:
        draw_a, draw_b, ref_a, ref_b = log_ratio_a, log_ratio_b, log_ratio_a, log_ratio_b
    estimator_fn = fraction_estimator(args.estimator, ref_a, ref_b)
    known, recovered = sweep_mixtures(
        draw_a, draw_b, args.fractions, key=jax.random.key(args.seed + 1), estimator=estimator_fn
    )
    estimator = args.estimator
    purity = fit_purity(known, recovered)
    accuracy = recovery_error(known, recovered)
    null = (
        _shuffle_null(
            log_ratio_a,
            log_ratio_b,
            args.fractions,
            estimator=args.estimator,
            holdout=args.holdout,
            n_shuffles=args.n_shuffles,
            key=jax.random.key(args.seed + 3),
        )
        if args.n_shuffles > 0
        else None
    )

    payload: dict = {
        "estimator": estimator,
        "holdout": bool(args.holdout),
        "shuffle_null": null,
        "snr": pools["snr"],
        "n_particles": int(log_ratio_a.shape[0]),
        # MARGINAL: `baseline` is one because a pose-marginalized Lambda has no closed-form target, so
        # one is a ceiling and `decoherence` charges the pool's heterogeneity to the templates (see `oracle`)
        "coherence": {
            "marginalized": True,
            "baseline_is_a_ceiling": True,
            # variance recorded beside the mean, as the derived anchors consume it; `b`'s mean is
            # ORIENTED (positive when correct) unlike `score_pool`'s raw one, so `_derive_purity` mirrors it
            "a": {
                "value": read_a.value,
                "mean": read_a.mean,
                "variance": read_a.variance,
                "points_at_its_own_bank": read_a.points_at_its_own_bank,
            },
            "b": {
                "value": read_b.value,
                "mean": read_b.mean,
                "variance": read_b.variance,
                "points_at_its_own_bank": read_b.points_at_its_own_bank,
            },
        },
        "sweep": {"p_true": known, "p_hat": recovered},
        "purity": {
            "f_a": purity.f_a,
            "f_b": purity.f_b,
            "f_a_err": purity.f_a_err,
            "f_b_err": purity.f_b_err,
            "separation": purity.separation,
            "separation_err": purity.separation_err,
            "midpoint_err": purity.midpoint_err,
            "midpoint": purity.midpoint,
            "residual_rms": purity.residual_rms,
        },
        # the headline accuracy: RMS error off the diagonal in percentage points, with the slope and
        # intercept the compression and any offset show up in
        "recovery_error": accuracy,
        "derived": _derive_purity(read_a, read_b, purity),
    }

    if pools["oracle"] is not None:
        payload["oracle"] = _read_oracle(pools["oracle"])

    print(
        f"coherence   A {read_a.value:+.3f}   B {read_b.value:+.3f}   "
        "(MARGINAL: 1.000 is the ceiling, not the target -- see `oracle`)"
    )
    if "oracle" in payload:
        for state in ("a", "b"):
            read = payload["oracle"][state]
            if read["baseline"] is None:
                # the twin control, whose d^2 is identically zero. Printed as no target rather than
                # skipped: a readout absent from the output reads as a readout that passed
                print(f"  oracle {state.upper()}  {read['value']:+.3f}   no target: d^2 is zero, the pair being a twin")
                continue
            print(
                f"  oracle {state.upper()}  {read['value']:+.3f}   against a predicted "
                f"{read['baseline']:.3f} from the templates (decoherence {read['decoherence']:+.3f}, "
                f"scale {read['scale_error']:.3f}x)"
            )
            print(
                f"            d^2 closed form {read['d2_closed_form']:.4g}   pool {read['d2_pool']:.4g}   "
                f"({100 * read['d2_pool'] / read['d2_closed_form'] - 100:+.2f}%)"
            )
    print(f"purity      f_a {purity.f_a:+.3f} +- {purity.f_a_err:.3f}   f_b {purity.f_b:+.3f} +- {purity.f_b_err:.3f}")
    # with their error bars, as `f_a` is printed with its: 0.98 is a compression only if it clears it
    print(
        f"axis        separation {purity.separation:+.3f} +/- {purity.separation_err:.3f}   "
        f"midpoint {purity.midpoint:+.3f} +/- {purity.midpoint_err:.3f}"
    )
    print(f"linearity   residual rms {purity.residual_rms:.4f}")
    print(
        f"recovery    rms {accuracy['rms_pp']:.2f} pp off the diagonal   "
        f"slope {accuracy['slope']:+.3f}   intercept {accuracy['intercept']:+.3f}"
    )
    if null is not None:
        z = (purity.separation - null["mean"]) / null["sd"] if null["sd"] > 0 else float("inf")
        print(
            f"shuffle null slope {null['mean']:+.3f} +/- {null['sd']:.3f} (95% {null['upper_95']:+.3f}) "
            f"over {null['n_shuffles']} perms; observed {purity.separation:+.3f} = {z:.1f} sigma above chance"
        )
    derived = payload["derived"]
    if derived["f_b"] is None:
        print(f"derived     no anchors: {derived['unavailable']}")
    else:
        # printed beside the measurement: P4 of docs/planning/marginalization_floor.md is a claim
        # about the derived and measured anchors together
        print(
            f"derived     f_a {derived['f_a']:+.3f} vs measured {purity.f_a:+.3f} "
            f"({derived['f_a_error']:+.4f})   f_b {derived['f_b']:+.3f} vs measured {purity.f_b:+.3f} "
            f"({derived['f_b_error']:+.4f})   from two moments, no fitted parameter"
        )

    # the REALIZED node count of the converged bank (`pose_bank` quantizes the request), else a reader
    # must rebuild it; kept at TOP LEVEL because the read arm reports it (as None) with no convergence
    # dict of its own, so a reader finds it the same way whichever arm ran
    conv = pools.get("convergence")
    if conv is not None:
        payload["nodes_realized"] = max(conv.curve)
        payload["convergence"] = conv.as_record()
    else:
        payload["nodes_realized"] = None

    write_result(args.out, payload, driver=DRIVER, arguments=vars(args))
    print(f"\n-> {args.out}")

    # raised AFTER the write, so a refused (expensive) run still records every number motivating the
    # refusal rather than being lost
    for state, read in payload.get("oracle", {}).items():
        if read["indicates_scale_error"]:
            raise ValueError(
                f"pool {state.upper()} reads an oracle coherence of {read['value']:+.4g} against a "
                f"predicted {read['baseline']:.4g}. Heterogeneity can only push a correct pool BELOW "
                f"one, so a reading this far above it is a scale error and not a property of the "
                f"pool: Lambda is too small by about {read['scale_error']:.3g}x, and the pool's d^2 "
                f"of {read['d2_pool']:.4g} against the templates' {read['d2_closed_form']:.4g} is "
                f"the same statement in the units being quoted. Every rung below is affected."
            )
    return 0


def _derive_purity(read_a: Coherence, read_b: Coherence, purity: Purity) -> dict:
    """Where the two pure pools SHOULD land, from their own two moments and no fitted parameter.

    `statistics.composition.anchor_from_moments` solves the estimator's fixed point on a Gaussian
    `Lambda`, so the prediction consumes each pool's mean and variance and nothing else -- not the
    sweep, not `p_true`, not the pair. It is recorded because `docs/planning/marginalization_floor.md`
    P4 is a claim about the predicted and the measured anchor TOGETHER.

    Pool B is mirrored, and that is the trap: `coherence` reports pool B's mean already NEGATED so both
    pools read positive when correct, and `anchor(-mu, s^2) = 1 - anchor(mu, s^2)`. So the anchor of
    the stored moments is where pool B sits on its OWN axis while `f_b` is on the state-A axis, the two
    being one minus each other.

    Args:
        read_a: Pool A's coherence readout, carrying the mean and variance of its `Lambda`.
        read_b: The same for pool B, whose mean is oriented and so is mirrored here.
        purity: The MEASURED anchors, so the per-configuration error is recorded beside the pair
            rather than left for a reader to subtract.

    Returns:
        The two derived anchors, the moments they were derived from, and `predicted - measured` for
        each. Every field is None with an `unavailable` note where a pool's variance is not positive:
        a twin control's `Lambda` is identically zero, the derivation is undefined on it, and the run
        whose purpose is to measure zeros must record that rather than die on it.
    """
    if not (read_a.variance > 0.0 and read_b.variance > 0.0):
        return {
            "f_a": None,
            "f_b": None,
            "f_a_error": None,
            "f_b_error": None,
            "mean_a": read_a.mean,
            "variance_a": read_a.variance,
            "mean_b": read_b.mean,
            "variance_b": read_b.variance,
            "unavailable": (
                f"a pool's Lambda has no spread (variance {read_a.variance:.4g} and "
                f"{read_b.variance:.4g}), so the fixed point the anchors solve is undefined"
            ),
        }
    derived_f_a = anchor_from_moments(read_a.mean, read_a.variance)
    derived_f_b = 1.0 - anchor_from_moments(read_b.mean, read_b.variance)
    return {
        "f_a": derived_f_a,
        "f_b": derived_f_b,
        "f_a_error": derived_f_a - purity.f_a,
        "f_b_error": derived_f_b - purity.f_b,
        "mean_a": read_a.mean,
        "variance_a": read_a.variance,
        "mean_b": read_b.mean,
        "variance_b": read_b.variance,
    }


def _read_oracle(oracle: dict) -> dict:
    """Both pools' coherence at their own poses, which is the only readout here with a derived target.

    The identity `2 E[Lambda] / Var[Lambda] = 1` is exact for the known-pose log-ratio of a pool whose
    particles share a `d_i^2`, and `predicted_coherence` carries it to a pool whose do not. Neither holds
    for the marginalized Lambda the rest of this driver reads, so this is where a calibration error is
    convicted MOST DIRECTLY. Under the pooled estimators (the default) `purity` and `sweep` are
    invariant under a SCALE on Lambda; only `--estimator no_pool`, which reads Lambda as a calibrated
    log-likelihood ratio, is not, and there the linearity residual is a calibration diagnostic too.

    Rationale: docs/rationale/reproduction/experiments/benchmark-and-fetch.md

    Args:
        oracle: The `"oracle"` block of `simulate_pools`: per-pool known-pose log-ratios and the
            `d_i^2` the same templates imply.

    Returns:
        One entry per pool, each carrying the reading, its predicted baseline, and the `d^2` the pool
        read beside the `d^2` it should have -- `E[Lambda] = E[d^2] / 2` at this rung, so the two are
        the same claim and the second is computed from no images at all.
    """
    read_out: dict = {}
    for state, is_a in (("a", True), ("b", False)):
        d2 = oracle[f"d2_{state}"]
        # a TWIN has no baseline (not a baseline of one): its `d^2` is identically zero and
        # `predicted_coherence` refuses it, so the target is written ABSENT rather than dying on it
        has_baseline = float(jnp.mean(d2)) > 0.0
        baseline = predicted_coherence(d2) if has_baseline else None
        # `indicates_scale_error` is unaffected: it is documented as reading against one and not
        # against the baseline, precisely so it stays valid where the baseline is unavailable
        read = coherence(oracle[f"log_ratio_{state}"], state_a=is_a, baseline=baseline if has_baseline else 1.0)
        read_out[state] = {
            "value": read.value,
            "mean": read.mean,
            "baseline": baseline,
            "decoherence": read.decoherence if has_baseline else None,
            "scale_error": read.scale_error if has_baseline else None,
            "points_at_its_own_bank": read.points_at_its_own_bank,
            "indicates_scale_error": read.indicates_scale_error(),
            "d2_closed_form": float(jnp.mean(d2)),
            # the SPREAD is what makes the baseline differ from one, recorded beside it so a
            # heterogeneous pool (baseline below one) is distinguishable from a bad one
            "d2_spread": float(jnp.var(d2)),
            "d2_pool": 2.0 * abs(read.mean),
        }
    return read_out


if __name__ == "__main__":
    raise SystemExit(main())
