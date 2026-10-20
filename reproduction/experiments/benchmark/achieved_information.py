"""Does a real pipeline's posterior carry the information the physics predicts? The central claim.
Measures achieved information `I_p_ach = mean((r - p)^2) / (p(1-p))^2` against the a-priori `I_p` from
`d^2`, on REAL particles from two pools believed pure, mixed at a fraction exact by construction; their
ratio is the retained fraction and nothing is fitted to make them agree.
    .venv/bin/python .../achieved_information.py --log-ratio-a a.npy --log-ratio-b b.npy --d2 D --out out.json
A retention needs an a-priori `--d2`; without it `d^2` comes from the pools' own gap, so `retained` is
null and the run is a `d^2` diagnostic only. The responsibilities `r_i = sigma(Lambda_i + logit(p_hat))`
use the soft-EM fixed point (the pipeline's posterior); `--at-true-fraction` scores at the known
fraction as an upper bound. Each rung mixes its own `n_a`/`n_b`, so `p_true` is exact.
"""

import argparse
from collections.abc import Iterable
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from cryodisc.rungs import achieved_information
from cryodisc.statistics.composition import fraction_no_pool
from cryodisc.statistics.information import information_ceiling
from cryodisc.statistics.power import information_per_particle
from reproduction.record import write_result

DRIVER = "reproduction/experiments/benchmark/achieved_information.py"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-ratio-a", type=Path, required=True, help="Lambda on a pool pure in state A")
    parser.add_argument("--log-ratio-b", type=Path, required=True, help="Lambda on a pool pure in state B")
    parser.add_argument(
        "--d2",
        type=float,
        default=None,
        help="the pair's a-priori d^2, for the PREDICTED I_p. REQUIRED for a retention: omit it and "
        "d^2 is taken from the pools themselves as mean(Lambda|A) - mean(Lambda|B), the scorer's own "
        "output on both sides of the ratio, so `retained` is written null and the run reports only "
        "the d^2 readings",
    )
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.2, 0.35, 0.5, 0.65, 0.8])
    parser.add_argument("--n-particles", type=int, default=1200, help="particles per mixture")
    parser.add_argument(
        "--at-true-fraction",
        action="store_true",
        help="build the responsibilities at the KNOWN fraction rather than the estimator's own fixed "
        "point. An upper bound, not the pipeline's number: it hands the estimator the answer",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser())
    return args


def responsibilities(lam: np.ndarray, prior: float) -> np.ndarray:
    """`sigma(Lambda + logit(p))`, the posterior that each particle is state A."""
    return 1.0 / (1.0 + np.exp(-(lam + np.log(prior / (1.0 - prior)))))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    lam_a = np.load(args.log_ratio_a).astype(float)
    lam_b = np.load(args.log_ratio_b).astype(float)
    measured_d2 = float(lam_a.mean() - lam_b.mean())
    # NO --d2 MEANS NO RETENTION: with d^2 read off the pools, `predicted` and `achieved` are both
    # functions of the same Lambda, so their ratio measures nothing about the physics.
    self_referential = args.d2 is None
    d2 = measured_d2 if self_referential else args.d2
    source = (
        "measured from the pools -- SELF-REFERENTIAL, no retention is reported"
        if self_referential
        else "a priori, from the templates"
    )
    compression = None if self_referential or not measured_d2 else args.d2 / measured_d2
    # (B9) caps `I_p` at `1/(p(1-p))`; where it saturates, cross-rung gaps must be quoted in d^2, not
    # I_p. The pools' mean gap over the a-priori d^2 is the (B7) currency, equal to 1/compression.
    mean_gap_retention = None if compression is None else measured_d2 / args.d2

    print(f"pools     A {len(lam_a)}, B {len(lam_b)} particles")
    print(f"d^2       {d2:.4g}   ({source}; the pools themselves read {measured_d2:.4g})")
    if self_referential:
        print("          the I_p below is built from that same measured gap, so I_p_ach / I_p is the")
        print("          scorer divided by itself: it is NOT a retained fraction and is not written")
    else:
        print(f"compression  a-priori d^2 / the pools' gap = {compression:.4g}x")
        print(f"mean gap  d^2 retained = {mean_gap_retention:.4%}  (the (B7) currency; the I_p ratio below is capped)")
    print(
        f"posterior at {'the KNOWN fraction (upper bound)' if args.at_true_fraction else 'the estimator fixed point'}\n"
    )

    rng = np.random.default_rng(args.seed)
    print(f"{'p_true':>8} {'n':>6} {'p_hat':>8} {'I_p pred':>10} {'I_p ach':>9} {'retained':>9} {'+bias':>9}")
    rungs = []
    for fraction in args.fractions:
        n_a = round(fraction * args.n_particles)
        n_b = args.n_particles - n_a
        if n_a > len(lam_a) or n_b > len(lam_b):
            print(f"{fraction:>8.3f}  needs {n_a}/{n_b} from pools of {len(lam_a)}/{len(lam_b)}, skipped")
            continue
        mixed = np.concatenate([rng.choice(lam_a, size=n_a, replace=False), rng.choice(lam_b, size=n_b, replace=False)])
        p_true = n_a / (n_a + n_b)
        p_hat = fraction_no_pool(mixed)
        r = responsibilities(mixed, p_true if args.at_true_fraction else p_hat)
        # AT THE FIXED POINT, not at p_true: `mean((r-p)^2)/(p(1-p))^2` is an information only when `r`
        # is unbiased for `p`; scored at the truth it absorbs a squared bias that can dominate here
        achieved = float(achieved_information(jnp.asarray(r)))
        at_truth = float(achieved_information(jnp.asarray(r), fraction=p_true))
        predicted = float(information_per_particle(jnp.asarray(d2), fraction=p_true))
        rungs.append(
            {
                "p_true": p_true,
                "n": len(mixed),
                "p_hat": p_hat,
                "predicted": predicted,
                # (B9): shows whether the DENOMINATOR is at the cap, which must be read before any
                # ratio against it -- at the cap the retention is a distance to perfect labels.
                "predicted_over_cap": predicted / float(information_ceiling(p_true)),
                "achieved": achieved,
                # kept for the record and NOT compared to `predicted`: it mixes variance with the
                # squared bias of a compressed axis
                "achieved_at_truth_includes_bias": at_truth,
            }
        )
        share = "      n/a" if self_referential else f"{achieved / predicted if predicted else float('nan'):>8.1%} "
        print(
            f"{p_true:>8.3f} {len(mixed):>6d} {p_hat:>8.4f} {predicted:>10.4f} {achieved:>9.4f} {share}{at_truth:>9.4f}"
        )

    if rungs:
        ach = float(np.mean([r["achieved"] for r in rungs]))
        pred = float(np.mean([r["predicted"] for r in rungs]))
        print(f"\nmean over {len(rungs)} mixtures: I_p_ach = {ach:.4f} against I_p = {pred:.4f}")
        if self_referential:
            print(f"  retained  NOT REPORTED: I_p came from the pools' own gap ({measured_d2:.4g}), so")
            print(
                f"            the {ach / pred if pred else float('nan'):.1%} that ratio would read is the scorer against itself,"
            )
            print("            not a fraction of the available information. Pass --d2 for a retention.")
        else:
            cap = float(np.mean([r["predicted"] / r["predicted_over_cap"] for r in rungs]))
            print(f"  retained {ach / pred:.1%} of the information the physics makes available")
            print(f"    I_p / I_p_cap = {pred / cap:.4f} for the DENOMINATOR: at the cap this reads as a")
            print(f"    distance to perfect labels, and the d^2 currency of (B7) reads {mean_gap_retention:.4%}")
        slope = (
            np.polyfit([r["p_true"] for r in rungs], [r["p_hat"] for r in rungs], 1)[0]
            if len(rungs) > 1
            else float("nan")
        )
        print(f"  p_hat against p_true, slope {slope:.4f}  (1 is an uncompressed axis)")
        write_result(
            args.out,
            {
                "d2": d2,
                "d2_source": source,
                "d2_a_priori": args.d2,
                "d2_measured_from_pools": measured_d2,
                "d2_self_referential": self_referential,
                "compression_a_priori_over_measured": compression,
                "mean_gap_retention": mean_gap_retention,
                "at_true_fraction": args.at_true_fraction,
                "rungs": rungs,
                "achieved_mean": ach,
                "predicted_mean": pred,
                "retained": None if self_referential or not pred else ach / pred,
                "slope": float(slope),
            },
            driver=DRIVER,
            arguments=vars(args),
        )
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
