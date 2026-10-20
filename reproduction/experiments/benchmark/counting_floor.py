"""Where the counting floor comes from, and why reporting at p = 0.5 hides discriminability.
The headline count `N = 1/(I_p sigma_p^2)` saturates because `I_p <= 1/[p(1-p)]`; this decomposes the
floor on a pool, from the log-ratios `realdata/score_pool.py --log-ratio` saves.
    .venv/bin/python .../counting_floor.py --log-ratio-a a.npy --log-ratio-b b.npy --out out.json
For a two-state mixture at fraction `p`, `Var[Lambda] = v + p(1-p) Delta^2`, so the linear-axis
estimator gives `sigma_p^2 = [ p(1-p) + v/Delta^2 ] / N`: a BINOMIAL floor `p(1-p)` no measurement
beats, plus the DISCRIMINABILITY term `v/Delta^2`, the only one `d^2` enters. The floor's share falls
away from p = 0.5, so that convention is the worst case for seeing `d^2`. `sigma_p` is measured by
resampling real particles into mixed pools and reported against the analytic prediction.
"""

import argparse
from pathlib import Path

import numpy as np

from reproduction.record import write_result

_ERROR_TARGET = 0.02
_TRIALS = 4000

DRIVER = "reproduction/experiments/benchmark/counting_floor.py"


def axis(pure_a: np.ndarray, pure_b: np.ndarray) -> dict:
    """The two numbers the whole decomposition rests on, from two pure-state pools.

    Args:
        pure_a: Per-particle log-ratios for a pool known to be all state A.
        pure_b: The same for state B.

    Returns:
        `delta` (the separation of the two means), `within_variance` (pooled, unbiased), and
        `discriminability_term` = `v / delta^2`, which is the only term `d^2` enters.

    Raises:
        SystemExit: If the two means coincide, which makes the axis undefined rather than narrow. A
            twin pool does this by construction and would otherwise divide by ~0.
    """
    delta = float(pure_a.mean() - pure_b.mean())
    within = float(0.5 * (pure_a.var(ddof=1) + pure_b.var(ddof=1)))
    # compared against the separation's OWN sampling error (a twin's means differ by ~sqrt(within/n));
    # `<=` so two identical pools -- zero delta and zero spread -- are refused, not divided by zero
    standard_error = float(np.sqrt(pure_a.var(ddof=1) / pure_a.size + pure_b.var(ddof=1) / pure_b.size))
    if abs(delta) <= 3.0 * standard_error:
        raise SystemExit(
            f"the two pools' means differ by {delta:.3g}, within 3 standard errors ({standard_error:.3g}) "
            "of no difference at all, so there is no axis to project on."
        )
    return {"delta": delta, "within_variance": within, "discriminability_term": within / delta**2}


def predicted(fraction: float, n: int, discriminability_term: float) -> dict:
    """`sigma_p` and its two components at one operating point.

    Args:
        fraction: Population fraction `p`.
        n: Pool size.
        discriminability_term: `v / delta^2` from `axis`.

    Returns:
        `sigma_p`, the `floor` it would have with states known exactly, the `floor_share` of the
        variance, `information` `I_p`, and its `cap`.

        `I_p / cap` is `binomial / total`, which is `floor_share` -- so it is not returned under a
        second name. A record that carried both would let a reader compare two spellings of one
        number and believe the agreement said something.
    """
    binomial = fraction * (1.0 - fraction)
    total = binomial + discriminability_term
    return {
        "sigma_p": float(np.sqrt(total / n)),
        "floor": float(np.sqrt(binomial / n)),
        "floor_share": float(binomial / total),
        "information": float(1.0 / total),
        "cap": float(1.0 / binomial) if binomial > 0 else float("inf"),
    }


def resample(
    pure_a: np.ndarray,
    pure_b: np.ndarray,
    *,
    fraction: float,
    n: int,
    trials: int,
    seed: int,
) -> dict:
    """Measure `sigma_p` by building mixed pools out of the real particles.

    The pool's composition is itself drawn binomially rather than fixed at `round(p n)`, because the
    binomial term of the variance IS that fluctuation -- fixing the count would remove the very thing
    being measured and the result would sit at the discriminability term alone.

    Args:
        pure_a: Per-particle log-ratios from a pure state-A pool.
        pure_b: The same for B.
        fraction: True `p`.
        n: Pool size.
        trials: How many pools to draw.
        seed: RNG seed.

    Returns:
        `sigma_p` (the spread of the estimates), `bias` (mean estimate minus `fraction`), and `trials`.
    """
    rng = np.random.default_rng(seed)
    mu_b = float(pure_b.mean())
    delta = float(pure_a.mean()) - mu_b
    estimates = np.empty(trials)
    for index in range(trials):
        drawn = int(rng.binomial(n, fraction))
        pool = np.concatenate([rng.choice(pure_a, drawn, replace=True), rng.choice(pure_b, n - drawn, replace=True)])
        estimates[index] = (pool.mean() - mu_b) / delta
    return {
        "sigma_p": float(estimates.std(ddof=1)),
        "bias": float(estimates.mean() - fraction),
        "trials": trials,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line. Every path is an argument; nothing is read from a constant."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-ratio-a", type=Path, required=True, help="pure state-A per-particle log-ratios")
    parser.add_argument("--log-ratio-b", type=Path, required=True, help="pure state-B per-particle log-ratios")
    parser.add_argument("--out", type=Path, default=Path("results/counting_floor.json"))
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=(0.5, 0.4, 0.3, 0.2, 0.1, 0.05),
        help="population fractions to report. 0.5 is the panel convention and the worst case",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=(25, 50, 100, 200, 400, 800, 1600),
        help="pool sizes for the N sweep, run at the FIRST --fractions value",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pure_a = np.load(args.log_ratio_a).ravel()
    pure_b = np.load(args.log_ratio_b).ravel()
    geometry = axis(pure_a, pure_b)
    term = geometry["discriminability_term"]

    print(f"n = {pure_a.size} + {pure_b.size} real particles")
    print(f"Delta {geometry['delta']:+.3f}, within-state variance {geometry['within_variance']:.2f}")
    print(f"discriminability term v/Delta^2 = {term:.5f}; the floor term is p(1-p)\n")

    print("N sweep at p = " + f"{args.fractions[0]:g}")
    print(f"{'N':>6} {'measured':>10} {'predicted':>10} {'floor':>9} {'meas/pred':>10} {'bias':>10}")
    sweep = []
    for size in args.sizes:
        want = predicted(args.fractions[0], size, term)
        got = resample(pure_a, pure_b, fraction=args.fractions[0], n=size, trials=_TRIALS, seed=args.seed)
        ratio = got["sigma_p"] / want["sigma_p"]
        sweep.append({"n": size, "predicted": want, "measured": got, "ratio": ratio})
        print(
            f"{size:6d} {got['sigma_p']:10.5f} {want['sigma_p']:10.5f} {want['floor']:9.5f} "
            f"{ratio:10.4f} {got['bias']:+10.5f}"
        )

    print("\nfraction sweep at N = " + f"{args.sizes[len(args.sizes) // 2]}")
    print(f"{'p':>6} {'floor share':>12} {'N at target':>12} {'meas/pred':>10}")
    middle = args.sizes[len(args.sizes) // 2]
    fractions = []
    for fraction in args.fractions:
        want = predicted(fraction, middle, term)
        got = resample(pure_a, pure_b, fraction=fraction, n=middle, trials=_TRIALS, seed=args.seed + 1)
        count = 1.0 / (want["information"] * _ERROR_TARGET**2)
        ratio = got["sigma_p"] / want["sigma_p"]
        fractions.append({"fraction": fraction, "predicted": want, "measured": got, "n_at_target": count})
        print(f"{fraction:6.2f} {want['floor_share']:12.3f} {count:12,.0f} {ratio:10.4f}")

    worst = predicted(0.5, middle, term)["floor_share"]
    best = predicted(min(args.fractions), middle, term)["floor_share"]
    print(f"\nfloor share {worst:.1%} at p = 0.5 against {best:.1%} at p = {min(args.fractions):g}")

    write_result(
        args.out,
        {
            "axis": geometry,
            "n_particles": [int(pure_a.size), int(pure_b.size)],
            "size_sweep": sweep,
            "fraction_sweep": fractions,
            "floor_share_at_half": worst,
            "floor_share_at_min_fraction": best,
            "error_target": _ERROR_TARGET,
        },
        driver=DRIVER,
        arguments=vars(args),
    )
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
