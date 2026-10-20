"""Test the particle calculator on real pools: its predicted error against the error actually reached.

    .venv/bin/python -m reproduction.experiments.benchmark.calculator_check --out results/calculator_check.json
The calculator sizes a collection from the Fisher information about the fraction, which the two-pool
maximum-likelihood estimate attains; so this measures that estimate, not the mean score. Every trial is
one simulated experiment: each pool is split afresh into disjoint reference and draw halves, as an
experiment's pure-state references are a finite sample and not its mixture, and a mixed pool of
binomial composition is drawn and fitted against those references. The measured error therefore
carries the references' own sampling error, which the prediction, the full pools' information, omits.
SpCas9 is swept over collection size at each fraction; every system is read as measured over predicted
at one size.
"""

from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import numpy as np

from cryodisc.statistics.composition import (
    fraction_from_reference,
    information_from_pools,
    two_pool_reference,
)
from cryodisc.statistics.information import deflection
from reproduction.record import write_result

DRIVER = "reproduction/experiments/benchmark/calculator_check.py"
SYSTEMS = ("msba", "fanac", "nkcc1", "slc19a3", "slco2a1", "ste2", "hk-atpase")
"""The seven real two-state systems the recovery figure reads."""


def split(pool: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A pool's disjoint reference and draw halves."""
    order = rng.permutation(pool.size)
    return pool[order[: pool.size // 2]], pool[order[pool.size // 2 :]]


def check(pool_a: np.ndarray, pool_b: np.ndarray, *, fraction: float, n: int, trials: int, seed: int) -> dict:
    """Predicted and measured standard error of the two-pool fraction at one operating point.

    Args:
        pool_a: Per-particle Lambda on the pure state-A pool.
        pool_b: The same for state B.
        fraction: True fraction of A in the drawn mixtures.
        n: Particles per mixture.
        trials: Simulated experiments, each with its own reference split.
        seed: RNG seed, for the splits and the draws.

    Returns:
        `predicted` and `measured` sigma_p, the counting `floor`, the measurement's own relative
        standard error, the `bias` and `trials`.
    """
    rng = np.random.default_rng(seed)
    information, _ = information_from_pools(pool_a, pool_b, fraction=fraction)
    estimates = np.empty(trials)
    for trial in range(trials):
        ref_a, draw_a = split(pool_a, rng)
        ref_b, draw_b = split(pool_b, rng)
        # the composition is drawn binomially: its fluctuation IS the counting term being tested
        drawn = rng.binomial(n, fraction)
        mixture = np.concatenate([rng.choice(draw_a, drawn), rng.choice(draw_b, n - drawn)])
        estimates[trial] = fraction_from_reference(mixture, two_pool_reference(ref_a, ref_b, span=mixture))
    measured = float(estimates.std(ddof=1))
    return {
        "predicted": float(1.0 / np.sqrt(n * information)),
        "measured": measured,
        "measured_rel_err": float(1.0 / np.sqrt(2.0 * (trials - 1))),
        "floor": float(np.sqrt(fraction * (1.0 - fraction) / n)),
        "bias": float(estimates.mean() - fraction),
        "trials": trials,
    }


def system_pools(root: Path, system: str) -> tuple[np.ndarray, np.ndarray]:
    """A system's two pure pools, in the order the recovery figure takes them."""
    pools = sorted(path for path in glob(str(root / system / "pool_*.npy")) if "_d2" not in path)
    if len(pools) != 2:
        raise SystemExit(f"{root / system} holds {len(pools)} pools, not two")
    return tuple(np.load(path).astype(float).ravel() for path in pools)


def main() -> int:
    """Run the SpCas9 size sweep and the seven-system check, and record both."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool-a", type=Path, default=Path("results/rerun_cas9_hoRepro_6nt.npy"))
    parser.add_argument("--pool-b", type=Path, default=Path("results/rerun_cas9_hoRepro_12nt.npy"))
    parser.add_argument("--pools-root", type=Path, default=Path("results/ribo_pools"))
    parser.add_argument("--fractions", type=float, nargs="+", default=(0.5, 0.05))
    parser.add_argument("--sizes", type=int, nargs="+", default=(25, 50, 100, 200, 400, 800, 1600, 3200, 6400))
    parser.add_argument("--system-size", type=int, default=800, help="particles collected for each system's check")
    parser.add_argument(
        "--cas9-size", type=int, default=20000, help="subsample each SpCas9 pool to this many particles for the sweep"
    )
    parser.add_argument(
        "--trials", type=int, default=2000, help="simulated experiments per point; 1.6% relative precision"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/calculator_check.json"))
    args = parser.parse_args()

    cas9_a, cas9_b = (np.load(path).astype(float).ravel() for path in (args.pool_a, args.pool_b))
    if args.cas9_size:
        sub = np.random.default_rng(args.seed)
        cas9_a, cas9_b = (sub.choice(pool, min(args.cas9_size, pool.size), replace=False) for pool in (cas9_a, cas9_b))
    sweep = {}
    for fraction in args.fractions:
        rows = []
        for index, n in enumerate(args.sizes):
            row = {"n": n, **check(cas9_a, cas9_b, fraction=fraction, n=n, trials=args.trials, seed=args.seed + index)}
            rows.append(row)
            print(
                f"SpCas9 p={fraction:g} N={n:5d}: measured {row['measured']:.5f} predicted {row['predicted']:.5f} "
                f"ratio {row['measured'] / row['predicted']:.3f}"
            )
        sweep[f"{fraction:g}"] = rows

    systems = []
    for system in SYSTEMS:
        pool_a, pool_b = system_pools(args.pools_root, system)
        entry = {"system": system, "d2_obs": float(deflection(pool_a, pool_b)), "by_fraction": {}}
        for fraction in args.fractions:
            row = check(pool_a, pool_b, fraction=fraction, n=args.system_size, trials=args.trials, seed=args.seed)
            row["ratio"] = row["measured"] / row["predicted"]
            entry["by_fraction"][f"{fraction:g}"] = row
            print(f"{system:10s} d'2 {entry['d2_obs']:6.2f} p={fraction:g}: measured/predicted {row['ratio']:.3f}")
        systems.append(entry)

    write_result(args.out, {"cas9_sweep": sweep, "systems": systems}, driver=DRIVER, arguments=vars(args))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
