"""Same-state (twin) null for the eight-system estimator ladder.

Splits each pure pool into two random halves, mixes them at known fractions, and recovers with the
two-pool held-out estimator. With no real state difference the recovered slope should be zero; its
spread across many random halvings is the same-state floor the two-state recovery must clear. The
single-halving fit error sees only the sweep resampling and understates that spread, so the floor is
quoted as the across-halving standard deviation. A random split tests self-reference only, not a
cross-session batch difference.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from reproduction.analysis.purity import fit_purity, fraction_estimator, holdout_split, sweep_mixtures
from reproduction.record import write_result

FRACTIONS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]


def _twin_slope(pool, seed):
    """One same-state recovery slope at halving `seed`."""
    idx = np.random.default_rng(seed).permutation(pool.shape[0])
    cut = len(idx) // 2
    h1 = jnp.asarray(pool[idx[:cut]])
    h2 = jnp.asarray(pool[idx[cut : 2 * cut]])
    draw_a, draw_b, ref_a, ref_b = holdout_split(h1, h2, jax.random.key(2 * seed + 2))
    estimator = fraction_estimator("two_pool", ref_a, ref_b)
    known, recovered = sweep_mixtures(draw_a, draw_b, FRACTIONS, key=jax.random.key(2 * seed + 1), estimator=estimator)
    return float(fit_purity(known, recovered).separation)


def main(argv=None):
    """Run the twin null over every system and write the floor record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools-root", type=Path, default=Path("results/ribo_pools"))
    parser.add_argument("--n-halvings", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("results/twin_null_ladder.json"))
    args = parser.parse_args(argv)

    systems, abs_means, sds, clearances = {}, [], [], []
    for system_dir in sorted(p for p in args.pools_root.iterdir() if p.is_dir()):
        pools = sorted(p for p in system_dir.glob("pool_*.npy") if "_d2" not in p.name)
        if len(pools) != 2:
            continue
        real_slope = json.loads((system_dir / "recovery.json").read_text())["result"]["purity"]["separation"]
        states, sds_here = {}, []
        for label, pool_path in zip(("state_a", "state_b"), pools):
            pool = np.load(pool_path).astype(np.float64)
            slopes = np.array([_twin_slope(pool, seed) for seed in range(args.n_halvings)])
            sd = float(slopes.std(ddof=1))
            states[label] = {
                "pool": pool_path.name,
                "n_particles": int(pool.shape[0]),
                "mean_slope": float(slopes.mean()),
                "sd_slope": sd,
                "se_slope": float(sd / np.sqrt(args.n_halvings)),
                "p5": float(np.percentile(slopes, 5)),
                "p95": float(np.percentile(slopes, 95)),
            }
            abs_means.append(abs(float(slopes.mean())))
            sds.append(sd)
            sds_here.append(sd)
        clearances.append(real_slope / max(sds_here))
        systems[system_dir.name] = {
            "real_two_pool_slope": real_slope,
            "clearance_over_floor": real_slope / max(sds_here),
            **states,
        }

    payload = {
        "n_halvings": args.n_halvings,
        "systems": systems,
        "max_abs_mean_slope": max(abs_means),
        "median_floor_sd": float(np.median(sds)),
        "min_clearance": min(clearances),
    }
    write_result(
        args.out, payload, driver="reproduction/experiments/benchmark/twin_null_ladder.py", arguments=vars(args)
    )
    print(
        f"wrote {args.out}: {len(systems)} systems, max|mean| {payload['max_abs_mean_slope']:.3f}, "
        f"median floor sd {payload['median_floor_sd']:.3f}, min clearance {payload['min_clearance']:.1f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
