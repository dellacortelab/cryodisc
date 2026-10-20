"""Small-n sweep: the soft two-pool estimator vs Forman-adjusted classify-and-count.

At the pool sizes of the recovery figure the two tie, because both are consistent prior-adjustment
estimators. The adjusted count corrects classification's bias but still hard-labels every particle, so
it is unbiased yet not efficient; the soft estimator weighs each particle's evidence and reaches the
information bound. The gap that opens is in variance, so it should show as the pool shrinks and as the
states separate less. This sweeps particles-per-pool for several real systems and reports the RMS
recovery error of each estimator against pool size, with the per-pool d'^2 as the separation axis.
"""

import argparse

import jax
import numpy as np

from cryodisc.statistics.composition import (
    fraction_classifier_adjusted,
    fraction_two_pool,
)
from reproduction.analysis.purity import SWEEP_FRACTIONS, sweep_mixtures
from reproduction.record import write_result

DRIVER = "reproduction/experiments/benchmark/small_n_sweep.py"

POOLS = {
    "cas9": ("results/rerun_cas9_hoRepro_6nt.npy", "results/rerun_cas9_hoRepro_12nt.npy"),
    "slco2a1": ("results/ribo_pools/slco2a1/pool_a.npy", "results/ribo_pools/slco2a1/pool_b.npy"),
    "fanac": ("results/ribo_pools/fanac/pool_a.npy", "results/ribo_pools/fanac/pool_b.npy"),
}

SIZES = (64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000)
ESTIMATORS = ("2-pool", "adjusted")


def pool_d2(a: np.ndarray, b: np.ndarray) -> float:
    """Per-particle discriminability of the two pools: squared mean gap over pooled within-pool variance."""
    gap = float(a.mean() - b.mean())
    within = 0.5 * float(a.var() + b.var())
    return gap * gap / within


def rms_error(draw_a: np.ndarray, draw_b: np.ndarray, estimator, key) -> float:
    """RMS recovery error, in percentage points, over the A->B mixing sweep for one estimator."""
    known, recovered = sweep_mixtures(draw_a, draw_b, SWEEP_FRACTIONS, key=key, estimator=estimator)
    return 100.0 * float(np.sqrt(np.mean((np.asarray(recovered) - np.asarray(known)) ** 2)))


def sweep(pool_a: np.ndarray, pool_b: np.ndarray, sizes, repeats: int) -> dict:
    """RMS error per estimator at each pool size n, averaged over `repeats` disjoint subsamples.

    Each repeat draws n particles from each full pool, splits them into a reference half and a disjoint
    draw half, and mixes the draw halves at the known fractions. Reference and mixture both shrink with
    n, the realistic small-collection case.
    """
    a_all, b_all = np.asarray(pool_a, float), np.asarray(pool_b, float)
    out = {}
    for n in sizes:
        if n > min(len(a_all), len(b_all)):
            continue
        acc = {name: [] for name in ESTIMATORS}
        for seed in range(repeats):
            rng = np.random.default_rng(seed)
            a = a_all[rng.choice(len(a_all), n, replace=False)]
            b = b_all[rng.choice(len(b_all), n, replace=False)]
            ia, ib, half = rng.permutation(n), rng.permutation(n), n // 2
            ref_a, draw_a = a[ia[:half]], a[ia[half:]]
            ref_b, draw_b = b[ib[:half]], b[ib[half:]]
            key = jax.random.key(seed)
            funcs = {
                "2-pool": lambda m, ra=ref_a, rb=ref_b: fraction_two_pool(np.asarray(m), ra, rb),
                "adjusted": lambda m, ra=ref_a, rb=ref_b: fraction_classifier_adjusted(m, ra, rb),
            }
            for name, fn in funcs.items():
                acc[name].append(rms_error(draw_a, draw_b, fn, key))
        out[str(n)] = {
            name: {
                "mean": float(np.mean(v)),
                "sd": float(np.std(v)),
                "sem": float(np.std(v) / np.sqrt(len(v))),
            }
            for name, v in acc.items()
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", default=["cas9", "slco2a1", "fanac"], choices=list(POOLS))
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--record", default="results/small_n_sweep.json")
    args = parser.parse_args()

    pool_paths, per_system = {}, {}
    for system in args.systems:
        pa, pb = POOLS[system]
        pool_paths[f"{system}_a"], pool_paths[f"{system}_b"] = pa, pb
        a, b = np.load(pa).astype(float), np.load(pb).astype(float)
        per_system[system] = {
            "d2_pool": pool_d2(a, b),
            "n_full": int(min(len(a), len(b))),
            "by_size": sweep(a, b, args.sizes, args.repeats),
        }
        min(args.sizes[-1], per_system[system]["n_full"])
        print(f"{system}: d2_pool={per_system[system]['d2_pool']:.2f}, n_full={per_system[system]['n_full']}")

    payload = {
        "systems": per_system,
        "estimators": list(ESTIMATORS),
        "note": "RMS recovery error (percentage points) vs particles per pool; mean over --repeats disjoint subsamples.",
    }
    args_record = vars(args) | pool_paths
    out = write_result(args.record, payload, driver=DRIVER, arguments=args_record)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
