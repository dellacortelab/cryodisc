"""Figure: population fractions recovered from real particles across systems, without classifying any.

    .venv/bin/python .../real_recovery.py --pool-a A.npy --pool-b B.npy --out docs/manuscript/figures/real_recovery
Pedagogical order, one hard pair before the breadth. (a) One hard pair worked up: SpCas9 with the guide
RNA displaced 6 or 12 bases, whose per-particle log-ratios overlap too heavily for any classifier, with
the best-threshold accuracy line. (b) The recovery ERROR against the true fraction, which magnifies the 0/2-pool effect: without a pure
pool the error is several times larger and does not vanish in the interior, a miscalibration bias no pool
is there to remove; two pure pools sit near zero. The
headline is the two-pool RMS of `p_hat - p_true` in percentage points, with the slope and intercept it
decomposes into. (c) The RMS recovery error across seven real two-state systems, falling as pure pools are
added -- large with no pool (the miscalibration bias), lowest with both; the same-state
(twin) null control from `results/twin_null_ladder.json` is reported in the body. Rationale: docs/rationale/figures.md
"""

import argparse
import json
from collections.abc import Iterable
from glob import glob
from pathlib import Path

import jax
import numpy as np

from cryodisc.statistics.composition import (
    fraction_classifier,
    fraction_classifier_adjusted,
    fraction_no_pool,
    fraction_two_pool,
)
from reproduction.analysis.fits import recovery_error
from reproduction.analysis.purity import SWEEP_FRACTIONS, fit_purity, sweep_mixtures, sweep_mixtures_repeated
from reproduction.record import write_result
from reproduction.reporting.figures import (
    DOUBLE_COLUMN,
    OTHER,
    PAD,
    PANEL_GAP,
    RULE,
    SERIES,
    SINGLE_COLUMN,
    TICK_PT,
    label_panels,
    save,
    use_house_style,
)
from reproduction.reporting.numbers import Ledger

DRIVER = "reproduction/experiments/figures/real_recovery.py"

STATE_A = "6 nt R-loop"
STATE_B = "12 nt R-loop"
"""What the two SpCas9 pools are, in the specimen's own words rather than as `A` and `B`."""

SYSTEMS = ("msba", "fanac", "nkcc1", "slc19a3", "slco2a1", "ste2", "hk-atpase")
"""The seven real two-state systems the breadth panel reads, in display order."""

LADDER_REPEATS = 20
"""Held-out splits averaged per system; the spread across them is the small-n uncertainty."""


def best_accuracy(pool_a: np.ndarray, pool_b: np.ndarray) -> tuple[float, float]:
    """The best single-particle assignment any threshold classifier could reach.

    Args:
        pool_a: Per-particle log-ratios from the pool believed pure in state A.
        pool_b: The same from the state-B pool.

    Returns:
        `(accuracy, threshold)`, the accuracy balanced over the two pools.
    """
    grid = np.linspace(min(pool_a.min(), pool_b.min()), max(pool_a.max(), pool_b.max()), 2001)
    scores = 0.5 * ((pool_a[None, :] > grid[:, None]).mean(1) + (pool_b[None, :] <= grid[:, None]).mean(1))
    best = int(np.argmax(scores))
    return float(scores[best]), float(grid[best])


def _held_out(pool_a: np.ndarray, pool_b: np.ndarray):
    """Split both pools into a reference half and a disjoint test half, at a fixed seed."""
    rng = np.random.default_rng(0)
    a_all, b_all = np.asarray(pool_a, float), np.asarray(pool_b, float)
    ia, ib = rng.permutation(len(a_all)), rng.permutation(len(b_all))
    a_ref, a_test = a_all[ia[: len(a_all) // 2]], a_all[ia[len(a_all) // 2 :]]
    b_ref, b_test = b_all[ib[: len(b_all) // 2]], b_all[ib[len(b_all) // 2 :]]
    return a_ref, a_test, b_ref, b_test


def recover(pool_a: np.ndarray, pool_b: np.ndarray, *, repeats: int = 200) -> dict:
    """Recover the SpCas9 fraction across the sweep with each estimator, with error bars on the two-pool.

    A held-out split builds each reference density from one half of a pool and mixes the disjoint half,
    so no particle is scored against a component it helped form (an in-sample fit reads far too
    optimistic here). The two-pool curve is the headline; the no-pool curve shows what dropping the
    references costs.

    Args:
        pool_a: Per-particle log-ratios, state-A pool.
        pool_b: Per-particle log-ratios, state-B pool.
        repeats: Independent resampled sweeps behind the two-pool error bar.

    Returns:
        `{p_true, two, none, classifier, adjusted, spread, f_a, f_b, separation, error, ...}` -- the
        recovery curves, the two-pool bootstrap spread, the fitted anchors, and a `recovery_error`
        accuracy dict for the two-pool, naive-classify, and adjusted-classify estimators.
    """
    a_ref, a_test, b_ref, b_test = _held_out(pool_a, pool_b)
    estimators = {
        "none": lambda mixture: fraction_no_pool(mixture),
        "two": lambda mixture: fraction_two_pool(np.asarray(mixture), a_ref, b_ref),
        "classifier": lambda mixture: fraction_classifier(np.asarray(mixture), a_ref, b_ref),
        "adjusted": lambda mixture: fraction_classifier_adjusted(mixture, a_ref, b_ref),
    }
    curves, spread = {}, None
    for name, estimator in estimators.items():
        reps = repeats if name == "two" else max(2, repeats // 4)
        p_true, p_hat, p_sd = sweep_mixtures_repeated(
            a_test, b_test, SWEEP_FRACTIONS, key=jax.random.key(0), repeats=reps, estimator=estimator
        )
        curves[name] = np.asarray(p_hat, float)
        if name == "two":
            spread = np.asarray(p_sd, float)
    p_true = np.asarray(SWEEP_FRACTIONS, float)
    purity = fit_purity(p_true, curves["two"])
    return {
        "p_true": p_true,
        "two": curves["two"],
        "none": curves["none"],
        "classifier": curves["classifier"],
        "adjusted": curves["adjusted"],
        "spread": spread,
        "f_a": purity.f_a,
        "f_b": purity.f_b,
        "separation": purity.separation,
        "error": recovery_error(p_true, curves["two"]),
        "classify_error": recovery_error(p_true, curves["classifier"]),
        "adjusted_error": recovery_error(p_true, curves["adjusted"]),
    }


def _rms(draw_a: np.ndarray, draw_b: np.ndarray, estimator) -> float:
    """The RMS recovery error, in percentage points, over the A->B mixing sweep for a given estimator."""
    known, recovered = sweep_mixtures(draw_a, draw_b, SWEEP_FRACTIONS, key=jax.random.key(0), estimator=estimator)
    return 100.0 * float(np.sqrt(np.mean((np.asarray(recovered) - np.asarray(known)) ** 2)))


def _halves(pool: np.ndarray, key) -> tuple[np.ndarray, np.ndarray]:
    """Split a pool into a draw half (mixed into the sweep) and a disjoint reference half."""
    index = np.asarray(jax.random.permutation(key, pool.shape[0]))
    half = pool.shape[0] // 2
    return pool[index[:half]], pool[index[half:]]


def ladder(pool_a: np.ndarray, pool_b: np.ndarray, *, repeats: int = LADDER_REPEATS) -> dict[str, float]:
    """Mean RMS error for naive and adjusted classify-and-count, no pool, and both pure pools, over splits."""
    zero, two, classify, adjusted = [], [], [], []
    for seed in range(repeats):
        key_a, key_b = jax.random.split(jax.random.key(seed))
        draw_a, ref_a = _halves(pool_a, key_a)
        draw_b, ref_b = _halves(pool_b, key_b)
        zero.append(_rms(draw_a, draw_b, fraction_no_pool))
        two.append(_rms(draw_a, draw_b, lambda mix, a=ref_a, b=ref_b: fraction_two_pool(np.asarray(mix), a, b)))
        classify.append(_rms(draw_a, draw_b, lambda mix, a=ref_a, b=ref_b: fraction_classifier(np.asarray(mix), a, b)))
        adjusted.append(_rms(draw_a, draw_b, lambda mix, a=ref_a, b=ref_b: fraction_classifier_adjusted(mix, a, b)))
    return {
        "0-pool": float(np.mean(zero)),
        "2-pool": float(np.mean(two)),
        "classify": float(np.mean(classify)),
        "adjusted": float(np.mean(adjusted)),
    }


def read_systems(root: Path) -> dict[str, dict[str, float]]:
    """The no-pool and two-pool recovered slope for every pair under `root`, in display order."""
    out = {}
    for system in [s for s in SYSTEMS if (root / s).is_dir()]:
        pools = sorted(p for p in glob(str(root / system / "pool_*.npy")) if "_d2" not in p)
        if len(pools) < 2:
            continue
        out[system] = ladder(np.load(pools[0]).astype(float), np.load(pools[1]).astype(float))
    return out


def twin_null_floor(path: Path) -> dict[str, float]:
    """The same-state (twin) null floor from `results/twin_null_ladder.json` -- its aggregate fields."""
    record = json.loads(path.read_text())["result"]
    return {
        "floor_sd": float(record["median_floor_sd"]),
        "max_abs_mean": float(record["max_abs_mean_slope"]),
        "min_clearance": float(record["min_clearance"]),
    }


def draw(systems: dict[str, dict[str, float]], pool_a, pool_b, real: dict, *, threshold: float):
    """The three panels, pedagogical order: one hard pair's scores, its recovery, then seven-system breadth."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure, (scores, curve, breadth) = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, SINGLE_COLUMN * 0.82))

    # ---- (a) one hard pair: overlapping per-particle scores, and the best a threshold could do ----
    reach = float(np.abs(np.percentile(np.concatenate([pool_a, pool_b]), [0.4, 99.6])).max())
    span = (-reach, reach)
    edges = np.linspace(span[0], span[1], 71)
    for pool, color in ((pool_b, SERIES[1]), (pool_a, SERIES[0])):
        scores.hist(pool, bins=edges, color=color, alpha=0.45, lw=0.0)
        scores.hist(pool, bins=edges, histtype="step", color=color, lw=0.9)
    scores.axvline(threshold, color=RULE, lw=0.8, ls=(0, (3, 2)), zorder=1)
    scores.set_xlabel("Log-likelihood ratio")
    scores.set_ylabel("Particles")
    scores.set_xlim(*span)
    scores.set_ylim(0, scores.get_ylim()[1] * 1.16)
    scores.annotate(
        "best\nthreshold",
        xy=(threshold, scores.get_ylim()[1]),
        xytext=(3, -1),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=TICK_PT,
        color=RULE,
        linespacing=0.95,
    )
    legend_order = [
        Patch(facecolor=SERIES[0], alpha=0.55, label=STATE_A),
        Patch(facecolor=SERIES[1], alpha=0.55, label=STATE_B),
    ]
    scores.legend(
        handles=legend_order,
        loc="upper left",
        handlelength=0.9,
        handletextpad=0.5,
        borderpad=0.2,
        labelspacing=0.25,
        fontsize=TICK_PT,
    )

    # ---- (b) the recovery ERROR, which magnifies what dropping the pure references costs ----
    pt = real["p_true"]
    curve.axhline(0.0, color=RULE, lw=0.9, ls=(0, (3, 2)), zorder=1, label="Exact recovery")
    curve.plot(
        pt,
        100.0 * (real["classifier"] - pt),
        color=SERIES[2],
        lw=1.1,
        ls=(0, (4, 2)),
        marker="^",
        ms=3.2,
        mew=0.0,
        zorder=2,
        label="Classify & count",
    )
    curve.plot(
        pt,
        100.0 * (real["adjusted"] - pt),
        color=SERIES[1],
        lw=1.1,
        ls=(0, (4, 2)),
        marker="D",
        ms=3.0,
        mew=0.0,
        zorder=4,
        label="Adjusted classify",
    )
    curve.plot(
        pt,
        100.0 * (real["none"] - pt),
        color=OTHER,
        lw=1.1,
        ls=(0, (1.5, 1.5)),
        marker="s",
        ms=3.0,
        mew=0.0,
        zorder=3,
        label="No pool",
    )
    curve.plot(
        pt,
        100.0 * (real["two"] - pt),
        color=SERIES[0],
        ls="none",
        marker="o",
        ms=4.2,
        mew=0.0,
        zorder=5,
        label="Two pools",
    )
    curve.set_xlabel(f"{STATE_A} fraction")
    curve.set_ylabel("Recovery error (pp)")
    curve.set_xlim(-0.05, 1.06)
    curve.set_xticks([0, 0.5, 1.0], ["0", "50%", "100%"])
    curve.set_ylim(top=curve.get_ylim()[1] * 1.22)  # headroom so the legend clears the no-pool series
    curve.legend(loc="upper right", handlelength=1.4, borderpad=0.2, labelspacing=0.3, fontsize=TICK_PT)

    # ---- (c) seven-system breadth: RMS drops from naive classify-and-count through the better estimators ----
    x = np.array([0.0, 1.0, 2.0, 3.0])
    for rungs in systems.values():
        breadth.plot(
            x,
            [rungs["classify"], rungs["adjusted"], rungs["0-pool"], rungs["2-pool"]],
            color=OTHER,
            lw=0.9,
            marker="o",
            ms=3.0,
            mew=0.0,
            alpha=0.7,
            zorder=3,
        )
    breadth.set_xticks(x, ["classify\n& count", "adj.\nclassify", "no\npool", "two\npool"])
    breadth.set_xlim(-0.25, 3.25)
    breadth.set_yscale("log")
    breadth.set_yticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    breadth.minorticks_off()
    breadth.set_ylabel("RMS recovery error (pp)")

    for panel in (scores, curve, breadth):
        panel.set_box_aspect(1.0)  # square boxes regardless of each panel's own data range
    figure.tight_layout(pad=PAD, w_pad=PANEL_GAP)
    label_panels(scores, curve, breadth)
    return figure


def _record_payload(systems: dict, real: dict, accuracy: float, threshold: float) -> dict:
    """The adjusted-classifier backing record: the panel-c ladder and the SpCas9 panel-b curves.

    The reviewer-requested adjusted classify-and-count (Forman 2008) alongside the estimators already
    plotted, so its per-system RMS and SpCas9 recovery curve are traceable to the run that drew them.
    """
    return {
        "ladder_repeats": LADDER_REPEATS,
        "ladder": {system: dict(rungs) for system, rungs in systems.items()},
        "cas9_panel_b": {
            "p_true": [float(v) for v in real["p_true"]],
            "two_pool": [float(v) for v in real["two"]],
            "no_pool": [float(v) for v in real["none"]],
            "classify": [float(v) for v in real["classifier"]],
            "adjusted": [float(v) for v in real["adjusted"]],
            "two_pool_error": real["error"],
            "classify_error": real["classify_error"],
            "adjusted_error": real["adjusted_error"],
            "best_accuracy": float(accuracy),
            "threshold": float(threshold),
        },
    }


def main(argv: Iterable[str] | None = None) -> int:
    """Draw the merged recovery figure and print every number a caption quotes."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool-a", type=Path, default=Path("results/rerun_cas9_hoRepro_6nt.npy"))
    parser.add_argument("--pool-b", type=Path, default=Path("results/rerun_cas9_hoRepro_12nt.npy"))
    parser.add_argument("--pools-root", type=Path, default=Path("results/ribo_pools"))
    parser.add_argument("--twin-null", type=Path, default=Path("results/twin_null_ladder.json"))
    parser.add_argument("--record", type=Path, default=Path("results/real_recovery_adjusted.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/manuscript/figures/real_recovery"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    use_house_style()
    val = Ledger()
    systems = read_systems(args.pools_root)
    floor = twin_null_floor(args.twin_null)
    pool_a, pool_b = np.load(args.pool_a), np.load(args.pool_b)
    accuracy_value, threshold = best_accuracy(pool_a, pool_b)
    real = recover(pool_a, pool_b)
    figure = draw(systems, pool_a, pool_b, real, threshold=threshold)
    written = save(figure, args.out)

    zero = [systems[s]["0-pool"] for s in systems]
    two = [systems[s]["2-pool"] for s in systems]
    classify = [systems[s]["classify"] for s in systems]
    adjusted = [systems[s]["adjusted"] for s in systems]
    print(f"systems         {len(systems)} pairs ({LADDER_REPEATS} held-out splits each): {', '.join(systems)}")
    print(
        f"classify (all)  rms {val('ladder_classify_lo', min(classify), '.1f')}-{val('ladder_classify_hi', max(classify), '.1f')} pts (naive counting)"
    )
    print(
        f"adjusted (all)  rms {val('ladder_adjusted_lo', min(adjusted), '.1f')}-{val('ladder_adjusted_hi', max(adjusted), '.1f')} pts (Forman-adjusted counting)"
    )
    print(
        f"0-pool (all)    rms {val('ladder_zero_lo', min(zero), '.1f')}-{val('ladder_zero_hi', max(zero), '.1f')} pts (miscalibrated)"
    )
    print(
        f"2-pool (all)    rms {val('ladder_two_lo', min(two), '.1f')}-{val('ladder_two_hi', max(two), '.1f')} pts (best)"
    )
    print(
        f"twin null       floor sd {val('twin_null_floor_sd', floor['floor_sd'], '.2f')}   "
        f"max |mean| {val('twin_null_max_slope', floor['max_abs_mean'], '.3f')}   "
        f"min clearance {val('twin_null_min_clearance', floor['min_clearance'], '.1f')}x"
    )

    pooled = np.sqrt(0.5 * (pool_a.var() + pool_b.var()))
    err = real["error"]
    print(f"pure pools      A {pool_a.mean():+.4f} +/- {pool_a.std():.4f} (n={pool_a.size:,})")
    print(f"                B {pool_b.mean():+.4f} +/- {pool_b.std():.4f} (n={pool_b.size:,})")
    print(f"per particle    {(pool_a.mean() - pool_b.mean()) / pooled:.3f} sigma apart")
    print(
        f"                best single-particle accuracy {val('cas9_accuracy', 100 * accuracy_value, '.1f')}% at threshold {threshold:+.3f}"
    )
    print(f"reference MLE   f_a {real['f_a']:+.4f}  f_b {real['f_b']:+.4f}  (two pure pools)")
    print(
        f"recovery        error {val('cas9_recovery_rms_pp', err['rms_pp'], '.1f')} pp   "
        f"slope {val('cas9_recovery_slope', err['slope'], '.3f')}   "
        f"intercept {val('cas9_recovery_intercept', err['intercept'], '.3f')}"
    )
    classify_err = real["classify_error"]
    print(
        f"classify&count  error {val('cas9_classify_rms_pp', classify_err['rms_pp'], '.1f')} pp   "
        f"slope {val('cas9_classify_slope', classify_err['slope'], '.3f')}"
    )
    adjusted_err = real["adjusted_error"]
    print(
        f"adjusted class  error {val('cas9_adjusted_rms_pp', adjusted_err['rms_pp'], '.2f')} pp   "
        f"slope {val('cas9_adjusted_slope', adjusted_err['slope'], '.3f')}   "
        f"intercept {val('cas9_adjusted_intercept', adjusted_err['intercept'], '.3f')}"
    )
    write_result(
        args.record, _record_payload(systems, real, accuracy_value, threshold), driver=DRIVER, arguments=vars(args)
    )
    print(f"  wrote {args.record}")
    val.dump(args.out.parent / "values" / f"{args.out.name}.json")
    print("  wrote " + ", ".join(str(path) for path in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
