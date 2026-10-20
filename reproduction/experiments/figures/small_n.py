"""Supplementary figure: the soft two-pool estimator vs Forman-adjusted classify-and-count at small n.

Reads `results/small_n_sweep.json` (the resampling sweep on the already-scored pools) and draws, one
panel per system in order of separation, the RMS recovery error against pool size for the two
estimators. The adjusted count corrects classification's bias but still hard-labels, so it is consistent
yet not efficient; the soft estimator weighs each particle and the variance gap it closes shows as the
pool shrinks. Numbers a caption quotes are printed and dumped for the build to substitute.
"""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from reproduction.reporting.figures import (
    DOUBLE_COLUMN,
    PAD,
    PANEL_GAP,
    SERIES,
    SINGLE_COLUMN,
    TICK_PT,
    label_panels,
    save,
    use_house_style,
)
from reproduction.reporting.numbers import Ledger

DRIVER = "reproduction/experiments/figures/small_n.py"

DISPLAY = {"cas9": "SpCas9", "slco2a1": "SLCO2A1", "fanac": "FaNaC"}
"""The three swept systems, in display (decreasing-separation) order, with their figure names."""


def _series(by_size: dict, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool sizes and the mean +/- sem RMS error for one estimator, sorted by size."""
    sizes = sorted(int(n) for n in by_size)
    mean = np.array([by_size[str(n)][name]["mean"] for n in sizes])
    sem = np.array([by_size[str(n)][name]["sem"] for n in sizes])
    return np.array(sizes), mean, sem


def draw(systems: dict):
    """One log-log panel per system: two-pool and adjusted RMS error against pool size."""
    import matplotlib.pyplot as plt

    order = [s for s in DISPLAY if s in systems]
    figure, axes = plt.subplots(1, len(order), figsize=(DOUBLE_COLUMN, SINGLE_COLUMN * 0.82), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, system in zip(axes, order, strict=True):
        by_size = systems[system]["by_size"]
        n, two, two_e = _series(by_size, "2-pool")
        _, adj, adj_e = _series(by_size, "adjusted")
        ax.errorbar(
            n,
            adj,
            yerr=adj_e,
            color=SERIES[1],
            lw=1.1,
            ls=(0, (4, 2)),
            marker="D",
            ms=3.0,
            mew=0.0,
            capsize=1.5,
            label="Adjusted classify",
        )
        ax.errorbar(
            n, two, yerr=two_e, color=SERIES[0], lw=1.1, marker="o", ms=3.6, mew=0.0, capsize=1.5, label="Two pools"
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Particles per pool")
        ax.set_title(f"{DISPLAY[system]}  ($d'^2 = {systems[system]['d2_pool']:.1f}$)", fontsize=TICK_PT + 1)
        ax.set_box_aspect(1.0)
    axes[0].set_ylabel("RMS recovery error (pp)")
    axes[0].legend(loc="upper right", handlelength=1.4, borderpad=0.2, labelspacing=0.3, fontsize=TICK_PT)
    figure.tight_layout(pad=PAD, w_pad=PANEL_GAP)
    label_panels(*axes)
    return figure


def gaps(systems: dict) -> dict:
    """The adjusted-minus-two-pool RMS gap at each size, per system, plus the overall maximum."""
    per_system, overall = {}, 0.0
    for system, record in systems.items():
        by_size = record["by_size"]
        per_system[system] = {int(n): by_size[n]["adjusted"]["mean"] - by_size[n]["2-pool"]["mean"] for n in by_size}
        overall = max(overall, max(per_system[system].values()))
    return {"by_system": per_system, "max": overall}


def main(argv: Iterable[str] | None = None) -> int:
    """Draw the supplementary small-n panel and print every number a caption quotes."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", type=Path, default=Path("results/small_n_sweep.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/manuscript/figures/small_n"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    use_house_style()
    systems = json.loads(args.sweep.read_text())["result"]["systems"]
    figure = draw(systems)
    written = save(figure, args.out)

    val = Ledger()
    gap = gaps(systems)
    n_min = min(int(n) for n in next(iter(systems.values()))["by_size"])
    print(f"small-n sweep  {len(systems)} systems, smallest pool {val('smalln_n_min', n_min, 'd')} particles")
    print(f"max gap (adj-2pool)  {val('smalln_gap_max', gap['max'], '.1f')} pp")
    for system in DISPLAY:
        if system not in systems:
            continue
        d2 = systems[system]["d2_pool"]
        g512 = gap["by_system"][system].get(512)
        print(
            f"  {system:<8} d2={val(f'smalln_d2_{system}', d2, '.1f')}  "
            f"gap@512={val(f'smalln_gap512_{system}', g512, '.1f')} pp"
        )
    val.dump(args.out.parent / "values" / f"{args.out.name}.json")
    print("  wrote " + ", ".join(str(path) for path in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
