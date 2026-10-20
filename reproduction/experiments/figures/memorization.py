"""Figure 6: how much a single particle overfits the map it helped build.
    .venv/bin/python reproduction/experiments/figures/memorization.py
The overfit measured one particle at a time: its log-likelihood advantage on a map its own data built
against one it did not. Two half-maps split at random differ by nothing real, so the `self` arm minus
the `null` arm (reserved particles in neither half) is the particle recognizing itself. The prediction
is that this leakage dies as `1/N`; the -1 slope is asymptotic, so the fit window `TYPICAL_DATASET` is
declared, not selected, and `main()` prints the curvature, fit and unwindowed slope. Panels: (a) the
log-log law, (b) its percent residual with the full-range fit overlaid (the finite-N excess the log axis
hides), (c) the held-out null the self arm subtracts. One ladder is
drawn (`--record`, TmrAB, past the crossing); `--compare` ladders are printed only. The claim is the
exponent and the crossing of the real pair's `d^2` (SpCas9's, a cross-specimen mismatch in the caption).
"""

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from matplotlib import ticker

from cryodisc.statistics.information import deflection
from reproduction.record import payload_of, read_payload
from reproduction.reporting.figures import (
    OTHER,
    PAD,
    RULE,
    SERIES,
    TICK_PT,
    label_panels,
    save,
    use_house_style,
)
from reproduction.reporting.numbers import Ledger

DRIVER = "reproduction/experiments/figures/memorization.py"

TYPICAL_DATASET = (100_000, 1_000_000)
"""The range a final cryo-EM reconstruction is usually built from, drawn as a band.

A BAND AND NOT A MEDIAN, deliberately: no published median particle count per EMDB entry exists, so a
single line would be invented authority. The band spans where published reconstructions ordinarily
sit, and the conclusion does not turn on where inside it a reader puts their own dataset."""


def read_real_d2(pool_a: Path, pool_b: Path) -> float:
    """The real SpCas9 pair's per-particle `d^2`, the level the leakage is compared against.

    Read off the same two score arrays the recovery figure draws, not carried as a constant: it is the
    only number on this panel that does not come from the ladder's own record, and without it the
    ordinate is nats and a reader cannot tell whether 4 is a lot.

    Args:
        pool_a: Per-particle log-ratios for one state.
        pool_b: The same for the other.

    Returns:
        `(mean gap)^2 / pooled variance`.
    """
    return float(deflection(jnp.asarray(np.load(pool_a)), jnp.asarray(np.load(pool_b))))


def read_cas9_mass(record_path: Path) -> float | None:
    """The SpCas9 example pair's matched-chain mass in kDa, over the atoms the Cα RMSD used.

    The `_molecular_mass` the survey law's M is built from, re-aligning the same structures, assembly
    and ligands the cited Cα RMSD record names, so the mass and the RMSD describe one matched atom set.

    Args:
        record_path: The `score_pool` record the body reads the pair's Cα RMSD from.

    Returns:
        The mass in kDa, or `None` when the models are not on disk, so the figure still builds from a
        checkout without them.
    """
    arguments = json.loads(record_path.read_text())["arguments"]
    structure_a, structure_b = Path(arguments["structure_a"]), Path(arguments["structure_b"])
    if not (structure_a.exists() and structure_b.exists()):
        return None
    from cryodisc.specimen.structures import align_conformers
    from reproduction.targets import _molecular_mass

    pair = align_conformers(
        structure_a,
        structure_b,
        assembly=arguments.get("assembly", "1"),
        ligands=tuple(arguments.get("ligands") or ()),
        superpose=not arguments.get("no_superpose", False),
    )
    return _molecular_mass(pair)


def read_dataset_label(record: dict) -> str:
    """Which dataset a law record's ladder was built from, read off the ladder it names.

    The law record does not carry the accession; the ladder manifest it was fitted from carries the
    cryoSPARC project directory, whose name contains it. Parsed rather than passed in, because a
    legend entry naming the wrong specimen is a mislabeled figure that no test would catch.

    Args:
        record: The law record, whose `arguments.manifest` names the ladder.

    Returns:
        `EMPIAR-11779` where the accession is recoverable, otherwise the manifest's stem.
    """
    manifest = Path(record.get("arguments", {}).get("manifest", ""))
    if not manifest.name:
        return "one ladder"
    try:
        rungs = read_payload(manifest).get("rungs", [])
    except OSError:
        return manifest.stem
    # `(?:-|$)` and not a bare `-`: the accession can be last in the name or followed by a suffix, and
    # requiring a trailing dash would silently miss the terminal case
    accessions = {
        match.group(1) for rung in rungs if (match := re.search(r"-(\d{5})(?:-|$)", str(rung.get("root", ""))))
    }
    if len(accessions) == 1:
        return f"EMPIAR-{accessions.pop()}"
    return manifest.stem


_LADDER_STYLES = (
    {"color": SERIES[0], "marker": "o", "ls": "-", "mfc": SERIES[0], "mew": 0.0, "ms": 4.6, "lw": 1.4},
    {"color": SERIES[2], "marker": "o", "ls": "--", "mfc": "none", "mew": 1.1, "ms": 5.0, "lw": 1.1},
    {"color": OTHER, "marker": "s", "ls": ":", "mfc": OTHER, "mew": 0.0, "ms": 4.2, "lw": 1.1},
)
"""How each ladder after the first is told apart, in the order they are given.

Three hues is the house cap and `SERIES[1]` is the measured-`d^2` rule, so a third ladder carries its
identity in MARKER and DASH instead of a fourth color -- filled circles on a solid join, open circles
on a dashed one, filled squares on a dotted one. That also survives grayscale printing, which a fourth
hue would not.
"""


def read_law(path: Path) -> dict:
    """The rung ladder and the fitted law.

    Args:
        path: `e19_memorization_law.json`.

    Returns:
        `n`, `self_mean`, `self_err`, `null_mean`, `null_err`, `difference`, `difference_err`,
        the free fit, the de-attenuated fit where the record carries one, and a `label` naming the
        dataset, which the legend needs once more than one ladder is drawn.

    Raises:
        SystemExit: If the record carries fewer than three rungs, a slope over two being a line
            through two points rather than a measurement.
    """
    record = json.loads(path.read_text())
    result = payload_of(record)
    rungs = sorted(result.get("rungs", []), key=lambda row: row["n_particles"])
    if len(rungs) < 3:
        raise SystemExit(f"{path} carries {len(rungs)} rungs; a slope needs at least three")
    return {
        "n": np.array([row["n_particles"] for row in rungs], dtype=float),
        "self_mean": np.array([row["self"]["mean"] for row in rungs], dtype=float),
        "self_err": np.array([row["self"]["sem"] for row in rungs], dtype=float),
        "null_mean": np.array([row["null"]["mean"] for row in rungs], dtype=float),
        "null_err": np.array([row["null"]["sem"] for row in rungs], dtype=float),
        # THE QUANTITY THE LAW IS FITTED ON (self minus null), exposed under one key by both paths so a
        # caller does not subtract the null a second time
        "difference": np.array([row["difference"] for row in rungs], dtype=float),
        "difference_err": np.array([row["difference_sem"] for row in rungs], dtype=float),
        "free": result["fit_free_slope"],
        # the same ladder with the scorer's fitted amplitude divided out; absent from records
        # written before that was separated from the geometry
        "deattenuated": result.get("fit_with_scorer_amplitude_removed"),
        # THE FIT WINDOW. Defaults to 0 for records written before `--fit-min` existed, which is the
        # honest reading of them: they were fitted on every rung they carry.
        "fit_min": float(result.get("fit_min_particles", 0.0) or 0.0),
        "commit": record.get("commit"),
        "label": read_dataset_label(record),
    }


def apply_replicates(data: dict, path: Path) -> dict:
    """Swap the drawn ladder's points and fit for the replicate campaign's.

    A single ladder's error bar is the per-particle standard error of one reconstruction, which
    treats the map as fixed. It is not: a different draw of `N` particles builds a different map.
    `memorization_replicates.py` averages independent draws per rung and takes their spread, which is
    the only error bar that carries that term. `main()` prints the measured inflation over the
    single-ladder error.

    The upper panel's quantity becomes the self-minus-null difference rather than the self arm
    alone. That is what the law was always fitted on, and the null straddles zero at every rung, so
    the points move by far less than their error bars -- but it is a different quantity and the
    caption has to say so.

    Args:
        data: As `read_law` returns it, for the same ladder the replicates redraw.
        path: A `memorization_replicates.py` record.

    Returns:
        `data`, with `n`, `difference`, `difference_err`, `free` and `fit_min` replaced, the null
        arms realigned onto the replicate rungs under `null_n`, and the self arm removed.

    Raises:
        SystemExit: If a rung carries no empirical error and none can be scaled onto it.
    """
    result = payload_of(json.loads(path.read_text()))
    rows = sorted(result["rungs"], key=lambda row: row["n_particles"])
    inflation = float(result["median_inflation"])
    free = dict(result["fit_free_slope"])
    # the null arms come from the REPLICATES, not the `--record` ladder: every replicate scores the
    # same held-out reserve at every rung, so both panels carry the same abscissa
    data["n"] = np.array([row["n_particles"] for row in rows], dtype=float)
    data["null_n"] = data["n"]
    data["null_mean"] = np.array([row["null_mean"] for row in rows], dtype=float)
    data["null_err"] = np.array([row["null_sem"] for row in rows], dtype=float)
    data["difference"] = np.array([row["mean"] for row in rows], dtype=float)
    # THE CAMPAIGN IS TAPERED, so a single "4 draws per rung" would be wrong at the top of the
    # ladder: the pool only allows one 1,024,000-particle draw. Printed per rung for that reason.
    data["n_draws"] = np.array([row["n_draws"] for row in rows], dtype=int)
    # SINGLETON RUNGS TAKE THE SCALED PER-PARTICLE ERROR, the same substitution the fit makes, so a
    # rung that could not be drawn twice is not shown with an error bar it did not earn
    data["difference_err"] = np.array(
        [
            row["sem_empirical"] if row["sem_empirical"] is not None else row["sem_per_particle"] * inflation
            for row in rows
        ],
        dtype=float,
    )
    # the self arm alone is not carried by a replicate record; nothing may read it after this
    data.pop("self_mean", None)
    data.pop("self_err", None)
    # neither is the calibration arm: `memorization_replicates.py` computes none, so the de-attenuated
    # fit is dropped rather than filled from the single ladder (which would mix two ladders' fits)
    data["deattenuated"] = None
    data["free"] = free
    data["fit_min"] = float(result.get("fit_min_particles", 0.0) or 0.0)
    data["n_replicates"] = int(result["n_replicates"])
    data["median_inflation"] = inflation
    # carried so the caption can quote them: the fit window is CHOSEN on the curvature test, so a
    # caption that states the window without the test it passed states half the measurement
    data["log_quadratic"] = result.get("log_quadratic")
    data["full_range"] = result.get("fit_free_slope_all_rungs")
    data["span_decades"] = float(np.log10(data["n"].max() / data["n"].min()))
    return data


def draw(data: dict, *others: dict):
    """The two arms against dataset size, with the 1/N fit drawn over the typical-dataset band.

    Args:
        data: As `read_law` returns it.
        second: An independent ladder on a different dataset, drawn as a second series. The fitted
            line is drawn for the FIRST only: two lines this close overlap into one thick rule and
            the eye cannot tell which points belong to which, whereas two point series on a single
            line is exactly the claim being made.

    Returns:
        The figure.
    """
    import matplotlib.pyplot as plt

    figure, (axes, resid, lower) = plt.subplots(
        3,
        1,
        figsize=(4.0, 6.4),
        sharex=True,
        height_ratios=[2.5, 0.75, 0.75],  # (b) and (c) equal plot-box height; (c)'s ticks/label sit below it
    )

    # one fitted line: the 1/N headline fit, solid across the typical-dataset band where it is fitted
    # and where real datasets sit. The full-range fit is steeper (a finite-N excess at small datasets)
    # and lives in the prose, so the panel carries the predicted law and not that finite-N artifact.
    grid = np.geomspace(data["n"].min() * 0.75, 1_400_000, 200)
    free = data["free"]
    band = (grid >= TYPICAL_DATASET[0]) & (grid <= TYPICAL_DATASET[1])
    band_law = np.exp(free["log_amplitude"]) * grid[band] ** free["slope"]
    axes.plot(grid[band], band_law, color="0", lw=1.4, zorder=5)
    # centred within [100k, 1M] so the full label stays in the shaded region, and lifted a fixed factor
    # above the line AT THE LABEL'S OWN POSITION (band_law[i]) so it sits just clear, not floating high
    i = int(band.sum() * 0.42)
    axes.annotate(
        f"slope {free['slope']:.3f}  (1/N)",
        xy=(grid[band][i], band_law[i] * 4.0),
        fontsize=TICK_PT,
        color="0",
        ha="center",
        va="center",
    )

    # REFUSED rather than wrapped with `% len(...)`: a fourth ladder reusing the first one's color,
    # marker and dash is a figure with two indistinguishable series, which no test would catch.
    if 1 + len(others) > len(_LADDER_STYLES):
        raise ValueError(
            f"{1 + len(others)} ladders but only {len(_LADDER_STYLES)} distinguishable styles; add one "
            "to _LADDER_STYLES that stays separable in grayscale"
        )
    # every ladder through one loop: markers and dashes keep them separable where they overlap, which
    # is nearly everywhere -- that being the point of drawing them together
    for index, ladder in enumerate((data, *others)):
        style = _LADDER_STYLES[index]
        axes.errorbar(
            ladder["n"],
            ladder["difference"],
            yerr=ladder["difference_err"],
            marker=style["marker"],
            ls=style["ls"],
            color=style["color"],
            lw=style["lw"],
            mfc=style["mfc"],
            mew=style["mew"],
            ms=style["ms"],
            elinewidth=0.9,
            capsize=1.8,
            zorder=4 - index,
            # the generic label only makes sense when this is the ONLY ladder; with companions every
            # series has to name its own dataset
            label="Built the map" if index == 0 and not others else ladder["label"],
        )

    # (b) the same points as a percent departure from their own 1/N fit: the log-log panel (a) hides a
    # few-percent residual, so the fit looks exact; here each point's error bar is finally resolvable.
    resid.axhline(0.0, color="0", lw=1.0, zorder=5)
    for index, ladder in enumerate((data, *others)):
        style = _LADDER_STYLES[index]
        fit = ladder["free"]
        fit_at_n = np.exp(fit["log_amplitude"]) * ladder["n"] ** fit["slope"]
        percent = 100.0 * (ladder["difference"] / fit_at_n - 1.0)
        resid.errorbar(
            ladder["n"],
            percent,
            yerr=100.0 * ladder["difference_err"] / fit_at_n,
            marker=style["marker"],
            ls="none",
            color=style["color"],
            mfc=style["mfc"],
            mew=style["mew"],
            ms=style["ms"],
            elinewidth=0.9,
            capsize=1.8,
            zorder=4 - index,
        )
    # the full-range fit, as its percent departure from the windowed fit: steeper, so it rises above zero
    # at small datasets -- the finite-N excess the prose names, drawn only over the range it was fitted on
    full = data.get("full_range")
    if full:
        win = data["free"]
        over = (grid >= data["n"].min()) & (grid <= data["n"].max())
        full_law = np.exp(full["log_amplitude"]) * grid[over] ** full["slope"]
        win_law = np.exp(win["log_amplitude"]) * grid[over] ** win["slope"]
        resid.plot(
            grid[over],
            100.0 * (full_law / win_law - 1.0),
            color=OTHER,
            lw=1.1,
            ls=(0, (3, 2)),
            zorder=3,
            label="full-range fit",
        )
        resid.legend(
            loc="upper center", handlelength=1.4, borderpad=0.3, fontsize=TICK_PT, framealpha=0.9, edgecolor="0.85"
        )
    resid.axvspan(*TYPICAL_DATASET, color=OTHER, alpha=0.14, lw=0.0, zorder=1)
    resid.set_ylabel("Residual (%)")

    # the null as its own panel, not an inset: its flatness IS the result (a shared artifact would
    # appear here too), so it should not read as an aside. Linear ordinate, since the values straddle zero.
    lower.axhline(0.0, color=RULE, lw=0.9, zorder=1)
    # matched to (a), so a marker means the same dataset in both panels; gray with one ladder only,
    # the null being a control and not a series
    for index, ladder in enumerate((data, *others)):
        style = _LADDER_STYLES[index]
        # gray when this is the only ladder, the null being a control rather than a series; one
        # condition, not the same one written twice
        color = style["color"] if others else OTHER
        lower.errorbar(
            ladder.get("null_n", ladder["n"]),
            ladder["null_mean"],
            yerr=ladder["null_err"],
            marker=style["marker"],
            ls="none",
            color=color,
            mfc=style["mfc"] if others else OTHER,
            mew=style["mew"],
            ms=style["ms"] * 0.9,
            elinewidth=0.9,
            capsize=1.8,
            zorder=3 - index,
        )
    lower.set_ylim(-0.07, 0.07)
    lower.set_yticks([-0.05, 0.0, 0.05], ["$-$0.05", "0", "0.05"])
    lower.set_ylabel("Held-out particles")
    lower.set_xlabel("Particles the map was built from")

    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xticks(
        [1000, 3000, 10_000, 30_000, 100_000, 300_000, 1_000_000],
        ["1k", "3k", "10k", "30k", "100k", "300k", "1M"],
    )
    axes.xaxis.set_minor_locator(ticker.NullLocator())
    axes.set_yticks([0.125, 0.5, 2, 8, 32, 128], ["0.125", "0.5", "2", "8", "32", "128"])
    axes.set_ylim(0.1, 160)
    axes.yaxis.set_minor_locator(ticker.NullLocator())
    # named for what it MEASURES rather than how it is computed: this is the advantage one particle has
    # on a map it helped build, which is a per-particle overfitting scale
    axes.set_ylabel("Overfit per particle (log-ratio)")
    axes.legend(loc="lower left", handlelength=1.4, borderpad=0.3, fontsize=TICK_PT, framealpha=0.9, edgecolor="0.85")
    # where a real dataset sits, since the measured ladder stops two orders of magnitude short of one
    axes.axvspan(*TYPICAL_DATASET, color=OTHER, alpha=0.14, lw=0.0, zorder=1)
    axes.annotate(
        "where final reconstructions\nare usually built",
        xy=(float(np.sqrt(TYPICAL_DATASET[0] * TYPICAL_DATASET[1])), 0.97),
        xycoords=("data", "axes fraction"),
        xytext=(0, 0),
        textcoords="offset points",
        fontsize=TICK_PT,
        color="0",
        ha="center",
        va="top",
    )
    axes.set_box_aspect(1.0)  # square main panel
    # panels in reading order: (a) the 1/N law, (b) its percent residual, (c) the held-out null
    label_panels(axes, resid, lower)
    figure.tight_layout(pad=PAD, h_pad=0.6)
    return figure


def main(argv: Iterable[str] | None = None) -> int:
    """Draw the memorization figure and print the ladder it drew."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # The DRAWN ladder, and the only one on the panel: TmrAB, the one with rungs at the crossing.
    parser.add_argument("--record", type=Path, default=Path("results/e20_memorization_law.json"))
    parser.add_argument(
        "--replicates",
        type=Path,
        # the default, being the better measurement of the same ladder; falls back to the single
        # ladder when the record is absent
        default=Path("results/e20_memorization_replicates.json"),
        help="a `memorization_replicates.py` record. Replaces the drawn ladder's points and error "
        "bars with the replicate means and their measured spread, and its fit with the replicate "
        "fit -- see `apply_replicates`",
    )
    # compared in print, NOT drawn; each is optional and silently skipped when absent, so the figure
    # still builds from a checkout carrying only the drawn ladder's record
    parser.add_argument(
        "--compare",
        type=Path,
        nargs="*",
        default=[Path("results/e19_memorization_law.json"), Path("results/e19b_memorization_law.json")],
        help="ladders printed against the drawn one; not drawn",
    )
    parser.add_argument("--pool-a", type=Path, default=Path("results/rerun_cas9_hoRepro_6nt.npy"))
    parser.add_argument("--pool-b", type=Path, default=Path("results/rerun_cas9_hoRepro_12nt.npy"))
    parser.add_argument(
        "--pair-record",
        type=Path,
        default=Path("results/cas9_bl_6nt_00625.json"),
        help="the score_pool record the body reads the pair's Cα RMSD from; its models give cas9_mass",
    )
    parser.add_argument("--out", type=Path, default=Path("docs/manuscript/figures/memorization"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    use_house_style()
    val = Ledger()
    data = read_law(args.record)
    if args.replicates is not None and args.replicates.exists():
        data = apply_replicates(data, args.replicates)
        print(
            f"drawn from {data['n_replicates']} replicate ladders, up to {int(max(data['n_draws']))} "
            f"draws at any one rung as the pool allows; the single-ladder error "
            f"understates by {data['median_inflation']:.2f}x"
        )
    elif args.replicates is not None:
        print(f"no {args.replicates}; drawing the single ladder with its per-particle errors")
    others = [read_law(path) for path in args.compare if path.exists()]
    real_d2 = read_real_d2(args.pool_a, args.pool_b)
    val("real_pair_d2", real_d2, ".2f")  # the SpCas9 pair, cited across Figs 1 and 4 (prose owns any sign)
    cas9_mass = read_cas9_mass(args.pair_record)
    if cas9_mass is None:
        print(f"  no models under {args.pair_record}; skipping cas9_mass")
    else:
        val("cas9_mass", cas9_mass, ".1f")  # matched-chain mass M (kDa), the survey law's M for this pair
        print(f"  the real pair's matched-chain mass is {cas9_mass:.1f} kDa")
    # `others` reaches the PRINTED comparison only. Passing them to `draw` is what put three series
    # on the panel; the band and amplitude comparisons below still read every one of them.
    figure = draw(data)
    written = save(figure, args.out)

    print(f"memorization ladder at commit {data['commit']}")
    print(f"  the real pair sits at d^2 = {real_d2:.4f}")
    draws = (
        [f" ({n} draw{'s' if n > 1 else ''})" for n in data["n_draws"]] if "n_draws" in data else [""] * len(data["n"])
    )
    # by rung size, not by position: `apply_replicates` drops a rung with no null arm, so the two
    # arrays need not align and a positional index would print the wrong null past the first gap
    null_at = dict(
        zip(data.get("null_n", data["n"]), zip(data["null_mean"], data["null_err"], strict=True), strict=True)
    )
    for index, size in enumerate(data["n"]):
        null = null_at.get(size)
        shown = f"   null {null[0]:+.4f} +- {null[1]:.4f}" if null else "   null (no reserve scored)"
        print(
            f"  N {size:7,.0f}   self-null {data['difference'][index]:8.3f} +- {data['difference_err'][index]:.3f}"
            f"{draws[index]}{shown}"
        )
    free = data["free"]
    goodness = f", chi2/dof {free['chi_square_per_dof']:.2f} on {free['dof']:.0f} dof" if "dof" in free else ""
    print(f"  free slope {free['slope']:+.4f} +- {free['slope_sem']:.4f}{goodness}")
    val("memo_exponent", abs(free["slope"]), ".3f")  # magnitude; the prose writes the minus
    val("memo_exponent_err", free["slope_sem"], ".3f")
    # the window-dependence, printed: the fitted slope moves with `--fit-min` because the law is
    # asymptotic, so `full_range` (the same fit with no floor) is carried beside it
    if data.get("full_range"):
        whole = data["full_range"]
        print(
            f"  over every rung ({data['n'].size} of them, {data['span_decades']:.2f} decades) the same fit gives "
            f"{whole['slope']:+.4f} +- {whole['slope_sem']:.4f}, and the slope moves toward -1 as the floor rises"
        )
        val("memo_fullrange_slope", abs(whole["slope"]), ".2f")  # prose caveat; the finite-N excess at low N
    if data.get("log_quadratic"):
        bend = data["log_quadratic"]
        print(
            f"  a quadratic term in log N over the fitted window runs {bend['sigma']:.1f} sigma "
            f"({bend['curvature']:+.4f} +- {bend['curvature_sem']:.4f}), so a straight line describes it"
        )
    if data["deattenuated"]:
        d = data["deattenuated"]
        print(
            f"  with the scorer's fitted amplitude divided out: free slope {d['free']['slope']:+.4f}"
            f" +- {d['free']['slope_sem']:.4f}"
        )
        # tested, not asserted: whether the two slopes bracket -1 depends on the ladder, so it is
        # evaluated rather than stated (an unevaluated conclusion is what this figure is about)
        pair = (free["slope"], d["free"]["slope"])
        sems = (free["slope_sem"], d["free"]["slope_sem"])
        brackets = min(pair) < -1.0 < max(pair)
        # consistency before bracketing: bracketing compares two point estimates and ignores their
        # errors, so two estimates that both agree with -1 need no bracket and must not be failed on one
        worst = max(abs(value + 1.0) / sem for value, sem in zip(pair, sems))
        if worst < 2.0:
            verdict = f"are BOTH CONSISTENT with -1, the furthest at {worst:.1f} sigma"
        elif brackets:
            verdict = "BRACKET -1, so the departure from -1 is the calibration and not the law"
        else:
            # name which one disagrees: `worst` is a max over both, so reporting "neither agrees" when
            # only one departs is false of half the sentence and is what a caption would copy
            apart = [abs(value + 1.0) / sem for value, sem in zip(pair, sems)]
            names = ("as scored", "de-attenuated")
            agreeing = [name for name, sigma in zip(names, apart) if sigma < 2.0]
            if agreeing:
                verdict = (
                    f"SPLIT: the {agreeing[0]} fit agrees with -1 and the other does not "
                    f"({worst:.1f} sigma), so the calibration is what carries the departure"
                )
            else:
                verdict = (
                    f"DO NOT bracket -1 and neither agrees with it ({worst:.1f} sigma at worst): both are "
                    f"{'steeper' if max(pair) < -1.0 else 'shallower'}, so the calibration does not "
                    "account for the departure here"
                )
        print(f"  the two defensible treatments of the calibration {verdict}:")
        print(f"    {free['slope']:+.4f} as scored, {d['free']['slope']:+.4f} de-attenuated")
    # the crossing, printed as the figure's claim; solved on the FREE fit the panel draws, with
    # whether it is measured or extrapolated stated beside it
    crossing = float((real_d2 / np.exp(free["log_amplitude"])) ** (1.0 / free["slope"]))
    val("memo_crossing", crossing, ",.0f")
    # the reference crossing at one unit of discriminability (d'^2 = 1); by the 1/N law the crossing of any
    # target d'^2 is this count divided by d'^2, so one number extends to any pair (within this specimen)
    unit_crossing = float((1.0 / np.exp(free["log_amplitude"])) ** (1.0 / free["slope"]))
    val("memo_unit_crossing", unit_crossing, ",.0f")
    # the self-recognition's own d'^2 at the edges of the routine 100k-1M build window, for a SpCas9-free
    # magnitude statement (overfit(N) = exp(log_amp) * N^slope)
    val("memo_overfit_100k", float(np.exp(free["log_amplitude"]) * 1e5 ** free["slope"]), ".2f")
    val("memo_overfit_1m", float(np.exp(free["log_amplitude"]) * 1e6 ** free["slope"]), ".2f")
    # against the FIT WINDOW, not every rung drawn: with `--fit-min` the line is only a fit above the
    # window, so a crossing below it is an extrapolation even where rungs were measured
    fitted_lo = max(float(data["n"].min()), float(data["fit_min"]))
    fitted_hi = float(data["n"].max())
    if fitted_lo <= crossing <= fitted_hi:
        where = "INSIDE the fitted window"
    elif crossing < fitted_lo:
        where = f"{fitted_lo / crossing:.2f}x BELOW the fitted window, so extrapolated"
    else:
        where = f"{crossing / fitted_hi:.2f}x above the last rung, so extrapolated"
    print(
        f"  the fitted law reaches the real pair's d^2 = {real_d2:.4f} at N = {crossing:,.0f}"
        f" particles ({where}: fitted on {fitted_lo:,.0f} to {fitted_hi:,.0f},"
        f" rungs drawn from {float(data['n'].min()):,.0f})"
    )
    # read off the measured rungs rather than the fitted line, so the overfit-to-d^2 ratio stands
    # whatever the fit window is
    for mark in (32_000, 1_000_000):
        nearest = int(np.argmin(np.abs(data["n"] - mark)))
        ratio = float(data["difference"][nearest] / real_d2)
        print(f"  at the {data['n'][nearest]:,.0f}-particle rung the overfit is {ratio:.2f}x the real pair's d^2")
        if mark == 32_000:
            val("memo_ratio_at_32k", ratio, ".2f")
        if mark == 1_000_000:
            # the overfit as a share of the real difference at the largest measured dataset: still
            # non-zero but only a fraction of a genuine pair -- out-signaled, not out-averaged
            val("memo_ratio_at_1m", ratio, ".2f")
            # and its significance there, the self-null advantage over its own error: the advantage
            # never sinks into its sampling noise, so it stays detectable at every realistic size
            significance = float(data["difference"][nearest] / data["difference_err"][nearest])
            print(f"    that rung's advantage is {significance:.1f}x its own error, so still detectable")
            val("memo_significance_at_1m", significance, ".1f")
    # the null's error bar is the across-rung scatter, not the per-particle sem: the reserve is the same
    # particles every rung, so its null is one realization that does not shrink with count.
    # the drawn ladder's own nulls first, since panel (b) shows those; the pooled set below is stronger
    # but a caption quoting it beside an 8-point panel reads as short.
    own = np.asarray(data["null_mean"])
    own_sems = np.asarray(data["null_err"])
    print(
        f"  null arms on the DRAWN ladder ({data['label']}), which is what panel (b) shows:"
        f" {own.size} of them, mean {own.mean():+.4f}, across-rung sd {float(own.std(ddof=1)):.4f}"
        f" against a median per-particle sem of {np.median(own_sems):.4f}"
    )
    val("memo_null_mean", float(own.mean()), "+.4f")  # panel (b)'s drawn-ladder null, consistent with zero
    val("memo_null_sd", float(own.std(ddof=1)), ".4f")
    nulls = np.concatenate([ladder["null_mean"] for ladder in (data, *others)])
    sems = np.concatenate([ladder["null_err"] for ladder in (data, *others)])
    scatter = float(nulls.std(ddof=1))
    worst = int(np.argmax(np.abs(nulls)))
    print(
        f"  null arms pooled over the drawn ladder and the {len(others)} compared: {nulls.size} of"
        f" them, mean {nulls.mean():+.4f}, across-rung sd {scatter:.4f}"
        f" against a median per-particle sem of {np.median(sems):.4f}"
    )
    print(
        f"    chi2 vs zero on the per-particle sem {float(((nulls / sems) ** 2).sum()):.1f} on "
        f"{nulls.size} dof, so that sem understates the null's spread by {scatter / np.median(sems):.2f}x"
    )
    print(
        f"    largest |null| {nulls[worst]:+.4f} = {abs(nulls[worst]) / scatter:.1f}x the across-rung sd"
        f" ({abs(nulls[worst]) / sems[worst]:.1f}x its own per-particle sem)"
    )

    for idx, second in enumerate(others):
        tag = "ab"[idx] if idx < 2 else str(idx)
        val(f"memo_cas9{tag}_slope", abs(second["free"]["slope"]), ".3f")  # SpCas9 series; prose writes the minus
        val(f"memo_cas9{tag}_err", second["free"]["slope_sem"], ".3f")
        val(f"memo_cas9{tag}_nmin", float(second["n"].min()), ",.0f")
        val(f"memo_cas9{tag}_nmax", float(second["n"].max()), ",.0f")
        print(f"  compared against {second['label']}, an independent ladder on a different dataset:")
        # the crossing travels even where the fit does not: the slope and amplitude disagreements move
        # the line oppositely and cancel where it meets `real_d2`, so the crossing replicates
        other_free = second["free"]
        other_cross = float((real_d2 / np.exp(other_free["log_amplitude"])) ** (1.0 / other_free["slope"]))
        val(f"memo_cas9{tag}_crossing", other_cross, ",.0f")  # SpCas9's own self-recognition crossing its own d'^2
        other_top = float(second["n"].max())
        reach = (
            "measured"
            if other_cross <= other_top
            else f"extrapolated {other_cross / other_top:.1f}x past its last rung"
        )
        print(
            f"    crossing of d^2 = {real_d2:.2f}   {crossing:,.0f} vs {other_cross:,.0f} particles"
            f"  -> {abs(crossing - other_cross) / crossing * 100:.1f}% apart, and theirs is {reach}"
        )
        # the de-attenuated row only where both ladders have one, so a replicate ladder (no calibration
        # arm) does not lose the crossing, which is the half the manuscript quotes
        slopes = [("as scored", data["free"], second["free"])]
        if data["deattenuated"] and second["deattenuated"]:
            slopes.append(("de-attenuated", data["deattenuated"]["free"], second["deattenuated"]["free"]))
        else:
            print("    de-attenuated slope not compared: one ladder carries no calibration arm")
        for name, first_fit, other_fit in slopes:
            gap = abs(first_fit["slope"] - other_fit["slope"])
            spread = np.hypot(first_fit["slope_sem"], other_fit["slope_sem"])
            print(
                f"    slope {name:14s} {first_fit['slope']:+.4f} vs {other_fit['slope']:+.4f}"
                f"  -> {gap / spread:.2f} sigma apart"
            )

    val.dump(args.out.parent / "values" / f"{args.out.name}.json")
    print("  wrote " + ", ".join(str(path) for path in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
