"""Figure 5: pose-bank convergence collapses onto one law, and its scale is derived from structure.
    .venv/bin/python .../kc_convergence.py --fit results/eq1_mechanistic.json --survey 'results/kc_survey/*.json' --pool 'results/survey_pool/*.json' --out .../bank
Each pair grows its own pose bank to convergence. (a) Every bank size of every pair, its d'^2 against its
own converged plateau and its bank against its own half-recovery count K*, collapses onto the single
crossover a surface-dominated difference predicts (`CROSSOVER_EXPONENT`). (b) K* is a fixed share of
the SO(3) cells the difference decorrelates across (`correlation_cells`), one constant fitted. The
retention d'^2/d'^2_known by motion class is printed for the appendix table, not drawn. Rationale: docs/rationale/figures.md
"""

import argparse
import glob
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from cryodisc.statistics.convergence import CROSSOVER_EXPONENT
from reproduction.analysis.kc_survey import correlation_cells, free_exponent, lever_arms, rms_frequency
from reproduction.record import payload_of
from reproduction.reporting.figures import (
    BAND,
    DOUBLE_COLUMN,
    PAD,
    PANEL_GAP,
    RULE,
    SEQUENTIAL,
    SERIES,
    SINGLE_COLUMN,
    TICK_PT,
    label_panels,
    save,
    shared_contours,
    use_house_style,
)
from reproduction.reporting.numbers import Ledger

DRIVER = "reproduction/experiments/figures/kc_convergence.py"
A_EXP = CROSSOVER_EXPONENT  # the fit's own exponent, so the drawn law is the one the plateaus came from


def master(reach):
    """The convergence law in reduced coordinates: d'^2/d'^2_inf against bank/K*."""
    return 1.0 / (1.0 + (1.0 / reach) ** A_EXP)


def read_pairs(fit_path):
    """Per-pair converged plateau and K* from the mechanistic fit record, keyed by name."""
    rows = payload_of(json.loads(Path(fit_path).read_text()))["rows"]
    return {r["name"]: r for r in rows}


def read_ladders(survey_glob):
    """Raw {K: d'^2} ladder per pair name, from the convergent-survey records."""
    ladders = {}
    for path in glob.glob(survey_glob):
        if "_tmp" in path:
            continue
        record = json.loads(Path(path).read_text())
        record = record.get("result", record)
        if record.get("status") != "converged":
            continue
        curve = record.get("curve") or {}
        if record.get("name") and curve:
            ladders[record["name"]] = {int(k): float(v) for k, v in curve.items()}
    return ladders


def read_ext(ext_glob):
    """Raw {K: d'^2} per pair name from ip_converge ladder-extension records (rows[].rungs[], their key).

    These are bank sizes scored past the survey's adaptive stop (build_ladder_floor.py, extend_ladders_orc.sh)
    at the pair's own dose/seed, so they extend that pair's curve rather than form a separate one.
    """
    ext = {}
    for path in glob.glob(ext_glob):
        try:
            rows = json.loads(Path(path).read_text()).get("rows", [])
        except (json.JSONDecodeError, OSError):
            continue
        for row in rows:
            name = row.get("name")
            for rung in row.get("rungs", []):
                if name and rung.get("k") and rung.get("d2") is not None:
                    ext.setdefault(name, {})[int(rung["k"])] = float(rung["d2"])
    return ext


def read_pool(pool_glob):
    """Per-pair radius, pose-known d'^2 and motion class from the survey pool, keyed by name."""
    pool = {}
    for path in glob.glob(pool_glob):
        try:
            rows = json.loads(Path(path).read_text()).get("result", {}).get("rows", [])
        except (json.JSONDecodeError, OSError):
            continue
        for row in rows:
            name = row.get("name")
            if name and name not in pool:
                pool[name] = row
    return pool


def retention_by_class(pairs, pool, min_members=20):
    """Converged retention d'^2/d'^2_known per motion class, and its dependence on d'^2.

    Returns:
        `(motion, median, count, values)` per class of at least `min_members`, least retained first;
        the overall median, its count, and the log-log slope of retention on d'^2.
    """
    from collections import defaultdict

    by, overall_ret, overall_d2 = defaultdict(list), [], []
    for name, row in pairs.items():
        info = pool.get(name, {})
        known, motion, plateau = info.get("d2_known_pose"), info.get("motion"), row.get("d2_plateau")
        if known and plateau and motion and known > 0:
            by[motion].append(plateau / known)
            overall_ret.append(plateau / known)
            overall_d2.append(plateau)
    classes = sorted(
        ((m, float(np.median(v)), len(v), np.array(v)) for m, v in by.items() if len(v) >= min_members),
        key=lambda entry: entry[1],
    )
    slope = float(np.polyfit(np.log(overall_d2), np.log(overall_ret), 1)[0])
    return classes, float(np.median(overall_ret)), len(overall_ret), slope


def read_cells(spectrum_path, features_path):
    """Per-pair SO(3) cell count, RMS difference frequency and lever arms, from structure alone."""
    record = json.loads(Path(spectrum_path).read_text())
    spectrum = payload_of(record)
    frequency = np.array(spectrum["frequency_cyc_px"]) / record["arguments"]["pixel_size"]  # 1/A
    weight = np.array(spectrum["shell_weight"])
    shapes = {r["name"]: r for r in payload_of(json.loads(Path(features_path).read_text()))["rows"]}
    cells = {}
    for row in spectrum["rows"]:
        shape = shapes.get(row["name"])
        if not shape or not shape.get("rg"):
            continue
        k_rms = rms_frequency(row["defocus"][0]["power"], weight, frequency)
        arms = lever_arms(shape["rg"], shape["asphericity"], shape["acylindricity"])
        cells[row["name"]] = (float(correlation_cells(k_rms, arms)), k_rms, arms)
    return cells


def kstar_law(pairs, cells, ladders):
    """K* as a fixed share of the cell count, fitted only where K* lies inside the bank sizes scored.

    Returns:
        The share, predicted and measured K*, which pairs were resolved, the freed exponents on
        (arm product, k_rms) over those, and names.
    """
    names = [name for name, row in pairs.items() if name in cells and name in ladders and row.get("kc")]
    measured = np.array([pairs[name]["kc"] for name in names])
    # a K* below the first bank scored is an extrapolation of the ladder, not a measurement
    resolved = measured >= np.array([min(ladders[name]) for name in names])
    count = np.array([cells[name][0] for name in names])
    share = float(np.exp(np.mean(np.log(measured[resolved] / count[resolved]))))
    design = np.column_stack(
        [
            np.ones(len(names)),
            [np.log(np.prod(cells[name][2])) for name in names],
            [np.log(cells[name][1]) for name in names],
        ]
    )
    freed, *_ = np.linalg.lstsq(design[resolved], np.log(measured[resolved]), rcond=None)
    return share, share * count, measured, resolved, freed[1:], names


def outlier_census(pairs, cells, ladders, names, resolved, residual):
    """Resolved pairs more than two SD below the law against a normal tail, and the unresolved count.

    Returns:
        Counts: low pairs, those a normal scatter predicts, unresolved pairs; the smallest first bank;
        and the coefficients Ca RMSD and mass take when added to the law over the resolved pairs.
    """
    from scipy.stats import norm

    fitted = [name for name, keep in zip(names, resolved) if keep]
    low = residual < -2 * residual.std()
    design = np.column_stack(
        [
            np.ones(len(fitted)),
            [np.log(np.prod(cells[name][2])) for name in fitted],
            [np.log(cells[name][1]) for name in fitted],
            np.log([pairs[name]["ca_rmsd"] for name in fitted]),
            np.log([pairs[name]["mass_kda"] for name in fitted]),
        ]
    )
    extra, *_ = np.linalg.lstsq(design, np.log([pairs[name]["kc"] for name in fitted]), rcond=None)
    return {
        "low": int(low.sum()),
        "low_expected": round(norm.cdf(-2.0) * len(fitted)),
        "unresolved": int((~resolved).sum()),
        "first_bank": int(min(min(ladders[name]) for name in names)),
        "rmsd_extra": float(extra[3]),
        "mass_extra": float(extra[4]),
    }


def empirical_r2(pairs, pool, names):
    """R^2 of the free fit ln K* ~ ln radius + ln rmsd, the size-and-motion law the derivation replaces."""
    y = np.log([pairs[name]["kc"] for name in names])
    design = np.column_stack(
        [
            np.ones(len(names)),
            np.log([pool[name]["radius"] for name in names]),
            np.log([pairs[name]["ca_rmsd"] for name in names]),
        ]
    )
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(1 - np.var(y - design @ coeffs) / np.var(y))


def reduced(pairs, ladders):
    """Every measurement of every pair as (K/K*, d'^2(K)/d'^2), the coordinates the collapse is drawn in."""
    reach, share = [], []
    for name, ladder in ladders.items():
        row = pairs.get(name)
        if not row or not row.get("kc") or not row.get("d2_plateau"):
            continue
        for k, d2 in ladder.items():
            reach.append(k / row["kc"])
            share.append(d2 / row["d2_plateau"])
    return np.array(reach), np.array(share)


def draw(reach, share, kstar_pred, kstar_meas, resolved, fold, r2):
    """(a) convergence collapse onto the crossover; (b) K* against the derived cell law, drawn as figure 3."""
    import matplotlib.pyplot as plt

    # same figure shape as fig:lawfulness (two square panels), and both panels squared the SAME way
    # (set_box_aspect below, not a mix of box_aspect and aspect="equal") so they come out equal
    figure, (left, right) = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, SINGLE_COLUMN * 1.14))

    # (a) every measurement in reduced coordinates
    left.plot(reach, share, "o", color=SEQUENTIAL, mew=0.0, ms=2.4, alpha=0.30, zorder=2)
    grid = np.geomspace(reach.min(), reach.max(), 240)
    left.plot(grid, master(grid), color=RULE, lw=1.3, zorder=3)
    left.axvline(1.0, color="0.35", lw=0.7, ls=(0, (3, 3)), zorder=1)
    left.axhline(1.0, color="0.35", lw=0.7, ls=(0, (1, 2)), zorder=1)
    left.set_xscale("log")
    left.set_xlabel("Bank size / half-recovery count  $K/K^\\ast$")
    left.set_ylabel("Recovered share  $d'^2(K) / d'^2$")
    left.set_ylim(0, 1.05)

    # (b) K* against the cell law
    span = [min(kstar_pred.min(), kstar_meas.min()) * 0.6, max(kstar_pred.max(), kstar_meas.max()) * 1.6]
    right.fill_between(span, [v / fold for v in span], [v * fold for v in span], color=BAND, lw=0, zorder=1)
    right.plot(span, span, color=RULE, lw=0.8, zorder=2)
    shared_contours(right, kstar_pred[resolved], kstar_meas[resolved], span, zorder=3)
    right.scatter(kstar_pred[~resolved], kstar_meas[~resolved], s=7, color=SERIES[1], linewidths=0.0, zorder=6)
    # a third legend row under the contour shares (0.86, 0.805, ...), its swatch the same orange as the points
    right.scatter(
        [0.07], [0.75], transform=right.transAxes, s=7, color=SERIES[1], linewidths=0.0, clip_on=False, zorder=6
    )
    right.annotate(
        "Below the smallest bank",
        xy=(0.12, 0.75),
        xycoords="axes fraction",
        fontsize=TICK_PT,
        va="center",
        color=SERIES[1],
    )
    right.set(xscale="log", yscale="log", xlim=span, ylim=span)
    right.annotate(
        f"$R^2 = {r2:.3f}$", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=TICK_PT, ha="left", va="top"
    )
    right.annotate(
        f"Geometric SD = {fold:.2f}",
        xy=(0.96, 0.05),
        xycoords="axes fraction",
        fontsize=TICK_PT,
        ha="right",
        va="bottom",
        color="0.45",
    )
    right.set_xlabel("Predicted $K^\\ast$")
    right.set_ylabel("Measured $K^\\ast$")

    for panel in (left, right):
        panel.set_box_aspect(1.0)  # square boxes so (a) matches (b)
    label_panels(left, right)
    figure.tight_layout(pad=PAD, w_pad=PANEL_GAP)
    return figure


def main(argv: Iterable[str] | None = None) -> int:
    """Draw figure 5 and print the convergence, K* and retention numbers for the build ledger."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", type=Path, default=Path("results/eq1_mechanistic.json"))
    parser.add_argument("--survey", default="results/kc_survey/*.json")
    parser.add_argument(
        "--ext",
        default="results/kc_ladder_ext/*.json",
        help="ip_converge bank sizes scored past the survey stop; unioned into each pair's curve",
    )
    parser.add_argument("--pool", default="results/survey_pool/*.json")
    parser.add_argument("--spectrum", type=Path, default=Path("results/difference_spectrum.json"))
    parser.add_argument("--features", type=Path, default=Path("results/pair_features.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/manuscript/figures/bank"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    use_house_style()
    pairs = read_pairs(args.fit)
    # the fit record's admitted pairs, so this figure and equation (1) describe one population
    admitted = {d["name"] for d in payload_of(json.loads(args.fit.read_text()))["converged_deflections"]}
    ladders = {name: ladder for name, ladder in read_ladders(args.survey).items() if name in admitted}
    for name, extra in read_ext(args.ext).items():  # extend a converged pair's curve past the survey stop
        if name in ladders:
            ladders[name].update(extra)
    pool = read_pool(args.pool)
    cells = read_cells(args.spectrum, args.features)
    share, kstar_pred, kstar_meas, resolved, freed, names = kstar_law(pairs, cells, ladders)

    measured = np.log(kstar_meas[resolved])
    residual = measured - np.log(kstar_pred[resolved])
    fold = float(np.exp(np.std(residual)))
    r2 = float(1 - np.sum(residual**2) / np.sum((measured - measured.mean()) ** 2))
    with_radius = [name for name, keep in zip(names, resolved) if keep and pool.get(name, {}).get("radius")]
    classes, overall, n_ret, slope = retention_by_class(pairs, pool)
    reach, recovered = reduced(pairs, ladders)
    collapse_r2 = float(1 - np.sum((recovered - master(reach)) ** 2) / np.sum((recovered - recovered.mean()) ** 2))
    figure = draw(reach, recovered, kstar_pred, kstar_meas, resolved, fold, r2)
    written = save(figure, args.out)

    n_points = sum(len(l) for name, l in ladders.items() if name in pairs)
    print(f"convergence law d'^2(K) = d'^2 / (1 + (K*/K)^{A_EXP:g}), the exponent of a surface-dominated difference")
    print(
        f"  collapse over {len(ladders)} converged pairs, {n_points} bank sizes, R^2 {collapse_r2:.3f}; "
        "the crossover exponent is fixed by theory, not fitted"
    )
    print(
        f"K* = {share:.3f} x SO(3) cells of the difference (n={resolved.sum()} resolved of {len(names)}), "
        "the share the one fitted constant"
    )
    print(f"  R^2 {r2:.3f}, geometric SD {fold:.2f}x")
    print(f"  freed: arm product ^{freed[0]:.2f} (derived 1), k_rms ^{freed[1]:.2f} (derived 3)")
    empirical = empirical_r2(pairs, pool, with_radius)
    print(f"  the free size-and-RMSD law it replaces: R^2 {empirical:.3f} over {len(with_radius)} pairs")
    print(f"converged retention d'^2/d'^2_known: median {overall:.3f} over {n_ret} pairs, scales as d'^2^{slope:.3f}")
    print("  by motion class (>=20 members), least to most retained:")
    for motion, median, count, _ in classes:
        print(f"    {motion:18s} {median:.3f}  (n={count})")
    val = Ledger()
    # the derived exponent checked against each pair's own curve, freed
    freed_p = np.array([p for p in (free_exponent(ladder) for ladder in ladders.values()) if p is not None])
    print(
        f"  exponent freed per pair: median {np.median(freed_p):.3f}, IQR {np.percentile(freed_p, 25):.3f}-"
        f"{np.percentile(freed_p, 75):.3f} over {freed_p.size} pairs, against the derived {A_EXP:g}"
    )
    val("kc_free_p", float(np.median(freed_p)), ".2f")
    val("kc_free_p_lo", float(np.percentile(freed_p, 25)), ".2f")
    val("kc_free_p_hi", float(np.percentile(freed_p, 75)), ".2f")
    val("kc_pairs", len(ladders), ",d")
    val("kc_points", n_points, ",d")
    val("kc_collapse_r2", collapse_r2, ".3f")
    val("kc_law_n", int(resolved.sum()), ",d")
    # pairs on the (a) collapse whose structure was never featured, so (b) cannot predict their K*
    plotted = {name for name in ladders if pairs.get(name, {}).get("kc") and pairs.get(name, {}).get("d2_plateau")}
    val("kc_no_shape", len(plotted - set(names)), ",d")
    val("kc_kstar_decades", float(np.log10(kstar_meas[resolved].max() / kstar_meas[resolved].min())), ".1f")
    val("kc_kstar_min", float(kstar_meas[resolved].min()), ",.0f")
    val("kc_kstar_max", float(kstar_meas[resolved].max()), ",.0f")
    val("kc_kstar_median", float(np.median(kstar_meas[resolved])), ",.0f")
    val("kc_cell_share", share, ".2f")
    val("kc_cell_spacing", share ** (-1 / 3), ".1f")  # bank spacing at half recovery, in decorrelation angles
    # the body states K* in the folded form K* = c * k_rms^3 * prod(rho_i); c absorbs the geometric cell
    # volume 8*pi^2*(2*pi/sqrt3)^3, so it equals kc_cell_share times that factor. Fit it directly here.
    simple = np.array([cells[name][1] ** 3 * float(np.prod(cells[name][2])) for name in names])
    cell_const = float(np.exp(np.mean(np.log(kstar_meas[resolved] / simple[resolved]))))
    val("kc_cell_const", cell_const, ".0f")
    val("kc_arm_exp", freed[0], ".2f")
    val("kc_k_exp", freed[1], ".2f")
    val("kc_empirical_r2", empirical, ".2f")
    census = outlier_census(pairs, cells, ladders, names, resolved, residual)
    print(
        f"  below the law by >2 SD: {census['low']} pairs (a normal scatter puts {census['low_expected']} there); "
        f"{census['unresolved']} more with K* under their first bank (smallest {census['first_bank']}), unfitted"
    )
    print(f"  added to the law, ln Ca RMSD takes {census['rmsd_extra']:+.2f} and ln mass {census['mass_extra']:+.2f}")
    for key in ("low", "low_expected", "unresolved", "first_bank"):
        val(f"kc_{key}", census[key], ",d")
    val("kc_rmsd_extra", abs(census["rmsd_extra"]), ".2f")  # magnitude; prose owns the sign
    val("kc_mass_extra", abs(census["mass_extra"]), ".2f")
    val("kc_r2", r2, ".2f")
    val("kc_residual", fold, ".2f")
    val("kc_retention_n", n_ret, ",d")
    val("kc_retention_exp", abs(slope), ".2f")  # magnitude; prose owns the minus
    val("kc_retention_median", overall, ".3f")
    for motion, median, count, _ in classes:  # per-class retention for the appendix table
        val(f"kc_ret_{motion}", median, ".3f")
        val(f"kc_ret_{motion}_n", count, ",d")
    val.dump(args.out.parent / "values" / f"{args.out.name}.json")
    print("  wrote " + ", ".join(str(path) for path in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
