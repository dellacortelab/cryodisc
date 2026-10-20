"""Figure 3: a mechanistic law predicts `d'^2` across the survey better than Cα RMSD alone.
    .venv/bin/python .../d2_lawfulness.py --fit results/eq1_mechanistic.json --out .../lawfulness
Each point is a pair's pose-marginal converged plateau. Panel (a) draws the frozen one-parameter law --
moving mass times squared displacement, saturating at the resolution; panel (b) the best power law in
Cα RMSD alone, on the same axes. The band is one geometric SD of measured over predicted, not an error
bar, and contours enclose fixed shares of the pairs. Numbers are read from the fit record, never
refitted here. Rationale: docs/rationale/figures.md
"""

import argparse
import json
import statistics
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from reproduction.record import payload_of
from reproduction.reporting.figures import (
    BAND,
    DOUBLE_COLUMN,
    PAD,
    PANEL_GAP,
    RULE,
    SINGLE_COLUMN,
    TICK_PT,
    label_panels,
    save,
    shared_contours,
    use_house_style,
)
from reproduction.reporting.numbers import Ledger

DRIVER = "reproduction/experiments/figures/d2_lawfulness.py"


def measured_at_reference(result: dict) -> np.ndarray:
    """Each pair's plateau d'^2; the survey is collected at the single reference dose, so no rescaling."""
    return np.array([row["d2_plateau"] for row in result["rows"]])


def law_prediction(result: dict) -> np.ndarray:
    """The frozen one-parameter law `C · M · RMSD² / (1 + (RMSD/2)²)` for each pair."""
    block = result["canonical"]
    mass = np.array([row["mass_kda"] for row in result["rows"]])
    rmsd = np.array([row["ca_rmsd"] for row in result["rows"]])
    return block["C"] * mass * rmsd**2 / (1 + (rmsd / block["r_star"]) ** block["q"])


def rmsd_alone_prediction(result: dict) -> np.ndarray:
    """Panel (b)'s abscissa: the best power law in Cα RMSD alone."""
    block = result["singles"]["ca_rmsd"]
    rmsd = np.array([row["ca_rmsd"] for row in result["rows"]])
    return np.exp(block["intercept"]) * rmsd ** block["exp"]


def draw(measured: np.ndarray, result: dict):
    """Measured against predicted `d^2` for the law and for Cα RMSD alone, on one pair of limits."""
    import matplotlib.pyplot as plt

    law, alone = law_prediction(result), rmsd_alone_prediction(result)
    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, SINGLE_COLUMN * 1.14))
    low = min(measured.min(), law.min(), alone.min()) * 0.6
    high = max(measured.max(), law.max(), alone.max()) * 1.6
    span = [low, high]

    panels = (
        (axes[0], law, result["canonical"]["residual_factor"], result["canonical"]["r2"]),
        (axes[1], alone, result["singles"]["ca_rmsd"]["residual_factor"], result["singles"]["ca_rmsd"]["r2"]),
    )
    for panel, predicted, factor, r2 in panels:
        panel.fill_between(span, [v / factor for v in span], [v * factor for v in span], color=BAND, lw=0, zorder=1)
        panel.plot(span, span, color=RULE, lw=0.8, zorder=2)
        shared_contours(panel, predicted, measured, span, zorder=3, bw_method=0.45)
        panel.set(xscale="log", yscale="log", xlim=span, ylim=span, aspect="equal")
        panel.annotate(
            f"$R^2 = {r2:.3f}$", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=TICK_PT, ha="left", va="top"
        )

    axes[0].set_xlabel("Predicted $d'^2$ (mass and C$\\alpha$ RMSD)")
    axes[0].set_ylabel("Measured $d'^2$")
    axes[1].set_xlabel("Predicted $d'^2$ (C$\\alpha$ RMSD alone)")
    axes[1].tick_params(labelleft=False)
    label_panels(axes[0], axes[1])
    figure.tight_layout(pad=PAD, w_pad=PANEL_GAP)
    return figure


def main(argv: Iterable[str] | None = None) -> int:
    """Draw figure 3 and print the fit it drew for the build's number ledger."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", type=Path, default=Path("results/eq1_mechanistic.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/manuscript/figures/lawfulness"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    use_house_style()
    result = payload_of(json.loads(args.fit.read_text()))
    measured = measured_at_reference(result)
    figure = draw(measured, result)
    written = save(figure, args.out)

    canonical, free, alone, power, power2 = (
        result["canonical"],
        result["free_shape"],
        result["singles"]["ca_rmsd"],
        result["power_law"],
        result["power_law_two"],
    )
    masses = [row["mass_kda"] for row in result["rows"]]
    rmsds = [row["ca_rmsd"] for row in result["rows"]]
    heavy = sum(1 for mass in masses if mass >= 100)
    print(f"mechanistic law at reference dose {result['reference_dose']:.0f}, n {result['n_rows']}")
    print(f"  measured d2 spans {measured.min():.2f} to {measured.max():.0f}")
    print(
        f"  law: d2 = {canonical['C']:.2f} * mass * RMSD^2 / (1 + (RMSD/{canonical['r_star']:.0f})^2), "
        f"R2 {canonical['r2']:.3f}, residual factor {canonical['residual_factor']:.2f}x"
    )
    print(
        f"  free shape: r* {free['r_star']:.2f}+-{free['r_star_err']:.2f} A, "
        f"q {free['q']:.2f}+-{free['q_err']:.2f}, mass exponent {result['free_mass']['mass_exp']:.3f}, "
        f"R2 {free['r2']:.3f} against {canonical['r2']:.3f}"
    )
    print(f"  product power law: exponent {power['product_exp']:.2f}, R2 {power['r2']:.3f}")
    print(
        f"  Ca RMSD alone: {np.exp(alone['intercept']):.3f} * RMSD^{alone['exp']:.2f}, R2 {alone['r2']:.3f}, residual factor {alone['residual_factor']:.2f}x"
    )
    print(
        f"  mass alone: R2 {result['singles']['mass_kda']['r2']:.3f}, residual factor {result['singles']['mass_kda']['residual_factor']:.2f}x"
    )
    print(
        f"  the scored rows' masses run {min(masses):.1f} to {max(masses):.1f} kDa, "
        f"median {statistics.median(masses):.1f}, {heavy} at or above 100"
    )
    val = Ledger()
    val("law_C", canonical["C"], ".2f")
    val("law_r_star", canonical["r_star"], ".0f")
    val("law_residual", canonical["residual_factor"], ".2f")
    val("law_r2", canonical["r2"], ".3f")
    val("law_n", result["n_rows"], ",d")
    val("law_converged", result["n_converged"], ",d")
    val("law_min_overlap", 100 * result["min_overlap"], ".0f")  # the admission floor, in percent
    # the floor's sensitivity: the same one-constant form on every converged pair
    val("law_unfloored_r2", result["unfloored"]["r2"], ".3f")
    val("law_unfloored_c_shift", 100 * abs(result["unfloored"]["C"] / result["canonical"]["C"] - 1), ".1f")
    val("law_attempted", result["n_attempted"], ",d")
    val("law_mass_min", min(masses), ".1f")
    val("law_mass_max", max(masses), ".1f")
    val("law_mass_median", float(np.median(masses)), ".1f")
    val("law_rmsd_min", min(rmsds), ".1f")
    val("law_rmsd_max", max(rmsds), ".1f")
    val("law_rmsd_median", float(np.median(rmsds)), ".1f")
    val("law_free_rstar", free["r_star"], ".2f")
    val("law_free_rstar_err", free["r_star_err"], ".2f")
    val("law_q", free["q"], ".2f")
    val("law_q_err", free["q_err"], ".2f")
    val("law_mass_exp", result["free_mass"]["mass_exp"], ".3f")
    val("law_mass_exp_err", result["free_mass"]["mass_exp_err"], ".3f")
    val("law_free_r2", free["r2"], ".3f")
    val("law_r2_gain", free["r2"] - canonical["r2"], ".3f")
    val("law_product_exp", power["product_exp"], ".2f")
    val("law_product_r2", power["r2"], ".3f")
    val("law_two_mass_exp", power2["mass_exp"], ".2f")
    val("law_two_rmsd_exp", power2["rmsd_exp"], ".2f")
    val("law_two_r2", power2["r2"], ".3f")
    val("law_rmsd_residual", alone["residual_factor"], ".2f")
    val("law_rmsd_r2", alone["r2"], ".3f")
    val("law_mass_residual", result["singles"]["mass_kda"]["residual_factor"], ".2f")
    val.dump(args.out.parent / "values" / f"{args.out.name}.json")
    print("  wrote " + ", ".join(str(path) for path in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
