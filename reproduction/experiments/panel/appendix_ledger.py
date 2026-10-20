"""The appendix constants no figure prints, read off the records rather than asserted.
    .venv/bin/python reproduction/experiments/panel/appendix_ledger.py
`check_numbers.py` holds the manuscript to what figure drivers print, which leaves a gap for numbers
the appendix states that no panel needs: the declared collection plan, the defocus quadrature it
implies, the bank quadrature the code runs. This driver draws nothing and only puts them in the
ledger. Nothing is retyped: the plan is read from the `arguments` every scored record carries, one
campaign at a time, and refused if a campaign disagrees with itself; the derived quadrature count is
printed per plan since two campaigns run two plans. The bank quadrature comes from `refine_poses`'
default and `B_conf` from the function the forward model applies.
"""

import argparse
import json
import math
from collections import Counter
from glob import glob
from pathlib import Path

from cryodisc.microscope.plan import CollectionPlan
from cryodisc.microscope.transfer import defocus_groups_for
from cryodisc.resolution.transfer import conformational_b_factor
from cryodisc.rungs.refine import BANK_RADII
from reproduction.operating_point import SHARED
from reproduction.record import write_result
from reproduction.reporting.numbers import Ledger
from reproduction.survey import SURVEY_ARMS

DRIVER = "reproduction/experiments/panel/appendix_ledger.py"

CAMPAIGNS: dict[str, tuple[str, ...]] = {"survey": tuple(SURVEY_ARMS)}
"""The record globs each campaign scored, named as the appendix names them."""

PLAN_KEYS = ("dose", "pixel_size", "rmsf")
"""The `arguments` that fix a campaign's plan, and so its quadrature.

`box_size` is deliberately NOT among them. It is a per-pair quantity, not a plan constant: the box
must hold the molecule, so it is bounded below by `minimum_box_size = 2*ceil(r/p)` and scales with the
radius. `box_sizes` reports the distribution rather than a single value, so a campaign that spans
several boxes is described honestly instead of collapsed to one."""


def box_sizes(patterns: tuple[str, ...]) -> dict[str, int]:
    """What boxes a campaign actually ran at.

    Args:
        patterns: Globs over the campaign's records.

    Returns:
        `{"px": n}` when every record shares one box, else `{"mode_px", "n_distinct", "min_px",
        "max_px", "mode_share_pct"}`, so a reader can see whether a single number is honest.
    """
    seen: Counter[int] = Counter()
    for path in sorted({found for pattern in patterns for found in glob(pattern)}):
        arguments = json.loads(Path(path).read_text()).get("arguments", {})
        if isinstance(arguments.get("box_size"), int):
            seen[arguments["box_size"]] += 1
    if not seen:
        return {}
    if len(seen) == 1:
        return {"px": next(iter(seen))}
    (mode, hits), total = seen.most_common(1)[0], sum(seen.values())
    return {
        "mode_px": mode,
        "n_distinct": len(seen),
        "min_px": min(seen),
        "max_px": max(seen),
        "mode_share_pct": round(100.0 * hits / total, 1),
    }


def recorded_plan(patterns: tuple[str, ...]) -> tuple[dict[str, float], int]:
    """The one plan every record a campaign wrote was run at.

    Args:
        patterns: Globs over the campaign's records.

    Returns:
        `({dose, pixel_size, rmsf}, n_records)`.

    Raises:
        SystemExit: If the globs match nothing, or if two records disagree on a plan key -- a
            campaign with two plans has no single quadrature to quote.
    """
    plans: dict[tuple[float, ...], str] = {}
    paths = sorted({path for pattern in patterns for path in glob(pattern)})
    scored = 0
    for path in paths:
        arguments = json.loads(Path(path).read_text()).get("arguments", {})
        # A pool glob also matches per-shard job-status records (returncode/dropped, no plan); those
        # are not scored pairs, so they carry no plan key and are skipped rather than crashing here.
        if not all(key in arguments for key in PLAN_KEYS):
            continue
        plan = tuple(float(arguments[key]) for key in PLAN_KEYS)
        plans.setdefault(plan, path)
        scored += 1
    if not plans:
        raise SystemExit(f"{list(patterns)} matched no scored record")
    if len(plans) > 1:
        raise SystemExit(f"{list(patterns)} ran more than one plan: {plans}")
    (plan,) = plans
    return dict(zip(PLAN_KEYS, plan)), scored


def main(argv: list[str] | None = None) -> int:
    """Print every appendix constant and record it.

    Args:
        argv: Command line, or None for `sys.argv`.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/appendix_ledger.json"))
    args = parser.parse_args(argv)

    # the appendix quotes u, D and B_conf from the operating point that DROVE the experiments, not from
    # a record it re-reads: reproduction.operating_point.SHARED is the one source the reproduce.py registry
    # launches from, so the manuscript and the runs cannot disagree.
    val = Ledger()
    val("op_rmsf", SHARED.rmsf, "")
    val("op_dose", SHARED.dose, ".0f")
    val("op_b_conf", conformational_b_factor(SHARED.rmsf), ".1f")

    campaigns = {}
    print("forward-model constants, read off each campaign's records")
    for name, patterns in CAMPAIGNS.items():
        recorded, count = recorded_plan(patterns)
        plan = CollectionPlan(dose=recorded["dose"], pixel_size=recorded["pixel_size"])
        groups = int(defocus_groups_for(plan))
        b_conf = float(conformational_b_factor(recorded["rmsf"]))
        print(
            f"  {name} ({count:,} records): dose {plan.dose:.0f} e/A^2   pixel {plan.pixel_size:.2f} A   Nyquist {plan.nyquist:.4f} 1/A"
        )
        print(f"    defocus window {plan.defocus_range[0]:,.0f} to {plan.defocus_range[1]:,.0f} A (a plan, uniform)")
        print(
            f"    voltage {plan.voltage_in_kv:.0f} kV   Cs {plan.spherical_aberration_in_mm:.1f} mm   amplitude contrast {plan.amplitude_contrast_ratio:.2f}"
        )
        print(f"    defocus quadrature: {groups} nodes, derived from this plan and the phase tolerance")
        print(f"    u = {recorded['rmsf']} A, so B_conf = 8 pi^2 u^2 = {b_conf:.1f} A^2")
        boxes = box_sizes(patterns)
        if name == "survey":  # the campaign whose quadrature and box the appendix cites
            val("op_quad_nodes", groups, "")
            val("survey_attempted", count, ",d")  # scored survey records -- the pairs the survey attempted
            if "px" in boxes:
                val("op_box", boxes["px"], "")
        if "px" in boxes:
            print(f"    box {boxes['px']} px, the same for every record")
        elif boxes:
            print(
                f"    box is per-pair: {boxes['n_distinct']} distinct, {boxes['min_px']} to "
                f"{boxes['max_px']} px, mode {boxes['mode_px']} on {boxes['mode_share_pct']}% of records"
            )
        campaigns[name] = {
            "n_records": count,
            "dose_e_per_a2": plan.dose,
            "pixel_size_a": plan.pixel_size,
            "defocus_range_a": list(plan.defocus_range),
            "voltage_kv": plan.voltage_in_kv,
            "spherical_aberration_mm": plan.spherical_aberration_in_mm,
            "amplitude_contrast": plan.amplitude_contrast_ratio,
            "defocus_quadrature_nodes": groups,
            "rmsf_a": recorded["rmsf"],
            "b_conf_a2": b_conf,
            "box": boxes,
        }
    print(f"  bank quadrature: {BANK_RADII} radii, the default every rung ran with")

    # the appendix's u = 1.0 A alternative and the 5 A amplitude each B leaves (exp(-B k^2 / 4), k = 1/5),
    # so the "matters little" sentence fills from the same forward model rather than three stale literals
    k = 1.0 / 5.0
    b_full = conformational_b_factor(1.0)
    val("op_b_conf_1a", b_full, ".1f")
    val("op_amp_half", math.exp(-conformational_b_factor(SHARED.rmsf) * k**2 / 4), ".3f")
    val("op_amp_full", math.exp(-b_full * k**2 / 4), ".3f")
    val.dump(Path("docs/manuscript/figures/values/appendix_ledger.json"))

    write_result(
        args.out,
        {"campaigns": campaigns, "bank_quadrature_radii": BANK_RADII},
        driver=DRIVER,
        arguments=vars(args),
    )
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
