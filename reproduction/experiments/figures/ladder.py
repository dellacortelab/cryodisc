"""Supplementary numbers: the structure-only prediction of d'^2 tested against the real measurement.

    .venv/bin/python reproduction/experiments/figures/ladder.py

Draws nothing, like `recovery_table.py`. It reads `results/ladder_real_atomic.json` -- the atomic
deposited-model prediction of each real system's d'^2 against the pose-marginalized d'^2_obs -- and emits
the summary tokens the Results subsection and Supplementary table quote. It also reads
`results/ladder_real_systems.json` for the per-system reconstructed-map route (the consistent-offset
route whose scatter is the 1.38x), one token per system for the Supplementary Note table. The claim is convention-robust: the
obs/pred scatter is a geometric standard deviation of 6--13x (range over 1000x) across every pose
convention, against the reconstructed-map route's 1.38x, and its sign is inconsistent within one convention
(known pose over-predicts SpCas9's 6.5 A motion yet under-predicts SLCO2A1's 0.42 A pair). Rationale:
docs/rationale/figures.md
"""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from reproduction.record import payload_of
from reproduction.reporting.numbers import Ledger

DRIVER = "reproduction/experiments/figures/ladder.py"


def main(argv: Iterable[str] | None = None) -> int:
    """Record the ladder's summary and two illustrative systems off the result, print, dump. Draws nothing."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", type=Path, default=Path("results/ladder_real_atomic.json"))
    parser.add_argument("--map-systems", type=Path, default=Path("results/ladder_real_systems.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/manuscript/figures/ladder"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = payload_of(json.loads(args.result.read_text()))
    spread = result["spread"]
    systems = {entry["system"]: entry for entry in result["systems"]}

    val = Ledger()
    print("structure-only prediction of d'^2 against the real measurement, off ladder_real_atomic.json")

    # The scatter magnitude is the convention-robust claim: the geometric SD of obs/pred runs across the
    # three pose conventions, and every one of them dwarfs the map route's 1.38x.
    gsds = [
        spread["marginalized_nyquist"]["geo_sd_factor"],
        spread["marginalized_band"]["geo_sd_factor"],
        spread["known_pose_band"]["geo_sd_factor"],
    ]
    print("  n_systems", val("ladder_n", spread["n_systems"], "d"))
    print("  atomic obs/pred geo-SD lo", val("ladder_atomic_gsd_lo", min(gsds), ".0f"))
    print("  atomic obs/pred geo-SD hi", val("ladder_atomic_gsd_hi", max(gsds), ".0f"))
    print("  atomic obs/pred range hi", val("ladder_atomic_range_hi", spread["known_pose_band"]["range_factor"], ".0f"))
    print("  map-route geo-SD", val("ladder_map_gsd", spread["map_route_for_contrast"]["geo_sd_factor"], ".2f"))

    # Inconsistent sign within one convention (known pose): a large rigid motion over-predicted, a near-zero
    # Ca motion under-predicted, so no single rescaling aligns them. ratio = d'^2_obs / atomic_known_pose.
    cas9, slco2a1 = systems["cas9"], systems["slco2a1"]
    print("  SpCas9 Ca RMSD", val("ladder_cas9_rmsd", cas9["ca_rmsd"], ".1f"))
    print(
        "  SpCas9 known-pose overshoot",
        val("ladder_cas9_overshoot_kp", 1.0 / cas9["ratio_obs_over_atomic_known_pose_band"], ".0f"),
    )
    print("  SLCO2A1 Ca RMSD", val("ladder_slco2a1_rmsd", slco2a1["ca_rmsd"], ".2f"))
    print("  SLCO2A1 d'^2_obs", val("ladder_slco2a1_d2obs", slco2a1["step4_d2_obs"], ".2f"))
    print(
        "  SLCO2A1 known-pose undershoot",
        val("ladder_slco2a1_undershoot_kp", slco2a1["ratio_obs_over_atomic_known_pose_band"], ".1f"),
    )

    # The per-system map route (prediction from the two reconstructed pure-pool maps vs the measured
    # d'^2_obs), the consistent-offset route whose scatter is the 1.38x above. One row per system so the
    # Supplementary Note [[note:fsc]] table fills from tokens, not retyped literals.
    map_result = payload_of(json.loads(args.map_systems.read_text()))
    map_systems = {entry["system"]: entry for entry in map_result["systems"]}
    print("  map-route per system (pred known-pose vs measured d'^2_obs), off ladder_real_systems.json")
    for name in ("msba", "fanac", "nkcc1", "slc19a3", "slco2a1", "ste2", "hk-atpase", "cas9"):
        entry = map_systems[name]
        key = name.replace("-", "")
        pred = val(f"ladder_map_{key}_pred", entry["step1_pred_known_pose"], ".1f")
        obs = val(f"ladder_map_{key}_obs", entry["step4_d2_obs"], ".2f")
        ratio = val(f"ladder_map_{key}_ratio", entry["ratio_obs_over_pred"], ".3f")
        print(f"    {name:10s} pred {pred}  obs {obs}  ratio {ratio}")
    print("  map-route obs/pred range", val("ladder_map_range", map_result["step1_spread"]["range_factor"], ".2f"))

    args.out.parent.joinpath("values").mkdir(parents=True, exist_ok=True)
    val.dump(args.out.parent / "values" / f"{args.out.name}.json")
    print(f"  wrote {args.out.parent / 'values' / (args.out.name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
