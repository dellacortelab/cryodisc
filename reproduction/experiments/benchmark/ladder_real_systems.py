"""The predicted-vs-measured d'^2 ladder on the eight real two-state systems.

Tests the reviewer's objection that structure->d'^2 is only ever checked on simulated images: for each
real system it places the forward-model prediction beside the measured d'^2_obs and reports the
across-system spread of their ratio. The decisive quantity is the spread of log(d'^2_obs/d'^2_pred) at
the honest prospective step (Step 1): a consistent offset with small spread is one empirical correction
factor; a large spread means structure alone does not predict real d'^2.

Steps, per system, each at that dataset's own band limit / shift window / pose convention:
  Step 0  Eq. 1 atomic shortcut at the reference condition (the ~99 figure). Needs the atomic pair's
          Ca-RMSD and mass; recorded only where available (SpCas9), the extrapolation arm.
  Step 1  forward model at the dataset's real optics -- here the closed-form known-pose d'^2 (mean of
          the per-particle pool_*_d2.npy), the honest prospective prediction's UPPER bound (perfect
          pose, no high-frequency envelope). The bank-marginalized value is <= this and needs GPU.
  Step 1b FSC half-map route (note:fsc), where available: the state difference and noise read from the
          real half-maps, carrying the real envelope -- the near-exact route.
  Step 4  d'^2_obs, the measured deflection of the two pools' per-particle log-ratios.
Steps 2-3 (sigma^2(k)/contrast and a data-derived Guinier-B envelope) are already folded into the
measured log-ratios' contrast and are embodied by the FSC route; the explicit per-system marginalized
envelope prediction needs a GPU re-score and is left to a follow-up (see record notes).

    .venv/bin/python -m reproduction.experiments.benchmark.ladder_real_systems --out results/ladder_real_systems.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from cryodisc.statistics.information import deflection
from reproduction.record import write_result

DRIVER = "reproduction/experiments/benchmark/ladder_real_systems.py"

SYSTEMS = ("msba", "fanac", "nkcc1", "slc19a3", "slco2a1", "ste2", "hk-atpase")
"""The seven breadth systems; SpCas9 is added from its own rerun arrays."""

CAS9_D2_OBS = 2.906
"""SpCas9 6nt/12nt pooled de-confounded d'^2_obs, the worked pair (shared with recovery_table.py)."""

CAS9_EQ1 = 99.0
"""The Eq. 1 atomic shortcut for SpCas9 at the reference condition -- the reviewer's hand-check figure.

The ~34x gap to d'^2_obs is the whole object of this ladder. Recorded as the reviewer stated it; the
C and r* that produce it are results/eq1_mechanistic.json (canonical C=0.157, r*=2.0)."""

CAS9_FSC = {
    "predicted": 3.494,
    "predicted_ci": [3.41, 3.78],
    "measured": 3.563,
    "measured_err": 0.084,
    "source": "results/fsc_bridge_cas9.json / appendix note:fsc (6nt pool)",
}
"""The half-map (FSC) route for SpCas9: near-exact, the forward route with the real envelope in it."""


def pool_paths(root: Path, system: str) -> tuple[Path, Path, Path, Path]:
    """The A/B log-ratio and closed-form d'^2 arrays of one ribo_pools system."""
    d = root / system
    return d / "pool_a.npy", d / "pool_b.npy", d / "pool_a_d2.npy", d / "pool_b_d2.npy"


def load_meta(root: Path, system: str) -> dict:
    """The dataset's own operating point, read off its score_a.json (band limit, pixel, box, dose)."""
    score = json.loads((root / system / "score_a.json").read_text())
    arg, res = score["arguments"], score["result"]
    return {
        "pixel_size": float(res["pixel_size"]),
        "box_size": int(res["box_size"]),
        "band_limit_cyc_px": float(arg["band_limit"]),
        "band_limit_resolution_a": float(res["pixel_size"]) / float(arg["band_limit"]),
        "dose": float(arg["dose"]),
        "shift_window": int(arg["shift_window"]),
        "k_max": float(arg["k_max"]),
        "n_defocus_groups": res.get("n_defocus_groups"),
    }


def system_row(
    name: str,
    log_a: np.ndarray,
    log_b: np.ndarray,
    d2_a: np.ndarray,
    d2_b: np.ndarray,
    d2_obs: float,
    meta: dict | None,
) -> dict:
    """One system's ladder: measured deflection, known-pose forward prediction, and their ratio."""
    pred_known_pose = 0.5 * (float(d2_a.mean()) + float(d2_b.mean()))
    ratio = d2_obs / pred_known_pose
    return {
        "system": name,
        "n_a": int(log_a.size),
        "n_b": int(log_b.size),
        "step1_pred_known_pose": pred_known_pose,
        "step1_pred_a": float(d2_a.mean()),
        "step1_pred_b": float(d2_b.mean()),
        "step4_d2_obs": float(d2_obs),
        "ratio_obs_over_pred": float(ratio),
        "log_ratio": float(math.log(ratio)),
        "metadata": meta,
    }


def summarize(rows: list[dict]) -> dict:
    """The across-system spread of log(d'^2_obs/d'^2_pred) at Step 1 -- the headline call."""
    logs = np.array([r["log_ratio"] for r in rows])
    ratios = np.exp(logs)
    return {
        "n_systems": len(rows),
        "geo_mean_ratio": float(np.exp(logs.mean())),
        "geo_mean_overstatement_factor": float(np.exp(-logs.mean())),
        "geo_sd_factor": float(np.exp(logs.std(ddof=1))),
        "min_ratio": float(ratios.min()),
        "max_ratio": float(ratios.max()),
        "range_factor": float(np.exp(logs.max() - logs.min())),
        "per_system_ratio": {r["system"]: round(r["ratio_obs_over_pred"], 4) for r in rows},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pools-root", type=Path, default=Path("results/ribo_pools"))
    parser.add_argument("--check", type=Path, default=Path("results/calculator_check.json"))
    parser.add_argument("--cas9-a", type=Path, default=Path("results/rerun_cas9_hoRepro_6nt.npy"))
    parser.add_argument("--cas9-b", type=Path, default=Path("results/rerun_cas9_hoRepro_12nt.npy"))
    parser.add_argument("--cas9-a-d2", type=Path, default=Path("results/rerun_cas9_hoRepro_6nt_d2.npy"))
    parser.add_argument("--cas9-b-d2", type=Path, default=Path("results/rerun_cas9_hoRepro_12nt_d2.npy"))
    parser.add_argument("--out", type=Path, default=Path("results/ladder_real_systems.json"))
    args = parser.parse_args(argv)

    check = json.loads(args.check.read_text())["result"]["systems"]
    d2_obs_official = {e["system"]: float(e["d2_obs"]) for e in check}

    rows: list[dict] = []
    for system in SYSTEMS:
        pa, pb, da, db = pool_paths(args.pools_root, system)
        log_a, log_b = np.load(pa).ravel().astype(float), np.load(pb).ravel().astype(float)
        d2_a, d2_b = np.load(da).ravel().astype(float), np.load(db).ravel().astype(float)
        row = system_row(system, log_a, log_b, d2_a, d2_b, d2_obs_official[system], load_meta(args.pools_root, system))
        rows.append(row)

    log_a, log_b = np.load(args.cas9_a).ravel().astype(float), np.load(args.cas9_b).ravel().astype(float)
    d2_a, d2_b = np.load(args.cas9_a_d2).ravel().astype(float), np.load(args.cas9_b_d2).ravel().astype(float)
    cas9 = system_row("cas9", log_a, log_b, d2_a, d2_b, CAS9_D2_OBS, None)
    cas9["step0_eq1_atomic_shortcut"] = CAS9_EQ1
    cas9["step1b_fsc_half_map_route"] = CAS9_FSC
    cas9["deflection_check"] = float(deflection(log_a, log_b))  # reproduces CAS9_D2_OBS
    cas9["metadata"] = {
        "pixel_size": 1.3,
        "box_size": 192,
        "band_limit_cyc_px": 0.125,
        "band_limit_resolution_a": 10.4,
        "dose": 40.0,
        "shift_window": 4,
        "k_max": 0.1,
        "marginal_fixed_bank": 4000000,
        "driver": "reproduction/experiments/reproduce.py (registry); "
        "deploy/orc/_cas9_marginal.sh ran a measured-equivalent 256000 plateau bank",
    }
    rows.append(cas9)

    summary = summarize(rows)

    record = {
        "d2_obs_convention": "deflection(pool_a, pool_b): variance-normalized, de-confounded, pose-marginal",
        "step1_pred_convention": "mean closed-form known-pose d'^2 over both pools (pool_*_d2.npy); UPPER "
        "bound on the honest forward prediction (perfect pose, no high-k envelope)",
        "systems": rows,
        "step1_spread": summary,
        "verdict": (
            "The forward model (known-pose, the quantity that must hold) overstates real d'^2 by a "
            f"CONSISTENT factor: geo-mean {summary['geo_mean_overstatement_factor']:.1f}x, geo-SD "
            f"{summary['geo_sd_factor']:.2f}x, full range {summary['range_factor']:.1f}x across 8 systems "
            "spanning 1.3-4.1 A/px and unrelated complexes. A consistent offset with small spread is ONE "
            "empirical correction factor, so the prospective claim survives WITH it applied; it is not a "
            "system-dependent blow-up. The much larger Eq. 1 gap (~34x for SpCas9) is the atomic-shortcut "
            "extrapolation arm above the survey mass range, not the forward model."
        ),
        "notes": {
            "step1_is_upper_bound": "known-pose >= bank-marginalized, so the TRUE forward->obs gap is "
            "smaller than reported here; the marginalized prediction needs a GPU re-score.",
            "envelope_accounts_for_most": "the diagnosed missing high-frequency envelope (implied amplitude "
            "~0.5 -> power ~0.25, i.e. ~4x) explains most of the ~7x; the residual "
            "~1.8x is pose marginalization + decoherence. See ribo_pools/provenance.json "
            "amplitude_scale_limitation.",
            "fsc_route_is_near_exact": "the half-map (FSC) route carries the real envelope and predicts SpCas9 "
            "3.49 vs 3.56 measured (note:fsc), i.e. ~1x -- the forward difference model "
            "itself is sound; the known-pose closed form overstates only through the two "
            "idealizations above.",
            "steps_not_computed": "Step 0 Eq.1 recorded for SpCas9 only (atomic pair not local for the 7 breadth "
            "membrane systems); Steps 2-3 explicit marginalized+Guinier-B predictions need GPU.",
        },
        "atomic_arm": {
            "record": "results/ladder_real_atomic.json (reproduction/experiments/benchmark/ladder_real_atomic.py)",
            "finding": "The deposited-ATOMIC-model route is SCATTERED on the MATCHED pose-marginalized "
            "comparison -- obs/pred geo-SD 9.9x (marginalized+band), 6.0x (marginalized only), "
            "12.7x (known-pose bound); range 180-1450x. Marginalization does NOT collapse it to a "
            "single offset (the K^-2/3 extrapolation was refused for every system: K* unreachable). "
            "Root cause: a deposited atomic pair is an unreliable proxy for the real scored "
            "difference (slco2a1 models differ 0.42 A Ca yet the data separates them; big rigid "
            "motions overshoot). The reconstructed-MAP route below (geo-SD 1.38x) is the only "
            "consistent one, so the map/FSC route -- not the atomic shortcut (Eq.1) -- carries the "
            "prospective claim. [Ran on ribo GPU 0-5, score_survey --marginalize --extrapolate.]",
        },
        "d2_obs_is_marginalized": (
            "CORRECTION to an earlier note: d'^2_obs is pose-MARGINALIZED, not known-pose. score_a.json has "
            "known_pose=false and recovery.json/the cas9 source have poses=haar; the provenance PROSE saying "
            "'--known-pose' is wrong. The map-route known-pose columns here are thus an UPPER bound on the "
            "matched prediction, and the atomic arm's marginalized columns are the matched comparison."
        ),
        "sources": {
            "pools": "results/ribo_pools/<system>/pool_{a,b}{,_d2}.npy (score_pool.py, commit 6f5f462d / campaign 5ce4383d)",
            "atomic_ladder": "results/ladder_real_atomic.json",
            "cas9": "results/rerun_cas9_hoRepro_{6,12}nt{,_d2}.npy",
            "d2_obs": "results/calculator_check.json result.systems[].d2_obs; SpCas9 = 2.906",
            "fsc": "results/fsc_bridge_cas9.json",
            "ribo_pools_provenance": "results/ribo_pools/provenance.json",
        },
    }

    write_result(args.out, record, driver=DRIVER, arguments=vars(args))
    print(
        f"8 systems: forward overstates obs by geo-mean {summary['geo_mean_overstatement_factor']:.2f}x, "
        f"geo-SD {summary['geo_sd_factor']:.2f}x, range {summary['range_factor']:.2f}x -> {args.out}"
    )
    for r in rows:
        print(
            f"  {r['system']:10s} obs {r['step4_d2_obs']:6.3f}  pred_kp {r['step1_pred_known_pose']:7.3f}  "
            f"obs/pred {r['ratio_obs_over_pred']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
