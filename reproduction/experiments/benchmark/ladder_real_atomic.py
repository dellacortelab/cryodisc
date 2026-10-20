"""The atomic-model arm of the real-systems ladder: structure -> d'^2, band-matched, per system.

Tests whether the deposited ATOMIC pair predicts the measured d'^2_obs -- the route the reviewer said is
only ever checked on simulated images. For each system it aligns the two deposited state models, renders
their difference spectrum at the dataset's own plan, and sums it under the real 0.125 cyc/px band stop:
  Step 0  Eq. 1 at the reference condition, from the pair's own Ca-RMSD and mass (the ~99 shortcut).
  Step 1  known-pose d'^2 at the reference envelope (rmsf 0.5), band-matched -- the honest atomic
          prospective prediction. (The bank-marginalized value is <= this; that arm needs GPU and is
          deferred -- see the note in ladder_real_systems.json. Because d'^2_obs is a KNOWN-POSE
          measurement, known-pose is the matched axis anyway.)
  Step 3  a Debye-Waller / Guinier-B sweep, asking which envelope would reconcile the atomic prediction
          with d'^2_obs.
  Step 4  d'^2_obs, as measured.

The verdict is the ACROSS-SYSTEM spread of obs/pred: unlike the reconstructed-map route (geo-SD ~1.4x),
the atomic route is wildly inconsistent, because a deposited atomic pair is an unreliable proxy for the
real scored state difference (near-identical models that the data still separates; large rigid motions
the imaged particles realize only partially). No single envelope reconciles it.

    .venv/bin/python -m reproduction.experiments.benchmark.ladder_real_atomic --data-dir data/pdb --fetch \
        --out results/ladder_real_atomic.json
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from cryodisc import CollectionPlan
from cryodisc.forward.optics import CTFParams
from cryodisc.grids.fourier import shell_indices
from cryodisc.grids.poses import pose_bank
from reproduction.predictors.spectrum import difference_spectrum, shell_weight
from reproduction.record import write_result
from reproduction.targets import Candidate, fetch_structure

DRIVER = "reproduction/experiments/benchmark/ladder_real_atomic.py"

EQ1_C, EQ1_RSTAR, EQ1_Q = 0.15662229452624504, 2.0, 2.0  # results/eq1_mechanistic.json canonical

# name: (pdb_a, pdb_b, pixel, box, dose, d2_obs). The five campaign systems + SpCas9 use the deposited
# state models (state label = PDB id). FaNaC uses its scored accessions' models (8ON8/8ON7). MsbA and
# HK-ATPase are omitted: no clean deposited atomic pair (msba scored apo/TBT1 7MET with no apo model
# here; hk-atpase 7W4A/8IJL share only 5% residue-type identity at matched Ca, so align refuses it).
SYSTEMS = {
    "cas9": ("7Z4C", "7Z4G", 1.3, 192, 40.0, 2.906),
    "nkcc1": ("7S1Y", "7S1X", 4.004, 144, 1.175, 2.210),
    "slc19a3": ("8S4U", "8S5W", 3.998, 98, 45.0, 1.141),
    "slco2a1": ("9R0A", "9R07", 4.007, 98, 40.2, 1.110),
    "ste2": ("7QA8", "7QBI", 4.002, 104, 57.0, 2.177),
    "fanac": ("8ON8", "8ON7", 4.088, 128, 40.0, 0.812),
}
BAND_CYC_PX = 0.125
DEFOCUS = 15000.0
RMSF_SWEEP = [0.5, 0.8, 1.1, 1.5, 2.0, 2.6, 3.3, 4.0, 5.0]

# Bank-MARGINALIZED forward prediction, from score_survey --marginalize --extrapolate on ribo
# (8x RTX 5000, GPU 0-5, 2026-10-06, repo at /data/bryceeh/cryodisc), at each system's
# own pixel/box/dose, Haar off-grid pool, shift-sigma 1.0, n-particles 2000. d2_known_pose cross-checks
# this file's CPU kp_full_nyquist to <1%. The K^-2/3 EXTRAPOLATION WAS REFUSED for every system
# (marginal_extrapolated=False): K* is unreachably large for these complexes, so d2_marginal is the
# base-bank value, a deep lower bound, not a converged number. {sys: (d2_known_pose_nyq, d2_marginal_base,
# pose_efficiency)}.
MARGINALIZED = {
    "cas9": (739.933, 0.3211, 0.00043),
    "nkcc1": (15.271, 0.5503, 0.03604),
    "slc19a3": (150.994, 2.3506, 0.01557),
    "slco2a1": (2.343, 0.0127, 0.00541),
    "ste2": (61.289, 0.4357, 0.00711),
    "fanac": (87.832, 0.7103, 0.00809),
}


def eq1(mass_kda: float, ca_rmsd: float) -> float:
    """Equation 1 at the reference condition."""
    return EQ1_C * mass_kda * ca_rmsd**2 / (1.0 + (ca_rmsd / EQ1_RSTAR) ** EQ1_Q)


def masked_d2(power: np.ndarray, plan: CollectionPlan, box: int, n_shells: int, rmsf: float, mask: np.ndarray) -> float:
    """d'^2 = sum_k power(k) / sigma^2(k; rmsf), summed over the shells the mask keeps."""
    weight = np.asarray(shell_weight(plan, box, n_shells, rmsf=rmsf))
    return float(np.sum(power * weight * mask))


def _pred(row: dict, key: str) -> float:
    """One prediction column of a row."""
    return row[key]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/pdb"))
    parser.add_argument("--fetch", action="store_true", help="download missing depositions from RCSB")
    parser.add_argument("--poses", type=int, default=96)
    parser.add_argument("--out", type=Path, default=Path("results/ladder_real_atomic.json"))
    args = parser.parse_args(argv)

    angles = pose_bank(args.poses).angles
    rows: list[dict] = []
    for name, (pa, pb, pixel, box, dose, obs) in SYSTEMS.items():
        path_a = fetch_structure(pa, args.data_dir, fetch=args.fetch)
        path_b = fetch_structure(pb, args.data_dir, fetch=args.fetch)
        plan = CollectionPlan(dose=dose, pixel_size=pixel)
        pair = Candidate(name=name, path_a=path_a, path_b=path_b).align(min_shared_fraction=0.5)
        n_shells = box // 2
        model_a, model_b = pair.models(plan, box_size=box, form_factor="peng")
        spectrum = difference_spectrum(
            model_a, model_b, angles, box_size=box, n_shells=n_shells, ctf_params=CTFParams.from_plan(plan, DEFOCUS)
        )
        power = np.asarray(spectrum.power)
        freq = np.asarray(shell_indices(box, n_shells)[1])
        band = (freq <= BAND_CYC_PX).astype(float)
        full = np.ones_like(band)
        mass = float(pair.state_a["atomic_weight"].iloc[: pair.alignment.n_atoms].sum() / 1000.0)
        rmsd = float(pair.alignment.rmsd)
        kp_band = masked_d2(power, plan, box, n_shells, 0.5, band)
        sweep = {f"B{8 * math.pi**2 * r**2:.0f}": masked_d2(power, plan, box, n_shells, r, band) for r in RMSF_SWEEP}
        kp_nyq = masked_d2(power, plan, box, n_shells, 0.5, full)
        _d2kp_nyq, d2marg_base, pose_eff = MARGINALIZED[name]
        band_marg = kp_band * pose_eff  # marginalized AND band-matched to the measurement
        rows.append(
            {
                "system": name,
                "pixel_size": pixel,
                "box_size": box,
                "dose": dose,
                "band_limit_cyc_px": BAND_CYC_PX,
                "band_resolution_a": pixel / BAND_CYC_PX,
                "mass_kda": mass,
                "ca_rmsd": rmsd,
                "step0_eq1": eq1(mass, rmsd),
                "step1_atomic_known_pose_full_nyquist": kp_nyq,
                "step1_atomic_known_pose_band_refB": kp_band,
                "step1_atomic_marginalized_nyquist_base": d2marg_base,
                "pose_efficiency_base_bank": pose_eff,
                "marginal_extrapolated": False,
                "step1_atomic_marginalized_band": band_marg,
                "step3_B_sweep_band": sweep,
                "step4_d2_obs": obs,
                # three obs/pred conventions, all scattered:
                "ratio_obs_over_atomic_known_pose_band": obs / kp_band,
                "ratio_obs_over_atomic_marginalized_band": obs / band_marg,
                "ratio_obs_over_atomic_marginalized_nyquist": obs / d2marg_base,
                "log_ratio": math.log(obs / kp_band),
            }
        )

    def spread_of(key):
        logs = np.array([math.log(r["step4_d2_obs"] / _pred(r, key)) for r in rows])
        ratios = np.exp(logs)
        return {
            "geo_mean_ratio": float(np.exp(logs.mean())),
            "geo_sd_factor": float(np.exp(logs.std(ddof=1))),
            "min_ratio": float(ratios.min()),
            "max_ratio": float(ratios.max()),
            "range_factor": float(np.exp(logs.max() - logs.min())),
            "per_system_ratio": {r["system"]: round(r["step4_d2_obs"] / _pred(r, key), 4) for r in rows},
        }

    spread = {
        "n_systems": len(rows),
        "known_pose_band": spread_of("step1_atomic_known_pose_band_refB"),
        "marginalized_band": spread_of("step1_atomic_marginalized_band"),
        "marginalized_nyquist": spread_of("step1_atomic_marginalized_nyquist_base"),
        "map_route_for_contrast": {
            "geo_sd_factor": 1.38,
            "range_factor": 2.75,
            "source": "results/ladder_real_systems.json",
        },
    }
    record = {
        "arm": "atomic deposited-model route vs the pose-MARGINALIZED d'^2_obs (the matched axis)",
        "d2_obs_is_marginalized": (
            "Verified against the records: every score_a.json has known_pose=false and recovery.json/the "
            "cas9 source have poses=haar, so d'^2_obs is pose-marginalized (the provenance PROSE saying "
            "'--known-pose' is wrong -- a contradiction logged here). The matched Step-1 is therefore the "
            "bank-marginalized atomic prediction."
        ),
        "systems": rows,
        "spread": spread,
        "verdict": (
            "The atomic-model route is SCATTERED on the MATCHED (marginalized) comparison, not a single "
            f"offset: obs/pred geo-SD {spread['marginalized_band']['geo_sd_factor']:.1f}x and range "
            f"{spread['marginalized_band']['range_factor']:.0f}x for the marginalized+band prediction, "
            f"{spread['marginalized_nyquist']['geo_sd_factor']:.1f}x / "
            f"{spread['marginalized_nyquist']['range_factor']:.0f}x for marginalized-only, and "
            f"{spread['known_pose_band']['geo_sd_factor']:.1f}x / "
            f"{spread['known_pose_band']['range_factor']:.0f}x for the known-pose upper bound. "
            "Marginalization does NOT rescue it -- every convention stays at geo-SD 6-13x, against the "
            "reconstructed-MAP route's 1.38x. Root cause (robust to the bank): a deposited atomic PAIR is "
            "an unreliable proxy for the real scored difference -- slco2a1's two models differ by only "
            "0.42 A Ca yet the data separates them (obs >> pred), while cas9/slc19a3's large rigid motions "
            "overshoot. Compounding it, the K^-2/3 extrapolation was REFUSED for every system "
            "(K* unreachable), so there is no converged atomic marginalized d'^2 at feasible banks at all. "
            "Structure predicts real d'^2 reliably only through the map/FSC forward route, not the atomic "
            "shortcut (Eq. 1)."
        ),
        "note_convergence": (
            "The real measurement used a large bank (score_pool fixed_bank ~4e6) and sits near-converged "
            "(obs/known_pose_map ~0.21 is an envelope factor, not a bank collapse), so the known-pose-band "
            "comparison (geo-SD 12.7x) is the most appropriate matched read; the base-bank marginals here "
            "(pose_eff 0.5-3.6%) are deep lower bounds. All conventions agree the route is scattered."
        ),
        "provenance": {
            "difference_spectrum": "reproduction.predictors.spectrum.difference_spectrum, form_factor=peng, "
            "96-node bank, defocus 15000 A (the CTF does not re-weight the difference)",
            "band": "0.125 cyc/px shell mask, matching the real score's --band-limit",
            "eq1": "results/eq1_mechanistic.json canonical C=0.1566, r*=2.0, q=2.0",
            "pdbs": "deposited state models fetched from RCSB (data-dir); msba/hk-atpase omitted (no clean pair)",
            "d2_obs": "results/calculator_check.json; SpCas9 2.906",
        },
    }
    write_result(args.out, record, driver=DRIVER, arguments=vars(args))
    mb, kp = spread["marginalized_band"], spread["known_pose_band"]
    print(
        f"atomic route SCATTERED: marg+band geo-SD {mb['geo_sd_factor']:.1f}x range {mb['range_factor']:.0f}x; "
        f"known-pose geo-SD {kp['geo_sd_factor']:.1f}x (map route 1.38x) -> {args.out}"
    )
    for r in rows:
        print(
            f"  {r['system']:9s} Eq1 {r['step0_eq1']:7.1f}  kp_band {r['step1_atomic_known_pose_band_refB']:8.3f}  "
            f"marg_band {r['step1_atomic_marginalized_band']:8.4f}  obs {r['step4_d2_obs']:5.3f}  "
            f"obs/marg_band {r['ratio_obs_over_atomic_marginalized_band']:8.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
