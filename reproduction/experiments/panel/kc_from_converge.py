"""Read K_c per pair from ip_converge mapping runs or a kc_survey survey dir, form-free.

    .venv/bin/python reproduction/experiments/panel/kc_from_converge.py results/kc_survey
    .venv/bin/python reproduction/experiments/panel/kc_from_converge.py results/converge_kc results/converge_kc_ext

Reads both output shapes, per file: an ip_converge mapping (`{"rows":[{rungs}]}`, fixed banks) and a
kc_survey per-pair survey record (`{name, curve, status}`, adaptive -- the production format, one file
per pair). In-progress records and scratch files are skipped, so it can be run against a survey still
filling. Pass several dirs to stitch their ladders (a shared bank must be bit-identical, which holds
when the runs share source commit and seeds). Median d^2 over seeds at each bank -- the survey is
single-seed, so the median is that one value unless a multi-seed spot-check dir is also passed. Then
K_c(95) = smallest realized bank reaching 0.95*d^2(top bank), the quantized bank-gap bracket; a pair
still gaining > RISE over its top octave has not plateaued, so its K_c is a lower bound. retention@8064
= d^2(8064)/d^2(top). theta = Ca RMSD/R_g is joined from pair_features.json for the K_c-vs-theta table.
"""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RISE = 0.15  # top-octave d^2 gain above this => still rising at the largest realized bank


def load_curves(result_dirs: list[Path]):
    """key -> {bank: [d2 per seed]}, and key -> pair name, merged across dirs and both file shapes.

    ip_converge files (`{"rows":[...]}`) key by their filename slug; kc_survey per-pair records
    (`{name, curve, status}`) key by pair name. In-progress records and scratch files (leading `_`)
    are skipped, so a survey still filling reads cleanly.
    """
    by_key = defaultdict(lambda: defaultdict(list))
    name_of = {}
    for result_dir in result_dirs:
        for f in sorted(glob.glob(str(result_dir / "*.json"))):
            if Path(f).name.startswith("_"):  # kc_survey's per-process ip_converge scratch file
                continue
            doc = json.loads(Path(f).read_text())
            if "rows" in doc:  # ip_converge mapping output
                slug = Path(f).name.split("ip_converge_")[1].rsplit("_a", 1)[0]
                for row in doc["rows"]:
                    name_of[slug] = row["name"]
                    for rung in row["rungs"]:
                        by_key[slug][rung["k"]].append(rung["d2"])
            elif doc.get("status") in ("converged", "censored"):  # kc_survey per-pair record
                name = doc["name"]
                name_of[name] = name
                for bank, d2 in doc["curve"].items():
                    by_key[name][int(bank)].append(d2)
            # else: in-progress kc_survey record or an unrecognized file -- skip
    return by_key, name_of


def kc_for_pair(bank_d2: dict) -> dict:
    """K_c(95) bracket, rising flag, and retention at 8064 from one pair's per-bank d^2 lists."""
    banks = np.array(sorted(bank_d2))
    med = np.array([np.median(bank_d2[b]) for b in banks])
    i_oct = int(np.argmin(np.abs(banks - banks[-1] / 2)))  # nearest realized half of the top bank
    top_gain = float((med[-1] - med[i_oct]) / max(med[i_oct], 1e-12))
    hit = np.where(med >= 0.95 * med[-1])[0]
    i = int(hit[0]) if len(hit) else len(banks) - 1
    bracket = "<=96" if i == 0 else f"{banks[i - 1]}-{banks[i]}"
    rising = top_gain >= RISE
    ret_8064 = float(med[banks.tolist().index(8064)] / med[-1]) if 8064 in banks else float("nan")
    return {
        "plateau_d2": float(med[-1]),
        "top_bank": int(banks[-1]),
        "top_octave_gain": top_gain,
        "kc": f">{banks[-1]}" if rising else bracket,
        "status": "rising" if rising else "converged",
        "retention_8064": ret_8064,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dirs", type=Path, nargs="+")
    parser.add_argument("--features", type=Path, default=Path("results/pair_features.json"))
    args = parser.parse_args()

    feat = {r["name"]: r for r in json.loads(args.features.read_text())["result"]["rows"]}
    by_key, name_of = load_curves(args.result_dirs)

    rows = []
    for key, bank_d2 in by_key.items():
        info = kc_for_pair(bank_d2)
        fr = feat.get(name_of[key], {})
        theta = fr["ca_rmsd"] / fr["rg"] if fr.get("rg") and fr.get("ca_rmsd") else float("nan")
        rows.append((theta, name_of[key], info))
    rows.sort(key=lambda r: (r[0], r[1]))

    print(f"{'theta':>6}  {'pair':<30}{'plateau':>9}{'top_bank':>9}  {'K_c(95)':>12}{'ret@8064':>9}  status")
    for theta, name, info in rows:
        r8 = "--" if info["retention_8064"] != info["retention_8064"] else f"{info['retention_8064'] * 100:.0f}%"
        print(
            f"{theta:>6.3f}  {name[:30]:<30}{info['plateau_d2']:>9.4f}{info['top_bank']:>9}"
            f"  {info['kc']:>12}{r8:>9}  {info['status']}"
        )
    conv = [r for _, _, r in rows if r["status"] == "converged"]
    if conv and all(r["status"] == "converged" for _, _, r in rows):
        ceiling = max((int(r["kc"].split("-")[-1]) for _, _, r in rows if "-" in r["kc"]), default=96)
        print(f"\nall {len(rows)} converge; universal bank (largest K_c upper bracket): {ceiling}")


if __name__ == "__main__":
    main()
