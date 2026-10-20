"""Summarize how much of the survey's d'^2 comes from beyond the projection-approximation limit.

    .venv/bin/python .../survey_band_share.py --spectrum results/difference_spectrum.json

The survey integrates d'^2 to Nyquist, so some signal sits finer than the 5 A limit the projection
approximation is valid to [@vulovic2014]. This reads the stored per-shell difference power and the
per-shell weight (difference_spectrum.py output, which reproduces each pair's d2_known_pose) and
reports the fraction of d'^2 above that limit -- per pair and for the high-d'^2 pairs that set the
magnitude. No re-rendering: the share is a resum of numbers already recorded.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from reproduction.record import write_result

DRIVER = "reproduction/experiments/benchmark/survey_band_share.py"


def shares(spectrum: dict, limit_resolution: float) -> dict:
    """Per-pair >limit share of d'^2, and the high-d'^2 subset, from a difference_spectrum record."""
    result = spectrum["result"]
    arguments = spectrum["arguments"]
    pixel = float(arguments["pixel_size"])
    freq = np.asarray(result["frequency_cyc_px"])  # cyc/px per shell
    weight = np.asarray(result["shell_weight"])  # 1/sigma^2 per shell, the survey's own noise side
    k = freq / pixel  # 1/A
    above = k >= 1.0 / limit_resolution

    totals, highs = [], []
    for row in result["rows"]:
        contribution = np.asarray(row["defocus"][0]["power"]) * weight
        total = float(contribution.sum())
        if total <= 0:
            continue
        totals.append(total)
        highs.append(float(contribution[above].sum()))
    totals, highs = np.asarray(totals), np.asarray(highs)
    share = highs / totals
    return {"total": totals, "share": share, "pixel": pixel, "nyquist_inv_a": float(0.5 / pixel)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spectrum", type=Path, default=Path("results/difference_spectrum.json"))
    parser.add_argument("--out", type=Path, default=Path("results/survey_band_share.json"))
    parser.add_argument(
        "--limit-resolution",
        type=float,
        default=5.0,
        help="the projection approximation's validity limit in A; d'^2 finer than this is the residual",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="how many of the highest-d'^2 pairs define the law-anchoring subset the prose cites as <1%%",
    )
    args = parser.parse_args(argv)

    spectrum = json.loads(args.spectrum.read_text())
    computed = shares(spectrum, args.limit_resolution)
    share, total = computed["share"], computed["total"]
    top = np.argsort(total)[::-1][: args.top_n]
    arguments = spectrum["arguments"]

    record = {
        "limit_resolution_a": args.limit_resolution,
        "limit_frequency_inv_a": 1.0 / args.limit_resolution,
        "n_pairs": int(share.size),
        "units": "every *_share_* field is a percent of d2_known_pose",
        # per-pair distribution of the >5 A share
        "median_share_above_5A_pct": float(np.median(share) * 100),
        "mean_share_above_5A_pct": float(share.mean() * 100),
        "p90_share_above_5A_pct": float(np.percentile(share, 90) * 100),
        "p99_share_above_5A_pct": float(np.percentile(share, 99) * 100),
        "max_share_above_5A_pct": float(share.max() * 100),
        # how much of the WHOLE survey's d'^2 is above the limit (each pair weighted by its d'^2)
        "d2_weighted_share_above_5A_pct": float(highs_total(computed) * 100),
        # the pairs that set the magnitude and anchor the law: their share is the <1% figure
        "high_d2_set": f"the {args.top_n} highest d2_known_pose pairs",
        "high_d2_set_size": int(args.top_n),
        "high_d2_min_d2": float(total[top].min()),
        "high_d2_share_above_5A_pct": float(share[top].max() * 100),  # the worst among them, i.e. still <1%
        "high_d2_median_share_above_5A_pct": float(np.median(share[top]) * 100),
        "operating_point": {
            "pixel_size": float(arguments["pixel_size"]),
            "box_size": int(arguments["box_size"]),
            "dose": float(arguments["dose"]),
            "rmsf": float(arguments["rmsf"]),
            "defocus_a": list(arguments["defocus"]),
            "poses": int(arguments["poses"]),
            "nyquist_inv_a": computed["nyquist_inv_a"],
        },
        "source": str(args.spectrum),
        "source_driver": spectrum.get("driver"),
        "source_commit": spectrum.get("commit"),
    }
    write_result(args.out, record, driver=DRIVER, arguments=vars(args))
    print(
        f"{record['n_pairs']} pairs: >{args.limit_resolution:g} A share "
        f"median {record['median_share_above_5A_pct']:.2f}%  max {record['max_share_above_5A_pct']:.2f}%  "
        f"high-d2 (top {args.top_n}) max {record['high_d2_share_above_5A_pct']:.2f}%  "
        f"d2-weighted {record['d2_weighted_share_above_5A_pct']:.2f}% -> {args.out}"
    )
    return 0


def highs_total(computed: dict) -> float:
    """The d'^2-weighted share: total d'^2 above the limit over total d'^2, across all pairs."""
    share, total = computed["share"], computed["total"]
    return float((share * total).sum() / total.sum())


if __name__ == "__main__":
    raise SystemExit(main())
