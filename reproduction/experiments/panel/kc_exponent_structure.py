"""Per-pair d'^2(K) exponent vs structure: is p universal, or a p=(gamma-1)/d family?

Reads the K_c survey records, estimates each converged pair's crossover exponent three ways (global
Hill fit, low-K rise slope, high-K approach slope) plus its crossover scale K*, and correlates them
against per-pair structural variables (rmsd, snr, and any radius/resolution/symmetry found in a panel
file). A universal exponent argues for one derived p on Fig 5; a correlation argues for p=(gamma-1)/d
with structure-dependent smoothness. Diagnostic only -- writes nothing the figure or fit depends on.
"""

import argparse
import glob
import json
import math
import os
import statistics as st
from pathlib import Path


def _load_records(survey_dir):
    """Yield converged survey records (those with a measured plateau A)."""
    for path in glob.glob(os.path.join(survey_dir, "*.json")):
        if "_tmp_" in path:
            continue
        try:
            rec = json.loads(Path(path).read_text())
        except (ValueError, OSError):
            continue
        if rec.get("status") == "converged":
            yield rec


def _curve(rec):
    """Return (banks, d2) sorted by bank size for a record's ladder."""
    curve = rec.get("curve", {})
    banks = sorted((int(k) for k in curve), key=int)
    d2 = [curve[str(k)].get("d2") if isinstance(curve[str(k)], dict) else curve[str(k)] for k in banks]
    return banks, d2


def _loglog_slope(xs, ys):
    """OLS slope of log(ys) on log(xs); None if degenerate."""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((x - mx) ** 2 for x in lx)
    if sxx == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sxx


def _hill_exponent(banks, d2, plateau, p_grid):
    """Best relative-LSQ exponent for A/(1+(K*/K)^p) over a per-pair log K* grid (A,K* nuisances)."""
    lo, hi = min(banks) / 30.0, max(banks) * 2.0
    kstar_grid = [lo * (hi / lo) ** (i / 47) for i in range(48)]
    best = (float("inf"), None)
    for p in p_grid:
        for kstar in kstar_grid:
            u = [(1.0 / (1.0 + (kstar / k) ** p)) / d for k, d in zip(banks, d2) if d > 0]
            su2 = sum(x * x for x in u)
            if su2 <= 0:
                continue
            a = sum(u) / su2
            sse = sum((1 - a * x) ** 2 for x in u)
            if sse < best[0]:
                best = (sse, p)
    return best[1]


def _crossover_bank(banks, d2, plateau):
    """Interpolate the bank K* at which d'^2 first reaches A/2 (the crossover scale)."""
    for (k1, d1), (k2, d2v) in zip(zip(banks, d2), list(zip(banks, d2))[1:]):
        if d1 < plateau / 2 <= d2v:
            frac = (plateau / 2 - d1) / (d2v - d1)
            return k1 * (k2 / k1) ** frac
    return None


def _pair_features(rec, p_grid):
    """Extract the exponent estimates, K*, and structural fields for one converged pair."""
    banks, d2 = _curve(rec)
    plateau = rec.get("d2_plateau") or max(d2)
    rise = [(k, d) for k, d in zip(banks, d2) if 0 < d < 0.4 * plateau]
    appr = [(k, plateau - d) for k, d in zip(banks, d2) if 0.5 * plateau <= d < 0.92 * plateau]
    p_rise = _loglog_slope([k for k, _ in rise], [d for _, d in rise]) if len(rise) >= 3 else None
    s = _loglog_slope([k for k, _ in appr], [d for _, d in appr]) if len(appr) >= 3 else None
    return {
        "name": rec.get("name"),
        "p_hill": _hill_exponent(banks, d2, plateau, p_grid) if len(banks) >= 5 else None,
        "p_rise": p_rise,
        "p_approach": (-s) if s is not None else None,
        "kstar": _crossover_bank(banks, d2, plateau),
        "A": plateau,
        "rmsd": rec.get("rmsd"),
        "snr": rec.get("snr"),
    }


def _merge_panel(feats, panel_path):
    """Attach radius/resolution/symmetry from a panel file when its entries carry them (by name)."""
    if not panel_path or not os.path.exists(panel_path):
        return
    panel = json.loads(Path(panel_path).read_text())
    entries = panel.values() if isinstance(panel, dict) else panel
    meta = {}
    for e in entries if isinstance(entries, list) else list(entries):
        if isinstance(e, dict) and "name" in e:
            meta[e["name"]] = e
    for f in feats:
        m = meta.get(f["name"], {})
        for key in ("radius", "resolution", "symmetry", "n_res"):
            if key in m:
                f[key] = m[key]


def _corr(feats, xkey, ykey, logx=True, logy=False):
    """Pearson r and OLS slope of ykey on xkey over pairs where both are positive/present."""
    pts = [(f[xkey], f[ykey]) for f in feats if f.get(xkey) and f.get(ykey)]
    pts = [(x, y) for x, y in pts if (x > 0 or not logx) and (y > 0 or not logy)]
    if len(pts) < 10:
        return None
    xs = [math.log(x) if logx else x for x, _ in pts]
    ys = [math.log(y) if logy else y for _, y in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx == 0 or syy == 0:
        return None
    return {"n": n, "slope": sxy / sxx, "r": sxy / math.sqrt(sxx * syy)}


def _summary(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    q = lambda f: vals[min(len(vals) - 1, int(f * len(vals)))]
    return {"n": len(vals), "median": st.median(vals), "iqr": (q(0.25), q(0.75))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--survey-dir", default="results/kc_survey")
    ap.add_argument(
        "--panel",
        default="data/survey_canonical.json",
        help="optional; adds radius/resolution/symmetry correlations when present",
    )
    args = ap.parse_args()

    p_grid = [round(0.50 + 0.02 * i, 2) for i in range(51)]
    feats = [_pair_features(rec, p_grid) for rec in _load_records(args.survey_dir)]
    _merge_panel(feats, args.panel)
    print(f"converged pairs: {len(feats)}")

    for key in ("p_hill", "p_rise", "p_approach"):
        s = _summary([f[key] for f in feats])
        if s:
            print(f"{key:11s} n={s['n']:4d}  median={s['median']:.3f}  IQR[{s['iqr'][0]:.3f},{s['iqr'][1]:.3f}]")

    print("\nexponent vs structure (Pearson r on the noted transform; |r| small => p ~ universal):")
    for xkey, logx in (
        ("rmsd", True),
        ("snr", True),
        ("A", True),
        ("kstar", True),
        ("radius", True),
        ("resolution", True),
        ("n_res", True),
    ):
        for ykey in ("p_hill", "p_approach"):
            c = _corr(feats, xkey, ykey, logx=logx, logy=False)
            if c:
                print(
                    f"  {ykey:11s} ~ {'log ' if logx else ''}{xkey:11s}  r={c['r']:+.2f}  slope={c['slope']:+.3f}  n={c['n']}"
                )

    print("\nK* structural law (log-log):")
    for xkey in ("rmsd", "radius", "resolution", "n_res"):
        c = _corr(feats, xkey, "kstar", logx=True, logy=True)
        if c:
            print(f"  log kstar ~ log {xkey:11s}  slope={c['slope']:+.2f}  r={c['r']:+.2f}  n={c['n']}")


if __name__ == "__main__":
    main()
