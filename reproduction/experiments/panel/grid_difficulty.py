"""Test whether D/d, a geometric scalar, predicts how hard a pair's pose bank is to converge.

    .venv/bin/python reproduction/experiments/panel/grid_difficulty.py \
        --spectrum results/diff_spectrum_fork.json --converge results/fork_deflection \
        --local results/ip_local_p1.json results/ip_local_p2.json

D = 2 * radius (molecular extent, A) and d = <k^2>^(-1/2) with <k^2> weighted by the whitened,
CTF-modulated difference power (the angular Nyquist scale of the difference density). D/d is the
geometry of grid difficulty, contrast-invariant. Against it: the top-octave d^2 gain from the
ip_converge ladders (still-rising = under-converged at the top bank) and rho_global = I_measured/I_local.
Reads a difference_spectrum record for the power and shell weight; no new render. Companion:
docs/planning/figure5_kc_convergence.md.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def load_gain_rho(converge_dir: Path, local_files: list[Path]) -> tuple[dict, dict]:
    """name -> top-octave d^2 gain, and name -> rho_global, from the a=1 ladders and ip_local."""
    i_local = {}
    for f in local_files:
        for row in json.loads(f.read_text())["rows"]:
            one = [a for a in row["amplitudes"] if a["amplitude"] == 1.0]
            if one:
                i_local[row["name"]] = one[0]["i_local"]
    gain, rho = {}, {}
    for f in glob.glob(str(converge_dir / "*_a1.0.json")):
        row = json.loads(Path(f).read_text())["rows"][0]
        rungs = row["rungs"]
        prev = rungs[-2]["d2"] if len(rungs) > 1 else 0.0  # single-bank ladder: unassessable, read as rising
        gain[row["name"]] = (rungs[-1]["d2"] - prev) / max(rungs[-1]["d2"], 1e-12)
        if row["name"] in i_local:
            rho[row["name"]] = rungs[-1]["information"] / i_local[row["name"]]
    return gain, rho


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spectrum", type=Path, required=True, help="a difference_spectrum record")
    parser.add_argument("--converge", type=Path, default=Path("results/fork_deflection"))
    parser.add_argument(
        "--local", type=Path, nargs="+", default=[Path("results/ip_local_p1.json"), Path("results/ip_local_p2.json")]
    )
    parser.add_argument("--pixel-size", type=float, default=2.0)
    args = parser.parse_args()

    spec = json.loads(args.spectrum.read_text())["result"]
    freq = np.array(spec["frequency_cyc_px"])
    weight = np.array(spec["shell_weight"])
    gain, rho = load_gain_rho(args.converge, args.local)

    print(f"{'pair':<44}{'D=2R':>8}{'d_rms':>7}{'D/d':>7}{'s_half':>7}{'top_oct':>9}{'rho_g':>7}")
    dd, ds, g, r = [], [], [], []
    for row in spec["rows"]:
        power = np.array(row["defocus"][0]["power"])
        wp = power * weight
        d_rms = args.pixel_size / np.sqrt((wp * freq**2).sum() / wp.sum())
        s_half = row["defocus"][0]["half_power_spacing_a"]
        big = 2 * row["radius"]
        nm = row["name"]
        print(
            f"{nm[:44]:<44}{big:>8.1f}{d_rms:>7.2f}{big / d_rms:>7.2f}{s_half:>7.2f}"
            f"{gain.get(nm, float('nan')) * 100:>+8.1f}%{rho.get(nm, float('nan')):>7.3f}"
        )
        if nm in gain:
            dd.append(big / d_rms)
            ds.append(big / s_half)
            g.append(gain[nm])
            r.append(rho.get(nm, np.nan))

    dd, ds, g, r = map(np.array, (dd, ds, g, r))
    ok = np.isfinite(r)
    print(f"\ncorrelations (n={len(g)}):")
    print(
        f"  D/d_rms  vs top-octave gain: {np.corrcoef(dd, g)[0, 1]:+.3f}   (D/s_half: {np.corrcoef(ds, g)[0, 1]:+.3f})"
    )
    print(
        f"  D/d_rms  vs rho_global:      {np.corrcoef(dd[ok], r[ok])[0, 1]:+.3f}   "
        f"(D/s_half: {np.corrcoef(ds[ok], r[ok])[0, 1]:+.3f})"
    )


if __name__ == "__main__":
    main()
