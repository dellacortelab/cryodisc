"""Predict `d^2` from FSC curves alone, by a route that shares no code with the scorer.

Over 3D voxels, `d^2 = (1/N_half) sum_k n3(k) |dS(k)|^2 / nu(k)` (1), with `nu = (1 - FSC_gs) P` the
half-map noise power and `|dS|^2 = |dT|^2 - nu` the state difference with template noise subtracted
(held-out noise inflates the variance, not the mean). No CTF, noise or forward model enters. Setting
`dS` to the raw map difference recovers the memorization null `d^2_null = 2 V_band / N_half` (2). The
three amplitude conventions (`none`, global, per-shell) are all reported, as states carry no common
scale. Usage: fsc_bridge.py --state-{a,b}-half-{a,b} *.mrc --n-half-a N --n-half-b N --out out.json
"""

import argparse
import math
from pathlib import Path

import numpy as np

from cryodisc.specimen.density import read_map
from reproduction.analysis.shells import Shells, shell_index
from reproduction.record import write_result

_SCALE_CORRELATION = 0.95

DRIVER = "reproduction/experiments/benchmark/fsc_bridge.py"

FSC_THRESHOLD = 0.143
"""The gold-standard half-map criterion, quoted only to locate the declared resolution.

Nothing in (1) uses it. It is here so the cumulative `d^2` can be read against the number the map
would be published at, which is the comparison the FSC-complement claim rests on.
"""


def scale_shells(cross_correlation: np.ndarray, n_band: int, floor: float) -> int:
    """The last shell over which a relative amplitude can be fitted without the difference biasing it.

    The two states are reconstructed separately and carry no common density scale, so one has to be
    fitted. Fitting it over the whole band uses the shells where the states DIFFER, which pulls the
    scale toward whichever map is larger there and feeds straight back into `|dS|^2`. The shells
    where the implied signal correlation is still at `floor` are the ones where the two maps agree,
    so a scale is identifiable there and the difference cannot bias it.

    `cross_correlation` is a ratio of correlations and is therefore already scale-free, so choosing
    the shells from it is not circular.

    Args:
        cross_correlation: Implied true signal correlation per shell.
        n_band: Last shell inside the band.
        floor: Correlation above which the states count as agreeing.

    Returns:
        The last shell to fit on, at least 3.
    """
    last = 2
    for shell in range(1, n_band + 1):
        # NOT `< floor` alone: an undefined correlation compares False against any threshold, so a bare
        # comparison walks past it and extends the fit into shells with no signal
        value = cross_correlation[shell]
        if not np.isfinite(value) or value < floor:
            break
        last = shell
    return max(last, 3)


def load_volume(path: Path, box_size: int | None, pixel_size: float | None) -> tuple[np.ndarray, float]:
    """One map, checked against the box and pixel the other three carry.

    Args:
        path: The `.mrc` to read.
        box_size: Box the previous maps had, or `None` for the first.
        pixel_size: Pixel the previous maps had, or `None` for the first.

    Returns:
        The volume as float64, and its pixel size.

    Raises:
        ValueError: If the map is not cubic, from `density.read_map`.
        SystemExit: If it carries no pixel size, or disagrees with the others.
    """
    volume, pixel, _ = read_map(path)
    volume = np.asarray(volume, dtype=np.float64)
    if pixel is None:
        raise SystemExit(f"{path} carries no pixel size, and (1) is quoted per angstrom")
    if box_size is not None and volume.shape[0] != box_size:
        raise SystemExit(f"{path} is box {volume.shape[0]} against {box_size}; the four maps must match")
    if pixel_size is not None and abs(pixel - pixel_size) > 1e-3:
        raise SystemExit(f"{path} is {pixel:.4f} A/px against {pixel_size:.4f}; the four maps must match")
    return volume, float(pixel)


def register(
    first: np.ndarray,
    second: np.ndarray,
    index: np.ndarray,
    radius: float,
    grids: tuple[np.ndarray, ...],
    box_size: int,
) -> tuple[np.ndarray, tuple[int, int, int], float]:
    """Shift the second transform onto the first, to whole voxels, inside the band.

    Whole voxels because that is the search the scorer runs on a map pair; a sub-voxel refinement
    here would credit the prediction with an alignment the measurement never had.

    Args:
        first: Reference transform.
        second: Transform to move.
        index: Shell index of every voxel.
        radius: Band radius in voxels.
        grids: The `(kz, ky, kx)` frequency grids.
        box_size: Cubic box edge.

    Returns:
        The phase ramp that applies the shift, the shift itself, and the correlation peak in units
        of its own rms. The ramp rather than the shifted transform, because both of state B's
        halves have to move by exactly the same amount as the full map that was matched.
    """
    inside = (index >= 1) & (index <= radius)
    correlation = np.fft.ifftn(np.where(inside, first * np.conj(second), 0)).real
    peak = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    wrapped = [int(p) if p <= box_size // 2 else int(p) - box_size for p in peak]
    shift = (wrapped[0], wrapped[1], wrapped[2])
    kz, ky, kx = grids
    ramp = np.exp(-2j * np.pi * (kz * shift[0] + ky * shift[1] + kx * shift[2]) / box_size)
    return ramp, shift, float(correlation.max() / correlation.std())


def measured(paths: list[Path]) -> dict | None:
    """Mean `d^2` and its standard error from stored per-particle log-ratios.

    The only route by which a MEASURED rung reaches this record, which is what the bridge prediction
    is checked against. `d^2 = E[Lambda|A] - E[Lambda|B]` and a single-pool run reads `|mean| * 2` by
    symmetry, the convention `oracle_closed_form.d2_pool` records.

    Args:
        paths: One `.npy` of per-particle `Lambda` per run.

    Returns:
        The runs, their pooled mean and the spread across them, or `None` if no path was given.
    """
    if not paths:
        return None
    runs = []
    for path in paths:
        values = np.asarray(np.load(path), dtype=float).ravel()
        runs.append(
            {
                "path": str(path),
                "n": int(values.size),
                "d2": float(2 * abs(values.mean())),
                "sem": float(2 * values.std(ddof=1) / math.sqrt(values.size)),
            }
        )
    d2 = np.array([run["d2"] for run in runs])
    return {
        "runs": runs,
        "mean": float(d2.mean()),
        "spread": float(d2.max() - d2.min()) / 2 if d2.size > 1 else 0.0,
    }


def main() -> int:
    """Read four half-maps and predict `d^2` from their FSC curves."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-a-half-a", type=Path, required=True)
    parser.add_argument("--state-a-half-b", type=Path, required=True)
    parser.add_argument("--state-b-half-a", type=Path, required=True)
    parser.add_argument("--state-b-half-b", type=Path, required=True)
    parser.add_argument("--n-half-a", type=int, required=True, help="particles in ONE half of state A's reconstruction")
    parser.add_argument("--n-half-b", type=int, required=True, help="particles in ONE half of state B's reconstruction")
    parser.add_argument(
        "--band-limit",
        type=float,
        default=0.125,
        help="cycles per pixel, matching the --band-limit the rung was scored at",
    )
    parser.add_argument(
        "--measured-a", type=Path, nargs="*", default=[], help="log-ratio arrays from runs scored on state A's pool"
    )
    parser.add_argument(
        "--measured-b", type=Path, nargs="*", default=[], help="log-ratio arrays from runs scored on state B's pool"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    seen_box: int | None = None
    seen_pixel: float | None = None
    volumes = {}
    for name, path in (
        ("a_half_a", args.state_a_half_a),
        ("a_half_b", args.state_a_half_b),
        ("b_half_a", args.state_b_half_a),
        ("b_half_b", args.state_b_half_b),
    ):
        volume, seen_pixel = load_volume(path, seen_box, seen_pixel)
        seen_box = volume.shape[0]
        volumes[name] = volume
    # the loop is over four fixed halves and `load_volume` refuses a disagreement, so both are set;
    # naming them here is what lets the rest of the driver read them as the numbers they are
    assert seen_box is not None and seen_pixel is not None
    box_size, pixel_size = seen_box, seen_pixel

    radius = args.band_limit * box_size
    if radius < 4 or radius > box_size // 2:
        raise SystemExit(f"band {args.band_limit} cyc/px is {radius:.1f} voxels in a box {box_size}")

    index, counts, grids = shell_index(box_size)
    shells = Shells(index, counts)
    n_band = round(radius)
    inside = slice(1, n_band + 1)
    band_voxels = int(counts[inside].sum())

    transforms = {name: np.fft.fftn(volume) for name, volume in volumes.items()}
    full_a = 0.5 * (transforms["a_half_a"] + transforms["a_half_b"])
    full_b = 0.5 * (transforms["b_half_a"] + transforms["b_half_b"])
    ramp, shift, peak = register(full_a, full_b, index, radius, grids, box_size)
    transforms["b_half_a"] = transforms["b_half_a"] * ramp
    transforms["b_half_b"] = transforms["b_half_b"] * ramp
    full_b = full_b * ramp

    fsc_a = shells.fsc(transforms["a_half_a"], transforms["a_half_b"])
    fsc_b = shells.fsc(transforms["b_half_a"], transforms["b_half_b"])
    fsc_cross = shells.fsc(full_a, full_b)
    power_a = 0.5 * (
        shells.mean(np.abs(transforms["a_half_a"]) ** 2) + shells.mean(np.abs(transforms["a_half_b"]) ** 2)
    )
    power_b = 0.5 * (
        shells.mean(np.abs(transforms["b_half_a"]) ** 2) + shells.mean(np.abs(transforms["b_half_b"]) ** 2)
    )
    noise_a, noise_b = (1 - fsc_a) * power_a, (1 - fsc_b) * power_b
    signal_a, signal_b = fsc_a * power_a, fsc_b * power_b

    # scale-free, so it can choose the shells the scale is fitted on; the FSC is clipped to >= 0
    # first because `2f/(1+f)` is meaningless below f = 0, so those shells read as no-signal
    positive_a = np.clip(fsc_a, 0.0, None)
    positive_b = np.clip(fsc_b, 0.0, None)
    correlation = fsc_cross / np.sqrt(
        np.maximum((2 * positive_a / (1 + positive_a)) * (2 * positive_b / (1 + positive_b)), 1e-30)
    )
    last_scale_shell = scale_shells(correlation, n_band, _SCALE_CORRELATION)
    agree = slice(1, last_scale_shell + 1)
    global_scale = float(
        np.sqrt(np.sum(signal_a[agree] * counts[agree]) / max(np.sum(signal_b[agree] * counts[agree]), 1e-30))
    )
    band_scale = float(
        np.sqrt(np.sum(signal_a[inside] * counts[inside]) / max(np.sum(signal_b[inside] * counts[inside]), 1e-30))
    )
    per_shell = np.sqrt(np.maximum(signal_a, 0) / np.maximum(signal_b, 1e-30))

    predictions = {}
    for label, scale in (
        ("none", np.ones_like(fsc_a)),
        ("global", np.full_like(fsc_a, global_scale)),
        ("per_shell", np.clip(per_shell, 0.5, 2.0)),
    ):
        scaled_b_noise = noise_b * scale**2
        scaled_b_signal = signal_b * scale**2
        # cross term denormalized by the FULL powers, not the signal powers: the two states' noises
        # are independent, so using signal powers would divide the noise out twice and inflate |dS|^2
        cross = fsc_cross * np.sqrt(np.maximum((signal_a + 0.5 * noise_a) * (signal_b + 0.5 * noise_b), 0.0))
        state = np.maximum(signal_a + scaled_b_signal - 2 * scale * cross, 0.0)
        per_shell_d2_a = counts[inside] * state[inside] / np.maximum(noise_a[inside], 1e-30)
        per_shell_d2_b = counts[inside] * state[inside] / np.maximum(scaled_b_noise[inside], 1e-30)
        # past the band `state` is a difference of two noise estimates, so this is a diagnostic (what
        # a sharper map would buy); its flattening where FSC hits zero checks that no signal is faked
        beyond = slice(1, shells.n_shells)
        extended = np.cumsum(counts[beyond] * state[beyond] / np.maximum(noise_a[beyond], 1e-30)) / args.n_half_a
        predictions[label] = {
            "scale": float(scale[1]) if label != "per_shell" else None,
            "d2_pool_a": float(per_shell_d2_a.sum()) / args.n_half_a,
            "d2_pool_b": float(per_shell_d2_b.sum()) / args.n_half_b,
            "cumulative_a": np.cumsum(per_shell_d2_a / args.n_half_a).tolist(),
            "cumulative_a_beyond_band": extended.tolist(),
        }

    def first_shell_below(curve: np.ndarray) -> int | None:
        """The first shell at which a half-map FSC has fallen below the criterion.

        The shell BELOW the crossing rather than an interpolated resolution, because that is the
        convention a declared resolution is quoted at here and the two must not drift apart.

        Args:
            curve: A half-map FSC per shell.

        Returns:
            The shell index, or None if the curve never falls below inside the computed band.
        """
        for shell in range(1, shells.n_shells - 1):
            if curve[shell] >= FSC_THRESHOLD > curve[shell + 1]:
                return shell + 1
        return None

    crossing = first_shell_below(fsc_a)
    # each state's OWN gold-standard resolution, which the cross-state crossing above is not: the
    # body quotes both, and at a narrow band limit neither curve reaches the criterion
    per_state: dict[str, float | None] = {}
    for label, curve in (("state_a", fsc_a), ("state_b", fsc_b)):
        shell = first_shell_below(curve)
        per_state[f"{label}_fsc_crossing_shell"] = shell
        per_state[f"{label}_fsc_crossing_resolution"] = box_size * pixel_size / shell if shell else None

    null = {
        "closed_form_a": 2 * band_voxels / args.n_half_a,
        "closed_form_b": 2 * band_voxels / args.n_half_b,
        "band_voxels": band_voxels,
    }

    print(
        f"box {box_size} at {pixel_size:.4f} A/px, band {args.band_limit} cyc/px "
        f"= shell {n_band} = {box_size * pixel_size / n_band:.2f} A, {band_voxels:,} 3D voxels"
    )
    print(f"registration: whole-voxel shift {shift}, correlation peak {peak:.1f} rms")
    print(
        f"relative amplitude of state B against state A: {global_scale:.4f} "
        f"(fitted on shells 1-{last_scale_shell}, correlation >= {_SCALE_CORRELATION}; "
        f"whole-band fit would give {band_scale:.4f})\n"
    )
    observed = {"pool_a": measured(args.measured_a), "pool_b": measured(args.measured_b)}

    print(f"{'amplitude':>12}{'d2 pool A':>12}{'d2 pool B':>12}")
    for label, entry in predictions.items():
        print(f"{label:>12}{entry['d2_pool_a']:>12.3f}{entry['d2_pool_b']:>12.3f}")
    for pool, record in observed.items():
        if record:
            print(
                f"{'MEASURED ' + pool[-1].upper():>12}{record['mean']:>12.3f}"
                f"  +/- {record['spread']:.3f} across {len(record['runs'])} runs"
            )
    print(f"\nmemorization null 2V/N: pool A {null['closed_form_a']:.3f}, pool B {null['closed_form_b']:.3f}")
    if crossing:
        extended = predictions["global"]["cumulative_a_beyond_band"]
        print(f"gold-standard FSC = {FSC_THRESHOLD} at shell {crossing}, {box_size * pixel_size / crossing:.2f} A")
        print(
            f"cumulative d^2 at the band edge {extended[n_band - 1]:.3f}, "
            f"at the FSC resolution {extended[crossing - 1]:.3f} "
            f"({100 * extended[crossing - 1] / extended[n_band - 1] - 100:+.0f}%), "
            f"at Nyquist {extended[-1]:.3f}"
        )

    cumulative = predictions["global"]["cumulative_a"]
    print(f"\n{'shell':>6}{'res A':>9}{'FSC_A':>8}{'FSC_B':>8}{'cross':>8}{'cum d2':>9}{'% band':>8}")
    for shell in range(1, n_band + 1):
        print(
            f"{shell:>6}{box_size * pixel_size / shell:>9.1f}{fsc_a[shell]:>8.3f}{fsc_b[shell]:>8.3f}"
            f"{fsc_cross[shell]:>8.3f}{cumulative[shell - 1]:>9.3f}"
            f"{100 * cumulative[shell - 1] / cumulative[-1]:>7.0f}%"
        )

    write_result(
        args.out,
        {
            "box_size": box_size,
            "pixel_size": pixel_size,
            "band_shell": n_band,
            "band_resolution": box_size * pixel_size / n_band,
            "band_voxels": band_voxels,
            "registration": {"shift": list(shift), "peak_over_rms": peak},
            "relative_amplitude": global_scale,
            "relative_amplitude_whole_band": band_scale,
            # the SPREAD of the per-shell ratio is the argument that no scalar absorbs a B-factor
            # difference between two separately refined reconstructions
            "relative_amplitude_per_shell_low": float(np.min(per_shell[inside])),
            "relative_amplitude_per_shell_high": float(np.max(per_shell[inside])),
            "scale_shells": last_scale_shell,
            "scale_correlation_floor": _SCALE_CORRELATION,
            "predictions": predictions,
            "measured": observed,
            "memorization_null": null,
            "fsc_threshold": FSC_THRESHOLD,
            "fsc_crossing_shell": crossing,
            **per_state,
            "fsc_crossing_resolution": box_size * pixel_size / crossing if crossing else None,
            "shells": {
                "resolution": [box_size * pixel_size / s if s else None for s in range(n_band + 1)],
                "fsc_a": fsc_a[: n_band + 1].tolist(),
                "fsc_b": fsc_b[: n_band + 1].tolist(),
                "fsc_cross": fsc_cross[: n_band + 1].tolist(),
                "noise_a": noise_a[: n_band + 1].tolist(),
                "signal_a": signal_a[: n_band + 1].tolist(),
                "voxels": counts[: n_band + 1].tolist(),
                "signal_correlation": correlation[: n_band + 1].tolist(),
            },
        },
        driver=DRIVER,
        arguments={
            k: [str(p) for p in v] if isinstance(v, list) else (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
