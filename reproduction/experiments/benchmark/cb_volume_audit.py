"""Density-conservation and radial-power audit of the CryoBench IgG-1D volumes (cluster-only).

The six deposited conformations should integrate to one total density; they do not (they span ~2%),
and the radial power flattens toward high frequency. Both are properties of the rendered volumes read
directly, not of any particle stack, so this reads the maps and sums them. RUNS ON THE CLUSTER: the
volumes live on scratch, not the repo. Rationale: docs/planning/cryobench_upstream_report.md
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from cryodisc.specimen.density import read_map
from reproduction.record import write_result

DRIVER = "reproduction/experiments/benchmark/cb_volume_audit.py"


def sha256_16(path: Path) -> str:
    """The first 16 hex digits of the file's SHA-256, so a table names the exact bytes read."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def max_face_abs(volume: np.ndarray) -> float:
    """The largest absolute value on any of the six box faces; nonzero means the box clips the map."""
    faces = (volume[0], volume[-1], volume[:, 0], volume[:, -1], volume[:, :, 0], volume[:, :, -1])
    return float(max(np.abs(face).max() for face in faces))


def radial_power(volume: np.ndarray) -> list[float]:
    """Mean power per mode in each integer frequency shell (shell 0 is the DC mode, so total**2)."""
    n = volume.shape[0]
    power = np.abs(np.fft.fftn(volume)) ** 2
    freq = np.fft.fftfreq(n) * n
    kz, ky, kx = np.meshgrid(freq, freq, freq, indexing="ij")
    shell = np.rint(np.sqrt(kz**2 + ky**2 + kx**2)).astype(int).ravel()
    total = np.bincount(shell, weights=power.ravel())
    count = np.bincount(shell)
    return [float(t / c) for t, c in zip(total[: n // 2], count[: n // 2], strict=True)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="the directory holding <state>.mrc")
    parser.add_argument("--states", nargs="+", default=["000", "010", "025", "050", "075", "090"])
    parser.add_argument("--radial-state", default="000", help="the volume the radial power is measured on")
    parser.add_argument("--out", type=Path, default=Path("results/cb_volume_audit.json"))
    args = parser.parse_args()

    volumes, radial = [], None
    for state in args.states:
        path = args.root / f"{state}.mrc"
        volume = np.asarray(read_map(path)[0])
        volumes.append(
            {
                "state": state,
                "sha256_16": sha256_16(path),
                "shape": list(volume.shape),
                "total": float(volume.sum()),
                "max_face_abs": max_face_abs(volume),
            }
        )
        if state == args.radial_state:
            radial = {"mean_per_mode": radial_power(volume)}
        print(f"  {state}: total {volume.sum():.4f}  sha {volumes[-1]['sha256_16']}", flush=True)

    write_result(
        args.out,
        {"root": str(args.root), "volumes": volumes, "radial_power": radial},
        driver=DRIVER,
        arguments=vars(args),
    )
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
