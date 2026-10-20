"""Pull a few real particles from each SpCas9 pool for the overview schematic's particle box.

    ribo: cd /data/bryceeh/cryodisc && .venv/bin/python -m reproduction.experiments.figures.overview_particles
Read through `pools.read_pool`, the reader every measured figure scores with, so the drawn particles
are the ones the method sees, recentered by the table's own shifts. Each candidate is correlated, after
a low-pass, with its own state's model projected at its refined pose and defocus, against a null that
pairs it with another particle's projection; the best-matched are kept, since a random single particle
at this SNR is usually invisible. Stored raw; `overview.py` owns display filtering.
Rationale: docs/rationale/figures.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from cryodisc.forward.optics import CTFParams
from cryodisc.grids.poses import euler_from_relion
from cryodisc.pools import read_pool
from cryodisc.specimen.structures import align_conformers
from reproduction.record import write_result

BASE = Path("/data/bryceeh/cas9")
POOLS = (
    ("6nt", BASE / "pools/cas9_6nt_splitB.cs", BASE / "CS-e9-spcas9-11779-2"),
    ("12nt", BASE / "pools/cas9_12nt_splitB.cs", BASE / "CS-e9-spcas9-11780-2"),
)
"""The two states' pools, in the order `overview.py` calls A and B (7Z4C, then 7Z4G)."""
PAIR = (Path("data/pdb/7Z4C.pdb"), Path("data/pdb/7Z4G.pdb"))
CHUNK = 16
"""Projections rendered per call: the assembly's render temporary does not fit a 16 GB card at more."""


def lowpass(images: np.ndarray, pixel_size: float, resolution: float) -> np.ndarray:
    """A Gaussian low-pass at `resolution` A, over the last two axes."""
    k = np.fft.fftfreq(images.shape[-1], d=pixel_size)
    radius = np.hypot(*np.meshgrid(k, k))
    return np.real(np.fft.ifft2(np.fft.fft2(images) * np.exp(-((radius * resolution) ** 2) / 2)))


def correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of each image pair, over pixels."""
    a = a - a.mean(axis=(-2, -1), keepdims=True)
    b = b - b.mean(axis=(-2, -1), keepdims=True)
    return (a * b).sum(axis=(-2, -1)) / np.sqrt((a * a).sum(axis=(-2, -1)) * (b * b).sum(axis=(-2, -1)))


def main() -> int:
    """Write the particles as `.npy` and their states, scores and provenance as `.json` beside it."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-state", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=400, help="random draw per state the kept ones come from")
    parser.add_argument("--resolution", type=float, default=20.0, help="low-pass before correlating, in A")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/overview_cas9_particles.npy"))
    args = parser.parse_args()

    images, states, record = [], [], {}
    models = None
    for index, (state, meta, root) in enumerate(POOLS):
        table, stack, pixel_size, _, _ = read_pool(meta, root=root, subsample=args.candidates, subsample_seed=args.seed)
        real = np.real(np.fft.ifft2(np.asarray(stack)))
        if models is None:
            pair = align_conformers(PAIR[0], PAIR[1], assembly="1")
            models = pair.models_on_grid(box_size=real.shape[-1], pixel_size=pixel_size)
        angles = jnp.asarray(euler_from_relion(table.euler))
        defocus = (np.asarray(table.defocus_u) + np.asarray(table.defocus_v)) / 2
        optics = {
            "spherical_aberration_in_mm": float(np.asarray(table.spherical_aberration_in_mm)[0]),
            "amplitude_contrast_ratio": float(np.asarray(table.amplitude_contrast_ratio)[0]),
        }
        projections = np.concatenate(
            [
                np.asarray(
                    models[index].forward_batch(
                        angles[start : start + CHUNK],
                        CTFParams(defocus_in_angstroms=jnp.asarray(defocus[start : start + CHUNK]), **optics),
                    )
                )
                for start in range(0, len(real), CHUNK)
            ]
        )
        seen, model = lowpass(real, pixel_size, args.resolution), lowpass(projections, pixel_size, args.resolution)
        matched = correlation(seen, model)
        # each particle against ANOTHER particle's projection: what a pose-blind match would score
        null = correlation(seen, np.roll(model, 1, axis=0))
        keep = np.argsort(matched)[::-1][: args.per_state]
        images.append(real[keep])
        states += [state] * len(keep)
        record[state] = {
            "pixel_size": float(pixel_size),
            "matched_median": float(np.median(matched)),
            "null_median": float(np.median(null)),
            "matched_above_null_p95": float(np.mean(matched > np.percentile(null, 95))),
            "kept_correlation": matched[keep].tolist(),
            "kept_defocus": defocus[keep].tolist(),
        }
        print(
            f"{state}: matched r median {np.median(matched):+.3f} vs null {np.median(null):+.3f}; "
            f"{100 * record[state]['matched_above_null_p95']:.0f}% above the null's 95th percentile; "
            f"kept r {matched[keep].min():.3f}-{matched[keep].max():.3f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # .npy because results/ tracks only .npy and .json; the states ride in the record
    np.save(args.out, np.concatenate(images).astype(np.float32))
    write_result(
        args.out.with_suffix(".json"),
        {"n": len(states), "states": states, "pools": [str(meta) for _, meta, _ in POOLS], "by_state": record},
        driver="reproduction/experiments/figures/overview_particles.py",
        arguments=vars(args),
    )
    print(f"wrote {args.out} and {args.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
