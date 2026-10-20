"""Supplementary table: the measured recovery statistics of every real two-state system and the controls.

    .venv/bin/python reproduction/experiments/figures/recovery_table.py

Draws nothing, like `appendix_ledger.py`. It reads each system's `d'^2_obs` from its record and recomputes
the recovery -- two-pool and no-pool together, on one held-out split so the two are comparable -- from the
system's own pools, so the Supplementary table fills from `[[val:...]]` tokens instead of retyped literals.
Rows: the seven breadth systems (MsbA, FaNaC, NKCC1, SLC19A3, SLCO2A1, Ste2, HK-ATPase), the SpCas9
6nt/12nt pair, and two controls -- the same-state (twin) null and CryoBench IgG-1D at SNR 0.001. Per row it
emits `d'^2_obs`, the two-pool fitted purities `f_A`/`f_B` (which carry the recovery slope `f_A - f_B` and
intercept `f_B`, so neither is tabulated separately) and RMS, and the same for no pool. A cell with no
measurement emits no token, so the table writes `---`. Rationale: docs/rationale/figures.md
"""

import argparse
import json
from collections.abc import Iterable
from glob import glob
from pathlib import Path

import jax
import numpy as np

from cryodisc.statistics.composition import fraction_classifier, fraction_no_pool, fraction_two_pool
from cryodisc.statistics.information import deflection
from reproduction.analysis.purity import SWEEP_FRACTIONS, fit_purity, sweep_mixtures
from reproduction.record import payload_of
from reproduction.reporting.numbers import Ledger

DRIVER = "reproduction/experiments/figures/recovery_table.py"

BREADTH = ("msba", "fanac", "nkcc1", "slc19a3", "slco2a1", "ste2", "hk-atpase")
"""The seven breadth systems, keyed as `calculator_check.json` and `results/ribo_pools/` record them."""

NAMES = {
    "msba": "MsbA",
    "fanac": "FaNaC",
    "nkcc1": "NKCC1",
    "slc19a3": "SLC19A3",
    "slco2a1": "SLCO2A1",
    "ste2": "Ste2",
    "hkatpase": "HK-ATPase",
    "spcas9": "SpCas9 (6nt/12nt)",
    "samestate": "Same-state null",
    "cryobench": "CryoBench IgG-1D (SNR 0.001)",
}
"""Display names, keyed by the row token (a system key with its hyphen dropped)."""

SPCAS9_D2_OBS = 2.906
"""The SpCas9 6nt/12nt pooled d'^2_obs, the worked pair of Figure 1 (shared with `calculator.py`'s EXAMPLE)."""

FIELDS = (
    ("d2obs", ".2f"),
    ("fa", ".3f"),
    ("fb", ".3f"),
    ("fa0", ".3f"),
    ("fb0", ".3f"),
    ("rms", ".2f"),
    ("rms0", ".2f"),
    ("rmsc", ".2f"),
)
"""The cells a row can fill, each with its format; a `None` value emits no token (an `---` cell).
Two-pool purities `fa`/`fb` and RMS `rms`; no-pool purities `fa0`/`fb0` and RMS `rms0`; classify-and-count
RMS `rmsc`."""

LADDER_SPLITS = 20
"""Held-out splits averaged per system, matching `real_recovery.ladder` so the table's RMS equals Fig. 3c's.
Both estimators share each split, so their RMS and purities are directly comparable."""


def emit_row(val: Ledger, row: str, cells: dict[str, float | None]) -> dict[str, str]:
    """Record every measured cell of one row and print it; skip cells whose value is `None`.

    Args:
        val: The ledger to record into.
        row: The row token, e.g. `msba` or `spcas9`.
        cells: `{field: value}`; a `None` value is an unmeasured cell and emits no token.

    Returns:
        `{field: formatted}` for the cells that were emitted.
    """
    emitted = {}
    for field, spec in FIELDS:
        value = cells.get(field)
        if value is None:
            continue
        if float(f"{value:{spec}}") == 0.0:
            value = 0.0  # a tiny negative would otherwise render "-0.000"
        emitted[field] = val(f"tbl_{row}_{field}", value, spec)
    shown = "   ".join(f"{field} {emitted.get(field, '---')}" for field, _ in FIELDS)
    print(f"  {NAMES.get(row, row):28s} {shown}")
    return emitted


def load_pools(system_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """The two pure-pool log-ratio arrays of a `results/ribo_pools/` system, A then B."""
    pools = sorted(p for p in glob(str(system_dir / "pool_*.npy")) if "_d2" not in p)
    return np.load(pools[0]).astype(float), np.load(pools[1]).astype(float)


def _halves(pool: np.ndarray, key) -> tuple[np.ndarray, np.ndarray]:
    """Split a pool into a draw half (mixed into the sweep) and a disjoint reference half."""
    index = np.asarray(jax.random.permutation(key, pool.shape[0]))
    half = pool.shape[0] // 2
    return pool[index[:half]], pool[index[half:]]


def recover_both(pool_a: np.ndarray, pool_b: np.ndarray) -> dict[str, float]:
    """Two-pool and no-pool recovery averaged over `LADDER_SPLITS` held-out splits, as Fig. 3c computes RMS.

    Each split mixes a draw half across the sweep and references the disjoint half; both estimators read
    the same mixtures, so their end-anchors (`fit_purity`) and RMS are comparable. The no-pool anchors come
    out near 1/0 like the two-pool's, so dropping the references does not compress the endpoints; the cost
    shows in the larger no-pool RMS. The RMS reproduces `real_recovery.ladder` exactly.
    """
    a, b = np.asarray(pool_a, float), np.asarray(pool_b, float)
    acc: dict[str, list[float]] = {}
    for seed in range(LADDER_SPLITS):
        key_a, key_b = jax.random.split(jax.random.key(seed))
        draw_a, ref_a = _halves(a, key_a)
        draw_b, ref_b = _halves(b, key_b)
        for estimator, suffix in (
            (lambda mix, ra=ref_a, rb=ref_b: fraction_two_pool(np.asarray(mix), ra, rb), ""),
            (lambda mix: fraction_no_pool(mix), "0"),
            (lambda mix, ra=ref_a, rb=ref_b: fraction_classifier(np.asarray(mix), ra, rb), "c"),
        ):
            known, recovered = sweep_mixtures(
                draw_a, draw_b, SWEEP_FRACTIONS, key=jax.random.key(0), estimator=estimator
            )
            known, recovered = np.asarray(known, float), np.asarray(recovered, float)
            acc.setdefault(f"rms{suffix}", []).append(100.0 * float(np.sqrt(np.mean((recovered - known) ** 2))))
            if suffix != "c":  # only the soft estimators get tabulated end-anchors; the classifier gets RMS only
                purity = fit_purity(known, recovered)
                acc.setdefault(f"fa{suffix}", []).append(float(purity.f_a))
                acc.setdefault(f"fb{suffix}", []).append(float(purity.f_b))
    return {field: float(np.mean(values)) for field, values in acc.items()}


def read_breadth(check_path: Path, pools_root: Path) -> dict[str, dict[str, float]]:
    """Each breadth system's cells: `d'^2_obs` from the check, the recovery recomputed from its pools."""
    check = payload_of(json.loads(check_path.read_text()))
    d2_obs = {entry["system"]: float(entry["d2_obs"]) for entry in check["systems"]}
    rows = {}
    for system in BREADTH:
        cells = {"d2obs": d2_obs[system]}
        cells.update(recover_both(*load_pools(pools_root / system)))
        rows[system.replace("-", "")] = cells
    return rows


def main(argv: Iterable[str] | None = None) -> int:
    """Record every table cell off its record and pools, print it, and dump the values JSON. Draws nothing."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", type=Path, default=Path("results/calculator_check.json"))
    parser.add_argument("--pools-root", type=Path, default=Path("results/ribo_pools"))
    parser.add_argument("--cas9-a", type=Path, default=Path("results/rerun_cas9_hoRepro_6nt.npy"))
    parser.add_argument("--cas9-b", type=Path, default=Path("results/rerun_cas9_hoRepro_12nt.npy"))
    parser.add_argument("--samestate", type=Path, default=Path("results/purity_cas9_samestate_null.json"))
    parser.add_argument("--cryobench-a", type=Path, default=Path("results/cb_lo_000.npy"))
    parser.add_argument("--cryobench-b", type=Path, default=Path("results/cb_lo_050.npy"))
    parser.add_argument("--out", type=Path, default=Path("docs/manuscript/figures/recovery_table"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    val = Ledger()
    print("consolidated recovery statistics, d'^2_obs off each record and recovery recomputed from the pools")
    rows = read_breadth(args.check, args.pools_root)
    for row in rows:
        emit_row(val, row, rows[row])

    # SpCas9: d'^2_obs is the Figure 1 pooled value; the recovery is recomputed from its two pools like the rest.
    cas9_cells = {"d2obs": SPCAS9_D2_OBS}
    cas9_cells.update(recover_both(np.load(args.cas9_a), np.load(args.cas9_b)))
    emit_row(val, "spcas9", cas9_cells)

    # Control 1: same-state null -- identical states, so the two-pool anchors coincide (f_A ~ f_B, slope ~ 0)
    # and recovery is flat; its message is that coincidence, not an RMS, and no-pool is undefined.
    null = payload_of(json.loads(args.samestate.read_text()))
    emit_row(val, "samestate", {"fa": float(null["purity"]["f_a"]), "fb": float(null["purity"]["f_b"])})

    # Control 2: CryoBench IgG-1D, the one row with a forward-model d'^2 (its two log-ratio pools' deflection).
    cb_cells = {"d2obs": float(deflection(np.load(args.cryobench_a), np.load(args.cryobench_b)))}
    cb_cells.update(recover_both(np.load(args.cryobench_a), np.load(args.cryobench_b)))
    emit_row(val, "cryobench", cb_cells)

    val.dump(args.out.parent / "values" / f"{args.out.name}.json")
    print(f"  wrote {args.out.parent / 'values' / (args.out.name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
