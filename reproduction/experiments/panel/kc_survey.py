"""Measure K_c per pair over a survey: one fixed dose, grid grown adaptively to a measured plateau.

    .venv/bin/python reproduction/experiments/panel/kc_survey.py \
        --panels data/survey_topup_1.json data/survey_topup_2.json \
        --dose 40 --rise 0.025 --min-stop-bank 100000 --cap-bank 2000000 \
        --seed 0 --out-dir results/kc_survey

The Figure 5 production run (docs/planning/figure5_kc_convergence.md). For each pair it fixes a DOSE
(contrast=1, rmsf=0.5 -- the ordered-protein Debye-Waller floor, B~20), then steps up the quantized
pose-bank ladder, scoring each new bank with `ip_converge.py` as a subprocess, until the top octave
gains less than `--rise` AND the ladder holds a rung at or above `--min-stop-bank` (so the plateau is
measured, not extrapolated), or a compute cap is hit. A pair still rising at the cap is recorded
right-censored (K_c > cap), not chased.

Drives ip_converge as a subprocess rather than importing it (reproduction/experiments/ forbids sibling imports). Each
escalation re-simulates the SAME seeded stack at the SAME dose and scores only the NEW banks, so no rung
is scored twice and the merged ladder is one comparable curve.

One JSON per pair in `--out-dir`, rewritten after every rung with a `status` of in_progress, converged
or censored. That file IS the checkpoint: a job killed mid-pair (the top rungs run hours) resumes from
the next unscored bank, and a finished pair is skipped. Writes are atomic (temp then rename) so a kill
mid-write cannot leave a torn checkpoint.

`--shard i/N` fans the survey across N workers (one GPU each): worker i takes every N-th pair by sorted
name, a static collision-free partition needing no cross-worker coordination. A worker killed by
preemption is restarted by Slurm `--requeue`; a finished pair is then skipped and an in-progress one
resumes from its checkpoint, so the restart picks up where the kill landed rather than redoing the shard.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from cryodisc.statistics.convergence import DEFAULT_LADDER, INITIAL, RISE, top_octave_gain
from reproduction.targets import load_panel

# Extends the shared ladder past its 1M top so high-K* pairs can be chased to flatness under a 2M cap.
SURVEY_LADDER: tuple[int, ...] = DEFAULT_LADDER + (1_400_000, 2_000_000)


def write_atomic(path: Path, obj) -> None:
    """Write `obj` as JSON so a preemption mid-write cannot leave a torn checkpoint: temp then rename."""
    scratch = path.with_name(f"{path.name}.{os.getpid()}.part")
    scratch.write_text(json.dumps(obj, indent=1))
    os.replace(scratch, path)


class ScoreCrash(RuntimeError):
    """ip_converge exited non-zero, wrote nothing, or wrote no rows: a transient failure (OOM, cold
    XLA compile, preemption) that must be retried, never recorded terminally."""


def score_banks(panel: Path, name: str, dose: float, banks: list[int], args, tmp: Path) -> dict:
    """Run ip_converge on one pair at one dose over `banks`; return its row dict.

    Raises ScoreCrash on a non-zero exit, missing output, or empty rows -- each a transient failure
    the caller retries rather than poisoning the pair with a terminal record. Empty rows are transient
    too: the driver only ever asks for names drawn from the panel, so a missing row is a failed run,
    never a genuinely unscorable pair.
    """
    cmd = [
        sys.executable,
        "reproduction/experiments/panel/ip_converge.py",
        "--panel",
        str(panel),
        "--rows",
        name,
        "--data-dir",
        str(args.data_dir),
        "--dose",
        f"{dose:g}",
        "--contrast",
        "1.0",
        "--rmsf",
        f"{args.rmsf:g}",
        "--n-particles",
        str(args.n_particles),
        "--seed",
        str(args.seed),
        "--banks",
        *[str(b) for b in banks],
        "--out",
        str(tmp),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=False
    )  # no check: a crash must not take the worker down, but it must be retryable
    if proc.returncode != 0 or not tmp.exists():
        raise ScoreCrash(f"ip_converge exit {proc.returncode} on {name!r}: {proc.stderr.strip()[-500:]}")
    rows = json.loads(tmp.read_text())["rows"]
    if not rows:
        raise ScoreCrash(f"ip_converge wrote no rows for {name!r} at banks {banks}")
    return rows[0]


def convergence(curve: dict[int, float], rise: float = RISE) -> dict:
    """K_c(95) bracket, top-octave gain and rising flag from a {realized_bank: d2} curve.

    A smaller `rise` stops further up the curve: 0.15 lands ~74% of the plateau, 0.025 ~95%.
    """
    banks = np.array(sorted(curve))
    d2 = np.array([curve[b] for b in banks])
    gain = top_octave_gain(curve)
    hit = np.where(d2 >= 0.95 * d2[-1])[0]
    i = int(hit[0]) if len(hit) else len(banks) - 1
    bracket = "<=96" if i == 0 else f"{banks[i - 1]}-{banks[i]}"
    return {"top_octave_gain": gain, "rising": gain >= rise, "kc_bracket": bracket, "top_bank": int(banks[-1])}


def _record(
    name: str, dose: float, snr, rmsd, curve: dict[int, float], step: int, status: str, seconds: float, args
) -> dict:
    """One pair's record, terminal or in-progress; `curve`/`step` are enough to resume it."""
    conv = convergence(curve, args.rise)
    kc = f">{conv['top_bank']}" if status == "censored" else conv["kc_bracket"]
    return {
        "name": name,
        "dose": dose,
        "snr": snr,
        "rmsd": rmsd,
        "kc": kc,
        "status": status,
        "step": step,
        "top_octave_gain": conv["top_octave_gain"],
        "d2_plateau": max(curve.values()),
        "curve": {str(k): curve[k] for k in sorted(curve)},
        "cap_bank": int(args.cap_bank),
        "seed": args.seed,
        "seconds": round(seconds, 1),
    }


def _resume(out: Path) -> dict | None:
    """An in-progress checkpoint to continue, or None (absent, terminal, or unreadable)."""
    try:
        rec = json.loads(out.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return rec if rec.get("status") == "in_progress" else None


def _done(out: Path) -> bool:
    """Whether this pair is finished (a terminal record on disk), so a worker skips it."""
    try:
        return json.loads(out.read_text()).get("status") in ("converged", "censored")
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def measure_pair(panel: Path, name: str, args, tmp: Path, out: Path) -> dict:
    """Grow the grid at the fixed dose to a measured plateau or the cap, checkpointing after each rung.

    Resumes from an in-progress checkpoint at `out` if one is there, so a job killed mid-pair (the
    ladder's top rungs run hours) picks up at the next unscored bank rather than re-scoring the pair.
    The resumed pair keeps its checkpoint dose, so a clean single-dose run needs a fresh out-dir.
    """
    ladder = [b for b in SURVEY_LADDER if b <= args.cap_bank]
    prior = _resume(out)
    if prior is not None:
        dose, snr, rmsd, step, seconds = prior["dose"], prior["snr"], prior["rmsd"], prior["step"], prior["seconds"]
        curve = {int(k): v for k, v in prior["curve"].items()}
    else:
        dose = args.dose
        curve, snr, rmsd, step, seconds = {}, None, None, INITIAL, 0.0

    while True:
        # first pass: score the INITIAL prefix; afterwards one new bank per pass
        banks = ladder[:INITIAL] if not curve else [ladder[step - 1]]
        t0 = time.time()
        row = score_banks(
            panel, name, dose, banks, args, tmp
        )  # raises ScoreCrash on a failed run; the checkpoint is left to resume
        seconds += time.time() - t0
        if snr is None:  # record from the first pass only: a fixed reference bank, not each escalation's
            snr, rmsd = row["snr"], row["rmsd"]
        for rung in row["rungs"]:
            curve[rung["k"]] = rung["d2"]
        conv = convergence(curve, args.rise)
        at_cap = step >= len(ladder)
        # a flattened ladder still climbs until it holds a rung >= the floor, so every pair gets a high-K point
        flattened = not conv["rising"] and max(curve) >= args.min_stop_bank
        if flattened or at_cap:
            rec = _record(
                name, dose, snr, rmsd, curve, step, "censored" if conv["rising"] else "converged", seconds, args
            )
            write_atomic(out, rec)
            return rec
        step += 1  # checkpoint the RESUME position: the next iteration scores ladder[step - 1]
        write_atomic(out, _record(name, dose, snr, rmsd, curve, step, "in_progress", seconds, args))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panels", type=Path, nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/pdb"))
    parser.add_argument("--only", nargs="+", default=None, help="score only these pair names (smoke test)")
    parser.add_argument("--dose", type=float, default=40.0, help="fixed dose applied to every pair")
    parser.add_argument(
        "--rise",
        type=float,
        default=RISE,
        help="top-octave gain to stop at: 0.15 lands ~74%% of the plateau, 0.025 ~95%%",
    )
    parser.add_argument(
        "--min-stop-bank",
        type=int,
        default=0,
        help="a flattened ladder keeps climbing until it holds a rung >= this (a high-K floor)",
    )
    parser.add_argument("--cap-bank", type=int, default=1_000_000, help="stop and censor above this bank")
    parser.add_argument(
        "--rmsf",
        type=float,
        default=0.5,
        help="specimen Debye-Waller floor in A (B=8 pi^2 rmsf^2 ~ 20 at 0.5); the ordered-protein floor",
    )
    parser.add_argument("--n-particles", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("results/kc_survey"))
    parser.add_argument(
        "--shard",
        default=None,
        metavar="i/N",
        help="0-indexed GPU fan-out: process pairs with index %% N == i, i in 0..N-1 (e.g. 3/8)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tmp = args.out_dir / f"_tmp_ip_converge_{os.getpid()}.json"  # per-process: shards must not share
    wanted = set(args.only) if args.only else None

    todo = []
    for panel in args.panels:
        candidates, _ = load_panel(panel, data_dir=args.data_dir, fetch=False)
        for c in candidates:
            if wanted is None or c.name in wanted:
                todo.append((panel, c.name))
    if args.shard:  # deterministic partition across GPUs; sorted first so every shard agrees on order
        i, n = (int(x) for x in args.shard.split("/"))
        if not 0 <= i < n:
            parser.error(f"--shard i/N is 0-indexed, needs 0 <= i < N; got {args.shard}")
        todo = sorted(todo, key=lambda pn: pn[1])[i::n]

    def out_for(item: tuple[Path, str]) -> Path:
        return args.out_dir / f"{hashlib.sha1(item[1].encode()).hexdigest()[:10]}.json"

    def run_pair(item: tuple[Path, str], out: Path) -> None:
        panel, name = item
        prior = _resume(out)
        if prior is not None:
            print(f"resume (step {prior['step']}): {name[:40]}")
        try:
            rec = measure_pair(panel, name, args, tmp, out)
        except (
            ScoreCrash
        ) as exc:  # transient (OOM/cold-compile/preempt): leave no terminal record so a requeue retries it
            print(f"{name[:40]:42s} CRASH, will retry -- {exc}")
            return
        print(
            f"{name[:40]:42s} dose={rec['dose']:5.1f} snr={rec['snr']:5.1f} "
            f"K_c={rec['kc']:>14} ({rec['status']}, {rec['seconds']:.0f}s)"
        )

    print(
        f"{len(todo)} pair(s); dose {args.dose:g}, rise {args.rise:g}, floor {args.min_stop_bank}, "
        f"cap {args.cap_bank}, seed {args.seed}"
    )
    did = 0
    for item in todo:
        out = out_for(item)
        if _done(out):  # a finished pair's terminal record is its skip marker; a requeue resumes the rest
            continue
        run_pair(item, out)
        did += 1
    print(f"worker did {did} pair(s)")
    tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
