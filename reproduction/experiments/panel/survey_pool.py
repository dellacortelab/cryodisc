"""Score the survey per pair: one score_survey --only per pair, sharded across workers on standby.

    .venv/bin/python reproduction/experiments/panel/survey_pool.py \
        --panel data/survey_canonical.json --out-dir results/survey_pool

Mirrors kc_survey's harness: --shard i/N fans the pairs across N workers by sorted name, and each
pair's result file IS its checkpoint -- at pair granularity, so a preempted worker loses only the one
in-flight pair (cheaper to redo than to protect) and Slurm --requeue resumes the rest of the shard.
Reuses score_survey UNCHANGED via --only; a reducer stitches the per-pair files into the survey table
and the d'^2 fit. The scoring arguments are the canonical survey's, matching
deploy/ribo/_survey_population.sh.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from reproduction.operating_point import SHARED
from reproduction.targets import load_panel

DRIVER = "reproduction/experiments/panel/score_survey.py"
# The canonical survey scoring point: the operating point (pixel/dose/rmsf) comes from SHARED, not
# hardcoded; 0.0 overlap + no RMSD guard admit the partial/RMSD rows; the marginal arm uses the
# extrapolation ladder. box-size 128 is a floor the scorer grows per-pair.
SCORE_ARGS = [
    "--pixel-size",
    f"{SHARED.pixel_size:g}",
    "--dose",
    f"{SHARED.dose:g}",
    "--box-size",
    "128",
    "--rmsf",
    f"{SHARED.rmsf:g}",
    "--sigma",
    "0.02",
    "--fraction",
    "0.5",
    "--budget",
    "100000",
    "--marginalize",
    "--poses",
    "haar",
    "--extrapolate",
    "--extrapolate-nodes",
    "1024",
    "3456",
    "8192",
    "--n-particles",
    "2000",
    "--shift-sigma",
    "1.0",
    "--min-shared-fraction",
    "0.0",
    "--no-rmsd-guard",
]


def score_one(name: str, out: Path, args) -> None:
    """Score one pair with score_survey --only, writing its result to `out`.

    Does NOT raise on a non-zero exit: a dropped pair (unresolved/refused) or a failure would
    otherwise take the whole worker down. Instead, if no result was written, it records a stub with
    the exit code and stderr tail -- so the pair is not retried forever AND the reason stays visible
    to the reducer (a systematic error surfaces as every pair carrying the same stub, not silence).
    """
    cmd = [
        sys.executable,
        DRIVER,
        "--panel",
        str(args.panel),
        "--only",
        name,
        "--data-dir",
        str(args.data_dir),
        "--out",
        str(out),
        "--seed",
        str(args.seed),
        *SCORE_ARGS,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if not out.exists():
        out.write_text(
            json.dumps(
                {
                    "result": {"rows": []},
                    "dropped": name,
                    "returncode": proc.returncode,
                    "stderr_tail": (proc.stderr or "")[-600:],
                },
                indent=1,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, default=Path("data/survey_canonical.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/pdb"))
    parser.add_argument("--only", nargs="+", default=None, help="score only these pair names (smoke test)")
    parser.add_argument("--out-dir", type=Path, default=Path("results/survey_pool"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shard",
        default=None,
        metavar="i/N",
        help="0-indexed GPU fan-out: process pairs with index %% N == i, i in 0..N-1 (e.g. 3/8)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates, _ = load_panel(args.panel, data_dir=args.data_dir, fetch=False)
    wanted = set(args.only) if args.only else None
    todo = [c.name for c in candidates if wanted is None or c.name in wanted]
    if args.shard:  # deterministic partition across GPUs; sorted first so every shard agrees on order
        i, n = (int(x) for x in args.shard.split("/"))
        if not 0 <= i < n:
            parser.error(f"--shard i/N is 0-indexed, needs 0 <= i < N; got {args.shard}")
        todo = sorted(todo)[i::n]

    def out_for(name: str) -> Path:
        return args.out_dir / f"{hashlib.sha1(name.encode()).hexdigest()[:10]}.json"

    print(f"{len(todo)} pair(s); out {args.out_dir}, seed {args.seed}", flush=True)
    did = 0
    for name in todo:
        out = out_for(name)
        if out.exists():  # its result file is its checkpoint; a requeue resumes the rest of the shard
            continue
        t0 = time.time()
        score_one(name, out, args)
        print(f"{name[:48]:50s} ({time.time() - t0:.0f}s)", flush=True)
        did += 1
    print(f"worker did {did} pair(s)", flush=True)


if __name__ == "__main__":
    main()
