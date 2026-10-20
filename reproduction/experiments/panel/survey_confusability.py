"""Aggregate the survey's pair rejections by the guard that refused each, over the scored records.

`score_survey` writes each refusal as one preformatted message; this pools them across the survey and
counts them by guard, because the guards mean opposite things about the axis: a correspondence failure
says the PAIR was mis-specified (a defect in the survey), while an RMSD or box refusal is a property of
the axis or the campaign, not the pairing. Only the first is worth fixing. Reads the `rejections` field
each survey record already carries -- no re-scoring. Rationale: docs/rationale/preprocess.md
"""

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

from reproduction.record import write_result
from reproduction.reporting.numbers import Ledger

DRIVER = "reproduction/experiments/panel/survey_confusability.py"


def guard_of(why: str) -> str:
    """The guard a rejection message names, recovered from the message `score_survey` wrote."""
    lower = why.lower()
    if "rmsd" in lower and ("exceed" in lower or "over the guard" in lower or "too large" in lower):
        return "ca_rmsd_over_guard"
    if "clip" in lower or "box" in lower:
        return "box_too_small"
    if "share only" in lower or "0 atoms" in lower or "no corresponding" in lower:
        return "atom_correspondence_zero"
    if "correspond" in lower or "residue type" in lower or "matched c-alpha" in lower:
        return "atom_correspondence"
    return "other"


def collect(paths: list[str]) -> list[str]:
    """Every `rejections` message from every scored record the globs resolve."""
    messages: list[str] = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)):
            record = json.loads(Path(path).read_text()).get("result", {})
            for item in record.get("rejections", []):
                messages.append(item if isinstance(item, str) else (item.get("why") or item.get("reason") or ""))
    return messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--scored", nargs="+", default=["results/survey_pool/*.json"], help="score_survey result files or globs"
    )
    parser.add_argument("--out", type=Path, default=Path("results/survey_confusability.json"))
    parser.add_argument("--values", type=Path, default=Path("docs/manuscript/figures/values/survey_confusability.json"))
    args = parser.parse_args(argv)

    messages = collect(args.scored)
    by_guard = Counter(guard_of(why) for why in messages)
    rejections = {"total": len(messages), "by_guard": dict(by_guard), "messages": messages}
    write_result(args.out, {"rejections": rejections}, driver=DRIVER, arguments=vars(args))
    val = Ledger()
    val("survey_refusals", len(messages), "")
    val("survey_refusals_correspondence", by_guard.get("atom_correspondence", 0), "")
    val("survey_refusals_box", by_guard.get("box_too_small", 0), "")
    val.dump(args.values)
    print(f"{rejections['total']} rejections: {dict(by_guard)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
