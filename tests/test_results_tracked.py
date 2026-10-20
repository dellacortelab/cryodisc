"""`.gitignore` tracks `results/` by extension, so a new extension is silently ignored, not untracked.

`git status` reports an untracked file loudly and an ignored file not at all, so a too-broad ignore
rule produces silence where the paper's evidence lives. The allow-list re-includes by extension, so a
driver writing an unlisted extension goes invisible; asserting the ignored set is exactly the one
file meant to be ignored turns that silence into a failure. The checks ask git what it would do,
since the property under test is git's own behavior.
"""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

REFERENCE_VOLUME = "results/cs_e9_spcas9/reference_consensus_20A.mrc"
"""The one file under `results/` that is meant to be ignored, and the reason the allow-list exists.

Rebuildable from the deposition `rank_targets --fetch` records, so it is an input rather than a
measurement. Nothing in the manuscript resolves to it.
"""

EXPECTED_IGNORED = {
    REFERENCE_VOLUME,
    # Residue `.stamp` files from a removed driver; nothing reads or writes them. Listed so this
    # guard is green on a tree that has them, not because they are meant to stay.
    "results/e8_R_11992_hHb_O2_Supervisor_20221203_151033_38.stamp",
    "results/e8_T_11993_Supervisor_20230506_154234_49_humandeoxy_2.stamp",
    "results/e8_T_11993_Supervisor_20230506_155543_86_humandeoxy_1.stamp",
    "results/e8_T_11993_hHb_deoxy-1_Supervisor_20230329_170109_0.stamp",
}
"""Every path under `results/` that git is allowed to be silent about."""


def _ignored_under_results() -> set[str]:
    """Paths under `results/` that git is ignoring, one line per file.

    `git status --ignored` collapses an ignored directory to a single entry, which would hide how many
    files are inside it and what they are. `ls-files --others --ignored` does not.

    Returns:
        Repository-relative paths.
    """
    finished = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "results/"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return set(finished.stdout.split())


def test_the_scan_sees_the_file_it_is_supposed_to_see() -> None:
    """A guard whose scan returns nothing passes by finding nothing.

    An empty result here is indistinguishable between "the tree is clean" and "the command ran in the
    wrong directory", and the reference volume is the one entry that must be present on any checkout
    with the data in place.
    """
    if not (ROOT / REFERENCE_VOLUME).exists():
        pytest.skip(f"{REFERENCE_VOLUME} is not in this checkout, so there is nothing to be ignored")
    assert REFERENCE_VOLUME in _ignored_under_results(), (
        f"{REFERENCE_VOLUME} is on disk but git does not report it ignored; this scan is not working"
    )


def test_nothing_under_results_is_silently_ignored() -> None:
    """Anything under `results/` that is neither tracked nor reported by `git status` fails here.

    The fix is almost never to extend `EXPECTED_IGNORED`. A measurement gets its extension added to
    `.gitignore`'s allow-list so it is tracked with the rest of the evidence; a large input gets
    written outside `results/`.
    """
    unexpected = sorted(_ignored_under_results() - EXPECTED_IGNORED)
    assert not unexpected, (
        "these files under results/ are ignored, so git status will never mention them -- track the "
        "extension in .gitignore or write them elsewhere:\n" + "\n".join(unexpected)
    )


@pytest.mark.parametrize(
    "path",
    ["tests/data/fixture.json", "src/cryodisc/logs/run.json", "reproduction/experiments/panel/data/input.json"],
)
def test_the_root_only_ignores_are_anchored(path: str) -> None:
    """`/data/`, `/banks/` and `/logs/` must match at the repository root and nowhere else.

    Written without the leading slash -- which is how they started -- they match a directory of that
    name at ANY depth, so the day somebody adds `tests/data/` fixtures they are ignored rather than
    untracked, which is the same failure this file's other checks exist for, one directory over.
    """
    assert subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT, check=False).returncode != 0, (
        f"{path} is ignored; a root-only rule in .gitignore has lost its leading slash"
    )


def test_the_root_ignores_still_cover_the_root() -> None:
    """The other half of the rule above: anchoring must not stop them working where they are meant to."""
    for path in ("data/raw.star", "banks/poses.npy", "logs/run.out", "build/lib", "graphify-out/graph.json"):
        assert subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT, check=False).returncode == 0, (
            f"{path} is no longer ignored"
        )
