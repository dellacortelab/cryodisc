"""Checks on the result envelope, which is the only durable output any driver has.

A recorded number that cannot be read back is worse than one that was never written, because the run
that produced it has already finished. So what is tested here is that the file parses STRICTLY -- not
merely that Python can reopen it, Python's own reader accepting non-standard literals that other
readers reject.
"""

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from reproduction.record import (
    SOURCE_COMMIT_ENV,
    git_commit,
    payload_of,
    read_payload,
    write_result,
)

STRICT = json.JSONDecoder(parse_constant=lambda literal: pytest.fail(f"non-standard JSON literal {literal!r}"))
"""A reader that refuses `NaN`, `Infinity` and `-Infinity`, as RFC 8259 has no production for any.

`json.loads` accepts all three as an extension and so does `jq`, which is why a file carrying one reads
fine in-house and fails from another language -- `JSON.parse` rejects `NaN` outright. This decoder is
that stricter reader, so the assertion below is about the FILE and not about Python.
"""


def test_a_non_finite_measurement_is_recorded_as_null_rather_than_as_NaN(tmp_path):
    """These arise legitimately: `coherence` returns NaN on a pool of zero variance and
    `Extrapolation.local_exponents` on a point at its own limit. Mapped, not refused."""
    out = write_result(
        tmp_path / "r.json",
        {
            "coherence": {"value": float("nan"), "mean": 1.5},
            "exponents": [0.5, float("nan"), float("inf"), float("-inf")],
        },
        driver="x.py",
        arguments={},
    )

    text = Path(out).read_text()
    assert "NaN" not in text and "Infinity" not in text

    result = STRICT.decode(text)["result"]
    assert result["coherence"]["value"] is None
    assert result["coherence"]["mean"] == 1.5
    assert result["exponents"] == [0.5, None, None, None]


def test_finite_numbers_pass_through_as_numbers(tmp_path):
    """The sanitizer must not stringify or round what it walks past: a reader comparing a recorded
    value to a number would otherwise be comparing it to a string and quietly finding no match."""
    payload = {"amplitude": 53.57, "count": 858, "flag": True, "absent": None, "nested": {"d2": [1.0, 2.5]}}
    out = write_result(tmp_path / "r.json", payload, driver="x.py", arguments={"grid": [3, 6, 3]})

    result = STRICT.decode(Path(out).read_text())["result"]
    assert result == payload
    assert isinstance(result["amplitude"], float) and isinstance(result["count"], int)


def test_a_numpy_scalar_that_is_not_finite_also_lands_as_null(tmp_path):
    """`np.float32("nan")` is not a `float`, so it reaches `json.dumps` through `_plain`'s pass-through
    and would be written as a bare NaN by a sanitizer that only checked the builtin type."""
    out = write_result(
        tmp_path / "r.json",
        {"value": np.float32("nan"), "fine": np.float64(2.0)},
        driver="x.py",
        arguments={},
    )

    result = STRICT.decode(Path(out).read_text())["result"]
    assert result["value"] is None
    assert result["fine"] == 2.0


def test_an_array_of_measurements_lands_as_nulls_where_it_is_not_finite(tmp_path):
    """`Extrapolation.local_exponents` is written whole, and one NaN in it must not cost the file."""
    out = write_result(
        tmp_path / "r.json",
        {"exponents": np.array([0.63, np.nan, 0.91]), "d2": np.arange(3.0)},
        driver="x.py",
        arguments={},
    )

    result = STRICT.decode(Path(out).read_text())["result"]
    assert result["exponents"] == [pytest.approx(0.63), None, pytest.approx(0.91)]
    assert result["d2"] == [0.0, 1.0, 2.0]


def test_the_envelope_still_carries_its_provenance(tmp_path):
    out = write_result(
        tmp_path / "r.json", {"d2": 1.0}, driver="reproduction/experiments/x.py", arguments={"path": Path("/a/b")}
    )

    envelope = STRICT.decode(Path(out).read_text())
    assert envelope["driver"] == "reproduction/experiments/x.py"
    assert envelope["arguments"]["path"] == "/a/b"
    assert math.isfinite(envelope["result"]["d2"])
    assert "written_at" in envelope and "commit" in envelope


def test_a_tree_that_is_not_a_checkout_records_the_stamp_it_was_synced_with(tmp_path, monkeypatch):
    """The cluster case, and the reason this fallback exists: `sync.sh` pushes source by rsync, so the
    tree a job imports has no `.git` and `git rev-parse` fails there for a reason that has nothing to
    do with the commit being unknown. Every result this repository recorded from a cluster says
    `unknown`, which is a published number that cannot be attributed to code."""
    monkeypatch.setenv(SOURCE_COMMIT_ENV, "abc1234-dirty\n")

    assert git_commit(tmp_path) == "abc1234-dirty"


def test_a_real_checkout_outranks_the_stamp(monkeypatch):
    """Precedence, not preference. A stamp left in a shell must not shadow the tree in front of it --
    the variable is an assertion by whoever synced and git is a reading. Skipped where the suite is
    not running inside a checkout, since git has nothing to read there.
    """
    monkeypatch.setenv(SOURCE_COMMIT_ENV, "0000000")
    here = Path(__file__).parent
    if not any(parent.joinpath(".git").exists() for parent in (here, *here.parents)):
        pytest.skip("not a git checkout, which is the case the stamp exists for")

    assert git_commit(here) not in ("0000000", "unknown")


def _throwaway_checkout(root: Path) -> Path:
    """Build a repository with the layout `git_commit` scopes against, and return a dir inside it.

    A real one rather than a mock, because the thing under test is a git pathspec: `CODE_PATHS` uses
    the `:/` root-relative prefix precisely so it resolves the same from a nested working directory,
    and a mock of `subprocess.run` would assert the arguments while proving nothing about what git
    does with them. The returned path is DEEP -- the package directory, not the root -- since that is
    where `git_commit` actually runs, and a bare `src` pathspec would silently match nothing there.

    Args:
        root: An empty directory to make the repository in.

    Returns:
        The nested package directory to pass as `repository`.
    """
    package = root / "src" / "pkg"
    package.mkdir(parents=True)
    (package / "thing.py").write_text("x = 1\n")
    (root / "reproduction" / "experiments").mkdir(parents=True)
    (root / "reproduction" / "experiments" / "driver.py").write_text("y = 2\n")
    (root / "results").mkdir()
    (root / "results" / "measured.json").write_text('{"d2": 1.0}\n')

    run = lambda *a: subprocess.run(a, cwd=root, capture_output=True, check=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "test")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "initial")
    return package


def test_a_modified_result_does_not_make_a_run_dirty(tmp_path):
    """The false alarm this scoping exists to stop. A batch fetching its own results, or a driver
    writing beside one already tracked, modifies `results/` constantly; under a whole-tree check every
    run in that shell then records `-dirty` while the code was never touched. `-dirty` is the one
    marker separating a reproducible number from an unattributable one, so a version of it that fires
    on the normal case is a version people learn to ignore."""
    package = _throwaway_checkout(tmp_path)
    (tmp_path / "results" / "measured.json").write_text('{"d2": 2.0}\n')

    assert not git_commit(package).endswith("-dirty")


def test_a_modified_driver_does_make_a_run_dirty(tmp_path):
    """The case the flag is for, asserted from a NESTED directory so it also pins the `:/` pathspec:
    read relative to the package, a bare `experiments` matches nothing and every run reports clean."""
    package = _throwaway_checkout(tmp_path)
    (tmp_path / "reproduction/experiments" / "driver.py").write_text("y = 3\n")

    assert git_commit(package).endswith("-dirty")


def test_an_untracked_driver_makes_a_run_dirty(tmp_path):
    """`deploy/common.sh` rsyncs whatever git does not ignore, so an untracked driver SHIPS and RUNS.
    Recording that as clean is worse than recording it dirty, because it is a positive claim of
    reproducibility about code that is in no commit."""
    package = _throwaway_checkout(tmp_path)
    (tmp_path / "reproduction/experiments" / "new_driver.py").write_text("z = 4\n")

    assert git_commit(package).endswith("-dirty")


@pytest.mark.parametrize("stamp", ["", "   \n"])
def test_an_empty_stamp_is_unknown_and_not_an_empty_commit(tmp_path, monkeypatch, stamp):
    """An unset variable and a blank one mean the same thing, and neither is a commit. Blank because
    `cat` of a missing SOURCE_COMMIT under `|| echo unknown` can still deliver whitespace."""
    monkeypatch.setenv(SOURCE_COMMIT_ENV, stamp)

    assert git_commit(tmp_path) == "unknown"


def test_a_payload_is_read_back_through_the_envelope_and_without_one(tmp_path):
    """`results/` holds both shapes -- payloads wrapped under `"result"` and bare payloads from before
    the envelope existed -- so both have to come back as the payload and nothing else.
    """
    wrapped = write_result(tmp_path / "wrapped.json", {"d2": 1.5}, driver="d.py", arguments={})
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"d2": 1.5}))

    assert read_payload(wrapped) == {"d2": 1.5}
    assert read_payload(bare) == {"d2": 1.5}
    assert payload_of(json.loads(bare.read_text())) == {"d2": 1.5}


def test_the_bare_payload_is_the_payload_and_not_an_empty_dict(tmp_path):
    """The spelling this replaces, and the reason the migration was worth doing.

    Three hand-unwrapping spellings disagree on the BARE shape: `record["result"]` raises,
    `record.get("result", {})` yields empty, and only `record.get("result", record)` returns the
    payload. The first fails loudly; the second does not, and an empty payload draws an empty figure
    and skips every row with no error anywhere. Asserted against both wrong spellings directly,
    because on an enveloped file all three agree, which is why the disagreement went unnoticed.
    """
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"rows": [{"name": "a"}], "d2": 1.5}))
    document = json.loads(bare.read_text())

    assert payload_of(document) == document
    assert document.get("result", {}) == {}, "the spelling this replaces silently empties a bare file"
    with pytest.raises(KeyError):
        _ = document["result"]

    wrapped = json.loads(
        write_result(tmp_path / "w.json", {"rows": [{"name": "a"}]}, driver="d.py", arguments={}).read_text()
    )
    assert payload_of(wrapped) == wrapped.get("result", {}) == wrapped["result"], (
        "on an enveloped file all three agree, which is why the disagreement stayed invisible"
    )


def test_a_file_that_is_not_a_result_object_is_refused_by_name(tmp_path):
    """A list or a bare number indexes as `TypeError` several frames from the file that caused it."""
    for name, content in (("list.json", "[1, 2]"), ("scalar.json", "3"), ("badresult.json", '{"result": []}')):
        path = tmp_path / name
        path.write_text(content)
        # the message names the FILE, which is the thing a reader several frames away cannot see
        with pytest.raises(ValueError, match=rf"{name}.*not a"):
            read_payload(path)


def test_the_commit_stamp_is_read_from_disk_and_bounded_at_the_tree_root(tmp_path, monkeypatch):
    """A synced tree has no `.git`, so the stamp on disk is the only provenance it can offer.

    Exporting the variable in each runner is fragile -- a runner that forgets records
    `commit: unknown` -- so reading `SOURCE_COMMIT` from disk covers runners that do not exist yet.
    BOUNDED at the root, because an unbounded walk reaches `$HOME`: a stray stamp in a parent would
    otherwise be reported as this tree's commit, and a plausible wrong hash is worse than `unknown`.
    """
    from reproduction.record import SOURCE_COMMIT_ENV, _stamped_commit

    monkeypatch.delenv(SOURCE_COMMIT_ENV, raising=False)
    tree = tmp_path / "tree"
    (tree / "src" / "pkg").mkdir(parents=True)
    (tree / "pyproject.toml").write_text("[project]\n")
    deep = tree / "src" / "pkg"

    assert _stamped_commit(deep) == "unknown"

    # a stray stamp ABOVE the root must not be adopted
    (tmp_path / "SOURCE_COMMIT").write_text("stray9\n")
    assert _stamped_commit(deep) == "unknown", "read a stamp from outside the tree"

    # the tree's own stamp is found from deep inside it, newline stripped
    (tree / "SOURCE_COMMIT").write_text("right123\n")
    assert _stamped_commit(deep) == "right123"

    # an empty stamp is not an answer
    (tree / "SOURCE_COMMIT").write_text("  \n")
    assert _stamped_commit(deep) == "unknown"

    # and the environment outranks the file, being the more specific assertion
    monkeypatch.setenv(SOURCE_COMMIT_ENV, "fromenv")
    (tree / "SOURCE_COMMIT").write_text("onfile\n")
    assert _stamped_commit(deep) == "fromenv"
