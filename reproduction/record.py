"""How a driver writes a number down, so the number can be traced back to the run that made it.

A result carries its provenance beside its payload, written by one function rather than by whatever
each driver remembered to include: the `git` commit (marked DIRTY when the tree is), the arguments,
and the driver. A cluster run is not in a checkout and `git` cannot answer there, so `git_commit`
falls back to a stamp from the environment; see `SOURCE_COMMIT_ENV`. `payload_of` reads a payload
back through the one envelope reading, so a bare pre-envelope file and a wrapped one unwrap the same
way in one place.
"""

import json
import math
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = 1
"""Version of the envelope below, not of any payload.

Bumped when a reader would break on an older file. Payload keys are the driver's own and are not
versioned here, because a driver that renames one has changed what it measured.
"""


SOURCE_COMMIT_ENV = "CRYOEM_SOURCE_COMMIT"
"""Environment variable holding the commit of a source tree that is not a checkout.

Set by the deploy scripts, so a cluster job records the commit of the tree pushed to it. Consulted
only when git cannot answer -- a real checkout outranks a stamp, so a stale variable in a shell
cannot shadow the tree in front of it.
"""


CODE_PATHS = (":/src", ":/reproduction", ":/deploy", ":/pyproject.toml")
"""What the dirty check looks at: the tree that decides what a run DOES.

Written with git's `:/` root-relative pathspec prefix, not plain paths, because `git_commit` runs
with `cwd` inside the package; a bare `src` there matches nothing and reports clean unconditionally.
Scoped rather than whole-tree so a modified RESULT (a batch fetching its own results, a driver
writing beside a tracked one) does not flag every subsequent run `-dirty` while the code is untouched.
`deploy` is included because the submit scripts choose the flags, which are part of the measurement.
"""


def git_commit(repository: str | Path | None = None) -> str:
    """The commit a run happened at, marked when the tree was dirty.

    Args:
        repository: Directory inside the working tree. `None` uses the package's own.

    Returns:
        The short hash, suffixed `-dirty` when anything under `CODE_PATHS` was modified or is
        untracked. Falling back to `SOURCE_COMMIT_ENV` when git cannot answer, and to `"unknown"`
        when that is unset or empty. Unknown rather than an exception: a result written outside a
        checkout is still a result, and refusing to write it would push drivers into recording
        nothing at all.
    """
    where = Path(repository) if repository is not None else Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=where, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", *CODE_PATHS],
            cwd=where,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        # An untracked driver still ships and runs (the deploy rsync copies whatever is not ignored),
        # so it must read dirty; `git status` alone would miss it.
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "--", *CODE_PATHS],
            cwd=where,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = dirty or untracked
    except (OSError, subprocess.CalledProcessError):
        return _stamped_commit(where)
    return f"{commit}-dirty" if dirty else commit


def _stamped_commit(where: Path) -> str:
    """The pushed tree's commit, from the environment or from `SOURCE_COMMIT` on disk.

    The file fallback is the point: `SOURCE_COMMIT_ENV` answers only when a runner remembered to
    export it, and a runner that forgets silently records `commit: unknown`, a provenance failure no
    later run repairs. `deploy/common.sh` writes `SOURCE_COMMIT` at the root of every synced tree, so
    reading it here covers every path into every driver, including runners that do not exist yet.

    Args:
        where: Directory inside the synced tree, from which the root is searched upward.

    Returns:
        The stamped commit, or `"unknown"`. Stripped, because the stamp arrives through a file and a
        shell and a trailing newline in a recorded commit defeats a later `grep` for the hash.
    """
    stamped = os.environ.get(SOURCE_COMMIT_ENV, "").strip()
    if stamped:
        return stamped
    # Search upward because `git_commit` runs with `cwd` inside the package while the stamp sits at
    # the tree root. Bounded at the root (marked by `pyproject.toml`): an unbounded walk could report
    # a stray `SOURCE_COMMIT` from a parent as this tree's provenance, and a wrong commit is worse
    # than a visibly-missing `unknown`.
    for directory in (where, *where.parents):
        try:
            candidate = directory / "SOURCE_COMMIT"
            if candidate.is_file():
                text = candidate.read_text().strip()
                if text:
                    return text
            if (directory / "pyproject.toml").is_file():
                break
        except OSError:
            break
    return "unknown"


def write_result(path: str | Path, payload: dict[str, Any], *, driver: str, arguments: dict[str, Any]) -> Path:
    """Write a result and the provenance that makes it traceable.

    Args:
        path: Where to write. Parent directories are created, a driver being run from wherever the
            data happens to sit.
        payload: What was measured. Written verbatim under `"result"`, so a reader indexes the
            driver's own keys and a rename is visible rather than silently absorbed.
        driver: The script that produced it, as a repository-relative path. Recorded so a result
            names the file that can regenerate it.
        arguments: The parsed arguments, as `vars(namespace)`. Paths are stringified; anything else
            that is not JSON-serialisable is recorded by its `repr`, since an argument that cannot
            be written is exactly the one worth seeing.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema": SCHEMA,
        "driver": driver,
        "commit": git_commit(),
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "arguments": {name: _plain(value) for name, value in arguments.items()},
        "result": _finite(payload),
    }
    # allow_nan=False so a non-finite value that `_finite` did not reach fails HERE, loudly, rather
    # than becoming a file that only some readers can open
    text = json.dumps(envelope, indent=2, default=_plain, allow_nan=False) + "\n"
    # Write to a sibling and rename: `write_text` truncates first, so a kill mid-write leaves a short
    # file with a new mtime that every "exists and is newer" skip check reads as finished work.
    # `os.replace` is atomic within a filesystem, so a reader sees the old file or the complete new one.
    # Sweep this result's own abandoned attempts first: a SIGKILL (standby preemption) skips the
    # `except` below, so each killed write leaves a `.partial` behind. Scoped to this destination's
    # siblings; two processes writing the same result file is already the bug.
    for abandoned in destination.parent.glob(f"{destination.name}.*.partial"):
        abandoned.unlink(missing_ok=True)
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.partial")
    try:
        temporary.write_text(text)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _finite(value: Any) -> Any:
    """The payload with every non-finite float as `null`, which JSON can hold and `NaN` cannot.

    `json.dumps` writes `NaN`/`Infinity` as bare words RFC 8259 has no production for, so a strict
    reader in another language cannot open the file; these arise legitimately (a zero-variance pool, a
    separation of zero), so they are mapped to `null` rather than refused. NaN and infinity collapse
    onto one value; the distinction survives in the neighboring field.

    Args:
        value: Any part of a payload.

    Returns:
        The same structure with non-finite floats replaced by `None`.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {name: _finite(item) for name, item in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(item) for item in value]
    # numpy/jax scalars are not `float` subclasses, so a NaN in one sails past the check above and
    # would hit `allow_nan=False`; convert them here so it becomes a recorded null, not a crash
    if hasattr(value, "tolist") and not isinstance(value, str | bytes | bytearray):
        return _finite(value.tolist())
    return value


def _plain(value: Any) -> Any:
    """Anything JSON cannot hold, as something it can -- and everything it can, untouched.

    The pass-through is the part that matters. A recorder that stringified every value would write
    `"0.05"` where a driver passed `0.05`, and a reader comparing an argument to a number would
    then be comparing it to a string and quietly finding no match.

    Args:
        value: The value.

    Returns:
        The value itself when JSON holds it natively, a list for anything with a `tolist`, a string
        for a path, and the `repr` otherwise -- an argument that cannot be written being exactly the
        one worth seeing.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(name): _plain(item) for name, item in value.items()}
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def payload_of(document: Any, path: str | Path = "<in memory>") -> dict[str, Any]:
    """The payload of an already-loaded record, whichever envelope it is wearing.

    `results/` holds two shapes: files wrapped by `write_result` under `"result"`, and bare payloads
    written before the envelope existed. This is the one reading that unwraps both -- `record.get(
    "result", record)` -- since the hand-written spellings did not agree on the bare shape.

    Args:
        document: A parsed JSON record.
        path: Where it came from, for the message only.

    Returns:
        The contents of `"result"` where the envelope has one, the whole document where it does not.

    Raises:
        ValueError: If the document, or its `"result"`, is not an object -- indexing a list or bare
            number gives `TypeError` several frames from the file that caused it.
    """
    if not isinstance(document, dict):
        raise ValueError(  # noqa: TRY004 -- the argument is fine; the CONTENT it carries is not
            f"{path} holds {type(document).__name__}, not a result object"
        )
    payload = document.get("result", document)
    if not isinstance(payload, dict):
        raise ValueError(  # noqa: TRY004 -- as above
            f"{path} has a 'result' that is {type(payload).__name__}, not an object"
        )
    return payload


def read_payload(path: str | Path) -> dict[str, Any]:
    """`payload_of` on a file. See it for which envelope shapes exist and why they disagree.

    Args:
        path: The record to read.

    Returns:
        The payload.

    Raises:
        ValueError: As `payload_of`, naming the file.
    """
    return payload_of(json.loads(Path(path).read_text()), path)
