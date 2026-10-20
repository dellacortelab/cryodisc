"""The rules `reproduction/experiments/` exists to enforce, checked rather than trusted to its README.

No driver may import another driver: a function two drivers both want belongs in `src/` when a PyPI
user would need it, or in `reproduction/` when it is paper-reproduction machinery. Dependencies flow
`reproduction/experiments/` -> `reproduction/` -> `src/`, never back up, so `src/` may not import either tier above it.

Checkable from the import statements alone -- no script is executed or imported -- because a driver
needs data or a GPU to run and a layout rule must not. The one exception is the last check, which
starts one driver as a bare script to prove the deploy step put the repo-root packages on its path.
"""

import ast
import subprocess
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parent.parent / "reproduction/experiments"


def _driver_paths() -> list[Path]:
    """Every driver in `reproduction/experiments/`, ignoring caches.

    Returns:
        Sorted paths.
    """
    # crosssession/ holds cluster-run whitening / half-map helper scripts (sys.argv, no main()), not
    # figure or analysis drivers, so it is outside the driver contract these checks enforce.
    return sorted(
        p for p in EXPERIMENTS.rglob("*.py") if "__pycache__" not in p.parts and "crosssession" not in p.parts
    )


def test_there_are_drivers_to_check() -> None:
    """A layout guard over an empty tree passes by finding nothing, which is what green looked like
    the last time this repository's CI was checking zero files."""
    assert _driver_paths(), f"no drivers found under {EXPERIMENTS}; the checks below would be vacuous"


def _imported_modules(path: Path) -> set[str]:
    """Top-level module names a file imports, from its syntax alone.

    Args:
        path: The file to parse.

    Returns:
        The dotted names, `from x.y import z` contributing `x.y`. Relative imports contribute the
        empty string, which is enough to catch one -- a driver has no package to be relative to.
    """
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add("" if node.level else (node.module or ""))
    return names


def test_no_driver_imports_another_driver() -> None:
    """A shared function goes to `src/` or `reproduction/`, not to a sibling a benchmark then depends on.

    `reproduction/experiments/` is not a package, so a sibling is reached as a plain `score_pool` rather than a
    dotted `reproduction.experiments.realdata.score_pool`. Any bare name that IS another driver's stem is an
    offender, as is any `experiments.*` import -- the drivers' own tier is not a library to import.
    """
    stems = {path.stem for path in _driver_paths()}
    offenders = [
        f"{path.relative_to(EXPERIMENTS)} imports {name!r}"
        for path in _driver_paths()
        for name in _imported_modules(path)
        if name == "reproduction.experiments"
        or name.startswith("reproduction.experiments.")
        or name == ""
        or name in stems - {path.stem}
    ]
    assert not offenders, (
        "drivers must not import each other; move shared code to src/ or reproduction/:\n" + "\n".join(offenders)
    )


def test_no_driver_loads_another_driver_off_the_filesystem() -> None:
    """The other way round the rule: a driver reaching a sibling through
    `spec_from_file_location` is an import that leaves no import statement for the check above to see.
    The shared code goes to `src/` or `reproduction/` instead.
    """
    offenders = [
        f"{path.relative_to(EXPERIMENTS)} calls {call}"
        for path in _driver_paths()
        for call in ("spec_from_file_location", "SourceFileLoader")
        if call in path.read_text()
    ]
    assert not offenders, "load shared code from src/ or reproduction/, not a sibling's file:\n" + "\n".join(offenders)


def test_readme_states_the_rule() -> None:
    """The rule is documented where somebody adding a driver will look for it."""
    readme = (EXPERIMENTS / "README.md").read_text()
    assert "No experiment may import another experiment" in readme


def test_every_driver_takes_its_arguments_rather_than_hardcoding_them() -> None:
    """`reproduction/experiments/README.md`: paths come from arguments, not from a constant somebody edited
    before the last run. A hand-edited path in a benchmark whose state label IS the dataset is how a
    pool gets scored against the wrong state's template, which nothing downstream catches -- it
    reads as a smaller signal."""
    missing = [
        str(path.relative_to(EXPERIMENTS)) for path in _driver_paths() if "argparse" not in _imported_modules(path)
    ]
    assert not missing, "a driver must parse arguments rather than carry paths as constants:\n" + "\n".join(missing)


def test_every_driver_is_runnable_and_importable_as_neither() -> None:
    """Two halves of the same rule. A driver needs a `main` so the suite can exercise it at a toy
    size, and a `__main__` guard so importing it -- which is what that exercise does -- cannot start
    a run that wanted a GPU and four hours."""
    offenders = []
    for path in _driver_paths():
        source = path.read_text()
        tree = ast.parse(source)
        has_main = any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body)
        if not has_main:
            offenders.append(f"{path.relative_to(EXPERIMENTS)} defines no main()")
        if '__name__ == "__main__"' not in source:
            offenders.append(f'{path.relative_to(EXPERIMENTS)} has no `if __name__ == "__main__"` guard')
    assert not offenders, "\n".join(offenders)


def test_src_does_not_import_the_upper_tiers() -> None:
    """The published package must not depend on the tiers that only validate or drive it.

    `src/` is what somebody installs; `reproduction/` and `reproduction/experiments/` hold machinery that reproduces
    this paper, not machinery that ships. An import in this direction would make the package depend
    on its own test rig and could drag an optional heavyweight dependency into an install.
    """
    source = EXPERIMENTS.parent / "src"
    offenders = [
        f"{path.relative_to(source)} imports {name!r}"
        for path in sorted(source.rglob("*.py"))
        if "__pycache__" not in path.parts
        for name in _imported_modules(path)
        if name == "reproduction" or name.startswith("reproduction.")
    ]
    assert not offenders, "src/ may not import reproduction/ or reproduction/experiments/:\n" + "\n".join(offenders)


def test_a_driver_that_imports_a_repo_root_package_can_actually_be_run() -> None:
    """A driver imports `reproduction`, a repo-root package a bare script start does not put on the path:
    running `python reproduction/experiments/<thread>/<script>.py` puts only that driver's own directory on
    `sys.path`, so `reproduction` is not importable. `deploy/link_repo_root.py` writes the `repo_root.pth`
    that fixes this, and this is the one check that it worked.

    A subprocess is the only honest check, since the failure is a property of how the interpreter was
    started and any in-process import reproduces the wrong conditions. The failure is environmental --
    identical for every driver that crosses the boundary -- so one representative proves it rather
    than paying a process per driver over a tier nearly every driver now imports.
    """
    crossing = [
        path
        for path in _driver_paths()
        if any(
            isinstance(node, ast.ImportFrom) and (node.module or "").startswith("reproduction")
            for node in ast.walk(ast.parse(path.read_text()))
        )
    ]
    assert crossing, "no driver imports research, so this guard is watching nothing -- delete it or fix the scan"

    representative = crossing[0]
    finished = subprocess.run(
        [sys.executable, str(representative), "--help"], capture_output=True, text=True, timeout=180, check=False
    )
    assert finished.returncode == 0, f"{representative} cannot be run as a script:\n{finished.stderr[-2000:]}"
