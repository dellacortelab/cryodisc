"""Put the repo root on the venv's import path, so `reproduction/` imports in script mode.

`src/cryodisc` is the only tier that ships; `pip install -e .` installs it editable. The `reproduction/`
and `reproduction/experiments/` tiers are deliberately NOT packaged, but drivers import them. Tests get
the repo root from pytest's `pythonpath = ["."]`; scripts (`python reproduction/experiments/.../driver.py`)
get it from the `.pth` this writes. Without it a driver run as a script sees only its own directory and
dies at `ModuleNotFoundError: reproduction`.

Run once after creating the venv and `pip install -e .`:

    .venv/bin/python deploy/link_repo_root.py

Reversible: delete `repo_root.pth` from the venv's site-packages. This writes a venv artifact, never the
built wheel, so it changes nothing about what ships.
"""

import pathlib
import site


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    pth = pathlib.Path(site.getsitepackages()[0]) / "repo_root.pth"
    pth.write_text(f"{repo_root}\n")
    print(f"wrote {pth}\n  -> {repo_root}")


if __name__ == "__main__":
    main()
