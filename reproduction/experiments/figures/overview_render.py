"""Render the two conformers of the overview schematic as ChimeraX cartoons, in one camera frame.

    .venv/bin/python -m reproduction.experiments.figures.overview_render
Cartoons rather than the C-alpha traces `overview.py` falls back to, which read as a scribble at panel
size. It is a driver, not a `.cxc` file, because ChimeraX resolves relative paths against its own
working directory, so the script is written with absolute paths and run. Not called from `build.py`,
because ChimeraX on macOS cannot render headless; `overview.py` uses these PNGs when present and draws
traces otherwise, so this is an improvement to Figure 1 rather than a prerequisite.
Rationale: docs/rationale/figures.md
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_WIDTH = 700
_HEIGHT = 900

CHIMERAX = Path("/Applications/ChimeraX-1.11.1.app/Contents/bin/ChimeraX")
from reproduction.reporting.overview_pair import CARTOONS, PAIR, PROTEIN

BLUE, ORANGE = "#2a78d6", "#eb6834"

ANGLES = (0, 30, 60, 90, 120, 150)
"""Rotations about the vertical to try. `view` alone points the camera down whatever axis the
deposit happens to use, which is as likely as not the hinge axis -- and a hinge viewed end-on is
two blobs. The panel's whole job is to make the change legible, so the view is chosen by measuring
which of these shows the two states differing most, not by taking the first one."""

HEAD = """\
open {a}
open {b}
matchmaker #2/{chain} to #1/{chain}

hide all atoms
hide all cartoons
show #1 cartoons
show #2 cartoons
# thick ribbons and tube helices: at panel size a fine ribbon breaks into hairlines, and the reader
# needs the domains' shapes, not their secondary structure
cartoon style protein modeHelix tube sides 16 arrows true arrowsHelix false xsection oval width 2.6 thickness 1.0
cartoon style nucleic xsection oval width 2.6 thickness 1.2
# missing-segment dashes and their residue-count labels are annotation, not structure
hide pbonds
label delete
color #1 {blue}
color #2 {orange}

# ambient occlusion and a soft key light rather than flat shading with a heavy outline: a tube
# helix under a 1.6 px silhouette reads as a clip-art protein, and the panel is asking a reader to
# believe these are real coordinates. `full` adds the directional shadow that gives the fold depth.
lighting full
lighting shadows true intensity 0.55
lighting depthCue false
graphics silhouettes true width 1.4 color #30303080
set bgColor white

# ONE frame for both states at every angle: the two renders of a pair must differ by the hinge and
# by nothing else, so `view` is issued once and only `turn` moves after it.
view #1
zoom 0.9
"""

STEP = """
turn y {step} models #1,2
hide #2 cartoons
save {out_a} width {width} height {height} supersample 3 transparentBackground true
show #2 cartoons
hide #1 cartoons
save {out_b} width {width} height {height} supersample 3 transparentBackground true
show #1 cartoons
"""


def render(chimerax: Path, width: int, height: int) -> list[Path]:
    """Write the ChimeraX script and run it; returns the PNGs that appeared.

    Raises:
        SystemExit: If ChimeraX is not where it was expected, or ran without writing both images --
            it reports a failed command in its log and still exits 0, so the files are the check.
    """
    if not chimerax.exists():
        raise SystemExit(f"{chimerax} does not exist; pass --chimerax with the path to ChimeraX")
    CARTOONS[0].parent.mkdir(parents=True, exist_ok=True)
    # the sweep and the generated script are working files, not artifacts: only the two chosen
    # written beside the figure, which is a build output directory rather than source
    scratch = Path(tempfile.mkdtemp(prefix="overview-render-"))
    tried = [(angle, scratch / f"a{angle:03d}.png", scratch / f"b{angle:03d}.png") for angle in ANGLES]
    for _, first, second in tried:
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)

    script = scratch / "overview_render.cxc"
    body = HEAD.format(a=PAIR[0].resolve(), b=PAIR[1].resolve(), chain=PROTEIN, blue=BLUE, orange=ORANGE)
    previous = 0
    for angle, first, second in tried:
        body += STEP.format(
            step=angle - previous, out_a=first.resolve(), out_b=second.resolve(), width=width, height=height
        )
        previous = angle
    script.write_text(body + "\nexit\n")
    done = subprocess.run([str(chimerax), "--exit", str(script)], capture_output=True, text=True, check=False)

    missing = [path for _, first, second in tried for path in (first, second) if not path.exists()]
    if missing:
        raise SystemExit(
            "ChimeraX wrote no image for "
            + ", ".join(str(path) for path in missing)
            + "\n--- stdout ---\n"
            + done.stdout
            + "\n--- stderr ---\n"
            + done.stderr
        )

    # the view that shows the change: the two silhouettes overlap least. A hinge seen end-on gives
    # two near-identical outlines and scores near zero here, which is exactly what to reject.
    import matplotlib.image as mpimg

    scored = []
    for angle, first, second in tried:
        alpha = [mpimg.imread(path)[..., 3] for path in (first, second)]
        scored.append((float(np.abs(alpha[0] - alpha[1]).mean()), angle, first, second))
    scored.sort(reverse=True)
    for share, angle, _, _ in scored:
        print(f"  turn y {angle:3d}: silhouettes differ over {share:.4f} of the frame")
    _, best, first, second = scored[0]
    shots = list(CARTOONS)
    for source, target in zip((first, second), shots):
        target.write_bytes(source.read_bytes())
    print(f"  chose turn y {best}")
    return shots


def main() -> int:
    """Render both conformers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chimerax", type=Path, default=CHIMERAX)
    args = parser.parse_args()
    shots = render(args.chimerax, _WIDTH, _HEIGHT)
    print(
        f"cartoons in one camera frame, {PAIR[0].stem} and {PAIR[1].stem} superposed on chain {PROTEIN} by matchmaker"
    )
    for shot in shots:
        print(f"  wrote {shot} ({shot.stat().st_size // 1024} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
