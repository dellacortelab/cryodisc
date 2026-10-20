"""Draw the schematic of what the calculation does, from two structures to a population fraction.

    .venv/bin/python -m reproduction.experiments.figures.overview
The only figure that draws no measurement, but not a cartoon: the templates are rendered through the
same `CryoJAXModel` path the measured figures use, and the particles are real, drawn from the pools the
recovery figure scores. The pair is SpCas9 with the guide RNA displaced six and twelve bases (7Z4C
against 7Z4G), so the schematic's example is the paper's real-data system. Laid out as the dependency graph `EDGES`, so a box that moves takes its
arrows with it: two sources converge on the score hub, which fans out to the count and the fraction.
Rationale: docs/rationale/figures.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cryodisc.forward.optics import CTFParams
from cryodisc.specimen.structures import align_conformers
from reproduction.reporting.figures import (
    DOUBLE_COLUMN,
    LABEL_PT,
    OTHER,
    SERIES,
    TICK_PT,
    save,
    use_house_style,
)
from reproduction.reporting.numbers import Ledger
from reproduction.reporting.overview_pair import CARTOONS, PAIR, PROTEIN

BLUE, ORANGE = SERIES[0], SERIES[1]
PARTICLES = Path("results/overview_cas9_particles.npy")
"""Real particles from both pools, written by `overview_particles.py` on ribo."""
SCORES = (Path("results/rerun_cas9_hoRepro_6nt.npy"), Path("results/rerun_cas9_hoRepro_12nt.npy"))
"""Measured Lambda on each pure pool, the same pair of files the recovery figure's panel (a) draws."""
MIX = 0.6
"""The illustrative mixture weight of A in the drawn score distribution; the components are measured."""
# CARTOONS are preferred over the C-alpha traces, which read as scribbles at panel size, but optional:
# ChimeraX on macOS cannot render without a GUI session, so absent renders fall back to the traces
SEED = 20260825
BOX, PIXEL = 96, 2.6
"""Twice the particles' 1.3 A/px at half their 192 box, so templates and particles show one field."""
FIGURE_SIZE = (DOUBLE_COLUMN, DOUBLE_COLUMN / 2.25)  # 2.25:1, the schematic's aspect; height tracks width
VIEWS = np.array(
    [[0.0, 70.0, 0.0], [55.0, 100.0, 25.0], [120.0, 45.0, 80.0], [200.0, 115.0, 140.0]],
    dtype=np.float32,
)
"""Four orientations, drawn as four template columns and as four particles; the ellipsis beside each
grid says the real bank and the real pool are larger than what fits."""

GUTTER = 0.030
WIDTHS = (0.150, 0.222, 0.250, 0.252)
"""One width per column, set by what has to fit: two traces, eight template tiles, three equations.

No arrow carries a label: a gutter wide enough for a legible one costs the boxes more width than
the label is worth, so the renderer is named under the title of the box whose contents it made and
`d^2` is named by the equation that uses it."""

FULL, HIGH, LOW = (0.02, 0.98), 0.7525, 0.2475
TOP, BOT = (0.525, 0.98), (0.02, 0.475)
FOOT = 0.037
HEAD, SUBGAP, CONTENT = 0.026, 0.010, 0.014
# where a box has them, the subtitle names its input or source and the footer the thing to take away
"""Box top to title top, title to subtitle, and heading to contents, as figure fractions. Titles are
placed by their TOP edge and later rows by the measured bottom of the one above, so a two-line title
keeps the same margin as a one-line one instead of growing into it."""
GRID = (0.026, 0.320)
"""A tile grid's slot, as offsets from its box's floor. Shared by the bank and the pool: both are
width-limited at the same width, so one slot puts them at the same height in their boxes and the
`CryoJAX` line under one title costs alignment nothing -- it fills space the pool leaves blank."""
"""Every box's italic footer sits FOOT above its own floor, so they read as lines and not as eight
independent placements."""

EDGES = (
    ("structures", "banks", HIGH),
    ("banks", "scores", HIGH),
    ("particles", "scores", LOW),
    ("scores", "count", HIGH),
    ("scores", "fraction", LOW),
)
"""The graph, as (from, to, height). Two sources converge on `scores`, which then fans out: the same
two heights carry arrows in on its left and out on its right."""


def _traces():
    """The two conformers' C-alpha coordinates, superposed, in their most informative plane."""
    pair = align_conformers(PAIR[0], PAIR[1], assembly=None, chains=(PROTEIN,))
    trace = []
    for state in (pair.state_a, pair.state_b):
        alpha = state[state["atom"] == "CA"]
        trace.append(alpha[["x", "y", "z"]].to_numpy())
    both = np.vstack(trace)
    # the plane of largest spread over BOTH states, so the domain swing is in view rather than
    # projected away -- picking a coordinate pair by hand shows a different motion for every entry
    _, _, basis = np.linalg.svd(both - both.mean(0), full_matrices=False)
    return [(coords - both.mean(0)) @ basis[:2].T for coords in trace], pair


def _stacked(traces):
    """A above B, each turned upright, with the gap set by their own height and not by hand.

    Turned because the pair's long axis is horizontal, and the box holding them is a tall narrow
    one: upright and stacked, the two of them fill it instead of sitting as two wide slivers.
    """
    turned = [np.column_stack((-coords[:, 1], coords[:, 0])) for coords in traces]
    half = max(np.ptp(turned[0][:, 1]), np.ptp(turned[1][:, 1])) / 2
    gap = 0.12 * half
    return [
        turned[index] - [0.0, turned[index][:, 1].mean() - sign * (half + gap)]
        for index, sign in enumerate((1.0, -1.0))
    ]


def _banks():
    """Real templates for both states at `VIEWS`, of the assembly the recovery figure scores.

    At the drawn particles' median defocus, read from their record, so both carry one contrast.
    """
    record = json.loads(PARTICLES.with_suffix(".json").read_text())["result"]["by_state"]
    defocus = float(np.median([value for state in record.values() for value in state["kept_defocus"]]))
    pair = align_conformers(PAIR[0], PAIR[1], assembly="1")
    model_a, model_b = pair.models_on_grid(box_size=BOX, pixel_size=PIXEL)
    ctf = CTFParams(defocus_in_angstroms=defocus)
    return (np.asarray(model_a.forward_batch(VIEWS, ctf)), np.asarray(model_b.forward_batch(VIEWS, ctf)))


DISPLAY_LOWPASS = 18.0
"""Angstrom. Particles are low-pass filtered for display only (caption): at raw SNR they read as static."""


def _binned(image):
    """A particle Fourier-cropped to `BOX` at the templates' pixel size, then low-passed for display."""
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    edge = (image.shape[0] - BOX) // 2
    spectrum = spectrum[edge : edge + BOX, edge : edge + BOX]
    k = np.fft.fftshift(np.fft.fftfreq(BOX, d=PIXEL))
    radius = np.hypot(*np.meshgrid(k, k))
    return np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * np.exp(-((radius * DISPLAY_LOWPASS) ** 2) / 2))))


def _density(scores, grid):
    """A smoothed histogram of `scores` on `grid`, integrating to one; Scott's bandwidth."""
    step = grid[1] - grid[0]
    counts, _ = np.histogram(scores, bins=np.append(grid - step / 2, grid[-1] + step / 2), density=True)
    width = 1.06 * scores.std() * len(scores) ** -0.2 / step
    kernel = np.exp(-0.5 * (np.arange(-4 * width, 4 * width + 1) / width) ** 2)
    return np.convolve(counts, kernel / kernel.sum(), mode="same")


def _panel(figure, rect):
    """A bare axes with no frame, in figure coordinates."""
    axes = figure.add_axes(rect)
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)
    return axes


def columns():
    """Left and right edge of each column: the columns and one shared gutter, centered in the figure so
    the margins are symmetric and the end boxes clear the edges."""
    total = sum(WIDTHS) + GUTTER * (len(WIDTHS) - 1)
    edges = np.cumsum([(1.0 - total) / 2.0] + [w for pair in zip(WIDTHS, [GUTTER] * len(WIDTHS)) for w in pair])
    return tuple((edges[2 * i], edges[2 * i + 1]) for i in range(len(WIDTHS)))


def fit(region, aspect):
    """The largest rect inside `region` that shows `aspect` (data height / width) undistorted.

    Figure coordinates are not square, so the aspect is converted through the figure's own inches
    before the fit; otherwise a "square" panel comes out as wide as the figure is.
    """
    x0, y0, x1, y1 = region
    room_wide, room_high = (x1 - x0) * FIGURE_SIZE[0], (y1 - y0) * FIGURE_SIZE[1]
    inches = (room_wide, aspect * room_wide) if room_high / room_wide > aspect else (room_high / aspect, room_high)
    width, height = inches[0] / FIGURE_SIZE[0], inches[1] / FIGURE_SIZE[1]
    return [x0 + (x1 - x0 - width) / 2, y0 + (y1 - y0 - height) / 2, width, height]


def pack(region, rows, columns_, head=0.0, tail=0.0, foot=0.0, gap=0.16):
    """Square tiles filling `region`, as large as fit; returns their rects and the margins.

    `head` and `tail` reserve that many tile-widths at the left and right, for row labels and for an
    ellipsis, and `foot` that many tile-heights at the bottom, so each is part of the packing rather
    than laid on top of it and hoped for.
    """
    x0, y0, x1, y1 = region
    across = columns_ + (columns_ - 1) * gap + (head + tail) * (1.0 + gap)
    down = rows + (rows - 1) * gap + foot * (1.0 + gap)
    side = min((x1 - x0) * FIGURE_SIZE[0] / across, (y1 - y0) * FIGURE_SIZE[1] / down)
    step, wide, high = side * (1.0 + gap), side / FIGURE_SIZE[0], side / FIGURE_SIZE[1]
    left = x0 + ((x1 - x0) - side * across / FIGURE_SIZE[0]) / 2 + head * step / FIGURE_SIZE[0]
    top = y1 - ((y1 - y0) - side * down / FIGURE_SIZE[1]) / 2
    rects = [
        [left + step / FIGURE_SIZE[0] * column, top - high - step / FIGURE_SIZE[1] * row, wide, high]
        for row in range(rows)
        for column in range(columns_)
    ]
    anchors = {
        "head_x": left - side * gap / FIGURE_SIZE[0],
        "tail_x": left + (columns_ * step + tail * side / 2) / FIGURE_SIZE[0],
        "middle_y": top - (rows * side + (rows - 1) * side * gap) / 2 / FIGURE_SIZE[1],
        "center_x": left + (columns_ * side + (columns_ - 1) * side * gap) / 2 / FIGURE_SIZE[0],
        "top_y": top + side * gap / FIGURE_SIZE[1],
        "foot_y": top - (rows * step + foot * side / 2) / FIGURE_SIZE[1],
    }
    return rects, anchors


def draw(out: Path) -> list[Path]:
    """Render the schematic and write it as pdf and png."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    use_house_style()
    rng = np.random.default_rng(SEED)
    traces, _ = _traces()
    bank_a, bank_b = _banks()
    COLUMNS = columns()
    sizes: dict[str, float] = {}
    # every display equation, its tint layers included, so all can be set to one size once all are fitted
    equations: list = []
    # where each node of EDGES sits: column index and vertical band
    PLACE = {
        "structures": (0, FULL),
        "banks": (1, TOP),
        "particles": (1, BOT),
        "scores": (2, FULL),
        "count": (3, TOP),
        "fraction": (3, BOT),
    }

    figure = plt.figure(figsize=FIGURE_SIZE)
    stage = _panel(figure, [0.0, 0.0, 1.0, 1.0])
    stage.set_xlim(0, 1)
    stage.set_ylim(0, 1)
    stage.patch.set_alpha(0.0)

    # the stage spans 0-1 over a non-square figure, so a corner radius in figure fractions draws as an
    # ellipse; mutation_aspect rescales the box before rounding and back after, making it a circle
    SQUARE = FIGURE_SIZE[0] / FIGURE_SIZE[1]

    def box(node):
        column, (y0, y1) = PLACE[node]
        x0, x1 = COLUMNS[column]
        stage.add_patch(
            FancyBboxPatch(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                boxstyle="round,pad=0.004,rounding_size=0.011",
                mutation_aspect=SQUARE,
                facecolor="white",
                edgecolor=OTHER,
                lw=0.7,
                zorder=0,
            )
        )
        return (x0 + x1) / 2

    def inside(node, y0, y1):
        """The region a box's contents may use: its column, minus the padding."""
        x0, x1 = COLUMNS[PLACE[node][0]]
        return (x0 + 0.010, y0, x1 - 0.010, y1)

    def floor(node):
        """The italic footer's height for this box: FOOT above its own floor."""
        return PLACE[node][1][0] + FOOT

    def bottom(artist):
        """Where a drawn text ends, as a figure fraction, so the next row can hang below it."""
        return artist.get_window_extent(figure.canvas.get_renderer()).y0 / figure.bbox.height

    def title(node, text, subtitle=None):
        """The heading HEAD below the box top, and its subtitle SUBGAP under it; returns their bottom."""
        column, (_, top) = PLACE[node]
        center = sum(COLUMNS[column]) / 2
        head = stage.text(
            center, top - HEAD, text, ha="center", va="top", fontsize=LABEL_PT, color="0.15", weight="bold"
        )
        if subtitle is None:
            return bottom(head)
        sub = stage.text(
            center,
            bottom(head) - SUBGAP,
            subtitle,
            ha="center",
            va="top",
            fontsize=TICK_PT,
            color=OTHER,
            style="italic",
        )
        return bottom(sub)

    def note(node, text, size=TICK_PT):
        column, _ = PLACE[node]
        stage.text(
            sum(COLUMNS[column]) / 2,
            floor(node),
            text,
            ha="center",
            va="center",
            fontsize=size,
            color=OTHER,
            style="italic",
        )

    def formula(node, y, tex, size=8.6):
        """A display equation centered in a box, shrunk until it fits the box's own width."""
        x0, x1 = COLUMNS[PLACE[node][0]]
        room = (x1 - x0) - 0.020
        artist = stage.text((x0 + x1) / 2, y, tex, ha="center", va="center", fontsize=size, color="0.20")
        equations.append(artist)
        renderer = figure.canvas.get_renderer()
        while size > 4.6:
            wide = artist.get_window_extent(renderer).width / figure.bbox.width
            if wide <= room:
                break
            size *= min(0.95, room / wide)
            artist.set_fontsize(size)
        return size

    def tinted(node, y, ink, layers, size=8.6):
        """A formula() whose state symbols keep their state's color, as mathtext has no \\color.

        Each layer repeats the formula, same structure, with every glyph but its own under \\phantom, so
        it registers exactly; layers sit beneath `ink`, which covers the bars and brackets they repeat.
        """
        size = formula(node, y, ink, size)
        for tex, color in layers:
            center = sum(COLUMNS[PLACE[node][0]]) / 2
            equations.append(stage.text(center, y, tex, ha="center", va="center", fontsize=size, color=color, zorder=2))
        return size

    def crop(image, keep=0.62):
        """The middle of a render. A box sized for the pair's radius is mostly solvent."""
        edge = int(image.shape[0] * (1.0 - keep) / 2)
        return image[edge : image.shape[0] - edge, edge : image.shape[1] - edge]

    def shared(images):
        """One gray scale for a set of tiles; stretched per tile, equal backgrounds read as different."""
        pixels = np.concatenate([crop(image).ravel() for image in images])
        return pixels.min(), pixels.max()

    def tile(image, rect, color, limits):
        panel = _panel(figure, rect)
        panel.imshow(crop(image), cmap="Greys_r", interpolation="bilinear", vmin=limits[0], vmax=limits[1])
        for spine in panel.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(0.5)

    def dots(anchors, vertical=True):
        """More of the same: under the grid when it grows downward, beside it when it grows right."""
        x, y = (anchors["center_x"], anchors["foot_y"]) if vertical else (anchors["tail_x"], anchors["middle_y"])
        stage.text(x, y, "⋮" if vertical else "⋯", ha="center", va="center", fontsize=LABEL_PT, color=OTHER)

    # ---- the two conformers, one above the other, in one camera frame ----
    box("structures")
    heading = title("structures", "Two candidate\nstructures", "PDB models")
    # each state gets half the box, its label on top, so a width-limited render takes the full width
    # the label sits at the render's top-left, inside its frame, where a globular fold leaves the corner
    # empty; dropped by DROP so the heading above and the render below both get air
    DROP, SPLIT = 0.022, 0.018
    region = inside("structures", floor("structures") + 0.010, heading - CONTENT)
    half = (region[3] - region[1]) / 2
    if all(path.exists() for path in CARTOONS):
        import matplotlib.image as mpimg

        shots = [mpimg.imread(path) for path in CARTOONS]
        # trim the transparent margin ChimeraX leaves, by the UNION of the two alphas: cropping each
        # render to its own bounds would re-frame them independently and undo the shared camera
        cover = np.maximum(shots[0][..., 3], shots[1][..., 3]) > 0.02
        rows, cols = np.flatnonzero(cover.any(1)), np.flatnonzero(cover.any(0))
        window = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        shots = [shot[window] for shot in shots]
        # a quarter turn applied to BOTH keeps the camera shared; whichever way fills a slot more wins
        room = (half - DROP - SPLIT) * FIGURE_SIZE[1] / ((region[2] - region[0]) * FIGURE_SIZE[0])
        upright = [np.rot90(shot) for shot in shots]
        pairs = {
            abs(np.log(shot.shape[0] / shot.shape[1] / room)): candidate
            for candidate, shot in ((shots, shots[0]), (upright, upright[0]))
        }
        shots = pairs[min(pairs)]
        for index, (shot, color, label) in enumerate(zip(shots, (BLUE, ORANGE), ("A", "B"))):
            top = region[3] - index * half
            slot_ = (region[0], top - half + SPLIT, region[2], top - DROP)
            stage.text(region[0] + 0.004, top - DROP, label, color=color, fontsize=TICK_PT, ha="left", va="top")
            panel = _panel(figure, fit(slot_, shot.shape[0] / shot.shape[1]))
            panel.imshow(shot)
            panel.patch.set_alpha(0.0)
    else:
        print(
            "no ChimeraX renders in figures/assets/; drawing C-alpha traces instead. "
            "Run reproduction/experiments/figures/overview_render.py on a machine with a ChimeraX GUI "
            "session to produce them."
        )
        laid = _stacked(traces)
        both = np.vstack(laid)
        head = 0.10 * np.ptp(both[:, 1])
        axes = _panel(figure, fit(region, (np.ptp(both[:, 1]) + head) / np.ptp(both[:, 0])))
        for coords, color, label in zip(laid, (BLUE, ORANGE), ("A", "B")):
            axes.plot(coords[:, 0], coords[:, 1], color=color, lw=0.75, alpha=0.95, solid_capstyle="round")
            # left of the trace, not at its leftmost atom: a non-convex outline puts that atom inside it
            axes.text(
                both[:, 0].min() - 0.06 * np.ptp(both[:, 0]),
                coords[:, 1].mean(),
                label,
                color=color,
                fontsize=TICK_PT,
                weight="bold",
                ha="right",
                va="center",
            )
        axes.set_xlim(both[:, 0].min(), both[:, 0].max())
        axes.set_ylim(both[:, 1].min(), both[:, 1].max() + head)
        axes.set_aspect("equal")
        axes.patch.set_alpha(0.0)

    # ---- the banks: a column per orientation, a row per state, so a column is a matched pair ----
    box("banks")
    top_banks = title("banks", "Template banks", "CryoJAX")
    note("banks", r"Near-uniform orientations $\phi$")
    # center the tile grid between the subtitle's bottom and the footer, so the gap above it equals the
    # gap below it instead of the grid being anchored to the floor with a large, title-dependent top gap
    half = (GRID[1] - GRID[0]) / 2
    mid = (top_banks + floor("banks") + 0.02) / 2
    rects, anchors = pack(inside("banks", mid - half, mid + half), 2, len(VIEWS), head=0.44, tail=0.62)
    limits = shared([*bank_a, *bank_b])
    for index, rect in enumerate(rects):
        bank, color = (bank_a, BLUE) if index < len(VIEWS) else (bank_b, ORANGE)
        tile(bank[index % len(VIEWS)], rect, color, limits)
    for row, (state, color) in enumerate((("A", BLUE), ("B", ORANGE))):
        rect = rects[row * len(VIEWS)]
        stage.text(
            anchors["head_x"],
            rect[1] + rect[3] / 2,
            "$\\mu_" + state + "$",
            ha="right",
            va="center",
            fontsize=TICK_PT,
            color=color,
        )
    dots(anchors, vertical=False)

    # ---- the pool: a mixture whose labels nobody knows ----
    box("particles")
    top_particles = title("particles", "Particle images", "Cryo-EM experiment")
    note("particles", "Assumed mixture of A and B")
    mid = (top_particles + floor("particles") + 0.02) / 2  # centered as the bank's grid is
    rects, anchors = pack(
        inside("particles", mid - half, mid + half),
        2,
        len(VIEWS),
        head=0.44,
        tail=0.62,
    )
    # a real pool holds many particles at every orientation and in either state, so both are drawn
    # at random: the point of the box is that which is which is unknown
    grains = [_binned(image) for image in rng.permutation(np.load(PARTICLES))[: len(rects)]]
    limits = shared(grains)
    for rect, grain in zip(rects, grains):
        # black: an unlabeled mixture, belonging to neither state's color
        tile(grain, rect, "0.10", limits)
    stage.text(anchors["head_x"], anchors["middle_y"], "$Y_i$", ha="right", va="center", fontsize=TICK_PT, color="0.15")
    dots(anchors, vertical=False)

    # ---- the hub: both sources converge, and everything downstream is a functional of Lambda ----
    box("scores")
    heading = title("scores", "Scores and their separation")
    note("scores", r"$d'^2$ from simulated particles")
    # the hub holds four things between its title and footer; one shared gap keeps the equations
    # the same distance from the diagram as from the frame
    # the diagram is width-limited, so its slot is the height it will actually draw at plus the room
    # its axis label hangs into; a guessed height left the surplus as a gap under it
    LABEL, ASPECT = 0.046, 0.52
    # read in order: the score, its distribution with gap and spread marked, then d'^2 built from them
    HEIGHTS = (
        ("ratio", 0.070),
        ("diagram", fit(inside("scores", 0.0, 1.0), ASPECT)[3] + LABEL),
        ("separation", 0.040),
    )
    ceiling, base = heading - CONTENT, 0.098
    spare = (ceiling - base - sum(h for _, h in HEIGHTS)) / (len(HEIGHTS) + 1)
    slot, edge = {}, ceiling
    for name, height in HEIGHTS:
        edge -= spare
        slot[name] = (edge - height, edge)
        edge -= height

    # equations (A6a)-(A6b) as one ratio. <.>_phi is the integral against the uniform pose prior,
    # which cancels between numerator and denominator and so is not written.
    head, tail = r"\langle\, P(Y_i \mid ", r"(\phi)) \,\rangle_\phi"
    sizes["score"] = tinted(
        "scores",
        sum(slot["ratio"]) / 2,
        rf"$\Lambda_i = \log \dfrac{{{head}\phantom{{\mu_A}}{tail}}}{{{head}\phantom{{\mu_B}}{tail}}}$",
        [
            (
                (
                    rf"$\phantom{{\Lambda_i = \log}} \dfrac{{\phantom{{{head}}}\mu_A\phantom{{{tail}}}}}"
                    rf"{{\phantom{{{head}\mu_B{tail}}}}}$"
                ),
                BLUE,
            ),
            (
                (
                    rf"$\phantom{{\Lambda_i = \log}} \dfrac{{\phantom{{{head}\mu_A{tail}}}}}"
                    rf"{{\phantom{{{head}}}\mu_B\phantom{{{tail}}}}}$"
                ),
                ORANGE,
            ),
        ],
    )

    # ONE distribution: the dataset is a mixture with unknown labels, so drawing two labeled curves
    # would assume the assignment this method refuses. The axis label sits inside the slot's spacing.
    where = fit(inside("scores", slot["diagram"][0] + LABEL, slot["diagram"][1]), ASPECT)
    hist = _panel(figure, where)
    # the measured pure pools, so every mark is the data's: the gap between their means and one
    # within-state SD, pooled as `statistics.information.deflection` pools it. No value is printed.
    pools = [np.load(path).astype(float) for path in SCORES]
    grid = np.linspace(*np.percentile(np.concatenate(pools), (0.5, 99.5)), 600)
    centers = [pool.mean() for pool in pools]
    sigma = np.sqrt(sum(pool.var(ddof=1) * len(pool) for pool in pools) / sum(len(pool) for pool in pools))
    parts = [weight * _density(pool, grid) for weight, pool in zip((MIX, 1 - MIX), pools)]
    scale = 0.72 / (parts[0] + parts[1]).max()
    parts = [part * scale for part in parts]
    hist.fill_between(grid, parts[0] + parts[1], color=OTHER, alpha=0.30, lw=0.0)
    hist.plot(grid, parts[0] + parts[1], color="0.25", lw=1.1)
    for part, center, color in zip(parts, centers, (BLUE, ORANGE)):
        hist.plot(grid, part, color=color, lw=0.75, ls=(0, (2.5, 1.6)))
        # to the axis, so each spread arrow starts on its own state's mean
        hist.plot([center, center], [0.0, 0.84], color=color, lw=0.6, alpha=0.7)
    hist.add_patch(
        FancyArrowPatch(
            (centers[1], 0.82),
            (centers[0], 0.82),
            arrowstyle="<|-|>",
            mutation_scale=5.0,
            lw=0.8,
            color="0.25",
            shrinkA=0,
            shrinkB=0,
        )
    )
    hist.text(np.mean(centers), 0.87, "gap", ha="center", va="bottom", fontsize=TICK_PT, color="0.20")
    heights = []
    for part, center, side, color in zip(parts, centers, (1, -1), (BLUE, ORANGE)):
        heights.append(np.interp(center + side * sigma, grid, part))
        hist.add_patch(
            FancyArrowPatch(
                (center, heights[-1]),
                (center + side * sigma, heights[-1]),
                arrowstyle="<|-|>",
                mutation_scale=4.0,
                lw=0.7,
                color=color,
                shrinkA=0,
                shrinkB=0,
            )
        )
    span = grid[-1] - grid[0]
    # outside the curve, off both mean lines, at the tip of the arrow it names
    hist.text(
        centers[0] + sigma + 0.02 * span,
        heights[0] + 0.03,
        "spread",
        ha="left",
        va="bottom",
        fontsize=TICK_PT,
        color="0.20",
    )
    hist.set_xlim(grid[0], grid[-1])
    hist.set_ylim(0, 1.02)
    hist.patch.set_alpha(0.0)
    stage.text(
        sum(COLUMNS[2]) / 2,
        where[1] - 0.022,  # hung from the drawn axis: a width-limited diagram need not fill its slot
        r"$\Lambda$",
        ha="center",
        va="center",
        fontsize=TICK_PT,
        color="0.15",
    )
    sizes["separation"] = formula(
        "scores", sum(slot["separation"]) / 2, r"$d'^{2} = (\mathrm{gap} \,/\, \mathrm{spread})^{2}$"
    )
    # the components named where they are drawn, so the caption's Fisher integral has its terms on the page
    for part, center, side, color, name in zip(parts, centers, (1, -1), (BLUE, ORANGE), ("A", "B")):
        x = center + side * 1.9 * sigma
        hist.text(
            x, np.interp(x, grid, part) + 0.05, rf"$f_{name}$", ha="center", va="bottom", fontsize=TICK_PT, color=color
        )

    # ---- branch one: the count, reachable with no images at all ----
    box("count")
    count_title = title("count", "Particles needed")
    note("count", "No images required")
    # eq:variance is the mean-score estimator's variance, which the MLE can only beat, and p(1-p) <= 1/4:
    # so this count suffices at any fraction (caption). Centered in the content area (title to footer),
    # not the box, so the title taking the top does not leave the equation sitting high
    sizes["count"] = formula(
        "count", (count_title + floor("count")) / 2, r"$N \geq \dfrac{1/4 + 1/d'^{2}}{\sigma_p^{2}}$"
    )

    # ---- branch two: the fraction, off the whole distribution and never off a label ----
    box("fraction")
    heading = title("fraction", "Population fraction")
    # the box's two outputs, stacked: the estimate, then its error bar. Hats and "obs" mark what is read
    # off data, against the count box's chosen sigma_p and bank d'^2
    # the two equations centered in the space the title leaves: from the title's bottom down to the box
    # bottom (this box has no footer, so floor's reserved FOOT would otherwise push them up toward the title)
    box_bottom = floor("fraction") - FOOT
    span_h = heading - box_bottom
    estimate_y, error_y = heading - 0.30 * span_h, box_bottom + 0.30 * span_h
    sizes["error"] = formula(
        "fraction", error_y, r"$\hat\sigma_p^{2} \approx \dfrac{\hat p(1-\hat p) + 1/d'^{2}_\mathrm{obs}}{N}$"
    )
    # eq:fraction, the mean posterior of state A, written out so the sum in the denominator shows p-hat
    # cannot cancel; <.>_i over particles, against the score's <.>_phi over orientations
    a_, b_, mid = r"\hat p\, ", r"(\Lambda_i)", r" + (1-\hat p)\, "
    # f_A and f_B here are the pools' score densities, not the colored templates, so the equation is
    # drawn in one ink rather than tinting them to the state colors
    sizes["fraction"] = formula(
        "fraction",
        estimate_y,
        rf"$\hat p = \left\langle \dfrac{{{a_}f_A{b_}}}{{{a_}f_A{b_}{mid}f_B{b_}}} \right\rangle_i$",
    )

    # ---- the graph's edges, drawn from EDGES so a box that moves takes its arrows with it ----
    # a box's rounded border is drawn BOX_PAD outside its nominal edge, so an arrow from the edge starts
    # inside the box; both ends clear the border by a further CLEAR
    BOX_PAD, CLEAR = 0.004, 0.005
    for source, target, height in EDGES:
        stage.add_patch(
            FancyArrowPatch(
                (COLUMNS[PLACE[source][0]][1] + BOX_PAD + CLEAR, height),
                (COLUMNS[PLACE[target][0]][0] - BOX_PAD - CLEAR, height),
                arrowstyle="-|>",
                mutation_scale=7.5,
                lw=0.9,
                color="0.35",
                zorder=6,
                shrinkA=0,
                shrinkB=0,
            )
        )

    # one size for every equation: the largest at which the tightest of them still fits its box
    common = min(sizes.values())
    for artist in equations:
        artist.set_fontsize(common)
    print(f"  equations drawn at {common:.1f} pt, set by the {min(sizes, key=sizes.get)} equation")
    if common < 6.0:
        print(f"!! equations shrank to {common:.1f} pt to fit")
    written = save(figure, out)
    plt.close(figure)
    return written


def main() -> int:
    """Draw the schematic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/manuscript/figures"))
    args = parser.parse_args()
    written = draw(args.out / "overview.pdf")
    val = Ledger()
    val("display_lowpass", DISPLAY_LOWPASS, ".0f")  # the particles' display filter, in the caption
    print("schematic: real conformers, templates and particles, no measured values")
    print(f"  {PAIR[0].stem}/{PAIR[1].stem} assembly 1, particles from {PARTICLES}, {len(EDGES)} graph edges")
    val.dump(args.out / "values" / "overview.json")
    print("  wrote " + ", ".join(str(path) for path in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
