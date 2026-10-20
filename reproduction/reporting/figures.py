"""House style for the manuscript figures, and the one save path they all take.

`reproduction/experiments/figures/*.py` may not import each other, so the style they share lives here. Every
figure is saved as both `.pdf` (for the journal) and `.png` (for the pandoc `.docx`) from one
`savefig` pair, so the two cannot disagree.

Rationale: docs/rationale/figures.md
"""

from pathlib import Path
from typing import Any, cast

import numpy as np

# Authored widths, in inches, which are also the SAVED widths: figures save at full canvas (see
# savefig.bbox below), and each figure's own tight/constrained layout fills that canvas, so a figure is
# exactly figsize wide. DOUBLE_COLUMN is the text width of the build's reference.docx (Letter, 1 in
# margins) -- NOT pandoc's narrower default column, which is why build.py passes that reference.docx;
# authoring at this width and never rescaling keeps the shared LABEL_PT type one size across every
# figure. SINGLE_COLUMN is the single-column half-width. Final journal widths would want a reference.docx
# that widens the page past 6.5 in to match.
SINGLE_COLUMN = 3.5
DOUBLE_COLUMN = 6.5

# Type scale, in points, one role per size so every figure follows the same rules instead of ad-hoc
# numbers. Figure type is smaller than the 12 pt body on purpose: a label sits beside a panel a few
# inches wide, not in a full-width text column, so body-sized type would swamp a multi-panel figure.
# One scale across all figures keeps a given role the same physical size regardless of panel count.
LABEL_PT = 7
"""Axis labels and panel letters."""
TICK_PT = 6
"""Tick labels, legends, and data annotations: R^2, contour shares, slopes, series and curve labels."""
FINE_PT = 5
"""Incidental fine print: sample counts and provenance-ish tags, subordinate to the data annotations."""

# Whitespace rule, in tight_layout units (a fraction of the font size): the same padding on every
# figure so the gaps read uniform across the set, and kept tight so the panels -- not the margins --
# fill the figure. PAD trims the canvas to the outermost labels; PANEL_GAP is the space between panels.
PAD = 0.4
PANEL_GAP = 1.2
GAP_IN = PANEL_GAP * LABEL_PT / 72.0
"""PANEL_GAP expressed in inches, for figures laid out with constrained_layout (whose pads are in
inches) rather than tight_layout (whose w_pad is in font-size units), so the gap is the same size."""

# Assigned in FIXED ORDER and never cycled; validated for color-vision distinguishability rather
# than chosen by eye.
#
# A scatter may use at most three: it interleaves every series with every other, where bars and
# lines only compare neighbors. More classes fold into `OTHER` or become a second panel.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
"""Categorical hues, in assignment order. Blue, orange, aqua."""

OTHER = "#8a8a86"
"""The fold-in gray for classes past the cap. Never a generated fourth hue."""

SEQUENTIAL = "#1b4f8a"
"""Single hue for magnitude, where the encoding is more-is-darker rather than identity."""

RULE = "0.72"
"""Reference lines, identity lines and baselines: recessive, never a series color."""

BAND = "0.88"
"""Fill for a quoted-scatter or error band, behind everything."""

STYLE: dict[str, Any] = {
    "figure.dpi": 150,
    "savefig.dpi": 400,
    # the bounding box is chosen per-save in save(): it crops the empty top/bottom band that a fixed
    # figsize leaves around square panels, but keeps the full authored width, so figures lose vertical
    # whitespace yet stay the same width and are never rescaled -- the type is one size across figures.
    "savefig.bbox": "standard",
    # type scale from the role constants above; figures are never rescaled, so these land at their
    # stated point size on the page and read the same across figures whatever their panel count
    "font.size": LABEL_PT,
    "axes.labelsize": LABEL_PT,
    "axes.titlesize": LABEL_PT,
    "xtick.labelsize": TICK_PT,
    "ytick.labelsize": TICK_PT,
    "legend.fontsize": TICK_PT,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "errorbar.capsize": 1.5,
    # Type 42 embeds the font as TrueType. The default Type 3 is a subsetted PostScript program that
    # several publishers reject outright and that no reviewer can select text out of.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
"""rcParams every manuscript figure sets, applied by `use_house_style`."""


def use_house_style() -> None:
    """Apply `STYLE` to the global matplotlib state.

    Imported inside the function because matplotlib is a `dev` extra: the prediction arm never draws
    anything, and a module-scope import would make it a hard dependency of the package.
    """
    import matplotlib.pyplot as plt

    # cast because RcParams keys are a Literal union, which no plain dict can satisfy
    plt.rcParams.update(cast(Any, STYLE))


def save(figure: Any, out: Path) -> list[Path]:
    """Write one figure as both `.pdf` and `.png`, creating the directory.

    The figure is cropped to its content VERTICALLY but kept at its authored width, so the top/bottom
    whitespace a fixed figsize leaves around square panels is removed while every figure keeps the same
    width and nothing is rescaled.

    Args:
        figure: The matplotlib figure.
        out: Target path. Any suffix is replaced; both formats are written beside it.

    Returns:
        The paths written, in the order pdf then png.
    """
    from matplotlib.transforms import Bbox

    out.parent.mkdir(parents=True, exist_ok=True)
    figure.canvas.draw()  # realize label extents so the content box is measured correctly
    content = figure.get_tightbbox(figure.canvas.get_renderer())  # inches, from the figure origin
    width, _ = figure.get_size_inches()
    crop = Bbox.from_extents(0.0, content.y0 - 0.02, width, content.y1 + 0.02)
    written = []
    for suffix in (".pdf", ".png"):
        path = out.with_suffix(suffix)
        figure.savefig(path, bbox_inches=crop)
        written.append(path)
    return written


def label_panels(*axes) -> None:
    """Mark each axes with its panel letter, in reading order.

    Set as a left-aligned title rather than text at fixed axes coordinates: a title cannot collide
    with the data whatever the axis limits do, and it moves with `tight_layout`.

    Args:
        *axes: The panels, in the order the caption names them. A panel drawn but not named is simply
            not passed.
    """
    for panel, letter in zip(axes, "abcdefgh", strict=False):
        panel.set_title(letter, loc="left", fontweight="bold", fontsize=LABEL_PT, pad=3)


SHARES = (0.5, 0.9)
"""The contours drawn: the smallest regions holding these fractions of the pairs."""
CONTOUR_STYLE = {0.5: ("0.05", 1.2), 0.9: ("0.35", 0.7)}
"""Color and width per share: the inner line reads heavier, and a key names both, since inline labels
cut into a thin contour read as belonging to the line beside it."""


def shared_contours(panel, x, y, span, zorder: int, bw_method=None) -> None:
    """Light points with contours enclosing each of `SHARES` of them, in log space.

    Shading by density only says "more here"; a contour holding half the pairs says how many, and every
    outlier stays visible as its own point, which a count per bin would wash out. `bw_method` widens the
    kernel to keep a sparse tail from splitting off its own contour lobe; the points are drawn regardless.
    """
    from scipy.stats import gaussian_kde

    log_pair = np.vstack([np.log10(x), np.log10(y)])
    kernel = gaussian_kde(log_pair, bw_method=bw_method)
    # the density a point must exceed to be inside the region holding that share of the pairs
    levels = sorted(np.quantile(kernel(log_pair), 1 - share) for share in SHARES)
    grid = np.linspace(*np.log10(span), 200)
    gx, gy = np.meshgrid(grid, grid)
    surface = kernel(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    panel.scatter(x, y, s=5, color=SEQUENTIAL, alpha=0.35, linewidths=0.0, zorder=zorder)
    # ascending levels are descending shares: the lowest density bounds the largest region
    shares = sorted(SHARES, reverse=True)
    panel.contour(
        10**gx,
        10**gy,
        surface,
        levels=levels,
        zorder=zorder + 1,
        colors=[CONTOUR_STYLE[share][0] for share in shares],
        linewidths=[CONTOUR_STYLE[share][1] for share in shares],
    )
    for index, share in enumerate(SHARES):
        color, width = CONTOUR_STYLE[share]
        y_key = 0.86 - 0.055 * index
        panel.plot([0.04, 0.10], [y_key, y_key], transform=panel.transAxes, color=color, lw=width)
        panel.annotate(
            f"{share:.0%} of pairs", xy=(0.12, y_key), xycoords="axes fraction", fontsize=TICK_PT, va="center"
        )
