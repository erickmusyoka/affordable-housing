"""
Chart style for every figure in the project, following the dataviz skill's reference palette (light mode).

Rules applied here: categorical colors in fixed slot order (blue, orange, aqua; the first three validate for all-pair
forms, the first four for adjacent forms, see CLAUDE.md Section 14); sequential = one hue, light to dark; diverging =
blue and red around a neutral gray; 2 point lines; hairline solid grid; text in ink tokens, never the series color; one
axis only (no dual-axis charts); a legend whenever there are two or more series, with direct end labels in addition.
Aqua and yellow fall below 3:1 contrast on the light surface, so every chart prints its values in a table beside it.

Usage:  from viz import apply_style, SERIES, save_figure, end_label
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
NEUTRAL = "#f0efec"  # diverging midpoint

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow (fixed order, never cycled)
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
RED_BLUE = ["#104281", "#256abf", "#6da7ec", NEUTRAL, "#f1a48a", "#e34948", "#a82a2a"]  # blue (low) to red (high), gray middle
DEEMPHASIS = "#c3c2b7"

SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
DIVERGING = LinearSegmentedColormap.from_list("blue_red", RED_BLUE)

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "outputs" / "figures"


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": BASELINE, "axes.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 12,
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.markersize": 6, "figure.dpi": 110, "savefig.dpi": 160,
        "figure.constrained_layout.use": True,
    })


def no_xgrid(ax):
    ax.grid(axis="x", visible=False)


def end_label(ax, x, y, text, dx=4, color=INK_2, **kw):
    """Label at the end of a line in the text ink color. The colored mark beside it carries identity."""
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points", va="center", ha="left", fontsize=9, color=color, **kw)


def dot(ax, x, y, color):
    """Marker with a 2 point ring in the surface color so it stays legible where it overlaps a line."""
    ax.plot([x], [y], marker="o", markersize=6, color=color, markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none", zorder=5)


def save_figure(fig, name, part="01_eda"):
    out = FIGURES / part
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path)
    return path


# Tile-grid map: one equal square per state at an approximate geographic position (row, column). 51 distinct cells.
TILE_GRID = {
    "AK": (0, 0), "ME": (0, 10),
    "WI": (1, 5), "VT": (1, 9), "NH": (1, 10),
    "WA": (2, 0), "ID": (2, 1), "MT": (2, 2), "ND": (2, 3), "MN": (2, 4), "IL": (2, 5), "MI": (2, 6), "NY": (2, 8), "MA": (2, 9),
    "OR": (3, 0), "NV": (3, 1), "WY": (3, 2), "SD": (3, 3), "IA": (3, 4), "IN": (3, 5), "OH": (3, 6), "PA": (3, 7), "NJ": (3, 8), "CT": (3, 9),
    "CA": (4, 0), "UT": (4, 1), "CO": (4, 2), "NE": (4, 3), "MO": (4, 4), "KY": (4, 5), "WV": (4, 6), "VA": (4, 7), "MD": (4, 8), "RI": (4, 9),
    "AZ": (5, 1), "NM": (5, 2), "KS": (5, 3), "AR": (5, 4), "TN": (5, 5), "NC": (5, 6), "SC": (5, 7), "DC": (5, 8), "DE": (5, 9),
    "OK": (6, 3), "LA": (6, 4), "MS": (6, 5), "AL": (6, 6), "GA": (6, 7),
    "HI": (7, 0), "TX": (7, 3), "FL": (7, 7),
}
assert len(TILE_GRID) == 51 and len(set(TILE_GRID.values())) == 51


def tile_map(ax, values, cmap=None, vmin=None, vmax=None, label_values=False):
    """Draw the tile-grid map of a state-indexed Series. Returns the ScalarMappable for a colorbar."""
    import matplotlib.colors as mcolors
    cmap = cmap or SEQUENTIAL
    vmin = values.min() if vmin is None else vmin
    vmax = values.max() if vmax is None else vmax
    norm = mcolors.Normalize(vmin, vmax)
    for st, (r, c) in TILE_GRID.items():
        v = values.get(st)
        face = cmap(norm(v)) if v == v and v is not None else "#f0efec"
        ax.add_patch(plt.Rectangle((c, -r), 0.94, 0.94, facecolor=face, edgecolor="none", zorder=2))
        lum = 0.299 * face[0] + 0.587 * face[1] + 0.114 * face[2]
        ax.text(c + 0.47, -r + 0.47, st, ha="center", va="center", fontsize=9, color="#ffffff" if lum < 0.55 else INK, zorder=3)
    ax.set_xlim(-0.1, 11.1); ax.set_ylim(-7.1, 1.1); ax.set_aspect("equal"); ax.axis("off")
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)
