"""Shared plotting style so we don't re-style every figure from scratch.
Every figure script imports from here -- change a color once, every plot in
the report picks it up.

Picked these colors off a colorblind-safe generator rather than eyeballing
them (CVD Delta-E >= 8, normal-vision Delta-E >= 15 between slots we actually
put next to each other).
"""

import matplotlib.pyplot as plt

# categorical slots, light surface
BLUE = "#2a78d6"      # headline / "ours" result
ORANGE = "#eb6834"    # contrasting second series
GREEN = "#008300"     # validated / as-expected result
NEUTRAL = "#aeada4"   # reference / "paper" bars, kept deliberately dull

# light->dark ramp of BLUE for 1/2/4 bits-per-cell
BLUE_RAMP = ["#6da7ec", "#2a78d6", "#184f95"]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

FIGSIZE_SINGLE = (4.6, 3.2)   # one IEEE column
FIGSIZE_WIDE = (7.2, 3.2)     # both IEEE columns
DPI = 300


def setup():
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "sans-serif",
        "text.color": INK,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_MUTED,
        "ytick.left": False,
        "xtick.bottom": False,
        "legend.frameon": False,
        "legend.labelcolor": INK_SECONDARY,
        "savefig.bbox": "tight",
        "savefig.dpi": DPI,
    })


def save(fig, name: str, outdir: str = "paper/figs"):
    """Dump a PDF straight into paper/figs for \\includegraphics."""
    fig.savefig(f"{outdir}/{name}.pdf")
    plt.close(fig)
    print(f"  wrote {outdir}/{name}.pdf")


def bar_value_labels(ax, bars, fmt="{:.1f}", fontsize=9, color=None):
    """Number on top of each bar -- easier to just read it off than squint at the axis."""
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), xy=(b.get_x() + b.get_width() / 2, h),
                   xytext=(0, 3), textcoords="offset points",
                   ha="center", va="bottom", fontsize=fontsize,
                   color=color or INK)
