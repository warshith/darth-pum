"""Shared plotting style: one consistent look across every figure in the
report. Every figure script imports from here rather than repeating style
setup, so a single change updates every plot at once.

Colors are a validated, colorblind-safe categorical set (CVD Delta-E >= 8,
normal-vision Delta-E >= 15 between any two slots actually used together
here), not picked by eye.
"""

import matplotlib.pyplot as plt

# -- validated categorical slots (light-surface set) ------------------------
BLUE = "#2a78d6"      # the headline / "ours" result
ORANGE = "#eb6834"    # a contrasting second series
GREEN = "#008300"     # a validated / correct-as-expected result
NEUTRAL = "#aeada4"   # reference / "paper" bars -- deliberately recessive

# an ordinal light->dark ramp of BLUE, for 1/2/4 bits-per-cell (low to high)
BLUE_RAMP = ["#6da7ec", "#2a78d6", "#184f95"]

# chart chrome
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
    """Save a PDF for \\includegraphics in the report."""
    fig.savefig(f"{outdir}/{name}.pdf")
    plt.close(fig)
    print(f"  wrote {outdir}/{name}.pdf")


def bar_value_labels(ax, bars, fmt="{:.1f}", fontsize=9, color=None):
    """Put the numeric value on top of each bar -- direct labels beat a
    reader having to read the value off the axis."""
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), xy=(b.get_x() + b.get_width() / 2, h),
                   xytext=(0, 3), textcoords="offset points",
                   ha="center", va="bottom", fontsize=fontsize,
                   color=color or INK)
