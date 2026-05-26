"""
fig_milp_comparison.py — MILP vs. alternatives comparison table, section 2.5.5.

Rows: Greedy, Hungarian, GA, RL, MILP.
Columns: Optimality guarantee, Hard constraint support, Handles RS×1/NS×∞
         inventory limits, Scalable to 120 slots, Computation time.
Rating symbols: ✓ (green) = full support, ~ (orange) = partial, ✗ (red) = no support.
MILP row highlighted as the chosen method.

Saves to figures/pdf/ and figures/png/.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import FIG_PDF_DIR, FIG_PNG_DIR

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_NS    = "#61788C"
C_RS    = "#F2994B"
C_DARK  = "#2F3E4F"
C_MUTED = "#9CA5A6"
BG      = "#FFFFFF"

# Rating colours (background, text/symbol)
RATING_FULL = ("#C8EBC5", "#1E6B18", "✓")   # green   — full support
RATING_PART = ("#FDECC0", "#7A4E00", "~")   # orange  — partial
RATING_NONE = ("#FFD8D8", "#9E1717", "✗")   # red     — no support

# ---------------------------------------------------------------------------
# Table data
# ---------------------------------------------------------------------------
METHODS = [
    ("Greedy",     "sequential, path-dependent"),
    ("Hungarian",  "1-to-1 optimal, no extra constraints"),
    ("GA",         "heuristic, no optimality proof"),
    ("RL",         "approximate, reward-engineering required"),
    ("MILP",       "branch-and-bound, globally exact"),
]

CRITERIA = [
    "Optimality\nguarantee",
    "Hard constraint\nsupport",
    "Inventory limits\n(RS ×1,  NS ×∞)",
    "Scalable to\n120 slots",
    "Computation\ntime",
]

# Each row: list of (bg, tc, symbol_or_text) per criterion
F, P, N = RATING_FULL, RATING_PART, RATING_NONE
RATINGS = [
    # Greedy
    [N, P, F, F, (RATING_FULL[0], RATING_FULL[1], "< 1 ms")],
    # Hungarian
    [P, N, N, F, (RATING_FULL[0], RATING_FULL[1], "< 1 ms")],
    # GA
    [N, N, P, P, (RATING_PART[0], RATING_PART[1], "seconds")],
    # RL
    [N, N, N, N, (RATING_NONE[0], RATING_NONE[1], "minutes")],
    # MILP  ← chosen method
    [F, F, F, F, (RATING_FULL[0], RATING_FULL[1], "< 1 s")],
]

N_ROW = len(METHODS)
N_COL = len(CRITERIA)

# ---------------------------------------------------------------------------
# Layout constants (axes-fraction coordinates)
# ---------------------------------------------------------------------------
# Column x positions (left edges, then widths)
METHOD_W = 0.230   # method name column
COL_W    = (1.0 - METHOD_W) / N_COL   # each criteria column

HEADER_H = 0.170   # header row height
LEG_RESERVE = 0.13  # space reserved at bottom for legend
ROW_H    = (1.0 - HEADER_H - LEG_RESERVE) / N_ROW
GAP      = 0.005   # gap between cells

MILP_ROW = N_ROW - 1   # last row = MILP

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 12.0, 5.2
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def col_x(ci):
    """Left edge of criteria column ci (0-indexed)."""
    return METHOD_W + ci * COL_W


def row_y(ri):
    """Bottom edge of data row ri (0 = top row = Greedy)."""
    return 1.0 - HEADER_H - (ri + 1) * ROW_H


def cell(ax, x0, y0, w, h, fc, ec=C_MUTED, lw=0.8, radius=0.015, zorder=2):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0 + GAP, y0 + GAP), w - 2 * GAP, h - 2 * GAP,
        boxstyle=f"round,pad={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
        transform=ax.transAxes, clip_on=False,
    ))


def txt(ax, x, y, s, **kw):
    kw.setdefault("transform", ax.transAxes)
    kw.setdefault("zorder", 5)
    ax.text(x, y, s, **kw)


# ---------------------------------------------------------------------------
# Header row
# ---------------------------------------------------------------------------
# Method column header
cell(ax, 0, 1.0 - HEADER_H, METHOD_W, HEADER_H,
     fc=C_DARK, ec=C_DARK, lw=1.2, radius=0.018, zorder=3)
txt(ax, METHOD_W / 2, 1.0 - HEADER_H / 2,
    "Method",
    ha="center", va="center", fontsize=10, fontweight="bold", color="white")

# Criteria column headers
for ci, lbl in enumerate(CRITERIA):
    x0 = col_x(ci)
    cell(ax, x0, 1.0 - HEADER_H, COL_W, HEADER_H,
         fc=C_DARK, ec=C_DARK, lw=1.2, radius=0.018, zorder=3)
    txt(ax, x0 + COL_W / 2, 1.0 - HEADER_H / 2,
        lbl,
        ha="center", va="center", fontsize=8.5, fontweight="bold",
        color="white", multialignment="center")


# ---------------------------------------------------------------------------
# Data rows
# ---------------------------------------------------------------------------
for ri, ((method_name, method_note), ratings) in enumerate(zip(METHODS, RATINGS)):
    y0 = row_y(ri)
    is_milp = (ri == MILP_ROW)

    # Alternating row tint (not for MILP — it gets its own style)
    row_bg = "#F0F4F8" if (ri % 2 == 0 and not is_milp) else BG

    # Method name cell
    if is_milp:
        fc_method = "#E8F0F7"
        ec_method = C_NS
        lw_method = 2.0
        name_col  = C_NS
    else:
        fc_method = row_bg
        ec_method = "#D0D8DF"
        lw_method = 0.8
        name_col  = C_DARK

    cell(ax, 0, y0, METHOD_W, ROW_H,
         fc=fc_method, ec=ec_method, lw=lw_method, radius=0.012, zorder=3)

    # Bold method name
    txt(ax, 0.012, y0 + ROW_H * 0.62,
        method_name + ("  ★" if is_milp else ""),
        ha="left", va="center", fontsize=9.5,
        fontweight="bold", color=name_col)
    # Italic description
    txt(ax, 0.012, y0 + ROW_H * 0.30,
        method_note,
        ha="left", va="center", fontsize=7,
        color=C_MUTED, style="italic")

    # Rating cells
    for ci, (bg, tc, sym) in enumerate(ratings):
        x0 = col_x(ci)

        if is_milp:
            ec_cell = C_NS
            lw_cell = 1.6
        else:
            ec_cell = "#D0D8DF"
            lw_cell = 0.8

        cell(ax, x0, y0, COL_W, ROW_H,
             fc=bg, ec=ec_cell, lw=lw_cell, radius=0.012, zorder=3)

        # Symbol or text label
        if sym in ("✓", "~", "✗"):
            txt(ax, x0 + COL_W / 2, y0 + ROW_H / 2,
                sym,
                ha="center", va="center",
                fontsize=18 if sym == "✓" else 16,
                fontweight="bold", color=tc)
        else:
            # Text label (time column)
            txt(ax, x0 + COL_W / 2, y0 + ROW_H / 2,
                sym,
                ha="center", va="center",
                fontsize=9, fontweight="bold", color=tc)


# ---------------------------------------------------------------------------
# MILP highlight: thick border around the entire MILP row
# ---------------------------------------------------------------------------
milp_y0 = row_y(MILP_ROW)
ax.add_patch(mpatches.FancyBboxPatch(
    (GAP * 0.5, milp_y0 + GAP * 0.5),
    1.0 - GAP, ROW_H - GAP,
    boxstyle="round,pad=0.012",
    facecolor="none", edgecolor=C_NS, linewidth=2.2,
    transform=ax.transAxes, clip_on=False, zorder=6,
))



# ---------------------------------------------------------------------------
# Legend for rating symbols
# ---------------------------------------------------------------------------
LEG_Y = 0.012
LEG_ITEMS = [
    (RATING_FULL[0], RATING_FULL[1], "✓  fully supported"),
    (RATING_PART[0], RATING_PART[1], "~  partial / limited"),
    (RATING_NONE[0], RATING_NONE[1], "✗  not supported"),
]
LEG_W = 0.145
LEG_X0 = 0.5 - (len(LEG_ITEMS) * LEG_W) / 2 + 0.10

for i, (bg, tc, label) in enumerate(LEG_ITEMS):
    bx = LEG_X0 + i * LEG_W
    ax.add_patch(mpatches.FancyBboxPatch(
        (bx, LEG_Y), LEG_W - 0.01, 0.042,
        boxstyle="round,pad=0.008",
        facecolor=bg, edgecolor=C_MUTED, linewidth=0.7,
        transform=ax.transAxes, clip_on=False, zorder=3,
    ))
    ax.text(bx + (LEG_W - 0.01) / 2, LEG_Y + 0.021,
            label,
            ha="center", va="center", fontsize=7.5,
            color=tc, fontweight="bold",
            transform=ax.transAxes, zorder=4)


# ---------------------------------------------------------------------------
# Title and caption
# ---------------------------------------------------------------------------
fig.text(0.5, 0.975,
    "Assignment method comparison  ·  MILP vs. alternatives for the inner-loop material matching problem",
    ha="center", va="top", fontsize=10.5, fontweight="bold", color=C_DARK)

plt.tight_layout(rect=[0.0, 0.06, 1.0, 0.96])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
stem = "fig_milp_comparison"
for fmt, out_dir in [("pdf", FIG_PDF_DIR), ("png", FIG_PNG_DIR)]:
    out = out_dir / f"{stem}.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close(fig)
