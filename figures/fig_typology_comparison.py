"""
fig_typology_comparison.py — Structural typology comparison for chapter 2.2.

Four typologies rated across four criteria. Space frame row highlighted as selected.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

C_NS     = "#61788C"
C_DARK   = "#2F3E4F"
C_ACCENT = "#D9653B"
C_MUTED  = "#9CA5A6"
C_LIGHT  = "#D7D9D9"
C_BIO    = "#A8B89A"
BG       = "#FFFFFF"

RATING_FC  = {3: C_BIO,    2: "#E8C97A", 1: C_LIGHT}
RATING_TC  = {3: "#2F4030", 2: "#5A4010", 1: C_MUTED}
RATING_LBL = {3: "High",   2: "Medium",  1: "Low"}

# (name, [length_tol, redundancy, geom_versatility, comp_suitability], selected)
data = [
    ("Reciprocal Frame",    [3, 2, 1, 1], False),
    ("Diagrid / Gridshell", [2, 3, 2, 2], False),
    ("Lamella (Zollinger)", [3, 2, 1, 1], False),
    ("Space Frame",         [3, 3, 3, 3], True),
]

col_headers = [
    "Variable length\ntolerance",
    "Structural\nredundancy",
    "Geometric\nversatility",
    "Computational\nsuitability",
    "",   # Selected checkmark column
]

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor(BG)

X_NAME_L  = 0.02
X_COLS_L  = 0.29
X_COLS_R  = 0.98
N_COLS    = len(col_headers)
COL_W     = (X_COLS_R - X_COLS_L) / N_COLS
Y_HEADER  = 0.88
Y_ROWS    = [0.67, 0.51, 0.35, 0.19]
ROW_H     = 0.12
PAD       = 0.007

# ---------------------------------------------------------------------------
# Column headers
# ---------------------------------------------------------------------------
for i, hdr in enumerate(col_headers):
    x = X_COLS_L + (i + 0.5) * COL_W
    ax.text(x, Y_HEADER + 0.02, hdr,
            ha="center", va="bottom", fontsize=8.5,
            fontweight="bold", color=C_DARK,
            transform=ax.transAxes)

ax.plot([X_NAME_L, X_COLS_R], [Y_HEADER, Y_HEADER],
        color=C_DARK, lw=0.9, transform=ax.transAxes, zorder=0)

# ---------------------------------------------------------------------------
# Data rows
# ---------------------------------------------------------------------------
for (name, ratings, selected), y in zip(data, Y_ROWS):

    # Row highlight
    if selected:
        hl = mpatches.FancyBboxPatch(
            (X_NAME_L, y - ROW_H / 2 - PAD),
            X_COLS_R - X_NAME_L, ROW_H + 2 * PAD,
            boxstyle="round,pad=0.004",
            facecolor=C_ACCENT, alpha=0.08,
            edgecolor=C_ACCENT, linewidth=0.9,
            zorder=1, transform=ax.transAxes,
        )
        ax.add_patch(hl)

    # Row name
    ax.text(X_NAME_L + 0.01, y, name,
            ha="left", va="center", fontsize=9.5,
            fontweight="bold" if selected else "normal",
            color=C_ACCENT if selected else C_DARK,
            transform=ax.transAxes, zorder=4)

    # Rating cells
    for i, r in enumerate(ratings):
        x  = X_COLS_L + (i + 0.5) * COL_W
        cw = COL_W - 2 * PAD
        ch = ROW_H - 2 * PAD
        cell = mpatches.FancyBboxPatch(
            (x - cw / 2, y - ch / 2), cw, ch,
            boxstyle="round,pad=0.004",
            facecolor=RATING_FC[r], edgecolor="white",
            linewidth=1.0, alpha=0.92,
            zorder=3, transform=ax.transAxes,
        )
        ax.add_patch(cell)
        ax.text(x, y, RATING_LBL[r],
                ha="center", va="center", fontsize=8,
                fontweight="bold", color=RATING_TC[r],
                transform=ax.transAxes, zorder=4)

    # Selected checkmark
    x_sel = X_COLS_L + 4.5 * COL_W
    ax.text(x_sel, y, "✓" if selected else "—",
            ha="center", va="center",
            fontsize=15 if selected else 10,
            fontweight="bold" if selected else "normal",
            color=C_ACCENT if selected else C_LIGHT,
            transform=ax.transAxes, zorder=4)

# Vertical column dividers (light dashed)
for i in range(1, N_COLS):
    x = X_COLS_L + i * COL_W
    ax.plot([x, x],
            [Y_ROWS[-1] - ROW_H / 2 - 0.02, Y_HEADER],
            color=C_LIGHT, lw=0.5, ls="--",
            transform=ax.transAxes, zorder=0)

# Bottom rule
ax.plot([X_NAME_L, X_COLS_R],
        [Y_ROWS[-1] - ROW_H / 2 - 0.015,
         Y_ROWS[-1] - ROW_H / 2 - 0.015],
        color=C_LIGHT, lw=0.7, transform=ax.transAxes)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
legend_patches = [
    mpatches.Patch(facecolor=C_BIO,     edgecolor="white", label="High"),
    mpatches.Patch(facecolor="#E8C97A", edgecolor="white", label="Medium"),
    mpatches.Patch(facecolor=C_LIGHT,   edgecolor=C_MUTED, label="Low"),
]
leg = ax.legend(handles=legend_patches, loc="lower right",
                bbox_to_anchor=(0.98, 0.01), ncol=3,
                fontsize=8, frameon=True, framealpha=0.95,
                edgecolor=C_MUTED)
leg.get_frame().set_linewidth(0.6)

plt.tight_layout(pad=0.8)

out_dir = Path(__file__).resolve().parent
for fmt in ["pdf", "png"]:
    out = out_dir / f"fig_typology_comparison.{fmt}"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close()
