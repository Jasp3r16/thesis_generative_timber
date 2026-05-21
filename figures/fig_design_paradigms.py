"""
fig_design_paradigms.py — Design paradigm spectrum for chapter 2.1.3.

Three paradigms on a horizontal spectrum: Stock Follows Design (left),
Design Adapts to Stock (centre / thesis position), Design Follows Stock (right).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

C_NS     = "#61788C"
C_RS     = "#F2994B"
C_DARK   = "#2F3E4F"
C_ACCENT = "#D9653B"
C_MUTED  = "#9CA5A6"
C_LIGHT  = "#D7D9D9"
BG       = "#FFFFFF"

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor(BG)

# ---------------------------------------------------------------------------
# Spectrum arrow
# ---------------------------------------------------------------------------
Y_ARROW = 0.72
ax.annotate("",
    xy=(0.93, Y_ARROW), xytext=(0.07, Y_ARROW),
    xycoords="axes fraction", textcoords="axes fraction",
    arrowprops=dict(arrowstyle="-|>", color=C_MUTED, lw=1.8, mutation_scale=16))

ax.text(0.07, Y_ARROW + 0.07, "← Design-led",
        ha="left", va="bottom", fontsize=9, color=C_MUTED,
        style="italic", transform=ax.transAxes)
ax.text(0.93, Y_ARROW + 0.07, "Stock-led →",
        ha="right", va="bottom", fontsize=9, color=C_MUTED,
        style="italic", transform=ax.transAxes)

# ---------------------------------------------------------------------------
# Three paradigm boxes
# ---------------------------------------------------------------------------
X_POS      = [0.18, 0.50, 0.82]
BOX_W      = 0.24
BOX_H      = 0.28
BOX_CY     = 0.38          # box centre y
BOX_TOP    = BOX_CY + BOX_H / 2   # 0.52

titles = [
    "Stock Follows Design",
    "Design Adapts\nto Stock",
    "Design Follows Stock",
]
subtitles = [
    "Geometry fixed first;\nmaterial sourced to match",
    "Geometry optimised\nagainst available inventory",
    "Geometry strictly dictated\nby existing stock",
]
contexts = [
    "conventional\nconstruction",
    "★  this thesis",
    "inventory-only\napproach",
]
colors = [C_NS, C_ACCENT, C_RS]

for x, title, subtitle, context, color in zip(X_POS, titles, subtitles, contexts, colors):
    # Drop line from arrow to box top
    ax.plot([x, x], [Y_ARROW - 0.01, BOX_TOP + 0.01],
            color=color, lw=1.5, transform=ax.transAxes, zorder=2)

    # Dot on arrow
    ax.plot(x, Y_ARROW, "o", color=color, ms=7, zorder=4,
            transform=ax.transAxes)

    # Box
    box = mpatches.FancyBboxPatch(
        (x - BOX_W / 2, BOX_CY - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.015",
        facecolor=color, edgecolor=color,
        linewidth=0, alpha=0.92, zorder=3,
        transform=ax.transAxes,
    )
    ax.add_patch(box)

    # Title (upper half of box)
    ax.text(x, BOX_CY + 0.06, title,
            ha="center", va="center", fontsize=9.5,
            fontweight="bold", color="white",
            transform=ax.transAxes, zorder=5)

    # Subtitle (lower half of box)
    ax.text(x, BOX_CY - 0.06, subtitle,
            ha="center", va="center", fontsize=8,
            color="white", alpha=0.88,
            transform=ax.transAxes, zorder=5)

    # Context label below box
    is_thesis = (color == C_ACCENT)
    ax.text(x, BOX_CY - BOX_H / 2 - 0.06, context,
            ha="center", va="top", fontsize=8.5,
            color=color,
            fontweight="bold" if is_thesis else "normal",
            transform=ax.transAxes, zorder=5)

# ---------------------------------------------------------------------------
# Caption
# ---------------------------------------------------------------------------
plt.tight_layout(pad=0.8)

out_dir = Path(__file__).resolve().parent
for fmt in ["pdf", "png"]:
    out = out_dir / f"fig_design_paradigms.{fmt}"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close()
