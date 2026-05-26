"""
fig_training_pipeline.py — Training data generation pipeline, section 3.2.

Vertical flowchart: geometry sampling → duplicate rejection → normalisation →
mean-EA FEM solve → random stock assignment → CSV export  (Python boundary)
→ Karamba3D FEA → utilisation label → GNN training-ready.
No caption embedded — provided separately.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import FIG_PDF_DIR, FIG_PNG_DIR

C_NS    = "#61788C"
C_RS    = "#F2994B"
C_DARK  = "#2F3E4F"
C_MUTED = "#9CA5A6"
BG      = "#FFFFFF"

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 8.0, 14.0
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 8)
ax.set_ylim(0, 14)
ax.axis("off")

CX = 4.0    # centre x
BW = 5.2    # main box width
BH = 0.78   # box height
R  = 0.07   # rounded corner radius

# Step y-centres (higher y = higher on page = earlier step)
Y1 = 12.6   # ① Sample geometry
Y2 = 10.8   # ② Unique? diamond
Y3 =  9.1   # ③ Centroid normalise
Y4 =  7.5   # ④ Mean-EA FEM
Y5 =  5.9   # ⑤ Random stock
Y6 =  4.3   # ⑥ Build CSV
BY =  3.35  # — boundary —
Y7 =  2.4   # ⑦ Karamba3D
Y8 =  1.0   # ⑧ Add utilisation

D_W = 2.5   # diamond half-width (x)
D_H = 0.82  # diamond half-height (y)

# ---------------------------------------------------------------------------
# Section backgrounds
# ---------------------------------------------------------------------------
py_top = Y1 + BH / 2 + 0.35
py_bot = Y6 - BH / 2 - 0.28
ax.add_patch(FancyBboxPatch(
    (0.22, py_bot), 7.56, py_top - py_bot,
    boxstyle="round,pad=0.05",
    facecolor=C_NS + "0D", edgecolor=C_NS + "50", linewidth=0.9,
    zorder=1, clip_on=False))
ax.text(0.52, (py_top + py_bot) / 2, "Python",
        ha="center", va="center", fontsize=8, color=C_NS,
        fontweight="bold", rotation=90, alpha=0.55, zorder=2)

gh_top = Y7 + BH / 2 + 0.28
gh_bot = Y8 - BH / 2 - 0.38
ax.add_patch(FancyBboxPatch(
    (0.22, gh_bot), 7.56, gh_top - gh_bot,
    boxstyle="round,pad=0.05",
    facecolor=C_RS + "0D", edgecolor=C_RS + "50", linewidth=0.9,
    zorder=1, clip_on=False))
ax.text(0.52, (gh_top + gh_bot) / 2, "Grasshopper",
        ha="center", va="center", fontsize=8, color=C_RS,
        fontweight="bold", rotation=90, alpha=0.65, zorder=2)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def draw_box(cx, cy, w, h, fc, ec, lw=1.5):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad={R}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        zorder=3, clip_on=False))


def draw_label(cx, cy, title, sub=None, tcol=C_DARK):
    dy = 0.13 if sub else 0.0
    ax.text(cx, cy + dy, title,
            ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=tcol, zorder=4)
    if sub:
        ax.text(cx, cy - 0.18, sub,
                ha="center", va="center", fontsize=7.5,
                color=C_MUTED, style="italic", zorder=4)


def arrow_down(x, y_from, y_to, col=C_DARK):
    ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5,
                        mutation_scale=13), zorder=5)

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
ax.text(CX, 13.45,
        "Training data generation pipeline  ·  20 000 samples",
        ha="center", va="center", fontsize=10.5,
        fontweight="bold", color=C_DARK, zorder=6)

# ---------------------------------------------------------------------------
# ① Sample geometry
# ---------------------------------------------------------------------------
draw_box(CX, Y1, BW, BH, fc=C_NS + "22", ec=C_NS, lw=1.6)
draw_label(CX, Y1,
           "① Sample geometry",
           "random discrete shift per movable node  ·  top and bottom layer")
arrow_down(CX, Y1 - BH / 2 - 0.04, Y2 + D_H + 0.04)

# ---------------------------------------------------------------------------
# ② Unique signature? (diamond)
# ---------------------------------------------------------------------------
ax.add_patch(plt.Polygon([
    [CX,         Y2 + D_H],
    [CX + D_W,   Y2      ],
    [CX,         Y2 - D_H],
    [CX - D_W,   Y2      ],
], closed=True, facecolor=C_NS + "22", edgecolor=C_NS, linewidth=1.6, zorder=3))
ax.text(CX, Y2 + 0.20, "② Unique signature?",
        ha="center", va="center", fontsize=9,
        fontweight="bold", color=C_DARK, zorder=4)
ax.text(CX, Y2 - 0.22, "coordinate hash  ·  reject if duplicate",
        ha="center", va="center", fontsize=7.5,
        color=C_MUTED, style="italic", zorder=4)

# "yes" label and downward arrow
ax.text(CX + 0.22, Y2 - D_H - 0.06, "yes ↓",
        ha="left", va="top", fontsize=8, color=C_NS,
        fontweight="bold", zorder=5)
arrow_down(CX, Y2 - D_H - 0.04, Y3 + BH / 2 + 0.04)

# "no" loop — exits right vertex, goes right then up then left back to step ①
LX = CX + D_W + 0.78
ax.annotate("", xy=(LX, Y2), xytext=(CX + D_W + 0.05, Y2),
    arrowprops=dict(arrowstyle="-", color=C_MUTED, lw=1.2), zorder=5)
ax.plot([LX, LX], [Y2, Y1], color=C_MUTED, lw=1.2, zorder=5)
ax.annotate("", xy=(CX + BW / 2 + 0.05, Y1), xytext=(LX, Y1),
    arrowprops=dict(arrowstyle="-|>", color=C_MUTED, lw=1.2,
                    mutation_scale=10), zorder=5)
ax.text(LX + 0.11, (Y1 + Y2) / 2, "no  —  resample",
        ha="left", va="center", fontsize=7.5,
        color=C_MUTED, style="italic", rotation=90, zorder=5, clip_on=False)

# ---------------------------------------------------------------------------
# ③ Centroid normalise
# ---------------------------------------------------------------------------
draw_box(CX, Y3, BW, BH, fc=C_NS + "22", ec=C_NS, lw=1.6)
draw_label(CX, Y3,
           "③ Centroid normalise",
           "shift all coordinates so geometric centroid → origin")
arrow_down(CX, Y3 - BH / 2 - 0.04, Y4 + BH / 2 + 0.04)

# ---------------------------------------------------------------------------
# ④ Mean-EA FEM solve
# ---------------------------------------------------------------------------
draw_box(CX, Y4, BW, BH, fc=C_NS + "22", ec=C_NS, lw=1.6)
draw_label(CX, Y4,
           "④ Mean-EA FEM solve",
           "direct stiffness · uniform EA  →  N_mean_EA per member · F_z per node")
arrow_down(CX, Y4 - BH / 2 - 0.04, Y5 + BH / 2 + 0.04)

# ---------------------------------------------------------------------------
# ⑤ Random stock assignment
# ---------------------------------------------------------------------------
draw_box(CX, Y5, BW, BH, fc=C_NS + "22", ec=C_NS, lw=1.6)
draw_label(CX, Y5,
           "⑤ Random stock assignment",
           "1 element drawn per slot from full catalogue  ·  120 assignments")
arrow_down(CX, Y5 - BH / 2 - 0.04, Y6 + BH / 2 + 0.04)

# ---------------------------------------------------------------------------
# ⑥ Build CSV  (output-style box)
# ---------------------------------------------------------------------------
draw_box(CX, Y6, BW, BH, fc=C_DARK + "14", ec=C_DARK, lw=1.8)
draw_label(CX, Y6,
           "⑥ Build node & edge feature CSV",
           "39 node rows · 120 edge rows per sample · no utilisation column yet",
           tcol=C_DARK)

# ---------------------------------------------------------------------------
# Python / Grasshopper boundary
# ---------------------------------------------------------------------------
ax.plot([0.35, 7.65], [BY, BY],
        color=C_RS, lw=1.8, ls="--", alpha=0.85, zorder=6)
ax.text(CX, BY + 0.13,
        "Python  /  Grasshopper  boundary",
        ha="center", va="bottom", fontsize=8.5,
        color=C_RS, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.16", fc=BG, ec="none", alpha=0.97),
        zorder=7)
arrow_down(CX, Y6 - BH / 2 - 0.04, Y7 + BH / 2 + 0.04, col=C_RS)

# ---------------------------------------------------------------------------
# ⑦ Karamba3D structural analysis
# ---------------------------------------------------------------------------
draw_box(CX, Y7, BW, BH, fc=C_RS + "22", ec=C_RS, lw=1.6)
draw_label(CX, Y7,
           "⑦ Karamba3D structural analysis",
           "full stiffness solve · actual cross-sections  →  per-member utilisation")

# Ground-truth label annotation (right side, clip_on=False for overflow)
ax.annotate("ground-truth\nstructural label\nenters here",
    xy=(CX + BW / 2 + 0.05, Y7),
    xytext=(CX + BW / 2 + 0.2, Y7 - 0.65),
    fontsize=7.5, color=C_RS, style="italic", ha="left", va="top",
    arrowprops=dict(arrowstyle="-|>", color=C_RS, lw=1.0, mutation_scale=9),
    zorder=6, clip_on=False)

arrow_down(CX, Y7 - BH / 2 - 0.04, Y8 + BH / 2 + 0.04)

# ---------------------------------------------------------------------------
# ⑧ Add utilisation → GNN training-ready
# ---------------------------------------------------------------------------
draw_box(CX, Y8, BW, BH, fc=C_RS + "33", ec=C_RS, lw=2.0)
draw_label(CX, Y8,
           "⑧ Add utilisation column  →  GNN training-ready",
           "20 000 uniquely labelled samples  ·  node + edge CSV pair complete",
           tcol=C_DARK)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
stem = "fig_training_pipeline"
for fmt, out_dir in [("pdf", FIG_PDF_DIR), ("png", FIG_PNG_DIR)]:
    out = out_dir / f"{stem}.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close(fig)
