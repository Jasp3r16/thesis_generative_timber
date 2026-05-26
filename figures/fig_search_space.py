"""
fig_search_space.py — Search space parameterisation, section 3.5.

Left:  plan view of the 6×4 top chord grid with node-type variable assignment
       and discrete increment ruler.
Right: 3D schematic of a single pyramid cell showing bottom-node parameters
       u, v (cell-local horizontal) and shift_z (vertical).
No caption embedded — provided separately.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
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

NX_TOP, NY_TOP = 6, 4
CW    = 3.0
DEPTH = 1.5

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 13.5, 8.0
fig = plt.figure(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 0.9], wspace=0.08)
ax2d = fig.add_subplot(gs[0])
ax3d = fig.add_subplot(gs[1], projection="3d")

# ===========================================================================
# LEFT — plan view of top chord with variable annotations
# ===========================================================================
ax2d.set_facecolor(BG)
ax2d.set_xlim(-5.5, 20.5)
ax2d.set_ylim(-5.8, 14.5)
ax2d.set_aspect("equal")
ax2d.axis("off")
ax2d.set_title("Top chord  —  variable assignment by node type  (28 variables)",
               fontsize=10, fontweight="bold", color=C_DARK, pad=6)


def node_type(ix, iy):
    corner = (ix in (0, NX_TOP-1)) and (iy in (0, NY_TOP-1))
    edge   = (ix in (0, NX_TOP-1)) or  (iy in (0, NY_TOP-1))
    if corner: return "corner"
    if edge:   return "edge"
    return "interior"


# Light grid lines
for iy in range(NY_TOP):
    ax2d.plot([0, (NX_TOP-1)*CW], [iy*CW, iy*CW],
              color=C_MUTED, lw=0.6, ls="--", alpha=0.35, zorder=1)
for ix in range(NX_TOP):
    ax2d.plot([ix*CW, ix*CW], [0, (NY_TOP-1)*CW],
              color=C_MUTED, lw=0.6, ls="--", alpha=0.35, zorder=1)

# Nodes and shift arrows
ARROW_L = 0.52
ap_int  = dict(arrowstyle="<->", lw=0.9, mutation_scale=7, color=C_NS)
ap_edge = dict(arrowstyle="<->", lw=0.9, mutation_scale=7, color=C_NS + "AA")

for iy in range(NY_TOP):
    for ix in range(NX_TOP):
        bx, by = ix * CW, iy * CW
        nt = node_type(ix, iy)
        if nt == "corner":
            ax2d.scatter(bx, by, s=90, color=C_DARK, marker="s",
                         zorder=5, edgecolors="white", linewidths=0.8)
        else:
            ax2d.scatter(bx, by, s=50, color=C_NS,
                         zorder=5, edgecolors="white", linewidths=0.8)
            if nt == "interior":
                ax2d.annotate("", xy=(bx+ARROW_L, by), xytext=(bx-ARROW_L, by),
                    arrowprops=ap_int, zorder=4)
                ax2d.annotate("", xy=(bx, by+ARROW_L), xytext=(bx, by-ARROW_L),
                    arrowprops=ap_int, zorder=4)
            else:
                if ix in (0, NX_TOP-1):  # left/right → Y
                    ax2d.annotate("", xy=(bx, by+ARROW_L), xytext=(bx, by-ARROW_L),
                        arrowprops=ap_edge, zorder=4)
                else:                     # top/bottom → X
                    ax2d.annotate("", xy=(bx+ARROW_L, by), xytext=(bx-ARROW_L, by),
                        arrowprops=ap_edge, zorder=4)

# ---------------------------------------------------------------------------
# Callout annotations — one representative per node type
# ---------------------------------------------------------------------------
ANN_KW = dict(zorder=6, clip_on=False)
BOX_KW = lambda col: dict(boxstyle="round,pad=0.22", fc=BG, ec=col + "55", lw=0.8)

# Corner (0, 0)
ax2d.annotate(
    "corner node  (×4)\nfixed  —  0 variables\npin support",
    xy=(0, 0), xytext=(-5.0, -0.8),
    fontsize=7.5, color=C_DARK, ha="right", style="italic",
    arrowprops=dict(arrowstyle="-", color=C_DARK + "70", lw=0.9),
    bbox=BOX_KW(C_DARK), **ANN_KW)

# Edge left/right (0, 1) — Y shift
ax2d.annotate(
    "edge node, left/right  (×4)\n1 variable  —  Δy",
    xy=(0, CW), xytext=(-5.0, CW + 1.2),
    fontsize=7.5, color=C_NS, ha="right",
    arrowprops=dict(arrowstyle="-", color=C_NS + "70", lw=0.9),
    bbox=BOX_KW(C_NS), **ANN_KW)

# Edge top/bottom (2, 0) — X shift
ax2d.annotate(
    "edge node, top/bottom  (×8)\n1 variable  —  Δx",
    xy=(2*CW, 0), xytext=(2*CW, -3.2),
    fontsize=7.5, color=C_NS, ha="center",
    arrowprops=dict(arrowstyle="-", color=C_NS + "70", lw=0.9),
    bbox=BOX_KW(C_NS), **ANN_KW)

# Interior (2, 1) — XY shifts
ax2d.annotate(
    "interior node  (×8)\n2 variables  —  Δx + Δy",
    xy=(2*CW, CW), xytext=(17.5, CW + 1.2),
    fontsize=7.5, color=C_NS, ha="left",
    arrowprops=dict(arrowstyle="-", color=C_NS + "70", lw=0.9),
    bbox=BOX_KW(C_NS), **ANN_KW)

# ---------------------------------------------------------------------------
# Discrete steps ruler — below the grid
# ---------------------------------------------------------------------------
RULER_Y  = -4.2
RULER_W  = 14.5
RULER_CX = (NX_TOP - 1) * CW / 2   # 7.5 m, centre of grid
STEPS = np.array([-1.125, -0.75, -0.375, 0.0, 0.375, 0.75, 1.125])
step_x = RULER_CX + (STEPS / 1.125) * (RULER_W / 2)

# Continuous band background
ax2d.add_patch(FancyBboxPatch(
    (RULER_CX - RULER_W/2, RULER_Y - 0.16), RULER_W, 0.32,
    boxstyle="round,pad=0.04",
    facecolor=C_NS + "26", edgecolor=C_NS + "65", lw=0.9, zorder=2))

ax2d.text(RULER_CX, RULER_Y + 0.52,
          "7 discrete training positions",
          ha="center", va="bottom", fontsize=7.5, color=C_MUTED, style="italic")
ax2d.text(RULER_CX, RULER_Y - 0.62,
          "continuous optimisation range  [−1.125, +1.125] m",
          ha="center", va="top", fontsize=7.5, color=C_NS)

ax2d.scatter(step_x, [RULER_Y]*7, s=22, color=C_DARK, zorder=4)
for sx, sv in zip(step_x, STEPS):
    ax2d.plot([sx, sx], [RULER_Y - 0.24, RULER_Y + 0.24],
              color=C_DARK, lw=1.5, zorder=3)
    lbl = "0" if sv == 0.0 else f"{sv:+.3f}"
    ax2d.text(sx, RULER_Y - 0.32, lbl,
              ha="center", va="top", fontsize=6.5, color=C_DARK, rotation=40)

# ===========================================================================
# RIGHT — 3D single pyramid cell, bottom node parameters
# ===========================================================================
ax3d.set_facecolor(BG)
for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor("#EBEBEB")

cell_top = np.array([[0, 0, 0], [CW, 0, 0], [CW, CW, 0], [0, CW, 0]])

U_EX, V_EX = 0.56, 0.43
bx_ex = U_EX * CW
by_ex = V_EX * CW
bz_ex = -DEPTH

# Top chord square
for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
    ax3d.plot([cell_top[i, 0], cell_top[j, 0]],
              [cell_top[i, 1], cell_top[j, 1]],
              [cell_top[i, 2], cell_top[j, 2]],
              color=C_NS, lw=2.0)

# Web members
for c in cell_top:
    ax3d.plot([c[0], bx_ex], [c[1], by_ex], [c[2], bz_ex],
              color=C_MUTED, lw=0.9, alpha=0.6)

# Top nodes
ax3d.scatter(cell_top[:, 0], cell_top[:, 1], cell_top[:, 2],
             color=C_NS, s=40, edgecolors="white", linewidths=0.6,
             depthshade=False, zorder=4)

# Bottom node
ax3d.scatter([bx_ex], [by_ex], [bz_ex],
             color=C_RS, s=65, edgecolors="white", linewidths=0.9,
             depthshade=False, zorder=5)

# u indicator — bracket along x at front of bottom plane
U_L, U_H = 0.25 * CW, 0.75 * CW
V_NEAR = 0.28
ax3d.plot([U_L, U_H], [V_NEAR, V_NEAR], [bz_ex, bz_ex],
          color=C_RS, lw=1.8)
ax3d.plot([U_L, U_L], [V_NEAR, V_NEAR], [bz_ex-0.08, bz_ex+0.08],
          color=C_RS, lw=1.2)
ax3d.plot([U_H, U_H], [V_NEAR, V_NEAR], [bz_ex-0.08, bz_ex+0.08],
          color=C_RS, lw=1.2)
ax3d.text((U_L+U_H)/2, V_NEAR - 0.2, bz_ex - 0.28,
          "u  ∈ [0.25, 0.75]",
          ha="center", va="top", fontsize=7.5, color=C_RS, fontweight="bold")

# v indicator — bracket along y at right of bottom plane
V_L, V_H = 0.25 * CW, 0.75 * CW
U_FAR = CW - 0.28
ax3d.plot([U_FAR, U_FAR], [V_L, V_H], [bz_ex, bz_ex],
          color=C_RS, lw=1.8)
ax3d.plot([U_FAR, U_FAR], [V_L, V_L], [bz_ex-0.08, bz_ex+0.08],
          color=C_RS, lw=1.2)
ax3d.plot([U_FAR, U_FAR], [V_H, V_H], [bz_ex-0.08, bz_ex+0.08],
          color=C_RS, lw=1.2)
ax3d.text(U_FAR + 0.22, (V_L+V_H)/2, bz_ex - 0.28,
          "v  ∈ [0.25, 0.75]",
          ha="left", va="top", fontsize=7.5, color=C_RS, fontweight="bold")

# shift_z indicator — vertical bracket outside cell at front-right corner
SZ = 1.125
SZ_X, SZ_Y = CW + 0.55, -0.05
ax3d.plot([SZ_X, SZ_X], [SZ_Y, SZ_Y], [bz_ex - SZ, bz_ex + SZ],
          color=C_DARK, lw=1.8)
ax3d.plot([SZ_X - 0.15, SZ_X + 0.15], [SZ_Y, SZ_Y],
          [bz_ex - SZ, bz_ex - SZ], color=C_DARK, lw=1.2)
ax3d.plot([SZ_X - 0.15, SZ_X + 0.15], [SZ_Y, SZ_Y],
          [bz_ex + SZ, bz_ex + SZ], color=C_DARK, lw=1.2)
ax3d.text(SZ_X + 0.25, SZ_Y, bz_ex,
          "shift_z\n±1.125 m",
          ha="left", va="center", fontsize=7.5, color=C_DARK, fontweight="bold")

# Dashed outer cell boundary on bottom plane
xs = [0, CW, CW, 0, 0]
ys = [0, 0, CW, CW, 0]
ax3d.plot(xs, ys, [bz_ex]*5, color=C_MUTED, lw=0.8, ls="--", alpha=0.45)

# Dotted inner bounds rectangle (allowed region for u, v)
ib_xs = [U_L, U_H, U_H, U_L, U_L]
ib_ys = [V_L, V_L, V_H, V_H, V_L]
ax3d.plot(ib_xs, ib_ys, [bz_ex]*5, color=C_RS, lw=0.9, ls=":", alpha=0.75)

ax3d.view_init(elev=28, azim=-48)
ax3d.set_box_aspect([1, 1, 0.55])
ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
ax3d.set_xlabel(""); ax3d.set_ylabel(""); ax3d.set_zlabel("")
ax3d.grid(False)
ax3d.set_title("Bottom node  —  cell-local parameters\n"
               "3 vars × 15 nodes = 45 variables",
               fontsize=9.5, fontweight="bold", color=C_DARK, pad=4)

fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.93)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
stem = "fig_search_space"
for fmt, out_dir in [("pdf", FIG_PDF_DIR), ("png", FIG_PNG_DIR)]:
    out = out_dir / f"{stem}.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close(fig)
