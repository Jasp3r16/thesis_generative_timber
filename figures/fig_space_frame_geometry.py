"""
fig_space_frame_geometry.py — Space frame geometry illustration, section 3.1.

Left:  annotated isometric view of the 5×3 space frame (regular configuration).
Right: plan view of top chord with node-shift direction arrows.
No caption embedded — provided separately.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import FIG_PDF_DIR, FIG_PNG_DIR

C_NS    = "#61788C"   # top chord
C_RS    = "#F2994B"   # bottom chord / highlight
C_DARK  = "#2F3E4F"
C_MUTED = "#9CA5A6"
BG      = "#FFFFFF"

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
NX_TOP, NY_TOP = 6, 4   # 6×4 → 24 top nodes
NX_BOT, NY_BOT = 5, 3   # 5×3 → 15 bottom nodes
CW    = 3.0              # cell width (m)
DEPTH = 1.5              # interlayer depth (m)

top_nodes = np.array([[ix*CW, iy*CW, 0.0]
                       for iy in range(NY_TOP)
                       for ix in range(NX_TOP)])

bot_nodes = np.array([[(ix+0.5)*CW, (iy+0.5)*CW, -DEPTH]
                       for iy in range(NY_BOT)
                       for ix in range(NX_BOT)])

top_edges = (
    [(iy*NX_TOP+ix, iy*NX_TOP+ix+1)
     for iy in range(NY_TOP) for ix in range(NX_TOP-1)] +
    [(iy*NX_TOP+ix, (iy+1)*NX_TOP+ix)
     for ix in range(NX_TOP) for iy in range(NY_TOP-1)]
)

bot_edges = (
    [(iy*NX_BOT+ix, iy*NX_BOT+ix+1)
     for iy in range(NY_BOT) for ix in range(NX_BOT-1)] +
    [(iy*NX_BOT+ix, (iy+1)*NX_BOT+ix)
     for ix in range(NX_BOT) for iy in range(NY_BOT-1)]
)

web_edges = [
    (iy*NX_BOT+ix, (iy+diy)*NX_TOP+(ix+dix))
    for iy in range(NY_BOT) for ix in range(NX_BOT)
    for diy in range(2) for dix in range(2)
]

corner_top = {0, NX_TOP-1, (NY_TOP-1)*NX_TOP, NY_TOP*NX_TOP-1}


def node_type(ix, iy):
    corner = (ix in (0, NX_TOP-1)) and (iy in (0, NY_TOP-1))
    edge   = (ix in (0, NX_TOP-1)) or  (iy in (0, NY_TOP-1))
    if corner:  return "corner"
    if edge:    return "edge"
    return "interior"


# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 12.0, 14.5
fig = plt.figure(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.0])
ax3d = fig.add_subplot(gs[0], projection="3d")
ax2d = fig.add_subplot(gs[1])

# ---------------------------------------------------------------------------
# LEFT — 3D isometric view
# ---------------------------------------------------------------------------
ax3d.set_facecolor(BG)
for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor("#E0E0E0")

# Web members
for b, t in web_edges:
    ax3d.plot([bot_nodes[b,0], top_nodes[t,0]],
              [bot_nodes[b,1], top_nodes[t,1]],
              [bot_nodes[b,2], top_nodes[t,2]],
              color=C_MUTED, lw=0.8, alpha=0.55)

# Bottom chord
for i, j in bot_edges:
    ax3d.plot([bot_nodes[i,0], bot_nodes[j,0]],
              [bot_nodes[i,1], bot_nodes[j,1]],
              [bot_nodes[i,2], bot_nodes[j,2]],
              color=C_RS, lw=1.8)

# Top chord
for i, j in top_edges:
    ax3d.plot([top_nodes[i,0], top_nodes[j,0]],
              [top_nodes[i,1], top_nodes[j,1]],
              [top_nodes[i,2], top_nodes[j,2]],
              color=C_NS, lw=1.8)

# Scatter — regular top nodes
non_corner = [i for i in range(len(top_nodes)) if i not in corner_top]
ax3d.scatter(top_nodes[non_corner, 0], top_nodes[non_corner, 1],
             top_nodes[non_corner, 2],
             color=C_NS, s=28, edgecolors="white", linewidths=0.6,
             depthshade=False, zorder=4)

# Scatter — bottom nodes
ax3d.scatter(bot_nodes[:, 0], bot_nodes[:, 1], bot_nodes[:, 2],
             color=C_RS, s=35, edgecolors="white", linewidths=0.6,
             depthshade=False, zorder=4)

# Support nodes (corners)
ci = list(corner_top)
ax3d.scatter(top_nodes[ci, 0], top_nodes[ci, 1], top_nodes[ci, 2],
             color=C_DARK, s=75, marker="o",
             edgecolors="white", linewidths=0.8, depthshade=False, zorder=6)

# ---------------------------------------------------------------------------
# Dimension annotations
# ---------------------------------------------------------------------------
# Interlayer depth — 1.5 m at right-front edge
ax3d.plot([15.7, 15.7], [0, 0], [-1.5, 0], color=C_MUTED, lw=0.9)
ax3d.plot([15.4, 15.7], [0, 0], [0, 0], color=C_MUTED, lw=0.9)
ax3d.plot([15.4, 15.7], [0, 0], [-1.5, -1.5], color=C_MUTED, lw=0.9)
ax3d.text(16.3, 0.0, -0.75, "1.5 m", fontsize=7.5, color=C_DARK,
          va="center", fontweight="bold")


ax3d.view_init(elev=22, azim=-55)
ax3d.set_box_aspect([10, 6, 1])    # true proportions: 15 m × 9 m × 1.5 m
ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
ax3d.set_xlabel(""); ax3d.set_ylabel(""); ax3d.set_zlabel("")
ax3d.grid(False)

# ---------------------------------------------------------------------------
# RIGHT — plan view of top chord with shift arrows
# ---------------------------------------------------------------------------
ax2d.set_facecolor(BG)
ax2d.set_xlim(-2.5, 17.5)
ax2d.set_ylim(-2.8, 12.0)
ax2d.set_aspect("equal")
ax2d.axis("off")

ARROW_L = 0.52   # visual arrow half-length (m)

# Light grid (base positions)
for iy in range(NY_TOP):
    for ix in range(NX_TOP - 1):
        ax2d.plot([ix*CW, (ix+1)*CW], [iy*CW, iy*CW],
                  color=C_MUTED, lw=0.7, ls="--", alpha=0.40, zorder=1)
for ix in range(NX_TOP):
    for iy in range(NY_TOP - 1):
        ax2d.plot([ix*CW, ix*CW], [iy*CW, (iy+1)*CW],
                  color=C_MUTED, lw=0.7, ls="--", alpha=0.40, zorder=1)

# Nodes + shift arrows
for iy in range(NY_TOP):
    for ix in range(NX_TOP):
        bx, by = ix * CW, iy * CW
        nt = node_type(ix, iy)

        if nt == "corner":
            ax2d.scatter(bx, by, s=100, color=C_DARK, marker="s",
                        zorder=5, edgecolors="white", linewidths=0.8)
        else:
            ax2d.scatter(bx, by, s=55, color=C_NS,
                        zorder=5, edgecolors="white", linewidths=0.8)

            ap = dict(arrowstyle="<->", lw=0.9, mutation_scale=7)

            if nt == "interior":
                ax2d.annotate("", xy=(bx+ARROW_L, by), xytext=(bx-ARROW_L, by),
                    arrowprops={**ap, "color": C_NS}, zorder=4)
                ax2d.annotate("", xy=(bx, by+ARROW_L), xytext=(bx, by-ARROW_L),
                    arrowprops={**ap, "color": C_NS}, zorder=4)
            else:  # edge
                if ix in (0, NX_TOP-1):   # left/right boundary → shift in Y (along boundary)
                    ax2d.annotate("", xy=(bx, by+ARROW_L), xytext=(bx, by-ARROW_L),
                        arrowprops={**ap, "color": C_NS + "CC"}, zorder=4)
                else:                      # top/bottom boundary → shift in X (along boundary)
                    ax2d.annotate("", xy=(bx+ARROW_L, by), xytext=(bx-ARROW_L, by),
                        arrowprops={**ap, "color": C_NS + "CC"}, zorder=4)

# Cell width — 3.0 m on interior member (ix=1→2, iy=1), above the grid line
ax2d.plot([3.0, 6.0], [4.25, 4.25], color=C_MUTED, lw=0.9)
ax2d.plot([3.0, 3.0], [4.15, 4.35], color=C_MUTED, lw=0.9)
ax2d.plot([6.0, 6.0], [4.15, 4.35], color=C_MUTED, lw=0.9)
ax2d.text(4.5, 4.42, "3.0 m", ha="center", va="bottom",
          fontsize=8, color=C_DARK, fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.10", fc=BG, ec="none", alpha=0.9))

# Footprint dimensions on plan view
# 15.0 m — below the grid
ax2d.plot([0, 15], [-1.55, -1.55], color=C_MUTED, lw=0.9)
ax2d.plot([0,  0],  [-1.55, -1.35], color=C_MUTED, lw=0.9)
ax2d.plot([15, 15], [-1.55, -1.35], color=C_MUTED, lw=0.9)
ax2d.text(7.5, -1.75, "15.0 m", ha="center", va="top",
          fontsize=8.5, color=C_DARK, fontweight="bold")

# 9.0 m — left of the grid
ax2d.plot([-1.55, -1.55], [0, 9], color=C_MUTED, lw=0.9)
ax2d.plot([-1.55, -1.35], [0,  0],  color=C_MUTED, lw=0.9)
ax2d.plot([-1.55, -1.35], [9,  9],  color=C_MUTED, lw=0.9)
ax2d.text(-1.75, 4.5, "9.0 m", ha="right", va="center",
          fontsize=8.5, color=C_DARK, fontweight="bold", rotation=90)

# ---------------------------------------------------------------------------
# Combined legend — shared between both panels
# ---------------------------------------------------------------------------
legend_handles = [
    Line2D([0], [0], color=C_NS, lw=2.5,
           label="Top chord  (24 nodes,  38 members)"),
    Line2D([0], [0], color=C_RS, lw=2.5,
           label="Bottom chord  (15 nodes,  22 members)"),
    Line2D([0], [0], color=C_MUTED, lw=1.5,
           label="Web members  (60 members)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_DARK,
           markersize=9, label="Corner node  —  pin support"),
    Line2D([0], [0], color=C_NS, lw=1.5,
           label="Node shift direction  (↔  edge: 1-axis   ✛  interior: 2-axes)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           fontsize=8.5, frameon=True, framealpha=0.95, edgecolor=C_MUTED,
           bbox_to_anchor=(0.5, 0.01))

fig.subplots_adjust(left=0.03, right=0.97, bottom=0.10, top=0.97, hspace=0.06)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
stem = "fig_space_frame_geometry"
for fmt, out_dir in [("pdf", FIG_PDF_DIR), ("png", FIG_PNG_DIR)]:
    out = out_dir / f"{stem}.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close(fig)
