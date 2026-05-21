"""
fig_graph_abstraction.py — Vertex table and edge index illustration for chapter 2.4.

Left:   3D space frame (6 nodes, 9 edges), colour-coded by member type.
Centre: Vertex Table V — Cartesian coordinates of each node.
Right:  Edge Index E — start/end node indices of each edge.
Colour coding is consistent across all three panels.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

C_NS     = "#61788C"
C_RS     = "#F2994B"
C_DARK   = "#2F3E4F"
C_MUTED  = "#9CA5A6"
C_LIGHT  = "#D7D9D9"
BG       = "#FFFFFF"

# Light row fills (each colour blended ~18% onto white)
RC_BOT = "#D2D8DD"   # bottom chord / bottom layer  (light C_DARK)
RC_TOP = "#C8D8E4"   # top chord    / top layer     (light C_NS)
RC_WEB = "#FAE3D0"   # web members                  (light C_RS)

# ---------------------------------------------------------------------------
# Geometry — double-layer triangular unit (6 nodes, 9 edges)
# ---------------------------------------------------------------------------
nodes = np.array([
    [0.0, 0.0, 0.0],   # 0  bottom layer
    [2.0, 0.0, 0.0],   # 1
    [1.0, 1.5, 0.0],   # 2
    [0.5, 0.5, 1.0],   # 3  top layer
    [1.5, 0.5, 1.0],   # 4
    [1.0, 1.2, 1.0],   # 5
])

# (start, end, line_colour, row_fill_colour)
edges = [
    (0, 1, C_DARK, RC_BOT),
    (1, 2, C_DARK, RC_BOT),
    (0, 2, C_DARK, RC_BOT),
    (3, 4, C_NS,   RC_TOP),
    (4, 5, C_NS,   RC_TOP),
    (3, 5, C_NS,   RC_TOP),
    (0, 3, C_RS,   RC_WEB),
    (1, 4, C_RS,   RC_WEB),
    (2, 5, C_RS,   RC_WEB),
]
N_COLORS = [C_DARK] * 3 + [C_NS] * 3

# ---------------------------------------------------------------------------
# Figure layout: [wide 3D panel | vertex table | edge index]
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 5.6))
fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.0, 1.0],
                      left=0.03, right=0.97, wspace=0.05)

ax3d = fig.add_subplot(gs[0], projection="3d")
ax_vt = fig.add_subplot(gs[1])
ax_ei = fig.add_subplot(gs[2])

# ---------------------------------------------------------------------------
# 3D space frame panel
# ---------------------------------------------------------------------------
ax3d.set_facecolor(BG)

# Transparent panes
for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor(C_LIGHT)

for i, j, ec, _ in edges:
    ax3d.plot([nodes[i, 0], nodes[j, 0]],
              [nodes[i, 1], nodes[j, 1]],
              [nodes[i, 2], nodes[j, 2]],
              color=ec, lw=2.8, zorder=2)

for idx in range(len(nodes)):
    ax3d.scatter(*nodes[idx], color=N_COLORS[idx], s=72,
                 edgecolors="white", linewidth=1.0, zorder=5)
    offset = np.array([0.09, 0.04, 0.09])
    ax3d.text(*(nodes[idx] + offset), str(idx),
              fontsize=9, fontweight="bold", color=N_COLORS[idx])

ax3d.view_init(elev=22, azim=38)
ax3d.set_box_aspect([2, 1.5, 1])
ax3d.tick_params(labelsize=6, pad=-2)
ax3d.set_xlabel("x", fontsize=7, labelpad=-4)
ax3d.set_ylabel("y", fontsize=7, labelpad=-4)
ax3d.set_zlabel("z", fontsize=7, labelpad=-4)
ax3d.set_title("Space frame  (6 nodes, 9 edges)",
               fontsize=10, fontweight="bold", pad=8, color=C_DARK)

ax3d.legend(handles=[
    mpatches.Patch(facecolor=C_DARK, label="Bottom chord"),
    mpatches.Patch(facecolor=C_NS,   label="Top chord"),
    mpatches.Patch(facecolor=C_RS,   label="Web members"),
], loc="lower left", fontsize=7.5, frameon=True,
   framealpha=0.92, edgecolor=C_MUTED)

# ---------------------------------------------------------------------------
# Table helper
# ---------------------------------------------------------------------------
def draw_table(ax, title, col_headers, rows, row_fills, col_widths=None):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BG)

    n_rows = len(rows)
    n_cols = len(col_headers)
    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols

    TITLE_H  = 0.09
    HEADER_H = 0.09
    DATA_H   = (0.95 - TITLE_H - HEADER_H) / n_rows

    y_top    = 0.97
    y_header = y_top - TITLE_H

    # Title
    ax.text(0.5, y_top, title,
            ha="center", va="top", fontsize=10,
            fontweight="bold", color=C_DARK,
            transform=ax.transAxes)

    # Header row
    x = 0.0
    for w, hdr in zip(col_widths, col_headers):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x + 0.006, y_header - HEADER_H + 0.005),
            w - 0.012, HEADER_H - 0.010,
            boxstyle="round,pad=0.003",
            facecolor=C_DARK, edgecolor="white",
            linewidth=0.8, zorder=3, transform=ax.transAxes,
        ))
        ax.text(x + w / 2, y_header - HEADER_H / 2, hdr,
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white",
                transform=ax.transAxes, zorder=4)
        x += w

    y_data = y_header - HEADER_H

    # Data rows
    for r, (row, rfc) in enumerate(zip(rows, row_fills)):
        y_row = y_data - r * DATA_H
        x = 0.0
        for c, (w, cell) in enumerate(zip(col_widths, row)):
            ax.add_patch(mpatches.FancyBboxPatch(
                (x + 0.006, y_row - DATA_H + 0.005),
                w - 0.012, DATA_H - 0.010,
                boxstyle="round,pad=0.003",
                facecolor=rfc, edgecolor="white",
                linewidth=0.8, zorder=3, transform=ax.transAxes,
            ))
            ax.text(x + w / 2, y_row - DATA_H / 2, str(cell),
                    ha="center", va="center", fontsize=8.5,
                    fontweight="bold" if c == 0 else "normal",
                    color=C_DARK,
                    transform=ax.transAxes, zorder=4)
            x += w

# ---------------------------------------------------------------------------
# Vertex Table
# ---------------------------------------------------------------------------
draw_table(
    ax_vt,
    "V — Vertex Table",
    ["node", "x", "y", "z"],
    rows=[
        ("0", "0.0", "0.0", "0.0"),
        ("1", "2.0", "0.0", "0.0"),
        ("2", "1.0", "1.5", "0.0"),
        ("3", "0.5", "0.5", "1.0"),
        ("4", "1.5", "0.5", "1.0"),
        ("5", "1.0", "1.2", "1.0"),
    ],
    row_fills=[RC_BOT, RC_BOT, RC_BOT, RC_TOP, RC_TOP, RC_TOP],
    col_widths=[0.22, 0.26, 0.26, 0.26],
)

# ---------------------------------------------------------------------------
# Edge Index
# ---------------------------------------------------------------------------
draw_table(
    ax_ei,
    "E — Edge Index",
    ["edge", "start", "end"],
    rows=[
        ("0", "0", "1"),
        ("1", "1", "2"),
        ("2", "0", "2"),
        ("3", "3", "4"),
        ("4", "4", "5"),
        ("5", "3", "5"),
        ("6", "0", "3"),
        ("7", "1", "4"),
        ("8", "2", "5"),
    ],
    row_fills=[RC_BOT] * 3 + [RC_TOP] * 3 + [RC_WEB] * 3,
    col_widths=[0.28, 0.36, 0.36],
)

# ---------------------------------------------------------------------------
# Caption
# ---------------------------------------------------------------------------
plt.tight_layout(pad=0.8)

out_dir = Path(__file__).resolve().parent
for fmt in ["pdf", "png"]:
    out = out_dir / f"fig_graph_abstraction.{fmt}"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close()
