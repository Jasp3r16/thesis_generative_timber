"""
fig_milp_bipartite.py — Publication-quality bipartite graph for chapter 3.6.

Slots on the left, stock elements on the right.
RS edges in orange (capacity ×1), NS edges in blue (capacity ×U).
Saves to fig_milp_bipartite.pdf and .png next to this script.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Colour palette (from config.py)
# ---------------------------------------------------------------------------
C_NS    = "#61788C"   # blue  — new stock
C_RS    = "#F2994B"   # orange — reclaimed stock
C_SLOT  = "#2F3E4F"   # deep navy — structural slots
C_EDGE  = "#9CA5A6"   # muted — unselected / no edge
BG      = "#FFFFFF"

# ---------------------------------------------------------------------------
# Graph topology
# ---------------------------------------------------------------------------
# 5 structural slots
slots = ["$s_1$", "$s_2$", "$s_3$", "$s_4$", "$s_5$"]

# Stock pool: 3 RS + 4 NS
rs_elements = ["$e_1^{RS}$", "$e_2^{RS}$", "$e_3^{RS}$"]
ns_elements = ["$e_1^{NS}$", "$e_2^{NS}$", "$e_3^{NS}$", "$e_4^{NS}$"]
all_elements = rs_elements + ns_elements          # order: RS first, then NS
n_slots = len(slots)
n_elem  = len(all_elements)

# Feasibility edges (slot_idx, elem_idx)
# RS edges are drawn dashed: an RS element can be a candidate for multiple slots
# but the ×1 constraint means only one dashed edge can be activated per element.
# NS edges are solid: freely available up to U times.
rs_edges = [
    (0, 0),   # s1 — e1_RS
    (1, 0),   # s2 — e1_RS  (competing for e1_RS)
    (2, 1),   # s3 — e2_RS
    (3, 2),   # s4 — e3_RS
    (4, 2),   # s5 — e3_RS  (competing for e3_RS)
]
ns_edges = [
    (0, 3),   # s1 — e1_NS
    (1, 3),   # s2 — e1_NS
    (2, 3),   # s3 — e1_NS
    (1, 4),   # s2 — e2_NS
    (3, 4),   # s4 — e2_NS
    (2, 5),   # s3 — e3_NS
    (4, 5),   # s5 — e3_NS
    (3, 6),   # s4 — e4_NS
    (4, 6),   # s5 — e4_NS
]

# ---------------------------------------------------------------------------
# Node positions
# ---------------------------------------------------------------------------
x_left  = 0.18   # x of slot column
x_right = 0.82   # x of element column

# Vertical spacing: centre nodes within [0.1, 0.9]
def ys(n):
    if n == 1:
        return [0.5]
    return [0.9 - i * (0.8 / (n - 1)) for i in range(n)]

slot_ys = ys(n_slots)
elem_ys = ys(n_elem)

slot_pos  = {i: (x_left,  slot_ys[i]) for i in range(n_slots)}
elem_pos  = {i: (x_right, elem_ys[i]) for i in range(n_elem)}

# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5.0))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(BG)

# --- edges ---
ALPHA_EDGE = 0.55
LW_EDGE    = 1.4
CAP_OFFSET = 0.065   # how far from the node centre to place the cap label

def draw_edge(ax, p0, p1, color, lw=LW_EDGE, alpha=ALPHA_EDGE, dashed=False):
    ls = (0, (4, 3)) if dashed else "solid"
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
            color=color, lw=lw, alpha=alpha, linestyle=ls,
            solid_capstyle="round", zorder=1)

for (si, ei) in rs_edges:
    p0 = slot_pos[si]
    p1 = elem_pos[ei]
    draw_edge(ax, p0, p1, C_RS, dashed=True)

for (si, ei) in ns_edges:
    p0 = slot_pos[si]
    p1 = elem_pos[ei]
    draw_edge(ax, p0, p1, C_NS, dashed=False)

# Draw cap labels once per element (mid-bundle, last edge touching that element)
# RS: each RS element's last edge
rs_last = {}
for (si, ei) in rs_edges:
    rs_last[ei] = (si, ei)
for ei, (si, _) in rs_last.items():
    # pick the middle-ish edge for label placement
    all_si = [s for (s, e) in rs_edges if e == ei]
    mid_si = all_si[len(all_si) // 2]
    p0 = slot_pos[mid_si]
    p1 = elem_pos[ei]
    dx = p1[0] - p0[0]; dy = p1[1] - p0[1]; n = np.hypot(dx, dy)
    cx = p1[0] - dx/n * CAP_OFFSET
    cy = p1[1] - dy/n * CAP_OFFSET
    ax.text(cx, cy, "×1",
            ha="center", va="center", fontsize=6.5, color=C_RS, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc=BG, ec=C_RS, lw=0.65, alpha=0.92),
            zorder=5)

# NS: each NS element's cap
ns_last = {}
for (si, ei) in ns_edges:
    ns_last[ei] = (si, ei)
for ei, (si, _) in ns_last.items():
    all_si = [s for (s, e) in ns_edges if e == ei]
    mid_si = all_si[len(all_si) // 2]
    p0 = slot_pos[mid_si]
    p1 = elem_pos[ei]
    dx = p1[0] - p0[0]; dy = p1[1] - p0[1]; n = np.hypot(dx, dy)
    cx = p1[0] - dx/n * CAP_OFFSET
    cy = p1[1] - dy/n * CAP_OFFSET
    ax.text(cx, cy, "×U",
            ha="center", va="center", fontsize=6.5, color=C_NS, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc=BG, ec=C_NS, lw=0.65, alpha=0.92),
            zorder=5)

# --- slot nodes ---
NODE_R = 0.030
for i, label in enumerate(slots):
    cx, cy = slot_pos[i]
    circ = plt.Circle((cx, cy), NODE_R, color=C_SLOT, zorder=3)
    ax.add_patch(circ)
    ax.text(cx - 0.055, cy, label,
            ha="right", va="center",
            fontsize=9, color=C_SLOT, fontweight="bold")

# --- element nodes ---
for i, label in enumerate(all_elements):
    cx, cy = elem_pos[i]
    color = C_RS if i < len(rs_elements) else C_NS
    circ = plt.Circle((cx, cy), NODE_R, color=color, zorder=3)
    ax.add_patch(circ)
    ax.text(cx + 0.055, cy, label,
            ha="left", va="center",
            fontsize=9, color=color, fontweight="bold")

# --- column headers ---
header_y = 0.96
ax.text(x_left,  header_y, "Structural slots",
        ha="center", va="bottom", fontsize=9.5, color=C_SLOT,
        style="italic")
ax.text(x_right, header_y, "Timber stock",
        ha="center", va="bottom", fontsize=9.5, color="#444444",
        style="italic")

# divider under RS / NS elements
div_y = elem_ys[len(rs_elements) - 1] - (elem_ys[len(rs_elements) - 1] - elem_ys[len(rs_elements)]) * 0.5
ax.axhline(div_y, xmin=0.73, xmax=1.0, color=C_EDGE, lw=0.6, ls="--", alpha=0.7, zorder=0)

# sub-headers for RS / NS on the right
mid_rs = np.mean([elem_ys[i] for i in range(len(rs_elements))])
mid_ns = np.mean([elem_ys[i] for i in range(len(rs_elements), n_elem)])
ax.text(0.975, mid_rs, "RS", ha="left", va="center",
        fontsize=8, color=C_RS, fontweight="bold", rotation=90)
ax.text(0.975, mid_ns, "NS", ha="left", va="center",
        fontsize=8, color=C_NS, fontweight="bold", rotation=90)

# --- legend ---
import matplotlib.lines as mlines
patch_rs = mlines.Line2D([], [], color=C_RS, lw=1.6, linestyle=(0, (4, 3)),
                          label="Reclaimed stock  (capacity = 1)")
patch_ns = mlines.Line2D([], [], color=C_NS, lw=1.6, linestyle="solid",
                          label="New stock  (capacity = U)")
leg = ax.legend(handles=[patch_rs, patch_ns],
                loc="lower center", ncol=2,
                frameon=True, framealpha=0.95,
                edgecolor=C_EDGE, fontsize=8,
                handlelength=1.0, handleheight=0.8,
                bbox_to_anchor=(0.5, -0.04))
leg.get_frame().set_linewidth(0.6)

plt.tight_layout(pad=0.3)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = Path(__file__).resolve().parent
for fmt in ["pdf", "png"]:
    out = out_dir / f"fig_milp_bipartite.{fmt}"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close()
