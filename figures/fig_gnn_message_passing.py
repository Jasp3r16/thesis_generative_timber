"""
fig_gnn_message_passing.py — GNN message passing schematic, section 2.5.3.

Three left panels show message passing over k=0, 1, 2 layers on a small
planar Warren truss section (5 nodes, 7 edges).  The colour-coded rings
illustrate how the receptive field of a target member expands with each
aggregation step.  The right panel contrasts a flat MLP that operates on
the same member in isolation, with no access to topology.

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
C_NS     = "#61788C"   # blue
C_RS     = "#F2994B"   # orange
C_DARK   = "#2F3E4F"   # deep navy
C_MUTED  = "#9CA5A6"   # grey
C_TEAL   = "#4F8A8B"   # muted teal — 2-hop ring
BG       = "#FFFFFF"

COL_TARGET  = C_RS
COL_HOP1    = C_NS
COL_HOP2    = C_TEAL
COL_FADED   = "#C5CDD0"   # edges outside receptive field

# ---------------------------------------------------------------------------
# Truss topology: 5 nodes, 7 edges (simple Warren section)
#
#         n4
#        /  \
#       /    \
#  n0--n1--n2--n3
#
# TARGET edge: e1 = (n1, n2)
# 1-hop:       e0=(n0,n1)  e4=(n1,n4)  e5=(n2,n4)  e2=(n2,n3)
# 2-hop:       e3=(n0,n4)  e6=(n3,n4)
# ---------------------------------------------------------------------------
NODES = np.array([
    [0.08, 0.30],   # n0  bottom-left
    [0.35, 0.30],   # n1  bottom-second  (target left endpoint)
    [0.65, 0.30],   # n2  bottom-third   (target right endpoint)
    [0.92, 0.30],   # n3  bottom-right
    [0.50, 0.78],   # n4  apex
])

EDGES = [
    (0, 1),  # e0  bottom-chord left    — 1-hop
    (1, 2),  # e1  bottom-chord middle  — TARGET
    (2, 3),  # e2  bottom-chord right   — 1-hop
    (0, 4),  # e3  left diagonal far    — 2-hop
    (1, 4),  # e4  left diagonal near   — 1-hop
    (2, 4),  # e5  right diagonal near  — 1-hop
    (3, 4),  # e6  right diagonal far   — 2-hop
]

IDX_TARGET = 1
IDX_HOP1   = {0, 2, 4, 5}
IDX_HOP2   = {3, 6}
NODE_TARGET = {1, 2}
NODE_HOP1   = {0, 3, 4}

NODE_R = 0.058   # node radius in axes-fraction units


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge_colour(ei, show):
    if ei == IDX_TARGET:
        return COL_TARGET
    if show >= 1 and ei in IDX_HOP1:
        return COL_HOP1
    if show >= 2 and ei in IDX_HOP2:
        return COL_HOP2
    return COL_FADED


def _node_colour(ni, show):
    if ni in NODE_TARGET:
        return COL_TARGET
    if show >= 1 and ni in NODE_HOP1:
        return COL_HOP1
    return COL_FADED


def _edge_lw(ei, show):
    if ei == IDX_TARGET:
        return 3.2
    if show >= 1 and ei in IDX_HOP1:
        return 2.4
    if show >= 2 and ei in IDX_HOP2:
        return 2.0
    return 1.4


def agg_arrow(ax, src, dst, col, rad=0.28, lw=1.6, alpha=0.85):
    """Curved aggregation arrow from src node to dst node (both in axes fraction)."""
    xs, ys = NODES[src]
    xd, yd = NODES[dst]
    vec = np.array([xd - xs, yd - ys], dtype=float)
    dist = float(np.linalg.norm(vec))
    pad = NODE_R + 0.016
    xs2 = xs + vec[0] / dist * pad
    ys2 = ys + vec[1] / dist * pad
    xd2 = xd - vec[0] / dist * pad
    yd2 = yd - vec[1] / dist * pad
    ax.annotate("",
        xy=(xd2, yd2), xytext=(xs2, ys2),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            connectionstyle=f"arc3,rad={rad}",
            color=col, lw=lw, mutation_scale=12, alpha=alpha,
        ),
        zorder=7,
    )


def draw_graph(ax, show, agg_arrows=None):
    """
    Draw the 5-node truss graph.
    show = 0 → only target highlighted
    show = 1 → target + 1-hop highlighted
    show = 2 → target + 1-hop + 2-hop highlighted
    agg_arrows: list of (src, dst, col, rad) tuples
    """
    # Edges
    for ei, (i, j) in enumerate(EDGES):
        col = _edge_colour(ei, show)
        lw  = _edge_lw(ei, show)
        ax.plot([NODES[i, 0], NODES[j, 0]],
                [NODES[i, 1], NODES[j, 1]],
                color=col, lw=lw, zorder=2,
                solid_capstyle="round",
                transform=ax.transAxes)

    # Aggregation arrows
    if agg_arrows:
        for (s, d, c, r) in agg_arrows:
            agg_arrow(ax, s, d, c, rad=r)

    # Nodes
    for ni in range(len(NODES)):
        cx, cy = NODES[ni]
        fc = _node_colour(ni, show)
        ec = "white"
        lw = 2.0 if fc != COL_FADED else 1.0
        circ = plt.Circle((cx, cy), NODE_R,
                           facecolor=fc, edgecolor=ec,
                           linewidth=lw, zorder=4,
                           transform=ax.transAxes)
        ax.add_patch(circ)
        # Node label
        lbl_col = "white" if fc != COL_FADED else "#999999"
        ax.text(cx, cy, f"$v_{ni}$",
                ha="center", va="center", fontsize=7,
                color=lbl_col, fontweight="bold",
                transform=ax.transAxes, zorder=5)

    # Target edge label
    i, j = EDGES[IDX_TARGET]
    mx = (NODES[i, 0] + NODES[j, 0]) / 2
    my = (NODES[i, 1] + NODES[j, 1]) / 2
    ax.text(mx, my - NODE_R - 0.06,
            "$e_{\\mathrm{tgt}}$",
            ha="center", va="top", fontsize=8,
            color=COL_TARGET, fontweight="bold",
            transform=ax.transAxes, zorder=5)


# ---------------------------------------------------------------------------
# Figure + GridSpec
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 13.0, 5.2
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)

gs = fig.add_gridspec(
    1, 5,
    width_ratios=[1.0, 1.0, 1.0, 0.06, 0.85],
    left=0.02, right=0.98, bottom=0.07, top=0.87,
    wspace=0.04,
)

ax_k0  = fig.add_subplot(gs[0, 0])
ax_k1  = fig.add_subplot(gs[0, 1])
ax_k2  = fig.add_subplot(gs[0, 2])
ax_gap = fig.add_subplot(gs[0, 3])
ax_mlp = fig.add_subplot(gs[0, 4])

ax_gap.axis("off")

for ax in [ax_k0, ax_k1, ax_k2, ax_mlp]:
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.02, 1.05)
    ax.axis("off")
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_visible(False)

# Subtle panel backgrounds
for ax in [ax_k0, ax_k1, ax_k2]:
    rect = mpatches.FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.02",
        facecolor="#F7F9FB", edgecolor=C_MUTED,
        linewidth=0.8, zorder=0,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(rect)


# ---------------------------------------------------------------------------
# Panel 0 — k = 0  (target edge only)
# ---------------------------------------------------------------------------
draw_graph(ax_k0, show=0)
ax_k0.set_title("$k = 0$  ·  initial features",
                fontsize=9.5, fontweight="bold", color=C_DARK, pad=5)

# Annotate feature vector on target nodes
for ni, xoff in [(1, -0.05), (2, +0.05)]:
    cx, cy = NODES[ni]
    ha = "right" if xoff < 0 else "left"
    ax_k0.text(cx + xoff, cy + NODE_R + 0.06,
               "[L, A, I, E]",
               ha=ha, va="bottom", fontsize=6.5,
               color=COL_TARGET, style="italic",
               transform=ax_k0.transAxes, zorder=5)

ax_k0.text(0.50, 0.10,
           "each node holds its\nown features only",
           ha="center", va="top", fontsize=7.5, color=C_MUTED,
           style="italic", transform=ax_k0.transAxes, zorder=5)


# ---------------------------------------------------------------------------
# Panel 1 — k = 1  (1-hop aggregation)
# ---------------------------------------------------------------------------
agg_k1 = [
    (0, 1, COL_HOP1,  0.25),
    (4, 1, COL_HOP1, -0.25),
    (4, 2, COL_HOP1,  0.25),
    (3, 2, COL_HOP1, -0.25),
]
draw_graph(ax_k1, show=1, agg_arrows=agg_k1)
ax_k1.set_title("$k = 1$  ·  1-hop aggregation",
                fontsize=9.5, fontweight="bold", color=COL_HOP1, pad=5)

ax_k1.text(0.50, 0.10,
           "target nodes absorb\ndirect neighbours",
           ha="center", va="top", fontsize=7.5, color=COL_HOP1,
           style="italic", transform=ax_k1.transAxes, zorder=5)


# ---------------------------------------------------------------------------
# Panel 2 — k = 2  (2-hop aggregation)
# ---------------------------------------------------------------------------
# First round: 2-hop → 1-hop (teal arrows, lighter)
# Second round: 1-hop → target (blue arrows)
agg_k2 = [
    # round 1: 2-hop nodes → 1-hop nodes (sends through apex)
    (0, 4, COL_HOP2,  0.30),
    (3, 4, COL_HOP2, -0.30),
    # round 2: 1-hop → target (same as k=1 arrows)
    (0, 1, COL_HOP1,  0.25),
    (4, 1, COL_HOP1, -0.25),
    (4, 2, COL_HOP1,  0.25),
    (3, 2, COL_HOP1, -0.25),
]
draw_graph(ax_k2, show=2, agg_arrows=agg_k2)
ax_k2.set_title("$k = 2$  ·  2-hop aggregation",
                fontsize=9.5, fontweight="bold", color=COL_HOP2, pad=5)

ax_k2.text(0.50, 0.10,
           "full load-path context\nreached in two layers",
           ha="center", va="top", fontsize=7.5, color=COL_HOP2,
           style="italic", transform=ax_k2.transAxes, zorder=5)


# ---------------------------------------------------------------------------
# Layer-transition arrows between GNN panels (in figure coordinates)
# ---------------------------------------------------------------------------
def inter_panel_arrow(ax_left, ax_right, label, fig):
    p_l = ax_left.get_position()
    p_r = ax_right.get_position()
    x = (p_l.x1 + p_r.x0) / 2
    y = (p_l.y0 + p_l.y1) / 2 + 0.04
    fig.text(x, y, "→", ha="center", va="center",
             fontsize=14, color=C_MUTED, transform=fig.transFigure)
    fig.text(x, y - 0.068, label, ha="center", va="center",
             fontsize=7, color=C_MUTED, style="italic",
             transform=fig.transFigure)

inter_panel_arrow(ax_k0, ax_k1, "layer 1\nmessage pass", fig)
inter_panel_arrow(ax_k1, ax_k2, "layer 2\nmessage pass", fig)


# ---------------------------------------------------------------------------
# MLP contrast panel
# ---------------------------------------------------------------------------
def _box(ax, cx, cy, w, h, fc, ec=C_MUTED, lw=1.2, zorder=3,
         label="", sublabel=""):
    patch = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.025",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(patch)
    tc = "white" if fc not in (BG, "#F7F9FB") else C_DARK
    if label:
        ax.text(cx, cy + (0.016 if sublabel else 0), label,
                ha="center", va="center", fontsize=7.5, fontweight="bold",
                color=tc, transform=ax.transAxes, zorder=zorder + 1)
    if sublabel:
        ax.text(cx, cy - 0.028, sublabel,
                ha="center", va="center", fontsize=6.5, color="#AAAAAA",
                transform=ax.transAxes, zorder=zorder + 1)


def _arr(ax, x, y0, y1, col=C_MUTED):
    ax.annotate("",
        xy=(x, y1), xytext=(x, y0),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5, mutation_scale=11),
        zorder=8,
    )


ax = ax_mlp

# Background
rect = mpatches.FancyBboxPatch(
    (0, 0), 1, 1,
    boxstyle="round,pad=0.02",
    facecolor="#FFF8F5", edgecolor=C_RS,
    linewidth=0.8, linestyle="dashed", zorder=0,
    transform=ax.transAxes, clip_on=False,
)
ax.add_patch(rect)

ax.set_title("MLP  (no context)",
             fontsize=9.5, fontweight="bold", color=C_DARK, pad=5)

# Isolated member icon
MEM_Y = 0.880
ax.plot([0.22, 0.78], [MEM_Y, MEM_Y],
        color=COL_TARGET, lw=3.8, solid_capstyle="round",
        transform=ax.transAxes, zorder=3)
for nx in (0.22, 0.78):
    ax.add_patch(plt.Circle((nx, MEM_Y), NODE_R * 0.85,
                             facecolor=COL_TARGET, edgecolor="white",
                             linewidth=1.5, zorder=4,
                             transform=ax.transAxes))
ax.text(0.50, MEM_Y + NODE_R + 0.04,
        "$e_{\\mathrm{tgt}}$  (isolated)",
        ha="center", va="bottom", fontsize=8, color=COL_TARGET,
        fontweight="bold", transform=ax.transAxes, zorder=5)

# Feature vector
_box(ax, 0.50, 0.725, 0.75, 0.075, fc=C_DARK,
     label="[L,  α,  A,  E·I,  ...]",
     sublabel="per-member features only")

_arr(ax, 0.50, 0.688, 0.640)

# Hidden layers
for y, lbl in [(0.600, "dense  64"), (0.490, "dense  64"), (0.378, "output  1")]:
    _box(ax, 0.50, y, 0.70, 0.072, fc=C_NS, label=lbl)
    if y != 0.378:
        _arr(ax, 0.50, y - 0.036, y - 0.086)

# Output label
ax.text(0.50, 0.302,
        "safe / unsafe",
        ha="center", va="center", fontsize=7.5, fontweight="bold",
        color=C_DARK, transform=ax.transAxes, zorder=5)

# "No context" warning badge
_box(ax, 0.50, 0.175, 0.86, 0.110, fc="#FFF0F0",
     ec="#CC3333", lw=0.9, zorder=4)
ax.text(0.50, 0.205, "no load-path information",
        ha="center", va="center", fontsize=7.5, color="#CC3333",
        fontweight="bold", transform=ax.transAxes, zorder=5)
ax.text(0.50, 0.160, "neighbour context discarded",
        ha="center", va="center", fontsize=7, color="#CC3333",
        style="italic", transform=ax.transAxes, zorder=5)


# ---------------------------------------------------------------------------
# Vertical divider between GNN section and MLP panel
# ---------------------------------------------------------------------------
p_k2  = ax_k2.get_position()
p_mlp = ax_mlp.get_position()
div_x = (p_k2.x1 + p_mlp.x0) / 2
line2d = plt.Line2D(
    [div_x, div_x], [0.05, 0.93],
    transform=fig.transFigure,
    color=C_MUTED, lw=1.0, ls="--",
)
fig.add_artist(line2d)


# ---------------------------------------------------------------------------
# Section header
# ---------------------------------------------------------------------------
p0 = ax_k0.get_position()
p2 = ax_k2.get_position()
fig.text(
    (p0.x0 + p2.x1) / 2, 0.94,
    "GNN — message passing expands receptive field across truss topology",
    ha="center", va="bottom", fontsize=10, fontweight="bold", color=C_DARK,
)


# ---------------------------------------------------------------------------
# Legend (below the three GNN panels)
# ---------------------------------------------------------------------------
legend_patches = [
    mpatches.Patch(facecolor=COL_TARGET, label="target member  $e_{\\mathrm{tgt}}$"),
    mpatches.Patch(facecolor=COL_HOP1,   label="1-hop  (direct neighbours)"),
    mpatches.Patch(facecolor=COL_HOP2,   label="2-hop  (neighbours of neighbours)"),
    mpatches.Patch(facecolor=COL_FADED,  label="outside receptive field"),
]
ax_k1.legend(
    handles=legend_patches,
    loc="lower center",
    ncol=4,
    frameon=True, framealpha=0.95,
    edgecolor=C_MUTED, fontsize=7.5,
    bbox_to_anchor=(0.50, -0.20),
    bbox_transform=ax_k1.transAxes,
)



# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
stem = "fig_gnn_message_passing"
for fmt, out_dir in [("pdf", FIG_PDF_DIR), ("png", FIG_PNG_DIR)]:
    out = out_dir / f"{stem}.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close(fig)
