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
# Figure size driven by a single panel-size constant so content is never condensed.
# PANEL_SIZE = physical width (and height) of one ratio-1.0 panel in inches.
PANEL_SIZE = 3.4

_L, _R, _B, _T = 0.02, 0.98, 0.13, 0.88
_RATIOS = [1.0, 1.0, 1.0, 0.06, 1.0]

FIG_W = PANEL_SIZE * sum(_RATIOS) / (_R - _L)
FIG_H = PANEL_SIZE / (_T - _B)

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)

gs = fig.add_gridspec(
    1, 5,
    width_ratios=_RATIOS,
    left=_L, right=_R, bottom=_B, top=_T,
    wspace=0.06,
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
                fontsize=9.5, fontweight="bold", color=C_DARK, pad=10)


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
                fontsize=9.5, fontweight="bold", color=COL_HOP1, pad=10)


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
                fontsize=9.5, fontweight="bold", color=COL_HOP2, pad=10)


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
# MLP contrast panel — same graph, frozen at k = 0 (no aggregation ever)
# ---------------------------------------------------------------------------
ax_mlp.add_patch(mpatches.FancyBboxPatch(
    (0, 0), 1, 1,
    boxstyle="round,pad=0.02",
    facecolor="#F7F9FB", edgecolor=C_RS,
    linewidth=1.2, zorder=0,
    transform=ax_mlp.transAxes, clip_on=False,
))
draw_graph(ax_mlp, show=0)
ax_mlp.set_title("MLP  ·  no aggregation",
                 fontsize=9.5, fontweight="bold", color=C_RS, pad=10)


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
# Legend (below the three GNN panels)
# ---------------------------------------------------------------------------
legend_patches = [
    mpatches.Patch(facecolor=COL_TARGET, label="target member  $e_{\\mathrm{tgt}}$"),
    mpatches.Patch(facecolor=COL_HOP1,   label="1-hop  (direct neighbours)"),
    mpatches.Patch(facecolor=COL_HOP2,   label="2-hop  (neighbours of neighbours)"),
    mpatches.Patch(facecolor=COL_FADED,  label="outside receptive field"),
]
fig.legend(
    handles=legend_patches,
    loc="lower center",
    ncol=4,
    frameon=True, framealpha=0.95,
    edgecolor=C_MUTED, fontsize=7.5,
    bbox_to_anchor=(0.50, 0.01),
    bbox_transform=fig.transFigure,
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
