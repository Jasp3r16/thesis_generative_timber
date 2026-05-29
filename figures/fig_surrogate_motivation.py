"""
fig_surrogate_motivation.py — Surrogate model motivation, section 4.1.1.

Left panel: flat MLP — coordinate vector input loses topology, produces a
single scalar feasibility score.  Right panel: GNN — operates on the graph
directly, message passing mirrors force propagation, produces per-member
probabilities.  The contrast makes the architectural motivation concrete
without requiring the reader to understand GNN mechanics in advance.

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
# Colour palette — consistent with project theme
# ---------------------------------------------------------------------------
C_NS    = "#61788C"   # blue
C_RS    = "#F2994B"   # orange
C_DARK  = "#2F3E4F"   # deep navy
C_MUTED = "#9CA5A6"   # grey
C_SAFE  = "#4F8A8B"   # muted teal — safe probability
C_FAIL  = "#D9653B"   # red-orange — high failure probability
C_WARN  = "#F2994B"   # amber — borderline
BG      = "#FFFFFF"

# ---------------------------------------------------------------------------
# Small Warren truss for the GNN panel
# 5 nodes (bottom chord + apex), 7 edges
#
#        n4
#       /  \  \
#      /    \  \
#  n0----n1----n2---n3
#
# Edge failure probabilities (illustrative, span low→high left→right)
# ---------------------------------------------------------------------------
NODES = np.array([
    [0.08, 0.30],   # n0  bottom-left
    [0.37, 0.30],   # n1  bottom-second
    [0.63, 0.30],   # n2  bottom-third
    [0.92, 0.30],   # n3  bottom-right
    [0.50, 0.76],   # n4  apex
])

EDGES = [
    (0, 1),   # e0  bottom-chord left
    (1, 2),   # e1  bottom-chord middle
    (2, 3),   # e2  bottom-chord right
    (0, 4),   # e3  left diagonal
    (1, 4),   # e4  inner-left diagonal
    (2, 4),   # e5  inner-right diagonal
    (3, 4),   # e6  right diagonal
]

# Illustrative failure probabilities: chords under higher load, diagonals safer
EDGE_PROBS = [0.12, 0.71, 0.45, 0.08, 0.22, 0.18, 0.61]

def _prob_colour(p):
    """Interpolate green→amber→red based on failure probability."""
    if p < 0.4:
        t = p / 0.4
        r = int(79  + t * (242 - 79))
        g = int(138 + t * (153 - 138))
        b = int(139 + t * (59  - 139))
    else:
        t = (p - 0.4) / 0.6
        r = int(242 + t * (217 - 242))
        g = int(153 + t * (101 - 153))
        b = int(59  + t * (59  - 59))
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Figure — 2-panel layout
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 13.0, 6.6
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)

gs = fig.add_gridspec(
    1, 3,
    width_ratios=[1.0, 0.04, 1.4],
    left=0.03, right=0.97, bottom=0.08, top=0.88,
    wspace=0.0,
)
ax_mlp = fig.add_subplot(gs[0, 0])
ax_div = fig.add_subplot(gs[0, 1])
ax_gnn = fig.add_subplot(gs[0, 2])

ax_div.axis("off")
for ax in (ax_mlp, ax_gnn):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BG)

# Panel background — light tint for the MLP (contrast / "wrong approach")
for ax, fc, ec, ls in [
    (ax_mlp, "#F9F6F4", C_MUTED, "solid"),
    (ax_gnn, "#F4F8F9", C_NS,    "solid"),
]:
    ax.add_patch(mpatches.FancyBboxPatch(
        (-0.02, -0.02), 1.04, 1.04,
        boxstyle="round,pad=0.015",
        facecolor=fc, edgecolor=ec,
        linewidth=0.9, linestyle=ls, zorder=0,
        transform=ax.transAxes, clip_on=False,
    ))


# ---------------------------------------------------------------------------
# Helper: rounded box with optional sublabel
# ---------------------------------------------------------------------------
def box(ax, cx, cy, w, h, fc, ec=None, lw=1.1, label="", sub="",
        tc="white", fontsize=8.5, zorder=3):
    ec = ec or fc
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.018",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, zorder=zorder,
        transform=ax.transAxes, clip_on=False,
    ))
    if label:
        ax.text(cx, cy + (0.014 if sub else 0), label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=tc,
                transform=ax.transAxes, zorder=zorder + 1)
    if sub:
        ax.text(cx, cy - 0.024, sub,
                ha="center", va="center", fontsize=6.8,
                color=tc, alpha=0.80,
                transform=ax.transAxes, zorder=zorder + 1)


def arr(ax, x, y0, y1, col=C_MUTED, lw=1.5):
    ax.annotate("",
        xy=(x, y1), xytext=(x, y0),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                        mutation_scale=11),
        zorder=8,
    )


# ===========================================================================
# LEFT PANEL — MLP
# ===========================================================================
ax = ax_mlp

# ── Input: coordinate vector ──────────────────────────────────────────────
VEC_Y = 0.885
VEC_W = 0.78
VEC_H = 0.085
box(ax, 0.50, VEC_Y, VEC_W, VEC_H, fc=C_DARK,
    label="[x₁, y₁,  x₂, y₂,  …,  xₙ, yₙ]",
    sub="flat coordinate vector — 78 values")

# "Topology lost" callout badge
ax.add_patch(mpatches.FancyBboxPatch(
    (0.50 - 0.38, VEC_Y + VEC_H/2 + 0.01),
    0.76, 0.055,
    boxstyle="round,pad=0.01",
    facecolor="#FDE8E0", edgecolor=C_FAIL,
    linewidth=0.9, zorder=4,
    transform=ax.transAxes, clip_on=False,
))
ax.text(0.50, VEC_Y + VEC_H/2 + 0.038,
        "connectivity discarded — nodes become anonymous numbers",
        ha="center", va="center", fontsize=6.8,
        color=C_FAIL, style="italic",
        transform=ax.transAxes, zorder=5)

# ── Truss icon above, crossed out ────────────────────────────────────────
ICON_Y = 0.975
for (i, j) in EDGES:
    nx0, ny0 = NODES[i]; nx1, ny1 = NODES[j]
    # rescale to a narrow strip
    sx = 0.10 + (nx0 * 0.80); ex = 0.10 + (nx1 * 0.80)
    sy = ICON_Y - 0.01 + (ny0 - 0.30) * 0.19
    ey = ICON_Y - 0.01 + (ny1 - 0.30) * 0.19
    ax.plot([sx, ex], [sy, ey], color=C_MUTED, lw=1.4,
            alpha=0.5, transform=ax.transAxes, zorder=2)

# Cross-out slash
ax.plot([0.12, 0.88], [ICON_Y - 0.025, ICON_Y + 0.025],
        color=C_FAIL, lw=2.2, transform=ax.transAxes, zorder=6)
ax.text(0.91, ICON_Y, "×",
        ha="left", va="center", fontsize=12,
        color=C_FAIL, fontweight="bold",
        transform=ax.transAxes, zorder=6)

# ── Dense layers ─────────────────────────────────────────────────────────
arr(ax, 0.50, VEC_Y - VEC_H/2 - 0.005, 0.715)
box(ax, 0.50, 0.685, 0.65, 0.075, fc=C_NS,
    label="Dense  ×  2", sub="ReLU  ·  256 units")

arr(ax, 0.50, 0.648, 0.570)
box(ax, 0.50, 0.540, 0.65, 0.075, fc=C_NS,
    label="Dense  ×  2", sub="ReLU  ·  128 units")

arr(ax, 0.50, 0.503, 0.425)

# ── Output ────────────────────────────────────────────────────────────────
box(ax, 0.50, 0.395, 0.55, 0.075, fc=C_MUTED,
    label="Output — 1 value", sub="global feasibility score")

# ── Limitation note ───────────────────────────────────────────────────────
ax.add_patch(mpatches.FancyBboxPatch(
    (0.04, 0.06), 0.92, 0.28,
    boxstyle="round,pad=0.015",
    facecolor="#FDE8E0", edgecolor=C_FAIL,
    linewidth=0.8, zorder=3,
    transform=ax.transAxes, clip_on=False,
))
for i, line in enumerate([
    "No structural correspondence:",
    "force propagation follows the graph —",
    "a flat vector carries no graph.",
    "Indeterminate trusses require neighbour",
    "context the MLP cannot access.",
]):
    ax.text(0.50, 0.295 - i * 0.042, line,
            ha="center", va="center",
            fontsize=7.2,
            color=C_FAIL if i == 0 else "#5C2C1A",
            fontweight="bold" if i == 0 else "normal",
            transform=ax.transAxes, zorder=4)


# ===========================================================================
# RIGHT PANEL — GNN
# ===========================================================================
ax = ax_gnn

NODE_R = 0.045

# ── Truss icon (top — topology preserved) ────────────────────────────────
TRUSS_Y_BASE = 0.75
TRUSS_SCALE_X = 0.75
TRUSS_SCALE_Y = 0.22

def truss_pos(raw_x, raw_y):
    """Map raw NODES coords into the upper portion of the GNN panel."""
    x = 0.12 + raw_x * TRUSS_SCALE_X
    y = TRUSS_Y_BASE + (raw_y - 0.30) * TRUSS_SCALE_Y
    return x, y


# Draw edges coloured by failure probability
for ei, (i, j) in enumerate(EDGES):
    x0, y0 = truss_pos(*NODES[i])
    x1, y1 = truss_pos(*NODES[j])
    col = _prob_colour(EDGE_PROBS[ei])
    ax.plot([x0, x1], [y0, y1], color=col, lw=5.0,
            solid_capstyle="round",
            transform=ax.transAxes, zorder=2)
    # Probability label near edge midpoint
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    offset_y = -0.055 if my < TRUSS_Y_BASE + 0.08 else 0.045
    ax.text(mx, my + offset_y, f"{EDGE_PROBS[ei]:.2f}",
            ha="center", va="center", fontsize=7.0,
            color=col, fontweight="bold",
            transform=ax.transAxes, zorder=5)

# Draw nodes
for ni in range(len(NODES)):
    cx, cy = truss_pos(*NODES[ni])
    ax.add_patch(plt.Circle((cx, cy), NODE_R,
                             facecolor=C_DARK, edgecolor="white",
                             linewidth=1.8, zorder=4,
                             transform=ax.transAxes))
    ax.text(cx, cy, f"$v_{ni}$",
            ha="center", va="center", fontsize=7,
            color="white", fontweight="bold",
            transform=ax.transAxes, zorder=5)

# Topology-preserved badge
ax.add_patch(mpatches.FancyBboxPatch(
    (0.04, TRUSS_Y_BASE + TRUSS_SCALE_Y * (0.78 - 0.30) + 0.025),
    0.92, 0.05,
    boxstyle="round,pad=0.01",
    facecolor="#E4F2F0", edgecolor=C_SAFE,
    linewidth=0.8, zorder=4,
    transform=ax.transAxes, clip_on=False,
))
ax.text(0.50,
        TRUSS_Y_BASE + TRUSS_SCALE_Y * (0.78 - 0.30) + 0.050,
        "graph topology preserved — each edge knows its neighbours",
        ha="center", va="center", fontsize=7.0,
        color=C_SAFE, style="italic",
        transform=ax.transAxes, zorder=5)

# ── Input feature boxes ───────────────────────────────────────────────────
arr(ax, 0.50, TRUSS_Y_BASE - 0.035, 0.645)
box(ax, 0.50, 0.618, 0.88, 0.075, fc=C_DARK,
    label="Node features: [x, y, support?]   ·   Edge features: [L, A, E·I, grade]",
    sub="one feature vector per node; one per member", fontsize=7.8)

# ── GNN layers ────────────────────────────────────────────────────────────
arr(ax, 0.50, 0.580, 0.508)
box(ax, 0.50, 0.478, 0.70, 0.075, fc=C_NS,
    label="Message passing  ×  3 layers",
    sub="nodes aggregate neighbours → mirrors force propagation")

arr(ax, 0.50, 0.440, 0.370)
box(ax, 0.50, 0.340, 0.70, 0.075, fc=C_NS,
    label="Edge readout MLP",
    sub="per-member hidden state → failure probability")

arr(ax, 0.50, 0.302, 0.230)

# ── Output: 120 coloured probability bars ─────────────────────────────────
# Mini bar chart: 120 members shown as thin vertical bars
BAR_Y0 = 0.065
BAR_H_MAX = 0.145
BAR_AREA_X0 = 0.04
BAR_AREA_W  = 0.92
N_BARS = 120

np.random.seed(42)
# Simulate realistic probability distribution: mostly safe, some failures
probs_120 = np.clip(
    np.concatenate([
        np.random.beta(1.5, 9, 60),    # web diagonals — mostly safe
        np.random.beta(3,   5, 38),    # top chord — moderate
        np.random.beta(2,   4, 22),    # bottom chord — variable
    ]),
    0, 1
)
probs_120 = probs_120[np.argsort(np.random.permutation(N_BARS))]  # shuffle order

bar_w = BAR_AREA_W / N_BARS * 0.72
for k, p in enumerate(probs_120):
    bx = BAR_AREA_X0 + (k + 0.5) * (BAR_AREA_W / N_BARS)
    bh = BAR_H_MAX * p
    ax.add_patch(mpatches.Rectangle(
        (bx - bar_w/2, BAR_Y0), bar_w, bh,
        facecolor=_prob_colour(p), edgecolor="none",
        transform=ax.transAxes, zorder=3,
    ))

# Threshold line at p = 0.5
thresh_y = BAR_Y0 + BAR_H_MAX * 0.5
ax.plot([BAR_AREA_X0, BAR_AREA_X0 + BAR_AREA_W],
        [thresh_y, thresh_y],
        color=C_FAIL, lw=1.0, ls="--",
        transform=ax.transAxes, zorder=4)
ax.text(BAR_AREA_X0 + BAR_AREA_W + 0.01, thresh_y,
        "p = 0.5",
        va="center", fontsize=6.5, color=C_FAIL,
        transform=ax.transAxes, zorder=5)

# x-axis label
ax.text(0.50, BAR_Y0 - 0.038,
        "120 members  ·  per-member failure probability  $\\hat{p}_e$",
        ha="center", va="top", fontsize=7.5, color=C_DARK,
        transform=ax.transAxes, zorder=5)

# Output box border
ax.add_patch(mpatches.FancyBboxPatch(
    (BAR_AREA_X0 - 0.015, BAR_Y0 - 0.010),
    BAR_AREA_W + 0.03, BAR_H_MAX + 0.030,
    boxstyle="round,pad=0.01",
    facecolor="none", edgecolor=C_MUTED,
    linewidth=0.7, zorder=2,
    transform=ax.transAxes, clip_on=False,
))


# ===========================================================================
# Divider line between panels
# ===========================================================================
p_mlp = ax_mlp.get_position()
p_gnn = ax_gnn.get_position()
div_x = (p_mlp.x1 + p_gnn.x0) / 2
fig.add_artist(plt.Line2D(
    [div_x, div_x], [0.06, 0.94],
    transform=fig.transFigure,
    color=C_MUTED, lw=1.0, ls="--",
))
fig.text(div_x, 0.955, "vs.",
         ha="center", va="bottom", fontsize=13,
         color=C_MUTED, fontweight="bold",
         transform=fig.transFigure)


# ===========================================================================
# Colour legend for failure probability
# ===========================================================================
legend_ax = fig.add_axes([0.56, 0.005, 0.38, 0.042])
legend_ax.set_xlim(0, 1); legend_ax.set_ylim(0, 1)
legend_ax.axis("off")

grad_n = 200
for k in range(grad_n):
    p = k / (grad_n - 1)
    legend_ax.add_patch(mpatches.Rectangle(
        (k / grad_n, 0.55), 1 / grad_n, 0.45,
        facecolor=_prob_colour(p), edgecolor="none",
        transform=legend_ax.transAxes,
    ))
for px, lbl in [(0.0, "0.0"), (0.5, "0.5"), (1.0, "1.0")]:
    legend_ax.text(px, 0.35, lbl, ha="center", va="top",
                   fontsize=7, color=C_DARK,
                   transform=legend_ax.transAxes)
legend_ax.text(0.50, 0.05,
               "member failure probability  $\\hat{p}_e$",
               ha="center", va="bottom", fontsize=7,
               color=C_MUTED, transform=legend_ax.transAxes)
legend_ax.add_patch(mpatches.FancyBboxPatch(
    (-0.01, 0.0), 1.02, 1.0,
    boxstyle="round,pad=0.01",
    facecolor="none", edgecolor=C_MUTED,
    linewidth=0.6,
    transform=legend_ax.transAxes, clip_on=False,
))


# ===========================================================================
# Figure title
# ===========================================================================
fig.text(0.50, 0.955,
         "Surrogate model motivation: MLP vs. GNN for per-member safety prediction",
         ha="center", va="bottom", fontsize=11.5,
         fontweight="bold", color=C_DARK,
         transform=fig.transFigure)


# ===========================================================================
# Save
# ===========================================================================
stem = "fig_surrogate_motivation"
for fmt, out_dir in [("pdf", FIG_PDF_DIR), ("png", FIG_PNG_DIR)]:
    out = out_dir / f"{stem}.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close(fig)
