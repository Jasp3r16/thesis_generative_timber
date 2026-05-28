"""
fig_node_connections.py — Rigid vs pinned node connection for chapter 2.2.

Left:  rigid connection — all DOF transferred, artificial bending moments introduced.
Right: pinned connection (internal hinge) — R_y, R_z released, pure axial transfer.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import FIG_PDF_DIR, FIG_PNG_DIR

C_NS     = "#61788C"
C_DARK   = "#2F3E4F"
C_ACCENT = "#D9653B"
C_MUTED  = "#9CA5A6"
C_LIGHT  = "#D7D9D9"
BG       = "#FFFFFF"

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 5.2))
fig.patch.set_facecolor(BG)

NODE_R = 0.13
MEM_L  = 0.72
MEM_LW = 16
PIN_R  = 0.08
DIRS   = [(1, 0), (-1, 0), (0, 1), (0, -1)]

for ax in (ax_l, ax_r):
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.25, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG)

# ---------------------------------------------------------------------------
# Shared: draw four members
# ---------------------------------------------------------------------------
def draw_members(ax):
    for dx, dy in DIRS:
        ax.plot([dx * (NODE_R + 0.02), dx * MEM_L],
                [dy * (NODE_R + 0.02), dy * MEM_L],
                color=C_NS, lw=MEM_LW,
                solid_capstyle="round", zorder=2)

# ---------------------------------------------------------------------------
# LEFT — Rigid connection
# ---------------------------------------------------------------------------
draw_members(ax_l)

# Solid node
ax_l.add_patch(plt.Circle((0, 0), NODE_R,
                           facecolor=C_DARK, edgecolor="white",
                           linewidth=2.0, zorder=5))

# Moment arcs (two arcs, opposite quadrants)
for theta1, theta2 in [(25, 155), (205, 335)]:
    arc = mpatches.Arc((0, 0), 0.44, 0.44, angle=0,
                       theta1=theta1, theta2=theta2,
                       color=C_ACCENT, lw=2.8, zorder=6)
    ax_l.add_patch(arc)
    # Arrowhead at arc end
    t_end  = np.radians(theta2)
    t_prev = np.radians(theta2 - 10)
    r = 0.22
    ax_l.annotate("",
        xy=(r * np.cos(t_end),   r * np.sin(t_end)),
        xytext=(r * np.cos(t_prev), r * np.sin(t_prev)),
        arrowprops=dict(arrowstyle="-|>", color=C_ACCENT,
                        lw=1.5, mutation_scale=11),
        zorder=7)

# "M" label
ax_l.text(0.36, 0.26, "M", ha="center", va="center",
          fontsize=13, fontweight="bold", color=C_ACCENT, zorder=7)

# Panel labels
ax_l.text(0, -0.96, "Rigid connection",
          ha="center", va="top", fontsize=11,
          fontweight="bold", color=C_DARK)

# ---------------------------------------------------------------------------
# RIGHT — Pinned connection
# ---------------------------------------------------------------------------
draw_members(ax_r)

# Node (lighter to distinguish from rigid)
ax_r.add_patch(plt.Circle((0, 0), NODE_R,
                           facecolor=C_NS, edgecolor="white",
                           linewidth=2.0, zorder=5))

# Pin release circles at member–node junctions
for dx, dy in DIRS:
    px = dx * (NODE_R + PIN_R + 0.03)
    py = dy * (NODE_R + PIN_R + 0.03)
    ax_r.add_patch(plt.Circle((px, py), PIN_R,
                               facecolor=BG, edgecolor=C_DARK,
                               linewidth=2.2, zorder=6))

# DOF annotation
ax_r.text(0.52, 0.50,
          "R$_y$, R$_z$\nreleased",
          ha="left", va="center", fontsize=9, color=C_DARK,
          bbox=dict(boxstyle="round,pad=0.28", fc=BG, ec=C_MUTED, lw=0.8),
          zorder=7)
ax_r.annotate("",
    xy=(PIN_R + 0.03, NODE_R + PIN_R + 0.01),
    xytext=(0.47, 0.50),
    arrowprops=dict(arrowstyle="-|>", color=C_MUTED,
                    lw=1.2, mutation_scale=9),
    zorder=6)

# "Axial only" annotation
ax_r.text(-0.60, 0.0,
          "N\n(axial)",
          ha="center", va="center", fontsize=9,
          color=C_NS, fontweight="bold", zorder=6)
ax_r.annotate("",
    xy=(-MEM_L + 0.08, 0),
    xytext=(-0.48, 0),
    arrowprops=dict(arrowstyle="-|>", color=C_NS,
                    lw=1.4, mutation_scale=10),
    zorder=6)

# Panel labels
ax_r.text(0, -0.96, "Pinned connection  (internal hinge)",
          ha="center", va="top", fontsize=11,
          fontweight="bold", color=C_DARK)

# ---------------------------------------------------------------------------
# Divider between panels
# ---------------------------------------------------------------------------
fig.add_artist(mlines.Line2D([0.5, 0.5], [0.06, 0.94],
                             transform=fig.transFigure,
                             color=C_LIGHT, lw=1.2, ls="--"))

# ---------------------------------------------------------------------------
# Caption
# ---------------------------------------------------------------------------
plt.tight_layout(pad=1.0)

out_dirs = {"pdf": FIG_PDF_DIR, "png": FIG_PNG_DIR}
for fmt in ["pdf", "png"]:
    out = out_dirs[fmt] / f"fig_node_connections.{fmt}"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close()
