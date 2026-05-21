"""
fig_domain_intersection.py — Three-domain intersection diagram for chapter 1.1.2.

Circular Logistics / Computational Design / Structural Engineering
arranged in a triangle with labelled information-flow arrows.
Central zone marks the thesis contribution.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Ellipse
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
C_NS     = "#61788C"   # blue  — Computational Design
C_RS     = "#F2994B"   # orange — Circular Logistics
C_DARK   = "#2F3E4F"   # navy  — Structural Engineering
C_ACCENT = "#D9653B"   # deep orange — thesis contribution centre
C_MUTED  = "#9CA5A6"
C_LIGHT  = "#D7D9D9"
BG       = "#FFFFFF"

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7.0))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor(BG)

# ---------------------------------------------------------------------------
# Triangle node positions (in axes coords)
# Top = Computational Design, BL = Circular Logistics, BR = Structural Eng
# ---------------------------------------------------------------------------
POS = {
    "comp":  np.array([0.50, 0.82]),   # top
    "circ":  np.array([0.15, 0.20]),   # bottom-left
    "struc": np.array([0.80, 0.20]),   # bottom-right
}
CENTER = np.mean(list(POS.values()), axis=0)

# ---------------------------------------------------------------------------
# Soft background ellipses per domain (glow effect)
# ---------------------------------------------------------------------------
glow_params = [
    ("comp",  C_NS,   0.36, 0.22),
    ("circ",  C_RS,   0.30, 0.22),
    ("struc", C_DARK, 0.30, 0.22),
]
for key, color, w, h in glow_params:
    cx, cy = POS[key]
    ell = Ellipse((cx, cy), w, h, color=color, alpha=0.08, zorder=0,
                  transform=ax.transAxes)
    ax.add_patch(ell)

# ---------------------------------------------------------------------------
# Domain boxes
# ---------------------------------------------------------------------------
BOX_W = 0.220
BOX_H = 0.095
CORNER = "round,pad=0.015"

def domain_box(cx, cy, title, subtitle, color):
    x = cx - BOX_W / 2
    y = cy - BOX_H / 2
    box = mpatches.FancyBboxPatch(
        (x, y), BOX_W, BOX_H,
        boxstyle=CORNER,
        facecolor=color, edgecolor=color,
        linewidth=0, alpha=1.0, zorder=4,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(cx, cy + 0.012, title,
            ha="center", va="center", fontsize=11,
            fontweight="bold", color="white",
            transform=ax.transAxes, zorder=5)
    ax.text(cx, cy - 0.020, subtitle,
            ha="center", va="center", fontsize=7.5,
            color="white", alpha=0.88,
            transform=ax.transAxes, zorder=5)

domain_box(*POS["comp"],  "Computational Design",  "CMA-ES · GNN · MILP",      C_NS)
domain_box(*POS["circ"],  "Circular Logistics",    "reclaimed stock inventory", C_RS)
domain_box(*POS["struc"], "Structural Engineering","EC5 · FEA · Karamba3D",     C_DARK)

# ---------------------------------------------------------------------------
# Central contribution badge
# ---------------------------------------------------------------------------
cx, cy = CENTER
badge_r = 0.098
badge = plt.Circle((cx, cy), badge_r,
                   facecolor=C_ACCENT, edgecolor=BG,
                   linewidth=2.5, alpha=0.95, zorder=6,
                   transform=ax.transAxes)
ax.add_patch(badge)
ax.text(cx, cy + 0.018, "Generative",
        ha="center", va="center", fontsize=9.5,
        fontweight="bold", color="white",
        transform=ax.transAxes, zorder=7)
ax.text(cx, cy - 0.010, "Workflow",
        ha="center", va="center", fontsize=9.5,
        fontweight="bold", color="white",
        transform=ax.transAxes, zorder=7)
ax.text(cx, cy - 0.038, "(this thesis)",
        ha="center", va="center", fontsize=7,
        color="white", alpha=0.80, style="italic",
        transform=ax.transAxes, zorder=7)

# ---------------------------------------------------------------------------
# Arrows with labels
# ---------------------------------------------------------------------------
def edge_point(frm, to, offset=0.075):
    """Return a point offset along the frm→to direction from frm."""
    d = to - frm
    d = d / np.linalg.norm(d)
    return frm + d * offset

def mid_perp(frm, to, perp_dist=0.06):
    """Midpoint shifted perpendicularly for label placement."""
    mid = (frm + to) / 2
    d = to - frm
    d = d / np.linalg.norm(d)
    perp = np.array([-d[1], d[0]])
    return mid + perp * perp_dist

# (from_key, to_key, label, rad, label_t, label_perp, color)
# rad: arc curvature (positive curves left of direction)
# label_t: parametric position along arc 0-1
# label_perp: extra perpendicular nudge for the text box
arrows = [
    ("circ",  "comp",
     "stock inventory\n(lengths · sections · grades)",
     0.10, 0.45, np.array([-0.10,  0.00]), C_RS),
    ("comp",  "struc",
     "candidate geometries\n+ assignments",
     0.25, 0.50, np.array([ 0.14,  0.04]), C_NS),
    ("struc", "comp",
     "feasibility score\n(fitness feedback)",
     0.25, 0.50, np.array([ 0.13, -0.04]), C_DARK),
    ("circ",  "struc",
     "material properties\n(strength class · EC5)",
     -0.10, 0.50, np.array([ 0.00, -0.08]), C_RS),
]

for frm_key, to_key, label, rad, _, lperp, color in arrows:
    p0 = edge_point(POS[frm_key], POS[to_key], offset=0.078)
    p1 = edge_point(POS[to_key],  POS[frm_key], offset=0.078)

    ax.annotate("", xy=p1, xytext=p0,
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=1.8,
                    mutation_scale=14,
                    connectionstyle=f"arc3,rad={rad}",
                ),
                zorder=3)

    # Label midpoint with perpendicular nudge
    mid = (p0 + p1) / 2 + lperp
    ax.text(mid[0], mid[1], label,
            ha="center", va="center", fontsize=7.5,
            color=color, style="italic",
            transform=ax.transAxes, zorder=8,
            bbox=dict(boxstyle="round,pad=0.25", fc=BG,
                      ec=color, lw=0.6, alpha=0.95))

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
legend_elements = [
    mpatches.Patch(facecolor=C_RS,     label="Circular Logistics"),
    mpatches.Patch(facecolor=C_NS,     label="Computational Design"),
    mpatches.Patch(facecolor=C_DARK,   label="Structural Engineering"),
    mpatches.Patch(facecolor=C_ACCENT, label="Thesis contribution"),
]
leg = ax.legend(handles=legend_elements,
                loc="lower center", ncol=4,
                frameon=True, framealpha=0.95,
                edgecolor=C_MUTED, fontsize=8,
                bbox_to_anchor=(0.5, 0.01))
leg.get_frame().set_linewidth(0.6)

# ---------------------------------------------------------------------------
# Caption
# ---------------------------------------------------------------------------
fig.text(0.5, 0.005,
         "Figure X — Three-domain intersection of Circular Logistics, Computational Design, and Structural Engineering. "
         "Arrows indicate primary information flows. The central badge marks the generative workflow developed in this thesis.",
         ha="center", va="bottom", fontsize=7.5, color="#555555", style="italic")

plt.tight_layout(pad=0.5)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = Path(__file__).resolve().parent
for fmt in ["pdf", "png"]:
    out = out_dir / f"fig_domain_intersection.{fmt}"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close()
