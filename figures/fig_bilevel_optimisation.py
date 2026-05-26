"""
fig_bilevel_optimisation.py — Conceptual bilevel structure diagram, section 2.5.1.

Two nested boxes:
  Outer box: upper level — CMA-ES continuous geometry optimiser
  Inner box: lower level — discrete timber assignment (MILP) + structural
             evaluation (FEA), with GNN surrogate intercepting the simulation
             bottleneck.

Annotates data flow between levels and the surrogate interception point.
Saves to fig_bilevel_optimisation.pdf and .png next to this script.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
        sys.path.append(str(ROOT_DIR))

from config import FIG_PDF_DIR, FIG_PNG_DIR

# ---------------------------------------------------------------------------
# Colour palette (consistent with config.py / other thesis figures)
# ---------------------------------------------------------------------------
C_NS    = "#61788C"   # blue  — upper level / geometry
C_RS    = "#F2994B"   # orange — surrogate
C_DARK  = "#2F3E4F"   # deep navy — lower level / assignment
C_MUTED = "#9CA5A6"   # grey — secondary arrows, borders
BG      = "#FFFFFF"

CORNER = 0.024   # FancyBboxPatch corner radius

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 9.5, 7.0
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor(BG)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def add_box(ax, x0, y0, w, h, fc, ec, lw=1.6, zorder=1, ls="solid"):
    patch = mpatches.FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle=f"round,pad={CORNER}",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, linestyle=ls, zorder=zorder,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(patch)


def arrow(ax, x0, y0, x1, y1, color=C_MUTED, lw=1.7, ms=13):
    ax.annotate("",
        xy=(x1, y1), xytext=(x0, y0),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=ms),
        zorder=12,
    )


def line(ax, x0, y0, x1, y1, color=C_MUTED, lw=1.7):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw,
            transform=ax.transAxes, zorder=12, solid_capstyle="round")


# ---------------------------------------------------------------------------
# OUTER BOX — upper level (continuous geometry optimiser)
# ---------------------------------------------------------------------------
OX, OY, OW, OH = 0.03, 0.08, 0.94, 0.87
add_box(ax, OX, OY, OW, OH,
        fc="#EDF2F7", ec=C_NS, lw=2.4, zorder=1)

ax.text(OX + 0.018, OY + OH - 0.018,
        "UPPER LEVEL",
        ha="left", va="top", fontsize=10, fontweight="bold",
        color=C_NS, transform=ax.transAxes, zorder=5)
ax.text(OX + 0.018, OY + OH - 0.058,
        "CMA-ES  ·  Continuous Geometry Optimiser",
        ha="left", va="top", fontsize=8.5, color=C_NS,
        style="italic", transform=ax.transAxes, zorder=5)


# ---------------------------------------------------------------------------
# INNER BOX — lower level (discrete assignment + structural evaluation)
# ---------------------------------------------------------------------------
IX, IY, IW, IH = 0.400, 0.105, 0.560, 0.800
add_box(ax, IX, IY, IW, IH,
        fc="#F5F1EC", ec=C_DARK, lw=2.0, zorder=2)

ax.text(IX + 0.018, IY + IH - 0.018,
        "LOWER LEVEL",
        ha="left", va="top", fontsize=10, fontweight="bold",
        color=C_DARK, transform=ax.transAxes, zorder=5)
ax.text(IX + 0.018, IY + IH - 0.058,
        "Discrete Assignment  +  Structural Evaluation",
        ha="left", va="top", fontsize=8, color=C_DARK,
        style="italic", transform=ax.transAxes, zorder=5)


# ---------------------------------------------------------------------------
# Left-side outer level boxes (geometry parameter source + fitness receiver)
# ---------------------------------------------------------------------------
GEO_CX, GEO_CY = 0.186, 0.660
GEO_W,  GEO_H  = 0.215, 0.180

add_box(ax, GEO_CX - GEO_W/2, GEO_CY - GEO_H/2, GEO_W, GEO_H,
        fc=C_NS, ec=C_NS, lw=1.2, zorder=3)
ax.text(GEO_CX, GEO_CY + 0.028, "Geometry",
        ha="center", va="center", fontsize=9.5, fontweight="bold",
        color="white", transform=ax.transAxes, zorder=5)
ax.text(GEO_CX, GEO_CY - 0.010, "parameters  θ",
        ha="center", va="center", fontsize=8.5, color="white",
        transform=ax.transAxes, zorder=5)
ax.text(GEO_CX, GEO_CY - 0.048, "continuous,  ℝⁿ",
        ha="center", va="center", fontsize=7, color="white",
        style="italic", transform=ax.transAxes, zorder=5)

FIT_CX, FIT_CY = 0.186, 0.305
FIT_W,  FIT_H  = 0.215, 0.180

add_box(ax, FIT_CX - FIT_W/2, FIT_CY - FIT_H/2, FIT_W, FIT_H,
        fc=C_NS, ec=C_NS, lw=1.2, zorder=3)
ax.text(FIT_CX, FIT_CY + 0.028, "Fitness  f(θ)",
        ha="center", va="center", fontsize=9.5, fontweight="bold",
        color="white", transform=ax.transAxes, zorder=5)
ax.text(FIT_CX, FIT_CY - 0.010, "→  update CMA-ES",
        ha="center", va="center", fontsize=8, color="white",
        transform=ax.transAxes, zorder=5)
ax.text(FIT_CX, FIT_CY - 0.048, "covariance  σ",
        ha="center", va="center", fontsize=7, color="white",
        style="italic", transform=ax.transAxes, zorder=5)


# ---------------------------------------------------------------------------
# Inner-level sub-box 1: Timber Assignment (MILP)
# ---------------------------------------------------------------------------
MILP_CX, MILP_CY = 0.572, 0.660
MILP_W,  MILP_H  = 0.210, 0.250

add_box(ax, MILP_CX - MILP_W/2, MILP_CY - MILP_H/2, MILP_W, MILP_H,
        fc=C_DARK, ec=C_DARK, lw=1.2, zorder=4)
ax.text(MILP_CX, MILP_CY + 0.045, "Timber",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color="white", transform=ax.transAxes, zorder=6)
ax.text(MILP_CX, MILP_CY + 0.005, "Assignment",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color="white", transform=ax.transAxes, zorder=6)
ax.text(MILP_CX, MILP_CY - 0.042, "MILP  ·  discrete",
        ha="center", va="center", fontsize=7.5, color=C_RS,
        fontweight="bold", transform=ax.transAxes, zorder=6)


# ---------------------------------------------------------------------------
# Inner-level sub-box 2: Structural Simulation (FEA)
# ---------------------------------------------------------------------------
FEA_CX, FEA_CY = 0.572, 0.305
FEA_W,  FEA_H  = 0.210, 0.250

add_box(ax, FEA_CX - FEA_W/2, FEA_CY - FEA_H/2, FEA_W, FEA_H,
        fc=C_DARK, ec=C_DARK, lw=1.2, zorder=4)
ax.text(FEA_CX, FEA_CY + 0.045, "Structural",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color="white", transform=ax.transAxes, zorder=6)
ax.text(FEA_CX, FEA_CY + 0.005, "Simulation",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color="white", transform=ax.transAxes, zorder=6)
ax.text(FEA_CX, FEA_CY - 0.038, "FEA  ·  K·u = f",
        ha="center", va="center", fontsize=7.5, color=C_MUTED,
        transform=ax.transAxes, zorder=6)
ax.text(FEA_CX, FEA_CY - 0.065, "stiffness assembly + factorisation",
        ha="center", va="center", fontsize=6.5, color=C_MUTED,
        style="italic", transform=ax.transAxes, zorder=6)


# ---------------------------------------------------------------------------
# Surrogate box (intercepts FEA simulation)
# ---------------------------------------------------------------------------
SUR_CX, SUR_CY = 0.840, 0.305
SUR_W,  SUR_H  = 0.155, 0.220

add_box(ax, SUR_CX - SUR_W/2, SUR_CY - SUR_H/2, SUR_W, SUR_H,
        fc=C_RS, ec=C_RS, lw=1.4, zorder=4)
ax.text(SUR_CX, SUR_CY + 0.048, "GNN",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color="white", transform=ax.transAxes, zorder=6)
ax.text(SUR_CX, SUR_CY + 0.010, "Surrogate",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color="white", transform=ax.transAxes, zorder=6)
ax.text(SUR_CX, SUR_CY - 0.030, "≈ FEA",
        ha="center", va="center", fontsize=8.5, color="white",
        transform=ax.transAxes, zorder=6)
ax.text(SUR_CX, SUR_CY - 0.062, "~100× faster",
        ha="center", va="center", fontsize=7, color="white",
        style="italic", transform=ax.transAxes, zorder=6)


# ---------------------------------------------------------------------------
# Arrows — internal flow (within inner box)
# ---------------------------------------------------------------------------

# MILP → FEA (vertical, within inner box)
arrow(ax, MILP_CX, MILP_CY - MILP_H/2,
         FEA_CX,  FEA_CY  + FEA_H/2,
         color=C_DARK, lw=1.7, ms=13)
INNER_MID_Y = (MILP_CY - MILP_H/2 + FEA_CY + FEA_H/2) / 2
ax.text(MILP_CX + 0.030, INNER_MID_Y,
        "element–slot\nmatching",
        ha="left", va="center", fontsize=7, color=C_DARK,
        style="italic", transform=ax.transAxes, zorder=6)

# FEA → Surrogate (horizontal, within inner box)
arrow(ax, FEA_CX + FEA_W/2, FEA_CY,
         SUR_CX - SUR_W/2, SUR_CY,
         color=C_RS, lw=1.7, ms=13)

# Annotation: surrogate intercepts bottleneck — placed above the connection
MID_X = (FEA_CX + FEA_W/2 + SUR_CX - SUR_W/2) / 2
ax.text(MID_X, SUR_CY + SUR_H/2 + 0.030,
        "surrogate intercepts\nsimulation bottleneck",
        ha="center", va="bottom", fontsize=7.5, color=C_RS,
        fontweight="bold", style="italic",
        transform=ax.transAxes, zorder=6)


# ---------------------------------------------------------------------------
# Arrows — cross-level flow (geometry ↓ and cost/feasibility ↑)
# ---------------------------------------------------------------------------

# Geometry θ → inner box left edge at MILP height
arrow(ax, GEO_CX + GEO_W/2, GEO_CY,
         IX, MILP_CY,
         color=C_NS, lw=1.9, ms=14)
MX = (GEO_CX + GEO_W/2 + IX) / 2
# Label above the arrow — in the gap between left boxes and inner box
ax.text(MX, GEO_CY + 0.065,
        "geometry  θ",
        ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        color=C_NS, transform=ax.transAxes, zorder=6,
        bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none", alpha=0.85))
ax.text(MX, GEO_CY + 0.025,
        "(continuous,  ℝⁿ)",
        ha="center", va="bottom", fontsize=7, color=C_NS,
        style="italic", transform=ax.transAxes, zorder=6,
        bbox=dict(boxstyle="round,pad=0.12", fc=BG, ec="none", alpha=0.85))

# Inner box left edge → Fitness box at FEA height
arrow(ax, IX, FEA_CY,
         FIT_CX + FIT_W/2, FIT_CY,
         color=C_DARK, lw=1.9, ms=14)
MX2 = (IX + FIT_CX + FIT_W/2) / 2
# Label below the arrow
ax.text(MX2, FIT_CY - 0.068,
        "assignment cost\n+  feasibility",
        ha="center", va="top", fontsize=8.5, fontweight="bold",
        color=C_DARK, transform=ax.transAxes, zorder=6,
        bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none", alpha=0.85))
ax.text(MX2, FIT_CY - 0.115,
        "non-smooth,  non-differentiable",
        ha="center", va="top", fontsize=7, color=C_DARK,
        style="italic", transform=ax.transAxes, zorder=6,
        bbox=dict(boxstyle="round,pad=0.12", fc=BG, ec="none", alpha=0.85))


# ---------------------------------------------------------------------------
# Feedback loop: Fitness → Geometry (left outer wall, U-shaped path)
# ---------------------------------------------------------------------------
FB_X = 0.054   # x of the vertical feedback leg

line(ax, FIT_CX - FIT_W/2, FIT_CY, FB_X, FIT_CY, color=C_NS, lw=1.5)
line(ax, FB_X, FIT_CY, FB_X, GEO_CY, color=C_NS, lw=1.5)
arrow(ax, FB_X, GEO_CY, GEO_CX - GEO_W/2, GEO_CY, color=C_NS, lw=1.5, ms=11)

ax.text(FB_X + 0.012, (FIT_CY + GEO_CY) / 2,
        "update\nstrategy",
        ha="left", va="center", fontsize=7, color=C_NS,
        style="italic", transform=ax.transAxes, zorder=6)

plt.tight_layout(pad=0.25)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dirs = {"pdf": FIG_PDF_DIR, "png": FIG_PNG_DIR}
for fmt in ("pdf", "png"):
        out = out_dirs[fmt] / f"fig_bilevel_optimisation.{fmt}"
        plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
        print(f"Saved: {out}")

plt.close()
