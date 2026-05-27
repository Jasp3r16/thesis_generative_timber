"""
fig_cost_matrix.py — Cost matrix illustration, section 2.5.4.

Grid of n_stock × n_slots cells.  Each cell encodes the CO₂e cost
of assigning a given stock element to a structural slot.  Cell colours
run green (low) → orange (high); infeasible pairings are grey + ×.
One cell is annotated to show its additive cost components.

Saves to figures/pdf/ and figures/png/.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
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
C_NS    = "#61788C"
C_RS    = "#F2994B"
C_DARK  = "#2F3E4F"
C_MUTED = "#9CA5A6"
BG      = "#FFFFFF"

COST_CMAP = LinearSegmentedColormap.from_list(
    "cost", ["#8DC88A", "#F5DC6E", "#E8855E"], N=256
)
INF_FC  = "#D8D8D8"
INF_EC  = "#AAAAAA"

# ---------------------------------------------------------------------------
# Data — 8 stock elements (rows: 3 RS + 5 NS) × 6 structural slots (cols)
# ---------------------------------------------------------------------------
INF = np.inf
COST = np.array([
    # s₁    s₂    s₃    s₄    s₅    s₆
    [ 2.1,  INF,  3.2,  INF,  4.8,  INF],   # e₁  RS
    [ INF,  1.8,  INF,  5.4,  3.1,  INF],   # e₂  RS
    [ INF,  4.3,  2.6,  INF,  INF,  3.7],   # e₃  RS
    [ 3.8,  5.1,  INF,  4.2,  6.1,  3.5],   # e₄  NS
    [ 5.2,  3.7,  4.4,  INF,  2.9,  5.8],   # e₅  NS
    [ INF,  4.8,  3.1,  5.7,  INF,  4.0],   # e₆  NS
    [ 4.1,  INF,  5.8,  3.9,  5.2,  INF],   # e₇  NS
    [ 6.3,  5.9,  INF,  4.5,  INF,  3.2],   # e₈  NS
])

N_ELEM, N_SLOT = COST.shape
N_RS = 3

ELEM_LABELS = (
    [f"$e_{i+1}^{{\\mathrm{{RS}}}}$" for i in range(N_RS)] +
    [f"$e_{i+1}^{{\\mathrm{{NS}}}}$" for i in range(N_ELEM - N_RS)]
)
SLOT_LABELS = [f"$s_{j+1}$" for j in range(N_SLOT)]

finite_vals = COST[np.isfinite(COST)]
V_MIN, V_MAX = finite_vals.min(), finite_vals.max()

# ---------------------------------------------------------------------------
# Figure geometry
# ---------------------------------------------------------------------------
CW = 0.90   # cell width  (data units)
CH = 0.82   # cell height (data units)
GAP = 0.10  # gap between cells

FIG_W, FIG_H = 11.5, 7.2
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(BG)

# Data axis limits: x = [−2.2, N_SLOT+1.8],  y = [−1.4, N_ELEM+2.2]
ax.set_xlim(-2.2, N_SLOT + 1.8)
ax.set_ylim(-1.0, N_ELEM + 1.0)
ax.set_aspect("equal")
ax.axis("off")


# ---------------------------------------------------------------------------
# Grid cells
# ---------------------------------------------------------------------------
def cell_y(row):
    """Convert row index to bottom-left y, so row 0 is at the top."""
    return (N_ELEM - 1 - row) + GAP / 2


for row in range(N_ELEM):
    for col in range(N_SLOT):
        val = COST[row, col]
        x0  = col + GAP / 2
        y0  = cell_y(row)
        if not np.isfinite(val):
            fc, ec, lw = INF_FC, INF_EC, 0.7
        else:
            t = (val - V_MIN) / (V_MAX - V_MIN)
            fc = COST_CMAP(t)
            ec = C_MUTED
            lw = 0.9

        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), CW, CH,
            boxstyle="round,pad=0.05",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
        ))

        if np.isfinite(val):
            ax.text(col + 0.5, y0 + CH / 2, f"{val:.1f}",
                    ha="center", va="center",
                    fontsize=8.5, color=C_DARK, fontweight="bold", zorder=3)
        else:
            # × mark
            m = 0.20
            ax.plot([col + m, col + 1 - m], [y0 + m * (CH/CW), y0 + CH - m * (CH/CW)],
                    color=INF_EC, lw=1.5, zorder=3, solid_capstyle="round")
            ax.plot([col + 1 - m, col + m], [y0 + m * (CH/CW), y0 + CH - m * (CH/CW)],
                    color=INF_EC, lw=1.5, zorder=3, solid_capstyle="round")



# ---------------------------------------------------------------------------
# Column headers — structural slots
# ---------------------------------------------------------------------------
for col, lbl in enumerate(SLOT_LABELS):
    ax.text(col + 0.5, N_ELEM + 0.50, lbl,
            ha="center", va="center",
            fontsize=9.5, color=C_DARK, fontweight="bold")

# Light separator bar under slot headers
ax.add_patch(mpatches.Rectangle(
    (-0.05, N_ELEM + 0.02), N_SLOT + 0.10, 0.03,
    facecolor=C_MUTED, alpha=0.35, zorder=1,
))


# ---------------------------------------------------------------------------
# Row headers — stock elements (left side)
# ---------------------------------------------------------------------------
for row, (lbl, col) in enumerate(zip(ELEM_LABELS,
                                     [C_RS]*N_RS + [C_NS]*(N_ELEM-N_RS))):
    y0 = cell_y(row)
    ax.text(-0.28, y0 + CH / 2, lbl,
            ha="right", va="center",
            fontsize=9.5, color=col, fontweight="bold")

# Group annotations — vertically centred next to each group's rows
ANN_X = -1.50

rs_y_center = (cell_y(0) + CH + cell_y(N_RS - 1)) / 2
ns_y_center = (cell_y(N_RS) + CH + cell_y(N_ELEM - 1)) / 2

ax.text(ANN_X, rs_y_center, "Reclaimed\nstock  (RS)",
        ha="center", va="center", fontsize=8.5,
        color=C_RS, fontweight="bold", linespacing=1.5)
ax.text(ANN_X, ns_y_center, "New\nstock  (NS)",
        ha="center", va="center", fontsize=8.5,
        color=C_NS, fontweight="bold", linespacing=1.5)

# Dashed divider between RS and NS rows
div_y = cell_y(N_RS - 1) - GAP / 2 + 0.02
ax.plot([-0.05, N_SLOT + 0.05], [div_y, div_y],
        color=C_MUTED, lw=0.8, ls="--", alpha=0.7, zorder=1)

ax.text(N_SLOT / 2, div_y - 0.04,
        "⋯  reclaimed / new-stock boundary  ⋯",
        ha="center", va="top", fontsize=7, color=C_MUTED, style="italic")


# ---------------------------------------------------------------------------
# Colour-ramp legend (right side)
# ---------------------------------------------------------------------------
RAMP_X0 = N_SLOT + 0.35
RAMP_W  = 0.38
N_STEPS = 50

for i in range(N_STEPS):
    t  = i / (N_STEPS - 1)
    y0 = i * (N_ELEM / N_STEPS)
    ax.add_patch(mpatches.Rectangle(
        (RAMP_X0, y0), RAMP_W, N_ELEM / N_STEPS + 0.01,
        facecolor=COST_CMAP(t), edgecolor="none", zorder=2,
    ))
ax.add_patch(mpatches.Rectangle(
    (RAMP_X0, 0), RAMP_W, N_ELEM,
    facecolor="none", edgecolor=C_MUTED, lw=0.9, zorder=3,
))

for val, y in [(V_MIN, 0), ((V_MIN+V_MAX)/2, N_ELEM/2), (V_MAX, N_ELEM)]:
    ax.plot([RAMP_X0 + RAMP_W, RAMP_X0 + RAMP_W + 0.12], [y, y],
            color=C_MUTED, lw=0.8)
    ax.text(RAMP_X0 + RAMP_W + 0.18, y, f"{val:.1f}",
            va="center", fontsize=7.5, color=C_DARK)
ax.text(RAMP_X0 + RAMP_W / 2, N_ELEM + 0.45, "kg CO₂e",
        ha="center", va="bottom", fontsize=7.5, color=C_DARK)

# Infeasible legend swatch (just below the ramp)
INF_Y0 = -0.75
ax.add_patch(mpatches.FancyBboxPatch(
    (RAMP_X0, INF_Y0), RAMP_W, 0.50,
    boxstyle="round,pad=0.03",
    facecolor=INF_FC, edgecolor=INF_EC, lw=0.8, zorder=2,
))
m = 0.07
ax.plot([RAMP_X0 + m, RAMP_X0 + RAMP_W - m],
        [INF_Y0 + m, INF_Y0 + 0.50 - m],
        color=INF_EC, lw=1.4, zorder=3)
ax.plot([RAMP_X0 + RAMP_W - m, RAMP_X0 + m],
        [INF_Y0 + m, INF_Y0 + 0.50 - m],
        color=INF_EC, lw=1.4, zorder=3)
ax.text(RAMP_X0 + RAMP_W + 0.18, INF_Y0 + 0.25,
        "infeasible  (∞)", va="center", fontsize=7.5, color=C_DARK)



plt.tight_layout(rect=[0.04, 0.04, 0.96, 0.97])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
stem = "fig_cost_matrix"
for fmt, out_dir in [("pdf", FIG_PDF_DIR), ("png", FIG_PNG_DIR)]:
    out = out_dir / f"{stem}.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close(fig)
