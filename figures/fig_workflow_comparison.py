"""
fig_workflow_comparison.py — Problem statement figure for chapter 1.2.

Side-by-side flowchart comparing the conventional design workflow (left)
with the inventory-driven generative workflow (right).
Saves to fig_workflow_comparison.pdf and .png next to this script.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# ---------------------------------------------------------------------------
# Colour palette (from config.py)
# ---------------------------------------------------------------------------
C_NS     = "#61788C"   # blue  — new stock / conventional
C_RS     = "#F2994B"   # orange — reclaimed / inventory
C_DARK   = "#2F3E4F"   # deep navy — shared / neutral nodes
C_MUTED  = "#9CA5A6"   # grey — arrows, borders
C_DANGER = "#D9653B"   # red-orange — gap / mismatch highlight
C_LIGHT  = "#D7D9D9"   # light grey — background boxes
BG       = "#FFFFFF"

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 10.0, 8.8
BOX_W  = 0.30     # box width in axes fraction
BOX_H  = 0.072    # box height in axes fraction
ARROW_LW = 1.6
CORNER_R = 0.02

# Column x-centres (in figure fraction 0-1)
X_LEFT  = 0.25
X_RIGHT = 0.75

# ---------------------------------------------------------------------------
# Helper: draw a rounded-rectangle box with centred text
# ---------------------------------------------------------------------------
def draw_box(ax, cx, cy, label, sublabel=None,
             facecolor=C_DARK, textcolor="white",
             bordercolor=None, alpha=1.0, fontsize=9.5):
    if bordercolor is None:
        bordercolor = facecolor
    x = cx - BOX_W / 2
    y = cy - BOX_H / 2
    box = mpatches.FancyBboxPatch(
        (x, y), BOX_W, BOX_H,
        boxstyle=f"round,pad={CORNER_R}",
        facecolor=facecolor, edgecolor=bordercolor,
        linewidth=1.2, alpha=alpha, zorder=3,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(cx, cy + 0.010, label,
                ha="center", va="center", fontsize=fontsize,
                color=textcolor, fontweight="bold",
                transform=ax.transAxes, zorder=4)
        ax.text(cx, cy - 0.018, sublabel,
                ha="center", va="center", fontsize=7.5,
                color=textcolor, alpha=0.82,
                transform=ax.transAxes, zorder=4)
    else:
        ax.text(cx, cy, label,
                ha="center", va="center", fontsize=fontsize,
                color=textcolor, fontweight="bold",
                transform=ax.transAxes, zorder=4)


def draw_arrow(ax, cx, y_top, y_bot, color=C_MUTED):
    ax.annotate("",
        xy=(cx, y_bot + BOX_H / 2 + 0.005),
        xytext=(cx, y_top - BOX_H / 2 - 0.005),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=ARROW_LW,
            mutation_scale=12,
        ),
        zorder=2,
    )


def draw_gap_arrow(ax, cx, y_top, y_bot):
    """Red dashed arrow indicating the design-to-feasibility gap."""
    ax.annotate("",
        xy=(cx, y_bot + BOX_H / 2 + 0.005),
        xytext=(cx, y_top - BOX_H / 2 - 0.005),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            color=C_DANGER,
            lw=ARROW_LW,
            linestyle="dashed",
            mutation_scale=12,
        ),
        zorder=2,
    )
    # gap label — placed left of the arrow to avoid overlap with right column
    mid_y = (y_top + y_bot) / 2
    ax.text(cx - 0.04, mid_y,
            "design-to-feasibility\ngap",
            ha="right", va="center", fontsize=7.5,
            color=C_DANGER, style="italic",
            transform=ax.transAxes, zorder=5)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor(BG)

# ── Column headers ────────────────────────────────────────────────────────────
ax.text(X_LEFT, 0.96, "Conventional workflow",
        ha="center", va="bottom", fontsize=11, fontweight="bold",
        color=C_NS, transform=ax.transAxes)
ax.text(X_RIGHT, 0.96, "Inventory-driven workflow",
        ha="center", va="bottom", fontsize=11, fontweight="bold",
        color=C_RS, transform=ax.transAxes)

# divider
ax.axvline(0.5, ymin=0.03, ymax=0.98,
           color=C_LIGHT, lw=1.2, ls="--", zorder=0)

# ── LEFT column — conventional ────────────────────────────────────────────────
# Box y-centres, top to bottom
L = [0.855, 0.725, 0.595, 0.465, 0.335, 0.195]

left_steps = [
    ("Design intent",        "programme, brief, spatial requirements"),
    ("Geometric form",       "parametric model — dimensions as free variables"),
    ("Structural analysis",  "FEA — sizing against code requirements"),
    ("Specify material",     "standard catalogue — unlimited, continuous supply"),
    ("Procure",              "order to specification"),
    ("Build",                None),
]

for i, (label, sub) in enumerate(left_steps):
    draw_box(ax, X_LEFT, L[i], label, sub,
             facecolor=C_NS, textcolor="white")

for i in range(len(L) - 2):
    draw_arrow(ax, X_LEFT, L[i], L[i + 1], color=C_NS)

# Gap arrow between "Specify material" and "Procure" when reclaimed is used
draw_gap_arrow(ax, X_LEFT, L[3], L[4])

# ── RIGHT column — inventory-driven ──────────────────────────────────────────
R = [0.855, 0.725, 0.595, 0.465, 0.335, 0.195]

right_steps = [
    ("Material inventory",   "reclaimed stock — discrete, finite, non-standard"),
    ("Inventory analysis",   "length, section, grade characterisation"),
    ("Geometry search",      "CMA-ES — node positions as optimisation variables"),
    ("Stock assignment",     "MILP — optimal element-to-slot matching"),
    ("Structural validation","GNN proxy + Karamba3D FEA verification"),
    ("Build",                None),
]

for i, (label, sub) in enumerate(right_steps):
    c = C_RS if i < 2 else (C_DARK if i < 5 else C_NS)
    draw_box(ax, X_RIGHT, R[i], label, sub,
             facecolor=c, textcolor="white")

for i in range(len(R) - 1):
    draw_arrow(ax, X_RIGHT, R[i], R[i + 1], color=C_RS)

# Feedback arrow: validation → geometry search (closed loop)
# Placed on the LEFT side of the right column to stay within margins
fb_x = X_RIGHT - BOX_W / 2 - 0.03
ax.annotate("",
    xy=(fb_x, R[2]),
    xytext=(fb_x, R[4]),
    xycoords="axes fraction", textcoords="axes fraction",
    arrowprops=dict(
        arrowstyle="-|>",
        color=C_MUTED,
        lw=1.2,
        mutation_scale=10,
    ),
    zorder=2,
)
ax.text(fb_x - 0.01, (R[2] + R[4]) / 2,
        "fitness\nfeedback",
        ha="right", va="center", fontsize=7.5,
        color=C_MUTED, style="italic",
        transform=ax.transAxes, zorder=5)

# ── Shared "Build" highlight ──────────────────────────────────────────────────
# Already drawn above; add a faint horizontal band to link both
ax.axhspan(R[5] - BOX_H / 2 - 0.01, R[5] + BOX_H / 2 + 0.01,
           xmin=0.04, xmax=0.96,
           color=C_LIGHT, alpha=0.35, zorder=0, transform=ax.transAxes)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=C_NS,     label="Conventional — new / standard material"),
    mpatches.Patch(facecolor=C_RS,     label="Inventory-driven — reclaimed stock input"),
    mpatches.Patch(facecolor=C_DARK,   label="Shared computational stage"),
    mpatches.Patch(facecolor=C_DANGER, label="Design-to-feasibility gap", alpha=0.7),
]
leg = ax.legend(handles=legend_elements,
                loc="lower center", ncol=2,
                frameon=True, framealpha=0.95,
                edgecolor=C_MUTED, fontsize=8,
                bbox_to_anchor=(0.5, 0.06))
leg.get_frame().set_linewidth(0.6)

# ── Caption area ──────────────────────────────────────────────────────────────
# Caption placed below figure via tight_layout pad — omit from axes text
# Add as figure suptitle at very bottom instead
fig.text(0.5, 0.01,
         "Figure X — Conventional workflow (left) vs. inventory-driven workflow (right). "
         "The dashed red arrow marks the design-to-feasibility gap that arises when reclaimed stock\n"
         "is substituted after geometry is fixed. The inventory-driven workflow resolves this by treating the stock as the primary design input.",
         ha="center", va="bottom", fontsize=7.5, color="#555555", style="italic")

plt.tight_layout(pad=0.3)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = Path(__file__).resolve().parent
for fmt in ["pdf", "png"]:
    out = out_dir / f"fig_workflow_comparison.{fmt}"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close()
