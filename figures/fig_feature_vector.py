"""
fig_feature_vector.py — GNN input feature vectors for node and edge (section 4.1.3).

Left:   Node feature vector f_v ∈ ℝ¹⁰ — spatial coordinates, boundary conditions, load.
Right:  Edge feature vector f_e ∈ ℝ⁹  — cross-section geometry, stiffness, pre-solved force.
Centre: Minimal node–edge–node schematic.

Example values are from support node v0 (pin support: Tx=Ty=Tz=1, Rx=Ry=Rz=0)
and a representative adjacent member (140 × 160 mm section, C24 timber, 1.03 m length).

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
C_DARK  = "#2F3E4F"
C_NS    = "#61788C"
C_RS    = "#F2994B"
C_MUTED = "#9CA5A6"
C_LIGHT = "#D7D9D9"
BG      = "#FFFFFF"

# Feature group row fills — light versions of the three main palette colours,
# matching the member-type fills used in fig_graph_abstraction.py
RC_TOP = "#C8D8E4"   # light C_NS  → spatial coordinates / section geometry
RC_BOT = "#D2D8DD"   # light C_DARK → restraint flags / stiffness properties
RC_WEB = "#FAE3D0"   # light C_RS   → applied load / pre-computed estimate

# Header fill
H_FILL = C_DARK
H_TEXT = BG

# ---------------------------------------------------------------------------
# Feature definitions
# (index, display_name, description, example_value, unit, row_fill)
# ---------------------------------------------------------------------------
NODE_FEATS = [
    # Example: support node v0 from sample 0 (pin support — translational DOFs fixed, rotational free)
    (0, "$x$",        "x-coordinate  (centroid-normalised)",   "−7.54",          "m",       RC_TOP),
    (1, "$y$",        "y-coordinate  (centroid-normalised)",   "−4.27",          "m",       RC_TOP),
    (2, "$z$",        "z-coordinate  (centroid-normalised)",   "0.66",           "m",       RC_TOP),
    (3, "$T_x$",      "Translational restraint  x  (1=fixed)", "1",              "{0,1}",   RC_BOT),
    (4, "$T_y$",      "Translational restraint  y  (1=fixed)", "1",              "{0,1}",   RC_BOT),
    (5, "$T_z$",      "Translational restraint  z  (1=fixed)", "1",              "{0,1}",   RC_BOT),
    (6, "$R_x$",      "Rotational restraint  x  (1=fixed)",    "0",              "{0,1}",   RC_BOT),
    (7, "$R_y$",      "Rotational restraint  y  (1=fixed)",    "0",              "{0,1}",   RC_BOT),
    (8, "$R_z$",      "Rotational restraint  z  (1=fixed)",    "0",              "{0,1}",   RC_BOT),
    (9, "$F_z$",      "Applied vertical load  (−=downward)",   "−6.42",          "N",       RC_WEB),
]

EDGE_FEATS = [
    (0, "$b$",        "Cross-section width",                    "0.14",           "m",     RC_TOP),
    (1, "$h$",        "Cross-section depth",                    "0.16",           "m",     RC_TOP),
    (2, "$L$",        "Installed member length",                "1.03",           "m",     RC_TOP),
    (3, "$E$",        "Elastic modulus",                        "11.0 × 10⁹",    "Pa",    RC_BOT),
    (4, "$I_y$",      "Second moment of area  (strong axis)",   "4.78 × 10⁻⁵",  "m⁴",   RC_BOT),
    (5, "$I_z$",      "Second moment of area  (weak axis)",     "3.66 × 10⁻⁵",  "m⁴",   RC_BOT),
    (6, "$J$",        "Torsional constant",                     "2.25 × 10⁻⁶",  "m⁴",   RC_BOT),
    (7, "$EA/L$",     "Axial stiffness per unit length",        "2.39 × 10⁸",   "N/m",   RC_BOT),
    (8, "$\\hat{N}$", "Mean-stiffness FEM force estimate",      "−35 000",       "N",     RC_WEB),
]

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 14.0, 6.2
fig = plt.figure(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(1, 3, width_ratios=[2.3, 0.9, 2.1],
                      left=0.02, right=0.98, wspace=0.04)

ax_node = fig.add_subplot(gs[0])
ax_ctr  = fig.add_subplot(gs[1])
ax_edge = fig.add_subplot(gs[2])

for ax in (ax_node, ax_ctr, ax_edge):
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

# ---------------------------------------------------------------------------
# Helper: draw a feature table inside an axes
# ---------------------------------------------------------------------------
def draw_feature_table(ax, title, subtitle, features, col_widths):
    """
    Draw a feature table.

    col_widths: list of 4 values summing to 1.0 → [idx, name, description, unit]
    features: list of (idx, name, desc, value, unit, fill)
    """
    n = len(features)

    TITLE_H   = 0.078
    SUB_H     = 0.048
    HEADER_H  = 0.072
    DATA_H    = (1.0 - TITLE_H - SUB_H - HEADER_H - 0.02) / n

    y = 0.995

    # Title
    ax.text(0.5, y, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=C_DARK,
            transform=ax.transAxes)
    y -= TITLE_H

    # Subtitle (dimension tag)
    ax.text(0.5, y, subtitle, ha="center", va="top",
            fontsize=8.5, color=C_NS, style="italic",
            transform=ax.transAxes)
    y -= SUB_H

    # Header row
    headers = ["#", "feature", "description", "unit"]
    x = 0.0
    for w, hdr in zip(col_widths, headers):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x + 0.005, y - HEADER_H + 0.004), w - 0.010, HEADER_H - 0.008,
            boxstyle="round,pad=0.002",
            facecolor=H_FILL, edgecolor=BG, linewidth=0.6,
            zorder=3, transform=ax.transAxes,
        ))
        ax.text(x + w / 2, y - HEADER_H / 2, hdr,
                ha="center", va="center", fontsize=7.8,
                fontweight="bold", color=H_TEXT,
                transform=ax.transAxes, zorder=4)
        x += w
    y -= HEADER_H

    # Data rows
    for idx, name, desc, val, unit, fill in features:
        row_y = y
        x = 0.0
        cells = [str(idx), name, desc, unit]
        aligns = ["center", "center", "left", "center"]
        x_offsets = [0.5, 0.5, 0.04, 0.5]  # fraction of cell width for text x
        for c, (w, cell, ha, xoff) in enumerate(zip(col_widths, cells, aligns, x_offsets)):
            ax.add_patch(mpatches.FancyBboxPatch(
                (x + 0.005, row_y - DATA_H + 0.003), w - 0.010, DATA_H - 0.006,
                boxstyle="round,pad=0.002",
                facecolor=fill, edgecolor=BG, linewidth=0.5,
                zorder=3, transform=ax.transAxes,
            ))
            text_x = x + w * xoff
            ax.text(text_x, row_y - DATA_H / 2, cell,
                    ha=ha, va="center", fontsize=8.0,
                    fontweight="bold" if c == 1 else "normal",
                    color=C_DARK,
                    transform=ax.transAxes, zorder=4)
            x += w

        # Example value — right-aligned in the description cell
        desc_x = col_widths[0] + col_widths[1]
        desc_w = col_widths[2]
        ax.text(desc_x + desc_w - 0.015, row_y - DATA_H / 2, val,
                ha="right", va="center", fontsize=7.5,
                color=C_NS, style="italic",
                transform=ax.transAxes, zorder=4)

        y -= DATA_H


# ---------------------------------------------------------------------------
# Node feature table
# ---------------------------------------------------------------------------
draw_feature_table(
    ax_node,
    title="Node feature vector",
    subtitle="$\\mathbf{f}_v \\in \\mathbb{R}^{10}$  —  one vector per node",
    features=NODE_FEATS,
    col_widths=[0.08, 0.13, 0.64, 0.15],
)

# ---------------------------------------------------------------------------
# Edge feature table
# ---------------------------------------------------------------------------
draw_feature_table(
    ax_edge,
    title="Edge feature vector",
    subtitle="$\\mathbf{f}_e \\in \\mathbb{R}^{9}$  —  one vector per member",
    features=EDGE_FEATS,
    col_widths=[0.08, 0.13, 0.64, 0.15],
)

# ---------------------------------------------------------------------------
# Centre schematic — node i → edge (i,j) → node j
# ---------------------------------------------------------------------------
ax = ax_ctr

# Vertical positions
y_node  = 0.72   # centre of nodes / edge
y_label = 0.60   # label below nodes
y_dim_n = 0.85   # dimension bracket for node
y_dim_e = 0.85   # dimension bracket for edge

# Node i (left, bottom-chord colour)
ni_x, ni_y = 0.18, y_node
ni_r = 0.09
ax.add_patch(mpatches.Circle((ni_x, ni_y), ni_r,
             color=C_DARK, zorder=4, transform=ax.transAxes, clip_on=False))
ax.add_patch(mpatches.Circle((ni_x, ni_y), ni_r,
             fill=False, edgecolor=BG, linewidth=1.5,
             zorder=5, transform=ax.transAxes, clip_on=False))

# Node j (right, top-chord colour)
nj_x, nj_y = 0.82, y_node
ax.add_patch(mpatches.Circle((nj_x, nj_y), ni_r,
             color=C_NS, zorder=4, transform=ax.transAxes, clip_on=False))
ax.add_patch(mpatches.Circle((nj_x, nj_y), ni_r,
             fill=False, edgecolor=BG, linewidth=1.5,
             zorder=5, transform=ax.transAxes, clip_on=False))

# Edge (i, j)
ax.annotate("", xy=(nj_x - ni_r, nj_y), xytext=(ni_x + ni_r, ni_y),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=C_NS,
                            lw=2.0, mutation_scale=12))

# Node labels
ax.text(ni_x, y_label, "node $i$", ha="center", va="top",
        fontsize=8, color=C_DARK, transform=ax.transAxes)
ax.text(nj_x, y_label, "node $j$", ha="center", va="top",
        fontsize=8, color=C_NS, transform=ax.transAxes)

# Edge label
ax.text(0.5, y_node + 0.14, "edge $(i,j)$", ha="center", va="bottom",
        fontsize=7.5, color=C_NS, style="italic", transform=ax.transAxes)

# Dimension tags showing vector lengths
ax.annotate("", xy=(0.0, y_dim_n), xytext=(ni_x - 0.01, y_dim_n),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="<->", color=C_DARK, lw=0.8))
ax.text(ni_x / 2, y_dim_n + 0.04, "10D", ha="center", va="bottom",
        fontsize=7.5, color=C_DARK, fontweight="bold", transform=ax.transAxes)

ax.annotate("", xy=(nj_x + 0.01, y_dim_n), xytext=(1.0, y_dim_n),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="<->", color=C_NS, lw=0.8))
ax.text((nj_x + 1.0) / 2, y_dim_n + 0.04, "9D", ha="center", va="bottom",
        fontsize=7.5, color=C_NS, fontweight="bold", transform=ax.transAxes)

# Separator lines at left and right edges of centre panel (visual boundary)
for x_sep in [0.0, 1.0]:
    ax.axvline(x_sep, color=C_LIGHT, lw=0.8, alpha=0.7)

# Feature group legend — single column below the schematic
legend_items = [
    (RC_TOP, "spatial  ·  section"),
    (RC_BOT, "restraints  ·  stiffness"),
    (RC_WEB, "load  ·  pre-computed"),
]
y_leg = 0.30
ax.text(0.5, y_leg + 0.06, "row colour groups",
        ha="center", va="bottom", fontsize=6.5, color=C_MUTED,
        transform=ax.transAxes)
for k, (fc, label) in enumerate(legend_items):
    ly = y_leg - k * 0.072
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.08, ly - 0.024), 0.18, 0.038,
        boxstyle="round,pad=0.002",
        facecolor=fc, edgecolor=C_LIGHT, linewidth=0.5,
        transform=ax.transAxes, zorder=3,
    ))
    ax.text(0.30, ly - 0.005, label, ha="left", va="center",
            fontsize=6.5, color=C_DARK, transform=ax.transAxes)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
stem = "fig_feature_vector"
for fmt, out_dir in [("pdf", FIG_PDF_DIR), ("png", FIG_PNG_DIR)]:
    out = out_dir / f"{stem}.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close(fig)
