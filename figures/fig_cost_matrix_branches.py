"""
fig_cost_matrix_branches.py — Two-branch cost formula, section 2.5.x.

Split-path diagram: C_ij branches at χ_i into new-stock and reclaimed-stock
paths, each listing its LCA component boxes (A1–A3, A4, A5_prep, A5_saw),
converging into shared end-of-life terms (C1, C2, C3+C4) → total C_ij.
Excluded / zero-burden terms shown greyed with strikethrough.
No caption embedded — provided separately.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import FIG_PDF_DIR, FIG_PNG_DIR

C_NS    = "#61788C"
C_RS    = "#F2994B"
C_DARK  = "#2F3E4F"
C_MUTED = "#9CA5A6"
C_EXCL  = "#E4E4E4"
BG      = "#FFFFFF"

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 12.0, 10.5
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 12)
ax.set_ylim(0, 10.5)
ax.axis("off")

CX_L = 3.0      # new-stock column centre
CX_R = 9.0      # reclaimed column centre
BW   = 4.6      # component box width
BH   = 0.82     # component box height
R    = 0.07     # corner radius

# y-centres (top to bottom)
Y_FORM  = 9.65
SPLIT_Y = 8.80
Y1      = 7.90   # A1–A3
Y2      = 6.72   # A4
Y3      = 5.54   # A5_prep
Y4      = 4.36   # A5_saw
MERGE_Y = 3.54   # horizontal merge bar
SHARED_Y = 2.62  # shared end-of-life boxes
TOTAL_Y  = 1.38  # total cost box

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def box(cx, cy, w, h, fc, ec, lw=1.5):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad={R}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        zorder=3, clip_on=False))


def label(cx, cy, title, sub=None, tcol=C_DARK, excluded=False):
    alpha = 0.50 if excluded else 1.0
    dy = 0.14 if sub else 0.0
    ax.text(cx, cy + dy, title,
            ha="center", va="center", fontsize=9.0,
            fontweight="bold", color=C_MUTED if excluded else tcol,
            alpha=alpha, zorder=4)
    if sub:
        ax.text(cx, cy - 0.19, sub,
                ha="center", va="center", fontsize=7.0,
                color=C_MUTED, style="italic", alpha=alpha, zorder=4)
    if excluded:
        ax.plot([cx - BW/2 + 0.32, cx + BW/2 - 0.32], [cy + dy, cy + dy],
                color=C_MUTED, lw=1.2, alpha=0.55, zorder=5)


def arr(x, y0, y1, col=C_DARK, lw=1.4):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
        arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                        mutation_scale=11), zorder=5)

# ---------------------------------------------------------------------------
# Section backgrounds (light tint behind each column of component boxes)
# ---------------------------------------------------------------------------
bg_top = Y1 + BH/2 + 0.18
bg_bot = Y4 - BH/2 - 0.18
ax.add_patch(FancyBboxPatch(
    (CX_L - BW/2 - 0.20, bg_bot), BW + 0.40, bg_top - bg_bot,
    boxstyle="round,pad=0.05",
    facecolor=C_NS + "0D", edgecolor=C_NS + "38", linewidth=0.8,
    zorder=1, clip_on=False))
ax.add_patch(FancyBboxPatch(
    (CX_R - BW/2 - 0.20, bg_bot), BW + 0.40, bg_top - bg_bot,
    boxstyle="round,pad=0.05",
    facecolor=C_RS + "0D", edgecolor=C_RS + "38", linewidth=0.8,
    zorder=1, clip_on=False))

# Shared section background
ax.add_patch(FancyBboxPatch(
    (0.42, SHARED_Y - BH/2 - 0.16), 11.16, BH + 0.32,
    boxstyle="round,pad=0.05",
    facecolor=C_DARK + "09", edgecolor=C_DARK + "28", linewidth=0.8,
    zorder=1, clip_on=False))

# ---------------------------------------------------------------------------
# Formula box
# ---------------------------------------------------------------------------
box(6.0, Y_FORM, 11.2, 0.86, fc=C_DARK + "18", ec=C_DARK, lw=1.8)
ax.text(6.0, Y_FORM + 0.20,
        r"$C_{ij} = \chi_i \cdot C_{ij}^{\mathrm{new}} + (1 - \chi_i) \cdot C_{ij}^{\mathrm{rec}}$",
        ha="center", va="center", fontsize=13,
        fontweight="bold", color=C_DARK, zorder=4)
ax.text(6.0, Y_FORM - 0.18,
        r"$\chi_i$ resolved from stock origin: $\;\chi_i = 1$ for NS  "
        r"$\quad\quad$ $\chi_i = 0$ for RS",
        ha="center", va="center", fontsize=8.0,
        color=C_MUTED, style="italic", zorder=4)

# ---------------------------------------------------------------------------
# Branch split
# ---------------------------------------------------------------------------
ax.plot([6.0, 6.0], [Y_FORM - 0.43, SPLIT_Y], color=C_DARK, lw=1.4, zorder=5)
ax.plot([CX_L, CX_R], [SPLIT_Y, SPLIT_Y], color=C_DARK, lw=1.4, zorder=5)

for cx, col, chi_lbl, route_lbl in [
    (CX_L, C_NS, r"$\chi_i = 1$", "new (virgin) stock"),
    (CX_R, C_RS, r"$\chi_i = 0$", "reclaimed stock"),
]:
    arr(cx, SPLIT_Y, Y1 + BH/2 + 0.06, col=col)
    mid_y = (SPLIT_Y + Y1 + BH/2) / 2
    ax.text(cx, mid_y + 0.09, chi_lbl,
            ha="center", va="center", fontsize=9.0,
            fontweight="bold", color=col, zorder=6,
            bbox=dict(boxstyle="round,pad=0.14", fc=BG, ec="none", alpha=0.92))
    ax.text(cx, mid_y - 0.13, route_lbl,
            ha="center", va="center", fontsize=7.5,
            color=col, style="italic", zorder=6,
            bbox=dict(boxstyle="round,pad=0.10", fc=BG, ec="none", alpha=0.92))

# ---------------------------------------------------------------------------
# Row 1 — A1–A3
# ---------------------------------------------------------------------------
box(CX_L, Y1, BW, BH, fc=C_NS + "28", ec=C_NS, lw=1.5)
label(CX_L, Y1, "A1–A3  ·  production",
      "raw material, manufacture, sawmill  (kg CO₂e / m³)")

box(CX_R, Y1, BW, BH, fc=C_EXCL, ec=C_MUTED, lw=0.9)
label(CX_R, Y1, "A1–A3  =  0  ·  zero-burden assumption",
      "production in prior life cycle — not re-charged to this use",
      excluded=True)

# ---------------------------------------------------------------------------
# Row 2 — A4 transport
# ---------------------------------------------------------------------------
arr(CX_L, Y1 - BH/2 - 0.04, Y2 + BH/2 + 0.04, col=C_NS)
arr(CX_R, Y1 - BH/2 - 0.04, Y2 + BH/2 + 0.04, col=C_RS)

box(CX_L, Y2, BW, BH, fc=C_NS + "28", ec=C_NS, lw=1.5)
label(CX_L, Y2, "A4  ·  transport  (supplier → site)",
      "tkm × emission factor  per element length")

box(CX_R, Y2, BW, BH, fc=C_RS + "28", ec=C_RS, lw=1.5)
label(CX_R, Y2, "A4  ·  transport  (donor → site)",
      "tkm × emission factor  per element length")

# ---------------------------------------------------------------------------
# Row 3 — A5_prep
# ---------------------------------------------------------------------------
arr(CX_L, Y2 - BH/2 - 0.04, Y3 + BH/2 + 0.04, col=C_NS)
arr(CX_R, Y2 - BH/2 - 0.04, Y3 + BH/2 + 0.04, col=C_RS)

box(CX_L, Y3, BW, BH, fc=C_EXCL, ec=C_MUTED, lw=0.9)
label(CX_L, Y3, "A5_prep  ·  preparation",
      "not applicable to virgin timber",
      excluded=True)

box(CX_R, Y3, BW, BH, fc=C_RS + "28", ec=C_RS, lw=1.5)
label(CX_R, Y3, "A5_prep  ·  preparation",
      "inspection, cleaning, surface treatment")

# ---------------------------------------------------------------------------
# Row 4 — A5_saw
# ---------------------------------------------------------------------------
arr(CX_L, Y3 - BH/2 - 0.04, Y4 + BH/2 + 0.04, col=C_NS)
arr(CX_R, Y3 - BH/2 - 0.04, Y4 + BH/2 + 0.04, col=C_RS)

box(CX_L, Y4, BW, BH, fc=C_NS + "28", ec=C_NS, lw=1.5)
label(CX_L, Y4, "A5_saw  ·  on-site sawing",
      "cutting to required length  ·  kg CO₂e per cut")

box(CX_R, Y4, BW, BH, fc=C_RS + "28", ec=C_RS, lw=1.5)
label(CX_R, Y4, "A5_saw  ·  on-site sawing",
      "cutting to required length  ·  kg CO₂e per cut")

# ---------------------------------------------------------------------------
# Merge into shared end-of-life section
# ---------------------------------------------------------------------------
# Vertical continuations from both columns down to merge bar
ax.plot([CX_L, CX_L], [Y4 - BH/2 - 0.04, MERGE_Y],
        color=C_DARK, lw=1.3, alpha=0.65, zorder=5)
ax.plot([CX_R, CX_R], [Y4 - BH/2 - 0.04, MERGE_Y],
        color=C_DARK, lw=1.3, alpha=0.65, zorder=5)
ax.plot([CX_L, CX_R], [MERGE_Y, MERGE_Y],
        color=C_DARK, lw=1.3, alpha=0.65, zorder=5)

ax.text(6.0, MERGE_Y + 0.14,
        "end-of-life  —  same for both routes",
        ha="center", va="bottom", fontsize=8.0,
        color=C_MUTED, style="italic", zorder=6,
        bbox=dict(boxstyle="round,pad=0.12", fc=BG, ec="none", alpha=0.95))

# Arrow from merge bar to shared boxes
arr(6.0, MERGE_Y, SHARED_Y + BH/2 + 0.06, col=C_DARK)

# ---------------------------------------------------------------------------
# Shared end-of-life boxes
# ---------------------------------------------------------------------------
BW_S = 3.20
for cx, tag, desc in [
    (2.10, "C1  ·  deconstruction",  "demolition / disassembly at end of service"),
    (6.00, "C2  ·  waste transport", "transport to disposal facility"),
    (9.90, "C3+C4  ·  disposal",     "waste processing and landfill"),
]:
    box(cx, SHARED_Y, BW_S, BH, fc=C_DARK + "14", ec=C_DARK, lw=1.4)
    ax.text(cx, SHARED_Y + 0.14, tag,
            ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=C_DARK, zorder=4)
    ax.text(cx, SHARED_Y - 0.19, desc,
            ha="center", va="center", fontsize=6.8,
            color=C_MUTED, style="italic", zorder=4)

# ---------------------------------------------------------------------------
# Total cost box
# ---------------------------------------------------------------------------
arr(6.0, SHARED_Y - BH/2 - 0.04, TOTAL_Y + 0.44, col=C_DARK)

box(6.0, TOTAL_Y, 7.2, 0.88, fc=C_DARK + "22", ec=C_DARK, lw=2.0)
ax.text(6.0, TOTAL_Y + 0.20,
        r"$C_{ij}$  —  total assignment cost",
        ha="center", va="center", fontsize=11.0,
        fontweight="bold", color=C_DARK, zorder=4)
ax.text(6.0, TOTAL_Y - 0.18,
        "kg CO₂e per slot–element pair  ·  entry in MILP cost matrix",
        ha="center", va="center", fontsize=7.5,
        color=C_MUTED, style="italic", zorder=4)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
stem = "fig_cost_matrix_branches"
for fmt, out_dir in [("pdf", FIG_PDF_DIR), ("png", FIG_PNG_DIR)]:
    out = out_dir / f"{stem}.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close(fig)
