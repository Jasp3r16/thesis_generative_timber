"""
fig_cost_matrix_branches.py — Two-branch cost formula, section 2.5.x.

Split-path diagram: C_ij branches at χ_i into new-stock and reclaimed-stock
paths. New stock incurs A1–A3 and A4 only (A5_prep and A5_saw excluded).
Reclaimed stock incurs A4, A5_prep, A5_saw (conditional), C1 recovery from
the donor structure, C2+C3+C4 offcut waste (conditional), and an optional
scarcity penalty (ω=0, currently inactive). Both branches converge to C_ij.
Excluded / zero-burden / inactive terms shown greyed with strikethrough.
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
FIG_W, FIG_H = 12.0, 13.5
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 12)
ax.set_ylim(0, 13.5)
ax.axis("off")

CX_L = 3.0      # new-stock column centre
CX_R = 9.0      # reclaimed column centre
BW   = 4.6      # component box width
BH   = 0.82     # component box height
R    = 0.07     # corner radius

# y-centres (top to bottom)
Y_FORM   = 13.10
SPLIT_Y  = 12.30
Y1       = 11.40   # A1–A3
Y2       = 10.20   # A4
Y3       =  9.00   # A5_prep
Y4       =  7.80   # A5_saw
# Reclaimed-only rows (right column only)
Y5       =  6.40   # C1: recovery from donor structure
Y6       =  5.20   # C2 + C3+C4: offcut waste
Y7       =  4.00   # ω: scarcity penalty (ω=0, inactive)
# Convergence
CONV_Y   =  3.10
TOTAL_Y  =  1.80

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
# Section backgrounds
# ---------------------------------------------------------------------------
bg_top       = Y1 + BH/2 + 0.18
bg_bot_comm  = Y4 - BH/2 - 0.18   # bottom of shared rows

# New-stock column (rows Y1–Y4)
ax.add_patch(FancyBboxPatch(
    (CX_L - BW/2 - 0.20, bg_bot_comm), BW + 0.40, bg_top - bg_bot_comm,
    boxstyle="round,pad=0.05",
    facecolor=C_NS + "0D", edgecolor=C_NS + "38", linewidth=0.8,
    zorder=1, clip_on=False))

# Reclaimed column — shared rows (Y1–Y4)
ax.add_patch(FancyBboxPatch(
    (CX_R - BW/2 - 0.20, bg_bot_comm), BW + 0.40, bg_top - bg_bot_comm,
    boxstyle="round,pad=0.05",
    facecolor=C_RS + "0D", edgecolor=C_RS + "38", linewidth=0.8,
    zorder=1, clip_on=False))

# Reclaimed column — reclaimed-only rows (Y5–Y7), dashed border
bg_bot_extra = Y7 - BH/2 - 0.18
bg_top_extra = Y5 + BH/2 + 0.18
ax.add_patch(FancyBboxPatch(
    (CX_R - BW/2 - 0.20, bg_bot_extra), BW + 0.40, bg_top_extra - bg_bot_extra,
    boxstyle="round,pad=0.05",
    facecolor=C_RS + "0D", edgecolor=C_RS + "55", linewidth=0.8,
    linestyle="--",
    zorder=1, clip_on=False))

# "reclaimed-only terms" divider label
ax.text(CX_R, (Y4 + Y5) / 2,
        "reclaimed-only terms  ↓",
        ha="center", va="center", fontsize=7.5,
        color=C_RS, style="italic", alpha=0.80, zorder=6)

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
      "tkm × emission factor  ·  required mass only  (cut-to-size)")

box(CX_R, Y2, BW, BH, fc=C_RS + "28", ec=C_RS, lw=1.5)
label(CX_R, Y2, "A4  ·  transport  (donor → site)",
      "tkm × emission factor  ·  full stock element mass shipped")

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
      "inspection, cleaning, surface treatment  (Bergman 2010)")

# ---------------------------------------------------------------------------
# Row 4 — A5_saw
# New stock: excluded (bought cut-to-size, no on-site sawing charged)
# Reclaimed: conditional — only when L_stock > L_req
# ---------------------------------------------------------------------------
arr(CX_L, Y3 - BH/2 - 0.04, Y4 + BH/2 + 0.04, col=C_NS)
arr(CX_R, Y3 - BH/2 - 0.04, Y4 + BH/2 + 0.04, col=C_RS)

box(CX_L, Y4, BW, BH, fc=C_EXCL, ec=C_MUTED, lw=0.9)
label(CX_L, Y4, "A5_saw  ·  on-site sawing",
      "new stock ordered cut-to-size — not applicable",
      excluded=True)

box(CX_R, Y4, BW, BH, fc=C_RS + "28", ec=C_RS, lw=1.5)
label(CX_R, Y4, "A5_saw  ·  on-site sawing",
      "conditional: only when L_stock > L_req  ·  kg CO₂e per cut")

# ---------------------------------------------------------------------------
# Reclaimed-only rows — right column only
# Left column: dashed pass-through connector (no cost terms added)
# ---------------------------------------------------------------------------

# Left-side dashed pass-through line (no boxes)
ax.plot([CX_L, CX_L],
        [Y4 - BH/2 - 0.04, CONV_Y],
        color=C_NS, lw=1.1, ls="--", alpha=0.45, zorder=2)

# Y5 — C1: recovery burden from donor structure
arr(CX_R, Y4 - BH/2 - 0.04, Y5 + BH/2 + 0.04, col=C_RS)
box(CX_R, Y5, BW, BH, fc=C_RS + "28", ec=C_RS, lw=1.5)
label(CX_R, Y5, "C1  ·  recovery from donor structure",
      "deconstruction of prior-use building  (Bergman 2010, kg CO₂e/kg)")

# Y6 — C2 + C3+C4: offcut waste (transport + disposal)
arr(CX_R, Y5 - BH/2 - 0.04, Y6 + BH/2 + 0.04, col=C_RS)
box(CX_R, Y6, BW, BH, fc=C_RS + "28", ec=C_RS, lw=1.5)
label(CX_R, Y6, "C2 + C3+C4  ·  offcut waste",
      "transport + disposal of cut-off  ·  conditional: L_stock > L_req")

# Y7 — ω: scarcity penalty (currently ω=0, shown inactive)
arr(CX_R, Y6 - BH/2 - 0.04, Y7 + BH/2 + 0.04, col=C_RS)
box(CX_R, Y7, BW, BH, fc=C_EXCL, ec=C_MUTED, lw=0.9)
label(CX_R, Y7, "ω  ·  scarcity penalty on cut-off volume",
      "ω = 0  ·  inactive  ·  penalises waste of scarce reclaimed stock",
      excluded=True)

# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------
ax.plot([CX_R, CX_R], [Y7 - BH/2 - 0.04, CONV_Y],
        color=C_RS, lw=1.3, alpha=0.65, zorder=5)
ax.plot([CX_L, CX_R], [CONV_Y, CONV_Y],
        color=C_DARK, lw=1.3, alpha=0.65, zorder=5)

ax.text(6.0, CONV_Y + 0.14,
        "cost per slot–element pair  (both routes)",
        ha="center", va="bottom", fontsize=8.0,
        color=C_MUTED, style="italic", zorder=6,
        bbox=dict(boxstyle="round,pad=0.12", fc=BG, ec="none", alpha=0.95))

arr(6.0, CONV_Y, TOTAL_Y + 0.44, col=C_DARK)

# ---------------------------------------------------------------------------
# Total cost box
# ---------------------------------------------------------------------------
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
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.98)

stem = "fig_cost_matrix_branches"
for fmt, out_dir in [("pdf", FIG_PDF_DIR), ("png", FIG_PNG_DIR)]:
    out = out_dir / f"{stem}.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close(fig)
