"""
fig_lca_modules.py — EN 15804 lifecycle module comparison for chapter 2.1.5.

Two rows: new timber (full A1-A3 burden) vs reclaimed timber (zero-burden A1-A3,
costs shifted to reprocessing in A4-A5). Qualitative / schematic — no numeric axis.
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

C_NS     = "#61788C"
C_RS     = "#F2994B"
C_DARK   = "#2F3E4F"
C_ACCENT = "#D9653B"
C_MUTED  = "#9CA5A6"
C_LIGHT  = "#D7D9D9"
C_BIO    = "#A8B89A"
BG       = "#FFFFFF"

fig, ax = plt.subplots(figsize=(12, 5.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor(BG)

# ---------------------------------------------------------------------------
# Module definitions
# Width proportions (qualitative, not to quantitative scale)
# ---------------------------------------------------------------------------
modules = [
    ("A1–A3",  "Product Stage",       0.265),
    ("A4–A5",  "Construction Stage",  0.130),
    ("B1–B7",  "Use Stage",           0.195),
    ("C1–C4",  "End of Life",         0.240),
    ("D",      "Beyond\nLifecycle",   0.095),
]

GAP     = 0.011
X_START = 0.12
BAR_H   = 0.13

# Scale widths to fit within [X_START, 0.97]
raw_total = sum(m[2] for m in modules) + GAP * (len(modules) - 1)
scale     = (0.97 - X_START) / raw_total

x = X_START
positions = []
for code, stage, w in modules:
    positions.append((x, w * scale))
    x += w * scale + GAP

# Row y-centres
Y_NEW = 0.70
Y_REC = 0.41

# ---------------------------------------------------------------------------
# Colours per module per row
# ---------------------------------------------------------------------------
new_fc = {
    "A1–A3":  C_NS,
    "A4–A5":  "#8FAFC2",   # lighter blue
    "B1–B7":  C_LIGHT,
    "C1–C4":  C_MUTED,
    "D":      C_BIO,
}
rec_fc = {
    "A1–A3":  None,        # zero burden — drawn as outline only
    "A4–A5":  C_RS,        # reprocessing / transport is the main cost
    "B1–B7":  C_LIGHT,
    "C1–C4":  C_MUTED,
    "D":      C_BIO,
}
# Text colour: white on saturated/dark backgrounds, dark on light ones
_DARK_BG = {C_NS, C_RS, C_DARK}

def _tc(fc):
    return "white" if fc in _DARK_BG else C_DARK

new_tc = {k: _tc(v) for k, v in new_fc.items()}
rec_tc = {k: _tc(v) if v is not None else C_MUTED for k, v in rec_fc.items()}

# Content labels inside bars
new_label = {
    "A1–A3": "forestry · mfg.",
    "A4–A5": "transport",
    "B1–B7": "maintenance",
    "C1–C4": "deconstruction",
    "D":     "reuse credit",
}
rec_label = {
    "A1–A3": "",
    "A4–A5": "reprocessing\n+ transport",
    "B1–B7": "maintenance",
    "C1–C4": "deconstruction",
    "D":     "reuse credit",
}

# ---------------------------------------------------------------------------
# Row labels (left margin)
# ---------------------------------------------------------------------------
ax.text(0.02, Y_NEW, "New\ntimber",
        ha="left", va="center", fontsize=9, fontweight="bold",
        color=C_NS, transform=ax.transAxes)
ax.text(0.02, Y_REC, "Reclaimed\ntimber",
        ha="left", va="center", fontsize=9, fontweight="bold",
        color=C_RS, transform=ax.transAxes)

# ---------------------------------------------------------------------------
# Draw bars
# ---------------------------------------------------------------------------
for (x_pos, w), (code, stage, _) in zip(positions, modules):

    # --- new timber bar ---
    fc = new_fc[code]
    rect = mpatches.FancyBboxPatch(
        (x_pos, Y_NEW - BAR_H / 2), w, BAR_H,
        boxstyle="round,pad=0.004",
        facecolor=fc, edgecolor="white", linewidth=1.2,
        alpha=0.93, zorder=3, transform=ax.transAxes,
    )
    ax.add_patch(rect)
    if new_label[code]:
        ax.text(x_pos + w / 2, Y_NEW, new_label[code],
                ha="center", va="center", fontsize=7,
                color=new_tc[code], transform=ax.transAxes, zorder=4)

    # --- reclaimed bar ---
    fc_r = rec_fc[code]
    if fc_r is None:
        # Zero-burden: white box with dashed border
        rect_r = mpatches.FancyBboxPatch(
            (x_pos, Y_REC - BAR_H / 2), w, BAR_H,
            boxstyle="round,pad=0.004",
            facecolor=BG, edgecolor=C_MUTED, linewidth=1.2,
            linestyle="--", alpha=0.9, zorder=3, transform=ax.transAxes,
        )
        ax.add_patch(rect_r)
        ax.text(x_pos + w / 2, Y_REC, "zero burden",
                ha="center", va="center", fontsize=7,
                color=C_MUTED, style="italic",
                transform=ax.transAxes, zorder=4)
    else:
        rect_r = mpatches.FancyBboxPatch(
            (x_pos, Y_REC - BAR_H / 2), w, BAR_H,
            boxstyle="round,pad=0.004",
            facecolor=fc_r, edgecolor="white", linewidth=1.2,
            alpha=0.93, zorder=3, transform=ax.transAxes,
        )
        ax.add_patch(rect_r)
        if rec_label[code]:
            ax.text(x_pos + w / 2, Y_REC, rec_label[code],
                    ha="center", va="center", fontsize=7,
                    color=rec_tc[code], transform=ax.transAxes, zorder=4)

    # --- module code above new bar ---
    ax.text(x_pos + w / 2, Y_NEW + BAR_H / 2 + 0.035, code,
            ha="center", va="bottom", fontsize=9.5,
            fontweight="bold", color=C_DARK, transform=ax.transAxes)

    # --- stage name below reclaimed bar ---
    ax.text(x_pos + w / 2, Y_REC - BAR_H / 2 - 0.03, stage,
            ha="center", va="top", fontsize=7.5,
            color=C_MUTED, transform=ax.transAxes)

# ---------------------------------------------------------------------------
# Zero-burden callout annotation
# ---------------------------------------------------------------------------
a1_x, a1_w = positions[0]
ax.annotate("",
    xy=(a1_x + a1_w / 2, Y_REC + BAR_H / 2 + 0.005),
    xytext=(a1_x + a1_w / 2, Y_NEW - BAR_H / 2 - 0.005),
    xycoords="axes fraction", textcoords="axes fraction",
    arrowprops=dict(arrowstyle="<->", color=C_ACCENT, lw=1.6))

ax.text(a1_x + a1_w + 0.012, (Y_NEW + Y_REC) / 2,
        "zero-burden\nassumption\nfor reclaimed",
        ha="left", va="center", fontsize=8,
        color=C_ACCENT, style="italic", transform=ax.transAxes)

# ---------------------------------------------------------------------------
# Lifecycle direction arrow at bottom
# ---------------------------------------------------------------------------
ax.annotate("",
    xy=(0.97, 0.20), xytext=(X_START, 0.20),
    xycoords="axes fraction", textcoords="axes fraction",
    arrowprops=dict(arrowstyle="-|>", color=C_LIGHT, lw=1.2, mutation_scale=12))
ax.text((X_START + 0.97) / 2, 0.18, "lifecycle direction →",
        ha="center", va="top", fontsize=7.5,
        color=C_MUTED, style="italic", transform=ax.transAxes)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
legend_items = [
    mpatches.Patch(facecolor=C_NS,    label="Full production burden (new timber)"),
    mpatches.Patch(facecolor=C_RS,    label="Reprocessing / transport (reclaimed)"),
    mpatches.Patch(facecolor=C_LIGHT, edgecolor=C_MUTED, label="Use-phase (both)"),
    mpatches.Patch(facecolor=C_MUTED, label="End-of-life processing"),
    mpatches.Patch(facecolor=C_BIO,   label="Beyond lifecycle (D)"),
    mpatches.Patch(facecolor=BG, edgecolor=C_MUTED,
                   linestyle="--", label="Zero-burden assumption"),
]
leg = ax.legend(handles=legend_items, loc="upper right",
                bbox_to_anchor=(0.99, 0.99),
                ncol=2, fontsize=7.5, frameon=True, framealpha=0.95,
                edgecolor=C_MUTED)
leg.get_frame().set_linewidth(0.6)

# ---------------------------------------------------------------------------
# Caption
# ---------------------------------------------------------------------------
plt.tight_layout(pad=0.8)

out_dirs = {"pdf": FIG_PDF_DIR, "png": FIG_PNG_DIR}
for fmt in ["pdf", "png"]:
    out = out_dirs[fmt] / f"fig_lca_modules.{fmt}"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close()
