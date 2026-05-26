"""
fig_pareto_curriculum.py — Pareto front + curriculum schedule, section 2.5.7.

Left:  two-objective Pareto front (CO₂e vs reuse fraction) with convex hull
       chord and shaded non-convex gap.
Right: structural penalty weight curriculum ramp over generations.
No caption embedded — provided separately.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
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
BG      = "#FFFFFF"

# ---------------------------------------------------------------------------
# Pareto front
# Trade-off: higher reuse fraction requires fitting suboptimal RS elements,
# increasing total material and transport CO₂e beyond an efficient threshold.
# ---------------------------------------------------------------------------
t = np.linspace(0, 1, 500)
co2e  = 25 + 65 * t
reuse = 0.12 + 0.76 * t - 0.19 * np.exp(-30 * (t - 0.52) ** 2)

# Identify the non-convex gap: where the front lies below the chord
chord_full = 0.12 + 0.76 * t
dip = chord_full - reuse
nc_mask = dip > 0.008

nc_idx   = np.where(nc_mask)[0]
t_left   = t[nc_idx[0]]
t_right  = t[nc_idx[-1]]

co2e_left  = 25 + 65 * t_left
co2e_right = 25 + 65 * t_right
reuse_left  = float(reuse[nc_idx[0]])
reuse_right = float(reuse[nc_idx[-1]])

nc_co2e  = co2e[nc_mask]
nc_front = reuse[nc_mask]
nc_chord = reuse_left + (reuse_right - reuse_left) * (nc_co2e - co2e_left) / (co2e_right - co2e_left)

# ---------------------------------------------------------------------------
# Curriculum schedule — linear ramp to w_max, then plateau
# ---------------------------------------------------------------------------
G_MAX    = 600
ramp_end = int(0.72 * G_MAX)
gens = np.linspace(0, G_MAX, 500)
w_s  = np.where(gens <= ramp_end, gens / ramp_end, 1.0)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 12.0, 5.0
fig, (ax_pf, ax_cs) = plt.subplots(
    1, 2, figsize=(FIG_W, FIG_H),
    gridspec_kw=dict(width_ratios=[1.1, 0.9], wspace=0.34),
)
fig.patch.set_facecolor(BG)


def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(labelsize=8, labelcolor="#555555", color=C_MUTED)
    for sp in ax.spines.values():
        sp.set_edgecolor("#CCCCCC")
        sp.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Left — Pareto front
# ---------------------------------------------------------------------------
style_ax(ax_pf)
ax_pf.set_xlim(18, 98)
ax_pf.set_ylim(0.02, 1.06)
ax_pf.set_xlabel("Embodied carbon  (kg CO₂e)", fontsize=9.5,
                  color=C_DARK, labelpad=6)
ax_pf.set_ylabel("Reuse fraction  (–)", fontsize=9.5,
                  color=C_DARK, labelpad=6)

# Non-convex gap shading
ax_pf.fill_between(nc_co2e, nc_front, nc_chord,
                    alpha=0.25, color=C_RS, zorder=2, linewidth=0)

# Convex hull chord (dashed)
ax_pf.plot([co2e_left, co2e_right], [reuse_left, reuse_right],
            color=C_RS, lw=1.5, ls="--", zorder=3)

# Pareto front curve
ax_pf.plot(co2e, reuse, color=C_NS, lw=2.5, zorder=4)

# Accessible solution dots on convex segments
for ti in np.linspace(0.03, t_left - 0.05, 5):
    c = 25 + 65 * ti
    r = 0.12 + 0.76 * ti - 0.19 * np.exp(-30 * (ti - 0.52) ** 2)
    ax_pf.scatter(c, r, s=52, color=C_NS, zorder=5,
                  edgecolors="white", linewidths=0.8)
for ti in np.linspace(t_right + 0.05, 0.97, 5):
    c = 25 + 65 * ti
    r = 0.12 + 0.76 * ti - 0.19 * np.exp(-30 * (ti - 0.52) ** 2)
    ax_pf.scatter(c, r, s=52, color=C_NS, zorder=5,
                  edgecolors="white", linewidths=0.8)

# Ideal point marker + annotation
ax_pf.scatter(25, 0.88, s=110, marker="*", color=C_DARK,
               zorder=6, edgecolors="white", linewidths=0.8)
ax_pf.annotate("ideal point\n(non-achievable)",
    xy=(25, 0.88), xytext=(34, 0.81),
    fontsize=7.5, color=C_DARK, style="italic", ha="left",
    arrowprops=dict(arrowstyle="-", color=C_DARK, lw=0.9))

# "Better" arrows — placed adjacent to the Pareto front so the relationship is clear
# Carbon arrow: near the upper-right end of the front, pointing left
t_ca = 0.90
ca_x = 25 + 65 * t_ca
ca_y = 0.12 + 0.76 * t_ca - 0.19 * np.exp(-30 * (t_ca - 0.52) ** 2)
ax_pf.annotate("", xy=(ca_x - 10, ca_y + 0.04), xytext=(ca_x, ca_y + 0.04),
    arrowprops=dict(arrowstyle="-|>", color=C_MUTED, lw=1.0, mutation_scale=9))
ax_pf.text(ca_x - 10.5, ca_y + 0.04, "better carbon",
            fontsize=7, color=C_MUTED, ha="right", va="center", style="italic")

# Reuse arrow: near the lower-left end of the front, pointing up
t_ru = 0.10
ru_x = 25 + 65 * t_ru
ru_y = 0.12 + 0.76 * t_ru - 0.19 * np.exp(-30 * (t_ru - 0.52) ** 2)
ax_pf.annotate("", xy=(ru_x - 4, ru_y + 0.10), xytext=(ru_x - 4, ru_y),
    arrowprops=dict(arrowstyle="-|>", color=C_MUTED, lw=1.0, mutation_scale=9))
ax_pf.text(ru_x - 4, ru_y + 0.11, "better reuse",
            fontsize=7, color=C_MUTED, ha="center", va="bottom", style="italic")

# Non-convex annotation
mid = len(nc_co2e) // 2
mid_x = nc_co2e[mid]
mid_y = (nc_front[mid] + nc_chord[mid]) / 2
ax_pf.annotate("non-convex gap\n— inaccessible to\nweighted sum",
    xy=(mid_x, mid_y), xytext=(mid_x + 11, mid_y - 0.14),
    fontsize=7.5, color=C_RS, style="italic", ha="left",
    arrowprops=dict(arrowstyle="-", color=C_RS, lw=0.9))

ax_pf.set_title("Pareto front  ·  two-objective trade-off",
                 fontsize=10.5, fontweight="bold", color=C_DARK, pad=8)

pf_legend = [
    Line2D([0], [0], color=C_NS, lw=2.5, label="Pareto front"),
    Line2D([0], [0], color=C_RS, lw=1.5, ls="--", label="Convex hull chord"),
    mpatches.Patch(facecolor=C_RS, alpha=0.25, edgecolor="none",
                   label="Non-convex gap"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_NS,
           markersize=7, label="Accessible solutions\n(weighted sum)"),
]
ax_pf.legend(handles=pf_legend, loc="lower right", fontsize=7.5,
              frameon=True, framealpha=0.95, edgecolor=C_MUTED)

# ---------------------------------------------------------------------------
# Right — Curriculum schedule
# ---------------------------------------------------------------------------
style_ax(ax_cs)
ax_cs.set_xlim(0, G_MAX)
ax_cs.set_ylim(-0.05, 1.22)
ax_cs.set_xlabel("Generation  $g$", fontsize=9.5, color=C_DARK, labelpad=6)
ax_cs.set_ylabel("Structural penalty weight  $w_s(g)$", fontsize=9.5,
                  color=C_DARK, labelpad=6)

ax_cs.fill_between(gens, 0, w_s, alpha=0.13, color=C_NS, zorder=1)
ax_cs.plot(gens, w_s, color=C_NS, lw=2.5, zorder=3)

ax_cs.axhline(1.0, color=C_MUTED, lw=0.8, ls="--", alpha=0.7, zorder=2)
ax_cs.text(G_MAX * 0.02, 1.05, "$w_{\\mathrm{max}}$",
            fontsize=8.5, color=C_MUTED, va="bottom")

# Vertical marker at ramp end
ax_cs.axvline(ramp_end, color=C_MUTED, lw=0.8, ls=":", alpha=0.6, zorder=2)
ax_cs.text(ramp_end + G_MAX * 0.015, 0.04, "$G$",
            fontsize=8.5, color=C_MUTED, va="bottom")

# Phase labels
ax_cs.text(ramp_end * 0.35, 0.22,
            "exploration\n(lenient)",
            ha="center", va="center", fontsize=8.5, color=C_NS,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.28", fc=BG,
                      ec=C_NS + "80", lw=0.9))
ax_cs.text(ramp_end + (G_MAX - ramp_end) * 0.50, 0.78,
            "feasibility\nenforced",
            ha="center", va="center", fontsize=8.5, color=C_DARK,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.28", fc=BG,
                      ec=C_DARK + "80", lw=0.9))

ax_cs.set_title("Curriculum schedule  ·  penalty weight ramp",
                 fontsize=10.5, fontweight="bold", color=C_DARK, pad=8)

fig.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
stem = "fig_pareto_curriculum"
for fmt, out_dir in [("pdf", FIG_PDF_DIR), ("png", FIG_PNG_DIR)]:
    out = out_dir / f"{stem}.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close(fig)
