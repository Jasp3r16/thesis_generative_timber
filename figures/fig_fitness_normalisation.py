"""
Fitness normalisation and band figures (ch 4.7.5):
  fig_fitness_scatter   – raw (C, R) population scatter with C_max / R_max bounds
  fig_fitness_bands     – horizontal F band diagram for a fixed weight config
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import FIG_PDF_DIR, FIG_PNG_DIR, PLOT_COLORS

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica Neue", "DejaVu Sans"],
    "font.size":         9,
    "mathtext.fontset":  "dejavusans",
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

C_PRIMARY  = PLOT_COLORS["primary"]                       # #61788C
C_DANGER   = PLOT_COLORS["danger"]                        # #D9653B
C_ACCENT   = PLOT_COLORS["accent"]                        # #F2994B
C_NEUTRAL  = PLOT_COLORS["neutral"]                       # #D7D9D9
C_DARK     = PLOT_COLORS["extra_colors"]["deep_navy"]     # #2F3E4F
C_SAGE     = PLOT_COLORS["extra_colors"]["soft_sage_green"]  # #A8B89A
C_TEAL     = PLOT_COLORS["extra_colors"]["muted_teal"]    # #4F8A8B

# ── data ─────────────────────────────────────────────────────────────────────
GA_ROOT = Path(
    r"c:\Users\jaspe\OneDrive\06 Building Technology TU\2.2 - 2.4\60_Research_Exports\03_ga_data"
)
frames = []
hist_files = sorted(GA_ROOT.rglob("GA_A_*_history.csv"))
histories  = [pd.read_csv(f) for f in hist_files]

cfg_path = next(GA_ROOT.rglob("GA_A_*_run_config.json"))
with open(cfg_path) as fh:
    cfg = json.load(fh)

C_MAX = cfg["normalization_constants"]["C_max"]   # 366.06 kg CO₂e
R_MAX = cfg["normalization_constants"]["R_max"]   # 0.6032

# ── FIGURE 1: population trajectory plot ─────────────────────────────────────
fig1, ax = plt.subplots(figsize=(5.5, 4.2))

cmap = matplotlib.colormaps["Blues"]

for h in histories:
    gen    = h["generation"].values
    c_vals = h["mean_cost"].values
    r_vals = h["mean_reuse"].values
    ax.scatter(c_vals, r_vals,
               c=gen, cmap=cmap, vmin=0, vmax=250,
               s=4, alpha=0.4, linewidths=0)

# reference lines
ax.axvline(C_MAX, color=C_DANGER, lw=1.2, ls="--", zorder=4)
ax.axhline(R_MAX, color=C_ACCENT, lw=1.2, ls="--", zorder=4)

ax.text(C_MAX - 3, 0.505,
        f"$C_{{\\rm max}}={C_MAX:.0f}$", ha="right", va="bottom",
        fontsize=8, color=C_DANGER)
ax.text(100, R_MAX + 0.006,
        f"$R_{{\\rm max}}={R_MAX:.2f}$", ha="left", va="bottom",
        fontsize=8, color=C_ACCENT)

# generation colourbar
sm = plt.cm.ScalarMappable(cmap=cmap,
                            norm=matplotlib.colors.Normalize(vmin=0, vmax=250))
sm.set_array([])
cbar = fig1.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
cbar.set_label("generation", fontsize=8)
cbar.ax.yaxis.set_major_locator(mticker.MultipleLocator(50))

ax.set_xlabel("population mean  $C$  (kg CO$_2$e)", labelpad=6)
ax.set_ylabel("population mean  $R$")

fig1.tight_layout()
fig1.savefig(FIG_PDF_DIR / "fig_fitness_scatter.pdf", bbox_inches="tight")
fig1.savefig(FIG_PNG_DIR / "fig_fitness_scatter.png", bbox_inches="tight", dpi=300)
print("saved fig_fitness_scatter")

# ── FIGURE 2: fitness band diagram ───────────────────────────────────────────
W1, W2, W4 = 1.0, 0.5, 0.3          # example weights from text
F_MIN =  -W2                          # -0.5  (zero cost, full reuse, safe)
F_MAX =   W1 + W4                     # +1.3  (max cost, zero reuse, fully unsafe)
F_RNG =   F_MAX - F_MIN               # 1.8
Q     =   F_RNG / 4                   # quartile width = 0.45

bands = [
    (F_MIN,         F_MIN + Q,     C_TEAL,    "EXCELLENT"),
    (F_MIN + Q,     F_MIN + 2*Q,   C_SAGE,    "GOOD"),
    (F_MIN + 2*Q,   F_MIN + 3*Q,   C_ACCENT,  "FAIR"),
    (F_MIN + 3*Q,   F_MAX,         C_DANGER,  "POOR"),
]

fig2, ax2 = plt.subplots(figsize=(7.5, 1.8))
ax2.set_xlim(F_MIN - 0.08, F_MAX + 0.08)
ax2.set_ylim(0, 1)
ax2.axis("off")

BAR_Y, BAR_H = 0.38, 0.36

for lo, hi, color, label in bands:
    ax2.add_patch(mpatches.Rectangle(
        (lo, BAR_Y), hi - lo, BAR_H,
        facecolor=color, edgecolor="white", linewidth=1.0, alpha=0.80,
    ))
    cx = (lo + hi) / 2
    ax2.text(cx, BAR_Y + BAR_H / 2, label,
             ha="center", va="center", fontsize=8,
             fontweight="bold", color="white")

# axis tick marks at band boundaries
boundaries = [F_MIN, F_MIN+Q, F_MIN+2*Q, F_MIN+3*Q, F_MAX]
labels_b   = [f"{v:.2f}" for v in boundaries]
for v, lbl in zip(boundaries, labels_b):
    ax2.plot([v, v], [BAR_Y - 0.06, BAR_Y], color=C_DARK, lw=0.8)
    ax2.text(v, BAR_Y - 0.10, lbl, ha="center", va="top", fontsize=7.5,
             color=C_DARK)

# F-axis label
ax2.text((F_MIN + F_MAX) / 2, BAR_Y - 0.32, "$F$",
         ha="center", va="top", fontsize=9, color=C_DARK)

# Endpoint annotations
ax2.text(F_MIN - 0.01, BAR_Y + BAR_H + 0.08,
         f"$-\\omega_2 = {F_MIN:.1f}$\n(best)", ha="center", va="bottom",
         fontsize=7.5, color=C_DARK)
ax2.text(F_MAX + 0.01, BAR_Y + BAR_H + 0.08,
         f"$\\omega_1+\\omega_4 = {F_MAX:.1f}$\n(worst)", ha="center", va="bottom",
         fontsize=7.5, color=C_DARK)

# Weight config note
ax2.text((F_MIN + F_MAX) / 2, 0.97,
         f"$\\omega_1={W1},\\;\\omega_2={W2},\\;\\omega_4={W4}$",
         ha="center", va="top", fontsize=8, color=C_DARK, style="italic")

# Sample candidate (using real median values with example weights)
F_sample = W1 * 0.311 - W2 * 1.0 + W4 * 0.240   # ≈ -0.117
ax2.plot([F_sample, F_sample], [BAR_Y, BAR_Y + BAR_H],
         color=C_DARK, lw=1.6, zorder=5)
ax2.text(F_sample, BAR_Y + BAR_H + 0.08, f"sample\n$F={F_sample:.2f}$",
         ha="center", va="bottom", fontsize=7.5, color=C_DARK)

fig2.tight_layout(pad=0.3)
fig2.savefig(FIG_PDF_DIR / "fig_fitness_bands.pdf", bbox_inches="tight")
fig2.savefig(FIG_PNG_DIR / "fig_fitness_bands.png", bbox_inches="tight", dpi=300)
print("saved fig_fitness_bands")
