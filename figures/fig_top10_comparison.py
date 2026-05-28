"""
Top-10 shortlist comparison figure (ch 4.9):
  fig_top10_comparison – three-panel: fitness bars | GNN feasibility |
                          cost vs structural infeasibility scatter
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
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
    "font.size":         8.5,
    "mathtext.fontset":  "dejavusans",
    "figure.dpi":        150,
    "axes.spines.top":   False,
})

C_PRIMARY = PLOT_COLORS["primary"]
C_DANGER  = PLOT_COLORS["danger"]
C_ACCENT  = PLOT_COLORS["accent"]
C_NEUTRAL = PLOT_COLORS["neutral"]
C_DARK    = PLOT_COLORS["extra_colors"]["deep_navy"]
C_SAGE    = PLOT_COLORS["extra_colors"]["soft_sage_green"]

# ── data ─────────────────────────────────────────────────────────────────────
RUN = Path(
    r"c:\Users\jaspe\OneDrive\06 Building Technology TU\2.2 - 2.4\60_Research_Exports"
    r"\03_ga_data\GA_A_20260519_163628_GEN250_EVAL7500_F-2_4697"
)
TOP = RUN / "top_k_designs"
s   = pd.read_csv(TOP / "GA_A_20260519_163628_GEN250_EVAL7500_F-2_4697_top10_summary.csv")

with open(RUN / "GA_A_20260519_163628_GEN250_EVAL7500_F-2_4697_run_config.json") as fh:
    cfg = json.load(fh)
w4 = cfg["ga_config"]["w_structural_end"]   # 0.8

# fitness advantage over worst in shortlist (rank 1 = best = largest value)
s["f_adv"] = s["fitness"] - s["fitness"].min()   # rank 10 → 0

# colour ramp: rank 1 = C_PRIMARY (darkest), rank 10 = lightest
def rank_color(rank, n=10, c_best=C_PRIMARY, c_worst=C_NEUTRAL):
    t = (rank - 1) / (n - 1)
    r1, g1, b1 = mcolors.to_rgb(c_best)
    r2, g2, b2 = mcolors.to_rgb(c_worst)
    return (r1 + t*(r2-r1), g1 + t*(g2-g1), b1 + t*(b2-b1))

colors = [rank_color(r) for r in s["rank"]]

# y positions: rank 1 at top
y = list(range(10, 0, -1))   # [10, 9, ..., 1] maps to rank [1,2,...,10]

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8),
                          gridspec_kw={"width_ratios": [2.5, 1.8, 2.5],
                                       "wspace": 0.45})
ax1, ax2, ax3 = axes

# ── panel 1: fitness advantage bars ──────────────────────────────────────────
ax1.barh(y, s["f_adv"] * 1000, color=colors, edgecolor="white",
         linewidth=0.4, height=0.75)
ax1.set_yticks(y)
ax1.set_yticklabels([f"{r}" for r in s["rank"]], fontsize=8)
ax1.set_xlabel(r"fitness advantage over rank 10  ($\times 10^{-3}$)", labelpad=5)
ax1.set_ylabel("rank")
ax1.spines["right"].set_visible(False)
ax1.xaxis.set_major_locator(mticker.MultipleLocator(0.5))

# ── panel 2: GNN feasibility dots ────────────────────────────────────────────
ax2.scatter(s["gnn_feasibility"], y, color=colors, s=38, zorder=3)
ax2.set_yticks(y)
ax2.set_yticklabels([])
ax2.set_xlabel("GNN feasibility  $f$", labelpad=5)
ax2.spines["left"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.tick_params(left=False)
# tight x range
x_lo = s["gnn_feasibility"].min() - 0.0005
x_hi = s["gnn_feasibility"].max() + 0.0005
ax2.set_xlim(x_lo, x_hi)
ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax2.xaxis.set_major_locator(mticker.MaxNLocator(3))

# horizontal grid lines shared between panel 1 and 2
for ax in (ax1, ax2):
    for yv in y:
        ax.axhline(yv, color=C_NEUTRAL, lw=0.4, zorder=0)

# ── panel 3: cost vs structural infeasibility scatter ────────────────────────
sc = ax3.scatter(
    s["total_cost"], s["structural_infeasibility"],
    c=[r for r in s["rank"]], cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "ramp", [C_PRIMARY, C_NEUTRAL]),
    s=45, zorder=3, edgecolors=C_DARK, linewidths=0.4,
)
for _, row in s.iterrows():
    ax3.text(row.total_cost + 0.01, row.structural_infeasibility + 0.00015,
             str(int(row["rank"])), fontsize=7, color=C_DARK, va="bottom")

ax3.set_xlabel("embodied carbon  $C$  (kg CO$_2$e)", labelpad=5)
ax3.set_ylabel("structural infeasibility  $S$")
ax3.spines["right"].set_visible(False)

cbar = fig.colorbar(sc, ax=ax3, pad=0.02, fraction=0.05)
cbar.set_label("rank", fontsize=8)
cbar.set_ticks([1, 5, 10])

# label panels
for ax, lbl in zip(axes, ["(a)", "(b)", "(c)"]):
    ax.text(0.03, 0.97, lbl, transform=ax.transAxes,
            va="top", ha="left", fontsize=8.5, color=C_DARK, fontweight="bold")

fig.tight_layout()
fig.savefig(FIG_PDF_DIR / "fig_top10_comparison.pdf", bbox_inches="tight")
fig.savefig(FIG_PNG_DIR / "fig_top10_comparison.png", bbox_inches="tight", dpi=300)
print("saved fig_top10_comparison")
