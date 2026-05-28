"""
GNN inference figures (ch 4.7.4):
  fig_gnn_failure_prob_hist  – histogram of per-member failure probabilities
  fig_gnn_omega_schedule     – ω₄ linear ramp + mean feasibility score
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
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

# ── colours ───────────────────────────────────────────────────────────────────
C_PRIMARY  = PLOT_COLORS["primary"]    # #61788C – safe bars, feasibility line
C_DANGER   = PLOT_COLORS["danger"]     # #D9653B – unsafe bars, ω₄ line
C_ACCENT   = PLOT_COLORS["accent"]     # #F2994B – threshold line
C_NEUTRAL  = PLOT_COLORS["neutral"]    # #D7D9D9 – annotation box border
C_DARK     = PLOT_COLORS["extra_colors"]["deep_navy"]   # #2F3E4F – axis labels

# ── data paths ────────────────────────────────────────────────────────────────
DATA_ROOT = Path(
    r"c:\Users\jaspe\OneDrive\06 Building Technology TU\2.2 - 2.4\60_Research_Exports"
)
PROBS_CSV   = DATA_ROOT / "02_surrogate_model_data" / \
    "ID20260512_193129_LR3e-04_EP150_BS32_FA0.50_ROC0.894" / "test_probs.csv"
HISTORY_CSV = DATA_ROOT / "03_ga_data" / \
    "GA_FINAL_BATCH_3PerStock_20260526_GEN250_EVAL_7500" / \
    "GA_A_20260520_214255_RUN1_GEN250_EVAL7500_F-0_4778" / \
    "GA_A_20260520_214255_RUN1_GEN250_EVAL7500_F-0_4778_history.csv"

THETA = 0.35

# ── load data ─────────────────────────────────────────────────────────────────
_raw    = pd.read_csv(PROBS_CSV, header=None).iloc[:, 0].values
p       = _raw[0:120]           # 120 per-member probabilities, first design
hist_df = pd.read_csv(HISTORY_CSV)

# ── FIGURE 1: failure-probability histogram ───────────────────────────────────
fig1, ax = plt.subplots(figsize=(6.5, 3.2))

bins           = np.linspace(0, 1, 21)
counts, edges  = np.histogram(p, bins=bins)

for cnt, left, right in zip(counts, edges[:-1], edges[1:]):
    mid    = (left + right) / 2
    unsafe = mid >= THETA
    ax.bar(mid, cnt, width=(right - left) * 0.88,
           color=C_ACCENT  if unsafe else C_PRIMARY,
           edgecolor=C_DARK, linewidth=0.5, alpha=0.45 if unsafe else 0.45)

ax.axvline(THETA, color=C_DANGER, lw=1.4, linestyle="--", zorder=5)
ax.text(THETA + 0.015, 0.96, f"$\\theta = {THETA}$",
        transform=ax.get_xaxis_transform(), va="top",
        fontsize=8.5, color=C_DANGER)

f_score  = 1 - p.mean()
n_unsafe = int((p >= THETA).sum())
stats_txt = (f"$\\bar{{p}} = {p.mean():.3f}$\n"
             f"$f = {f_score:.3f}$\n"
             f"unsafe: {n_unsafe}/120")
ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
        ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_NEUTRAL, lw=0.7))

ax.set_xlabel("predicted failure probability  $p_i$", labelpad=6)
ax.set_ylabel("member count")
ax.set_xlim(0, 1)
ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

fig1.tight_layout()
fig1.savefig(FIG_PDF_DIR / "fig_gnn_failure_prob_hist.pdf", bbox_inches="tight")
fig1.savefig(FIG_PNG_DIR / "fig_gnn_failure_prob_hist.png", bbox_inches="tight", dpi=300)
print("saved fig_gnn_failure_prob_hist")

# ── FIGURE 2: ω₄ schedule + mean feasibility ─────────────────────────────────
fig2, ax1 = plt.subplots(figsize=(6.5, 3.2))
ax2 = ax1.twinx()

gen = hist_df["generation"].values
w   = hist_df["w_structural"].values
gnn = hist_df["mean_gnn"].values

l1, = ax1.plot(gen, w,   color=C_DANGER, lw=1.5,              label="$\\omega_4$")
l2, = ax2.plot(gen, gnn, color=C_PRIMARY,  lw=1.5, ls="--",
               label="mean feasibility $f$")

ax1.annotate(f"$\\omega_4 = {w[0]:.1f}$",
             xy=(gen[0], w[0]), xytext=(18, 4),
             textcoords="offset points", fontsize=8, color=C_DANGER)
ax1.annotate(f"$\\omega_4 = {w[-1]:.1f}$",
             xy=(gen[-1], w[-1]), xytext=(-60, -4),
             textcoords="offset points", fontsize=8, color=C_DANGER)

ax1.set_xlabel("generation")
ax1.set_ylabel("$\\omega_4$  (structural weight)", color=C_DANGER)
ax2.set_ylabel("mean feasibility score  $f$",      color=C_PRIMARY)

ax1.tick_params(axis="y", colors=C_DANGER)
ax2.tick_params(axis="y", colors=C_PRIMARY)
ax2.spines["right"].set_visible(True)
ax2.spines["top"].set_visible(False)
ax1.spines["top"].set_visible(False)

ax1.set_xlim(1, 250)
ax1.set_ylim(0.6, 2.2)
ax2.set_ylim(0.45, 0.85)

lines  = [l1, l2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="center right", fontsize=8.5,
           frameon=True, framealpha=0.9, edgecolor=C_NEUTRAL)

fig2.tight_layout()
fig2.savefig(FIG_PDF_DIR / "fig_gnn_omega_schedule.pdf", bbox_inches="tight")
fig2.savefig(FIG_PNG_DIR / "fig_gnn_omega_schedule.png", bbox_inches="tight", dpi=300)
print("saved fig_gnn_omega_schedule")
