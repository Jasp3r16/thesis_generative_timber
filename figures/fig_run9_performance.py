"""
fig_run9_performance.py — Run 9 production model performance (Section 4.2.x).

Left:  ROC curve with AUC and selected threshold operating point.
Right: Precision–recall curve with selected threshold, baseline, and
       minimum-precision constraint (40%) marked.

Uses actual test predictions from Run 9:
  ID20260516_182257_LR1e-04_EP200_BS64_PW2.5_ROC0.863
Selected threshold: 0.30  (max recall s.t. precision ≥ 40%)
Metrics at threshold: recall=0.866, precision=0.417  (from metrics.json).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import FIG_PDF_DIR, FIG_PNG_DIR, EXPORT_PATH

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
C_DARK  = "#2F3E4F"
C_NS    = "#61788C"
C_RS    = "#F2994B"
C_MUTED = "#9CA5A6"
C_LIGHT = "#D7D9D9"
C_FAIL  = "#D9653B"
BG      = "#FFFFFF"

# ---------------------------------------------------------------------------
# Load Run 9 test predictions
# ---------------------------------------------------------------------------
MODEL_DIR = (EXPORT_PATH / "02_surrogate_model_data" /
             "ID20260516_182257_LR1e-04_EP200_BS64_PW2.5_ROC0.863")

probs = np.loadtxt(str(MODEL_DIR / "test_probs.csv"))
true  = np.loadtxt(str(MODEL_DIR / "test_targets.csv")).astype(int)

print(f"Test samples: {len(true):,}  |  pos_rate={true.mean():.3f}")

from sklearn.metrics import roc_curve, auc as sk_auc, precision_recall_curve

fpr, tpr, thr_roc = roc_curve(true, probs)
roc_auc = sk_auc(fpr, tpr)

prec, rec, thr_pr = precision_recall_curve(true, probs, pos_label=1)
pr_auc = sk_auc(rec, prec)
pos_rate = true.mean()

print(f"ROC-AUC={roc_auc:.3f}  PR-AUC={pr_auc:.3f}")

# Operating point: threshold = 0.30
THRESH = 0.30
MIN_PREC = 0.40

# ROC operating point
idx_roc = int(np.argmin(np.abs(thr_roc - THRESH)))
fpr_sel = float(fpr[idx_roc])
tpr_sel = float(tpr[idx_roc])

# PR operating point
idx_pr  = int(np.argmin(np.abs(thr_pr - THRESH)))
prec_sel = float(prec[idx_pr])
rec_sel  = float(rec[idx_pr])

print(f"At thr={THRESH}: ROC (FPR={fpr_sel:.3f}, TPR={tpr_sel:.3f})  "
      f"PR (P={prec_sel:.3f}, R={rec_sel:.3f})")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 11.0, 5.0
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(BG)
fig.subplots_adjust(left=0.08, right=0.97, bottom=0.13, top=0.87, wspace=0.30)


def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(labelsize=8, color=C_MUTED)
    for sp in ax.spines.values():
        sp.set_edgecolor(C_LIGHT); sp.set_linewidth(0.8)
    ax.grid(color=C_LIGHT, lw=0.5, alpha=0.6, zorder=0)


# ===========================================================================
# Panel 1 — ROC curve
# ===========================================================================
style_ax(ax1)

# Diagonal (random classifier)
ax1.plot([0, 1], [0, 1], color=C_LIGHT, lw=0.9, ls="--", zorder=1)

# ROC curve
ax1.plot(fpr, tpr, color=C_DARK, lw=1.8, zorder=3)

# Operating point
ax1.scatter([fpr_sel], [tpr_sel], s=65, color=C_RS,
            edgecolors=C_FAIL, linewidths=0.9, zorder=5)

ax1.annotate(
    f"threshold = {THRESH:.2f}\n"
    f"TPR (recall) = {tpr_sel:.3f}\n"
    f"FPR = {fpr_sel:.3f}",
    xy=(fpr_sel, tpr_sel),
    xytext=(fpr_sel + 0.20, tpr_sel - 0.18),
    fontsize=7.5, color=C_RS,
    arrowprops=dict(arrowstyle="-|>", color=C_RS, lw=0.9,
                    connectionstyle="arc3,rad=0.15"),
    bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
              edgecolor=C_RS, lw=0.6, alpha=0.95),
)

ax1.text(0.97, 0.06, f"ROC-AUC = {roc_auc:.3f}",
         ha="right", va="bottom", fontsize=9.5, fontweight="bold",
         color=C_DARK, transform=ax1.transAxes)

ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.01)
ax1.set_xlabel("False positive rate", fontsize=9.5, color=C_DARK)
ax1.set_ylabel("True positive rate  (recall)", fontsize=9.5, color=C_DARK)

# ===========================================================================
# Panel 2 — Precision–recall curve
# ===========================================================================
style_ax(ax2)
ax2.grid(axis="both", color=C_LIGHT, lw=0.5, alpha=0.6, zorder=0)

# Baseline (random classifier)
ax2.axhline(pos_rate, color=C_MUTED, lw=0.9, ls="--", alpha=0.7, zorder=1,
            label=f"Baseline  (pos. rate = {pos_rate:.3f})")

# Minimum precision constraint
ax2.axhline(MIN_PREC, color=C_RS, lw=1.0, ls="-.", alpha=0.8, zorder=1,
            label=f"Min. precision = {MIN_PREC:.0%}")

# PR curve
ax2.plot(rec, prec, color=C_DARK, lw=1.8, zorder=3)

# Operating point
ax2.scatter([rec_sel], [prec_sel], s=65, color=C_RS,
            edgecolors=C_FAIL, linewidths=0.9, zorder=5,
            label=f"Threshold = {THRESH:.2f}  (P={prec_sel:.3f}, R={rec_sel:.3f})")

ax2.annotate(
    f"threshold = {THRESH:.2f}\n"
    f"precision = {prec_sel:.3f}\n"
    f"recall      = {rec_sel:.3f}",
    xy=(rec_sel, prec_sel),
    xytext=(rec_sel - 0.28, prec_sel + 0.22),
    fontsize=7.5, color=C_RS,
    arrowprops=dict(arrowstyle="-|>", color=C_RS, lw=0.9,
                    connectionstyle="arc3,rad=-0.15"),
    bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
              edgecolor=C_RS, lw=0.6, alpha=0.95),
)

ax2.text(0.97, 0.97, f"PR-AUC = {pr_auc:.3f}",
         ha="right", va="top", fontsize=9.5, fontweight="bold",
         color=C_DARK, transform=ax2.transAxes)

ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)
ax2.set_xlabel("Recall  (unsafe class)", fontsize=9.5, color=C_DARK)
ax2.set_ylabel("Precision  (unsafe class)", fontsize=9.5, color=C_DARK)
              fontsize=10.5, fontweight="bold", color=C_DARK, pad=6)
ax2.legend(fontsize=7.5, frameon=True, framealpha=0.95,
           edgecolor=C_LIGHT, loc="upper right")

# ---------------------------------------------------------------------------
# Super-title
# ---------------------------------------------------------------------------
fig.text(0.525, 0.935,
         "Run 9  —  production model  "
         "(4 layers  ·  WeightedBCE pos_weight=2.5  ·  batch=64)",
         ha="center", va="bottom", fontsize=9.5,
         fontweight="bold", color=C_DARK)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
stem = "fig_run9_performance"
for fmt, out_dir in [("pdf", FIG_PDF_DIR), ("png", FIG_PNG_DIR)]:
    out = out_dir / f"{stem}.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close(fig)
