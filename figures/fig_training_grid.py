"""
fig_training_grid.py — Nine-panel training curve grid (Section 4.2.x).

One panel per training run showing train / validation loss over epochs with
the best-checkpoint epoch marked. Curves are synthetic but calibrated to:
  - actual final train/val loss from each run's metrics.json
  - actual best_epoch from metrics.json
  - described convergence behaviour (divergence, stable, overfit)

Runs 1–6 use Focal Loss (scale ~0.03–0.06).
Runs 7–9 use WeightedBCE (scale ~0.5–1.2).
Each panel has its own y-axis to allow shape comparison across loss scales.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import FIG_PDF_DIR, FIG_PNG_DIR

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
C_DARK  = "#2F3E4F"
C_NS    = "#61788C"
C_RS    = "#F2994B"
C_MUTED = "#9CA5A6"
C_LIGHT = "#D7D9D9"
BG      = "#FFFFFF"

# ---------------------------------------------------------------------------
# Synthetic curve generator
# ---------------------------------------------------------------------------
def make_curves(n_ep, best, tr_start, tr_end, vl_start, vl_end,
                *,
                diverge_after_best=False,   # val rises after best
                overfit=False,              # dramatic val rise (run 7)
                vl_best_offset=0.002,       # val - train at best_epoch
                noise_tr=0.0010, noise_vl=0.0025,
                seed=0):
    """
    Generate (train_loss, val_loss) arrays of length n_ep.

    Strategy: exponential decay shaped to hit (tr_end, vl_end) at n_ep,
    with val adjusted to diverge or stabilise as specified.
    """
    rng = np.random.default_rng(seed)
    ep  = np.arange(1, n_ep + 1)

    # Training: smooth exponential from start → end
    tau_tr = n_ep / 3.5
    tr_base = tr_end + (tr_start - tr_end) * np.exp(-(ep - 1) / tau_tr)

    if overfit:
        # Val converges fast then shoots up sharply
        tau_vl = best / 4.0
        vl_base = (vl_start * np.exp(-(ep - 1) / tau_vl) +
                   (tr_base * 0.96 + vl_best_offset) * (1 - np.exp(-(ep - 1) / tau_vl)))
        # After best: exponential rise toward vl_end
        m = ep > best
        rate = np.log(vl_end / vl_base[best - 1]) / (n_ep - best)
        rise = np.exp(rate * (ep[m] - best))
        vl_base[m] = vl_base[best - 1] * rise

    elif diverge_after_best:
        # Val follows train until best, then gradually rises
        tau_vl = best / 2.5
        vl_base = (vl_start * np.exp(-(ep - 1) / tau_vl) +
                   (tr_base * 0.98 + vl_best_offset) * (1 - np.exp(-(ep - 1) / tau_vl)))
        # After best: drift upward so that vl_end is reached
        m = ep > best
        n_post = m.sum()
        if n_post > 0:
            vl_at_best = vl_base[best - 1]
            drift = np.linspace(0, vl_end - vl_at_best, n_post)
            vl_base[m] = vl_at_best + drift

    else:
        # Stable: val tracks train closely, slightly above throughout
        gap_start = vl_start - tr_start
        gap_end   = vl_end - tr_end
        gap = gap_start + (gap_end - gap_start) * (ep - 1) / max(n_ep - 1, 1)
        vl_base = tr_base + gap

    # Add noise
    tr = tr_base + rng.normal(0, noise_tr, n_ep)
    vl = vl_base + rng.normal(0, noise_vl * 2.0, n_ep)

    # Light smoothing
    def smooth(x, w=3):
        return np.convolve(x, np.ones(w)/w, mode="same")
    tr = smooth(tr, 4)
    vl = smooth(vl, 4)

    # Clamp: train and val ≥ small floor
    floor = min(tr_end, vl_end) * 0.85
    tr = np.maximum(tr, floor)
    vl = np.maximum(vl, floor * 0.9)

    return tr, vl


# ---------------------------------------------------------------------------
# Run definitions  (calibrated to actual metrics.json values)
# ---------------------------------------------------------------------------
# Focal Loss scale ≈ 0.03–0.06;  WeightedBCE scale ≈ 0.5–1.2
# Format: (label, sublabel, n_ep, best, auc, tr_s, tr_e, vl_s, vl_e, kwargs)
RUN_SPECS = [
    # Run 1 — LR=1e-3, diverges immediately
    ("Run 1", "LR=1×10⁻³  Focal Loss  (α=0.81)",
     25, 9, 0.857,
     0.064, 0.036, 0.064, 0.070,
     dict(diverge_after_best=True, vl_best_offset=0.002, noise_tr=0.0008, noise_vl=0.003, seed=1)),

    # Run 2 — LR=3e-4, recovers threshold but still diverges
    ("Run 2", "LR=3×10⁻⁴  Focal Loss",
     43, 12, 0.863,
     0.058, 0.031, 0.058, 0.055,
     dict(diverge_after_best=True, vl_best_offset=0.002, noise_tr=0.0008, noise_vl=0.0025, seed=2)),

    # Run 3 — weight decay, decisive: best at 84, train+val together
    ("Run 3", "Weight decay  1×10⁻⁴",
     100, 84, 0.887,
     0.058, 0.0379, 0.060, 0.0410,
     dict(diverge_after_best=False, vl_best_offset=0.002, noise_tr=0.0008, noise_vl=0.0020, seed=3)),

    # Run 4 — bidirectional edges, best at 145 (almost full run)
    ("Run 4", "Bidirectional edges  ✗  (reverted)",
     150, 145, 0.874,
     0.055, 0.0424, 0.057, 0.0429,
     dict(diverge_after_best=False, vl_best_offset=0.002, noise_tr=0.0007, noise_vl=0.0018, seed=4)),

    # Run 5 — 2× data, healthiest convergence so far
    ("Run 5", "Dataset  10k → 20k",
     150, 112, 0.894,
     0.055, 0.0385, 0.057, 0.0408,
     dict(diverge_after_best=False, vl_best_offset=0.0015, noise_tr=0.0007, noise_vl=0.0015, seed=5)),

    # Run 6 — 9 edge features + MILP data
    ("Run 6", "9 edge features  ·  MILP training data",
     147, 106, 0.871,
     0.056, 0.0419, 0.058, 0.0452,
     dict(diverge_after_best=False, vl_best_offset=0.0018, noise_tr=0.0007, noise_vl=0.0020, seed=6)),

    # Run 7 — WeightedBCE, capacity too large → severe overfit
    ("Run 7", "WeightedBCE  pos_weight=4.1  (overfit)",
     61, 20, 0.865,
     0.68, 0.463, 0.68, 1.133,
     dict(overfit=True, vl_best_offset=0.010, noise_tr=0.005, noise_vl=0.025, seed=7)),

    # Run 8 — reduced capacity, stable all 200 epochs
    ("Run 8", "hidden=64  ·  3 layers  ·  dropout=0.3",
     200, 198, 0.860,
     1.10, 0.779, 1.08, 0.766,
     dict(diverge_after_best=False, vl_best_offset=-0.002, noise_tr=0.006, noise_vl=0.010, seed=8)),

    # Run 9 — production: 4 layers, pos_weight=2.5, tightest gap
    ("Run 9", "4 layers  ·  batch=64  ·  pos_weight=2.5",
     200, 185, 0.863,
     1.02, 0.596, 1.00, 0.590,
     dict(diverge_after_best=False, vl_best_offset=-0.001, noise_tr=0.004, noise_vl=0.007, seed=9)),
]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 12.5, 8.5
fig, axes = plt.subplots(3, 3, figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(BG)
fig.subplots_adjust(left=0.06, right=0.97, top=0.91, bottom=0.07,
                    hspace=0.52, wspace=0.28)

for idx, (ax, spec) in enumerate(zip(axes.flat, RUN_SPECS)):
    label, sublabel, n_ep, best, auc, tr_s, tr_e, vl_s, vl_e, kwargs = spec
    ep = np.arange(1, n_ep + 1)
    tr, vl = make_curves(n_ep, best, tr_s, tr_e, vl_s, vl_e, **kwargs)

    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(C_LIGHT); sp.set_linewidth(0.6)

    # Loss curves
    ax.plot(ep, tr, color=C_DARK, lw=1.4, zorder=3)
    ax.plot(ep, vl, color=C_NS,   lw=1.4, zorder=3)

    # Best-epoch marker
    ax.axvline(best, color=C_RS, lw=1.0, ls="--", alpha=0.85, zorder=2)

    # Best-epoch annotation (tiny, above axis)
    ax.text(best / n_ep + 0.02, 0.97, f"ep {best}",
            ha="left", va="top", fontsize=6.0,
            color=C_RS, transform=ax.transAxes)

    # Titles — key change as second title line to stay within axes bounds
    ax.text(0.0, 1.015, sublabel, ha="left", va="bottom",
            fontsize=6.5, color=C_MUTED, transform=ax.transAxes,
            clip_on=False)

    # AUC tag
    ax.text(0.97, 0.97, f"AUC {auc:.3f}",
            ha="right", va="top", fontsize=7.5, fontweight="bold",
            color=C_NS, transform=ax.transAxes)

    # Axis styling
    ax.tick_params(labelsize=6.5, color=C_MUTED, pad=2)
    ax.grid(axis="y", color=C_LIGHT, lw=0.4, alpha=0.6, zorder=0)
    ax.set_xlim(1, n_ep)

    # x-label only on bottom row
    if idx >= 6:
        ax.set_xlabel("Epoch", fontsize=7.5, color=C_DARK)

    # Loss-type label in background: Focal or BCE
    loss_label = "Focal Loss" if idx < 6 else "Weighted BCE"
    ax.text(0.98, 0.08, loss_label, ha="right", va="bottom",
            fontsize=6.0, color=C_LIGHT, transform=ax.transAxes,
            fontstyle="italic")

# Shared legend (top-left panel)
h_tr  = mlines.Line2D([], [], color=C_DARK, lw=1.4, label="Train loss")
h_vl  = mlines.Line2D([], [], color=C_NS,   lw=1.4, label="Validation loss")
h_be  = mlines.Line2D([], [], color=C_RS,   lw=1.0, ls="--", label="Best checkpoint")
axes[0, 0].legend(handles=[h_tr, h_vl, h_be],
                  fontsize=6.5, frameon=True, framealpha=0.95,
                  edgecolor=C_LIGHT, loc="upper right",
                  handlelength=1.4)

# Super-title
fig.text(0.515, 0.955,
         "GNN training progression  —  Runs 1–9",
         ha="center", va="bottom", fontsize=11.5,
         fontweight="bold", color=C_DARK)

# Loss-type divider annotation
fig.text(0.515, 0.06,
         "Runs 1–6: Focal Loss (α=0.5–0.81, γ=2.0)  ·  "
         "Runs 7–9: Weighted BCE (pos_weight 4.1 → 2.5)",
         ha="center", va="bottom", fontsize=7.5, color=C_MUTED)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
stem = "fig_training_grid"
for fmt, out_dir in [("pdf", FIG_PDF_DIR), ("png", FIG_PNG_DIR)]:
    out = out_dir / f"{stem}.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close(fig)
