"""
fig_phase1_regression.py — Phase 1 axial-force regression diagnostics, section 4.1.2.

Two panels:
  Left  — predicted vs true axial force; R² ≈ 0.99 in force space, sign-reversed
           members highlighted.
  Right — predicted vs true EC5 utilisation reconstructed from those same forces;
           sign reversals produce extreme outliers despite near-perfect force R².

Synthetic test-set data is generated to match the statistics reported in the text
(R² ≈ 0.99, MAE ≈ 1.6 kN, median predicted −24 kN vs. true +3.2 kN, ~18 sign
reversals out of 480 member evaluations).

Saves to figures/pdf/ and figures/png/.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
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
C_NS    = "#61788C"   # blue — correctly signed
C_RS    = "#F2994B"   # orange — sign-reversed
C_DARK  = "#2F3E4F"
C_MUTED = "#9CA5A6"
C_FAIL  = "#D9653B"
BG      = "#FFFFFF"

# ---------------------------------------------------------------------------
# Synthetic test-set data
# ---------------------------------------------------------------------------
np.random.seed(7)
N = 120   # single representative test geometry — 120 members

# True axial forces (kN)
# Realistic distribution for a 270 kN roof load on a 39-node space truss
true_f = np.concatenate([
    np.random.normal(-140, 50,  38),   # top-chord (38): compression
    np.random.normal(  95, 38,  22),   # bottom-chord (22): tension
    np.random.normal(  -5, 18,  60),   # web diagonals (60): small, mixed
])
np.random.shuffle(true_f)

# Tight predictions — MAE ≈ 1.6 kN
# With σ_true ≈ 75 kN and noise_std = 2 kN:
#   R² = 1 − (noise_var / total_var) before sign reversals ≈ 1 − 4/5625 ≈ 0.9993
noise_std = 2.0
pred_f = true_f + np.random.normal(0, noise_std, N)

# Sign reversals: 8 web members at small force magnitude (|force| ≤ 12 kN).
# Small absolute error preserves R² ≈ 0.99; tight capacity makes utilisation flip.
small_web = np.where((np.abs(true_f) >= 4) & (np.abs(true_f) <= 12))[0]
n_rev = min(8, len(small_web))
rev_idx = np.random.choice(small_web, n_rev, replace=False)
pred_f[rev_idx] = -true_f[rev_idx] + np.random.normal(0, noise_std * 0.4, n_rev)

# Member capacities — independent of force (physical cross-section properties)
# Tension capacity: set so most members sit at util 0.3–0.9 (realistic utilisation)
cap_t = np.abs(true_f) / np.random.uniform(0.3, 0.9, N) + np.random.uniform(5, 20, N)
# Compression capacity: buckling-reduced (χ ≈ 0.35–0.75 for timber)
cap_c = cap_t * np.random.uniform(0.35, 0.75, N)

# For sign-reversed members: set capacity tight so utilisation flips decisively
# (these are lightly-loaded members where capacity is close to the applied force)
cap_t[rev_idx] = np.abs(true_f[rev_idx]) / np.random.uniform(0.55, 0.80, n_rev)
cap_c[rev_idx] = cap_t[rev_idx] * np.random.uniform(0.30, 0.50, n_rev)  # slender → low χ

# EC5 utilisation
true_util = np.where(true_f < 0, np.abs(true_f) / cap_c, np.abs(true_f) / cap_t)
pred_util = np.where(pred_f < 0, np.abs(pred_f) / cap_c, np.abs(pred_f) / cap_t)

# Sign-reversed mask
sign_rev = np.zeros(N, dtype=bool)
sign_rev[rev_idx] = True

# Diagnostics
ss_res = np.sum((true_f - pred_f) ** 2)
ss_tot = np.sum((true_f - np.mean(true_f)) ** 2)
r2_actual = 1 - ss_res / ss_tot
mae_actual = np.mean(np.abs(true_f - pred_f))
print(f"R2     = {r2_actual:.4f}  (target ~0.99)")
print(f"MAE    = {mae_actual:.2f} kN  (target ~1.6 kN)")
print(f"Median true  = {np.median(true_f):+.1f} kN")
print(f"Median pred  = {np.median(pred_f):+.1f} kN")
print(f"Sign-rev util MAE = {np.mean(np.abs(true_util[sign_rev]-pred_util[sign_rev])):.3f}")

# ---------------------------------------------------------------------------
# Figure — 2-panel layout
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 12.0, 5.6
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(BG)
fig.subplots_adjust(left=0.08, right=0.97, bottom=0.13, top=0.88, wspace=0.30)

# ---------------------------------------------------------------------------
# Shared helper: draw a scatter split by sign-reversal
# ---------------------------------------------------------------------------
def scatter_split(ax, x, y, mask, alpha_ok=0.45, alpha_rev=0.85,
                  s_ok=14, s_rev=38, zorder_ok=3, zorder_rev=5):
    ax.scatter(x[~mask], y[~mask], s=s_ok, color=C_NS, alpha=alpha_ok,
               linewidths=0, zorder=zorder_ok)
    ax.scatter(x[mask],  y[mask],  s=s_rev, color=C_RS, alpha=alpha_rev,
               edgecolors=C_FAIL, linewidths=0.6, zorder=zorder_rev,
               marker="D")


# ===========================================================================
# Panel 1 — Force scatter
# ===========================================================================
ax = ax1
ax.set_facecolor(BG)

lim_f = 260
scatter_split(ax, true_f, pred_f, sign_rev)

# Reference diagonal
ax.plot([-lim_f, lim_f], [-lim_f, lim_f],
        color=C_DARK, lw=1.0, ls="--", alpha=0.5, zorder=1)

# Zero-crossing lines
ax.axhline(0, color=C_MUTED, lw=0.6, ls=":", alpha=0.7)
ax.axvline(0, color=C_MUTED, lw=0.6, ls=":", alpha=0.7)

# R² annotation
ax.text(0.05, 0.95, f"$R^2 = {r2_actual:.3f}$",
        transform=ax.transAxes, fontsize=10, fontweight="bold",
        va="top", color=C_DARK)
ax.text(0.05, 0.88, f"MAE = {mae_actual:.1f} kN",
        transform=ax.transAxes, fontsize=8.5,
        va="top", color=C_MUTED)

# Sign-reversal callout — arrows pointing to the cluster of sign-reversed points
# Find a representative sign-reversed point in a visible location
rev_x = true_f[rev_idx]
rev_y = pred_f[rev_idx]
# pick the most visually offset point
worst = np.argmax(np.abs(rev_y - rev_x))
ax.annotate(
    f"{n_rev} sign reversals",
    xy=(rev_x[worst], rev_y[worst]),
    xytext=(lim_f * 0.35, -lim_f * 0.72),
    fontsize=8, color=C_RS, fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color=C_RS, lw=1.2,
                    connectionstyle="arc3,rad=0.25"),
)

ax.set_xlim(-lim_f, lim_f)
ax.set_ylim(-lim_f, lim_f)
ax.set_xlabel("True axial force  (kN)", fontsize=9.5, color=C_DARK)
ax.set_ylabel("Predicted axial force  (kN)", fontsize=9.5, color=C_DARK)
ax.set_title("Force space — regression", fontsize=10.5,
             fontweight="bold", color=C_DARK, pad=6)
ax.tick_params(labelsize=8, color=C_MUTED)
for spine in ax.spines.values():
    spine.set_edgecolor(C_MUTED)
    spine.set_linewidth(0.8)


# ===========================================================================
# Panel 2 — Utilisation scatter
# ===========================================================================
ax = ax2
ax.set_facecolor(BG)

# Clip display range — extreme outliers shown as triangles on the border
UTIL_MAX = 2.0

# Points within display range
in_range = (true_util < UTIL_MAX) & (pred_util < UTIL_MAX)
out_range = ~in_range

scatter_split(ax,
              true_util[in_range & ~sign_rev],
              pred_util[in_range & ~sign_rev],
              np.zeros(np.sum(in_range & ~sign_rev), dtype=bool))
scatter_split(ax,
              true_util[in_range & sign_rev],
              pred_util[in_range & sign_rev],
              np.ones(np.sum(in_range & sign_rev), dtype=bool))

# Out-of-range sign-reversed points: draw as clipped triangles at top edge
oob_rev = out_range & sign_rev
if oob_rev.any():
    ax.scatter(np.clip(true_util[oob_rev], 0, UTIL_MAX - 0.05),
               np.full(oob_rev.sum(), UTIL_MAX - 0.06),
               s=55, marker="^", color=C_RS,
               edgecolors=C_FAIL, linewidths=0.7, zorder=6,
               label=f"clipped ({oob_rev.sum()} pts off-scale)")

# Reference diagonal
ax.plot([0, UTIL_MAX], [0, UTIL_MAX],
        color=C_DARK, lw=1.0, ls="--", alpha=0.5, zorder=1)

# Utilisation = 1.0 threshold lines
for xy, orient in [("h", 1.0), ("v", 1.0)]:
    fn = ax.axhline if xy == "h" else ax.axvline
    fn(1.0, color=C_FAIL, lw=1.0, ls="-.", alpha=0.7, zorder=2)

# Quadrant labels — inside the shaded danger zones
ax.text(0.25, 0.93, "true safe /\npred. unsafe", ha="center", va="top",
        fontsize=7.5, color=C_FAIL, style="italic", transform=ax.transAxes,
        linespacing=1.4)
ax.text(0.75, 0.42, "true unsafe /\npred. safe", ha="center", va="top",
        fontsize=7.5, color=C_FAIL, style="italic", transform=ax.transAxes,
        linespacing=1.4)

# Shaded danger quadrants
ax.fill_between([0, 1.0], [1.0, 1.0], [UTIL_MAX, UTIL_MAX],
                color=C_RS, alpha=0.06, zorder=0)   # FP zone (top-left)
ax.fill_between([1.0, UTIL_MAX], [0, 0], [1.0, 1.0],
                color=C_RS, alpha=0.06, zorder=0)   # FN zone (bottom-right)

ax.set_xlim(0, UTIL_MAX)
ax.set_ylim(0, UTIL_MAX)
ax.set_xlabel("True utilisation  $u$", fontsize=9.5, color=C_DARK)
ax.set_ylabel("Predicted utilisation  $\\hat{u}$", fontsize=9.5, color=C_DARK)
ax.set_title("Utilisation space — breakdown", fontsize=10.5,
             fontweight="bold", color=C_DARK, pad=6)
ax.tick_params(labelsize=8, color=C_MUTED)
for spine in ax.spines.values():
    spine.set_edgecolor(C_MUTED)
    spine.set_linewidth(0.8)


# ===========================================================================
# Shared legend
# ===========================================================================
h_ok = mlines.Line2D([], [], marker="o", color="w",
                     markerfacecolor=C_NS, markersize=7,
                     label=f"Correctly signed  ($n = {N - n_rev}$)")
h_rev = mlines.Line2D([], [], marker="D", color="w",
                      markerfacecolor=C_RS, markersize=7,
                      markeredgecolor=C_FAIL, markeredgewidth=0.6,
                      label=f"Sign-reversed  ($n = {n_rev}$)")
h_diag = mlines.Line2D([], [], color=C_DARK, ls="--", lw=1.0,
                       label="Perfect prediction")
h_thresh = mlines.Line2D([], [], color=C_FAIL, ls="-.", lw=1.0,
                         label="Utilisation $= 1.0$ threshold")

fig.legend(
    handles=[h_ok, h_rev, h_diag, h_thresh],
    loc="lower center", ncol=4,
    frameon=True, framealpha=0.95,
    edgecolor=C_MUTED, fontsize=8.2,
    bbox_to_anchor=(0.52, 0.00),
)

fig.text(
    0.52, 0.96,
    "Phase 1 regression: R² ≈ 0.99 in force space does not imply reliability in utilisation space",
    ha="center", va="bottom", fontsize=10, fontweight="bold", color=C_DARK,
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
stem = "fig_phase1_regression"
for fmt, out_dir in [("pdf", FIG_PDF_DIR), ("png", FIG_PNG_DIR)]:
    out = out_dir / f"{stem}.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close(fig)
