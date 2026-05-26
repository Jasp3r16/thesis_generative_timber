"""
fig_cma_es.py — CMA-ES sampling ellipsoid evolution, section 2.5.6.
Three panels: k=0 (circular, uninformed), k=1 (elongating), k=2 (converging).
No caption embedded — provided separately.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
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
# Fitness landscape — rotated elongated quadratic bowl
# ---------------------------------------------------------------------------
OPT = np.array([2.4, 1.8])
_a  = 0.55
_R  = np.array([[np.cos(_a), -np.sin(_a)],
                [np.sin(_a),  np.cos(_a)]])
_D  = np.diag([1.0, 6.0])
_M  = _R @ _D @ _R.T


def fitness(X, Y):
    pts = np.stack([X.ravel() - OPT[0], Y.ravel() - OPT[1]], axis=1)
    vals = np.einsum("ij,jk,ik->i", pts, _M, pts)
    return vals.reshape(X.shape)


# ---------------------------------------------------------------------------
# CMA-ES generation states
# ---------------------------------------------------------------------------
GEN_DATA = [
    dict(
        mean=np.array([-1.6, -1.1]),
        cov=np.array([[2.0, 0.0], [0.0, 2.0]]),
        label="$k = 0$  ·  uninformed",
        note="large, circular  —  no directional information",
    ),
    dict(
        mean=np.array([0.6, 0.6]),
        cov=np.array([[1.6, 0.8], [0.8, 0.65]]),
        label="$k = 1$  ·  adapting",
        note="ellipse elongates along fitness-improving direction",
    ),
    dict(
        mean=np.array([2.1, 1.6]),
        cov=np.array([[0.30, 0.19], [0.19, 0.18]]),
        label="$k = 2$  ·  converging",
        note="tight, aligned with landscape  —  near optimum",
    ),
]

np.random.seed(42)

XLIM = (-4.5, 5.0)
YLIM = (-3.2, 4.2)
xx, yy = np.meshgrid(np.linspace(*XLIM, 220), np.linspace(*YLIM, 220))
zz = fitness(xx, yy)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 13.0, 5.0
fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H),
                          gridspec_kw=dict(wspace=0.06))
fig.patch.set_facecolor(BG)

for k, (ax, gd) in enumerate(zip(axes, GEN_DATA)):
    ax.set_facecolor(BG)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_edgecolor("#E0E0E0")
        sp.set_linewidth(0.8)

    # Fitness landscape
    ax.contourf(xx, yy, zz, levels=14, cmap="YlOrRd", alpha=0.18, zorder=0)
    ax.contour(xx, yy, zz, levels=8, colors=[C_MUTED],
               linewidths=0.45, alpha=0.45, zorder=1)

    # Optimum star
    ax.scatter(*OPT, s=160, marker="*", color=C_RS,
               zorder=8, edgecolors="white", linewidths=0.8)

    mean = gd["mean"]
    cov  = gd["cov"]

    # Candidate samples — evaluate fitness, highlight top-k
    samples  = np.random.multivariate_normal(mean, cov, size=45)
    fit_vals = fitness(samples[:, 0:1], samples[:, 1:2]).ravel()
    top_k    = np.argsort(fit_vals)[:12]
    rest     = np.argsort(fit_vals)[12:]

    ax.scatter(samples[rest, 0], samples[rest, 1],
               s=16, color=C_NS, alpha=0.28, zorder=3)
    ax.scatter(samples[top_k, 0], samples[top_k, 1],
               s=28, color=C_NS, alpha=0.72, zorder=4)

    # Sampling ellipse at 2σ
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    ew = 4 * np.sqrt(eigvals[1])   # full width along major axis
    eh = 4 * np.sqrt(eigvals[0])   # full height along minor axis
    ell = Ellipse(mean, width=ew, height=eh, angle=angle,
                  facecolor=C_NS + "2A", edgecolor=C_NS, linewidth=2.2, zorder=5)
    ax.add_patch(ell)

    # Distribution mean
    ax.scatter(*mean, s=100, color=C_NS, zorder=6,
               edgecolors="white", linewidths=1.5)

    # Arrow toward optimum (scaled to stay within axes)
    if k < 2:
        d = OPT - mean
        d = d / np.linalg.norm(d) * 1.1
        ax.annotate("",
            xy=mean + d, xytext=mean,
            arrowprops=dict(arrowstyle="-|>", color=C_DARK,
                            lw=1.6, mutation_scale=14),
            zorder=9)

    ax.set_title(gd["label"], fontsize=10.5, fontweight="bold",
                 color=C_DARK, pad=8)
    # Note below the panel (outside axes, clip_on=False)
    ax.text(0.5, -0.04, gd["note"],
            ha="center", va="top", fontsize=7.8, color=C_MUTED,
            style="italic", transform=ax.transAxes, clip_on=False)

# ---------------------------------------------------------------------------
# Legend — below the per-panel notes
# ---------------------------------------------------------------------------
legend_handles = [
    mpatches.Patch(facecolor=C_NS + "2A", edgecolor=C_NS, linewidth=1.5,
                   label="Sampling distribution  (2σ ellipse)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_NS,
           markersize=9, label="Distribution mean  $\\mathbf{m}$"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_NS,
           markersize=6, alpha=0.45, label="Candidate samples"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor=C_RS,
           markersize=11, label="Optimum  $\\mathbf{x}^*$"),
    Line2D([0], [0], color=C_DARK, lw=1.6, marker=">",
           markersize=6, label="Mean update direction"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
           fontsize=8.5, frameon=True, framealpha=0.95,
           edgecolor=C_MUTED, bbox_to_anchor=(0.5, 0.07))

fig.subplots_adjust(left=0.01, right=0.99, bottom=0.22, top=0.93, wspace=0.06)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
stem = "fig_cma_es"
for fmt, out_dir in [("pdf", FIG_PDF_DIR), ("png", FIG_PNG_DIR)]:
    out = out_dir / f"{stem}.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close(fig)
