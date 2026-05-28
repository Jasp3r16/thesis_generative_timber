"""
LCA per-pair component breakdown (ch 4.5.7):
  fig_lca_breakdown – mean stacked bars (NS vs RS) + per-pair individual bars,
                      split into separate y-scales for NS and RS
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
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
    "font.size":         8.5,
    "mathtext.fontset":  "dejavusans",
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

C_NS    = PLOT_COLORS["NS"]
C_RS    = PLOT_COLORS["RS"]
C_DARK  = PLOT_COLORS["extra_colors"]["deep_navy"]
C_NEUT  = PLOT_COLORS["neutral"]
C_TEAL  = PLOT_COLORS["extra_colors"]["muted_teal"]
C_SAGE  = PLOT_COLORS["extra_colors"]["soft_sage_green"]
C_PRIM  = PLOT_COLORS["primary"]
C_DANG  = PLOT_COLORS["danger"]
C_ACCENT= PLOT_COLORS["accent"]

COMP_COLORS = {
    "e_embodied":  C_PRIM,
    "e_c1":        C_SAGE,
    "e_transport": C_TEAL,
    "e_prep":      C_ACCENT,
    "e_saw":       "#F7C59F",
    "e_waste":     C_DANG,
}
COMP_LABELS = {
    "e_embodied":  "A1–A3  embodied",
    "e_c1":        "C1  deconstruction",
    "e_transport": "A4  transport",
    "e_prep":      "A5  preparation",
    "e_saw":       "A5  sawing",
    "e_waste":     "C2+C3+C4  waste",
}
COMPS = list(COMP_COLORS.keys())

# ── data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(Path(__file__).parent / "assignment_breakdown.csv")
ns = df[df["state"] == 0].sort_values("total").reset_index(drop=True)
rs = df[df["state"] == 1].sort_values("total").reset_index(drop=True)
ns_means = ns[COMPS].mean()
rs_means = rs[COMPS].mean()

def draw_stacked_bars(ax, subset, x_arr, bar_w=0.85):
    bottom = np.zeros(len(subset))
    for comp in COMPS:
        vals = subset[comp].values
        mask = vals > 1e-6
        if mask.any():
            ax.bar(x_arr[mask], vals[mask], bottom=bottom[mask],
                   color=COMP_COLORS[comp], edgecolor="none", width=bar_w)
        bottom += vals
    return bottom.max()

# ── figure: 1×3 panels ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(13.5, 4.4))
gs  = gridspec.GridSpec(1, 3, figure=fig,
                         width_ratios=[1.1, 2.2, 4.5],
                         wspace=0.35)
ax_mean = fig.add_subplot(gs[0])
ax_ns   = fig.add_subplot(gs[1])
ax_rs   = fig.add_subplot(gs[2])

# ── (a) mean stacked bars ─────────────────────────────────────────────────────
bar_w_m = 0.55
for xi, (means, label, n, col) in enumerate([
        (ns_means, "NS", len(ns), C_NS),
        (rs_means, "RS", len(rs), C_RS),
]):
    bottom = 0.0
    for comp in COMPS:
        v = means[comp]
        if v < 1e-6:
            continue
        ax_mean.bar(xi, v, bottom=bottom, width=bar_w_m,
                    color=COMP_COLORS[comp], edgecolor="white", linewidth=0.5)
        if v > 0.08:
            ax_mean.text(xi, bottom + v / 2, f"{v:.2f}",
                         ha="center", va="center", fontsize=6.5,
                         color="white", fontweight="bold")
        bottom += v
    ax_mean.text(xi, bottom + 0.2, f"{bottom:.2f}",
                 ha="center", va="bottom", fontsize=8, color=C_DARK,
                 fontweight="bold")
    ax_mean.text(xi, -1.2, f"{label}\n(n={n})",
                 ha="center", va="top", fontsize=8, color=col,
                 fontweight="bold")

ax_mean.set_xlim(-0.7, 1.7)
ax_mean.set_ylim(-2.0, 9.0)
ax_mean.set_ylabel("mean LCA cost  (kg CO$_2$e)")
ax_mean.set_xticks([])
ax_mean.spines["bottom"].set_visible(False)
ax_mean.text(0.5, 1.02, "(a)", transform=ax_mean.transAxes,
             ha="center", fontsize=8.5, fontweight="bold", color=C_DARK)

# ── (b) NS per-pair ───────────────────────────────────────────────────────────
x_ns = np.arange(len(ns), dtype=float)
top_ns = draw_stacked_bars(ax_ns, ns, x_ns)
ax_ns.set_ylim(0, top_ns * 1.12)
ax_ns.set_xlim(-1, len(ns))
ax_ns.set_xticks([])
ax_ns.set_ylabel("LCA cost  (kg CO$_2$e)")
ax_ns.text(0.5, 0.97, f"(b)  new stock  (NS, n={len(ns)})",
           transform=ax_ns.transAxes, ha="center", va="top",
           fontsize=8.5, fontweight="bold", color=C_NS)
ax_ns.set_xlabel("assignments sorted by total cost", labelpad=4)
ax_ns.spines["bottom"].set_visible(False)

# ── (c) RS per-pair ───────────────────────────────────────────────────────────
x_rs = np.arange(len(rs), dtype=float)
top_rs = draw_stacked_bars(ax_rs, rs, x_rs)
ax_rs.set_ylim(0, top_rs * 1.12)
ax_rs.set_xlim(-1, len(rs))
ax_rs.set_xticks([])
ax_rs.set_ylabel("LCA cost  (kg CO$_2$e)")
ax_rs.text(0.5, 0.97, f"(c)  reclaimed stock  (RS, n={len(rs)})",
           transform=ax_rs.transAxes, ha="center", va="top",
           fontsize=8.5, fontweight="bold", color=C_RS)
ax_rs.set_xlabel("assignments sorted by total cost", labelpad=4)
ax_rs.spines["bottom"].set_visible(False)

# ── legend ────────────────────────────────────────────────────────────────────
patches = [mpatches.Patch(fc=COMP_COLORS[c], label=COMP_LABELS[c]) for c in COMPS]
fig.legend(handles=patches, loc="lower center", ncol=6,
           fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.04))

fig.savefig(FIG_PDF_DIR / "fig_lca_breakdown.pdf", bbox_inches="tight")
fig.savefig(FIG_PNG_DIR / "fig_lca_breakdown.png", bbox_inches="tight", dpi=300)
print("saved fig_lca_breakdown")
