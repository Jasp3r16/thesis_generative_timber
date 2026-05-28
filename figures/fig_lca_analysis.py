"""
LCA analysis figures (ch 4.5.6–4.5.7):
  fig_transport_scatter  – transport distance vs emission factor, NS/RS
  fig_lca_constants      – LCA constants magnitude, log-scale bar chart
  fig_cost_matrix_heatmap – 120×524 cost matrix heatmap, sparsity & NS/RS split
"""
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
C_SAGE  = PLOT_COLORS["extra_colors"]["soft_sage_green"]
C_TEAL  = PLOT_COLORS["extra_colors"]["muted_teal"]
C_ACCENT = PLOT_COLORS["accent"]
C_PRIM  = PLOT_COLORS["primary"]
C_DANG  = PLOT_COLORS["danger"]

# ── paths ─────────────────────────────────────────────────────────────────────
STOCK_CSV = Path(
    r"c:\Users\jaspe\OneDrive\06 Building Technology TU\2.2 - 2.4"
    r"\30_Data_Inventory\03_timber_data\complete_timber_A.csv"
)
CM_NPY = Path(__file__).parent / "cost_matrix_sample.npy"

# ── ① TRANSPORT SCATTER ───────────────────────────────────────────────────────
stock = pd.read_csv(STOCK_CSV, sep=";")

ns = stock[stock["State"] == 0]
rs = stock[stock["State"] == 1]

fig1, ax = plt.subplots(figsize=(6.0, 4.0))

ax.scatter(ns["Transport_Dist"], ns["EmissionFactor"],
           color=C_NS, s=14, alpha=0.55, linewidths=0, label=f"new stock (NS,  n={len(ns)})")
ax.scatter(rs["Transport_Dist"], rs["EmissionFactor"],
           color=C_RS, s=14, alpha=0.65, linewidths=0, label=f"reclaimed (RS,  n={len(rs)})")

ax.set_xlabel("transport distance  (km)", labelpad=5)
ax.set_ylabel("emission factor  (kg CO$_2$e t$^{-1}$km$^{-1}$)", labelpad=5)
ax.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor=C_NEUT)

fig1.tight_layout()
fig1.savefig(FIG_PDF_DIR / "fig_transport_scatter.pdf", bbox_inches="tight")
fig1.savefig(FIG_PNG_DIR / "fig_transport_scatter.png", bbox_inches="tight", dpi=300)
print("saved fig_transport_scatter")

# ── ② LCA CONSTANTS BAR CHART ─────────────────────────────────────────────────
CONSTANTS = [
    ("A1–A3\nembodied carbon",    0.250,  "EPD",           C_PRIM),
    ("C3+C4\noffcut disposal",    0.031,  "Ecoinvent v3",  C_TEAL),
    ("A5\npreparation",           0.010,  "Bergman 2010",  C_SAGE),
    ("C1\ndeconstruction",        0.0085, "Bergman 2010",  C_SAGE),
    ("C2\nwaste transport*",      0.00875,"EN 15978",      C_NEUT),
    ("A5\nsawing",                0.004,  "Calculated",    C_ACCENT),
]
labels  = [c[0] for c in CONSTANTS]
values  = [c[1] for c in CONSTANTS]
sources = [c[2] for c in CONSTANTS]
colors  = [c[3] for c in CONSTANTS]

fig2, ax2 = plt.subplots(figsize=(7.5, 3.6))

y = np.arange(len(labels))
bars = ax2.barh(y, values, color=colors, edgecolor="white", linewidth=0.4, height=0.65)

ax2.set_xscale("log")
ax2.set_xlim(0.002, 0.6)
ax2.set_yticks(y)
ax2.set_yticklabels(labels, fontsize=8)
ax2.set_xlabel("kg CO$_2$e  kg$^{-1}$  (log scale)", labelpad=5)
ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4g"))
ax2.tick_params(axis="x", which="minor", length=0)

# value and source annotations
for i, (val, src) in enumerate(zip(values, sources)):
    ax2.text(val * 1.15, i, f"{val}",
             va="center", ha="left", fontsize=7.5, color=C_DARK)
    ax2.text(0.0022, i, src,
             va="center", ha="left", fontsize=7, color="#777777", style="italic")

ax2.axvline(0.003, color=C_NEUT, lw=0.5, ls="--")  # source label separator

ax2.text(0.003, -0.7, "* 50 km × mean emission factor",
         fontsize=6.5, color="#888888", style="italic", va="top")

ax2.invert_yaxis()
fig2.tight_layout()
fig2.savefig(FIG_PDF_DIR / "fig_lca_constants.pdf", bbox_inches="tight")
fig2.savefig(FIG_PNG_DIR / "fig_lca_constants.png", bbox_inches="tight", dpi=300)
print("saved fig_lca_constants")

# ── ③ COST MATRIX HEATMAP ─────────────────────────────────────────────────────
cm = np.load(CM_NPY)          # (120, 524)
INF = 1e9

# sort slots (rows) by length — already in edge order, keep as-is
# sort columns: NS first, RS second; within each, sort by median finite cost
ns_idx = np.where(stock["State"].values == 0)[0]
rs_idx = np.where(stock["State"].values == 1)[0]

def sort_by_median(idx):
    meds = []
    for j in idx:
        col = cm[:, j]; finite = col[col < INF]
        meds.append(np.median(finite) if len(finite) > 0 else INF)
    return idx[np.argsort(meds)]

ns_sorted = sort_by_median(ns_idx)
rs_sorted = sort_by_median(rs_idx)
col_order = np.concatenate([ns_sorted, rs_sorted])
cm_sorted = cm[:, col_order]

# masked array: INF → NaN
cm_plot = np.where(cm_sorted >= INF, np.nan, cm_sorted)
vmax = np.nanpercentile(cm_plot, 95)

cmap = mcolors.LinearSegmentedColormap.from_list(
    "cm_ramp", [C_SAGE, C_PRIM, C_DANG], N=256)
cmap.set_bad(color=C_NEUT, alpha=0.55)

fig3, ax3 = plt.subplots(figsize=(13, 3.8))
im = ax3.imshow(cm_plot, aspect="auto", cmap=cmap,
                vmin=0, vmax=vmax, interpolation="nearest")

# NS / RS divider
n_ns = len(ns_sorted)
ax3.axvline(n_ns - 0.5, color="white", lw=1.2)
ax3.text(n_ns / 2,           -4,  "new stock (NS)",      ha="center",
         fontsize=8, color=C_NS, fontweight="bold")
ax3.text(n_ns + len(rs_sorted)/2, -4, "reclaimed (RS)", ha="center",
         fontsize=8, color=C_RS, fontweight="bold")

ax3.set_xlabel("stock element index  (sorted by median cost within NS / RS)", labelpad=5)
ax3.set_ylabel("slot index")
ax3.set_yticks([0, 30, 60, 90, 119])

cbar = fig3.colorbar(im, ax=ax3, pad=0.01, fraction=0.015)
cbar.set_label("LCA cost  (kg CO$_2$e)", fontsize=8)

# INF patch for legend
from matplotlib.patches import Patch
ax3.legend(
    handles=[Patch(facecolor=C_NEUT, alpha=0.55, label="infeasible (∞)")],
    loc="lower right", fontsize=7.5, frameon=True,
    framealpha=0.9, edgecolor=C_NEUT,
)

fig3.tight_layout()
fig3.savefig(FIG_PDF_DIR / "fig_cost_matrix_heatmap.pdf", bbox_inches="tight")
fig3.savefig(FIG_PNG_DIR / "fig_cost_matrix_heatmap.png", bbox_inches="tight", dpi=300)
print("saved fig_cost_matrix_heatmap")
