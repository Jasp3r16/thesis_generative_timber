"""
fig_feasibility_filter.py — Feasibility filter pipeline (Section 4.4).

Left:  Three 120×524 heatmaps showing the Boolean feasibility mask at successive
       filter stages (initial → after length → after all EC5 checks).
       Rows = structural slots (sorted by length), columns = stock items (NS then RS).

Right: Horizontal waterfall chart showing cumulative feasible slot–stock
       combinations remaining after each of the six filter steps.

Data: best-found Stock A design geometry (39 nodes, 120 members) from the GA,
      evaluated against Stock A inventory (524 items → 62,880 combinations total).
Stage counts derived from c24_stage_feasibility.build_cost_filter() logic.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import FIG_PDF_DIR, FIG_PNG_DIR, EXPORT_PATH

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_NS    = "#61788C"
C_RS    = "#F2994B"
C_DARK  = "#2F3E4F"
C_MUTED = "#9CA5A6"
C_LIGHT = "#D7D9D9"
C_FAIL  = "#D9653B"
BG      = "#FFFFFF"
C_FEAS  = C_NS
C_INFEAS = "#E8ECEE"

# ---------------------------------------------------------------------------
# Data — 120-slot best GA_A geometry × Stock A (524 items)
# ---------------------------------------------------------------------------
GA_OUT    = EXPORT_PATH / "03_ga_data" / "c30_output"
STOCK_PATH = (Path.home() / "OneDrive" / "06 Building Technology TU" / "2.2 - 2.4" /
              "30_Data_Inventory" / "03_timber_data" / "complete_timber_A.csv")

verts  = pd.read_csv(GA_OUT / "design_best_SA_vertices.csv")
bom    = pd.read_csv(GA_OUT / "design_best_SA_bom.csv")
stock  = pd.read_csv(STOCK_PATH, sep=";")

node_pos      = verts.sort_values("vi")[["x", "y", "z"]].values
support_nodes = verts[verts["attribute"] == "support"]["vi"].tolist()
load_nodes    = verts[verts["attribute"] == "load"]["vi"].tolist()
edges_v1      = bom["V1"].values
edges_v2      = bom["V2"].values

from workflows import c24_stage_feasibility as feas_mod

slot_lengths_m = feas_mod.compute_member_lengths(node_pos, edges_v1, edges_v2)
mean_EA = (stock["E_modulus_eff"].mean() * 1e6 *
           (stock["Depth"] * stock["Width"]).mean() * 1e-6)
forces_n = feas_mod.estimate_member_forces(
    node_pos, edges_v1, edges_v2,
    support_nodes, load_nodes, feas_mod.TOTAL_LOAD_N, mean_EA,
)

N_SLOTS, N_STOCK = len(slot_lengths_m), len(stock)
total = N_SLOTS * N_STOCK
print(f"Slots: {N_SLOTS}, Stock: {N_STOCK}, Total: {total:,}")

# c24 filter constants
MAX_OVERSIZE = 0.50; MAX_SLEN = 150; MAX_DL = 40; MAX_WD = 5
FSF = 2.0; GM = 1.3; KM = 0.8; BC = 0.2

slot_mm  = slot_lengths_m * 1000.0
stock_mm = stock["Length"].values
d  = stock["Depth"].values
w  = stock["Width"].values
ftk  = stock["f_tk"].values
fc0k = stock["f_c0k"].values
E005 = stock["E_modulus_005"].values
Nd   = forces_n * FSF

mask_len = ((stock_mm[None, :] >= slot_mm[:, None]) &
            (stock_mm[None, :] <= slot_mm[:, None] * (1 + MAX_OVERSIZE)))

iz = np.minimum(d, w) / np.sqrt(12.0)
comp = Nd < -1.0
mask_3a = np.ones((N_SLOTS, N_STOCK), dtype=bool)
if comp.any():
    mask_3a[comp] = (slot_mm[comp, None] / iz[None, :] <= MAX_SLEN)

mask_3b = d[None, :] >= (slot_mm / MAX_DL)[:, None]
mask_3c = w[None, :] >= (d / MAX_WD)[None, :]

ftd = KM * ftk / GM; fcd = KM * fc0k / GM; A = d * w
mask_str = np.ones((N_SLOTS, N_STOCK), dtype=bool)
tens = Nd >= 1.0
if tens.any():
    mask_str[tens] &= (A[None, :] >= Nd[tens, None] / ftd[None, :])
if comp.any():
    Nc  = np.abs(Nd[comp])
    lam = slot_mm[comp, None] / iz[None, :]
    lr  = (lam / np.pi) * np.sqrt(fc0k[None, :] / E005[None, :])
    k   = 0.5 * (1.0 + BC * (lr - 0.3) + lr ** 2)
    kc  = np.where(lr <= 0.3, 1.0,
                   1.0 / (k + np.sqrt(np.maximum(k**2 - lr**2, 0.0))))
    kc  = np.clip(kc, 0.05, 1.0)
    mask_str[comp] &= (A[None, :] >= Nc[:, None] / (kc * fcd[None, :]))

mask_initial  = np.ones((N_SLOTS, N_STOCK), dtype=bool)
mask_after_len = mask_len
mask_after_ec5 = mask_len & mask_3a & mask_3b & mask_3c & mask_str

n_len = int(mask_len.sum())
n_3a  = int((mask_len & mask_3a).sum())
n_3b  = int((mask_len & mask_3a & mask_3b).sum())
n_3c  = int((mask_len & mask_3a & mask_3b & mask_3c).sum())
n_fin = int(mask_after_ec5.sum())

for label, n in [("Total", total), ("After length", n_len), ("After 3a", n_3a),
                 ("After 3b", n_3b), ("After 3c", n_3c), ("After 3d/e", n_fin)]:
    print(f"  {label:<18}: {n:6,}  ({100*n/total:.1f}%)")

# Slot sort order: short → long
slot_order  = np.argsort(slot_lengths_m)
NS_BOUNDARY = 421    # first RS column index

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 14.0, 7.0
fig = plt.figure(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(BG)

outer = gridspec.GridSpec(1, 2, left=0.03, right=0.97,
                          top=0.91, bottom=0.10,
                          wspace=0.08, width_ratios=[1.0, 1.35])

gs_left = gridspec.GridSpecFromSubplotSpec(3, 1,
              subplot_spec=outer[0], hspace=0.40)
ax_h = [fig.add_subplot(gs_left[i]) for i in range(3)]
ax_wf = fig.add_subplot(outer[1])

# ---------------------------------------------------------------------------
# Heatmap helper
# ---------------------------------------------------------------------------
def draw_heatmap(ax, mask, title, pct_label):
    img = mask[slot_order, :].astype(np.float32)
    feas_rgb  = np.array([0x61, 0x78, 0x8C]) / 255.0
    infeas_rgb = np.array([0xE8, 0xEC, 0xEE]) / 255.0
    rgba = np.zeros((*img.shape, 4))
    rgba[..., :3] = img[..., None] * feas_rgb + (1 - img[..., None]) * infeas_rgb
    rgba[..., 3]  = 1.0

    ax.imshow(rgba, aspect="auto", interpolation="nearest", origin="upper")
    ax.axvline(NS_BOUNDARY - 0.5, color=C_DARK, lw=0.8, alpha=0.5)

    ax.set_yticks([0, 59, 119])
    ax.set_yticklabels(["short", "mid", "long"], fontsize=6.5, color=C_DARK)
    ax.set_xticks([NS_BOUNDARY // 2, NS_BOUNDARY + 51])
    ax.set_xticklabels(["NS  (421)", "RS  (103)"], fontsize=6.5, color=C_DARK)
    ax.tick_params(length=2, pad=2, color=C_MUTED)
    for sp in ax.spines.values():
        sp.set_edgecolor(C_LIGHT)
        sp.set_linewidth(0.6)

    ax.text(1.0, 1.04, pct_label, ha="right", va="bottom",
            fontsize=7.5, color=C_NS, fontweight="bold",
            transform=ax.transAxes)


masks      = [mask_initial, mask_after_len, mask_after_ec5]
hm_titles  = ["Initial  (all combinations)",
               "After length filter  (Stage 1)",
               "After EC5 checks  (Stages 3a–3e)"]
pct_labels = [f"{total:,} / {total:,}  =  100.0%",
              f"{n_len:,} / {total:,}  =  {100*n_len/total:.1f}%",
              f"{n_fin:,} / {total:,}  =  {100*n_fin/total:.1f}%"]

for i, (m, t, p) in enumerate(zip(masks, hm_titles, pct_labels)):
    draw_heatmap(ax_h[i], m, t, p)

ax_h[1].set_ylabel("slots  (sorted by length)", fontsize=7.5,
                    color=C_DARK, labelpad=4)

# ---------------------------------------------------------------------------
# Waterfall chart
# ---------------------------------------------------------------------------
ax = ax_wf
ax.set_facecolor(BG)
for sp in ax.spines.values():
    sp.set_edgecolor(C_LIGHT); sp.set_linewidth(0.7)
ax.tick_params(labelsize=7.5, color=C_MUTED)
ax.grid(axis="x", color=C_LIGHT, lw=0.5, alpha=0.6, zorder=0)

stages = [
    ("Total  (120 × 524)",         "",                             total),
    ("Stage 1 — length",           "stock ≥ slot, ≤ slot × 1.5",  n_len),
    ("Stage 3a — slenderness",     "λ ≤ 150",                      n_3a),
    ("Stage 3b — depth / length",  "d ≥ L / 40",                   n_3b),
    ("Stage 3c — width / depth",   "w ≥ d / 5",                    n_3c),
    ("Stage 3d/e — strength",      "tension & compression  (EC5)", n_fin),
]
remainders  = [s[2] for s in stages]
elim_counts = [0] + [remainders[i] - remainders[i+1]
                     for i in range(len(remainders) - 1)]
y_pos = np.arange(len(stages))[::-1]

for i, ((lbl, cond, remaining), elim, y) in enumerate(
        zip(stages, elim_counts, y_pos)):

    bar_col = C_DARK if i == 0 else C_NS

    # Ghost bar
    ax.barh(y, total, left=0, height=0.65,
            color=C_INFEAS, alpha=0.45, zorder=1)

    # Coloured remaining bar
    ax.barh(y, remaining, left=0, height=0.65,
            color=bar_col, alpha=0.88, zorder=2)

    # Stage label — wrap at " — " so long labels stay within the bar
    wrapped = lbl.replace(" — ", "\n", 1) if " — " in lbl else lbl
    first_line = wrapped.split("\n")[0]
    # White on colour if bar covers the first line; dark on ghost bar if bar is short
    _lbl_end = total * 0.012 + len(first_line) * total * 0.014
    lbl_color = BG if remaining > _lbl_end else C_DARK
    ax.text(total * 0.012, y, wrapped,
            ha="left", va="center", fontsize=7.8,
            color=lbl_color, fontweight="bold", zorder=3,
            linespacing=1.25)

    # Count + % right-aligned inside ghost bar — never extends beyond total
    # White if the colored bar covers the text position, dark if it sits on ghost bar
    count_color = BG if remaining >= total * 0.85 else C_DARK
    ax.text(total * 0.984, y,
            f"{remaining:,}  ({100*remaining/total:.0f}%)",
            ha="right", va="center", fontsize=7.8,
            color=count_color, fontweight="bold", zorder=3)

    # Elimination delta just outside ghost bar (compact, red)
    if elim > 0:
        ax.text(total * 1.012, y,
                f"−{elim:,}",
                ha="left", va="center", fontsize=7.2,
                color=C_FAIL, zorder=4)

ax.set_yticks([])
ax.set_xlim(0, total * 1.18)
ax.set_xlabel("Feasible slot–stock combinations  (of 62,880 total)",
              fontsize=8.5, color=C_DARK)
ax.set_xticks([0, 20000, 40000, 60000])
ax.xaxis.set_tick_params(labelsize=7.5)

# ---------------------------------------------------------------------------
# Heatmap legend
# ---------------------------------------------------------------------------
leg_feas  = mpatches.Patch(facecolor=C_NS,    edgecolor=C_DARK, lw=0.5,
                            label="Feasible")
leg_infeas = mpatches.Patch(facecolor=C_INFEAS, edgecolor=C_MUTED, lw=0.5,
                             label="Eliminated")
ax_h[2].legend(handles=[leg_feas, leg_infeas],
               loc="lower right", ncol=2,
               frameon=True, framealpha=0.95, edgecolor=C_LIGHT,
               fontsize=7.5, handlelength=1.0, handletextpad=0.4)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
stem = "fig_feasibility_filter"
for fmt, out_dir in [("pdf", FIG_PDF_DIR), ("png", FIG_PNG_DIR)]:
    out = out_dir / f"{stem}.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close(fig)
