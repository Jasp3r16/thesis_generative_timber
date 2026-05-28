"""
Plan-view member assignment figure (ch 5.3):
  fig_member_assignment_plan – colour-coded plan view of rank-1 design
                               distinguishing NS and RS members.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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
    "font.size":         8.0,
    "mathtext.fontset":  "dejavusans",
    "figure.dpi":        150,
})

C_NS   = PLOT_COLORS["NS"]
C_RS   = PLOT_COLORS["RS"]
C_DARK = PLOT_COLORS["extra_colors"]["deep_navy"]
C_NEUT = PLOT_COLORS["neutral"]

# ── data ─────────────────────────────────────────────────────────────────────
RUN = Path(
    r"c:\Users\jaspe\OneDrive\06 Building Technology TU\2.2 - 2.4\60_Research_Exports"
    r"\03_ga_data\GA_A_20260519_163628_GEN250_EVAL7500_F-2_4697"
)
TOP = RUN / "top_k_designs"

verts = pd.read_csv(TOP / "GA_A_20260519_163628_GEN250_EVAL7500_F-2_4697_top10_vertices.csv")
edges = pd.read_csv(TOP / "GA_A_20260519_163628_GEN250_EVAL7500_F-2_4697_top10_edges_assigned.csv")
stock = pd.read_csv(RUN / "GA_A_20260519_163628_GEN250_EVAL7500_F-2_4697_stock.csv", sep=";")

v = verts[verts["rank"] == 1].copy().reset_index(drop=True)
e = edges[edges["rank"] == 1].copy().reset_index(drop=True)

v_idx = {int(r.vertex_index.replace("v", "")): (r.x, r.y, r.layer)
         for _, r in v.iterrows()}

e["state"] = e["assigned_timber"].str[:2]
e = e.merge(
    stock[["Member_ID", "Width", "Depth", "Origin_Country"]],
    left_on="assigned_timber", right_on="Member_ID", how="left",
)
e["mx"] = e.apply(lambda r: (v_idx[r.V1][0] + v_idx[r.V2][0]) / 2, axis=1)
e["my"] = e.apply(lambda r: (v_idx[r.V1][1] + v_idx[r.V2][1]) / 2, axis=1)

# ── layout bounds ─────────────────────────────────────────────────────────────
x_min, x_max = v["x"].min(), v["x"].max()
y_min, y_max = v["y"].min(), v["y"].max()
PAD_X, PAD_TOP, PAD_BOT = 0.4, 0.4, 2.8   # extra space at bottom for legend

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.0, 5.8))
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(x_min - PAD_X, x_max + PAD_X)
ax.set_ylim(y_min - PAD_BOT, y_max + PAD_TOP)

# draw edges
for _, row in e.iterrows():
    x1, y1, _ = v_idx[row.V1]
    x2, y2, _ = v_idx[row.V2]
    color = C_RS if row.state == "RS" else C_NS
    ax.plot([x1, x2], [y1, y2], color=color, lw=1.1, alpha=0.75, zorder=2)

# draw nodes
for _, row in v.iterrows():
    marker = "o" if row.layer == "top" else "s"
    ax.plot(row.x, row.y, marker=marker, ms=3.5, color=C_DARK,
            markeredgewidth=0, zorder=3)

# ── annotations ───────────────────────────────────────────────────────────────
# (edge_id, label_lines, state, text_xy offset from midpoint)
ANNOT = [
    ("e38",  "RS_00040\n110×230 mm\nNL", "RS",  (-2.8,  1.8)),   # SW → text upper-left
    ("e116", "RS_00100\n60×170 mm\nNL",  "RS",  ( 2.5,  1.6)),   # NE → text upper-right
    ("e3",   "NS_00388\n75×175 mm\nNL",  "NS",  (-2.5, -1.8)),   # SW → text lower-left
    ("e55",  "NS_00198\n250×250 mm\nNL", "NS",  ( 2.2, -1.8)),   # NE → text lower-right
]

ABOX = dict(boxstyle="round,pad=0.3", fc="white", ec=C_NEUT, lw=0.6, alpha=0.95)
ARROW = dict(arrowstyle="-", color=C_DARK, lw=0.9,
             linestyle=(0, (4, 3)))   # dashed

for eid, label, state, (dx, dy) in ANNOT:
    row  = e[e["edge_id"] == eid].iloc[0]
    mx, my = row.mx, row.my
    color = C_RS if state == "RS" else C_NS
    # dot at member midpoint
    ax.plot(mx, my, "o", ms=4.5, color=C_DARK, zorder=6)
    ax.annotate(
        label,
        xy=(mx, my),
        xytext=(mx + dx, my + dy),
        fontsize=7.5, color=color, ha="center", va="center",
        bbox=ABOX,
        arrowprops=ARROW,
        zorder=7,
    )

# ── legend — bottom right, right edge flush with structure ────────────────────
n_rs = (e["state"] == "RS").sum()
n_ns = (e["state"] == "NS").sum()
legend_handles = [
    mlines.Line2D([], [], color=C_RS, lw=2,
                  label=f"reclaimed stock  (RS,  n = {n_rs})"),
    mlines.Line2D([], [], color=C_NS, lw=2,
                  label=f"new stock  (NS,  n = {n_ns})"),
    mlines.Line2D([], [], color=C_DARK, marker="o", ms=4, lw=0,
                  markeredgewidth=0, label="top-chord node"),
    mlines.Line2D([], [], color=C_DARK, marker="s", ms=4, lw=0,
                  markeredgewidth=0, label="bottom-chord node"),
]

# anchor upper-right corner of legend just below the structure's bottom-right,
# using data coordinates so placement is exact regardless of tight_layout
ax.legend(
    handles=legend_handles,
    loc="upper right",
    bbox_to_anchor=(x_max + PAD_X, y_min - 0.25),
    bbox_transform=ax.transData,
    fontsize=7.5,
    frameon=True, framealpha=0.95, edgecolor=C_NEUT,
)

fig.tight_layout(pad=0.3)
fig.savefig(FIG_PDF_DIR / "fig_member_assignment_plan.pdf", bbox_inches="tight")
fig.savefig(FIG_PNG_DIR / "fig_member_assignment_plan.png", bbox_inches="tight", dpi=300)
print("saved fig_member_assignment_plan")
