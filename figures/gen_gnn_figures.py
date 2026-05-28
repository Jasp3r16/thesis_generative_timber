"""Generate GNN feature-table and normalisation-pipeline figures (ch 4.7)."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import FIG_PDF_DIR, FIG_PNG_DIR, PLOT_COLORS

matplotlib.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "Helvetica Neue", "DejaVu Sans"],
    "font.size":        8.5,
    "mathtext.fontset": "dejavusans",
    "figure.dpi":       150,
})

# ── palette ───────────────────────────────────────────────────────────────────
# Row background tints derived from PLOT_COLORS roles
C_GEO  = "#D6E8F0"   # primary-tinted  – geometry-derived
C_BC   = "#D5EDE0"   # teal-tinted     – boundary condition
C_LOAD = "#FEF3E2"   # accent-tinted   – applied load
C_STK  = "#F9E8E2"   # danger-tinted   – stock property
C_HEAD = PLOT_COLORS["neutral"]          # #D7D9D9 – header row

C_DARK = PLOT_COLORS["extra_colors"]["deep_navy"]   # #2F3E4F – text / boxes

# ── feature data ──────────────────────────────────────────────────────────────
node_rows = [
    ["$x$",   "m",  "Joint $x$-coordinate",          "geom", C_GEO],
    ["$y$",   "m",  "Joint $y$-coordinate",          "geom", C_GEO],
    ["$z$",   "m",  "Joint $z$-coordinate",          "geom", C_GEO],
    ["$T_x$", "—", "Fixed-DOF flag, translate $x$", "BC",   C_BC],
    ["$T_y$", "—", "Fixed-DOF flag, translate $y$", "BC",   C_BC],
    ["$T_z$", "—", "Fixed-DOF flag, translate $z$", "BC",   C_BC],
    ["$R_x$", "—", "Fixed-DOF flag, rotate $x$",    "BC",   C_BC],
    ["$R_y$", "—", "Fixed-DOF flag, rotate $y$",    "BC",   C_BC],
    ["$R_z$", "—", "Fixed-DOF flag, rotate $z$",    "BC",   C_BC],
    ["$F_z$", "N", "Nodal load (tributary area)",   "load", C_LOAD],
]

edge_rows = [
    ["$b$",           "m",          "Section width",                   "stock", C_STK],
    ["$h$",           "m",          "Section depth",                   "stock", C_STK],
    ["$L$",           "m",          "Installed member length",         "geom",  C_GEO],
    ["$E$",           "N m$^{-2}$", "Elastic modulus",                 "stock", C_STK],
    ["$I_y$",         "m$^{4}$",    "2nd moment of area ($y$-axis)",   "stock", C_STK],
    ["$I_z$",         "m$^{4}$",    "2nd moment of area ($z$-axis)",   "stock", C_STK],
    ["$J$",           "m$^{4}$",    "Torsional constant (St. Venant)", "stock", C_STK],
    ["$EA/L$",        "N m$^{-1}$", "Axial stiffness",                 "geom",  C_GEO],
    ["$N_{\\rm EA}$", "N",          "Mean-EA axial force estimate",    "geom",  C_GEO],
]

HEADERS = ["Symbol", "Unit", "Description", "Source"]
COL_W   = [0.10, 0.13, 0.57, 0.14]


def _make_table(ax, rows, subtitle):
    ax.axis("off")
    tbl = ax.table(
        cellText=  [[r[0], r[1], r[2], r[3]] for r in rows],
        colLabels=HEADERS,
        cellColours=[[r[4]] * 4 for r in rows],
        colColours=[C_HEAD] * 4,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.55)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_linewidth(0.35)
        cell.set_edgecolor("#CCCCCC")
        cell.PAD = 0.06
        cell.set_width(COL_W[col])
        if row == 0:
            cell.set_text_props(fontweight="bold", color="#222222")
    ax.set_title(subtitle, fontsize=9, fontweight="bold", color="#222222",
                 pad=5, loc="left")


# ── FIGURE 1: feature table ───────────────────────────────────────────────────
fig1, (ax_n, ax_e) = plt.subplots(1, 2, figsize=(13.5, 4.8))
fig1.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.09, wspace=0.06)

_make_table(ax_n, node_rows, "Node features  (d = 10)")
_make_table(ax_e, edge_rows, "Edge features  (d = 9)")

fig1.add_artist(matplotlib.lines.Line2D(
    [0.505, 0.505], [0.05, 0.97],
    transform=fig1.transFigure, color="#CCCCCC", linewidth=0.8,
))
fig1.legend(
    handles=[
        mpatches.Patch(fc=C_GEO,  ec="#BBBBBB", lw=0.5, label="geometry-derived"),
        mpatches.Patch(fc=C_BC,   ec="#BBBBBB", lw=0.5, label="boundary condition"),
        mpatches.Patch(fc=C_LOAD, ec="#BBBBBB", lw=0.5, label="applied load"),
        mpatches.Patch(fc=C_STK,  ec="#BBBBBB", lw=0.5, label="stock property"),
    ],
    loc="lower center", ncol=4, fontsize=7.5, frameon=False,
    bbox_to_anchor=(0.5, 0.005),
)

fig1.savefig(FIG_PDF_DIR / "fig_gnn_feature_table.pdf", bbox_inches="tight")
fig1.savefig(FIG_PNG_DIR / "fig_gnn_feature_table.png", bbox_inches="tight", dpi=300)
print("saved fig_gnn_feature_table")

# ── FIGURE 2: normalisation pipeline ─────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(8, 2.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.2)
ax.axis("off")

BOX_KW   = dict(boxstyle="round,pad=0.5", facecolor="#F5F7FA",
                edgecolor=C_DARK, linewidth=0.9)
ARROW_KW = dict(arrowstyle="-|>", color=C_DARK, lw=1.1, mutation_scale=11)

box_specs = [
    (1.5,  1.9, "raw features\n$\\mathbf{x} \\in \\mathbb{R}^{n \\times d}$"),
    (5.0,  1.9, "$z = (x - \\mu)\\;/\\;\\sigma$"),
    (8.5,  1.9, "$\\hat{x} = \\mathrm{clip}(z,\\;{-5},\\;{+5})$"),
]
for cx, cy, lbl in box_specs:
    ax.text(cx, cy, lbl, ha="center", va="center",
            fontsize=10, linespacing=1.6, bbox=BOX_KW)

for x0, x1 in [(2.75, 3.45), (6.55, 7.25)]:
    ax.annotate("", xy=(x1, 1.9), xytext=(x0, 1.9), arrowprops=ARROW_KW)

ax.text(4.0,  2.2,  "standardise", ha="center", fontsize=7.5, color="#777777")
ax.text(6.9,  2.2,  "clip",        ha="center", fontsize=7.5, color="#777777")

ax.text(5.0, 3.05,
        r"$\mu,\;\sigma$ from training set — stored in model checkpoint",
        ha="center", va="center", fontsize=7.5, color="#888888", style="italic")

example = [
    (1.5,  "$EA/L = 8.2\\times10^{6}$"),
    (5.0,  "$z = 1.73$"),
    (8.5,  "$\\hat{x} = 1.73$"),
]
for ex_x, ex_txt in example:
    ax.text(ex_x, 0.55, ex_txt, ha="center", va="center",
            fontsize=7.5, color="#999999", style="italic")
    ax.annotate("", xy=(ex_x, 1.2), xytext=(ex_x, 0.85),
                arrowprops=dict(arrowstyle="-|>", color="#CCCCCC",
                                lw=0.7, mutation_scale=7))

fig2.savefig(FIG_PDF_DIR / "fig_gnn_norm_pipeline.pdf", bbox_inches="tight")
fig2.savefig(FIG_PNG_DIR / "fig_gnn_norm_pipeline.png", bbox_inches="tight", dpi=300)
print("saved fig_gnn_norm_pipeline")
