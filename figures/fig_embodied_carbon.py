"""
fig_embodied_carbon.py — Production-phase embodied carbon bar chart for chapter 1.3.

Single panel: A1-A3 production emissions, new softwood vs reclaimed timber.

Values from:
  TDUK (2024) Embodied Carbon Data for Timber Products — new softwood A1-A3: 107 kgCO2e/m3
  De Wolf, Brütting & Fivet (2020) — reclaimed reprocessing ~10-40 kgCO2e/m3
  Bergman & Bowe (2010) USDA FPL-GTR-177 — reclaimed GWP ~1/3 of virgin
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
C_NS     = "#61788C"   # blue  — new stock
C_RS     = "#F2994B"   # orange — reclaimed
C_ACCENT = "#D9653B"
C_MUTED  = "#9CA5A6"
BG       = "#FFFFFF"

fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

categories  = ["New structural\ntimber (softwood)", "Reclaimed\ntimber"]
fossil_vals = [107, 20]
colors      = [C_NS, C_RS]

bars = ax.bar(categories, fossil_vals, color=colors, width=0.45,
              edgecolor="white", linewidth=1.2, zorder=3, alpha=0.92)

# Value labels on bars
for bar, val in zip(bars, fossil_vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            val + 2.5,
            f"{val} kgCO₂e/m³",
            ha="center", va="bottom", fontsize=10,
            fontweight="bold", color=bar.get_facecolor())

# Savings annotation — bottom anchor raised above the "20 kgCO₂e/m³" value label
arrow_bot = fossil_vals[1] + 8
ax.annotate("",
    xy=(1, arrow_bot),
    xytext=(1, fossil_vals[0]),
    arrowprops=dict(arrowstyle="<->", color=C_ACCENT, lw=1.8))
ax.text(1.28, (fossil_vals[0] + arrow_bot) / 2,
        f"−{fossil_vals[0] - fossil_vals[1]} kgCO₂e/m³\n(~81% reduction)",
        ha="left", va="center", fontsize=9,
        color=C_ACCENT, fontweight="bold")

ax.set_ylabel("A1–A3 production carbon  (kgCO₂e / m³)", fontsize=10)
ax.set_ylim(0, 140)
ax.set_xlim(-0.55, 1.8)
ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
ax.tick_params(labelsize=10)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout(pad=1.2)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = Path(__file__).resolve().parent
for fmt in ["pdf", "png"]:
    out = out_dir / f"fig_embodied_carbon.{fmt}"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close()
