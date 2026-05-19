"""
c23_ga_analysis_export — GA run analysis and export.

Two entry points:
    run_analysis(result, ga_config, fixed_norm_constants, optimizer_search_space,
                 stagnation_limit=20) -> dict
        Creates and displays all analysis figures. Returns a figures dict.

    run_export(analysis_out, result, ga_config, fixed_norm_constants,
               model_prefix, bounds_source_info="unknown", es=None) -> dict
        Saves figures, JSONs, CSVs, and a human-readable report to
        config.GA_DATA_PATH / {artifact_stem}. Returns paths dict.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import config
from config import PLOT_COLORS as C, PLOT_STYLE as S
import c00_headquarter_params as _lca
import workflows.c24_stage_feasibility as _feas


# =============================================================================
# ANALYSIS
# =============================================================================

def run_analysis(
    result:                 dict,
    fixed_norm_constants:   dict,
    optimizer_search_space: dict,
    stagnation_limit:       int = 20,
) -> dict:
    """
    Create and display all GA analysis figures.

    Parameters
    ----------
    result                : dict returned by EvolutionStrategy.run()
    fixed_norm_constants  : FIXED_NORMALIZATION_CONSTANTS dict
    optimizer_search_space: search space dict (used for parameter normalisation)
    stagnation_limit      : ES stagnation patience for the reference line in Fig 1c

    Returns
    -------
    dict with keys: fig_conv_fitness, fig_conv_sigma, fig_conv_stagnation,
                    fig_best_design, fig_params, fig_top_k (None if unavailable)
    """
    history    = result["history"]
    best       = result["best_individual"]
    best_eval  = result["best_eval_result"] or {}
    n_evals    = result["n_evals"]
    n_gens     = result["n_generations"]
    n_restarts = result["n_restarts"]
    top_k      = result.get("top_k", [])
    norm       = fixed_norm_constants

    plt.rcParams.update({
        "figure.dpi":        S["dpi"],
        "axes.grid":         True,
        "grid.alpha":        S["grid_alpha"],
        "grid.color":        C["neutral"],
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    C["black"],
        "axes.labelcolor":   C["black"],
        "xtick.color":       C["black"],
        "ytick.color":       C["black"],
        "text.color":        C["black"],
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.titleweight":  "bold",
        "lines.linewidth":   S["line_width"],
        "lines.markersize":  S["marker_size"],
    })

    gens       = [h["generation"]    for h in history]
    best_fit   = [h["best_fitness"]  for h in history]
    mean_fit   = [h["mean_fitness"]  for h in history]
    worst_fit  = [h["worst_fitness"] for h in history]
    best_ever  = [h["best_ever"]     for h in history]
    mean_sigma = [h["mean_sigma"]    for h in history]
    stagnation = [h["stagnation"]    for h in history]

    # ── Fig 1a: Fitness convergence ───────────────────────────────────────────
    fig_conv_fitness, ax = plt.subplots(figsize=S["figsize_medium"])
    fig_conv_fitness.suptitle("Figure 1a — Fitness Convergence",
                               fontweight="bold", fontsize=13)
    # Clip penalty values (1e6) from y-axis so real fitness variation is visible
    penalty_threshold = 1e5
    real_fits = [f for f in best_fit + worst_fit + best_ever if f is not None and abs(f) < penalty_threshold]
    y_min = min(real_fits) if real_fits else -1
    y_max = max(real_fits) if real_fits else 1
    y_pad = max(abs(y_max - y_min) * 0.1, 0.05)

    ax.fill_between(gens, best_fit, worst_fit, alpha=0.15, color=C["primary"],
                    label="Population range")
    ax.plot(gens, best_ever, color=C["primary"],   lw=S["line_width"], label="Best ever")
    ax.plot(gens, mean_fit,  color=C["secondary"], lw=S["line_width"],
            linestyle="--", label="Generation mean")
    ax.plot(gens, best_fit,  color=C["accent"],    lw=1.2,
            linestyle=":",  label="Generation best")
    # Only show restart markers when actual restarts occurred
    if n_restarts > 0:
        _restart_labeled = False
        for h in history:
            if h["stagnation"] == 0 and h["generation"] > 1:
                ax.axvline(h["generation"], color=C["danger"], lw=1.0,
                           linestyle="--", alpha=0.6,
                           label="Restart" if not _restart_labeled else "_nolegend_")
                _restart_labeled = True
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (lower = better)")
    ax.set_title("Fitness over generations")
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    print("fig_conv_fitness ready")

    # ── Fig 1b: Self-adaptive step size ──────────────────────────────────────
    fig_conv_sigma, ax = plt.subplots(figsize=S["figsize_small"])
    fig_conv_sigma.suptitle("Figure 1b — Self-Adaptive Step Size",
                             fontweight="bold", fontsize=13)
    ax.plot(gens, mean_sigma, color=C["primary"], lw=S["line_width"])
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean σ")
    ax.set_title("Population mean step size — spikes mark restarts")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.show()
    print("fig_conv_sigma ready")

    # ── Fig 1c: Stagnation counter ────────────────────────────────────────────
    fig_conv_stagnation, ax = plt.subplots(figsize=S["figsize_small"])
    fig_conv_stagnation.suptitle("Figure 1c — Stagnation per Generation",
                                  fontweight="bold", fontsize=13)
    ax.bar(gens, stagnation, color=C["secondary"], width=0.8, edgecolor="none")
    ax.axhline(stagnation_limit, color=C["danger"], lw=1.5, linestyle="--",
               label=f"Stagnation limit = {stagnation_limit}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Stagnation counter")
    ax.set_title("Resets to 0 on improvement or restart")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    print("fig_conv_stagnation ready")

    # ── Fig 2: Best design breakdown ─────────────────────────────────────────
    best_cost    = float(best_eval.get("total_cost",     0))
    best_reuse   = float(best_eval.get("reuse_fraction", 0))
    best_waste   = float(best_eval.get("waste_total",    0))
    best_gnn     = float(best_eval.get("gnn_feasibility", 1.0) or 1.0)
    best_fitness = float(best.fitness)
    fr           = best_eval.get("fitness_result") or {}

    fig_best_design, axes = plt.subplots(1, 2, figsize=S["figsize_medium"])
    fig_best_design.suptitle("Figure 2 — Best Design Breakdown",
                              fontweight="bold", fontsize=13)

    component_labels = ["Cost\n(ω1·cost_norm)", "Reuse\n(ω2·reuse_norm)", "Structural\n(ω4·infeas.)"]
    component_values = [
        float(fr.get("cost_norm",               0)),
        float(fr.get("reuse_norm",              0)),
        float(fr.get("structural_infeasibility", 0)),
    ]
    ax = axes[0]
    bars = ax.bar(component_labels, component_values,
                  color=[C["primary"], C["accent"], C["secondary"], C["danger"]],
                  edgecolor=C["black"], linewidth=0.5)
    for bar, val in zip(bars, component_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Weighted component value")
    ax.set_title(f"Fitness Components  (total = {best_fitness:.4f})")
    ax.set_ylim(0, max(component_values) * 1.2 if max(component_values) > 0 else 1)

    ax = axes[1]
    ax.axis("off")
    metrics_text = [
        ("Best fitness",      f"{best_fitness:.4f}"),
        ("Total cost",        f"{best_cost:.2f}"),
        ("Reuse rate",        f"{best_reuse * 100:.1f}%"),
        ("Waste total",       f"{best_waste:.4f}"),
        ("GNN feasibility",   f"{best_gnn * 100:.1f}%"),
        ("",                  ""),
        ("Generations run",   str(n_gens)),
        ("Total evaluations", str(n_evals)),
        ("Restarts",          str(n_restarts)),
        ("",                  ""),
        ("C_max (norm)",      f"{norm.get('C_max', '?')}"),
        ("R_max (norm)",      f"{norm.get('R_max', '?')}"),
    ]
    y = 0.95
    for label, value in metrics_text:
        if label == "":
            y -= 0.04
            continue
        ax.text(0.05, y, label + ":", fontsize=10, color=C["black"],
                transform=ax.transAxes, va="top")
        ax.text(0.60, y, value, fontsize=10, color=C["primary"],
                transform=ax.transAxes, va="top", fontweight="bold")
        y -= 0.07
    ax.set_title("Best Design Metrics")
    plt.tight_layout()
    plt.show()
    print("fig_best_design ready")

    # ── Fig 3: Best design parameters ────────────────────────────────────────
    param_names  = list(best.params.keys())
    param_values = list(best.params.values())
    # CMA-ES stores one global sigma; replicate it across all parameters for the bar chart
    raw_sigma    = list(best.sigma)
    sigma_values = raw_sigma * len(param_names) if len(raw_sigma) == 1 else raw_sigma

    def _get_bound(entry):
        if entry["type"] == "discrete":
            return float(min(entry["options"])), float(max(entry["options"]))
        return float(entry["min"]), float(entry["max"])

    try:
        bounds_lo   = np.array([_get_bound(optimizer_search_space[k])[0] for k in param_names])
        bounds_hi   = np.array([_get_bound(optimizer_search_space[k])[1] for k in param_names])
        param_range = bounds_hi - bounds_lo
        param_norm  = ((np.array(param_values) - bounds_lo)
                       / np.where(param_range > 0, param_range, 1))
    except Exception:
        param_norm = np.array(param_values)

    n_params = len(param_names)
    x_pos    = np.arange(n_params)

    fig_params, axes = plt.subplots(2, 1, figsize=(S["figsize_large"][0], 8))
    fig_params.suptitle("Figure 3 — Best Design Parameters",
                         fontweight="bold", fontsize=13)

    ax = axes[0]
    ax.bar(x_pos, param_norm, color=C["primary"], edgecolor="none", width=0.8)
    ax.axhline(0.5, color=C["secondary"], lw=1.2, linestyle="--",
               label="Midpoint of search space")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(param_names, rotation=90, fontsize=7)
    ax.set_ylabel("Normalised value [0=min, 1=max]")
    ax.set_title("Parameter Values (normalised to search space bounds)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x_pos, sigma_values, color=C["secondary"], edgecolor="none", width=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(param_names, rotation=90, fontsize=7)
    ax.set_ylabel("σ (step size at convergence)")
    sigma_title = (
        "Final Step Size (CMA-ES global σ — same value for all parameters)"
        if len(raw_sigma) == 1
        else "Final Step Sizes — small σ = converged, large σ = still exploring"
    )
    ax.set_title(sigma_title)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.show()
    print("fig_params ready")

    # ── Fig 4: Top-k designs comparison ──────────────────────────────────────
    fig_top_k = None
    if top_k:
        n_show     = len(top_k)
        ranks      = [f"#{i+1}" for i in range(n_show)]
        fitnesses  = [ind.fitness for ind in top_k]
        gnn_scores = [
            float((ind.eval_result or {}).get("gnn_feasibility", 0) or 0)
            for ind in top_k
        ]
        costs      = [
            float((ind.eval_result or {}).get("total_cost",  0) or 0)
            for ind in top_k
        ]
        reuse_rates = [
            float((ind.eval_result or {}).get("reuse_fraction", 0) or 0)
            for ind in top_k
        ]
        gens_found = [ind.generation for ind in top_k]

        fig_top_k, axes = plt.subplots(1, 3, figsize=(S["figsize_large"][0], 5))
        fig_top_k.suptitle(f"Figure 4 — Top-{n_show} Designs (FEM candidate shortlist)",
                            fontweight="bold", fontsize=13)

        y_pos = np.arange(n_show)[::-1]  # rank 1 at top

        ax = axes[0]
        bars = ax.barh(y_pos, fitnesses, color=C["primary"], edgecolor="none", height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ranks, fontsize=9)
        ax.set_xlabel("Fitness (lower = better)")
        ax.set_title("Fitness by rank")
        for bar, v in zip(bars, fitnesses):
            ax.text(bar.get_width() + max(fitnesses) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}", va="center", fontsize=8)

        ax = axes[1]
        bars = ax.barh(y_pos, gnn_scores, color=C["secondary"], edgecolor="none", height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ranks, fontsize=9)
        ax.set_xlabel("GNN feasibility (higher = safer)")
        ax.set_title("GNN feasibility by rank")
        ax.set_xlim(0, 1.05)
        for bar, v in zip(bars, gnn_scores):
            ax.text(bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8)

        ax = axes[2]
        sc = ax.scatter(costs, gnn_scores,
                        c=fitnesses, cmap="viridis_r",
                        s=60, edgecolors=C["black"], linewidths=0.4, zorder=3)
        for i, (cx, cy) in enumerate(zip(costs, gnn_scores)):
            ax.annotate(f"#{i+1}", (cx, cy),
                        textcoords="offset points", xytext=(5, 3), fontsize=7)
        plt.colorbar(sc, ax=ax, label="Fitness")
        ax.set_xlabel("Total cost (kg CO₂e)")
        ax.set_ylabel("GNN feasibility")
        ax.set_title("Cost vs GNN feasibility")

        plt.tight_layout()
        plt.show()
        print("fig_top_k ready")

    # ── Fig 5: Per-generation component breakdown ─────────────────────────────
    fig_components = None
    has_components = any("best_cost_norm" in h for h in history)
    if has_components:
        cost_curve    = [h.get("best_cost_norm",         float("nan")) for h in history]
        reuse_curve   = [h.get("best_reuse_norm",        float("nan")) for h in history]
        struct_curve  = [h.get("best_structural_infeas", float("nan")) for h in history]
        w4_curve      = [h.get("w_structural",           float("nan")) for h in history]
        mean_reuse    = [h.get("mean_reuse",             float("nan")) for h in history]
        mean_gnn      = [h.get("mean_gnn",               float("nan")) for h in history]
        n_penalty     = [h.get("n_penalty",              0)            for h in history]

        fig_components, axes = plt.subplots(2, 2, figsize=(S["figsize_large"][0], 9))
        fig_components.suptitle("Figure 5 — Per-Generation Component Breakdown",
                                 fontweight="bold", fontsize=13)

        # Top-left: fitness components of best-in-generation
        ax = axes[0, 0]
        ax.plot(gens, cost_curve,   color=C["primary"],   lw=S["line_width"], label="cost_norm (ω1·cost)")
        ax.plot(gens, reuse_curve,  color=C["accent"],    lw=S["line_width"], label="reuse_norm (ω2·reuse, subtracted)")
        ax.plot(gens, struct_curve, color=C["danger"],    lw=S["line_width"], label="structural_infeas (ω4·infeas)")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Component value")
        ax.set_title("Fitness components — best-in-generation")
        ax.legend(fontsize=8)

        # Top-right: ω4 curriculum
        ax = axes[0, 1]
        ax.plot(gens, w4_curve, color=C["danger"], lw=S["line_width"])
        ax.set_xlabel("Generation")
        ax.set_ylabel("ω4")
        ax.set_title("Structural penalty weight (ω4) over generations")
        ax.set_ylim(0, 1)

        # Bottom-left: population reuse + GNN trends
        ax = axes[1, 0]
        ax.plot(gens, mean_reuse, color=C["accent"],   lw=S["line_width"], label="mean reuse fraction (population)")
        ax2 = ax.twinx()
        ax2.plot(gens, mean_gnn, color=C["secondary"], lw=S["line_width"],
                 linestyle="--", label="mean GNN feasibility (population)")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Mean reuse fraction", color=C["accent"])
        ax2.set_ylabel("Mean GNN feasibility", color=C["secondary"])
        ax.set_title("Population reuse & structural quality trend")
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)

        # Bottom-right: penalty count
        ax = axes[1, 1]
        ax.bar(gens, n_penalty, color=C["danger"], width=0.8, edgecolor="none", alpha=0.7)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Individuals with penalty fitness")
        ax.set_title("Failed evaluations per generation (in µ survivors)")

        plt.tight_layout()
        plt.show()
        print("fig_components ready")

    # ── Summary print ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("GA RUN SUMMARY")
    print("=" * 65)
    print(f"  Generations completed:  {n_gens}")
    print(f"  Total evaluations:      {n_evals}")
    print(f"  Restarts triggered:     {n_restarts}")
    print()
    print(f"  Best fitness:           {best_fitness:.6f}")
    print(f"  Best total cost:        {best_cost:.2f}")
    print(f"  Best reuse rate:        {best_reuse * 100:.1f}%")
    print(f"  Best waste total:       {best_waste:.4f}")
    print(f"  GNN feasibility:        {best_gnn * 100:.1f}%")
    print()
    print(f"  MILP status:            {best_eval.get('milp_status', 'n/a')}")
    print(f"  Unsafe members:         "
          f"{len(best_eval.get('gnn_unsafe_members') or [])} / 120")
    print()
    print(f"  Normalisation constants:  C_max={norm.get('C_max')}  R_max={norm.get('R_max')}")
    if top_k:
        print()
        print(f"  Top-{len(top_k)} designs (for FEM verification):")
        for i, ind in enumerate(top_k):
            ev = ind.eval_result or {}
            print(f"    #{i+1:2d}  fitness={ind.fitness:.4f}  "
                  f"GNN={float(ev.get('gnn_feasibility', 0) or 0):.3f}  "
                  f"cost={float(ev.get('total_cost', 0) or 0):.4f}  "
                  f"gen={ind.generation}")
    print("=" * 65)

    return {
        "fig_conv_fitness":    fig_conv_fitness,
        "fig_conv_sigma":      fig_conv_sigma,
        "fig_conv_stagnation": fig_conv_stagnation,
        "fig_best_design":     fig_best_design,
        "fig_params":          fig_params,
        "fig_top_k":           fig_top_k,
        "fig_components":      fig_components,
        "top_k":               top_k,
    }


# =============================================================================
# EXPORT
# =============================================================================

def run_export(
    analysis_out:         dict,
    result:               dict,
    ga_config:            dict,
    fixed_norm_constants: dict,
    model_prefix:         str   = None,
    bounds_source_info:   object = "unknown",
    es:                   object = None,
    df_stock:             object = None,   # pd.DataFrame — stock used by this run
    stock_source_path:    object = None,   # path the stock was loaded from (for report)
    run_tag:              str   = None,    # e.g. "RUN1" — inserted before GEN in artifact name
) -> dict:
    """
    Save all figures, metrics, MILP assignment, and a human-readable report.

    Parameters
    ----------
    analysis_out        : dict returned by run_analysis()
    result              : dict returned by EvolutionStrategy.run()
    ga_config           : GA_CONFIG pipeline dict
    fixed_norm_constants: FIXED_NORMALIZATION_CONSTANTS dict
    model_prefix        : surrogate model artifact stem string
    bounds_source_info  : BOUNDS_SOURCE_INFO dict or string (optional)
    es                  : EvolutionStrategy instance (optional, for ESConfig details)

    Returns
    -------
    dict with keys: export_dir (Path), artifact_stem (str), all_files (list[Path])
    """
    def _to_builtin(v):
        if hasattr(v, "item"):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    best      = result["best_individual"]
    best_eval = result["best_eval_result"] or {}
    top_k     = result.get("top_k") or analysis_out.get("top_k") or []
    n_gens    = result["n_generations"]
    n_evals   = result["n_evals"]
    norm      = fixed_norm_constants

    ts            = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_f        = f"{best.fitness:.4f}".replace(".", "_")
    # Derive dataset label from stock filename: "complete_timber_A.csv" → "A"
    if stock_source_path is not None:
        dataset_label = Path(str(stock_source_path)).stem.split("_")[-1]
    else:
        dataset_label = "?"
    run_part      = f"_RUN{run_tag}" if run_tag else ""
    artifact_stem = f"GA_{dataset_label}_{ts}{run_part}_GEN{n_gens}_EVAL{n_evals}_F{best_f}"
    export_dir    = config.GA_DATA_PATH / artifact_stem
    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"Export directory: {export_dir}")

    # ── Figures ───────────────────────────────────────────────────────────────
    fig_map = {
        "fig_conv_fitness":    "fig1a_fitness_convergence",
        "fig_conv_sigma":      "fig1b_sigma",
        "fig_conv_stagnation": "fig1c_stagnation",
        "fig_best_design":     "fig2_best_design",
        "fig_params":          "fig3_parameters",
        "fig_top_k":           "fig4_top_k_comparison",
        "fig_components":      "fig5_component_breakdown",
    }
    for var_name, stem in fig_map.items():
        fig = analysis_out.get(var_name)
        if fig is not None:
            out = export_dir / f"{artifact_stem}_{stem}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Saved: {out.name}")
        else:
            print(f"  Skipped: {var_name} not in analysis_out")

    # ── Best design JSON ──────────────────────────────────────────────────────
    best_design_payload = {
        "fitness":            float(best.fitness),
        "generation":         int(best.generation),
        "total_cost":         float(best_eval.get("total_cost",  0)),
        "reuse_rate":         float(best_eval.get("reuse_fraction", 0)),
        "waste_total":        float(best_eval.get("waste_total", 0)),
        "gnn_feasibility":    float(best_eval.get("gnn_feasibility", 1.0) or 1.0),
        "gnn_unsafe_members": list(best_eval.get("gnn_unsafe_members") or []),
        "milp_status":        str(best_eval.get("milp_status", "n/a")),
        "w_structural":       float(best_eval.get("w_structural", 0)),
        "params":             {k: float(v) for k, v in best.params.items()},
        "sigma":              [float(s) for s in best.sigma],
        "fitness_result":     {
            k: _to_builtin(v)
            for k, v in (best_eval.get("fitness_result") or {}).items()
            if not isinstance(v, pd.DataFrame)
        },
    }
    best_design_path = export_dir / f"{artifact_stem}_best_design.json"
    with open(best_design_path, "w", encoding="utf-8") as f:
        json.dump(best_design_payload, f, indent=2, default=str)
    print(f"  Saved: {best_design_path.name}")

    # ── History CSV ───────────────────────────────────────────────────────────
    history_path = export_dir / f"{artifact_stem}_history.csv"
    pd.DataFrame(result["history"]).to_csv(history_path, index=False)
    print(f"  Saved: {history_path.name}")

    # ── Stock dataset copy ────────────────────────────────────────────────────
    stock_path = None
    stock_info = {}
    if df_stock is not None:
        stock_path = export_dir / f"{artifact_stem}_stock.csv"
        df_stock.to_csv(stock_path, index=False, sep=";")
        n_total = len(df_stock)
        n_ns    = int((df_stock["State"] == 0).sum()) if "State" in df_stock.columns else "?"
        n_rs    = int((df_stock["State"] == 1).sum()) if "State" in df_stock.columns else "?"
        stock_info = {
            "file":       stock_path.name,
            "source":     str(stock_source_path) if stock_source_path else "unknown",
            "n_total":    n_total,
            "n_ns":       n_ns,
            "n_rs":       n_rs,
        }
        print(f"  Saved: {stock_path.name}  ({n_total} elements, NS={n_ns}, RS={n_rs})")

    # ── MILP topology helper (used by top-k export) ───────────────────────────
    def _enrich_milp(df_milp, df_edges):
        """Merge V1/V2 topology into a MILP result DataFrame."""
        if df_edges is None or df_edges.empty:
            return df_milp
        topo = df_edges[["edge_id", "V1", "V2"]] if "V1" in df_edges.columns else df_edges[["edge_id"]]
        out  = df_milp.merge(topo, on="edge_id", how="left")
        front = [c for c in ["edge_id", "V1", "V2"] if c in out.columns]
        rest  = [c for c in out.columns if c not in front]
        return out[front + rest]

    # ── Top-k designs ─────────────────────────────────────────────────────────
    if top_k:
        top_k_dir = export_dir / "top_k_designs"
        top_k_dir.mkdir(exist_ok=True)

        # Summary JSON (all ranks, all metadata)
        top_k_payload = []
        for rank, ind in enumerate(top_k, 1):
            ev = ind.eval_result or {}
            fr = ev.get("fitness_result") or {}
            entry = {
                "rank":              rank,
                "fitness":           float(ind.fitness),
                "generation":        int(ind.generation),
                "total_cost":        float(ev.get("total_cost",      0) or 0),
                "reuse_rate":            float(ev.get("reuse_fraction",  0) or 0),
                "waste_total":           float(ev.get("waste_total",     0) or 0),
                "gnn_feasibility":        float(ev.get("gnn_feasibility", 0) or 0),
                "n_unsafe_members":       len(ev.get("gnn_unsafe_members") or []),
                "gnn_unsafe_members":     list(ev.get("gnn_unsafe_members") or []),
                "milp_status":            str(ev.get("milp_status",        "n/a")),
                "w_structural":           float(ev.get("w_structural",      float("nan")) or float("nan")),
                "cost_norm":              _to_builtin(fr.get("cost_norm",                0)),
                "reuse_norm":             _to_builtin(fr.get("reuse_norm",               0)),
                "structural_infeasibility": _to_builtin(fr.get("structural_infeasibility", 0)),
                "structural_penalty":     _to_builtin(fr.get("structural_penalty",        0)),
                "params":             {k: float(v) for k, v in ind.params.items()},
            }
            top_k_payload.append(entry)

        summary_json = top_k_dir / f"{artifact_stem}_top{len(top_k)}_summary.json"
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(top_k_payload, f, indent=2, default=str)
        print(f"  Saved: top_k_designs/{summary_json.name}")

        # Summary CSV (flat, one row per design)
        summary_df = pd.DataFrame([
            {k: v for k, v in e.items() if k not in ("params", "gnn_unsafe_members")}
            for e in top_k_payload
        ])
        summary_csv = top_k_dir / f"{artifact_stem}_top{len(top_k)}_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"  Saved: top_k_designs/{summary_csv.name}")

        # Combined geometry + MILP assignment CSVs (one file per type, all ranks)
        vert_chunks: list[pd.DataFrame] = []
        edge_chunks: list[pd.DataFrame] = []

        for rank, ind in enumerate(top_k, 1):
            ev  = ind.eval_result or {}
            dfr = ev.get("df_results")

            df_v = ev.get("df_vertices")
            if df_v is not None and not df_v.empty:
                vert_chunks.append(df_v)

            if dfr is not None and not dfr.empty:
                edge_chunks.append(_enrich_milp(dfr.copy(), ev.get("df_edges")))

        def _concat_with_rank(chunks, keep_cols):
            records = []
            for rank, chunk in enumerate(chunks, 1):
                for row in chunk.to_dict(orient="records"):
                    rec = {"rank": rank}
                    for col in keep_cols:
                        if col in row:
                            rec[col] = row[col]
                    records.append(rec)
            return pd.DataFrame(records)

        if vert_chunks:
            verts_csv = top_k_dir / f"{artifact_stem}_top{len(top_k)}_vertices.csv"
            vert_cols = ["vertex_index", "layer", "attribute", "x", "y", "z"]
            _concat_with_rank(vert_chunks, vert_cols).to_csv(verts_csv, index=False)
            print(f"  Saved: top_k_designs/{verts_csv.name}")

        if edge_chunks:
            edges_csv = top_k_dir / f"{artifact_stem}_top{len(top_k)}_edges_assigned.csv"
            edge_cols = [c for c in edge_chunks[0].columns if c != "sample_id"]
            _concat_with_rank(edge_chunks, edge_cols).to_csv(edges_csv, index=False)
            print(f"  Saved: top_k_designs/{edges_csv.name}")

        print(f"  → {len(top_k)} designs consolidated into top_k_designs/")

    # ── Run config JSON ───────────────────────────────────────────────────────
    es_cfg = {}
    if es is not None:
        cfg = getattr(es, "config", None)
        if cfg is not None:
            es_cfg = {
                "mu":              cfg.mu,
                "lam":             cfg.lam,
                "n_generations":   cfg.n_generations,
                "sigma_init":      cfg.sigma_init,
                "sigma_min":       cfg.sigma_min,
                "tolfun":          getattr(cfg, "tolfun", None),
                "stagnation_limit":cfg.stagnation_limit,
                "n_restarts_max":  cfg.n_restarts_max,
                "top_k_size":      getattr(cfg, "top_k_size", None),
            }

    n_search_params = es.n_params if es is not None and hasattr(es, "n_params") else "?"

    config_payload = {
        "artifact_stem":           artifact_stem,
        "timestamp":               ts,
        "ga_config":               ga_config,
        "es_config":               es_cfg,
        "normalization_constants": {k: float(v) for k, v in norm.items()},
        "bounds_source":           bounds_source_info,
        "n_generations":           n_gens,
        "n_evals":                 n_evals,
        "n_restarts":              result["n_restarts"],
        "best_fitness":            float(best.fitness),
        "model_prefix":            model_prefix,
        "n_search_params":         n_search_params,
        "stock":                   stock_info,
    }
    config_path = export_dir / f"{artifact_stem}_run_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, default=str)
    print(f"  Saved: {config_path.name}")

    # ── Human-readable report ─────────────────────────────────────────────────
    fr = best_eval.get("fitness_result") or {}
    fw = ga_config["fitness_weights"]

    run_start   = result.get("start_time", "unknown")
    elapsed_s   = result.get("elapsed_seconds")
    elapsed_str = f"{elapsed_s / 60:.1f} min  ({elapsed_s} s)" if elapsed_s else "unknown"

    report_lines = [
        "GA OPTIMISATION RUN REPORT",
        "=" * 70,
        f"Artifact:              {artifact_stem}",
        f"Run started:           {run_start}",
        f"Export generated:      {ts}",
        f"Elapsed:               {elapsed_str}",
        f"Model prefix:          {model_prefix}",
        f"Search space vars:     {n_search_params}",
        f"Testing mode:          {ga_config.get('testing', False)}",
        "",
        "STOCK DATASET",
        "-" * 70,
    ]
    if stock_info:
        report_lines += [
            f"Source file:           {stock_info['source']}",
            f"Saved copy:            {stock_info['file']}",
            f"Total elements:        {stock_info['n_total']}  "
            f"(NS={stock_info['n_ns']}, RS={stock_info['n_rs']})",
        ]
    else:
        report_lines += ["  (no df_stock passed to run_export)"]

    report_lines += [
        "",
        "RUN STATISTICS",
        "-" * 70,
        f"Generations completed: {n_gens}",
        f"Total evaluations:     {n_evals}",
        f"Restarts triggered:    {result['n_restarts']}",
        "",
        "BEST DESIGN",
        "-" * 70,
        f"Fitness:               {best.fitness:.6f}",
        f"Found at generation:   {best.generation}",
        f"Total cost:            {best_design_payload['total_cost']:.4f}",
        f"Reuse rate:            {best_design_payload['reuse_rate']:.4f}",
        f"Waste total:           {best_design_payload['waste_total']:.4f}",
        f"GNN feasibility:       {best_design_payload['gnn_feasibility']:.4f}",
        f"Unsafe members:        {len(best_design_payload['gnn_unsafe_members'])} / 120",
        f"MILP status:           {best_design_payload['milp_status']}",
        f"omega_4 used:          {best_design_payload['w_structural']:.4f}",
        "",
        "FITNESS COMPONENTS",
        "-" * 70,
        f"Cost (cost_norm):         {fr.get('cost_norm',               'n/a')}",
        f"Reuse (reuse_norm):       {fr.get('reuse_norm',              'n/a')}",
        f"Structural (infeas.):     {fr.get('structural_infeasibility', 'n/a')}",
        f"Structural penalty:       {fr.get('structural_penalty',       'n/a')}",
        "",
        "NORMALISATION CONSTANTS",
        "-" * 70,
        f"C_max: {norm.get('C_max')}  R_max: {norm.get('R_max')}",
        f"Source: {bounds_source_info}",
        "",
        "LCA PARAMETERS",
        "-" * 70,
        f"A1-A3 embodied (new timber):     {_lca.IMPACT_FACTOR_A1_A3}  kg CO2e/kg  [EPD]",
        f"C1 deconstruction penalty:        {_lca.IMPACT_FACTOR_RECOVERED_C1}  kg CO2e/kg  [Bergman2010]",
        f"A5 preparation energy:            {_lca.ENERGY_PREP_A5}   kg CO2e/kg  [Bergman2010]",
        f"A5 saw energy:                    {_lca.ENERGY_SAW_A5}   kg CO2e/kg  [calc.]",
        f"C3-C4 offcut disposal:            {_lca.ENERGY_OFFCUT_FACTOR_C3_C4}   kg CO2e/kg  [Ecoinvent v3]",
        f"C2 waste transport distance:      {_lca.WASTE_TRANSPORT_DIST_KM} km              [EN15978]",
        f"Scarcity penalty (omega):         {_lca.SCARCITY_PENALTY}",
        "",
        "FEASIBILITY FILTER (c24)",
        "-" * 70,
        f"Force safety factor:              {_feas.FORCE_SAFETY_FACTOR}×   (forces multiplied before EC5 checks)",
        f"Max depth-to-length ratio:        L / {_feas.MAX_DEPTH_TO_LENGTH_RATIO}  (depth >= L/{_feas.MAX_DEPTH_TO_LENGTH_RATIO} per member)",
        f"Max slenderness (compression):    {_feas.MAX_SLENDERNESS}  (lambda = L/i_z, weak-axis i_z = min(w,d)/sqrt(12))",
        f"Max width-to-depth ratio:         1 / {_feas.MAX_WIDTH_DEPTH_RATIO}  (width >= depth/{_feas.MAX_WIDTH_DEPTH_RATIO})",
        f"Max oversize fraction:            {_feas.MAX_OVERSIZE_FRAC:.0%}  (stock may exceed slot length by this)",
        f"Design load:                      {_feas.LOAD_KN_PER_M2} kN/m²",
        f"GNN enabled:                      {'Yes' if ga_config.get('use_gnn', True) else 'No (structural term = 0)'}",
        "",
        "GA CONFIGURATION",
        "-" * 70,
        "  ".join(f"ω{i+1}={fw[k]}" for i, k in enumerate(sorted(fw))),
        f"Structural schedule:   ω4 {ga_config.get('w_structural_start', '?')} → "
        f"{ga_config.get('w_structural_end', '?')}",
        f"New stock max uses:    {ga_config.get('new_stock_max_uses')}",
        f"Min reuse fraction:    {ga_config.get('min_reuse_fraction', 'None')}",
        f"Penalty fitness:       {ga_config.get('penalty_fitness')}",
        f"GNN in fitness:        {'Yes (ω4=' + str(ga_config.get('w_structural_end', '?')) + ')' if ga_config.get('use_gnn', True) else 'No (ω4 zeroed)'}",
    ]

    if es_cfg:
        report_lines += [
            "",
            "ES CONFIGURATION",
            "-" * 70,
            f"μ={es_cfg['mu']}  λ={es_cfg['lam']}  generations={es_cfg['n_generations']}",
            f"sigma_init={es_cfg['sigma_init']}  sigma_min={es_cfg['sigma_min']}  "
            f"tolfun={es_cfg.get('tolfun', '?')}",
            f"stagnation_limit={es_cfg['stagnation_limit']}  "
            f"n_restarts_max={es_cfg.get('n_restarts_max', '?')}",
            f"top_k_size={es_cfg.get('top_k_size', '?')}",
        ]

    report_lines += ["", "BEST DESIGN PARAMETERS", "-" * 70]
    for k, v in best.params.items():
        report_lines.append(f"  {k:<40} {v:.6f}")

    if top_k:
        report_lines += ["", f"TOP-{len(top_k)} DESIGNS (FEM CANDIDATE SHORTLIST)", "-" * 70]
        report_lines.append(
            f"  {'Rank':<6} {'Fitness':>10} {'Cost':>10} {'Reuse%':>8} "
            f"{'GNN':>8} {'MILP':>12} {'Gen':>5}"
        )
        for rank, ind in enumerate(top_k, 1):
            ev = ind.eval_result or {}
            report_lines.append(
                f"  #{rank:<5} {ind.fitness:>10.4f} "
                f"{float(ev.get('total_cost',  0) or 0):>10.4f} "
                f"{float(ev.get('reuse_fraction', 0) or 0):>8.2f} "
                f"{float(ev.get('gnn_feasibility', 0) or 0):>8.3f} "
                f"{str(ev.get('milp_status', 'n/a')):>12} "
                f"{ind.generation:>5}"
            )
        report_lines += [
            "",
            "  Combined CSVs saved to top_k_designs/:",
            f"    {artifact_stem}_top{len(top_k)}_vertices.csv",
            f"    {artifact_stem}_top{len(top_k)}_edges_assigned.csv",
            f"    {artifact_stem}_top{len(top_k)}_summary.csv",
            f"    {artifact_stem}_top{len(top_k)}_summary.json",
        ]

    report_path = export_dir / f"{artifact_stem}_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {report_path.name}")

    # ── Summary ───────────────────────────────────────────────────────────────
    all_files = sorted(export_dir.glob("*"))
    print(f"\n{'='*65}")
    print("EXPORT COMPLETE")
    print(f"{'='*65}")
    print(f"  Directory: {export_dir}")
    print(f"\n  Files saved:")
    for fp in all_files:
        print(f"    {fp.name:<55} {fp.stat().st_size / 1024:6.1f} KB")

    return {
        "export_dir":    export_dir,
        "artifact_stem": artifact_stem,
        "all_files":     all_files,
    }
