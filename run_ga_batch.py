"""
run_ga_batch.py — Unattended multi-run GA batch runner.

Runs the full CMA-ES optimizer N times on a chosen stock dataset, exporting
each run to its own timestamped directory. Normalization bounds and the GNN
bundle are computed/loaded once and shared across all runs. Seeds increment
per run so each exploration is independent.

Usage (from repo root, with venv active):
    python run_ga_batch.py

Edit the CONFIG block below to change scenario, run count, or GA parameters.
"""

import importlib
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no windows, figures saved to file only
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG — edit here before each batch
# ---------------------------------------------------------------------------

TRAINING_SCENARIO = "A"          # "A", "B", or "new"
N_RUNS            = 1            # number of independent GA runs
BASE_SEED         = 42           # seeds will be 42, 43, 44, 45, 46

MODEL_PREFIX = "ID20260516_182257_LR1e-04_EP200_BS64_PW2.5_ROC0.863"
USE_GNN      = True              # set False to skip GNN (cost+reuse only)

GA_CONFIG = {
    "fitness_weights":    {"omega_1": 1.0, "omega_2": 1.0},
    "new_stock_max_uses": 10,
    "min_reuse_fraction": 0.0,
    "penalty_fitness":    1e6,
    "use_one_time_bounds":   True,
    "bounds_probe_attempts": 40,
    "w_structural_start": 2.0,   # high early — steers away from structural holes
    "w_structural_end":   0.8,   # relaxes as search converges
    "max_structural_infeas": 1.0,  # hard floor: infeas > 0.60 → penalty regardless
    "use_gnn":            USE_GNN,
}

CMAES_POPSIZE      = 30
CMAES_GENERATIONS  = 250
CMAES_SIGMA_INIT   = 0.25
CMAES_SIGMA_MIN    = 1e-8
CMAES_STAGNATION   = 30
CMAES_LOG_EVERY    = 10         # print progress every N generations

# ---------------------------------------------------------------------------
# Repo path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
for candidate in [REPO_ROOT, *REPO_ROOT.parents]:
    if (candidate / "config.py").exists():
        REPO_ROOT = candidate
        break

for p in [REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "workflows"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config
import c00_headquarter_params as c11_params
from workflows import c22_stage_geometry             as stage_geometry
from workflows import c24_stage_feasibility          as stage_feas
from workflows import c25_stage_cost_matrix          as stage_cost
from workflows import c26_stage_MILP                 as stage_milp
from workflows import c27_stage_GNN                  as stage_gnn
from workflows import c28_stage_fitness_score        as stage_fitness
from workflows import c28_stage_normalization_bounds as stage_bounds
from workflows import c23_ga_evaluator               as ga_eval
from workflows import c23_ga_analysis_export         as ga_ae
from workflows import c23_ga_algorithm               as ga_algo
from c21_surrogate_io import load_surrogate_bundle

for _mod in [stage_geometry, stage_feas, stage_cost, stage_milp,
             stage_gnn, stage_fitness, stage_bounds, ga_eval, ga_ae, ga_algo]:
    importlib.reload(_mod)

# ---------------------------------------------------------------------------
# Load stock + search space (once)
# ---------------------------------------------------------------------------

stock_file = config.TIMBER_STOCK_PATH / f"complete_timber_{TRAINING_SCENARIO}.csv"
if not stock_file.exists():
    raise FileNotFoundError(f"Stock CSV not found: {stock_file}")

df_input_stock = None
for opts in [{"sep": ";", "encoding": "utf-8"}, {"sep": ",", "encoding": "utf-8"},
             {"sep": ";", "encoding": "latin1"}, {"sep": ",", "encoding": "latin1"}]:
    try:
        _df = pd.read_csv(stock_file, **opts)
        if _df.shape[1] > 1:
            df_input_stock = _df
            break
    except Exception:
        pass

if df_input_stock is None:
    raise ValueError(f"Could not parse {stock_file}")

df_input_stock.columns = df_input_stock.columns.str.strip()
ns_n = int((df_input_stock["State"] == 0).sum()) if "State" in df_input_stock.columns else "?"
rs_n = int((df_input_stock["State"] == 1).sum()) if "State" in df_input_stock.columns else "?"
print(f"Stock {TRAINING_SCENARIO}: {len(df_input_stock)} elements  NS={ns_n}  RS={rs_n}")

PREPARED_GNN_STOCK = stage_gnn.prepare_stock_for_gnn(df_input_stock)

json_path = config.DATA_IO_PATH / f"search_space_{c11_params.GRID}.json"
with open(json_path, "r", encoding="utf-8") as f:
    optimizer_search_space = json.load(f)
print(f"Search space: {len(optimizer_search_space)} variables")

def _ss_to_bounds(ss: dict) -> dict:
    out = {}
    for k, v in ss.items():
        if v["type"] == "discrete":
            out[k] = (float(min(v["options"])), float(max(v["options"])))
        else:
            out[k] = (float(v["min"]), float(v["max"]))
    return out

es_search_space = _ss_to_bounds(optimizer_search_space)

# ---------------------------------------------------------------------------
# One-time normalization bounds (shared across all runs — same stock)
# ---------------------------------------------------------------------------

print("\nComputing one-time normalization bounds...")
random.seed(BASE_SEED); np.random.seed(BASE_SEED)

FIXED_NORM, BOUNDS_SOURCE = ga_eval._compute_one_time_normalization_constants(
    search_space = optimizer_search_space,
    df_stock     = df_input_stock,
    config_dict  = GA_CONFIG,
)
print(f"Bounds: {FIXED_NORM}  source: {BOUNDS_SOURCE}")

# ---------------------------------------------------------------------------
# Load GNN bundle once (if USE_GNN)
# ---------------------------------------------------------------------------

SURROGATE_BUNDLE = None
if USE_GNN and MODEL_PREFIX:
    print(f"\nLoading GNN bundle: {MODEL_PREFIX}")
    SURROGATE_BUNDLE = load_surrogate_bundle(prefix_sm=MODEL_PREFIX)
    print("Bundle loaded.")

# ---------------------------------------------------------------------------
# Batch loop
# ---------------------------------------------------------------------------

run_summaries = []

print(f"\n{'='*65}")
print(f"BATCH: Stock={TRAINING_SCENARIO}  N_RUNS={N_RUNS}  "
      f"GNN={'Yes' if USE_GNN else 'No'}")
print(f"Budget per run: popsize={CMAES_POPSIZE} × gens={CMAES_GENERATIONS} "
      f"= {CMAES_POPSIZE * CMAES_GENERATIONS:,} evaluations")
print(f"GA_CONFIG:")
for k, v in GA_CONFIG.items():
    print(f"  {k}: {v}")
print(f"{'='*65}\n")

for run_idx in range(N_RUNS):
    seed = BASE_SEED + run_idx
    random.seed(seed); np.random.seed(seed)

    print(f"\n{'─'*65}")
    print(f"RUN {run_idx + 1}/{N_RUNS}   seed={seed}")
    print(f"{'─'*65}")
    t_run_start = time.time()

    _base_fn = ga_algo.make_evaluate_fn(
        evaluate_fn_raw      = ga_eval.evaluate_design_candidate,
        df_stock             = df_input_stock,
        fixed_norm_constants = FIXED_NORM,
        config_dict          = GA_CONFIG,
        bundle               = SURROGATE_BUNDLE,
        prepared_gnn_stock   = PREPARED_GNN_STOCK,
        verbose              = False,
    )

    _run_label = f"[Run {run_idx + 1}/{N_RUNS}]"

    def evaluate_fn(params, generation=0, max_generations=1,
                    _fn=_base_fn, _label=_run_label):
        t0 = time.time()
        fitness, res = _fn(params, generation, max_generations)
        elapsed = time.time() - t0
        if res:
            status = res.get("status",        "?")
            milp   = res.get("milp_status",   "?")
            gnn    = res.get("gnn_feasibility")
            cost   = res.get("total_cost",     float("nan"))
            reuse  = res.get("reuse_fraction", float("nan"))
            w4     = res.get("w_structural",   float("nan"))
            reason = res.get("reason",         "")
            gnn_s  = f"{gnn:.3f}" if gnn is not None else " n/a"
            reason_tag = ""
            if status == "PENALIZED" and reason:
                if "structural infeasibility" in reason:
                    reason_tag = " [STRUCT_FLOOR]"
                elif "MILP" in reason:
                    reason_tag = " [MILP]"
                else:
                    reason_tag = f" [{reason[:20]}]"
            print(f"  {_label} gen={generation:>3} | {elapsed:>4.1f}s | {status}{reason_tag} | "
                  f"MILP={milp} | GNN={gnn_s} | "
                  f"cost={cost:>7.2f} | reuse={reuse:.3f} | "
                  f"ω4={w4:.2f} | fit={fitness:.4f}")
        else:
            print(f"  {_label} gen={generation:>3} | {elapsed:>4.1f}s | fit={fitness:.4f}  (no result)")
        return fitness, res

    es = ga_algo.CMAEvolutionStrategy(
        search_space = es_search_space,
        evaluate_fn  = evaluate_fn,
        config       = ga_algo.CMAESConfig(
            popsize          = CMAES_POPSIZE,
            n_generations    = CMAES_GENERATIONS,
            sigma_init       = CMAES_SIGMA_INIT,
            sigma_min        = CMAES_SIGMA_MIN,
            stagnation_limit = CMAES_STAGNATION,
            log_every        = CMAES_LOG_EVERY,
        ),
        seed = seed,
    )

    try:
        result = es.run()
    except Exception as exc:
        elapsed_min = (time.time() - t_run_start) / 60
        print(f"\nRun {run_idx + 1} CRASHED after {elapsed_min:.1f} min: {exc}")
        plt.close("all")
        run_summaries.append({
            "run": run_idx + 1, "seed": seed, "fitness": float("nan"),
            "cost": float("nan"), "reuse": float("nan"), "gnn_feas": None,
            "elapsed_min": elapsed_min, "export": "CRASHED",
        })
        continue

    elapsed_min = (time.time() - t_run_start) / 60
    best        = result["best_individual"]
    best_eval   = result["best_eval_result"] or {}
    fitness     = float(best.fitness)
    cost        = float(best_eval.get("total_cost", 0))
    reuse       = float(best_eval.get("reuse_fraction", 0))
    gnn_feas    = best_eval.get("gnn_feasibility")
    gnn_str     = f"{gnn_feas:.4f}" if gnn_feas is not None else "n/a"

    print(f"\nRun {run_idx + 1} complete  ({elapsed_min:.1f} min)")
    print(f"  Fitness:  {fitness:.4f}")
    print(f"  Cost:     {cost:.2f} kg CO2e")
    print(f"  Reuse:    {reuse:.3f}")
    print(f"  GNN feas: {gnn_str}")

    # Export
    export_dir = None
    try:
        analysis_out = ga_ae.run_analysis(
            result                 = result,
            fixed_norm_constants   = FIXED_NORM,
            optimizer_search_space = optimizer_search_space,
            stagnation_limit       = es.config.stagnation_limit,
        )
        export_out = ga_ae.run_export(
            analysis_out         = analysis_out,
            result               = result,
            ga_config            = GA_CONFIG,
            fixed_norm_constants = FIXED_NORM,
            model_prefix         = MODEL_PREFIX if USE_GNN else None,
            bounds_source_info   = BOUNDS_SOURCE,
            es                   = es,
            df_stock             = df_input_stock,
            stock_source_path    = stock_file,
            run_tag              = f"RUN{run_idx + 1}" if N_RUNS > 1 else None,
        )
        export_dir = export_out["export_dir"]
        print(f"  Exported: {export_dir.name}")
    except Exception as exc:
        print(f"  Export failed: {exc}")
    finally:
        plt.close("all")  # release figure memory between runs

    run_summaries.append({
        "run":      run_idx + 1,
        "seed":     seed,
        "fitness":  fitness,
        "cost":     cost,
        "reuse":    reuse,
        "gnn_feas": gnn_feas,
        "elapsed_min": elapsed_min,
        "export":   export_dir.name if export_dir else "FAILED",
    })

# ---------------------------------------------------------------------------
# Final summary table
# ---------------------------------------------------------------------------

print(f"\n{'='*65}")
print(f"BATCH COMPLETE — Stock {TRAINING_SCENARIO}  ({N_RUNS} runs)")
print(f"{'='*65}")
print(f"  {'Run':>4}  {'Seed':>6}  {'Fitness':>10}  {'Cost':>8}  {'Reuse':>7}  {'GNN':>7}  {'Min':>6}")
print(f"  {'─'*4}  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*6}")
for s in run_summaries:
    gnn_str = f"{s['gnn_feas']:.4f}" if s["gnn_feas"] is not None else "   n/a"
    print(f"  {s['run']:>4}  {s['seed']:>6}  {s['fitness']:>10.4f}  "
          f"{s['cost']:>8.2f}  {s['reuse']:>7.4f}  {gnn_str:>7}  {s['elapsed_min']:>6.1f}")

fitnesses = [s["fitness"] for s in run_summaries]
print(f"\n  Best fitness:  {min(fitnesses):.4f}  (run {fitnesses.index(min(fitnesses)) + 1})")
print(f"  Mean fitness:  {np.mean(fitnesses):.4f}")
print(f"  Std  fitness:  {np.std(fitnesses):.4f}")
print(f"  Total time:    {sum(s['elapsed_min'] for s in run_summaries):.1f} min")
