"""
Local per-evaluation timing benchmark for c23_run_ga_batch.py.

Mirrors the batch's setup (scenario A, same GA_CONFIG, same GNN bundle) and
times a sample of evaluations on random candidates drawn from the search space.
CUDA is disabled so the timing reflects the DelftBlue `compute` (CPU) partition
rather than the local GPU.

Run:  python delftblue/benchmark_eval.py
"""
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path

# Force CPU so the estimate matches the cluster compute partition.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd

# --- Repo path setup (copied from c23_run_ga_batch.py) ---------------------
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
from workflows import c27_stage_GNN as stage_gnn
from workflows import c23_ga_evaluator as ga_eval
from workflows import c23_ga_algorithm as ga_algo
from c21_surrogate_io import load_surrogate_bundle

# --- Config (must match c23_run_ga_batch.py) -------------------------------
TRAINING_SCENARIO = "A"
BASE_SEED = 42
MODEL_PREFIX = "ID20260516_182257_LR1e-04_EP200_BS64_PW2.5_ROC0.863"
GA_CONFIG = {
    "fitness_weights":    {"omega_1": 1.0, "omega_2": 1.0},
    "new_stock_max_uses": 120,
    "min_reuse_fraction": 0.0,
    "penalty_fitness":    1e6,
    "use_one_time_bounds":   True,
    "bounds_probe_attempts": 40,
    "w_structural_start": 2.0,
    "w_structural_end":   0.8,
    "max_structural_infeas": 0.6,
    "use_gnn":            True,
}
CMAES_POPSIZE = 30
CMAES_GENERATIONS = 500
N_RUNS = 3
N_EVALS = 25          # sample size for timing
N_WARMUP = 3          # excluded (first calls pay lazy-import / cache costs)

# --- Load search space + bounds --------------------------------------------
json_path = config.DATA_IO_PATH / f"search_space_{c11_params.GRID}.json"
with open(json_path, "r", encoding="utf-8") as f:
    optimizer_search_space = json.load(f)

def _ss_to_bounds(ss):
    out = {}
    for k, v in ss.items():
        if v["type"] == "discrete":
            out[k] = (float(min(v["options"])), float(max(v["options"])))
        else:
            out[k] = (float(v["min"]), float(v["max"]))
    return out

es_search_space = _ss_to_bounds(optimizer_search_space)
print(f"Search space: {len(optimizer_search_space)} variables")

# --- Load GNN bundle (CPU) -------------------------------------------------
print(f"Loading GNN bundle: {MODEL_PREFIX}")
bundle = load_surrogate_bundle(prefix_sm=MODEL_PREFIX, device="cpu")

# --- Load stock A ----------------------------------------------------------
stock_file = config.TIMBER_STOCK_PATH / f"complete_timber_{TRAINING_SCENARIO}.csv"
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
print(f"Stock {TRAINING_SCENARIO}: {len(df_input_stock)} elements")

prepared_gnn_stock = stage_gnn.prepare_stock_for_gnn(df_input_stock)

# --- One-time normalization bounds -----------------------------------------
random.seed(BASE_SEED); np.random.seed(BASE_SEED)
fixed_norm, bounds_source = ga_eval._compute_one_time_normalization_constants(
    search_space=optimizer_search_space,
    df_stock=df_input_stock,
    config_dict=GA_CONFIG,
)

# --- Build the evaluate fn (identical wiring to the batch) ------------------
_base_fn = ga_algo.make_evaluate_fn(
    evaluate_fn_raw=ga_eval.evaluate_design_candidate,
    df_stock=df_input_stock,
    fixed_norm_constants=fixed_norm,
    config_dict=GA_CONFIG,
    bundle=bundle,
    prepared_gnn_stock=prepared_gnn_stock,
    verbose=False,
)

# --- Time random evaluations -----------------------------------------------
param_names = list(es_search_space.keys())
lo = np.array([es_search_space[k][0] for k in param_names])
hi = np.array([es_search_space[k][1] for k in param_names])
rng = np.random.default_rng(BASE_SEED)

print(f"\nTiming {N_EVALS} evaluations ({N_WARMUP} warmup excluded)...\n")
times = []
statuses = []
for i in range(N_EVALS):
    x = lo + rng.random(len(param_names)) * (hi - lo)
    params = dict(zip(param_names, x.tolist()))
    t0 = time.time()
    fitness, res = _base_fn(params, 0, CMAES_GENERATIONS)
    dt = time.time() - t0
    status = (res or {}).get("status", "?")
    statuses.append(status)
    if i >= N_WARMUP:
        times.append(dt)
    print(f"  eval {i+1:>2}: {dt:6.2f}s  status={status:<10} fit={fitness:.4f}")

times = np.array(times)
print("\n" + "=" * 60)
print(f"Warm evals: {len(times)}   "
      f"PENALIZED={statuses.count('PENALIZED')}  "
      f"others={len(statuses)-statuses.count('PENALIZED')}")
print(f"per-eval  mean={times.mean():.2f}s  median={np.median(times):.2f}s  "
      f"min={times.min():.2f}s  max={times.max():.2f}s  p90={np.percentile(times,90):.2f}s")

evals_per_run = CMAES_POPSIZE * CMAES_GENERATIONS
print("\nExtrapolation (no early-stop; worst case):")
for label, t in [("median", np.median(times)), ("mean", times.mean()),
                 ("p90", np.percentile(times, 90))]:
    per_run_h = evals_per_run * t / 3600
    print(f"  @{label:>6} {t:5.2f}s/eval -> {evals_per_run:,} evals/run = "
          f"{per_run_h:5.2f} h/run  |  {N_RUNS} runs = {per_run_h*N_RUNS:5.2f} h")
print("=" * 60)
print("NOTE: random candidates skew toward early-generation (often penalized,")
print("cheaper) evals. Converged generations run the full pipeline more often,")
print("so real walltime trends toward the upper end. Cluster CPU per-core speed")
print("also differs from this machine.")
