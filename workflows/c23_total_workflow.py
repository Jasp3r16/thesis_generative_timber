'''
python script that runs all stages one time, importing dataset, setting up geometry, prefiltering feasibility, calculation of the cost matrix, and finally calculating the fitness
'''
from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

from networkx import config
import numpy as np
import pandas as pd
import json


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
	sys.path.append(str(SRC_PATH))

from workflows.c24_stage_geometry import run_geometry_from_design, run_random_geometry_stage
from workflows.c25_stage_feasibility import build_cost_filter
from workflows.c26_stage_cost_matrix import build_cost_matrix
from workflows.c27_stage_MILP import run_milp_stage
from workflows.c28_stage_GNN import prepare_stock_for_gnn, run_gnn_stage, SUPPORT_NODES, LOAD_NODES
from workflows.c29_stage_fitness_score import run_fitness_stage

def import_stock_and_search_space() -> tuple[pd.DataFrame, dict]:
    stock_path = config.TIMBER_STOCK_PATH / 'complete_timber.csv'
    json_path = config.DATA_IO_PATH / "search_space.json"

    with open(json_path, "r") as f:
        optimizer_search_space = json.load(f)

    print(f"Search space loaded. The optimizer can control {len(optimizer_search_space)} parameters.")

    # Try common combinations
    read_attempts = [
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "latin1"},
        {"sep": ";", "encoding": "latin1"},
    ]

    df_input_stock = None
    for opts in read_attempts:
        try:
            df_try = pd.read_csv(stock_path, **opts)  # type: ignore
            # Valid if we get more than 1 column
            if df_try.shape[1] > 1:
                df_input_stock = df_try
                print(f"Loaded with sep='{opts['sep']}' and encoding='{opts['encoding']}'")
                break
        except Exception:
            pass

    if df_input_stock is None:
        raise ValueError("Could not parse CSV with tested delimiter/encoding combinations.")

    # Clean column names
    df_input_stock.columns = df_input_stock.columns.str.strip()


def run_one_iteration(
	df_input_stock: pd.DataFrame,
	design_params: dict | None = None,
	sample_id: int = 0,
	support_nodes: list[int] | None = None,
	load_nodes: list[int] | None = None,
	weight_config: dict[str, float] | None = None,
	derive_normalization_constants: bool = True,
	new_stock_max_uses: int | None = 1,
	solver_msg: bool = False,
	model_bundle: dict | None = None,
	verbose: bool = False,
 ) -> dict[str, Any]:
	"""Run a single pipeline iteration from geometry → fitness.

	Minimal, notebook-free runner that wires stages c24 → c29 together.
	Returns a dictionary with intermediate and final results.
	"""
	if support_nodes is None:
		support_nodes = SUPPORT_NODES
	if load_nodes is None:
		load_nodes = LOAD_NODES
	if weight_config is None:
		weight_config = {"omega_1": 1.0, "omega_2": 1.0, "omega_3": 1.0}

	# --- Geometry stage ---
	if design_params is None:
		geo_out = run_random_geometry_stage(sample_id=sample_id)
	else:
		geo_out = run_geometry_from_design(design_params=design_params, sample_id=sample_id)
	if verbose:
		print(f"[geometry] generated vertices={len(geo_out['df_vertices'])} edges={len(geo_out['df_edges'])}")

	df_vertices = geo_out["df_vertices"]
	df_edges = geo_out["df_edges"]

	# Build node positions array expected by the feasibility stage
	if "vertex_index" in df_vertices.columns:
		df_v_sorted = df_vertices.sort_values("vertex_index")
	else:
		df_v_sorted = df_vertices.reset_index(drop=True)
	node_positions = df_v_sorted[["x", "y", "z"]].to_numpy(dtype=float)

	# --- Feasibility / prefilter stage ---
	if verbose:
		print("[feasibility] starting feasibility prefilter")
	df_slots, feasibility_mask, member_forces, filter_stats = build_cost_filter(
		node_positions, df_edges, df_input_stock, support_nodes, load_nodes
	)
	if verbose:
		print(f"[feasibility] df_slots={len(df_slots)} feasible_pairs={int(feasibility_mask.sum())}")

	# --- Cost matrix stage ---
	if verbose:
		print("[cost_matrix] building cost matrix")
	cost_matrix, enriched_stock, df_logs = build_cost_matrix(
		df_slots, df_input_stock, feasibility_mask, build_logs=False
	)
	if verbose:
		print(f"[cost_matrix] shape={cost_matrix.shape}")

	# --- MILP assignment stage ---
	if verbose:
		print("[milp] solving assignment MILP")
	milp_out = run_milp_stage(
		cost_matrix=cost_matrix,
		enriched_stock=enriched_stock,
		df_slots=df_slots,
		stock_df_raw=df_input_stock,
		new_stock_max_uses=new_stock_max_uses,
		solver_msg=solver_msg,
	)
	if verbose:
		print(f"[milp] status={milp_out.get('status')} total_cost={milp_out.get('total_cost')}")

	# --- Optional GNN structural check ---
	structural_infeasibility = 0.0
	gnn_out = None
	if model_bundle is not None and milp_out.get("milp_assignment") is not None:
		stock_for_gnn = prepare_stock_for_gnn(enriched_stock)
		if verbose:
			print("[gnn] running structural GNN check")
		gnn_out = run_gnn_stage(
			node_positions=node_positions,
			milp_assignment=milp_out["milp_assignment"],
			df_input_stock=stock_for_gnn,
			model_bundle=model_bundle,
		)
		# structural infeasibility = fraction of unsafe members
		structural_infeasibility = 1.0 - float(gnn_out.get("feasibility_score", 1.0))
		if verbose:
			print(f"[gnn] feasibility_score={gnn_out.get('feasibility_score')}")

	# --- Fitness evaluation ---
	if verbose:
		print("[fitness] evaluating fitness")
	fitness_out = run_fitness_stage(
		df_results=milp_out.get("df_results", pd.DataFrame()),
		enriched_stock=enriched_stock,
		df_slots=df_slots,
		total_cost=float(milp_out.get("total_cost", float("inf"))),
		weight_config=weight_config,
		normalization_margin=0.20,
		normalization_constants=None,
		derive_normalization_constants=derive_normalization_constants,
		run_sanity_checks=True,
		print_breakdown=False,
		structural_infeasibility=structural_infeasibility,
	)
	if verbose:
		print(f"[fitness] fitness={fitness_out['fitness_result']['fitness']}")

	return {
		"geometry": geo_out,
		"df_slots": df_slots,
		"enriched_stock": enriched_stock,
		"cost_matrix": cost_matrix,
		"milp": milp_out,
		"gnn": gnn_out,
		"fitness": fitness_out,
		"filter_stats": filter_stats,
		"df_logs": df_logs,
	}


def _parse_args() -> dict[str, Any]:
	import argparse
	parser = argparse.ArgumentParser(description="Run one iteration of the full pipeline (c24→c29)")
	parser.add_argument("--stock-csv", required=False, help="Path to stock CSV file (optional, default from config)")
	parser.add_argument("--design-json", required=False, help="Path to design JSON (optional)")
	parser.add_argument("--sample-id", type=int, default=0)
	parser.add_argument("--verbose", action="store_true", help="Print start/end phase messages")
	parser.add_argument("--disable-gnn", action="store_true", help="Do not run the GNN structural check")
	return vars(parser.parse_args())


def main() -> None:
	args = _parse_args(
		"verbose": True,
    )
	np.random.seed(42)
	
	# Resolve stock CSV: user-specified or default from config
	import config
	stock_arg = args.get("stock_csv")
	if stock_arg:
		stock_path = Path(stock_arg)
	else:
		stock_path = config.TIMBER_STOCK_PATH / "complete_timber.csv"
	df_stock = pd.read_csv(stock_path)

	# Resolve design JSON: user-specified or default search_space.json from config
	design = None
	design_arg = args.get("design_json")
	if design_arg:
		with open(design_arg, "r", encoding="utf-8") as f:
			design = json.load(f)
	else:
		default_json = config.DATA_IO_PATH / "search_space.json"
		if default_json.exists():
			with open(default_json, "r", encoding="utf-8") as f:
				design = json.load(f)

	model_bundle = None
	if not args.get("disable_gnn"):
		# No automatic model loading here — user can pass a prepared model_bundle
		model_bundle = None

	out = run_one_iteration(
		df_input_stock=df_stock,
		design_params=design,
		sample_id=int(args.get("sample_id", 0)),
		model_bundle=model_bundle,
		verbose=bool(args.get("verbose", False)),
	)

	# Minimal summary on exit
	if args.get("verbose"):
		fitness = out.get("fitness", {}).get("fitness_result", {}).get("fitness")
		print(f"[done] fitness={fitness}")


if __name__ == "__main__":
	main()

