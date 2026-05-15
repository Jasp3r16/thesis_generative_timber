from __future__ import annotations

# =============================================================================
# c12_generate_training_data.py — Phase 1: GNN Retraining Data Generator
# =============================================================================
#
# Generates training samples for GNN surrogate model retraining.
# Produces node and edge feature CSVs to be processed by Grasshopper/Karamba
# FEA (which adds the Utilization column) before c21_surrogate_training.py.
#
# Sample strategy:
#   MILP samples   — cost-optimal assignment; matches GA inference distribution
#   Random samples — feasibility-constrained random assignment; provides
#                    unsafe-heavy examples the MILP would never produce
#
# New edge features vs existing model:
#   Width_m    m    — section width  (replaces Area; exposes aspect ratio)
#   Depth_m    m    — section depth
#   N_mean_EA  N    — mean-EA axial force from c24 FEM (structural demand)
#   Length, E, Iy, Iz, J, EA/L — unchanged, SI units
#
# Outputs (saved to config.GH_DATA_PATH):
#   training_nodes_raw.csv  — node features, one row per node per sample
#   training_edges_raw.csv  — edge features without Utilization column
#                             (Karamba FEA fills in Utilization after simulation)
#
# After Grasshopper adds Utilization, run c21_surrogate_training.run_preprocessing()
# with edge_csv="training_edges_raw" and node_csv="training_nodes_raw".

import json
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH  = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.append(str(_SRC_PATH))

import config
import c00_headquarter_params as _params
from c12_geometry_truss import generate_sample_vertices, get_valid_shifts
from c12_reconstruction  import reconstruct_edges, sample_random_design

from workflows import c24_stage_feasibility as stage_feas
from workflows import c25_stage_cost_matrix as stage_cost
from workflows import c26_stage_MILP        as stage_milp
from workflows import c27_stage_GNN         as stage_gnn


# =============================================================================
# CONSTANTS
# =============================================================================

_LOAD_PER_NODE_N = stage_gnn.LOAD_PER_NODE_N       # -13 500 N per load node

_STOCK_CSV    = config.TIMBER_STOCK_PATH / "complete_timber.csv"
_SEARCH_SPACE = config.DATA_IO_PATH      / "search_space.json"

# GNN edge feature columns written to the edge CSV.
# Utilization is absent here — Karamba adds it after FEA simulation.
NEW_EDGE_COLS = ["Width_m", "Depth_m", "Length", "E", "Iy", "Iz", "J", "EA/L", "N_mean_EA"]
NODE_COLS     = ["x", "y", "z", "Tx", "Ty", "Tz", "Rx", "Ry", "Rz", "Fz"]


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _derive_node_roles(
    df_vertices: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, list[int], list[int]]:
    """Extract sorted vertex table, node_positions, support_nodes, load_nodes."""
    verts = df_vertices.copy()
    verts["v_idx"] = (
        verts["vertex_index"].str.replace("v", "", regex=False).astype(int)
    )
    verts = verts.sort_values("v_idx").reset_index(drop=True)
    node_positions = verts[["x", "y", "z"]].values
    support_nodes  = verts[verts["attribute"] == "support"]["v_idx"].tolist()
    load_nodes     = verts[verts["attribute"] == "load"]["v_idx"].tolist()
    return verts, node_positions, support_nodes, load_nodes


def _geometry_signature(df_vertices: pd.DataFrame, decimals: int = 2) -> tuple:
    """Coordinate-based signature for near-duplicate detection."""
    verts = df_vertices.copy()
    verts["v_idx"] = (
        verts["vertex_index"].str.replace("v", "", regex=False).astype(int)
    )
    verts = verts.sort_values("v_idx").reset_index(drop=True)
    return tuple(
        (round(float(r["x"]), decimals),
         round(float(r["y"]), decimals),
         round(float(r["z"]), decimals))
        for _, r in verts.iterrows()
    )


def _member_lengths_m(node_positions: np.ndarray, df_edges: pd.DataFrame) -> np.ndarray:
    """Euclidean length of each member in metres."""
    v1 = df_edges["V1"].values.astype(int)
    v2 = df_edges["V2"].values.astype(int)
    return np.linalg.norm(node_positions[v2] - node_positions[v1], axis=1)


def _build_node_rows(
    verts_sorted:  pd.DataFrame,
    support_nodes: list[int],
    load_nodes:    list[int],
    sample_id:     int,
) -> list[dict]:
    """One node feature dict per vertex, ordered by v_idx (matches edge_index.json).

    Includes layer and attribute columns so Grasshopper can identify supports,
    load nodes, and hinges without re-deriving them from the geometry.
    """
    load_set    = set(load_nodes)
    support_set = set(support_nodes)
    rows = []
    for _, row in verts_sorted.iterrows():
        idx        = int(row["v_idx"])
        is_support = idx in support_set
        is_load    = idx in load_set
        rows.append({
            "sample_id":    sample_id,
            "vertex_index": row["vertex_index"],
            "layer":        row["layer"],       # "top" or "bottom"
            "attribute":    row["attribute"],   # "support", "load", or "hinges"
            "x":  float(row["x"]),
            "y":  float(row["y"]),
            "z":  float(row["z"]),
            "Tx": 1.0 if is_support else 0.0,
            "Ty": 1.0 if is_support else 0.0,
            "Tz": 1.0 if is_support else 0.0,
            "Rx": 1.0 if is_support else 0.0,
            "Ry": 1.0 if is_support else 0.0,
            "Rz": 1.0 if is_support else 0.0,
            "Fz": float(_LOAD_PER_NODE_N) if is_load else 0.0,
        })
    return rows


def _build_edge_rows(
    df_edges:         pd.DataFrame,
    lengths_m:        np.ndarray,
    assignment:       np.ndarray,
    stock_gnn:        pd.DataFrame,
    member_forces:    np.ndarray,
    sample_id:        int,
    assignment_type:  str,
) -> list[dict]:
    """One edge feature dict per member (e0..eN), ordered by edge position.

    GNN features use stock element properties (consistent with c27 inference):
      - Length is stock length (>= actual slot length due to cutting allowance)
      - Width_m, Depth_m from assigned stock cross-section
      - N_mean_EA is geometry-derived mean-EA force (actual structural demand)
    Reference columns (ignored by c21_surrogate_training): member_length_m,
    assigned_stock, assignment_type.
    """
    stock_gnn = stock_gnn.reset_index(drop=True)
    rows = []
    for i, (_, edge) in enumerate(df_edges.iterrows()):
        s = stock_gnn.iloc[int(assignment[i])]
        rows.append({
            "sample_id":       sample_id,
            "edge_id":         edge["edge_id"],
            "V1":              int(edge["V1"]),
            "V2":              int(edge["V2"]),
            # GNN edge features (NEW_EDGE_COLS):
            "Width_m":         float(s["Width_m"]),
            "Depth_m":         float(s["Depth_m"]),
            "Length":          float(s["Length"]),
            "E":               float(s["E"]),
            "Iy":              float(s["Iy"]),
            "Iz":              float(s["Iz"]),
            "J":               float(s["J"]),
            "EA/L":            float(s["EA/L"]),
            "N_mean_EA":       float(member_forces[i]),
            # Reference columns (not GNN features):
            "member_length_m": float(lengths_m[i]),
            "assigned_stock":  str(s.get("Member_ID", "")),
            "assignment_type": assignment_type,
            # Utilization — to be filled by Karamba FEA in Grasshopper
        })
    return rows


def _random_feasible_assignment(
    feasibility_mask: np.ndarray,
    rng:              random.Random,
) -> np.ndarray:
    """Randomly assign one feasible stock element per slot.

    If a slot has no feasible element (extreme geometry), falls back to a
    uniformly random element so the sample is not silently dropped.
    """
    n_slots, n_stock = feasibility_mask.shape
    assignment = np.zeros(n_slots, dtype=int)
    for i in range(n_slots):
        feasible = np.where(feasibility_mask[i])[0]
        if len(feasible) == 0:
            assignment[i] = rng.randint(0, n_stock - 1)
        else:
            assignment[i] = int(rng.choice(feasible.tolist()))
    return assignment


def _save_checkpoint(
    node_rows: list[dict],
    edge_rows: list[dict],
    output_dir: Path,
    label: str = "checkpoint",
) -> None:
    """Write partial CSVs to disk for crash recovery."""
    ts = datetime.now().strftime("%H%M%S")
    pd.DataFrame(node_rows).to_csv(output_dir / f"training_nodes_{label}_{ts}.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(output_dir / f"training_edges_{label}_{ts}.csv", index=False)
    print(f"  [checkpoint saved: {len(node_rows)} node rows, {len(edge_rows)} edge rows]")


# =============================================================================
# MAIN — TRAINING DATA GENERATOR
# =============================================================================

def generate_training_samples(
    n_milp:             int                = 500,
    n_random:           int                = 200,
    stock_csv_path:     "Path | str | None" = None,
    search_space_path:  "Path | str | None" = None,
    output_dir:         "Path | str | None" = None,
    solver_time_limit:  int                = 15,
    random_seed:        int                = 42,
    verbose:            bool               = False,
    save_intermediate:  int                = 0,
) -> dict[str, Any]:
    """
    Generate training samples for GNN surrogate model retraining.

    Parameters
    ----------
    n_milp            : number of MILP-assigned samples to generate
    n_random          : number of randomly-assigned samples to generate
    stock_csv_path    : path to complete_timber.csv (None -> config default)
    search_space_path : path to search_space.json (None -> config default)
    output_dir        : directory for output CSVs (None -> config.GH_DATA_PATH)
    solver_time_limit : MILP CBC time limit in seconds (default 15)
                        Shorter than GA default (30 s) — exact optimum not
                        required, plausible assignment is sufficient.
    random_seed       : seed for reproducible geometry sampling and random assignment
    verbose           : print per-sample details
    save_intermediate : save checkpoint CSVs every N samples (0 = disabled)

    Returns
    -------
    dict with keys:
        nodes_csv_path      — Path written for node features
        edges_csv_path      — Path written for edge features
        n_milp_generated    — actual MILP samples written
        n_random_generated  — actual random samples written
        n_milp_skipped      — MILP attempts that failed (infeasible / timeout)
        n_random_skipped    — random attempts that raised exceptions
        df_nodes            — pd.DataFrame of node feature rows
        df_edges_out        — pd.DataFrame of edge feature rows
    """
    stock_csv_path    = Path(stock_csv_path)    if stock_csv_path    else _STOCK_CSV
    search_space_path = Path(search_space_path) if search_space_path else _SEARCH_SPACE
    output_dir        = Path(output_dir)        if output_dir        else config.GH_DATA_PATH
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(random_seed)
    np.random.seed(random_seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[c12_generate_training_data] {ts}")
    print(f"  Stock:         {stock_csv_path.name}")
    print(f"  Search space:  {search_space_path.name}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Target:        {n_milp} MILP + {n_random} random = {n_milp + n_random} samples")
    print(f"  MILP limit:    {solver_time_limit} s  |  seed: {random_seed}")

    # ---- Load stock ----
    df_stock = pd.read_csv(stock_csv_path, sep=";")
    n_stock  = len(df_stock)
    print(f"\n  Stock loaded: {n_stock} elements")

    # ---- Pre-compute stock representations (once before loop) ----
    # GNN stock: SI units + Width_m / Depth_m added for new edge feature columns
    stock_gnn = stage_gnn.prepare_stock_for_gnn(df_stock)
    stock_gnn["Width_m"]   = df_stock["Width"].values  * 1e-3
    stock_gnn["Depth_m"]   = df_stock["Depth"].values  * 1e-3
    stock_gnn["Member_ID"] = df_stock["Member_ID"].values

    # Cost stock: validated and enriched for MILP + cost matrix
    stock_cost = stage_cost.prepare_stock_cost_inputs(df_stock)

    # ---- Load search space ----
    with open(search_space_path, "r", encoding="utf-8") as f:
        search_space = json.load(f)

    # ---- Fixed topology (same edge ordering for all samples) ----
    df_edges = reconstruct_edges(_params.GRID_CELLS_X, _params.GRID_CELLS_Y)
    n_edges  = len(df_edges)
    print(f"  Topology:     {n_edges} edges, "
          f"{_params.GRID_CELLS_X}x{_params.GRID_CELLS_Y} grid")

    all_node_rows: list[dict] = []
    all_edge_rows: list[dict] = []
    seen_signatures: set      = set()

    # ==========================================================================
    # PHASE A: MILP SAMPLES
    # ==========================================================================
    n_milp_generated  = 0
    n_milp_skipped    = 0
    milp_attempts     = 0
    max_milp_attempts = n_milp * 6  # safety cap on geometry retries

    print(f"\n--- Phase A: MILP samples (target={n_milp}) ---")

    while n_milp_generated < n_milp and milp_attempts < max_milp_attempts:
        milp_attempts += 1
        sample_id = n_milp_generated

        try:
            design_params = sample_random_design(search_space)
            vertices_list = generate_sample_vertices(
                sample_id=sample_id, params=design_params
            )
            df_vertices = pd.DataFrame(vertices_list)

            sig = _geometry_signature(df_vertices)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)

            verts_sorted, node_positions, support_nodes, load_nodes = (
                _derive_node_roles(df_vertices)
            )

            # Feasibility filter — also computes member forces via mean-EA FEM
            df_slots, feasibility_mask, member_forces, _ = stage_feas.build_cost_filter(
                node_positions = node_positions,
                edges_df       = df_edges,
                stock_df       = df_stock,
                support_nodes  = support_nodes,
                load_nodes     = load_nodes,
            )

            # Cost matrix — use pre-computed stock_cost, no logs needed
            cost_matrix, _, _ = stage_cost.build_cost_matrix(
                df_slots         = df_slots,
                df_input_stock   = df_stock,
                feasibility_mask = feasibility_mask,
                build_logs       = False,
                prepared_stock   = stock_cost,
            )

            # MILP — find cost-optimal stock assignment
            milp_out = stage_milp.run_milp_stage(
                cost_matrix               = cost_matrix,
                enriched_stock            = stock_cost,
                df_slots                  = df_slots,
                stock_df_raw              = df_stock,
                new_stock_max_uses        = None,
                solver_msg                = False,
                solver_time_limit         = solver_time_limit,
                raise_on_infeasible_slots = False,
            )

            if milp_out["status"] != "Optimal" or milp_out["milp_assignment"] is None:
                if verbose:
                    print(f"  [skip MILP #{milp_attempts}] "
                          f"status={milp_out['status']}")
                n_milp_skipped += 1
                continue

            milp_assignment = milp_out["milp_assignment"]
            lengths_m       = _member_lengths_m(node_positions, df_edges)

            all_node_rows.extend(
                _build_node_rows(verts_sorted, support_nodes, load_nodes, sample_id)
            )
            all_edge_rows.extend(
                _build_edge_rows(
                    df_edges, lengths_m, milp_assignment,
                    stock_gnn, member_forces, sample_id, "milp",
                )
            )
            n_milp_generated += 1

            if n_milp_generated % 10 == 0 or verbose:
                print(f"  MILP {n_milp_generated:>4}/{n_milp}  "
                      f"(attempts={milp_attempts}, skipped={n_milp_skipped})")

            if save_intermediate > 0 and n_milp_generated % save_intermediate == 0:
                _save_checkpoint(all_node_rows, all_edge_rows, output_dir)

        except Exception as exc:
            warnings.warn(
                f"MILP attempt {milp_attempts}: {type(exc).__name__}: {exc}",
                stacklevel=1,
            )
            n_milp_skipped += 1

    print(f"  Phase A done: {n_milp_generated} generated, "
          f"{n_milp_skipped} skipped, {milp_attempts} total attempts")

    # ==========================================================================
    # PHASE B: RANDOM ASSIGNMENT SAMPLES
    # ==========================================================================
    n_random_generated  = 0
    n_random_skipped    = 0
    random_attempts     = 0
    max_random_attempts = n_random * 4

    print(f"\n--- Phase B: Random samples (target={n_random}) ---")

    while n_random_generated < n_random and random_attempts < max_random_attempts:
        random_attempts += 1
        sample_id = n_milp_generated + n_random_generated  # IDs continue from MILP

        try:
            design_params = sample_random_design(search_space)
            vertices_list = generate_sample_vertices(
                sample_id=sample_id, params=design_params
            )
            df_vertices = pd.DataFrame(vertices_list)

            sig = _geometry_signature(df_vertices)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)

            verts_sorted, node_positions, support_nodes, load_nodes = (
                _derive_node_roles(df_vertices)
            )

            # Force estimation — random samples skip the full cost matrix / MILP
            df_slots, feasibility_mask, member_forces, _ = stage_feas.build_cost_filter(
                node_positions = node_positions,
                edges_df       = df_edges,
                stock_df       = df_stock,
                support_nodes  = support_nodes,
                load_nodes     = load_nodes,
            )

            rand_assignment = _random_feasible_assignment(feasibility_mask, rng)
            lengths_m       = _member_lengths_m(node_positions, df_edges)

            all_node_rows.extend(
                _build_node_rows(verts_sorted, support_nodes, load_nodes, sample_id)
            )
            all_edge_rows.extend(
                _build_edge_rows(
                    df_edges, lengths_m, rand_assignment,
                    stock_gnn, member_forces, sample_id, "random",
                )
            )
            n_random_generated += 1

            if n_random_generated % 10 == 0 or verbose:
                print(f"  Random {n_random_generated:>4}/{n_random}  "
                      f"(attempts={random_attempts}, skipped={n_random_skipped})")

            if save_intermediate > 0 and n_random_generated % save_intermediate == 0:
                _save_checkpoint(all_node_rows, all_edge_rows, output_dir)

        except Exception as exc:
            warnings.warn(
                f"Random attempt {random_attempts}: {type(exc).__name__}: {exc}",
                stacklevel=1,
            )
            n_random_skipped += 1

    print(f"  Phase B done: {n_random_generated} generated, "
          f"{n_random_skipped} skipped, {random_attempts} total attempts")

    # ==========================================================================
    # SAVE OUTPUTS
    # ==========================================================================
    df_nodes     = pd.DataFrame(all_node_rows)
    df_edges_out = pd.DataFrame(all_edge_rows)

    nodes_path = output_dir / "training_nodes_raw.csv"
    edges_path = output_dir / "training_edges_raw.csv"

    df_nodes.to_csv(nodes_path, index=False)
    df_edges_out.to_csv(edges_path, index=False)

    total_samples = n_milp_generated + n_random_generated
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total samples:   {total_samples}")
    print(f"  MILP samples:    {n_milp_generated}")
    print(f"  Random samples:  {n_random_generated}")
    print(f"  Node rows:       {len(df_nodes)}  "
          f"({len(df_nodes) // max(total_samples, 1)} nodes/sample)")
    print(f"  Edge rows:       {len(df_edges_out)}  "
          f"({len(df_edges_out) // max(total_samples, 1)} edges/sample)")
    print(f"\n  Nodes CSV:  {nodes_path.name}")
    print(f"  Edges CSV:  {edges_path.name}")
    print(f"\n  Edge feature columns: {NEW_EDGE_COLS}")
    print(f"\n  Next step: open Grasshopper, load both CSVs, run Karamba FEA,")
    print(f"  and export a new edges CSV with a 'Utilization' column added.")
    print(f"  Then call c21_surrogate_training.run_preprocessing().")
    print(f"{'='*60}")

    return {
        "nodes_csv_path":     nodes_path,
        "edges_csv_path":     edges_path,
        "n_milp_generated":   n_milp_generated,
        "n_random_generated": n_random_generated,
        "n_milp_skipped":     n_milp_skipped,
        "n_random_skipped":   n_random_skipped,
        "df_nodes":           df_nodes,
        "df_edges_out":       df_edges_out,
    }


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate GNN retraining data (Phase 1)."
    )
    parser.add_argument("--n-milp",   type=int, default=500,
                        help="Number of MILP samples (default 500)")
    parser.add_argument("--n-random", type=int, default=200,
                        help="Number of random samples (default 200)")
    parser.add_argument("--time-limit", type=int, default=15,
                        help="MILP solver time limit in seconds (default 15)")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed (default 42)")
    parser.add_argument("--verbose",  action="store_true",
                        help="Print per-sample details")
    parser.add_argument("--checkpoint", type=int, default=0,
                        help="Save checkpoint CSV every N samples (0 = off)")
    args = parser.parse_args()

    generate_training_samples(
        n_milp            = args.n_milp,
        n_random          = args.n_random,
        solver_time_limit = args.time_limit,
        random_seed       = args.seed,
        verbose           = args.verbose,
        save_intermediate = args.checkpoint,
    )
