from __future__ import annotations

# =============================================================================
# c12_generate_training_data.py — Phase 1: GNN Retraining Data Generator
# =============================================================================
#
# Generates training samples for GNN surrogate model retraining.
# Produces node and edge feature CSVs to be processed by Grasshopper/Karamba
# FEA (which adds the Utilization column) before c21_surrogate_training.py.
#
# Strategy: random stock assignment across randomly sampled geometries.
# No MILP or cost-matrix pipeline — the GNN learns structural physics
# (geometry + cross-section → utilization), not cost-optimality logic.
# Random assignment gives full coverage of safe and unsafe combinations
# across the entire stock pool. MILP would bias toward barely-feasible,
# cost-minimal sections, underrepresenting clearly safe and clearly unsafe cases.
#
# Mean-EA force estimation (c24) is retained as a fast single FEM solve per
# geometry to produce N_mean_EA — the structural demand per member. This is a
# geometry-derived feature independent of stock assignment and gives the GNN
# the load signal it needs to distinguish lightly from heavily loaded members.
#
# New edge features vs existing model:
#   Width_m    m    — section width  (replaces Area; exposes aspect ratio)
#   Depth_m    m    — section depth
#   N_mean_EA  N    — mean-EA axial force from c24 FEM (structural demand)
#   Length, E, Iy, Iz, J, EA/L — unchanged, SI units
#
# Outputs (saved to config.GH_DATA_PATH):
#   training_nodes_raw.csv  — node features, one row per node per sample
#                             includes layer and attribute columns for Grasshopper
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
from workflows import c27_stage_GNN         as stage_gnn


# =============================================================================
# CONSTANTS
# =============================================================================

_LOAD_PER_NODE_N = stage_gnn.LOAD_PER_NODE_N       # -13 500 N per load node
_TOTAL_LOAD_N    = stage_feas.TOTAL_LOAD_N          # 270 000 N total

_STOCK_CSV    = config.TIMBER_STOCK_PATH / "complete_timber_v2.csv"
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


def _random_assignment(n_slots: int, n_stock: int, rng: random.Random) -> np.ndarray:
    """Assign one stock element per slot, drawn uniformly from the full stock pool."""
    return np.array([rng.randint(0, n_stock - 1) for _ in range(n_slots)], dtype=int)


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
            "x":  round(float(row["x"]), 4),
            "y":  round(float(row["y"]), 4),
            "z":  round(float(row["z"]), 4),
            "Tx": 1.0 if is_support else 0.0,
            "Ty": 1.0 if is_support else 0.0,
            "Tz": 1.0 if is_support else 0.0,
            "Rx": 0.0,
            "Ry": 0.0,
            "Rz": 0.0,
            "Fz": float(_LOAD_PER_NODE_N) if is_load else 0.0,
        })
    return rows


def _build_edge_rows(
    df_edges:        pd.DataFrame,
    lengths_m:       np.ndarray,
    assignment:      np.ndarray,
    stock_gnn:       pd.DataFrame,
    member_forces:   np.ndarray,
    sample_id:       int,
) -> list[dict]:
    """One edge feature dict per member (e0..eN), ordered by edge position.

    stock_gnn must be reset_index'd so iloc[assignment[i]] is row-index safe.
    Length in the GNN features is the actual installed member length (from geometry).
    EA/L is recomputed from E * A / member_length — not the stock-length-based value.
    stock_length_m is included as a reference column only — not a GNN feature.
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
            "strength_class":  str(s["strength_class"]),
            # GNN edge features (NEW_EDGE_COLS):
            "Depth_m":         round(float(s["Depth_m"]), 4),
            "Width_m":         round(float(s["Width_m"]), 4),
            "Length":          round(float(lengths_m[i]), 3),   # actual installed member length
            "E":               round(float(s["E"])),
            "Iy":              round(float(s["Iy"]), 8),
            "Iz":              round(float(s["Iz"]), 9),
            "J":               round(float(s["J"]), 8),
            "EA/L":            round(float(s["E"] * s["Width_m"] * s["Depth_m"] / lengths_m[i]), 1),
            "N_mean_EA":       round(float(member_forces[i]), 2),
            # Reference columns (ignored by c21_surrogate_training):
            "stock_length_m":  round(float(s["Length"]), 3),   # original stock piece length
            "assigned_stock":  str(s.get("Member_ID", "")),
        })
    return rows


def _save_checkpoint(
    node_rows: list[dict],
    edge_rows: list[dict],
    output_dir: Path,
) -> None:
    """Write partial CSVs to disk for crash recovery."""
    ts = datetime.now().strftime("%H%M%S")
    pd.DataFrame(node_rows).to_csv(
        output_dir / f"training_nodes_checkpoint_{ts}.csv", index=False
    )
    pd.DataFrame(edge_rows).to_csv(
        output_dir / f"training_edges_checkpoint_{ts}.csv", index=False
    )
    print(f"  [checkpoint: {len(node_rows)} node rows, {len(edge_rows)} edge rows]")


# =============================================================================
# MAIN — TRAINING DATA GENERATOR
# =============================================================================

def generate_training_samples(
    n_samples:          int                = 10_000,
    stock_csv_path:     "Path | str | None" = None,
    search_space_path:  "Path | str | None" = None,
    output_dir:         "Path | str | None" = None,
    random_seed:        int                = 42,
    verbose:            bool               = False,
    save_intermediate:  int                = 0,
) -> dict[str, Any]:
    """
    Generate training samples for GNN surrogate model retraining.

    For each sample:
      1. Draw a random geometry from the search space.
      2. Estimate member forces via mean-EA FEM (c24) → N_mean_EA per member.
      3. Randomly assign one stock element per slot from the full stock pool.
      4. Build node and edge feature rows.

    No feasibility filtering or MILP — random assignment gives full coverage
    of safe and unsafe cross-section combinations across the stock pool.

    Parameters
    ----------
    n_samples         : total number of samples to generate (default 10 000)
    stock_csv_path    : path to complete_timber.csv (None -> config default)
    search_space_path : path to search_space.json   (None -> config default)
    output_dir        : directory for output CSVs   (None -> config.GH_DATA_PATH)
    random_seed       : seed for reproducibility
    verbose           : print per-sample details
    save_intermediate : save checkpoint CSVs every N samples (0 = disabled)

    Returns
    -------
    dict with keys:
        nodes_csv_path   — Path written for node features
        edges_csv_path   — Path written for edge features
        n_generated      — actual samples written
        n_skipped        — attempts that raised exceptions
        df_nodes         — pd.DataFrame of node feature rows
        df_edges_out     — pd.DataFrame of edge feature rows
    """
    stock_csv_path    = Path(stock_csv_path)    if stock_csv_path    else _STOCK_CSV
    search_space_path = Path(search_space_path) if search_space_path else _SEARCH_SPACE
    output_dir        = Path(output_dir)        if output_dir        else config.GH_DATA_PATH
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(random_seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[c12_generate_training_data] {ts}")
    print(f"  Stock:         {stock_csv_path.name}")
    print(f"  Search space:  {search_space_path.name}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Target:        {n_samples} samples  |  seed: {random_seed}")

    # ---- Load stock ----
    df_stock = pd.read_csv(stock_csv_path, sep=";")
    n_stock  = len(df_stock)
    print(f"\n  Stock loaded: {n_stock} elements")

    # ---- Pre-compute stock in GNN SI units (once before loop) ----
    stock_gnn = stage_gnn.prepare_stock_for_gnn(df_stock)
    stock_gnn["Width_m"]        = df_stock["Width"].values  * 1e-3
    stock_gnn["Depth_m"]        = df_stock["Depth"].values  * 1e-3
    stock_gnn["Member_ID"]      = df_stock["Member_ID"].values
    stock_gnn["strength_class"] = df_stock["f_mk"].apply(lambda v: f"c{int(v)}")

    # ---- Mean-EA for force estimation (stock-average, constant across geometries) ----
    mean_E_Pa  = float(df_stock["E_modulus_eff"].mean()) * 1e6
    mean_A_m2  = float((df_stock["Depth"] * df_stock["Width"]).mean()) * 1e-6
    mean_EA_SI = mean_E_Pa * mean_A_m2

    # ---- Load search space ----
    with open(search_space_path, "r", encoding="utf-8") as f:
        search_space = json.load(f)

    # ---- Fixed topology (same edge ordering for all samples) ----
    df_edges = reconstruct_edges(_params.GRID_CELLS_X, _params.GRID_CELLS_Y)
    n_edges  = len(df_edges)
    edges_v1 = df_edges["V1"].values.astype(int)
    edges_v2 = df_edges["V2"].values.astype(int)
    print(f"  Topology:      {n_edges} edges, "
          f"{_params.GRID_CELLS_X}x{_params.GRID_CELLS_Y} grid")

    all_node_rows: list[dict] = []
    all_edge_rows: list[dict] = []
    seen_signatures: set      = set()

    n_generated = 0
    n_skipped   = 0
    attempts    = 0
    max_attempts = n_samples * 4

    print(f"\n--- Generating {n_samples} samples ---")

    while n_generated < n_samples and attempts < max_attempts:
        attempts += 1
        sample_id = n_generated

        try:
            design_params = sample_random_design(search_space)
            vertices_list = generate_sample_vertices(
                sample_id=sample_id, params=design_params
            )
            df_vertices = pd.DataFrame(vertices_list)

            # Deduplication — skip near-identical geometries
            sig = _geometry_signature(df_vertices)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)

            verts_sorted, node_positions, support_nodes, load_nodes = (
                _derive_node_roles(df_vertices)
            )

            # Mean-EA force estimation — fast single FEM solve, no stock dependency
            member_forces = stage_feas.estimate_member_forces(
                node_positions = node_positions,
                edges_v1       = edges_v1,
                edges_v2       = edges_v2,
                support_nodes  = support_nodes,
                load_nodes     = load_nodes,
                total_load_n   = _TOTAL_LOAD_N,
                mean_EA_SI     = mean_EA_SI,
            )

            # Random stock assignment — uniform draw from full pool, no filtering
            assignment = _random_assignment(n_edges, n_stock, rng)
            lengths_m  = _member_lengths_m(node_positions, df_edges)

            all_node_rows.extend(
                _build_node_rows(verts_sorted, support_nodes, load_nodes, sample_id)
            )
            all_edge_rows.extend(
                _build_edge_rows(
                    df_edges, lengths_m, assignment,
                    stock_gnn, member_forces, sample_id,
                )
            )
            n_generated += 1

            if n_generated % 100 == 0 or verbose:
                ts_now = datetime.now().strftime("%H:%M:%S")
                print(f"  [{ts_now}]  {n_generated:>6}/{n_samples}  "
                      f"(attempts={attempts}, skipped={n_skipped})", flush=True)

            if save_intermediate > 0 and n_generated % save_intermediate == 0:
                _save_checkpoint(all_node_rows, all_edge_rows, output_dir)

        except Exception as exc:
            warnings.warn(
                f"Sample attempt {attempts}: {type(exc).__name__}: {exc}",
                stacklevel=1,
            )
            n_skipped += 1

    # ---- Save outputs ----
    df_nodes     = pd.DataFrame(all_node_rows)
    df_edges_out = pd.DataFrame(all_edge_rows)

    nodes_path = output_dir / "training_nodes_raw.csv"
    edges_path = output_dir / "training_edges_raw.csv"

    df_nodes.to_csv(nodes_path, index=False)
    df_edges_out.to_csv(edges_path, index=False)

    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Samples generated: {n_generated}")
    print(f"  Attempts:          {attempts}  (skipped: {n_skipped})")
    print(f"  Node rows:         {len(df_nodes)}  "
          f"({len(df_nodes) // max(n_generated, 1)} nodes/sample)")
    print(f"  Edge rows:         {len(df_edges_out)}  "
          f"({len(df_edges_out) // max(n_generated, 1)} edges/sample)")
    print(f"\n  Nodes CSV:  {nodes_path.name}")
    print(f"  Edges CSV:  {edges_path.name}")
    print(f"\n  Edge feature columns: {NEW_EDGE_COLS}")
    print(f"\n  Next step: open Grasshopper, load both CSVs, run Karamba FEA,")
    print(f"  and export an updated edges CSV with a 'Utilization' column.")
    print(f"  Then call c21_surrogate_training.run_preprocessing().")
    print(f"{'='*60}")

    return {
        "nodes_csv_path": nodes_path,
        "edges_csv_path": edges_path,
        "n_generated":    n_generated,
        "n_skipped":      n_skipped,
        "df_nodes":       df_nodes,
        "df_edges_out":   df_edges_out,
    }


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate GNN retraining data (Phase 1)."
    )
    parser.add_argument("--n-samples", type=int, default=10_000,
                        help="Number of samples to generate (default 10000)")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed (default 42)")
    parser.add_argument("--verbose",   action="store_true",
                        help="Print per-sample details")
    parser.add_argument("--checkpoint", type=int, default=0,
                        help="Save checkpoint CSV every N samples (0 = off)")
    args = parser.parse_args()

    generate_training_samples(
        n_samples         = args.n_samples,
        random_seed       = args.seed,
        verbose           = args.verbose,
        save_intermediate = args.checkpoint,
    )
