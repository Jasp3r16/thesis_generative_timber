# =============================================================================
# Step 5 — GNN Structural Feasibility Check
# =============================================================================
#
# Evaluates the structural feasibility of a complete MILP assignment using
# the trained TrussEdgeSafetyGNN surrogate model. Called once per GA iteration
# after MILP produces a complete slot→stock assignment.
#
# One forward pass → [120, 1] predictions → feasibility score for fitness function.
#
# ┌─────────────────────────────────────────────────────────────────┐
# │  DATA IO — to be completed by Claude Code                       │
# │  Sections marked TODO need path resolution for your project     │
# │  structure. Search for "# TODO" to find all integration points. │
# └─────────────────────────────────────────────────────────────────┘
#
# Inputs (all available from earlier steps in the GA loop):
#   node_positions   — [39, 3] current xyz from GA (metres)
#   milp_assignment  — [120] integer array, stock_df row index per slot
#   stock_df         — full stock DataFrame (complete_timber.csv)
#   model            — loaded TrussEdgeSafetyGNN (load once, reuse every iteration)
#   norm_stats       — normalisation stats from training (norm_stats.pt)
#   edge_index       — [2, 240] bidirectional topology tensor (edge_index.json)
#
# Outputs:
#   feasibility_score — float in [0, 1], fraction of members predicted safe
#                       1.0 = all members safe, 0.0 = all members unsafe
#   unsafe_member_ids — list[int], physical member indices predicted unsafe
#   preds_physical    — np.ndarray [120], raw P(unsafe) per physical member
#
# Usage in fitness function:
#   score, unsafe_ids, preds = gnn_feasibility(node_positions, milp_assignment, ...)
#   fitness = w_lca * lca_cost_norm + w_struct * (1.0 - score)

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# =============================================================================
# TODO — DATA IO (complete these paths for your project structure)
# =============================================================================

import config
from c21_surrogate_model_v4 import TrussEdgeSafetyGNN, create_model

# =============================================================================
# CONFIGURATION
# =============================================================================

# Decision threshold — P(unsafe) >= threshold → member predicted unsafe.
# Use the val-tuned threshold from your training run (best_threshold in train script).
# Default 0.35 based on your training results.
THRESHOLD = 0.35

# Number of physical members (before bidirectional duplication)
NUM_EDGES_PHYSICAL = 120

# Edge feature columns — must match training order exactly
EDGE_COLS = ["Area", "Length", "E", "Iy", "Iz", "J", "EA/L"]

# Node feature columns — must match training order exactly
NODE_COLS = ["x", "y", "z", "Tx", "Ty", "Tz", "Rx", "Ry", "Rz", "Fz"]

# Boundary condition layout (fixed per problem — update if BCs change)
# Tx, Ty, Tz, Rx, Ry, Rz: 1 = fixed, 0 = free
# Based on df_vertices.csv: support nodes [0,5,18,23] = fully fixed
SUPPORT_NODES = [0, 5, 18, 23]

# Applied load in z-direction per load node (N)
# 270,000 N total / 20 load nodes = 13,500 N per node (downward = negative)
LOAD_PER_NODE_N = -13_500.0   # negative = downward

# Load node indices (all top-chord non-support nodes)
LOAD_NODES = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22]


# =============================================================================
# MODEL LOADER — call once at GA startup, reuse every iteration
# =============================================================================

def load_gnn_model(
    ckpt_path: Path,
    norm_stats_path: Path,
    edge_index_path: Path,
    inference_config_path: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Load the trained GNN model and all inference artefacts.
    Call this ONCE before the GA loop starts.

    Returns a bundle dict with everything needed for gnn_feasibility():
        {
            "model":       TrussEdgeSafetyGNN (eval mode, on device),
            "norm_stats":  dict with node/edge means and stds,
            "edge_index":  torch.Tensor [2, 240] on device,
            "device":      str,
            "config":      dict from inference_config.json,
        }
    """
    # Load inference config (architecture parameters)
    with open(inference_config_path, "r") as f:
        inf_config = json.load(f)

    model = create_model(
        node_features_dim = inf_config["node_features_dim"],
        edge_features_dim = inf_config["edge_features_dim"],
        hidden_dim        = inf_config["hidden_dim"],
        num_layers        = inf_config["num_layers"],
        dropout_p         = inf_config["dropout_p"],
        device            = device,
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[GNN] Loaded checkpoint from epoch {ckpt.get('best_epoch', '?')+1}  "
          f"val_loss={ckpt.get('best_val_loss', float('nan')):.6f}")

    # Load normalisation stats
    norm_stats = torch.load(norm_stats_path, map_location="cpu")
    node_means = np.array([norm_stats["node_means"][c] for c in NODE_COLS])
    node_stds  = np.array([norm_stats["node_stds"][c]  for c in NODE_COLS])
    edge_means = np.array([norm_stats["edge_means"][c] for c in EDGE_COLS])
    edge_stds  = np.array([norm_stats["edge_stds"][c]  for c in EDGE_COLS])

    # Load bidirectional edge index
    with open(edge_index_path, "r") as f:
        edge_index = torch.tensor(json.load(f), dtype=torch.long).to(device)

    model.cache_topology(edge_index)

    print(f"[GNN] Model ready on {device}  |  "
          f"edge_index: {edge_index.shape}  |  "
          f"threshold: {THRESHOLD}")

    return {
        "model":      model,
        "norm_stats": {
            "node_means": node_means,
            "node_stds":  node_stds,
            "edge_means": edge_means,
            "edge_stds":  edge_stds,
        },
        "edge_index": edge_index,
        "device":     device,
        "config":     inf_config,
    }


# =============================================================================
# FEATURE BUILDERS
# =============================================================================

def build_node_features(node_positions: np.ndarray, device: str,
                        norm_stats: dict) -> torch.Tensor:
    """
    Build normalised node feature tensor [39, 10] from current GA geometry.

    Node features: [x, y, z, Tx, Ty, Tz, Rx, Ry, Rz, Fz]
    - xyz:          current node positions from GA (metres)
    - Tx..Rz:       boundary condition flags (fixed per problem)
    - Fz:           applied load in z-direction (N)

    All features are z-score normalised using training statistics,
    then clipped to ±5σ to match preprocessing.
    """
    n_nodes   = node_positions.shape[0]
    x_raw     = np.zeros((n_nodes, len(NODE_COLS)), dtype=np.float32)

    # Columns: x, y, z
    x_raw[:, 0] = node_positions[:, 0]
    x_raw[:, 1] = node_positions[:, 1]
    x_raw[:, 2] = node_positions[:, 2]

    # Columns: Tx, Ty, Tz, Rx, Ry, Rz — 1 = fixed DOF, 0 = free
    # Support nodes: all 6 DOF fixed
    for node in SUPPORT_NODES:
        x_raw[node, 3:9] = 1.0   # Tx, Ty, Tz, Rx, Ry, Rz

    # Column: Fz — applied vertical load per load node (N)
    for node in LOAD_NODES:
        x_raw[node, 9] = LOAD_PER_NODE_N

    # Normalise: z-score then clip to ±5σ
    x_norm = (x_raw - norm_stats["node_means"]) / norm_stats["node_stds"]
    x_norm = np.clip(x_norm, -5.0, 5.0)

    return torch.tensor(x_norm, dtype=torch.float32, device=device)


def build_edge_features(milp_assignment: np.ndarray, stock_df: pd.DataFrame,
                        device: str, norm_stats: dict) -> torch.Tensor:
    """
    Build normalised edge feature tensor [240, 7] from MILP assignment.

    milp_assignment: [120] integer array — row index into stock_df per slot.
    Edge features per member: [Area, Length, E, Iy, Iz, J, EA/L]

    The assignment gives properties for the 120 physical members.
    The second 120 rows are duplicates for the reverse (bidirectional) edges.

    Units expected (matching training data):
        Area   m²        Length  m         E      N/m²
        Iy     m⁴        Iz     m⁴        J      m⁴
        EA/L   N/m
    TODO: verify units match your GH export — adjust conversion if needed.
    """
    # Extract assigned stock properties for 120 physical members
    assigned = stock_df.iloc[milp_assignment][EDGE_COLS].values   # [120, 7]

    # Duplicate for reverse (bidirectional) edges
    assigned_bi = np.concatenate([assigned, assigned], axis=0)    # [240, 7]

    # Normalise
    edge_norm = (assigned_bi - norm_stats["edge_means"]) / norm_stats["edge_stds"]
    edge_norm = np.clip(edge_norm, -5.0, 5.0).astype(np.float32)

    return torch.tensor(edge_norm, dtype=torch.float32, device=device)


# =============================================================================
# GNN FEASIBILITY — call every GA iteration
# =============================================================================

def gnn_feasibility(
    node_positions:  np.ndarray,
    milp_assignment: np.ndarray,
    stock_df:        pd.DataFrame,
    model_bundle:    dict,
    threshold:       float = THRESHOLD,
) -> tuple[float, list[int], np.ndarray]:
    """
    Evaluate structural feasibility of a complete MILP assignment.
    One forward pass through the GNN — essentially free compute.

    Parameters
    ----------
    node_positions : np.ndarray [39, 3]
        Current node xyz from GA (metres).
    milp_assignment : np.ndarray [120]
        Row index into stock_df for each slot (output of MILP solver).
    stock_df : pd.DataFrame
        Full stock inventory (complete_timber.csv).
    model_bundle : dict
        Output of load_gnn_model() — contains model, norm_stats, edge_index, device.
    threshold : float
        Decision threshold. P(unsafe) >= threshold → member predicted unsafe.
        Default: THRESHOLD (0.35, val-tuned from training).

    Returns
    -------
    feasibility_score : float [0, 1]
        Fraction of physical members predicted SAFE.
        1.0 = fully feasible, 0.0 = all members predicted unsafe.
        Use as: fitness += w_structural * (1.0 - feasibility_score)

    unsafe_member_ids : list[int]
        Physical member indices (0–119) predicted unsafe.
        Useful for debugging which members consistently fail.

    preds_physical : np.ndarray [120]
        Raw P(unsafe) probability per physical member.
        Can be logged or used for a softer feasibility penalty.
    """
    device     = model_bundle["device"]
    norm_stats = model_bundle["norm_stats"]
    edge_index = model_bundle["edge_index"]

    model = model_bundle["model"]

    # Build input tensors
    x         = build_node_features(node_positions, device, norm_stats)
    edge_attr = build_edge_features(milp_assignment, stock_df, device, norm_stats)

    # Forward pass
    with torch.no_grad():
        preds = model(x, edge_index=edge_index, edge_attr=edge_attr)  # [240, 1]

    # Slice to physical members only (first 120 — reverse edges are duplicates)
    preds_physical = preds[:NUM_EDGES_PHYSICAL, 0].cpu().numpy()   # [120]

    # Apply threshold
    unsafe_flags      = preds_physical >= threshold
    unsafe_member_ids = np.where(unsafe_flags)[0].tolist()
    feasibility_score = float(1.0 - unsafe_flags.mean())

    return feasibility_score, unsafe_member_ids, preds_physical


# =============================================================================
# FITNESS INTEGRATION HELPER
# =============================================================================

def structural_fitness_penalty(
    feasibility_score: float,
    unsafe_member_ids: list[int],
    w_structural:      float = 0.3,
) -> float:
    """
    Convert GNN feasibility score into a fitness penalty term.

    feasibility_score: float [0,1] from gnn_feasibility()
    w_structural:      weight of structural term in total fitness
                       (0.3 = 30% structural, 70% LCA cost)

    Returns: structural penalty to ADD to fitness
             (0.0 = fully feasible, w_structural = fully infeasible)
    """
    return w_structural * (1.0 - feasibility_score)


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    import pandas as pd

    print("=" * 60)
    print("Step 5 — GNN Feasibility: standalone test")
    print("=" * 60)
    print()
    print("TODO items remaining before this script is fully operational:")
    print("  1. Set CKPT_PATH, NORM_STATS_PATH, EDGE_INDEX_PATH,")
    print("     INFERENCE_CONFIG to your artifact folder paths.")
    print("  2. Uncomment the model loading block in load_gnn_model().")
    print("  3. Uncomment the forward pass in gnn_feasibility().")
    print("  4. Remove the random placeholder predictions.")
    print("  5. Verify EDGE_COLS unit system matches your GH export")
    print("     (Area in m² or mm², E in N/m² or N/mm², etc.)")
    print()

    # Demonstrate with placeholder data
    stock_df       = pd.read_csv("complete_timber.csv", sep=";")
    node_positions = np.random.randn(39, 3)           # placeholder geometry
    milp_assignment = np.random.randint(0, len(stock_df), size=120)  # random assignment

    print(f"Stock pool: {len(stock_df)} elements")
    print(f"Random assignment: {milp_assignment[:5]} ... (first 5 slots)")
    print()

    # Placeholder model bundle (no real model loaded)
    model_bundle = {
        "norm_stats": {
            "node_means": np.zeros(len(NODE_COLS)),
            "node_stds":  np.ones(len(NODE_COLS)),
            "edge_means": np.zeros(len(EDGE_COLS)),
            "edge_stds":  np.ones(len(EDGE_COLS)),
        },
        "edge_index": torch.zeros((2, 240), dtype=torch.long),
        "device":     "cpu",
    }

    score, unsafe_ids, preds = gnn_feasibility(
        node_positions  = node_positions,
        milp_assignment = milp_assignment,
        stock_df        = stock_df,
        model_bundle    = model_bundle,
    )

    print(f"Feasibility score:    {score:.4f}")
    print(f"Unsafe members:       {len(unsafe_ids)} / {NUM_EDGES_PHYSICAL}")
    print(f"Structural penalty:   {structural_fitness_penalty(score):.4f}")
    print()
    print("=" * 60)
    print("GA LOOP INTEGRATION TEMPLATE")
    print("=" * 60)
    print("""
# --- At GA startup (run once) ---
model_bundle = load_gnn_model(
    ckpt_path            = CKPT_PATH,
    norm_stats_path      = NORM_STATS_PATH,
    edge_index_path      = EDGE_INDEX_PATH,
    inference_config_path= INFERENCE_CONFIG,
    device               = DEVICE,
)

# --- Inside GA iteration ---
def evaluate_candidate(node_positions, stock_df, edges_df, model_bundle):

    # Step 2 — pre-filter
    df_slots, df_feasibility, member_forces, stats = build_cost_filter(
        node_positions = node_positions, edges_df = edges_df,
        stock_df = stock_df, support_nodes = SUPPORT_NODES,
        load_nodes = LOAD_NODES,
    )
    if len(stats["slots_no_feasible_stock"]) > 0:
        return LARGE_PENALTY   # geometry has unassignable slots

    # Step 3 — LCA cost matrix
    cost_matrix, _, _ = build_cost_matrix(
        df_slots = df_slots, df_stock_raw = stock_df,
        df_feasibility = df_feasibility,
    )

    # Step 4 — MILP
    milp_assignment = solve_milp(cost_matrix)   # your MILP solver
    lca_cost        = cost_matrix[
        np.arange(120), milp_assignment
    ][np.isfinite(cost_matrix[np.arange(120), milp_assignment])].sum()

    # Step 5 — GNN structural feasibility
    score, unsafe_ids, preds = gnn_feasibility(
        node_positions  = node_positions,
        milp_assignment = milp_assignment,
        stock_df        = stock_df,
        model_bundle    = model_bundle,
    )

    # Step 6 — Fitness
    lca_norm   = lca_cost / LCA_REFERENCE        # normalise to [0,1]
    w_lca      = 0.7
    w_struct   = 0.3
    fitness    = w_lca * lca_norm + w_struct * (1.0 - score)
    return fitness
""")