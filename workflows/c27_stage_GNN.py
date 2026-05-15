# =============================================================================
# Step 5 — GNN Structural Feasibility Check
# =============================================================================
#
# Evaluates the structural feasibility of a complete MILP assignment using
# the trained TrussEdgeSafetyGNN surrogate model. Called once per GA iteration
# after MILP produces a complete slot→stock assignment.
#
# One forward pass → [120, 1] predictions → feasibility score for fitness.
#
# Changes vs v2:
#   - _build_edge_features() now conditional on edge_index size — no longer
#     hardcodes bidirectional duplication. If edge_index has 120 edges
#     (unidirectional, current best config), edge_attr is [120, 7]. If 240
#     (bidirectional), edge_attr is [240, 7]. Determined at runtime from
#     model_bundle["edge_index"].shape[1].
#   - gnn_feasibility() preds slice now conditional — takes [:120] only when
#     edge_index has 240 edges; takes all 120 directly when unidirectional.
#   - NUM_EDGES added alongside NUM_EDGES_PHYSICAL — set from edge_index at
#     load time and stored in model_bundle for use in feature building.
#   - load_gnn_model() stores num_edges in model_bundle.
#   - BIDIRECTIONAL helper property derived from num_edges at runtime.
#
# Notebook cell usage (unchanged from v2):
#
#   gnn_out = stage_gnn.run_gnn_stage(
#       node_positions  = node_positions,
#       milp_assignment = milp_out["milp_assignment"],
#       df_input_stock  = df_input_stock,
#       model_bundle    = model_bundle,
#   )
#   feasibility_score  = gnn_out["feasibility_score"]
#   structural_penalty = gnn_out["structural_penalty"]

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import config
from c21_surrogate_model_v4 import create_model

# =============================================================================
# CONFIGURATION
# =============================================================================

THRESHOLD          = 0.35    # P(unsafe) >= threshold → member predicted unsafe
NUM_EDGES_PHYSICAL = 120     # physical members — always 120 regardless of bi/uni

# Edge feature columns — must match training order exactly
EDGE_COLS = ["Area", "Length", "E", "Iy", "Iz", "J", "EA/L"]

# Node feature columns — must match training order exactly
NODE_COLS = ["x", "y", "z", "Tx", "Ty", "Tz", "Rx", "Ry", "Rz", "Fz"]

# Boundary conditions (fixed per problem)
SUPPORT_NODES   = [0, 5, 18, 23]
LOAD_NODES      = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22]
LOAD_PER_NODE_N = -13_500.0   # 270 kN / 20 nodes, downward = negative


# =============================================================================
# STOCK PREPARATION — unit conversion + section properties
# =============================================================================

def prepare_stock_for_gnn(df_input_stock: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw stock CSV (mm / N/mm²) to SI units (m / N/m²) and compute
    section properties needed as GNN edge features.

    Must be called once before the GA loop (stock properties don't change).
    Pass the returned DataFrame as stock_df to gnn_feasibility() and
    run_gnn_stage().

    Input columns required:
        Width, Depth, Length  — mm
        E_modulus_eff         — N/mm²

    Output columns added / overwritten:
        Area   m²   — cross-sectional area
        Length m    — member length
        E      N/m² — elastic modulus
        Iy     m⁴   — second moment of area (strong axis)
        Iz     m⁴   — second moment of area (weak axis)
        J      m⁴   — torsional constant (Saint-Venant approximation)
        EA/L   N/m  — axial stiffness per unit length
    """
    stock = df_input_stock.copy()

    b_m  = stock["Width"].values         * 1e-3   # mm → m
    h_m  = stock["Depth"].values         * 1e-3   # mm → m
    L_m  = stock["Length"].values        * 1e-3   # mm → m
    E_pa = stock["E_modulus_eff"].values  * 1e6   # N/mm² → N/m² (Pa)

    area_m2 = b_m * h_m
    a_m = np.minimum(b_m, h_m)
    c_m = np.maximum(b_m, h_m)

    stock["Area"]   = area_m2
    stock["Length"] = L_m
    stock["E"]      = E_pa
    stock["Iy"]     = b_m * h_m**3 / 12
    stock["Iz"]     = h_m * b_m**3 / 12
    stock["J"]      = a_m**3 * c_m / 3 * (1 - 0.63 * a_m / c_m)
    stock["EA/L"]   = E_pa * area_m2 / L_m

    return stock


# =============================================================================
# MILP ASSIGNMENT BUILDER — fallback if c27 didn't produce milp_assignment
# =============================================================================

def build_milp_assignment(
    df_results:     pd.DataFrame,
    df_slots:       pd.DataFrame,
    df_input_stock: pd.DataFrame,
) -> np.ndarray:
    """
    Convert MILP result DataFrame to [n_slots] integer index array.

    Prefer using milp_out["milp_assignment"] from c27_stage_milp_v2 directly.
    Use this function only as a fallback when stock_df_raw was not passed to
    run_milp_stage(), or when constructing the assignment manually.

    Parameters
    ----------
    df_results     : MILP output with columns edge_id, assigned_timber
    df_slots       : slot table with column edge_id (order = slot position)
    df_input_stock : raw stock CSV (Member_ID order = stock position)

    Returns
    -------
    milp_assignment : np.ndarray int [n_slots]
        milp_assignment[i] = row index into df_input_stock for slot i.
    """
    slot_id_to_idx  = {
        str(eid): int(i)
        for i, eid in enumerate(df_slots["edge_id"].astype(str))
    }
    stock_id_to_idx = {
        str(mid): int(i)
        for i, mid in enumerate(df_input_stock["Member_ID"].astype(str))
    }

    n_slots         = len(df_slots)
    milp_assignment = np.full(n_slots, -1, dtype=int)

    for _, row in df_results.iterrows():
        s_idx = slot_id_to_idx.get(str(row["edge_id"]), -1)
        k_idx = stock_id_to_idx.get(str(row["assigned_timber"]), -1)
        if s_idx >= 0 and k_idx >= 0:
            milp_assignment[s_idx] = k_idx

    unassigned = int((milp_assignment == -1).sum())
    if unassigned > 0:
        import warnings
        warnings.warn(
            f"build_milp_assignment: {unassigned} slot(s) unassigned (index=-1). "
            "GNN edge features for these slots will use row 0 of stock_df.",
            stacklevel=2,
        )
        milp_assignment[milp_assignment == -1] = 0

    return milp_assignment


# =============================================================================
# MODEL LOADER — call once at GA startup
# =============================================================================

def load_gnn_model(
    ckpt_path:             Path,
    norm_stats_path:       Path,
    edge_index_path:       Path,
    inference_config_path: Path,
    device:                str = "cpu",
) -> dict[str, Any]:
    """
    Load trained GNN model and all inference artefacts.
    Call ONCE before the GA loop; pass the returned bundle to run_gnn_stage().

    num_edges is derived from edge_index.json at load time and stored in the
    bundle — this determines whether edge features are built as [120,7] or
    [240,7] and whether predictions are sliced or used directly.

    Returns
    -------
    model_bundle : dict
        {
          "model":       TrussEdgeSafetyGNN in eval mode,
          "norm_stats":  dict with node/edge means and stds as np.ndarray,
          "edge_index":  torch.Tensor [2, num_edges] on device,
          "num_edges":   int — 120 (unidirectional) or 240 (bidirectional),
          "bidirectional": bool,
          "device":      str,
          "config":      dict from inference_config.json,
        }
    """
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
    print(f"[GNN] Loaded checkpoint from epoch {ckpt.get('best_epoch', '?') + 1}  "
          f"val_loss={ckpt.get('best_val_loss', float('nan')):.6f}")

    norm_stats = torch.load(norm_stats_path, map_location="cpu")
    node_means = np.array([norm_stats["node_means"][c] for c in NODE_COLS])
    node_stds  = np.array([norm_stats["node_stds"][c]  for c in NODE_COLS])
    edge_means = np.array([norm_stats["edge_means"][c] for c in EDGE_COLS])
    edge_stds  = np.array([norm_stats["edge_stds"][c]  for c in EDGE_COLS])

    with open(edge_index_path, "r") as f:
        edge_index = torch.tensor(json.load(f), dtype=torch.long).to(device)

    num_edges     = int(edge_index.shape[1])
    bidirectional = num_edges == 2 * NUM_EDGES_PHYSICAL

    model.cache_topology(edge_index)

    print(f"[GNN] Model ready on {device}  |  "
          f"edge_index: {edge_index.shape}  |  "
          f"{'bidirectional' if bidirectional else 'unidirectional'}  |  "
          f"threshold: {THRESHOLD}")

    return {
        "model":         model,
        "norm_stats":    {
            "node_means": node_means,
            "node_stds":  node_stds,
            "edge_means": edge_means,
            "edge_stds":  edge_stds,
        },
        "edge_index":    edge_index,
        "num_edges":     num_edges,
        "bidirectional": bidirectional,
        "device":        device,
        "config":        inf_config,
    }


# =============================================================================
# FEATURE BUILDERS (internal)
# =============================================================================

def _build_node_features(
    node_positions: np.ndarray,
    device:         str,
    norm_stats:     dict,
) -> torch.Tensor:
    """Build normalised node feature tensor [39, 10]."""
    n_nodes = node_positions.shape[0]
    x_raw   = np.zeros((n_nodes, len(NODE_COLS)), dtype=np.float32)

    x_raw[:, 0] = node_positions[:, 0]   # x
    x_raw[:, 1] = node_positions[:, 1]   # y
    x_raw[:, 2] = node_positions[:, 2]   # z

    for node in SUPPORT_NODES:
        x_raw[node, 3:9] = 1.0           # Tx Ty Tz Rx Ry Rz = fixed

    for node in LOAD_NODES:
        x_raw[node, 9] = LOAD_PER_NODE_N # Fz

    x_norm = (x_raw - norm_stats["node_means"]) / norm_stats["node_stds"]
    x_norm = np.clip(x_norm, -5.0, 5.0)
    return torch.tensor(x_norm, dtype=torch.float32, device=device)


def _build_edge_features(
    milp_assignment: np.ndarray,
    stock_df:        pd.DataFrame,
    device:          str,
    norm_stats:      dict,
    bidirectional:   bool,
) -> torch.Tensor:
    """
    Build normalised edge feature tensor from MILP assignment.

    stock_df must already be in SI units — pass output of prepare_stock_for_gnn().
    milp_assignment[i] = row index into stock_df for slot i (always 120 slots).

    Output shape:
        [120, 7] if unidirectional (current best config, BIDIRECTIONAL=False)
        [240, 7] if bidirectional  (reverse edges duplicate forward features)
    """
    assigned = stock_df.iloc[milp_assignment][EDGE_COLS].values   # [120, 7]

    if bidirectional:
        assigned = np.concatenate([assigned, assigned], axis=0)   # [240, 7]

    edge_norm = (assigned - norm_stats["edge_means"]) / norm_stats["edge_stds"]
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
    Single forward pass — evaluate structural feasibility of MILP assignment.

    Parameters
    ----------
    node_positions  : [39, 3] current node xyz from GA (metres)
    milp_assignment : [120] row indices into stock_df (SI units)
    stock_df        : stock DataFrame in SI units (from prepare_stock_for_gnn)
    model_bundle    : output of load_gnn_model()
    threshold       : decision threshold (default THRESHOLD = 0.35)

    Returns
    -------
    feasibility_score  : float [0,1] — fraction of members predicted safe
    unsafe_member_ids  : list[int]   — member indices predicted unsafe (0–119)
    preds_physical     : np.ndarray [120] — raw P(unsafe) per physical member
    """
    device        = model_bundle["device"]
    norm_stats    = model_bundle["norm_stats"]
    edge_index    = model_bundle["edge_index"]
    bidirectional = model_bundle["bidirectional"]
    model         = model_bundle["model"]

    x         = _build_node_features(node_positions, device, norm_stats)
    edge_attr = _build_edge_features(
        milp_assignment, stock_df, device, norm_stats, bidirectional
    )

    with torch.no_grad():
        preds = model(x, edge_index=edge_index, edge_attr=edge_attr)

    # Unidirectional: preds is [120, 1] — use all
    # Bidirectional:  preds is [240, 1] — take first 120 (physical members)
    preds_physical    = preds[:NUM_EDGES_PHYSICAL, 0].cpu().numpy()   # [120] always
    unsafe_flags      = preds_physical >= threshold
    unsafe_member_ids = np.where(unsafe_flags)[0].tolist()
    feasibility_score = float(1.0 - unsafe_flags.mean())

    return feasibility_score, unsafe_member_ids, preds_physical


# =============================================================================
# ORCHESTRATION — single call for notebook cell
# =============================================================================

def run_gnn_stage(
    node_positions:  np.ndarray,
    milp_assignment: np.ndarray,
    df_input_stock:  pd.DataFrame,
    model_bundle:    dict,
    threshold:       float = THRESHOLD,
    w_structural:    float = 0.3,
    print_summary:   bool  = True,
) -> dict[str, Any]:
    """
    Orchestrate the full GNN feasibility stage for one GA iteration.

    Parameters
    ----------
    node_positions  : [39, 3] current node xyz from GA (metres)
    milp_assignment : [120] integer array from milp_out["milp_assignment"]
    df_input_stock  : raw stock inventory in original units (mm / N/mm²).
                      Unit conversion is done internally.
    model_bundle    : output of load_gnn_model() — load once, reuse every iteration.
    threshold       : decision threshold (default 0.35)
    w_structural    : weight for structural penalty in fitness (default 0.3)
    print_summary   : whether to print the GNN result summary

    Returns
    -------
    dict with keys:
        feasibility_score  — float [0,1]: fraction of members predicted safe
        structural_penalty — float: w_structural * (1 - feasibility_score)
        unsafe_member_ids  — list[int]: member indices predicted unsafe (0–119)
        preds_physical     — np.ndarray [120]: raw P(unsafe) per member
        n_unsafe           — int: number of unsafe members
        n_safe             — int: number of safe members
    """
    stock_gnn = prepare_stock_for_gnn(df_input_stock)

    feasibility_score, unsafe_member_ids, preds_physical = gnn_feasibility(
        node_positions  = node_positions,
        milp_assignment = milp_assignment,
        stock_df        = stock_gnn,
        model_bundle    = model_bundle,
        threshold       = threshold,
    )

    structural_penalty = float(w_structural * (1.0 - feasibility_score))
    n_unsafe = len(unsafe_member_ids)
    n_safe   = NUM_EDGES_PHYSICAL - n_unsafe

    if print_summary:
        bidir_str = "bidirectional" if model_bundle["bidirectional"] else "unidirectional"
        print(f"\n[GNN] Feasibility Results  ({bidir_str}, {model_bundle['num_edges']} edges):")
        print(f"  Feasibility score:  {feasibility_score:.4f}  "
              f"(1.0 = all members predicted safe)")
        print(f"  Safe members:       {n_safe} / {NUM_EDGES_PHYSICAL}")
        print(f"  Unsafe members:     {n_unsafe} / {NUM_EDGES_PHYSICAL}")
        print(f"  Structural penalty: {structural_penalty:.4f}  "
              f"(w_structural={w_structural})")
        if unsafe_member_ids:
            preview = unsafe_member_ids[:20]
            suffix  = "..." if n_unsafe > 20 else ""
            print(f"  Unsafe member IDs:  {preview}{suffix}")

    return {
        "feasibility_score":  feasibility_score,
        "structural_penalty": structural_penalty,
        "unsafe_member_ids":  unsafe_member_ids,
        "preds_physical":     preds_physical,
        "n_unsafe":           n_unsafe,
        "n_safe":             n_safe,
    }