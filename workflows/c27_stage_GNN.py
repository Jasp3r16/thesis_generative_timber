from __future__ import annotations

# =============================================================================
# c27_stage_GNN.py — GNN Structural Feasibility Stage
# =============================================================================
#
# Changes vs v2:
#   1. load_gnn_model() removed — dead loader that duplicated c21_surrogate_io.
#      Use load_surrogate_bundle() from c21_surrogate_io exclusively.
#      Removed json / Path / config / create_model imports (only needed there).
#   2. prepare_stock_for_gnn() now called once before the GA loop, not inside
#      run_gnn_stage(). Pass pre-converted stock as stock_df to run_gnn_stage()
#      and gnn_feasibility() to avoid a full DataFrame copy per evaluation.
#   3. support_nodes / load_nodes accepted as explicit parameters in
#      _build_node_features, gnn_feasibility, and run_gnn_stage — defaulting to
#      the hardcoded 5x3-grid constants. GA evaluator now passes the dynamically
#      derived values so the GNN receives correct boundary condition features.
#   4. build_milp_assignment raises ValueError on unassigned slots instead of
#      silently substituting row 0 (corrupted GNN features with no warning).
#   5. threshold=None in gnn_feasibility — resolves from bundle["config"]
#      ["recommended_threshold"] when not explicitly set, so the training-tuned
#      threshold is used automatically.
#   6. w_structural default changed 0.3 -> 0.0; structural_penalty is for
#      notebook convenience only — the GA evaluator reads feasibility_score
#      directly and passes structural_infeasibility to run_fitness_stage.
#   7. Stale module name references fixed in docstrings.

import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch

import c24_stage_feasibility
from c24_stage_feasibility import compute_nodal_fz, LOAD_KN_PER_M2


# =============================================================================
# CONFIGURATION
# =============================================================================

THRESHOLD          = 0.35
NUM_EDGES_PHYSICAL = 120

EDGE_COLS = ["Width_m", "Depth_m", "Length", "E", "Iy", "Iz", "J", "EA/L", "N_mean_EA"]
NODE_COLS = ["x", "y", "z", "Tx", "Ty", "Tz", "Rx", "Ry", "Rz", "Fz"]

# Default boundary conditions for the 5×3 grid topology.
# Pass support_nodes / load_nodes explicitly when the geometry differs.
_DEFAULT_SUPPORT_NODES = [0, 5, 18, 23]
_DEFAULT_LOAD_NODES    = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                          16, 17, 19, 20, 21, 22]


# =============================================================================
# STOCK PREPARATION — unit conversion + section properties
# =============================================================================

def prepare_stock_for_gnn(df_input_stock: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw stock CSV (mm / N/mm²) to SI units (m / N/m²) and compute
    section properties needed as GNN edge features.

    Call ONCE before the GA loop and pass the returned DataFrame as stock_df
    to gnn_feasibility() / run_gnn_stage(). Stock properties do not change
    between GA evaluations.

    Note: "Length" in the returned copy is in metres (not mm). The input
    DataFrame is not modified.

    Input columns required:
        Width, Depth, Length  — mm
        E_modulus_eff         — N/mm²

    Output columns added:
        Width_m m    — section width
        Depth_m m    — section depth
        Area    m²   — cross-sectional area
        E       N/m² — elastic modulus
        Iy      m⁴   — second moment of area (strong axis)
        Iz      m⁴   — second moment of area (weak axis)
        J       m⁴   — torsional constant (Saint-Venant approximation)

    Note: Length and EA/L are NOT included here — they depend on the actual
    installed member length (geometry-derived) and are computed per member
    inside _build_edge_features() at inference time.
    """
    stock = df_input_stock.copy()

    b_m  = stock["Width"].values         * 1e-3
    h_m  = stock["Depth"].values         * 1e-3
    E_pa = stock["E_modulus_eff"].values  * 1e6

    area_m2 = b_m * h_m
    a_m     = np.minimum(b_m, h_m)
    c_m     = np.maximum(b_m, h_m)

    stock["Width_m"] = b_m
    stock["Depth_m"] = h_m
    stock["Area"]    = area_m2
    stock["E"]       = E_pa
    stock["Iy"]      = b_m * h_m**3 / 12
    stock["Iz"]      = h_m * b_m**3 / 12
    stock["J"]       = a_m**3 * c_m / 3 * (1 - 0.63 * a_m / c_m)

    return stock


# =============================================================================
# MILP ASSIGNMENT BUILDER — fallback if stock_df_raw not passed to c26
# =============================================================================

def build_milp_assignment(
    df_results:     pd.DataFrame,
    df_slots:       pd.DataFrame,
    df_input_stock: pd.DataFrame,
) -> np.ndarray:
    """
    Convert MILP result DataFrame to [n_slots] integer index array.

    Prefer using milp_out["milp_assignment"] from c26_stage_MILP directly —
    it is built inside run_milp_stage() when stock_df_raw is provided.
    Use this function only as a fallback when building the assignment manually.

    Parameters
    ----------
    df_results     : MILP output with columns edge_id, assigned_timber
    df_slots       : slot table with column edge_id (order = slot position)
    df_input_stock : raw stock CSV (Member_ID order = stock position)

    Returns
    -------
    milp_assignment : np.ndarray int [n_slots]
        milp_assignment[i] = row index into df_input_stock for slot i.

    Raises
    ------
    ValueError if any slot has no matching assignment in df_results.
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
        raise ValueError(
            f"build_milp_assignment: {unassigned} slot(s) have no matching stock "
            "assignment in df_results. Ensure the MILP completed with Optimal status "
            "and that df_results contains an entry for every edge_id in df_slots."
        )

    return milp_assignment


# =============================================================================
# FEATURE BUILDERS (internal)
# =============================================================================

def _build_node_features(
    node_positions: np.ndarray,
    device:         str | "torch.device",
    norm_stats:     dict,
    support_nodes:  list[int] | None = None,
    load_nodes:     list[int] | None = None,
) -> torch.Tensor:
    """Build normalised node feature tensor [n_nodes, 10]."""
    support_nodes = support_nodes if support_nodes is not None else _DEFAULT_SUPPORT_NODES
    load_nodes    = load_nodes    if load_nodes    is not None else _DEFAULT_LOAD_NODES

    n_nodes = node_positions.shape[0]
    x_raw   = np.zeros((n_nodes, len(NODE_COLS)), dtype=np.float32)

    x_raw[:, 0] = node_positions[:, 0]   # x
    x_raw[:, 1] = node_positions[:, 1]   # y
    x_raw[:, 2] = node_positions[:, 2]   # z

    for node in support_nodes:
        x_raw[node, 3:9] = 1.0           # Tx Ty Tz Rx Ry Rz = fixed

    fz = compute_nodal_fz(node_positions, support_nodes, load_nodes, LOAD_KN_PER_M2)
    x_raw[:, 9] = fz.astype(np.float32)  # Fz: tributary area loads (N), matches training

    x_norm = (x_raw - norm_stats["node_means"]) / norm_stats["node_stds"]
    x_norm = np.clip(x_norm, -5.0, 5.0)
    return torch.tensor(x_norm, dtype=torch.float32, device=device)


def _build_edge_features(
    milp_assignment:  np.ndarray,
    stock_df:         pd.DataFrame,
    member_lengths_m: np.ndarray,
    member_forces:    np.ndarray,
    device:           str | "torch.device",
    norm_stats:       dict,
    bidirectional:    bool,
) -> torch.Tensor:
    """
    Build normalised edge feature tensor from MILP assignment.

    Assembles the 9 EDGE_COLS features per member in the correct order:
        Width_m, Depth_m  — from stock (milp_assignment → stock_df)
        Length            — actual installed member length (geometry-derived)
        E, Iy, Iz, J      — from stock
        EA/L              — recomputed as E * A / member_length (geometry-derived)
        N_mean_EA         — mean-EA FEM axial force estimate (geometry-derived)

    Parameters
    ----------
    milp_assignment  : [120] row indices into stock_df
    stock_df         : SI-unit stock table from prepare_stock_for_gnn()
    member_lengths_m : [120] actual installed member lengths in metres
    member_forces    : [120] N_mean_EA per member (N), from estimate_member_forces
    """
    _STOCK_COLS = ["Width_m", "Depth_m", "E", "Iy", "Iz", "J"]
    s = stock_df.iloc[milp_assignment][_STOCK_COLS].values.astype(np.float64)

    w_m  = s[:, 0]
    d_m  = s[:, 1]
    E    = s[:, 2]
    ea_l = E * (w_m * d_m) / member_lengths_m   # E * A / installed_length

    # Assemble in EDGE_COLS order: Width_m, Depth_m, Length, E, Iy, Iz, J, EA/L, N_mean_EA
    assigned = np.column_stack([
        w_m, d_m,
        member_lengths_m,
        s[:, 2], s[:, 3], s[:, 4], s[:, 5],
        ea_l,
        member_forces,
    ]).astype(np.float64)   # [120, 9]

    if bidirectional:
        assigned = np.concatenate([assigned, assigned], axis=0)   # [240, 9]

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
    threshold:       float | None = None,
    support_nodes:   list[int] | None = None,
    load_nodes:      list[int] | None = None,
) -> tuple[float, list[int], np.ndarray]:
    """
    Single forward pass — evaluate structural feasibility of MILP assignment.

    Parameters
    ----------
    node_positions  : [n_nodes, 3] current node xyz from GA (metres)
    milp_assignment : [120] row indices into stock_df
    stock_df        : stock DataFrame from prepare_stock_for_gnn() — must contain
                      Width_m, Depth_m, E, Iy, Iz, J columns in SI units
    model_bundle    : output of load_surrogate_bundle() from c21_surrogate_io
    threshold       : decision threshold for P(unsafe). When None, reads from
                      bundle["config"]["recommended_threshold"], falling back
                      to module constant THRESHOLD (0.35).
    support_nodes   : node indices with fixed supports. None → default 5x3 grid.
    load_nodes      : node indices with applied load. None → default 5x3 grid.

    Geometry-derived features computed internally each call:
        member_lengths_m — Euclidean length of each installed member (m)
        EA/L             — E * A / member_length (uses assigned stock E and area)
        N_mean_EA        — mean-EA FEM axial force estimate (N)

    Returns
    -------
    feasibility_score  : float [0,1] — fraction of members predicted safe
    unsafe_member_ids  : list[int]   — member indices predicted unsafe (0–119)
    preds_physical     : np.ndarray [120] — raw P(unsafe) per physical member
    """
    if threshold is None:
        threshold = float(
            model_bundle.get("config", {}).get("recommended_threshold", THRESHOLD)
        )

    device        = model_bundle["device"]
    norm_stats    = model_bundle["norm_stats"]
    edge_index    = model_bundle["edge_index"]
    bidirectional = model_bundle["bidirectional"]
    model         = model_bundle["model"]

    support_nodes = support_nodes if support_nodes is not None else _DEFAULT_SUPPORT_NODES
    load_nodes    = load_nodes    if load_nodes    is not None else _DEFAULT_LOAD_NODES

    # Member lengths from current geometry (geometry-derived, varies per GA iteration)
    ei_np            = edge_index[:, :NUM_EDGES_PHYSICAL].cpu().numpy()
    member_lengths_m = np.linalg.norm(
        node_positions[ei_np[1]] - node_positions[ei_np[0]], axis=1
    )  # [120]

    # N_mean_EA: mean-EA FEM axial force estimate — depends on current geometry AND stock
    assigned_E    = stock_df.iloc[milp_assignment]["E"].values
    assigned_area = (stock_df.iloc[milp_assignment]["Width_m"].values *
                     stock_df.iloc[milp_assignment]["Depth_m"].values)
    mean_EA_SI    = float((assigned_E * assigned_area).mean())
    member_forces = c24_stage_feasibility.estimate_member_forces(
        node_positions  = node_positions,
        edges_v1        = ei_np[0],
        edges_v2        = ei_np[1],
        support_nodes   = support_nodes,
        load_nodes      = load_nodes,
        total_load_n    = c24_stage_feasibility.TOTAL_LOAD_N,
        mean_EA_SI      = mean_EA_SI,
    )  # [120]

    x         = _build_node_features(
        node_positions, device, norm_stats, support_nodes, load_nodes
    )
    edge_attr = _build_edge_features(
        milp_assignment, stock_df, member_lengths_m, member_forces,
        device, norm_stats, bidirectional,
    )

    with torch.no_grad():
        preds = model(x, edge_index=edge_index, edge_attr=edge_attr)

    preds_physical    = preds[:NUM_EDGES_PHYSICAL, 0].cpu().numpy()   # [120] always
    unsafe_flags      = preds_physical >= threshold
    unsafe_member_ids = np.where(unsafe_flags)[0].tolist()
    feasibility_score = float(1.0 - unsafe_flags.mean())

    return feasibility_score, unsafe_member_ids, preds_physical


# =============================================================================
# ORCHESTRATION — single call per GA iteration
# =============================================================================

def run_gnn_stage(
    node_positions:  np.ndarray,
    milp_assignment: np.ndarray,
    df_input_stock:  pd.DataFrame,
    model_bundle:    dict,
    threshold:       float | None = None,
    w_structural:    float        = 0.0,
    print_summary:   bool         = True,
    stock_df:        "pd.DataFrame | None" = None,
    support_nodes:   list[int] | None      = None,
    load_nodes:      list[int] | None      = None,
) -> dict[str, Any]:
    """
    Orchestrate the full GNN feasibility stage for one GA iteration.

    Parameters
    ----------
    node_positions  : [n_nodes, 3] current node xyz from GA (metres)
    milp_assignment : [120] integer array from milp_out["milp_assignment"]
    df_input_stock  : raw stock inventory (mm / N/mm²). Ignored when stock_df
                      is provided.
    model_bundle    : output of load_surrogate_bundle() — load once, reuse.
    threshold       : P(unsafe) decision threshold. None → read from bundle.
    w_structural    : weight for structural_penalty in return dict.
                      For notebook convenience only — the GA evaluator reads
                      feasibility_score directly and ignores structural_penalty.
                      Default 0.0 (penalty term off unless explicitly set).
    print_summary   : whether to print the GNN result summary.
    stock_df        : pre-prepared stock table (output of prepare_stock_for_gnn).
                      When provided, df_input_stock is ignored and
                      prepare_stock_for_gnn() is not called. Pass this to avoid
                      repeating the conversion on every GA evaluation.
    support_nodes   : node indices with fixed supports. None → default 5x3 grid.
    load_nodes      : node indices with applied load. None → default 5x3 grid.

    Returns
    -------
    dict with keys:
        feasibility_score  — float [0,1]: fraction of members predicted safe
        structural_penalty — float: w_structural * (1 - feasibility_score)
        unsafe_member_ids  — list[int]: member indices predicted unsafe (0–119)
        preds_physical     — np.ndarray [120]: raw P(unsafe) per member
        n_unsafe           — int
        n_safe             — int
    """
    if stock_df is None:
        stock_df = prepare_stock_for_gnn(df_input_stock)

    feasibility_score, unsafe_member_ids, preds_physical = gnn_feasibility(
        node_positions  = node_positions,
        milp_assignment = milp_assignment,
        stock_df        = stock_df,
        model_bundle    = model_bundle,
        threshold       = threshold,
        support_nodes   = support_nodes,
        load_nodes      = load_nodes,
    )

    structural_penalty = float(w_structural * (1.0 - feasibility_score))
    n_unsafe = len(unsafe_member_ids)
    n_safe   = NUM_EDGES_PHYSICAL - n_unsafe

    if print_summary:
        bidir_str = "bidirectional" if model_bundle["bidirectional"] else "unidirectional"
        thr       = threshold if threshold is not None else float(
            model_bundle.get("config", {}).get("recommended_threshold", THRESHOLD)
        )
        print(f"\n[GNN] Feasibility Results  ({bidir_str}, {model_bundle['num_edges']} edges):")
        print(f"  Feasibility score:  {feasibility_score:.4f}  "
              f"(1.0 = all members predicted safe)")
        print(f"  Safe members:       {n_safe} / {NUM_EDGES_PHYSICAL}")
        print(f"  Unsafe members:     {n_unsafe} / {NUM_EDGES_PHYSICAL}")
        print(f"  Threshold:          {thr:.2f}")
        if w_structural > 0.0:
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
