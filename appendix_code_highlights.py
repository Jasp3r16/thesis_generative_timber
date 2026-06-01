"""
Appendix — Code Highlights
==========================
MSc Thesis: Deep Generative Design for Reclaimed Timber Structures
Author: Jasper Cluistra | Building Technology, TU Delft

This file contains four representative excerpts from the full pipeline codebase,
presented in the same sequence as the methodology chapter:

    1. LCA cost formula       — c25_stage_cost_matrix.py  (§4.2)
    2. MILP assignment        — c26_stage_MILP.py          (§4.3)
    3. GNN model definition   — c21_surrogate_model_v4.py  (§3.2)
    4. Pipeline evaluator     — c23_ga_evaluator.py        (§4.4)

The complete source code, notebooks, data, and requirements are available
in the GitHub repository. These excerpts are copied verbatim from the
production codebase and are intended to give a practical impression of
how the workflow is implemented.
"""


# =============================================================================
# 1. LCA COST FORMULA  —  c25_stage_cost_matrix.py
# =============================================================================
#
# _compute_lca_vectors() is the single source of truth for embodied carbon
# costs across all (slot, stock) pairs. It is called inside build_cost_matrix()
# to populate the [n_slots × n_stock] cost matrix that the MILP minimises.
#
# Transport basis:
#   New stock:       required mass only  (bought cut-to-size; waste not shipped)
#   Reclaimed stock: full stock mass     (whole physical element moved to site)
#
# LCA constants (from c00_headquarter_params.py):
#   M_A1_A3       = 0.25     kg CO₂e/kg  — embodied carbon, new softwood (A1–A3)
#   M_RECOVER     = 0.0085   kg CO₂e/kg  — C1 deconstruction energy, reclaimed
#   E_PREP        = 0.010    kg CO₂e/kg  — A5 preparation (cleaning, de-nailing)
#   E_SAW         = 0.004    kg CO₂e/kg  — A5 sawing (cross-cut, only if needed)
#   E_OFFCUT      = 0.031    kg CO₂e/kg  — C3+C4 offcut disposal/incineration
#   WASTE_DIST_KM = 50       km          — C2 offcut transport distance
#   SCARCITY_PENALTY = 0.0              — disabled in all reported runs
# =============================================================================

import numpy as np


def _compute_lca_vectors(slot_rows, stock_rows):
    """Compute all LCA quantities for a set of aligned (slot, stock) pairs.

    Both DataFrames must have the same length; row i in slot_rows corresponds
    to row i in stock_rows. stock_rows must be a prepared stock table
    (output of prepare_stock_cost_inputs).

    Returns a dict of NumPy arrays covering individual LCA components
    and the summed total_costs.
    """
    # LCA constants
    M_A1_A3, M_RECOVER  = 0.25, 0.0085
    E_PREP, E_SAW       = 0.010, 0.004
    E_OFFCUT            = 0.031
    WASTE_DIST_KM       = 50
    SCARCITY_PENALTY    = 0.0

    # Slot geometry (required cut dimensions)
    req_length_m = slot_rows["length_m"].values.astype(float)
    req_area_m2  = (slot_rows["Width_Req"].values.astype(float) *
                    slot_rows["Depth_Req"].values.astype(float)) / 1_000_000.0

    # Stock geometry (physical piece dimensions)
    stk_length_m = stock_rows["Length_Resolved"].values.astype(float) / 1000.0
    stk_area_m2  = (stock_rows["Width_Resolved"].values.astype(float) *
                    stock_rows["Depth_Resolved"].values.astype(float)) / 1_000_000.0

    # Material and transport properties
    density      = stock_rows["Density_Resolved"].values.astype(float)
    distance_km  = stock_rows["Distance_Resolved"].values.astype(float)
    trans_factor = stock_rows["TransportFactor_Resolved"].values.astype(float) / 1000.0
    is_reclaimed = stock_rows["State_Resolved"].values.astype(float) >= 0.5

    # Volumes and masses
    v_req        = req_area_m2  * req_length_m
    v_stock      = stk_area_m2  * stk_length_m
    v_waste      = np.maximum(0.0, stk_area_m2 * (stk_length_m - req_length_m))
    needs_sawing = v_waste > 0.0
    mass_req     = v_req   * density
    mass_stock   = v_stock * density
    mass_waste   = v_waste * density

    # LCA components — vectorised, branching on is_reclaimed
    e_embodied   = np.where(is_reclaimed, 0.0,        mass_req   * M_A1_A3)
    e_transport  = np.where(is_reclaimed, mass_stock,  mass_req)  * distance_km * trans_factor
    e_recovered  = np.where(is_reclaimed, mass_stock  * M_RECOVER,  0.0)
    e_prep       = np.where(is_reclaimed, mass_stock  * E_PREP,     0.0)
    e_saw        = np.where(is_reclaimed & needs_sawing, mass_stock * E_SAW, 0.0)
    e_waste_c2   = np.where(is_reclaimed, mass_waste  * WASTE_DIST_KM * trans_factor, 0.0)
    e_waste_c3c4 = np.where(is_reclaimed, mass_waste  * E_OFFCUT,   0.0)
    e_scarcity   = np.where(is_reclaimed, v_waste     * SCARCITY_PENALTY, 0.0)
    e_waste      = e_waste_c2 + e_waste_c3c4

    total_costs = np.where(
        is_reclaimed,
        e_recovered + e_transport + e_prep + e_saw + e_waste + e_scarcity,
        e_embodied  + e_transport,
    )

    return {
        "v_req": v_req, "v_stock": v_stock, "v_waste": v_waste,
        "mass_req": mass_req, "mass_stock": mass_stock, "mass_waste": mass_waste,
        "e_embodied": e_embodied, "e_transport": e_transport,
        "e_recovered": e_recovered, "e_prep": e_prep, "e_saw": e_saw,
        "e_waste_c2": e_waste_c2, "e_waste_c3c4": e_waste_c3c4,
        "e_waste": e_waste, "e_scarcity": e_scarcity,
        "is_reclaimed": is_reclaimed, "total_costs": total_costs,
    }


# =============================================================================
# 2. MILP ASSIGNMENT  —  c26_stage_MILP.py
# =============================================================================
#
# run_milp_stage() formulates and solves the integer programme that assigns
# one stock element to each of the 120 structural slots. The objective
# minimises total LCA cost subject to four constraint classes:
#
#   (a) Coverage:    every slot receives exactly one stock element
#   (b) Uniqueness:  each reclaimed piece is used at most once
#   (c) New-use cap: each NS section ID is used at most new_stock_max_uses times
#   (d) Reuse floor: optionally enforces a minimum reclaimed fraction
# =============================================================================

import pulp
import pandas as pd


def run_milp_stage(
    cost_matrix,
    enriched_stock,
    df_slots,
    stock_df_raw=None,
    new_stock_max_uses=None,
    min_reuse_fraction=None,
    solver_msg=False,
    solver_time_limit=30,
    raise_on_infeasible_slots=True,
):
    """
    Run MILP timber assignment for one cost matrix.

    Parameters
    ----------
    cost_matrix : np.ndarray float [n_slots, n_stock]
        LCA cost matrix. inf entries are infeasible and will not be selected.
    enriched_stock : pd.DataFrame
        Prepared stock table with Member_ID and State_Resolved columns.
    df_slots : pd.DataFrame
        Slot table with column edge_id.
    new_stock_max_uses : int | None
        Maximum times a new stock element can be assigned across all slots.
        None = unlimited. Set to 10 to enforce the new_stock_max_uses=10 scenario.
    min_reuse_fraction : float | None
        Minimum fraction of slots that must be assigned a reclaimed element.
    """
    stock_items        = enriched_stock["Member_ID"].astype(str).tolist()
    construction_slots = df_slots["edge_id"].astype(str).tolist()

    # Separate RS (reclaimed) and NS (new) items
    stock_state     = (enriched_stock["State_Resolved"].astype(float) >= 0.5).astype(int)
    reclaimed_items = enriched_stock.loc[stock_state == 1, "Member_ID"].astype(str).tolist()
    new_items       = enriched_stock.loc[stock_state == 0, "Member_ID"].astype(str).tolist()

    # Build valid (stock, slot) match lists from finite cost matrix entries
    finite_positions = np.argwhere(np.isfinite(cost_matrix))
    valid_matches, costs = [], {}
    slot_to_stocks = {sid: [] for sid in construction_slots}
    stock_to_slots = {sid: [] for sid in stock_items}

    for i, j in finite_positions:
        slot_id  = construction_slots[int(i)]
        stock_id = stock_items[int(j)]
        match    = (stock_id, slot_id)
        valid_matches.append(match)
        costs[match] = float(cost_matrix[int(i), int(j)])
        slot_to_stocks[slot_id].append(stock_id)
        stock_to_slots[stock_id].append(slot_id)

    # Early exit: abort if any slot has no feasible stock option
    infeasible_slots = [sid for sid, opts in slot_to_stocks.items() if not opts]
    if infeasible_slots and raise_on_infeasible_slots:
        raise ValueError(f"MILP aborted: {len(infeasible_slots)} slot(s) have no feasible stock.")

    # ── Build PuLP problem ────────────────────────────────────────────────────
    problem = pulp.LpProblem("Timber_Matching", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("Match", valid_matches, lowBound=0, upBound=1, cat=pulp.LpBinary)

    # Objective: minimise total LCA cost
    problem += pulp.lpSum(x[m] * costs[m] for m in valid_matches)

    # (a) Coverage: each slot must be assigned exactly one stock element
    for slot_id, options in slot_to_stocks.items():
        problem += pulp.lpSum(x[(stock_id, slot_id)] for stock_id in options) == 1

    # (b) Uniqueness: each reclaimed piece can only be used once
    for stock_id in reclaimed_items:
        if stock_to_slots.get(stock_id):
            problem += (
                pulp.lpSum(x[(stock_id, slot_id)] for slot_id in stock_to_slots[stock_id])
                <= 1
            )

    # (c) New-use cap: add constraint only when explicitly restricted
    #     new_stock_max_uses=None means unlimited — no constraint added
    if new_stock_max_uses is not None:
        for stock_id in new_items:
            if stock_to_slots.get(stock_id):
                problem += (
                    pulp.lpSum(x[(stock_id, slot_id)] for slot_id in stock_to_slots[stock_id])
                    <= int(new_stock_max_uses)
                )

    # (d) Reuse floor: capped by what the stock pool physically allows
    if min_reuse_fraction is not None and min_reuse_fraction > 0.0:
        reclaimed_set   = set(reclaimed_items)
        n_slots_with_rs = sum(1 for sid in construction_slots
                              if any(s in reclaimed_set for s in slot_to_stocks[sid]))
        min_reuse_target = min(
            int(min_reuse_fraction * len(construction_slots)),
            min(n_slots_with_rs, len(reclaimed_items)),
        )
        if min_reuse_target > 0:
            problem += (
                pulp.lpSum(
                    x[(sid, slot_id)]
                    for sid in reclaimed_items
                    for slot_id in stock_to_slots.get(sid, [])
                ) >= min_reuse_target,
                "min_reuse_constraint",
            )

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(msg=solver_msg, timeLimit=solver_time_limit,
                               options=["RandomSeed 0"])
    solver.solve(problem)
    status = pulp.LpStatus[problem.status]

    if status == "Optimal":
        total_cost = float(pulp.value(problem.objective))
        rows = [
            {"edge_id": slot_id, "assigned_timber": stock_id,
             "CO2_Penalty": float(costs[(stock_id, slot_id)])}
            for stock_id, slot_id in valid_matches
            if x[(stock_id, slot_id)].varValue is not None
            and x[(stock_id, slot_id)].varValue > 0.5
        ]
        df_results = pd.DataFrame(rows)
    else:
        total_cost = float("inf")
        df_results = pd.DataFrame(columns=["edge_id", "assigned_timber", "CO2_Penalty"])

    return {"status": status, "total_cost": total_cost, "df_results": df_results}


# =============================================================================
# 3. GNN MODEL DEFINITION  —  c21_surrogate_model_v4.py
# =============================================================================
#
# TrussEdgeSafetyGNN predicts per-member failure probability P(UC > 1.0) for
# the 120-member timber truss. It replaces a full Karamba3D FEA call (~seconds)
# with a forward pass (~milliseconds), making 7,500 GA evaluations per run
# computationally tractable.
#
# Architecture: NodeEncoder → NNConv stack (4 layers) → EdgeDecoder
#   Node features (10D): x, y, z + boundary condition flags + applied load Fz
#   Edge features  (9D): Width_m, Depth_m, Length, E, Iy, Iz, J, EA/L, N_mean_EA
#
# Key design decisions:
#   - NNConv with adaptive edge weights mimics FEA stiffness assembly
#   - Residual connections from layer 0 prevent over-smoothing
#   - Symmetric EdgeDecoder (|h_i−h_j|, h_i⊙h_j) is invariant to edge direction
#   - Topology cached once; edge_attr varies per sample (cross-sections change)
# =============================================================================

import torch
import torch.nn as nn
from torch_geometric.nn import NNConv, BatchNorm


class NodeEncoder(nn.Module):
    """Two-layer MLP projecting heterogeneous node features into latent space."""
    def __init__(self, node_features_dim, hidden_dim):
        super().__init__()
        self.fc1       = nn.Linear(node_features_dim, hidden_dim)
        self.fc2       = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class EdgeFeatureMLPFilter(nn.Module):
    """Three-layer MLP mapping edge features to adaptive NNConv weight matrices.

    Stiff members (high EA/L) receive higher filter weights and exert stronger
    influence during message passing — mimicking FEA stiffness assembly.
    """
    def __init__(self, edge_features_dim, out_channels, hidden=64):
        super().__init__()
        self.fc1       = nn.Linear(edge_features_dim, hidden)
        self.fc2       = nn.Linear(hidden, hidden)
        self.fc3       = nn.Linear(hidden, out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, edge_attr):
        h = self.activation(self.fc1(edge_attr))
        h = self.activation(self.fc2(h))
        return self.fc3(h)


class EdgeDecoder(nn.Module):
    """Symmetric prediction head using |h_i−h_j| and h_i⊙h_j.

    Direction-invariant: member AB and BA produce identical predictions,
    consistent with the undirected nature of truss members.
    """
    def __init__(self, hidden_dim, edge_features_dim, dropout_p=0.1):
        super().__init__()
        concat_dim      = 2 * hidden_dim + edge_features_dim
        self.fc1        = nn.Linear(concat_dim, hidden_dim)
        self.fc2        = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3        = nn.Linear(hidden_dim // 2, 1)
        self.activation = nn.LeakyReLU(0.1)
        self.dropout    = nn.Dropout(p=dropout_p)
        self.sigmoid    = nn.Sigmoid()

    def forward(self, h_i, h_j, e_ij):
        diff = torch.abs(h_i - h_j)        # force gradient across member
        prod = h_i * h_j                   # shared activation patterns
        x    = torch.cat([diff, prod, e_ij], dim=1)
        x    = self.dropout(self.activation(self.fc1(x)))
        x    = self.dropout(self.activation(self.fc2(x)))
        return self.sigmoid(self.fc3(x))


class TrussEdgeSafetyGNN(nn.Module):
    """
    End-to-end GNN for per-edge structural safety prediction in timber trusses.

    Input:
        x          [num_nodes, 10]   node features (coordinates + BCs + load)
        edge_index [2, num_edges]    graph connectivity (fixed topology)
        edge_attr  [num_edges, 9]    material + geometric edge features

    Output:
        [num_edges, 1]  P(unsafe) = P(Utilisation > 1.0) per member
    """
    def __init__(self, node_features_dim=10, edge_features_dim=9,
                 hidden_dim=128, num_layers=4, use_batch_norm=True,
                 use_residuals=True, dropout_p=0.1):
        super().__init__()
        self.num_layers     = num_layers
        self.use_batch_norm = use_batch_norm
        self.use_residuals  = use_residuals

        self.node_encoder  = NodeEncoder(node_features_dim, hidden_dim)
        self.nnconv_layers = nn.ModuleList()
        self.batch_norms   = nn.ModuleList() if use_batch_norm else None
        self.dropout       = nn.Dropout(p=dropout_p)
        self.activation    = nn.LeakyReLU(0.1)

        for _ in range(num_layers):
            edge_mlp = EdgeFeatureMLPFilter(
                edge_features_dim=edge_features_dim,
                out_channels=hidden_dim * hidden_dim,
                hidden=64,
            )
            self.nnconv_layers.append(
                NNConv(in_channels=hidden_dim, out_channels=hidden_dim,
                       nn=edge_mlp, aggr='add')
            )
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_dim))

        self.edge_decoder = EdgeDecoder(hidden_dim, edge_features_dim, dropout_p)
        self.register_buffer('edge_index_cache', torch.zeros((2, 1), dtype=torch.long))
        self._is_topology_cached = False

    def cache_topology(self, edge_index):
        """Call once with the fixed edge_index before training or inference."""
        self.edge_index_cache    = edge_index.clone()
        self._is_topology_cached = True

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        if self._is_topology_cached and edge_index is None:
            edge_index = self.edge_index_cache
        if edge_index is None or edge_attr is None:
            raise ValueError("edge_index must be cached or provided; edge_attr always required.")

        h = self.node_encoder(x)

        for layer_idx in range(self.num_layers):
            h_residual = h
            h = self.nnconv_layers[layer_idx](h, edge_index, edge_attr)
            if self.use_batch_norm:
                h = self.batch_norms[layer_idx](h)
            h = self.activation(h)
            h = self.dropout(h)
            if self.use_residuals:
                h = h + h_residual

        src, dst = edge_index[0], edge_index[1]
        return self.edge_decoder(h[src], h[dst], edge_attr)


# =============================================================================
# 4. PIPELINE EVALUATOR  —  c23_ga_evaluator.py
# =============================================================================
#
# evaluate_design_candidate() is the fitness oracle called by CMA-ES on every
# candidate solution. It runs all pipeline stages in sequence and returns a
# scalar fitness value. A penalty is returned immediately if the MILP fails
# or if structural infeasibility exceeds the hard ceiling.
#
# Fitness function:
#   F = ω₁ · Ĉ  −  ω₂ · R̂  +  ω₄ · S      (minimised)
#
#   Ĉ = total_cost / C_max          normalised LCA cost
#   R̂ = reuse_fraction / R_max      normalised reclaimed fraction
#   S  = 1 − GNN_feasibility        structural infeasibility
#   ω₄ annealed from w_structural_start → w_structural_end over max_generations
# =============================================================================

def evaluate_design_candidate(
    design_params,
    df_stock,
    fixed_norm_constants,
    config_dict,
    bundle=None,
    model_prefix=None,
    generation=0,
    max_generations=1,
    sample_id=0,
    verbose=False,
    prepared_stock=None,
    prepared_gnn_stock=None,
):
    """
    Evaluate one design candidate through the full pipeline.

    Pipeline:
        geometry → feasibility filter → cost matrix → MILP → GNN → fitness

    Returns a result dict with keys: status, fitness, total_cost,
    reuse_fraction, gnn_feasibility, waste_total, df_vertices, df_edges,
    df_results.
    """
    from workflows import c22_stage_geometry      as stage_geometry
    from workflows import c24_stage_feasibility   as stage_feas
    from workflows import c25_stage_cost_matrix   as stage_cost
    from workflows import c26_stage_MILP          as stage_milp
    from workflows import c27_stage_GNN           as stage_gnn
    from workflows import c28_stage_fitness_score as stage_fitness

    penalty = float(config_dict["penalty_fitness"])
    result  = {
        "status": "UNKNOWN", "fitness": penalty, "reason": None,
        "milp_status": None, "total_cost": float("inf"),
        "reuse_fraction": 0.0, "gnn_feasibility": None,
        "waste_total": 0.0, "df_vertices": None, "df_edges": None, "df_results": None,
    }

    try:
        # ── Stage 1: geometry ────────────────────────────────────────────────
        geo_out        = stage_geometry.run_geometry_from_design(design_params, sample_id)
        df_edges       = geo_out["df_edges"]
        df_vertices    = geo_out["df_vertices"].copy()
        df_vertices["v_idx"] = (df_vertices["vertex_index"]
                                .str.replace("v", "", regex=False).astype(int))
        df_vertices    = df_vertices.sort_values("v_idx").reset_index(drop=True)
        node_positions = df_vertices[["x", "y", "z"]].values
        support_nodes  = df_vertices[df_vertices["attribute"] == "support"]["v_idx"].tolist()
        load_nodes     = df_vertices[df_vertices["attribute"] == "load"]["v_idx"].tolist()

        # ── Stage 2: feasibility filter (member forces + EC5 check) ──────────
        df_slots, feasibility_mask, _, _ = stage_feas.build_cost_filter(
            node_positions=node_positions, edges_df=df_edges,
            stock_df=df_stock, support_nodes=support_nodes,
            load_nodes=load_nodes, verbose=verbose,
        )

        # ── Stage 3: LCA cost matrix ──────────────────────────────────────────
        cost_matrix, stock_prepared, _ = stage_cost.build_cost_matrix(
            df_slots=df_slots, df_input_stock=df_stock,
            feasibility_mask=feasibility_mask, prepared_stock=prepared_stock,
        )

        # ── Stage 4: MILP assignment ──────────────────────────────────────────
        milp_out = stage_milp.run_milp_stage(
            cost_matrix=cost_matrix, enriched_stock=stock_prepared,
            df_slots=df_slots, stock_df_raw=df_stock,
            new_stock_max_uses=config_dict.get("new_stock_max_uses"),
            min_reuse_fraction=config_dict.get("min_reuse_fraction"),
            solver_msg=False, raise_on_infeasible_slots=False,
        )
        result["milp_status"] = milp_out["status"]
        if milp_out["status"] != "Optimal":
            result.update({"status": "PENALIZED", "reason": f"MILP: {milp_out['status']}"})
            return result

        # ── Stage 5: GNN structural proxy ────────────────────────────────────
        gnn_feasibility = 1.0
        if config_dict.get("use_gnn", True) and (bundle is not None or model_prefix):
            from src.c21_surrogate_io import load_surrogate_bundle
            model_bundle = bundle or load_surrogate_bundle(prefix_sm=model_prefix)
            gnn_out = stage_gnn.run_gnn_stage(
                node_positions=node_positions,
                milp_assignment=milp_out.get("milp_assignment"),
                df_input_stock=df_stock, model_bundle=model_bundle,
                stock_df=prepared_gnn_stock,
                support_nodes=support_nodes, load_nodes=load_nodes,
            )
            gnn_feasibility = float(gnn_out["feasibility_score"])

        # Hard structural floor: penalise if infeasibility exceeds ceiling
        structural_infeasibility = 1.0 - gnn_feasibility
        max_infeas = float(config_dict.get("max_structural_infeas", 1.0))
        if config_dict.get("use_gnn", True) and structural_infeasibility > max_infeas:
            result.update({"status": "PENALIZED",
                           "reason": f"infeas {structural_infeasibility:.3f} > {max_infeas:.3f}"})
            return result

        # ── Stage 6: fitness score ────────────────────────────────────────────
        # ω₄ annealed linearly from w_structural_start → w_structural_end
        t            = min(generation / max(max_generations, 1), 1.0)
        w_start      = float(config_dict.get("w_structural_start", 2.0))
        w_end        = float(config_dict.get("w_structural_end",   0.8))
        w_structural = w_start + (w_end - w_start) * t

        weight_config = {
            "omega_1": float(config_dict.get("fitness_weights", {}).get("omega_1", 1.0)),
            "omega_2": float(config_dict.get("fitness_weights", {}).get("omega_2", 1.0)),
            "omega_4": w_structural,
        }

        fitness_out = stage_fitness.run_fitness_stage(
            df_results=milp_out["df_results"], enriched_stock=stock_prepared,
            df_slots=df_slots, total_cost=milp_out["total_cost"],
            weight_config=weight_config,
            normalization_constants=fixed_norm_constants,
            structural_infeasibility=structural_infeasibility,
            derive_normalization_constants=False,
        )
        fr = fitness_out["fitness_result"]
        result.update({
            "status":          "OK",
            "fitness":         float(fr["fitness"]),
            "total_cost":      float(fr.get("cost_raw", milp_out["total_cost"])),
            "reuse_fraction":  float(fr.get("reuse_fraction", 0.0)),
            "gnn_feasibility": gnn_feasibility,
            "df_vertices":     df_vertices,
            "df_edges":        df_edges,
            "df_results":      milp_out["df_results"],
        })

    except Exception as exc:
        result["status"] = "PENALIZED"
        result["reason"] = str(exc)

    return result
