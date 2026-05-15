# =============================================================================
# c23_ga_evaluator.py — Design Evaluator + One-Time Bounds
# =============================================================================
#
# Changes vs v4:
#   5. stock_df_raw=df_stock added to run_milp_stage() in both
#      evaluate_design_candidate() and _compute_one_time_normalization_constants().
#      milp_assignment is now built once inside run_milp_stage() and read from
#      milp_out["milp_assignment"] directly — stage_gnn.build_milp_assignment()
#      is no longer called in the evaluator.
#   6. _normalize_bounds_constants() splits validation into specific error
#      messages per constant (C_max / R_max / W_max) instead of a single
#      catch-all. R_max = 0.0 now names the stock composition cause and fix.
#   7. _compute_one_time_normalization_constants() catches stock-composition
#      ValueErrors immediately (skips remaining probes) vs transient errors
#      (continues retrying).
#   8. evaluate_design_candidate() accepts prepared_stock (pre-computed output
#      of stage_cost.prepare_stock_cost_inputs) and forwards it to
#      build_cost_matrix() — avoids repeating the stock prep on every GA call.
#   9. evaluate_design_candidate() accepts prepared_gnn_stock (pre-computed
#      output of stage_gnn.prepare_stock_for_gnn) and passes it to
#      run_gnn_stage() — avoids copying and converting the stock DataFrame on
#      every GNN call. Also passes support_nodes / load_nodes from the derived
#      geometry so GNN receives correct boundary condition features.
#
# Changes vs v3 (carried forward):
#   1. v_idx derivation added to _compute_one_time_normalization_constants().
#   2. Duplicate v_idx derivation removed from evaluate_design_candidate().
#   3. build_logs=True added to build_cost_matrix() in bounds probe.

import warnings
import numpy as np
import pandas as pd
import config

from c21_surrogate_io import load_surrogate_bundle
from workflows import c22_stage_geometry             as stage_geometry
from workflows import c24_stage_feasibility          as stage_feas       # stage_feas throughout
from workflows import c25_stage_cost_matrix          as stage_cost
from workflows import c26_stage_MILP                 as stage_milp
from workflows import c27_stage_GNN                  as stage_gnn
from workflows import c28_stage_fitness_score        as stage_fitness
from workflows import c28_stage_normalization_bounds as stage_bounds


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _resolve_weight_config(
    config_dict:  dict,
    w_structural: float = 0.3,
) -> dict:
    weights = config_dict.get("fitness_weights", {})
    if isinstance(weights, dict):
        return {
            "omega_1": float(weights.get("omega_1", 1.0)),
            "omega_2": float(weights.get("omega_2", 1.0)),
            "omega_3": float(weights.get("omega_3", 1.0)),
            "omega_4": float(w_structural),
        }
    strategy = str(config_dict.get("weight_strategy", "balanced")).strip().lower()
    base = {
        "cost-dominant":  {"omega_1": 1.0, "omega_2": 0.6, "omega_3": 0.8},
        "reuse-dominant": {"omega_1": 0.8, "omega_2": 1.0, "omega_3": 0.8},
        "waste-dominant": {"omega_1": 0.8, "omega_2": 0.6, "omega_3": 1.0},
    }.get(strategy, {"omega_1": 1.0, "omega_2": 1.0, "omega_3": 1.0})
    base["omega_4"] = float(w_structural)
    return base


def _resolve_w_structural(
    config_dict:     dict,
    generation:      int = 0,
    max_generations: int = 1,
) -> float:
    w_start = float(config_dict.get("w_structural_start", 0.2))
    w_end   = float(config_dict.get("w_structural_end",   0.8))
    if max_generations <= 1:
        return w_end
    t = min(generation / max_generations, 1.0)
    return w_start + (w_end - w_start) * t


def _normalize_bounds_constants(constants: dict) -> dict:
    out = {
        "C_max": float(constants.get("C_max", np.nan)),
        "R_max": float(constants.get("R_max", np.nan)),
        "W_max": float(constants.get("W_max", np.nan)),
    }
    if not all(np.isfinite(list(out.values()))):
        raise ValueError(f"Normalization constants contain non-finite values: {out}")
    if out["R_max"] <= 0.0:
        raise ValueError(
            "R_max = 0.0: the stock pool contains no reclaimed elements and the "
            "reuse objective cannot be normalised. Either supply reclaimed stock, "
            "or set GA_CONFIG['fitness_weights']['omega_2'] = 0.0 to disable the "
            "reuse component before running the GA."
        )
    if out["C_max"] <= 0.0:
        raise ValueError(
            f"C_max = {out['C_max']}: maximum assignment cost is zero or negative. "
            "Check that the cost matrix contains at least one finite positive entry."
        )
    if out["W_max"] <= 0.0:
        raise ValueError(
            f"W_max = {out['W_max']}: maximum waste volume is zero or negative. "
            "Check that the stock pool contains elements longer than their assigned slots."
        )
    return out


def _derive_node_roles(df_vertices: pd.DataFrame) -> tuple:
    """
    Derive node_positions, support_nodes, load_nodes from df_vertices.
    Handles v_idx derivation from vertex_index column (e.g. 'v0', 'v1'...).
    """
    verts = df_vertices.copy()
    verts["v_idx"] = verts["vertex_index"].str.replace("v", "", regex=False).astype(int)
    verts = verts.sort_values("v_idx").reset_index(drop=True)
    node_positions = verts[["x", "y", "z"]].values
    support_nodes  = verts[verts["attribute"] == "support"]["v_idx"].tolist()
    load_nodes     = verts[verts["attribute"] == "load"]["v_idx"].tolist()
    return verts, node_positions, support_nodes, load_nodes


# =============================================================================
# ONE-TIME NORMALISATION BOUNDS
# =============================================================================

def _compute_one_time_normalization_constants(
    search_space: dict,
    df_stock,
    config_dict:  dict,
) -> tuple[dict, dict]:
    """
    Derive C_max / R_max / W_max by running the full pipeline on random
    probe designs once before the GA loop starts.
    """
    defaults = stage_fitness.get_default_normalization_constants()

    if not bool(config_dict.get("use_one_time_bounds", True)):
        return defaults, {
            "source": "defaults",
            "reason": "one-time bounds disabled in GA_CONFIG",
        }

    attempts           = max(int(config_dict.get("bounds_probe_attempts", 8)), 1)
    new_stock_max_uses = config_dict.get("new_stock_max_uses", None)

    for attempt_idx in range(attempts):
        try:
            probe_design = stage_geometry.sample_random_design(search_space)

            geo_out = stage_geometry.run_geometry_from_design(
                design_params = probe_design,
                sample_id     = 10_000 + attempt_idx,
            )

            # Fix #1: derive v_idx here — was missing, causing KeyError 'v_idx'
            _, node_positions, support_nodes, load_nodes = _derive_node_roles(
                geo_out["df_vertices"]
            )

            df_slots, feasibility_mask, member_forces, _ = stage_feas.build_cost_filter(
                node_positions = node_positions,
                edges_df       = geo_out["df_edges"],
                stock_df       = df_stock,
                support_nodes  = support_nodes,
                load_nodes     = load_nodes,
            )

            cost_matrix, stock_prepared, logs = stage_cost.build_cost_matrix(
                df_slots         = df_slots,
                df_input_stock   = df_stock,
                feasibility_mask = feasibility_mask,
                build_logs       = True,
            )

            milp_out = stage_milp.run_milp_stage(
                cost_matrix    = cost_matrix,
                enriched_stock = stock_prepared,
                df_slots       = df_slots,
                stock_df_raw              = df_stock,
                reclaimed_marker          = "RS",
                new_marker                = "NS",
                new_stock_max_uses        = (
                    None if new_stock_max_uses is None
                    else int(new_stock_max_uses)
                ),
                solver_msg                = False,
                raise_on_infeasible_slots = False,
            )

            if milp_out["status"] != "Optimal":
                print(f"  [bounds probe {attempt_idx+1}] "
                      f"MILP {milp_out['status']} — retrying")
                continue

            bounds_out = stage_bounds.run_normalization_bounds_stage(
                cost_matrix    = cost_matrix,
                df_logs        = logs,
                enriched_stock = stock_prepared,
                df_slots       = df_slots,
                reclaimed_marker   = "RS",
                new_marker         = "NS",
                new_stock_max_uses = (
                    None if new_stock_max_uses is None
                    else int(new_stock_max_uses)
                ),
                solver_msg    = False,
                print_summary = False,
            )

            if str(bounds_out.get("status", "")).lower() not in {"optimal", "partial"}:
                print(f"  [bounds probe {attempt_idx+1}] bounds status "
                      f"'{bounds_out.get('status')}' — retrying")
                continue

            normalized = _normalize_bounds_constants(
                bounds_out["normalization_constants"]
            )
            print(f"  [bounds probe {attempt_idx+1}] success → {normalized}")
            return normalized, {
                "source":  "one-time-bounds",
                "status":  bounds_out.get("status"),
                "attempt": attempt_idx + 1,
            }

        except ValueError as exc:
            msg = str(exc)
            if any(k in msg for k in ("R_max", "C_max", "W_max")):
                # Stock composition problem — retrying will not help.
                warnings.warn(
                    f"Bounds probe failed due to stock composition: {exc}\n"
                    "Falling back to defaults. Check GA_CONFIG fitness weights.",
                    stacklevel=2,
                )
                return defaults, {
                    "source": "defaults",
                    "reason": f"stock composition: {exc}",
                }
            print(f"  [bounds probe {attempt_idx+1}] value error: {exc} — retrying")
            continue

        except Exception as exc:
            print(f"  [bounds probe {attempt_idx+1}] exception: {exc} — retrying")
            continue

    warnings.warn(
        f"Could not derive valid one-time bounds after {attempts} attempts. "
        "Falling back to default normalization constants.",
        stacklevel=2,
    )
    return defaults, {
        "source": "defaults",
        "reason": f"all {attempts} probe attempts failed",
    }


# =============================================================================
# DESIGN EVALUATOR
# =============================================================================

def evaluate_design_candidate(
    design_params:        dict,
    df_stock,
    fixed_norm_constants: dict,
    config_dict:          dict,
    bundle                    = None,
    model_prefix               = None,
    generation:           int  = 0,
    max_generations:      int  = 1,
    sample_id:            int  = 0,
    verbose:              bool = False,
    prepared_stock:       "pd.DataFrame | None" = None,
    prepared_gnn_stock:  "pd.DataFrame | None" = None,
) -> dict:
    """
    Evaluate one design candidate through the full pipeline.

    Pipeline:
        geometry → feasibility (slots + forces) → cost matrix
        → MILP → GNN (on MILP assignment) → fitness
    """
    penalty = float(config_dict["penalty_fitness"])
    result  = {
        "design_params":       design_params,
        "status":              "UNKNOWN",
        "fitness":             penalty,
        "reason":              None,
        "fitness_result":      None,
        "milp_status":         None,
        "total_cost":          float("inf"),
        "reuse_rate":          0.0,
        "waste_total":         float("inf"),
        "gnn_feasibility":     None,
        "gnn_unsafe_members":  None,
        "preds_physical":      None,
        "w_structural":        None,
        "df_vertices":         None,
        "df_edges":            None,
        "df_results":          None,
    }
    
    MODEL_PREFIX = model_prefix or config_dict.get("MODEL_PREFIX", None)

    try:
        # ---- geometry -------------------------------------------------------
        geo_out = stage_geometry.run_geometry_from_design(
            design_params = design_params,
            sample_id     = int(sample_id),
        )
        df_edges = geo_out["df_edges"]

        # Fix #2: single clean v_idx derivation (was duplicated in v3)
        df_vertices, node_positions, support_nodes, load_nodes = _derive_node_roles(
            geo_out["df_vertices"]
        )
        if verbose:
            print(f"    ✓ geometry    | {len(df_vertices)} nodes, {len(df_edges)} edges")

        # ---- feasibility (slots + member forces) ----------------------------
        df_slots, feasibility_mask, member_forces, _ = stage_feas.build_cost_filter(
            node_positions = node_positions,
            edges_df       = df_edges,
            stock_df       = df_stock,
            support_nodes  = support_nodes,
            load_nodes     = load_nodes,
        )
        if verbose:
            n_feasible = int(feasibility_mask.sum())
            print(f"    ✓ feasibility | {n_feasible:,} feasible slot/stock pairs")

        # ---- cost matrix ----------------------------------------------------
        cost_matrix, stock_prepared, _ = stage_cost.build_cost_matrix(
            df_slots         = df_slots,
            df_input_stock   = df_stock,
            feasibility_mask = feasibility_mask,
            prepared_stock   = prepared_stock,
        )
        if verbose:
            finite_entries = int(np.isfinite(cost_matrix).sum())
            print(f"    ✓ cost matrix | {finite_entries:,} finite entries")

        # ---- MILP -----------------------------------------------------------
        new_stock_max_uses = config_dict.get("new_stock_max_uses", None)
        milp_out = stage_milp.run_milp_stage(
            cost_matrix    = cost_matrix,
            enriched_stock = stock_prepared,
            df_slots       = df_slots,
            stock_df_raw              = df_stock,
            reclaimed_marker          = "RS",
            new_marker                = "NS",
            new_stock_max_uses        = (
                None if new_stock_max_uses is None
                else int(new_stock_max_uses)
            ),
            solver_msg                = False,
            raise_on_infeasible_slots = False,
        )

        result["milp_status"] = milp_out["status"]
        if milp_out["status"] != "Optimal":
            if verbose:
                print(f"    ✗ MILP        | status={milp_out['status']} → PENALIZED")
            result["status"] = "PENALIZED"
            result["reason"] = f"MILP status: {milp_out['status']}"
            return result

        df_results = milp_out["df_results"]
        total_cost = milp_out["total_cost"]
        if verbose:
            print(f"    ✓ MILP        | status=Optimal, cost={total_cost:.4f}, "
                  f"{len(df_results)} assignments")

        # ---- GNN feasibility (on MILP assignment built inside run_milp_stage) --
        gnn_feasibility    = 1.0
        gnn_unsafe_members = []

        if MODEL_PREFIX:
            milp_assignment = milp_out.get("milp_assignment")
            if milp_assignment is None:
                warnings.warn(
                    "milp_out['milp_assignment'] is None — GNN stage skipped. "
                    "Ensure stock_df_raw is passed to run_milp_stage().",
                    stacklevel=2,
                )
                if verbose:
                    print(f"    - GNN         | skipped (milp_assignment is None)")
            else:
                model_bundle = bundle if bundle is not None else load_surrogate_bundle(prefix_sm=MODEL_PREFIX)
                gnn_out = stage_gnn.run_gnn_stage(
                    node_positions  = node_positions,
                    milp_assignment = milp_assignment,
                    df_input_stock  = df_stock,
                    model_bundle    = model_bundle,
                    print_summary   = False,
                    stock_df        = prepared_gnn_stock,
                    support_nodes   = support_nodes,
                    load_nodes      = load_nodes,
                )
                gnn_feasibility    = float(gnn_out["feasibility_score"])
                gnn_unsafe_members = gnn_out["unsafe_member_ids"]
                preds_physical     = gnn_out["preds_physical"]
                if verbose:
                    print(f"    ✓ GNN         | feasibility={gnn_feasibility:.2%}, "
                          f"unsafe={len(gnn_unsafe_members)} members")
        else:
            warnings.warn(
                "MODEL_PREFIX not set — GNN stage skipped. "
                "Set MODEL_PREFIX in GA_CONFIG.",
                stacklevel=2,
            )
            if verbose:
                print(f"    - GNN         | skipped (MODEL_PREFIX not set)")

        result["gnn_feasibility"]    = gnn_feasibility
        result["gnn_unsafe_members"] = gnn_unsafe_members

        # ---- w_structural curriculum ----------------------------------------
        w_structural = _resolve_w_structural(
            config_dict     = config_dict,
            generation      = generation,
            max_generations = max_generations,
        )
        result["w_structural"] = w_structural

        # ---- fitness --------------------------------------------------------
        weight_config = _resolve_weight_config(config_dict, w_structural)

        fitness_out = stage_fitness.run_fitness_stage(
            df_results               = df_results,
            enriched_stock           = stock_prepared,
            df_slots                 = df_slots,
            total_cost               = total_cost,
            weight_config            = weight_config,
            normalization_constants  = fixed_norm_constants,
            structural_infeasibility = 1.0 - gnn_feasibility,
            derive_normalization_constants = False,
            run_sanity_checks        = False,
            print_breakdown          = False,
        )

        fitness_result = fitness_out["fitness_result"]
        result.update({
            "status":         "OK",
            "fitness":        float(fitness_result["fitness"]),
            "reason":         None,
            "fitness_result": fitness_result,
            "total_cost":     float(fitness_result.get("cost_raw", total_cost)),
            "reuse_rate":     float(fitness_result.get("reuse_rate", 0.0)),
            "waste_total":    float(fitness_result.get("waste_total", 0.0)),
            "df_vertices":    df_vertices,
            "df_edges":       df_edges,
            "df_results":     df_results,
            "preds_physical": preds_physical if MODEL_PREFIX else None,
        })

        if verbose:
            print(
                f"    ✓ fitness     | fitness={result['fitness']:.4f}, "
                f"cost={result['total_cost']:.2f}, "
                f"reuse={result['reuse_rate']:.1f}%, "
                f"waste={result['waste_total']:.4f}, "
                f"ω4={w_structural:.2f}"
            )

    except Exception as exc:
        result["status"] = "PENALIZED"
        result["reason"] = str(exc)

    return result