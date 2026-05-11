"""Fitness aggregation and evaluation for the MILP workflow.

This module orchestrates fitness calculation by combining:
- Individual metric evaluation (cost, reuse rate, waste)
- Normalization of metrics to [0, 1] range
- Multi-objective fitness function with configurable weights
- Sanity checks for normalization and weight configuration
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from workflows.c29_stage_normalization_bounds import get_default_normalization_constants


# ============================================================================
# Column Resolution Helpers
# ============================================================================


def _column_by_lower(df: pd.DataFrame, name: str) -> str | None:
    """Find a DataFrame column case-insensitively."""
    mapping = {str(col).strip().lower(): str(col) for col in df.columns}
    return mapping.get(name.strip().lower())


# ============================================================================
# Weight Configuration
# ============================================================================


def validate_weight_config(weight_config: dict[str, float]) -> dict[str, float]:
    """Validate and normalize a notebook-provided weight config.

    omega_4 is optional (default 0.0 = structural penalty disabled).
    Set omega_4 > 0 to include the GNN structural feasibility penalty.
    Recommended: omega_4 = 0.5 gives structural penalty ~50% weight of each
    LCA term, enough to meaningfully penalise infeasible geometries without
    dominating the LCA objectives.
    """
    required = ("omega_1", "omega_2", "omega_3")
    missing = [key for key in required if key not in weight_config]
    if missing:
        raise ValueError(f"Missing keys in weight_config: {', '.join(missing)}")

    omega_1 = float(weight_config["omega_1"])
    omega_2 = float(weight_config["omega_2"])
    omega_3 = float(weight_config["omega_3"])
    omega_4 = float(weight_config.get("omega_4", 0.0))   # structural — optional

    if omega_1 < 0 or omega_2 < 0 or omega_3 < 0 or omega_4 < 0:
        raise ValueError("All weights must be >= 0")

    return {
        "omega_1": omega_1,
        "omega_2": omega_2,
        "omega_3": omega_3,
        "omega_4": omega_4,
    }


def weights_from_config(weight_config: dict[str, float]) -> tuple[float, float, float, float]:
    """Convert a weight config dictionary to the (omega_1, omega_2, omega_3, omega_4) tuple."""
    validated = validate_weight_config(weight_config)
    return (
        float(validated["omega_1"]),
        float(validated["omega_2"]),
        float(validated["omega_3"]),
        float(validated["omega_4"]),
    )


# ============================================================================
# Reclaimed State Resolution
# ============================================================================


def _resolve_stock_state_map(enriched_stock_df: pd.DataFrame) -> dict[str, int]:
    """Build a mapping of stock item IDs to their reclaimed state (0 or 1)."""
    member_col = _column_by_lower(enriched_stock_df, "member_id")
    if member_col is None:
        raise ValueError("enriched_stock_df is missing Member_ID")

    state_col = _column_by_lower(enriched_stock_df, "state_resolved") or _column_by_lower(enriched_stock_df, "state")
    member_ids = enriched_stock_df[member_col].astype(str).str.strip()

    if state_col is not None:
        state_values = pd.to_numeric(enriched_stock_df[state_col], errors="coerce").fillna(0.0)
        resolved = state_values.clip(lower=0.0, upper=1.0).astype(int)
        return {mid: int(state) for mid, state in zip(member_ids, resolved)}

    return {mid: int(mid.upper().startswith("RS")) for mid in member_ids}


# ============================================================================
# Metric Calculation
# ============================================================================


def calculate_reuse_rate(
    milp_results_df: pd.DataFrame,
    enriched_stock_df: pd.DataFrame,
) -> float:
    """Calculate reuse rate as percentage of reclaimed assignments (count-based)."""
    if milp_results_df.empty:
        return 0.0

    if "assigned_timber" not in milp_results_df.columns:
        raise ValueError("milp_results_df is missing assigned_timber")

    state_map = _resolve_stock_state_map(enriched_stock_df)

    assigned = milp_results_df["assigned_timber"].astype(str).str.strip()
    reclaimed_count = int(sum(state_map.get(timber_id, int(timber_id.upper().startswith("RS"))) == 1 for timber_id in assigned))
    total_assignments = int(len(assigned))

    if total_assignments <= 0:
        return 0.0
    return float((reclaimed_count / total_assignments) * 100.0)


def calculate_total_waste(
    milp_results_df: pd.DataFrame,
    df_slots: pd.DataFrame,
    stock_inventory_df: pd.DataFrame,
) -> float:
    """Calculate total waste volume (m3) from MILP assignments."""
    if milp_results_df.empty:
        return 0.0

    required_slot_cols = {
        "edge_id": _column_by_lower(df_slots, "edge_id"),
        "length_req": _column_by_lower(df_slots, "length_req"),
        "width_req": _column_by_lower(df_slots, "width_req"),
        "depth_req": _column_by_lower(df_slots, "depth_req"),
    }
    if any(value is None for value in required_slot_cols.values()):
        raise ValueError("df_slots is missing one or more required columns: edge_id, Length_Req, Width_Req, Depth_Req")

    required_stock_cols = {
        "member_id": _column_by_lower(stock_inventory_df, "member_id"),
        "length": _column_by_lower(stock_inventory_df, "length"),
        "width": _column_by_lower(stock_inventory_df, "width"),
        "depth": _column_by_lower(stock_inventory_df, "depth"),
    }
    if any(value is None for value in required_stock_cols.values()):
        raise ValueError("stock_inventory_df is missing one or more required columns: Member_ID, Length, Width, Depth")

    slot_indexed = df_slots.set_index(required_slot_cols["edge_id"], drop=False)
    stock_indexed = stock_inventory_df.set_index(required_stock_cols["member_id"], drop=False)

    total_waste_m3 = 0.0
    for _, assignment in milp_results_df.iterrows():
        edge_id = str(assignment["edge_id"]).strip()
        timber_id = str(assignment["assigned_timber"]).strip()

        if edge_id not in slot_indexed.index or timber_id not in stock_indexed.index:
            continue

        slot_row = slot_indexed.loc[edge_id]
        stock_row = stock_indexed.loc[timber_id]

        l_req = float(slot_row[required_slot_cols["length_req"]]) / 1000.0
        w_req = float(slot_row[required_slot_cols["width_req"]]) / 1000.0
        d_req = float(slot_row[required_slot_cols["depth_req"]]) / 1000.0

        l_stock = float(stock_row[required_stock_cols["length"]]) / 1000.0
        w_stock = float(stock_row[required_stock_cols["width"]]) / 1000.0
        d_stock = float(stock_row[required_stock_cols["depth"]]) / 1000.0

        oversizing_waste = max(0.0, (w_stock * d_stock - w_req * d_req) * l_req)
        length_waste = max(0.0, w_stock * d_stock * (l_stock - l_req))
        total_waste_m3 += float(oversizing_waste + length_waste)

    return float(total_waste_m3)


def get_inner_cost(milp_result: float) -> float:
    """Return MILP objective value as float."""
    return float(milp_result)


# ============================================================================
# Normalization
# ============================================================================


def normalize_metrics(
    cost: float,
    reuse_rate: float,
    waste: float,
    normalization_constants: dict[str, float],
) -> tuple[float, float, float]:
    """Normalize cost, reuse rate, and waste to [0, 1]."""
    c_max = max(float(normalization_constants["C_max"]), 1e-9)
    r_max = max(float(normalization_constants["R_max"]), 1e-9)
    w_max = max(float(normalization_constants["W_max"]), 1e-9)

    cost_norm = float(np.clip(float(cost) / c_max, 0.0, 1.0))
    reuse_norm = float(np.clip(float(reuse_rate) / r_max, 0.0, 1.0))
    waste_norm = float(np.clip(float(waste) / w_max, 0.0, 1.0))
    return cost_norm, reuse_norm, waste_norm


def derive_normalization_constants_from_solution(
    milp_results_df: pd.DataFrame,
    enriched_stock_df: pd.DataFrame,
    df_slots: pd.DataFrame,
    milp_objective_value: float,
    margin: float = 0.20,
) -> dict[str, float]:
    """Derive normalization constants from the current solution with a margin."""
    if margin < 0:
        raise ValueError("margin must be >= 0")

    cost_raw = get_inner_cost(milp_objective_value)
    reuse_rate = calculate_reuse_rate(milp_results_df, enriched_stock_df)
    waste_total = calculate_total_waste(milp_results_df, df_slots, enriched_stock_df)

    scale = 1.0 + float(margin)
    c_max = max(float(cost_raw) * scale, 1e-9)
    r_max = max(float(reuse_rate) * scale, 100.0)
    w_max = max(float(waste_total) * scale, 1e-9)

    return {
        "C_max": float(c_max),
        "R_max": float(r_max),
        "W_max": float(w_max),
    }


# ============================================================================
# Multi-Objective Fitness
# ============================================================================


def fitness_function_multi_objective(
    cost_norm: float,
    reuse_norm: float,
    waste_norm: float,
    weights: tuple[float, float, float, float],
    structural_infeasibility: float = 0.0,
) -> float:
    """Compute weighted fitness.

    F(x) = omega_1*cost - omega_2*reuse + omega_3*waste + omega_4*structural_infeasibility

    structural_infeasibility: float in [0, 1]
        Fraction of members predicted UNSAFE by the GNN surrogate.
        = 1.0 - feasibility_score from gnn_feasibility()
        = 0.0 when all members predicted safe  (no penalty)
        = 1.0 when all members predicted unsafe (maximum penalty)

    omega_4 = 0.0 (default): structural penalty disabled, fully backward compatible.
    omega_4 = 0.5: structural penalty equivalent to half an LCA term.
    omega_4 = 1.0: structural penalty has same weight as each LCA term.

    The structural term is added (same sign as cost and waste) because higher
    infeasibility = worse fitness. GA pressure will evolve geometries that
    produce structurally sound MILP assignments.
    """
    omega_1, omega_2, omega_3, omega_4 = weights
    lca_fitness = omega_1 * cost_norm - omega_2 * reuse_norm + omega_3 * waste_norm
    struct_term = omega_4 * float(structural_infeasibility)
    return float(lca_fitness + struct_term)


def interpret_fitness_score(fitness: float) -> tuple[str, str]:
    """Map a fitness score to a label and short explanation."""
    if fitness <= -0.25:
        return "EXCELLENT", "Strong reclaimed reuse and low waste relative to cost"
    if fitness <= 0.25:
        return "GOOD", "Balanced trade-off with no major penalty dominance"
    if fitness <= 0.75:
        return "FAIR", "One objective is starting to dominate the trade-off"
    return "POOR", "Cost, waste, or low reuse is dominating the solution"


# ============================================================================
# Solution Evaluation
# ============================================================================


def evaluate_milp_solution(
    milp_results_df: pd.DataFrame,
    enriched_stock_df: pd.DataFrame,
    df_slots: pd.DataFrame,
    milp_objective_value: float,
    weights: tuple[float, float, float, float],
    normalization_constants: dict[str, float],
    structural_infeasibility: float = 0.0,
) -> dict[str, Any]:
    """Extract metrics, normalize them, and return fitness plus breakdown.

    structural_infeasibility: float in [0, 1], optional.
        Pass (1.0 - feasibility_score) from gnn_feasibility().
        Default 0.0 = structural penalty disabled (backward compatible).
    """
    reuse_rate = calculate_reuse_rate(milp_results_df, enriched_stock_df)
    waste_total = calculate_total_waste(milp_results_df, df_slots, enriched_stock_df)
    cost_raw = get_inner_cost(milp_objective_value)

    cost_norm, reuse_norm, waste_norm = normalize_metrics(
        cost_raw,
        reuse_rate,
        waste_total,
        normalization_constants,
    )

    fitness = fitness_function_multi_objective(
        cost_norm,
        reuse_norm,
        waste_norm,
        weights,
        structural_infeasibility=float(structural_infeasibility),
    )

    omega_4 = weights[3] if len(weights) > 3 else 0.0

    return {
        "fitness": float(fitness),
        "cost_raw": float(cost_raw),
        "reuse_rate": float(reuse_rate),
        "waste_total": float(waste_total),
        "cost_norm": float(cost_norm),
        "reuse_norm": float(reuse_norm),
        "waste_norm": float(waste_norm),
        "structural_infeasibility": float(structural_infeasibility),
        "structural_penalty": float(omega_4 * structural_infeasibility),
        "weights": tuple(float(v) for v in weights),
        "objective": float(cost_raw),
        "is_feasible": bool(np.isfinite(cost_raw) and not milp_results_df.empty),
    }


# ============================================================================
# Sanity Checks
# ============================================================================


def run_fitness_sanity_checks(
    normalization_constants: dict[str, float],
    weight_config: dict[str, float],
) -> dict[str, Any]:
    """Run quick checks for normalization and fitness ordering."""
    c_max = float(normalization_constants["C_max"])
    r_max = float(normalization_constants["R_max"])
    w_max = float(normalization_constants["W_max"])

    cost_1, reuse_1, waste_1 = normalize_metrics(
        cost=0.5 * c_max,
        reuse_rate=0.5 * r_max,
        waste=0.5 * w_max,
        normalization_constants=normalization_constants,
    )

    weights = weights_from_config(weight_config)
    fitness_excellent = fitness_function_multi_objective(
        cost_norm=0.25,
        reuse_norm=1.0,
        waste_norm=0.1,
        weights=weights,
        structural_infeasibility=0.0,
    )
    fitness_poor = fitness_function_multi_objective(
        cost_norm=0.8,
        reuse_norm=0.0,
        waste_norm=0.9,
        weights=weights,
        structural_infeasibility=1.0,
    )

    return {
        "normalization_mid_range": {
            "cost_norm": float(cost_1),
            "reuse_norm": float(reuse_1),
            "waste_norm": float(waste_1),
            "passes": bool(0.45 < cost_1 < 0.55 and 0.45 < reuse_1 < 0.55 and 0.45 < waste_1 < 0.55),
        },
        "fitness_ordering": {
            "excellent": float(fitness_excellent),
            "poor": float(fitness_poor),
            "passes": bool(fitness_excellent < fitness_poor),
        },
    }


# ============================================================================
# Printing & Debugging
# ============================================================================


def print_fitness_breakdown(result: dict[str, Any]) -> None:
    """Pretty-print a fitness result dictionary for debugging."""
    print("\n" + "=" * 70)
    print("MULTI-OBJECTIVE FITNESS EVALUATION")
    print("=" * 70)

    print("\nRaw Metrics:")
    print(f"  MILP Cost:        {result['cost_raw']:>8.3f} kg CO2e")
    print(f"  Reuse Rate:       {result['reuse_rate']:>8.1f} %")
    print(f"  Total Waste:      {result['waste_total']:>8.4f} m3")

    print("\nNormalized (0-1 range):")
    print(f"  Cost (norm):      {result['cost_norm']:>8.3f}")
    print(f"  Reuse (norm):     {result['reuse_norm']:>8.3f}")
    print(f"  Waste (norm):     {result['waste_norm']:>8.3f}")

    weights       = result["weights"]
    omega_1       = weights[0]
    omega_2       = weights[1]
    omega_3       = weights[2]
    omega_4       = weights[3] if len(weights) > 3 else 0.0
    struct_infeas  = float(result.get("structural_infeasibility", 0.0))
    struct_penalty = float(result.get("structural_penalty", 0.0))

    print("\nRaw Metrics:")
    print(f"  MILP Cost:              {result['cost_raw']:>8.3f} kg CO2e")
    print(f"  Reuse Rate:             {result['reuse_rate']:>8.1f} %")
    print(f"  Total Waste:            {result['waste_total']:>8.4f} m3")
    print(f"  Structural infeasible:  {struct_infeas:>8.3f}  (fraction of unsafe members)")

    print("\nNormalized (0-1 range):")
    print(f"  Cost (norm):      {result['cost_norm']:>8.3f}")
    print(f"  Reuse (norm):     {result['reuse_norm']:>8.3f}")
    print(f"  Waste (norm):     {result['waste_norm']:>8.3f}")
    print(f"  Structural:       {struct_infeas:>8.3f}  (already in [0,1])")

    print("\nWeights Applied:")
    print(f"  omega_1 (cost):        {omega_1:>8.3f}")
    print(f"  omega_2 (reuse):       {omega_2:>8.3f}")
    print(f"  omega_3 (waste):       {omega_3:>8.3f}")
    print(f"  omega_4 (structural):  {omega_4:>8.3f}")

    print("\nWeighted Components:")
    term1 = omega_1 * result["cost_norm"]
    term2 = omega_2 * result["reuse_norm"]
    term3 = omega_3 * result["waste_norm"]
    print(f"  omega_1 x cost:        {term1:>8.3f}")
    print(f"  omega_2 x reuse:       {term2:>8.3f} (subtracted)")
    print(f"  omega_3 x waste:       {term3:>8.3f}")
    print(f"  omega_4 x structural:  {struct_penalty:>8.3f} (added)")

    print("\nFinal Fitness:")
    if omega_4 > 0:
        print(f"  F(x) = {term1:.3f} - {term2:.3f} + {term3:.3f} + {struct_penalty:.3f}")
    else:
        print(f"  F(x) = {term1:.3f} - {term2:.3f} + {term3:.3f}")
    print(f"  F(x) = {result['fitness']:>8.3f}")
    print("\nInterpretation:")
    label, explanation = interpret_fitness_score(float(result["fitness"]))
    print(f"  [{label}] {explanation}")
    print("  Bands: <= -0.25 excellent, <= 0.25 good, <= 0.75 fair, > 0.75 poor")
    print("=" * 70 + "\n")


# ============================================================================
# Orchestration
# ============================================================================


def run_fitness_stage(
    df_results: pd.DataFrame,
    enriched_stock: pd.DataFrame,
    df_slots: pd.DataFrame,
    total_cost: float,
    weight_config: dict[str, float],
    normalization_margin: float = 0.20,
    normalization_constants: dict[str, float] | None = None,
    derive_normalization_constants: bool = True,
    run_sanity_checks: bool = True,
    print_breakdown: bool = False,
    structural_infeasibility: float = 0.0,
) -> dict[str, Any]:
    """Run fitness stage and return fitness values plus metadata.

    structural_infeasibility: float in [0, 1], optional.
        Pass (1.0 - feasibility_score) from gnn_feasibility() to include
        the GNN structural penalty in the fitness score.
        Requires omega_4 > 0 in weight_config to have effect.
        Default 0.0 = structural penalty disabled (backward compatible).
    """
    validated_weight_config = validate_weight_config(weight_config)
    weights = weights_from_config(validated_weight_config)

    if normalization_constants is not None:
        norm_constants = {
            "C_max": float(normalization_constants["C_max"]),
            "R_max": float(normalization_constants["R_max"]),
            "W_max": float(normalization_constants["W_max"]),
        }
    elif derive_normalization_constants:
        norm_constants = derive_normalization_constants_from_solution(
            milp_results_df=df_results,
            enriched_stock_df=enriched_stock,
            df_slots=df_slots,
            milp_objective_value=float(total_cost),
            margin=float(normalization_margin),
        )
    else:
        norm_constants = get_default_normalization_constants()

    sanity = None
    if run_sanity_checks:
        sanity = run_fitness_sanity_checks(
            normalization_constants=norm_constants,
            weight_config=validated_weight_config,
        )
        if not sanity["fitness_ordering"]["passes"] or not sanity["normalization_mid_range"]["passes"]:
            raise ValueError(
                "Fitness sanity check failed. Inspect normalization constants and weight configuration."
            )

    fitness_result = evaluate_milp_solution(
        milp_results_df=df_results,
        enriched_stock_df=enriched_stock,
        df_slots=df_slots,
        milp_objective_value=float(total_cost),
        weights=weights,
        normalization_constants=norm_constants,
        structural_infeasibility=float(structural_infeasibility),
    )

    if print_breakdown:
        print_fitness_breakdown(fitness_result)

    return {
        "fitness_result": fitness_result,
        "weight_config": validated_weight_config,
        "weights": weights,
        "normalization_constants": norm_constants,
        "sanity": sanity,
    }