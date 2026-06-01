"""Multi-objective fitness: F = ω₁·Ĉ − ω₂·R̂ + ω₄·S (minimised).

Waste is excluded from the fitness function — it is already captured by the
LCA cost components (C3/C4 streams) in c25_stage_cost_matrix.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from workflows.c28_stage_normalization_bounds import get_default_normalization_constants

# Column Resolution Helpers

def _column_by_lower(df: pd.DataFrame, name: str) -> str | None:
    """Find a DataFrame column case-insensitively."""
    mapping = {str(col).strip().lower(): str(col) for col in df.columns}
    return mapping.get(name.strip().lower())

# Weight Configuration

def validate_weight_config(weight_config: dict[str, float]) -> dict[str, float]:
    """Validate a notebook-provided weight config.

    Required keys: omega_1 (LCA cost), omega_2 (reuse, subtracted).
    Optional:      omega_4 (structural infeasibility, default 0.0).

    omega_3 is accepted for backward compatibility but silently ignored —
    waste is no longer a fitness term (captured by the LCA cost in c25).
    """
    required = ("omega_1", "omega_2")
    missing = [key for key in required if key not in weight_config]
    if missing:
        raise ValueError(f"Missing keys in weight_config: {', '.join(missing)}")

    if "omega_3" in weight_config:
        warnings.warn(
            "omega_3 (waste weight) is deprecated and will be ignored. "
            "Waste is captured by the LCA cost in c25_stage_cost_matrix.",
            DeprecationWarning,
            stacklevel=2,
        )

    omega_1 = float(weight_config["omega_1"])
    omega_2 = float(weight_config["omega_2"])
    omega_4 = float(weight_config.get("omega_4", 0.0))

    if omega_1 < 0 or omega_2 < 0 or omega_4 < 0:
        raise ValueError("All weights must be >= 0")

    return {
        "omega_1": omega_1,
        "omega_2": omega_2,
        "omega_4": omega_4,
    }

def weights_from_config(weight_config: dict[str, float]) -> tuple[float, float, float]:
    """Convert a weight config dictionary to the (omega_1, omega_2, omega_4) tuple."""
    validated = validate_weight_config(weight_config)
    return (
        float(validated["omega_1"]),
        float(validated["omega_2"]),
        float(validated["omega_4"]),
    )

# Reclaimed State Resolution

def _resolve_stock_state_map(enriched_stock_df: pd.DataFrame) -> dict[str, int]:
    """Build a mapping of stock item IDs to their reclaimed state (0 or 1)."""
    member_col = _column_by_lower(enriched_stock_df, "member_id")
    if member_col is None:
        raise ValueError("enriched_stock_df is missing Member_ID")

    state_col = (
        _column_by_lower(enriched_stock_df, "state_resolved")
        or _column_by_lower(enriched_stock_df, "state")
    )
    member_ids = enriched_stock_df[member_col].astype(str).str.strip()

    if state_col is not None:
        state_values = pd.to_numeric(enriched_stock_df[state_col], errors="coerce").fillna(0.0)
        resolved = state_values.clip(lower=0.0, upper=1.0).astype(int)
        return {mid: int(state) for mid, state in zip(member_ids, resolved)}

    return {mid: int(mid.upper().startswith("RS")) for mid in member_ids}

# Metric Calculation

def calculate_reuse_volume_fraction(
    milp_results_df:   pd.DataFrame,
    enriched_stock_df: pd.DataFrame,
) -> float:
    """Volume-weighted reclaimed reuse fraction: Σ V_reclaimed / Σ V_assigned [0, 1].

    Each assigned stock element contributes its volume (Width × Depth × Length,
    units in mm — they cancel in the ratio). Falls back to count-based fraction
    when dimension columns are absent from enriched_stock_df.
    """
    if milp_results_df.empty:
        return 0.0

    if "assigned_timber" not in milp_results_df.columns:
        raise ValueError("milp_results_df is missing assigned_timber")

    state_map  = _resolve_stock_state_map(enriched_stock_df)
    member_col = _column_by_lower(enriched_stock_df, "member_id")
    width_col  = _column_by_lower(enriched_stock_df, "width")
    depth_col  = _column_by_lower(enriched_stock_df, "depth")
    length_col = _column_by_lower(enriched_stock_df, "length")

    has_dims = all(c is not None for c in [member_col, width_col, depth_col, length_col])

    if not has_dims:
        warnings.warn(
            "enriched_stock_df missing Width/Depth/Length — falling back to count-based "
            "reuse fraction.",
            stacklevel=2,
        )
        assigned = milp_results_df["assigned_timber"].astype(str).str.strip()
        rec = sum(
            state_map.get(tid, int(tid.upper().startswith("RS")))
            for tid in assigned
        )
        return float(rec / len(assigned)) if len(assigned) > 0 else 0.0

    # Build volume lookup (mm³ — units cancel in ratio)
    vol_lookup: dict[str, float] = {}
    for _, row in enriched_stock_df.iterrows():
        mid = str(row[member_col]).strip()
        w   = float(row[width_col])
        d   = float(row[depth_col])
        l   = float(row[length_col])
        vol_lookup[mid] = w * d * l

    total_vol    = 0.0
    reclaimed_vol = 0.0
    for _, assignment in milp_results_df.iterrows():
        tid = str(assignment["assigned_timber"]).strip()
        vol = vol_lookup.get(tid, 0.0)
        total_vol    += vol
        if state_map.get(tid, int(tid.upper().startswith("RS"))) == 1:
            reclaimed_vol += vol

    return float(reclaimed_vol / total_vol) if total_vol > 0.0 else 0.0

def calculate_total_waste(
    milp_results_df:    pd.DataFrame,
    df_slots:           pd.DataFrame,
    stock_inventory_df: pd.DataFrame,
) -> float:
    """Calculate total waste volume (m³) from MILP assignments.

    Informational only — waste is NOT included in the fitness function.
    It is already captured in the LCA cost (c25_stage_cost_matrix).
    """
    if milp_results_df.empty:
        return 0.0

    required_slot_cols = {
        "edge_id":    _column_by_lower(df_slots, "edge_id"),
        "length_req": _column_by_lower(df_slots, "length_req"),
        "width_req":  _column_by_lower(df_slots, "width_req"),
        "depth_req":  _column_by_lower(df_slots, "depth_req"),
    }
    if any(value is None for value in required_slot_cols.values()):
        raise ValueError(
            "df_slots is missing one or more required columns: "
            "edge_id, Length_Req, Width_Req, Depth_Req"
        )

    required_stock_cols = {
        "member_id": _column_by_lower(stock_inventory_df, "member_id"),
        "length":    _column_by_lower(stock_inventory_df, "length"),
        "width":     _column_by_lower(stock_inventory_df, "width"),
        "depth":     _column_by_lower(stock_inventory_df, "depth"),
    }
    if any(value is None for value in required_stock_cols.values()):
        raise ValueError(
            "stock_inventory_df is missing one or more required columns: "
            "Member_ID, Length, Width, Depth"
        )

    slot_indexed  = df_slots.set_index(required_slot_cols["edge_id"], drop=False)
    stock_indexed = stock_inventory_df.set_index(required_stock_cols["member_id"], drop=False)

    total_waste_m3 = 0.0
    for _, assignment in milp_results_df.iterrows():
        edge_id   = str(assignment["edge_id"]).strip()
        timber_id = str(assignment["assigned_timber"]).strip()

        if edge_id not in slot_indexed.index or timber_id not in stock_indexed.index:
            continue

        slot_row  = slot_indexed.loc[edge_id]
        stock_row = stock_indexed.loc[timber_id]

        l_req   = float(slot_row[required_slot_cols["length_req"]]) / 1000.0
        w_req   = float(slot_row[required_slot_cols["width_req"]]) / 1000.0
        d_req   = float(slot_row[required_slot_cols["depth_req"]]) / 1000.0
        l_stock = float(stock_row[required_stock_cols["length"]]) / 1000.0
        w_stock = float(stock_row[required_stock_cols["width"]]) / 1000.0
        d_stock = float(stock_row[required_stock_cols["depth"]]) / 1000.0

        oversizing_waste = max(0.0, (w_stock * d_stock - w_req * d_req) * l_req)
        length_waste     = max(0.0, w_stock * d_stock * (l_stock - l_req))
        total_waste_m3  += oversizing_waste + length_waste

    return float(total_waste_m3)

def get_inner_cost(milp_result: float) -> float:
    """Return MILP objective value as float."""
    return float(milp_result)

# Normalization

def normalize_metrics(
    cost:                    float,
    reuse_fraction:          float,
    normalization_constants: dict[str, float],
) -> tuple[float, float]:
    """Normalize cost and reuse fraction to [0, 1].

    Parameters
    ----------
    cost             : MILP objective value (kg CO2e)
    reuse_fraction   : volume-weighted reclaimed fraction [0, 1]
    normalization_constants : dict with keys C_max, R_max
    """
    c_max = max(float(normalization_constants["C_max"]), 1e-9)
    r_max = max(float(normalization_constants["R_max"]), 1e-9)

    cost_norm  = float(np.clip(float(cost)            / c_max, 0.0, 1.0))
    reuse_norm = float(np.clip(float(reuse_fraction)  / r_max, 0.0, 1.0))
    return cost_norm, reuse_norm

def derive_normalization_constants_from_solution(
    milp_results_df:     pd.DataFrame,
    enriched_stock_df:   pd.DataFrame,
    milp_objective_value: float,
    margin:              float = 0.20,
) -> dict[str, float]:
    """Derive normalization constants from the current solution with a margin.

    WARNING: This function should NOT be used inside the GA loop — each candidate
    normalizes itself, making fitness values incomparable across individuals.
    Use compute_normalization_bounds() once before the loop instead.
    """
    if margin < 0:
        raise ValueError("margin must be >= 0")

    cost_raw       = get_inner_cost(milp_objective_value)
    reuse_fraction = calculate_reuse_volume_fraction(milp_results_df, enriched_stock_df)

    scale = 1.0 + float(margin)
    c_max = max(float(cost_raw)       * scale, 1e-9)
    r_max = max(float(reuse_fraction) * scale, 1.0)   # overestimate cap at 1.0

    return {
        "C_max": float(c_max),
        "R_max": float(min(r_max, 1.0)),
    }

# Multi-Objective Fitness

def fitness_function_multi_objective(
    cost_norm:               float,
    reuse_norm:              float,
    weights:                 tuple[float, float, float],
    structural_infeasibility: float = 0.0,
) -> float:
    """Compute weighted fitness (lower = better).

    F(x) = omega_1 * cost_norm
           - omega_2 * reuse_norm
           + omega_4 * structural_infeasibility

    structural_infeasibility: float in [0, 1]
        Mean P(unsafe) across all members from the GNN surrogate.
        = 1.0 - feasibility_score from gnn_feasibility()
        = 0.0 when all members predicted fully safe  (no penalty)
        = 1.0 when all members predicted fully unsafe (maximum penalty)

    omega_4 = 0.0 (default): structural penalty disabled.
    omega_4 > 0: curriculum-scheduled via c23_ga_evaluator (_resolve_w_structural).
    """
    omega_1, omega_2, omega_4 = weights
    return float(
        omega_1 * cost_norm
        - omega_2 * reuse_norm
        + omega_4 * float(structural_infeasibility)
    )

def interpret_fitness_score(
    fitness: float,
    weights: tuple[float, float, float] | None = None,
) -> tuple[str, str]:
    """Map a fitness score to a label and short explanation.

    Bands are derived from the theoretical range of the fitness function
    given the current weights, divided into equal quartiles:
        EXCELLENT : bottom 25% of range (strong reuse, low cost, safe)
        GOOD      : 25–50%
        FAIR      : 50–75%
        POOR      : top 25% of range
    """
    if weights is None:
        weights = (1.0, 1.0, 0.0)

    omega_1, omega_2, omega_4 = weights
    f_min   = -omega_2                    # best case: cost=0, reuse=1, struct=0
    f_range = omega_1 + omega_2 + omega_4 # worst case: cost=1, reuse=0, struct=1

    if f_range < 1e-9:
        return "UNKNOWN", "All weights are zero — fitness is undefined"

    q25 = f_min + 0.25 * f_range
    q50 = f_min + 0.50 * f_range
    q75 = f_min + 0.75 * f_range

    if fitness <= q25:
        return "EXCELLENT", "Strong reclaimed reuse relative to cost and structural quality"
    if fitness <= q50:
        return "GOOD", "Balanced trade-off with no major objective dominance"
    if fitness <= q75:
        return "FAIR", "Cost, low reuse, or structural infeasibility starting to dominate"
    return "POOR", "Solution dominated by cost, low reuse, or structural infeasibility"

# Solution Evaluation

# Fraction of members predicted unsafe above which a solution is flagged
# structurally infeasible in `is_feasible`. Applies only when omega_4 > 0.
STRUCTURAL_FEASIBILITY_THRESHOLD = 0.5

def evaluate_milp_solution(
    milp_results_df:         pd.DataFrame,
    enriched_stock_df:       pd.DataFrame,
    milp_objective_value:    float,
    weights:                 tuple[float, float, float],
    normalization_constants: dict[str, float],
    structural_infeasibility: float = 0.0,
) -> dict[str, Any]:
    """Extract metrics, normalize them, and return fitness plus breakdown.

    structural_infeasibility: float in [0, 1], optional.
        Pass (1.0 - feasibility_score) from gnn_feasibility().
        Default 0.0 = structural penalty disabled (backward compatible).

    is_feasible is True when:
        - MILP produced a valid assignment (finite cost, non-empty results)
        - AND either omega_4 == 0.0 (structural penalty disabled)
          OR structural_infeasibility < STRUCTURAL_FEASIBILITY_THRESHOLD (0.5)
          — i.e. fewer than 50 % of members predicted unsafe.
    """
    reuse_fraction = calculate_reuse_volume_fraction(milp_results_df, enriched_stock_df)
    cost_raw       = get_inner_cost(milp_objective_value)

    cost_norm, reuse_norm = normalize_metrics(
        cost_raw,
        reuse_fraction,
        normalization_constants,
    )

    fitness = fitness_function_multi_objective(
        cost_norm,
        reuse_norm,
        weights,
        structural_infeasibility=float(structural_infeasibility),
    )

    omega_1, omega_2, omega_4 = weights
    milp_feasible = bool(np.isfinite(cost_raw) and not milp_results_df.empty)
    struct_ok     = (omega_4 == 0.0 or float(structural_infeasibility) < STRUCTURAL_FEASIBILITY_THRESHOLD)
    is_feasible   = milp_feasible and struct_ok

    return {
        "fitness":                  float(fitness),
        "cost_raw":                 float(cost_raw),
        "reuse_fraction":           float(reuse_fraction),
        "cost_norm":                float(cost_norm),
        "reuse_norm":               float(reuse_norm),
        "structural_infeasibility": float(structural_infeasibility),
        "structural_penalty":       float(omega_4 * structural_infeasibility),
        "weights":                  tuple(float(v) for v in weights),
        "is_feasible":              is_feasible,
    }

# Sanity Checks

def run_fitness_sanity_checks(
    normalization_constants: dict[str, float],
    weight_config:           dict[str, float],
) -> dict[str, Any]:
    """Run quick checks for normalization and fitness ordering."""
    c_max = float(normalization_constants["C_max"])
    r_max = float(normalization_constants["R_max"])

    cost_1, reuse_1 = normalize_metrics(
        cost=0.5 * c_max,
        reuse_fraction=0.5 * r_max,
        normalization_constants=normalization_constants,
    )

    weights = weights_from_config(weight_config)
    fitness_excellent = fitness_function_multi_objective(
        cost_norm=0.2,
        reuse_norm=1.0,
        weights=weights,
        structural_infeasibility=0.0,
    )
    fitness_poor = fitness_function_multi_objective(
        cost_norm=0.9,
        reuse_norm=0.0,
        weights=weights,
        structural_infeasibility=1.0,
    )

    return {
        "normalization_mid_range": {
            "cost_norm":  float(cost_1),
            "reuse_norm": float(reuse_1),
            "passes":     bool(0.45 < cost_1 < 0.55 and 0.45 < reuse_1 < 0.55),
        },
        "fitness_ordering": {
            "excellent": float(fitness_excellent),
            "poor":      float(fitness_poor),
            "passes":    bool(fitness_excellent < fitness_poor),
        },
    }

# Printing & Debugging

def print_fitness_breakdown(result: dict[str, Any]) -> None:
    """Pretty-print a fitness result dictionary for debugging."""
    print("\n" + "=" * 70)
    print("MULTI-OBJECTIVE FITNESS EVALUATION")
    print("=" * 70)

    weights       = result["weights"]
    omega_1       = weights[0]
    omega_2       = weights[1]
    omega_4       = weights[2] if len(weights) > 2 else 0.0
    struct_infeas  = float(result.get("structural_infeasibility", 0.0))
    struct_penalty = float(result.get("structural_penalty", 0.0))

    print("\nRaw Metrics:")
    print(f"  MILP Cost:              {result['cost_raw']:>8.3f} kg CO2e")
    print(f"  Reuse Fraction:         {result['reuse_fraction']:>8.3f}  (volume-weighted, 0–1)")
    print(f"  Structural infeasible:  {struct_infeas:>8.3f}  (fraction of unsafe members)")

    print("\nNormalized (0–1 range):")
    print(f"  Cost (norm):      {result['cost_norm']:>8.3f}")
    print(f"  Reuse (norm):     {result['reuse_norm']:>8.3f}")
    print(f"  Structural:       {struct_infeas:>8.3f}  (already in [0,1])")

    print("\nWeights Applied:")
    print(f"  omega_1 (cost):        {omega_1:>8.3f}")
    print(f"  omega_2 (reuse):       {omega_2:>8.3f}  (subtracted)")
    print(f"  omega_4 (structural):  {omega_4:>8.3f}")

    term1 = omega_1 * result["cost_norm"]
    term2 = omega_2 * result["reuse_norm"]
    print("\nWeighted Components:")
    print(f"  omega_1 x cost:        {term1:>8.3f}")
    print(f"  omega_2 x reuse:      -{term2:>8.3f}")
    if omega_4 > 0:
        print(f"  omega_4 x structural:  {struct_penalty:>8.3f}")

    print("\nFinal Fitness (lower = better):")
    if omega_4 > 0:
        print(f"  F(x) = {term1:.3f} - {term2:.3f} + {struct_penalty:.3f}")
    else:
        print(f"  F(x) = {term1:.3f} - {term2:.3f}")
    print(f"  F(x) = {result['fitness']:>8.3f}")

    print(f"\n  is_feasible: {result.get('is_feasible', '—')}")
    print("\nInterpretation:")
    label, explanation = interpret_fitness_score(float(result["fitness"]), tuple(weights))
    f_min = -omega_2
    f_range = omega_1 + omega_2 + omega_4
    q25 = f_min + 0.25 * f_range
    q50 = f_min + 0.50 * f_range
    q75 = f_min + 0.75 * f_range
    print(f"  [{label}] {explanation}")
    print(f"  Bands: ≤ {q25:.3f} excellent, ≤ {q50:.3f} good, ≤ {q75:.3f} fair, > {q75:.3f} poor")
    print("=" * 70 + "\n")

# Orchestration

def run_fitness_stage(
    df_results:                     pd.DataFrame,
    enriched_stock:                 pd.DataFrame,
    df_slots:                       pd.DataFrame,
    total_cost:                     float,
    weight_config:                  dict[str, float],
    normalization_margin:           float = 0.20,
    normalization_constants:        dict[str, float] | None = None,
    derive_normalization_constants: bool = False,
    run_sanity_checks:              bool = True,
    print_breakdown:                bool = False,
    structural_infeasibility:       float = 0.0,
) -> dict[str, Any]:
    """Run fitness stage and return fitness values plus metadata.

    normalization_constants should be pre-computed via
    compute_normalization_bounds() once before the GA loop and passed
    explicitly. This is the only mode that produces comparable fitness scores
    across individuals — required for correct GA selection.

    derive_normalization_constants=True: constants are derived from the current
    solution's cost/reuse scaled by (1 + margin). Valid for single-solution
    inspection and notebooks, but produces incomparable scores inside the GA
    loop because every candidate normalizes itself. Default False.

    structural_infeasibility: float in [0, 1], optional.
        Pass (1.0 - feasibility_score) from gnn_feasibility().
        Requires omega_4 > 0 in weight_config to have effect.
        Default 0.0 = structural penalty disabled (backward compatible).

    df_slots is accepted for API compatibility but not used in fitness
    computation (waste is not a fitness term).
    """
    validated_weight_config = validate_weight_config(weight_config)
    weights = weights_from_config(validated_weight_config)

    if normalization_constants is not None:
        norm_constants = {
            "C_max": float(normalization_constants["C_max"]),
            "R_max": float(normalization_constants["R_max"]),
        }
    elif derive_normalization_constants:
        warnings.warn(
            "derive_normalization_constants=True: each candidate normalizes itself. "
            "Fitness values are NOT comparable across individuals. "
            "Use compute_normalization_bounds() once before the GA loop and pass "
            "normalization_constants explicitly.",
            UserWarning,
            stacklevel=2,
        )
        norm_constants = derive_normalization_constants_from_solution(
            milp_results_df=df_results,
            enriched_stock_df=enriched_stock,
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
            warnings.warn(
                "Fitness sanity check failed — normalization constants or weights may produce "
                "a flat or inverted fitness landscape. Inspect normalization_constants and "
                "weight_config before running the GA. "
                f"Ordering passes: {sanity['fitness_ordering']['passes']}, "
                f"Mid-range passes: {sanity['normalization_mid_range']['passes']}",
                UserWarning,
                stacklevel=2,
            )

    fitness_result = evaluate_milp_solution(
        milp_results_df=df_results,
        enriched_stock_df=enriched_stock,
        milp_objective_value=float(total_cost),
        weights=weights,
        normalization_constants=norm_constants,
        structural_infeasibility=float(structural_infeasibility),
    )

    if print_breakdown:
        print_fitness_breakdown(fitness_result)

    return {
        "fitness_result":         fitness_result,
        "weight_config":          validated_weight_config,
        "weights":                weights,
        "normalization_constants": norm_constants,
        "sanity":                 sanity,
    }
