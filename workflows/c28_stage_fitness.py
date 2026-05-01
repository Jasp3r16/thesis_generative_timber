from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))


from c28_normalization_bounds import compute_normalization_bounds
from c28_fitness_aggregation import (
    derive_normalization_constants_from_solution,
    evaluate_milp_solution,
    get_normalization_constants,
    print_fitness_breakdown,
    run_fitness_sanity_checks,
    validate_weight_config,
    weights_from_config,
)


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
) -> dict[str, Any]:
    """Run fitness stage and return fitness values plus metadata."""
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
        norm_constants = get_normalization_constants()

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


def get_default_normalization_constants() -> dict[str, float]:
    """Expose default fixed normalization constants for GA workflows."""
    defaults = get_normalization_constants()
    return {
        "C_max": float(defaults["C_max"]),
        "R_max": float(defaults["R_max"]),
        "W_max": float(defaults["W_max"]),
    }

def run_normalization_bounds_stage(
    *,
    cost_matrix: np.ndarray,
    df_logs: pd.DataFrame,
    enriched_stock: pd.DataFrame,
    df_slots: pd.DataFrame,
    reclaimed_marker: str = "RS",
    new_marker: str = "NS",
    new_stock_max_uses: int | None = 1,
    solver_msg: bool = False,
    print_summary: bool = True,
) -> dict[str, Any]:
    """Notebook-facing wrapper for computing exact normalization bounds."""
    out = compute_normalization_bounds(
        cost_matrix=cost_matrix,
        df_logs=df_logs,
        enriched_stock=enriched_stock,
        df_slots=df_slots,
        reclaimed_marker=reclaimed_marker,
        new_marker=new_marker,
        new_stock_max_uses=new_stock_max_uses,
        solver_msg=solver_msg,
    )

    if print_summary:
        status = out.get("status", "unknown")
        constants = out.get("normalization_constants", {})
        print(f"Bounds status: {status}")
        print(
            "Normalization constants "
            f"C_max={constants.get('C_max')}, "
            f"R_max={constants.get('R_max')}, "
            f"W_max={constants.get('W_max')}"
        )

    return out
