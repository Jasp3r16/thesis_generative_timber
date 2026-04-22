from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from c28_fitness_aggregation import (
    derive_normalization_constants_from_solution,
    evaluate_milp_solution,
    get_normalization_constants,
    get_weight_config,
    print_fitness_breakdown,
    run_fitness_sanity_checks,
    weights_from_config,
)


def run_fitness_stage(
    df_results: pd.DataFrame,
    enriched_stock: pd.DataFrame,
    df_slots: pd.DataFrame,
    total_cost: float,
    weight_strategy: str = "cost-dominant",
    normalization_margin: float = 0.20,
    normalization_constants: dict[str, float] | None = None,
    run_sanity_checks: bool = True,
    print_breakdown: bool = False,
    export_json_path: Path | None = None,
    export_csv_path: Path | None = None,
) -> dict[str, Any]:
    """Run fitness stage and return fitness values plus metadata."""
    weight_config = get_weight_config(weight_strategy)
    weights = weights_from_config(weight_config)

    if normalization_constants is not None:
        norm_constants = {
            "C_max": float(normalization_constants["C_max"]),
            "R_max": float(normalization_constants["R_max"]),
            "W_max": float(normalization_constants["W_max"]),
        }
    else:
        norm_constants = derive_normalization_constants_from_solution(
            milp_results_df=df_results,
            enriched_stock_df=enriched_stock,
            df_slots=df_slots,
            milp_objective_value=float(total_cost),
            margin=float(normalization_margin),
        )

    sanity = None
    if run_sanity_checks:
        sanity = run_fitness_sanity_checks(normalization_constants=norm_constants)
        if not sanity["fitness_ordering"]["passes"]:
            raise ValueError(
                "Fitness sanity check failed: excellent design did not outperform poor design."
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

    if export_json_path is not None:
        export_json_path.parent.mkdir(parents=True, exist_ok=True)
        fitness_export = {
            key: (float(value) if isinstance(value, (np.floating, np.integer)) else value)
            for key, value in fitness_result.items()
        }
        with open(export_json_path, "w", encoding="utf-8") as f:
            json.dump(fitness_export, f, indent=2)

    if export_csv_path is not None:
        export_csv_path.parent.mkdir(parents=True, exist_ok=True)
        fitness_df = pd.DataFrame([{**fitness_result, **weight_config}])
        fitness_df.to_csv(export_csv_path, index=False)

    return {
        "fitness_result": fitness_result,
        "weight_config": weight_config,
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
