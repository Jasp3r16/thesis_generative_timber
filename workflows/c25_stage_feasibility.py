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

import c25_feasibility_check as feasibility_check

assign_roof_load_fz = feasibility_check.assign_roof_load_fz
geometry_df_to_design_row = feasibility_check.geometry_df_to_design_row


def _validate_feasibility_stage_notebook_inputs(
    df_input_stock: pd.DataFrame | None,
    df_vertices: pd.DataFrame | None,
) -> None:
    helper = getattr(feasibility_check, "validate_feasibility_stage_notebook_inputs", None)
    if callable(helper):
        helper(df_input_stock=df_input_stock, df_vertices=df_vertices)
        return

    missing: list[str] = []
    if df_input_stock is None:
        missing.append("df_input_stock")
    if df_vertices is None:
        missing.append("df_vertices")
    if missing:
        raise ValueError("Missing required feasibility inputs: " + ", ".join(missing))

def run_feasibility_stage(
    df_input_stock: pd.DataFrame,
    df_vertices: pd.DataFrame | None = None,
    df_edges: pd.DataFrame | None = None,
    bundle: dict[str, Any] | None = None,
    model_prefix_complex: str | None = None,
    model_prefix_simple: str | None = None,
    gnn_margin: float = 1.10,
    utilization_threshold: float = 1.0,
    export_slots_path: Path | None = None,
    force_mode: str = "surrogate",
    surrogate_edge_feature_mode: str = "length_only",
) -> dict[str, Any]:
    """Run feasibility stage and return reusable tables.

    This is a notebook-independent wrapper around compute_utilization_outputs.
    """
    if str(force_mode).lower() != "surrogate":
        raise ValueError("c25 run_feasibility_stage currently supports force_mode='surrogate' only.")

    _validate_feasibility_stage_notebook_inputs(df_input_stock=df_input_stock, df_vertices=df_vertices)

    if df_vertices is None:
        raise ValueError("df_vertices is required.")
    if df_edges is None:
        raise ValueError("df_edges is required for surrogate inference.")

    feature_mode = str(surrogate_edge_feature_mode).strip().lower()
    if feature_mode not in {"length_only", "area_length"}:
        raise ValueError("surrogate_edge_feature_mode must be 'length_only' or 'area_length'.")

    # Determine active model prefix for surrogate inference.
    # Keep a deterministic fallback so legacy notebook calls continue to run.
    if feature_mode == "area_length":
        active_prefix = model_prefix_complex
    else:
        active_prefix = model_prefix_simple

    print(f"Using surrogate model prefix: {active_prefix} with edge feature mode: {feature_mode}")
    
    if "Fz" not in df_vertices.columns:
        df_vertices = assign_roof_load_fz(df_vertices)

    df_forces = outputs["df_forces"]
    active_bundle = outputs["bundle"]
    prediction_mode = f"surrogate:{feature_mode}"

    if export_slots_path is not None:
        export_slots_path.parent.mkdir(parents=True, exist_ok=True)
        outputs["df_slots"].to_csv(export_slots_path, index=False)

    summary = {
        "mode": prediction_mode,
        "model_prefix": active_prefix,
        "slots": int(len(outputs["df_slots"])),
        "stock_items": int(len(df_input_stock)),
        "feasible_pairs": int(outputs["df_safe_options"].shape[0]),
        "total_pairs": int(len(outputs["df_slots"]) * len(df_input_stock)),
        "utilization_threshold": float(utilization_threshold),
    }

    return {
        "bundle": active_bundle,
        "df_vertices": df_vertices,
        "df_slots": outputs["df_slots"],
        "df_feasibility_matrix": outputs["df_feasibility_matrix_display"],
        "df_feasibility_matrix_values": outputs["df_feasibility_matrix"],
        "df_safe_options": outputs["df_safe_options"],
        "df_failure_reasons": outputs["df_failure_reasons"],
        "summary": summary,
    }