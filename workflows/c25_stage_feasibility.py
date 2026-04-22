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

import c21_surrogate_io as surrogate_io
import c25_feasibility_check as structural_check

assign_roof_load_fz = structural_check.assign_roof_load_fz
geometry_df_to_design_row = structural_check.geometry_df_to_design_row
compute_utilization_outputs = structural_check.compute_utilization_outputs


def _validate_structural_stage_notebook_inputs(
    df_input_stock: pd.DataFrame | None,
    df_vertices: pd.DataFrame | None,
) -> None:
    helper = getattr(structural_check, "validate_structural_stage_notebook_inputs", None)
    if callable(helper):
        helper(df_input_stock=df_input_stock, df_vertices=df_vertices)
        return

    missing: list[str] = []
    if df_input_stock is None:
        missing.append("df_input_stock")
    if df_vertices is None:
        missing.append("df_vertices")
    if missing:
        raise ValueError("Missing required structural inputs: " + ", ".join(missing))

def _package_structural_outputs_for_notebook(
    structural_out: dict[str, Any],
    bundle_error: str | None = None,
) -> dict[str, Any]:
    helper = getattr(structural_check, "package_structural_outputs_for_notebook", None)
    if callable(helper):
        return helper(structural_out=structural_out, bundle_error=bundle_error)

    summary = structural_out["summary"]
    return {
        "SURROGATE_BUNDLE": structural_out["bundle"],
        "SURROGATE_BUNDLE_ERROR": bundle_error,
        "structural_out": structural_out,
        "df_forces": structural_out["df_forces"],
        "df_inventory": structural_out["df_inventory"],
        "df_forces_local": structural_out["df_forces_local"],
        "df_utilization_long": structural_out["df_utilization_long"],
        "df_utilization_matrix": structural_out["df_utilization_matrix"],
        "df_utilization_matrix_display": structural_out["df_utilization_matrix_display"],
        "safe_options": structural_out["safe_options"],
        "df_slots": structural_out["df_slots"],
        "summary": summary,
        "forces_source": structural_out["forces_source"],
    }

def _build_demand_slots(
    df_vertices: pd.DataFrame | None,
    df_edges: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build edge demand slots without surrogate force prediction.

    This path keeps c25 demand-only when section area is not assigned yet.
    """
    if df_edges is None or len(df_edges) == 0:
        raise ValueError("demand-only mode requires a non-empty df_edges input.")

    slots = df_edges.copy().reset_index(drop=True)
    if "edge_id" not in slots.columns:
        slots["edge_id"] = [f"e{i}" for i in range(len(slots))]
    else:
        slots["edge_id"] = slots["edge_id"].astype(str)

    if "length_m" not in slots.columns:
        if all(col in slots.columns for col in ("V1", "V2")) and df_vertices is not None and all(
            col in df_vertices.columns for col in ("x", "y", "z")
        ):
            coords = df_vertices[["x", "y", "z"]].astype(float).reset_index(drop=True)

            def _edge_length(row: pd.Series) -> float:
                try:
                    i = int(row["V1"])
                    j = int(row["V2"])
                    p0 = coords.iloc[i].to_numpy(dtype=float)
                    p1 = coords.iloc[j].to_numpy(dtype=float)
                    return float(np.linalg.norm(p0 - p1))
                except Exception:
                    return float("nan")

            slots["length_m"] = slots.apply(_edge_length, axis=1)
        else:
            raise ValueError("demand-only mode needs length_m in df_edges or V1/V2 plus vertex xyz coordinates.")

    slots["length_m"] = pd.to_numeric(slots["length_m"], errors="coerce")
    slots["Length_Req"] = (slots["length_m"] * 1000.0).round(0)

    if "axial_force_kn" not in slots.columns:
        slots["axial_force_kn"] = np.nan
    else:
        slots["axial_force_kn"] = pd.to_numeric(slots["axial_force_kn"], errors="coerce")

    slots["Depth_Req"] = np.nan
    slots["Width_Req"] = np.nan
    slots["Utilization_Req"] = np.nan

    return slots[["edge_id", "length_m", "axial_force_kn", "Length_Req", "Depth_Req", "Width_Req", "Utilization_Req"]]


def run_structural_stage(
    df_input_stock: pd.DataFrame,
    df_vertices: pd.DataFrame | None = None,
    df_edges: pd.DataFrame | None = None,
    df_forces: pd.DataFrame | None = None,
    bundle: dict[str, Any] | None = None,
    model_prefix: str | None = None,
    gnn_margin: float = 1.10,
    export_slots_path: Path | None = None,
    force_mode: str = "surrogate",
) -> dict[str, Any]:
    """Run structural utilization stage and return reusable tables.

    This is a notebook-independent wrapper around compute_utilization_outputs.
    """
    if str(force_mode).lower() == "demand-only":
        df_slots = _build_demand_slots(df_vertices=df_vertices, df_edges=df_edges)
        if export_slots_path is not None:
            export_slots_path.parent.mkdir(parents=True, exist_ok=True)
            df_slots.to_csv(export_slots_path, index=False)

        return {
            "df_forces": pd.DataFrame(columns=["edge_id", "length_m", "axial_force_kn"]),
            "df_inventory": df_input_stock.copy(),
            "df_forces_local": pd.DataFrame(columns=["edge_id", "length_m", "axial_force_kn"]),
            "df_utilization_long": pd.DataFrame(),
            "df_utilization_matrix": None,
            "df_utilization_matrix_display": None,
            "safe_options": pd.DataFrame(),
            "df_slots": df_slots,
            "bundle": bundle,
            "forces_source": "demand-only",
            "summary": {
                "members": int(len(df_slots)),
                "stock_items": int(len(df_input_stock)),
                "safe_combinations": 0,
            },
        }

    if df_forces is None:
        if df_vertices is None:
            raise ValueError("Provide either df_forces or df_vertices for structural stage")
        df_forces, active_bundle, forces_source = _predict_forces_with_surrogate(
            df_vertices=df_vertices,
            df_edges=df_edges,
            bundle=bundle,
            model_prefix=model_prefix,
        )
    else:
        active_bundle = bundle
        forces_source = "provided"

    outputs = compute_utilization_outputs(
        df_forces=df_forces,
        df_input_stock=df_input_stock,
        gnn_margin=float(gnn_margin),
    )

    df_inventory = outputs["df_inventory"]
    df_forces_local = outputs["df_forces_local"]
    df_utilization_long = outputs["df_utilization_long"]
    df_utilization_matrix = outputs["df_utilization_matrix"]
    df_utilization_matrix_display = outputs["df_utilization_matrix_display"]
    safe_options = outputs.get("safe_options", outputs.get("veilige_opties"))
    if safe_options is None:
        raise KeyError("compute_utilization_outputs did not return safe options.")
    df_slots = outputs["df_slots"].copy()

    if export_slots_path is not None:
        export_slots_path.parent.mkdir(parents=True, exist_ok=True)
        df_slots.to_csv(export_slots_path, index=False)

    return {
        "df_forces": df_forces,
        "df_inventory": df_inventory,
        "df_forces_local": df_forces_local,
        "df_utilization_long": df_utilization_long,
        "df_utilization_matrix": df_utilization_matrix,
        "df_utilization_matrix_display": df_utilization_matrix_display,
        "safe_options": safe_options,
        "df_slots": df_slots,
        "bundle": active_bundle,
        "forces_source": forces_source,
        "summary": {
            "members": int(len(df_forces_local)),
            "stock_items": int(len(df_inventory)),
            "safe_combinations": int(len(safe_options)),
        },
    }


def run_structural_stage_notebook(
    df_input_stock: pd.DataFrame | None,
    df_vertices: pd.DataFrame | None,
    df_edges: pd.DataFrame | None = None,
    model_prefix: str | None = None,
    bundle: dict[str, Any] | None = None,
    gnn_margin: float = 1.10,
    export_slots_path: Path | None = None,
    verbose: bool = True,
    force_mode: str = "surrogate",
) -> dict[str, Any]:
    """Notebook-friendly single entry point for the structural stage.

    This wraps bundle loading, structural execution, output unpacking and status prints
    so notebook cells can call a single function.
    """
    _validate_structural_stage_notebook_inputs(
        df_input_stock=df_input_stock,
        df_vertices=df_vertices,
    )

    active_bundle = bundle
    if str(force_mode).lower() == "demand-only":
        bundle_error = None
    else:
        if active_bundle is None:
            active_bundle, bundle_error = prepare_surrogate_bundle(model_prefix=model_prefix)
            if active_bundle is None:
                raise RuntimeError(f"Surrogate bundle unavailable. Load reason: {bundle_error}")
        else:
            bundle_error = None

    structural_out = run_structural_stage(
        df_input_stock=df_input_stock,
        df_vertices=df_vertices,
        df_edges=df_edges,
        bundle=active_bundle,
        model_prefix=model_prefix,
        gnn_margin=gnn_margin,
        export_slots_path=export_slots_path,
        force_mode=force_mode,
    )

    notebook_outputs = _package_structural_outputs_for_notebook(
        structural_out=structural_out,
        bundle_error=bundle_error,
    )

    if verbose:
        if notebook_outputs["SURROGATE_BUNDLE"] is not None:
            print("Surrogate bundle loaded and cached for iterative structural calls.")
        elif str(force_mode).lower() == "demand-only":
            print("Demand-only structural mode active (no surrogate force prediction in c25).")

        summary = notebook_outputs["summary"]
        print(f"Force source: {notebook_outputs['forces_source']}")
        print(
            f"Utilization: {summary['members']} members, "
            f"{summary['stock_items']} stock -> {summary['safe_combinations']} safe combinations"
        )

    return notebook_outputs
