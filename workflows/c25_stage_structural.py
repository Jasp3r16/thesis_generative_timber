from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

import c21_surrogate_io as surrogate_io
import c25_structural_check as structural_check

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


def _predict_forces_with_surrogate(
    df_vertices: pd.DataFrame,
    df_edges: pd.DataFrame | None,
    bundle: dict[str, Any] | None,
    model_prefix: str | None,
) -> tuple[pd.DataFrame, dict[str, Any] | None, str]:
    """Predict forces via the surrogate model."""
    df_geometry = df_vertices.copy().reset_index(drop=True)

    # Apply distributed roof load as nodal Fz before surrogate inference.
    # By convention, top-layer nodes receive tributary load and bottom-layer nodes remain zero.
    df_geometry = assign_roof_load_fz(df_geometry, roof_load_kn_m2=2.0)

    active_bundle = bundle if bundle is not None else surrogate_io.load_surrogate_bundle(prefix_sm=model_prefix)

    design_row = geometry_df_to_design_row(
        df_geometry=df_geometry,
        df_edges=df_edges,
    )
    df_forces = surrogate_io.predict_edge_forces_kn(design_row, active_bundle).copy()
    df_forces["V1"] = df_forces["V1"].astype(str)
    df_forces["V2"] = df_forces["V2"].astype(str)
    df_forces["length_m"] = df_forces["length_m"].round(3)
    df_forces["axial_force_kn"] = df_forces["axial_force_kn"].round(2)
    return df_forces, active_bundle, "surrogate"


def prepare_surrogate_bundle(model_prefix: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """Try loading surrogate bundle once for re-use in iterative runs."""
    try:
        return surrogate_io.load_surrogate_bundle(prefix_sm=model_prefix), None
    except Exception as exc:
        return None, str(exc)


def run_structural_stage(
    df_input_stock: pd.DataFrame,
    df_vertices: pd.DataFrame | None = None,
    df_edges: pd.DataFrame | None = None,
    df_forces: pd.DataFrame | None = None,
    bundle: dict[str, Any] | None = None,
    model_prefix: str | None = None,
    gnn_margin: float = 1.10,
    export_slots_path: Path | None = None,
) -> dict[str, Any]:
    """Run structural utilization stage and return reusable tables.

    This is a notebook-independent wrapper around compute_utilization_outputs.
    """
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
        gnn_marge=float(gnn_margin),
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
    )

    notebook_outputs = _package_structural_outputs_for_notebook(
        structural_out=structural_out,
        bundle_error=bundle_error,
    )

    if verbose:
        if notebook_outputs["SURROGATE_BUNDLE"] is not None:
            print("Surrogate bundle loaded and cached for iterative structural calls.")

        summary = notebook_outputs["summary"]
        print(f"Force source: {notebook_outputs['forces_source']}")
        print(
            f"Utilization: {summary['members']} members, "
            f"{summary['stock_items']} stock -> {summary['safe_combinations']} safe combinations"
        )

    return notebook_outputs
