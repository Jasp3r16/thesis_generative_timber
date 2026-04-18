from __future__ import annotations

import json
import importlib
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

import config
import c21_surrogate_io as surrogate_io
import c25_structural_check as structural_check

assign_roof_load_fz = structural_check.assign_roof_load_fz
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


def _geometry_df_to_design_row(
    df_geometry: pd.DataFrame,
    df_edges: pd.DataFrame | None = None,
    edge_count: int | None = None,
    default_edge_area_m2: float | None = None,
) -> pd.Series:
    """Convert vertex table to design-row format expected by surrogate IO."""
    required = ["x", "y", "z"]
    missing = [c for c in required if c not in df_geometry.columns]
    if missing:
        raise ValueError(f"Geometry dataframe misses required columns: {missing}")

    numeric_columns = list(df_geometry.select_dtypes(include=[np.number]).columns)
    if not all(column in numeric_columns for column in required):
        raise ValueError("Geometry dataframe must store x, y and z as numeric columns.")

    coords = df_geometry[numeric_columns].reset_index(drop=True).astype(float)
    payload: dict[str, float] = {}
    for idx, row in coords.iterrows():
        for column_name in coords.columns:
            payload[f"v{idx}_{column_name}"] = float(row[column_name])
    if df_edges is not None and len(df_edges) > 0:
        edge_table = df_edges.reset_index(drop=True)
        has_edge_id = "edge_id" in edge_table.columns
        has_area = "Area" in edge_table.columns
        if has_area:
            edge_table["Area"] = pd.to_numeric(edge_table["Area"], errors="coerce")

        for idx, edge_row in edge_table.iterrows():
            edge_key = str(edge_row["edge_id"]) if has_edge_id else f"e{idx}"
            if not edge_key.startswith("e"):
                edge_key = f"e{idx}"
            if has_area and pd.notna(edge_row["Area"]):
                payload[f"{edge_key}_Area"] = float(edge_row["Area"])

    if edge_count is not None and default_edge_area_m2 is not None:
        for edge_idx in range(int(edge_count)):
            payload.setdefault(f"e{edge_idx}_Area", float(default_edge_area_m2))

    return pd.Series(payload, dtype=np.float32)


def _estimate_default_edge_area_m2(df_input_stock: pd.DataFrame) -> float | None:
    required = {"Width", "Depth"}
    if not required.issubset(df_input_stock.columns):
        return None

    width_mm = pd.to_numeric(df_input_stock["Width"], errors="coerce")
    depth_mm = pd.to_numeric(df_input_stock["Depth"], errors="coerce")
    area_m2 = (width_mm * depth_mm) / 1_000_000.0
    area_m2 = area_m2.replace([np.inf, -np.inf], np.nan).dropna()
    if area_m2.empty:
        return None
    return float(area_m2.median())


def _load_edge_index_raw(edge_index_path: Path) -> tuple[list[int], list[int]]:
    with open(edge_index_path, "r", encoding="utf-8") as f:
        edge_index_raw = json.load(f)

    if isinstance(edge_index_raw, dict):
        start_nodes = (
            edge_index_raw.get("start_nodes")
            or edge_index_raw.get("source")
            or edge_index_raw.get("V1")
            or edge_index_raw.get("start")
        )
        end_nodes = (
            edge_index_raw.get("end_nodes")
            or edge_index_raw.get("target")
            or edge_index_raw.get("V2")
            or edge_index_raw.get("end")
        )
        if start_nodes is None or end_nodes is None:
            raise ValueError("edge_index.json missing required keys")
        return list(start_nodes), list(end_nodes)

    if isinstance(edge_index_raw, list) and len(edge_index_raw) == 2:
        return list(edge_index_raw[0]), list(edge_index_raw[1])

    raise ValueError("Unexpected edge_index.json format")


def _edge_length(df_geometry: pd.DataFrame, idx_a: int, idx_b: int) -> float:
    coord_a = df_geometry.iloc[idx_a][["x", "y", "z"]].values.astype(float)
    coord_b = df_geometry.iloc[idx_b][["x", "y", "z"]].values.astype(float)
    return float(np.linalg.norm(coord_b - coord_a))


def _predict_forces_with_fallback(
    df_vertices: pd.DataFrame,
    df_edges: pd.DataFrame | None,
    df_input_stock: pd.DataFrame,
    bundle: dict[str, Any] | None,
    model_prefix: str | None,
    use_synthetic_fallback: bool,
) -> tuple[pd.DataFrame, dict[str, Any] | None, str]:
    """Predict forces via surrogate; optionally fallback to synthetic values."""
    df_geometry = df_vertices.copy().reset_index(drop=True)

    # Apply distributed roof load as nodal Fz before surrogate inference.
    # By convention, top-layer nodes receive tributary load and bottom-layer nodes remain zero.
    df_geometry = assign_roof_load_fz(df_geometry, roof_load_kn_m2=2.0)

    try:
        active_bundle = bundle if bundle is not None else surrogate_io.load_surrogate_bundle(prefix_sm=model_prefix)
    except Exception:
        if not use_synthetic_fallback:
            raise

        edge_index_path = config.DATA_IO_PATH / "edge_index.json"
        start_nodes, end_nodes = _load_edge_index_raw(edge_index_path)

        predictions_records: list[dict[str, Any]] = []
        for i in range(len(start_nodes)):
            length_m = _edge_length(df_geometry, int(start_nodes[i]), int(end_nodes[i]))
            predictions_records.append(
                {
                    "edge_id": f"e{i}",
                    "V1": f"{start_nodes[i]}",
                    "V2": f"{end_nodes[i]}",
                    "length_m": round(length_m, 3),
                    "axial_force_kn": round(float(np.random.uniform(10, 50)), 2),
                }
            )

        return pd.DataFrame(predictions_records), bundle, "synthetic"

    edge_index_path = config.DATA_IO_PATH / "edge_index.json"
    start_nodes, _ = _load_edge_index_raw(edge_index_path)
    default_edge_area_m2 = _estimate_default_edge_area_m2(df_input_stock)

    design_row = _geometry_df_to_design_row(
        df_geometry=df_geometry,
        df_edges=df_edges,
        edge_count=len(start_nodes),
        default_edge_area_m2=default_edge_area_m2,
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
        importlib.reload(surrogate_io)
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
    use_synthetic_fallback: bool = True,
    gnn_margin: float = 1.10,
    swap_width_depth_req: bool = True,
    export_slots_path: Path | None = None,
) -> dict[str, Any]:
    """Run structural utilization stage and return reusable tables.

    This is a notebook-independent wrapper around compute_utilization_outputs.
    """
    if df_forces is None:
        if df_vertices is None:
            raise ValueError("Provide either df_forces or df_vertices for structural stage")
        df_forces, active_bundle, forces_source = _predict_forces_with_fallback(
            df_vertices=df_vertices,
            df_edges=df_edges,
            df_input_stock=df_input_stock,
            bundle=bundle,
            model_prefix=model_prefix,
            use_synthetic_fallback=use_synthetic_fallback,
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

    if swap_width_depth_req and {"Width_Req", "Depth_Req"}.issubset(df_slots.columns):
        df_slots[["Width_Req", "Depth_Req"]] = df_slots[["Depth_Req", "Width_Req"]].to_numpy()

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
    use_synthetic_fallback: bool = True,
    gnn_margin: float = 1.10,
    swap_width_depth_req: bool = True,
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
    bundle_error: str | None = None
    if active_bundle is None:
        active_bundle, bundle_error = prepare_surrogate_bundle(model_prefix=model_prefix)
        if active_bundle is None and not use_synthetic_fallback:
            raise RuntimeError(
                "Surrogate bundle unavailable and fallback is disabled. "
                f"Load reason: {bundle_error}"
            )

    structural_out = run_structural_stage(
        df_input_stock=df_input_stock,
        df_vertices=df_vertices,
        df_edges=df_edges,
        bundle=active_bundle,
        model_prefix=model_prefix,
        use_synthetic_fallback=use_synthetic_fallback,
        gnn_margin=gnn_margin,
        swap_width_depth_req=swap_width_depth_req,
        export_slots_path=export_slots_path,
    )

    notebook_outputs = _package_structural_outputs_for_notebook(
        structural_out=structural_out,
        bundle_error=bundle_error,
    )

    if verbose:
        if notebook_outputs["SURROGATE_BUNDLE"] is not None:
            print("Surrogate bundle loaded and cached for iterative structural calls.")
        else:
            print("Surrogate bundle unavailable; structural stage used synthetic fallback.")
            if notebook_outputs["SURROGATE_BUNDLE_ERROR"] is not None:
                print(f"Load reason: {notebook_outputs['SURROGATE_BUNDLE_ERROR']}")

        summary = notebook_outputs["summary"]
        print(f"Force source: {notebook_outputs['forces_source']}")
        print(
            f"Utilization: {summary['members']} members, "
            f"{summary['stock_items']} stock -> {summary['safe_combinations']} safe combinations"
        )

    return notebook_outputs
