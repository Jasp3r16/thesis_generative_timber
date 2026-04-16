from __future__ import annotations

import json
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
from c21_surrogate_io import load_surrogate_bundle, predict_edge_forces_kn
from c25_structural_check import compute_utilization_outputs


def _geometry_df_to_design_row(df_geometry: pd.DataFrame) -> pd.Series:
    """Convert vertex table to design-row format expected by surrogate IO."""
    required = ["x", "y", "z"]
    missing = [c for c in required if c not in df_geometry.columns]
    if missing:
        raise ValueError(f"Geometry dataframe misses required columns: {missing}")

    coords = df_geometry[required].reset_index(drop=True).astype(float)
    payload: dict[str, float] = {}
    for idx, row in coords.iterrows():
        payload[f"v{idx}_x"] = float(row["x"])
        payload[f"v{idx}_y"] = float(row["y"])
        payload[f"v{idx}_z"] = float(row["z"])
    return pd.Series(payload, dtype=np.float32)


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
    bundle: dict[str, Any] | None,
    model_prefix: str | None,
    use_synthetic_fallback: bool,
) -> tuple[pd.DataFrame, dict[str, Any] | None, str]:
    """Predict forces via surrogate; optionally fallback to synthetic values."""
    df_geometry = df_vertices.copy().reset_index(drop=True)

    try:
        active_bundle = bundle if bundle is not None else load_surrogate_bundle(prefix_sm=model_prefix)
        design_row = _geometry_df_to_design_row(df_geometry)
        df_forces = predict_edge_forces_kn(design_row, active_bundle).copy()
        df_forces["V1"] = df_forces["V1"].astype(str)
        df_forces["V2"] = df_forces["V2"].astype(str)
        df_forces["length_m"] = df_forces["length_m"].round(3)
        df_forces["axial_force_kn"] = df_forces["axial_force_kn"].round(2)
        return df_forces, active_bundle, "surrogate"
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


def prepare_surrogate_bundle(model_prefix: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """Try loading surrogate bundle once for re-use in iterative runs."""
    try:
        return load_surrogate_bundle(prefix_sm=model_prefix), None
    except Exception as exc:
        return None, str(exc)


def run_structural_stage(
    df_input_stock: pd.DataFrame,
    df_vertices: pd.DataFrame | None = None,
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
    safe_options = outputs["veilige_opties"]
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
