from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd

import c21_surrogate_io as surrogate_io

def _to_numeric_vertex_id(value: Any) -> int:
    text = str(value).strip()
    if text.lower().startswith("v"):
        text = text[1:]
    return int(text)


def _resolve_edge_id_column(df_edges: pd.DataFrame) -> str | None:
    for candidate in ("edge_id", "Edge_ID", "Element_ID"):
        if candidate in df_edges.columns:
            return candidate
    return None


def _resolve_edge_area_column(df_edges: pd.DataFrame) -> str | None:
    for candidate in ("Area", "area", "cross_section_area", "A"):
        if candidate in df_edges.columns:
            return candidate
    return None


def _resolve_edge_length_column(df_edges: pd.DataFrame) -> str | None:
    for candidate in ("Length", "length_m", "length", "Length_m"):
        if candidate in df_edges.columns:
            return candidate
    return None


def validate_feasibility_stage_notebook_inputs(
    df_input_stock: pd.DataFrame | None,
    df_vertices: pd.DataFrame | None,
    df_edges: pd.DataFrame | None = None,
) -> None:
    """Validate required stage inputs and expected schema before expensive compute."""
    missing: list[str] = []
    if df_input_stock is None:
        missing.append("df_input_stock")
    if df_vertices is None:
        missing.append("df_vertices")
    if missing:
        raise ValueError("Missing required feasibility inputs: " + ", ".join(missing))

    assert df_input_stock is not None
    assert df_vertices is not None

    required_stock_cols = {"Member_ID", "Length", "Depth", "Width", "f_c0k", "f_tk", "E_modulus_eff"}
    missing_stock_cols = [c for c in required_stock_cols if c not in df_input_stock.columns]
    if missing_stock_cols:
        raise ValueError("df_input_stock missing required columns: " + ", ".join(missing_stock_cols))

    required_vertex_cols = {"x", "y", "z"}
    missing_vertex_cols = [c for c in required_vertex_cols if c not in df_vertices.columns]
    if missing_vertex_cols:
        raise ValueError("df_vertices missing required columns: " + ", ".join(missing_vertex_cols))

    if "vertex_index" not in df_vertices.columns and "node_id" not in df_vertices.columns:
        raise ValueError("df_vertices must include 'vertex_index' or 'node_id' for surrogate conversion.")

    if "Fz" not in df_vertices.columns and "layer" not in df_vertices.columns:
        raise ValueError("df_vertices must include either 'Fz' or 'layer' so nodal load can be resolved.")

    if df_edges is not None and _resolve_edge_id_column(df_edges) is None and len(df_edges) == 0:
        raise ValueError("df_edges must contain at least one edge row.")


def assign_roof_load_fz(
    df_vertices: pd.DataFrame,
    roof_load_kn_m2: float = 2.0,
    layer_column: str = "layer",
    top_layer_value: str = "top",
    bottom_layer_value: str = "bottom",
) -> pd.DataFrame:
    """Assign nodal Fz from distributed roof load over the top-node roof footprint.

    The roof load is projected on the XY plane and distributed to top vertices using
    a triangulation-based tributary area method:
    - Triangulate the top-node XY points with `matplotlib.tri.Triangulation`
    - For each triangle, distribute one-third of its area to each triangle node
    - Nodal force is `Fz = -roof_load_kn_m2 * tributary_area_m2`
    - Bottom nodes are explicitly set to `Fz = 0`

    Args:
        df_vertices: Vertex table containing at least `x`, `y`, `z` and `layer`.
        roof_load_kn_m2: Uniform roof surface load in kN/m2.
        layer_column: Column that stores top/bottom layer labels.
        top_layer_value: Value that marks upper (roof) vertices.
        bottom_layer_value: Value that marks lower vertices.

    Returns:
        A copy of `df_vertices` with columns:
        - `Fz`: nodal vertical load in kN (negative downward)
        - `roof_tributary_area_m2`: tributary roof area per node in m2

    Raises:
        ValueError: If required columns are missing or top-node geometry is invalid.
    """
    required_cols = {"x", "y", "z", layer_column}
    missing = [c for c in required_cols if c not in df_vertices.columns]
    if missing:
        raise ValueError("assign_roof_load_fz missing columns: " + ", ".join(missing))

    df_out = df_vertices.copy()
    df_out["roof_tributary_area_m2"] = 0.0
    df_out["Fz"] = 0.0

    top_mask = df_out[layer_column].astype(str).str.lower() == str(top_layer_value).lower()
    bottom_mask = df_out[layer_column].astype(str).str.lower() == str(bottom_layer_value).lower()

    df_top = df_out.loc[top_mask, ["x", "y"]].astype(float)
    if len(df_top) < 3:
        raise ValueError("Need at least 3 top-layer vertices to distribute roof load.")

    xy = df_top.to_numpy(dtype=float)
    if np.linalg.matrix_rank(xy - xy.mean(axis=0, keepdims=True)) < 2:
        raise ValueError("Top-layer vertices are collinear in XY; cannot define roof area.")

    # Local import keeps this utility lightweight unless structural stage uses it.
    import matplotlib.tri as mtri

    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    if tri.triangles is None or len(tri.triangles) == 0:
        raise ValueError("Could not triangulate top-layer roof footprint.")

    tributary_area = np.zeros(len(df_top), dtype=float)
    for a, b, c in tri.triangles:
        pa = xy[a]
        pb = xy[b]
        pc = xy[c]
        area = 0.5 * abs(np.cross(pb - pa, pc - pa))
        share = area / 3.0
        tributary_area[a] += share
        tributary_area[b] += share
        tributary_area[c] += share

    top_indices = df_out.index[top_mask].to_numpy()
    df_out.loc[top_indices, "roof_tributary_area_m2"] = tributary_area
    df_out.loc[top_indices, "Fz"] = -float(roof_load_kn_m2) * tributary_area

    # Preserve training convention: lower vertices carry no vertical roof load.
    df_out.loc[bottom_mask, "Fz"] = 0.0

    return df_out

def _collect_feasibility_reasons(slot, stock_item, utilization_failed):
    """Collect all active feasibility constraints for this slot-stock combination."""
    reasons = []
    if utilization_failed:
        reasons.append('Utilization')

    geometry_reason = _classify_geometry_constraint(slot, stock_item)
    if geometry_reason == 'Length':
        reasons.append(geometry_reason)

    return reasons if reasons else ['Passed']


def _classify_geometry_constraint(slot: pd.Series, stock_item: pd.Series) -> str:
    """Classify only length feasibility; cross-section is enforced by utilization."""
    req_length_m = float(slot.get("length_m", np.nan))
    if not np.isfinite(req_length_m):
        req_length_m = float(slot.get("Length_Req", np.nan)) / 1000.0
    stock_length_m = float(stock_item["Length"]) / 1000.0
    if np.isfinite(req_length_m) and stock_length_m < req_length_m:
        return "Length"
    return "Passed"

def _validate_surrogate_feature_availability(
    df_vertices: pd.DataFrame,
    df_edges: pd.DataFrame,
    bundle: dict[str, Any],
) -> None:
    """Fail fast when required surrogate features/topology are not fully available."""
    run_manifest = bundle.get("run_manifest") or {}
    scalers = bundle.get("scalers") or {}

    node_id_col = "vertex_index" if "vertex_index" in df_vertices.columns else "node_id"
    if node_id_col not in df_vertices.columns:
        raise ValueError("Surrogate guard: df_vertices must contain 'vertex_index' or 'node_id'.")

    required_node_features = tuple(
        scalers.get("node_cols")
        or run_manifest.get("selected_node_continuous_cols")
        or ("x", "y", "z", "Tx", "Ty", "Tz", "Rx", "Ry", "Rz", "Fz")
    )
    missing_node_features = [f for f in required_node_features if f not in df_vertices.columns]
    if missing_node_features:
        raise ValueError(
            "Surrogate guard: missing required node features for model inference: "
            + ", ".join(missing_node_features)
        )

    required_edge_features = tuple(
        scalers.get("edge_cols")
        or run_manifest.get("selected_edge_feature_cols")
        or ("Area", "Length", "E", "Iy", "Iz", "J", "EA/L")
    )
    supported_edge_features = {"Area", "Length", "E", "Iy", "Iz", "J", "EA/L"}
    unsupported_edge_features = [f for f in required_edge_features if f not in supported_edge_features]
    if unsupported_edge_features:
        raise ValueError(
            "Surrogate guard: unsupported edge features requested by model: "
            + ", ".join(unsupported_edge_features)
        )

    edge_index = bundle.get("edge_index")
    if edge_index is None:
        raise ValueError("Surrogate guard: bundle missing edge_index.")

    expected_edge_count = int(getattr(edge_index, "size", lambda dim: edge_index.shape[dim])(1))
    if int(len(df_edges)) != expected_edge_count:
        raise ValueError(
            "Surrogate guard: df_edges row count does not match model topology. "
            f"expected={expected_edge_count}, received={len(df_edges)}"
        )


def _build_edge_feature_frame(
    edge_ids: list[str],
    lengths_m: np.ndarray,
    stock_item: pd.Series,
    edge_cols: tuple[str, ...],
) -> pd.DataFrame:
    width_mm = float(stock_item["Width"])
    depth_mm = float(stock_item["Depth"])
    width_m = width_mm / 1000.0
    depth_m = depth_mm / 1000.0
    area_m2 = width_m * depth_m
    iy_m4 = (width_m * (depth_m ** 3)) / 12.0
    iz_m4 = (depth_m * (width_m ** 3)) / 12.0
    j_m4 = iy_m4 + iz_m4
    modulus_e = float(stock_item["E_modulus_eff"])

    lengths_m = np.asarray(lengths_m, dtype=float)
    lengths_m = np.where(lengths_m > 0.0, lengths_m, np.nan)
    lengths_mm = lengths_m * 1000.0
    area_mm2 = area_m2 * 1_000_000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ea_over_l = (modulus_e * area_mm2) / lengths_mm

    payload: dict[str, Any] = {"edge_id": edge_ids}
    for feature in edge_cols:
        if feature == "Length":
            payload[feature] = lengths_m
        elif feature == "Area":
            payload[feature] = float(area_m2)
        elif feature == "E":
            payload[feature] = float(modulus_e)
        elif feature == "Iy":
            payload[feature] = float(iy_m4)
        elif feature == "Iz":
            payload[feature] = float(iz_m4)
        elif feature == "J":
            payload[feature] = float(j_m4)
        elif feature == "EA/L":
            payload[feature] = ea_over_l
        else:
            raise ValueError(f"Unsupported edge feature '{feature}' for surrogate inference.")

    return pd.DataFrame(payload)

def prepare_surrogate_bundle(model_prefix: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """Try loading surrogate bundle once for re-use in iterative runs."""
    active_prefix = model_prefix
    try:
        bundle = surrogate_io.load_surrogate_bundle(prefix_sm=active_prefix)
        return bundle, None
    except Exception as exc:
        return None, f"Failed to load surrogate bundle '{active_prefix}': {exc}"