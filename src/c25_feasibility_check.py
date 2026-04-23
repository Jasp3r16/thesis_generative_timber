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

def geometry_df_to_design_row(
    df_geometry: pd.DataFrame,
    df_edges: pd.DataFrame | None = None,
    edge_area_m2: float | None = None,
) -> pd.Series:
    """Convert long-format geometry tables to wide surrogate design-row schema."""
    node_id_col = "vertex_index" if "vertex_index" in df_geometry.columns else "node_id"
    if node_id_col not in df_geometry.columns:
        raise ValueError("df_geometry must include 'vertex_index' or 'node_id'.")

    required_node_cols = {"x", "y", "z"}
    missing_node_cols = [c for c in required_node_cols if c not in df_geometry.columns]
    if missing_node_cols:
        raise ValueError("geometry_df_to_design_row missing node columns: " + ", ".join(missing_node_cols))

    df_nodes = df_geometry.copy()
    if "Fz" not in df_nodes.columns:
        if "layer" not in df_nodes.columns:
            raise ValueError("df_geometry requires Fz or layer to derive Fz values.")
        df_nodes = assign_roof_load_fz(df_nodes)

    df_nodes = df_nodes.sort_values(by=node_id_col, key=lambda s: s.map(_to_numeric_vertex_id)).reset_index(drop=True)

    payload: dict[str, float] = {}
    for _, row in df_nodes.iterrows():
        node_idx = _to_numeric_vertex_id(row[node_id_col])
        payload[f"v{node_idx}_x"] = float(row["x"])
        payload[f"v{node_idx}_y"] = float(row["y"])
        payload[f"v{node_idx}_z"] = float(row["z"])
        payload[f"v{node_idx}_Fz"] = float(row["Fz"])

    if df_edges is not None:
        edge_area_col = _resolve_edge_area_column(df_edges)
        if edge_area_col is None and edge_area_m2 is None:
            raise ValueError("df_edges missing edge area column (Area/cross_section_area/A).")

        edge_id_col = _resolve_edge_id_column(df_edges)
        df_edges_local = df_edges.copy()
        if edge_id_col is not None:
            df_edges_local["_edge_numeric"] = (
                df_edges_local[edge_id_col].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
            )
            df_edges_local = df_edges_local.sort_values(by="_edge_numeric", kind="stable")
        else:
            df_edges_local = df_edges_local.reset_index(drop=True)

        for edge_idx, (_, edge_row) in enumerate(df_edges_local.iterrows()):
            if edge_area_col is not None:
                payload[f"e{edge_idx}_Area"] = float(edge_row[edge_area_col])
            else:
                payload[f"e{edge_idx}_Area"] = float(edge_area_m2)

    payload["num_vertices"] = float(len(df_nodes))
    payload["num_edges"] = float(len(df_edges)) if df_edges is not None else 0.0
    return pd.Series(payload, dtype="float64")


def geometry_df_to_design_row_edge_mode(
    df_geometry: pd.DataFrame,
    df_edges: pd.DataFrame | None = None,
    edge_feature_mode: str = "area_length",
    edge_area_m2: float | None = None,
) -> pd.Series:
    """Build a surrogate design row for different edge-feature layouts.

    Modes:
    - ``area_length``: keep the existing area-aware behavior used by the complex model.
    - ``length_only``: omit area-dependent edge features and only provide geometry length.
    """
    mode = str(edge_feature_mode).strip().lower()
    if mode == "area_length":
        return geometry_df_to_design_row(df_geometry=df_geometry, df_edges=df_edges, edge_area_m2=edge_area_m2)

    if mode != "length_only":
        raise ValueError("Unsupported edge_feature_mode. Use 'area_length' or 'length_only'.")

    node_id_col = "vertex_index" if "vertex_index" in df_geometry.columns else "node_id"
    if node_id_col not in df_geometry.columns:
        raise ValueError("df_geometry must include 'vertex_index' or 'node_id'.")

    required_node_cols = {"x", "y", "z"}
    missing_node_cols = [c for c in required_node_cols if c not in df_geometry.columns]
    if missing_node_cols:
        raise ValueError("geometry_df_to_design_row_edge_mode missing node columns: " + ", ".join(missing_node_cols))

    df_nodes = df_geometry.copy()
    if "Fz" not in df_nodes.columns:
        if "layer" not in df_nodes.columns:
            raise ValueError("df_geometry requires Fz or layer to derive Fz values.")
        df_nodes = assign_roof_load_fz(df_nodes)

    df_nodes = df_nodes.sort_values(by=node_id_col, key=lambda s: s.map(_to_numeric_vertex_id)).reset_index(drop=True)

    payload: dict[str, float] = {}
    for _, row in df_nodes.iterrows():
        node_idx = _to_numeric_vertex_id(row[node_id_col])
        payload[f"v{node_idx}_x"] = float(row["x"])
        payload[f"v{node_idx}_y"] = float(row["y"])
        payload[f"v{node_idx}_z"] = float(row["z"])
        payload[f"v{node_idx}_Fz"] = float(row["Fz"])

    if df_edges is not None:
        edge_id_col = _resolve_edge_id_column(df_edges)
        df_edges_local = df_edges.copy()
        if edge_id_col is not None:
            df_edges_local["_edge_numeric"] = (
                df_edges_local[edge_id_col].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
            )
            df_edges_local = df_edges_local.sort_values(by="_edge_numeric", kind="stable")
        else:
            df_edges_local = df_edges_local.reset_index(drop=True)

        for edge_idx, _ in enumerate(df_edges_local.itertuples(index=False)):
            # Length-only models need no edge feature payload beyond geometry-derived length.
            payload[f"e{edge_idx}_Length"] = float("nan")

    payload["num_vertices"] = float(len(df_nodes))
    payload["num_edges"] = float(len(df_edges)) if df_edges is not None else 0.0
    return pd.Series(payload, dtype="float64")

def _predict_forces_with_surrogate(
    df_vertices: pd.DataFrame,
    df_edges: pd.DataFrame | None,
    bundle: dict[str, Any] | None,
    model_prefix: str | None,
    edge_area_m2: float | None = None,
    edge_feature_mode: str = "area_length",
) -> tuple[pd.DataFrame, dict[str, Any] | None, str]:
    """Predict forces via the surrogate model."""
    if df_edges is None:
        raise ValueError("df_edges is required for surrogate force prediction in c25.")

    bundle_local = bundle
    active_prefix = model_prefix
    if bundle_local is None:
        bundle_local, bundle_error = prepare_surrogate_bundle(active_prefix)
        if bundle_local is None:
            raise RuntimeError(bundle_error or "Could not load surrogate bundle.")

    design_row = geometry_df_to_design_row_edge_mode(
        df_geometry=df_vertices,
        df_edges=df_edges,
        edge_feature_mode=edge_feature_mode,
        edge_area_m2=edge_area_m2,
    )
    df_forces = surrogate_io.predict_edge_forces_kn(design_row=design_row, bundle=bundle_local)

    required_force_cols = {"edge_id", "length_m", "axial_force_kn"}
    missing_force_cols = [c for c in required_force_cols if c not in df_forces.columns]
    if missing_force_cols:
        raise ValueError("Surrogate prediction missing columns: " + ", ".join(missing_force_cols))

    return df_forces.copy(), bundle_local, f"surrogate:{active_prefix}"


def _validate_surrogate_feature_availability(
    df_vertices: pd.DataFrame,
    df_edges: pd.DataFrame,
    bundle: dict[str, Any],
) -> None:
    """Fail fast when required surrogate features/topology are not fully available."""
    run_manifest = bundle.get("run_manifest") or {}

    node_id_col = "vertex_index" if "vertex_index" in df_vertices.columns else "node_id"
    if node_id_col not in df_vertices.columns:
        raise ValueError("Surrogate guard: df_vertices must contain 'vertex_index' or 'node_id'.")

    required_node_features = tuple(run_manifest.get("selected_node_continuous_cols") or ("x", "y", "z", "Fz"))
    missing_node_features = [f for f in required_node_features if f not in df_vertices.columns]
    if missing_node_features:
        raise ValueError(
            "Surrogate guard: missing required node features for model inference: "
            + ", ".join(missing_node_features)
        )

    required_edge_features = tuple(run_manifest.get("selected_edge_feature_cols") or ("Area", "Length"))
    unsupported_edge_features = [f for f in required_edge_features if f not in {"Area", "Length"}]
    if unsupported_edge_features:
        raise ValueError(
            "Surrogate guard: current c25 stock-specific mode only supports edge features Area and Length. "
            "Model requires unsupported edge features: "
            + ", ".join(unsupported_edge_features)
        )

    expected_edge_count = int(bundle["edge_index"].size(1) // 2)
    if int(len(df_edges)) != expected_edge_count:
        raise ValueError(
            "Surrogate guard: df_edges row count does not match model topology. "
            f"expected={expected_edge_count}, received={len(df_edges)}"
        )


def _validate_length_only_surrogate_compatibility(bundle: dict[str, Any], model_prefix: str) -> None:
    """Fail fast when length-only mode is used with a non-length surrogate checkpoint."""
    run_manifest = bundle.get("run_manifest") or {}
    selected_edge_features = run_manifest.get("selected_edge_feature_cols")

    if isinstance(selected_edge_features, (list, tuple)) and len(selected_edge_features) > 0:
        normalized_features = [str(feature).strip().lower() for feature in selected_edge_features]
        if normalized_features != ["length"]:
            raise ValueError(
                "Length-only surrogate mode requires a checkpoint trained with edge feature schema ['Length']. "
                f"Prefix '{model_prefix}' was trained with edge features: {selected_edge_features}"
            )
        return

    edge_in_dim = int(bundle.get("edge_in_dim", 0) or 0)
    if edge_in_dim != 1:
        raise ValueError(
            "Length-only surrogate mode requires a checkpoint with a single edge input feature. "
            f"Prefix '{model_prefix}' reports edge_in_dim={edge_in_dim}."
        )


def _validate_area_length_surrogate_compatibility(bundle: dict[str, Any], model_prefix: str) -> None:
    """Fail fast when area-length mode is used with a non area-aware checkpoint."""
    run_manifest = bundle.get("run_manifest") or {}
    selected_edge_features = run_manifest.get("selected_edge_feature_cols")

    if isinstance(selected_edge_features, (list, tuple)) and len(selected_edge_features) > 0:
        normalized_features = [str(feature).strip().lower() for feature in selected_edge_features]
        if "area" not in normalized_features:
            raise ValueError(
                "Area-length surrogate mode requires a checkpoint trained with an 'Area' edge feature. "
                f"Prefix '{model_prefix}' was trained with edge features: {selected_edge_features}"
            )
        return

    edge_in_dim = int(bundle.get("edge_in_dim", 0) or 0)
    if edge_in_dim < 2:
        raise ValueError(
            "Area-length surrogate mode requires a checkpoint with at least two edge input features "
            "(typically Area and Length). "
            f"Prefix '{model_prefix}' reports edge_in_dim={edge_in_dim}."
        )


def compute_utilization_outputs_with_stock_specific_area(
    df_vertices: pd.DataFrame,
    df_edges: pd.DataFrame,
    df_input_stock: pd.DataFrame,
    bundle: dict[str, Any] | None = None,
    model_prefix: str | None = None,
    gnn_margin: float = 1.10,
    utilization_threshold: float = 1.0,
) -> dict[str, Any]:
    """Compute feasibility by predicting surrogate force for each stock-specific edge area.

    Each stock candidate contributes an area value A = Depth*Width (m2), which is injected
    as the edge `Area` feature before surrogate force prediction.
    """
    validate_feasibility_stage_notebook_inputs(
        df_input_stock=df_input_stock,
        df_vertices=df_vertices,
        df_edges=df_edges,
    )

    vertices = df_vertices.copy()
    if "Fz" not in vertices.columns:
        vertices = assign_roof_load_fz(vertices)

    stock = df_input_stock.copy()
    stock["Member_ID"] = stock["Member_ID"].astype(str)
    for numeric_col in ("Length", "Depth", "Width", "f_c0k", "f_tk", "E_modulus_eff"):
        stock[numeric_col] = pd.to_numeric(stock[numeric_col], errors="coerce")

    active_prefix = model_prefix
    bundle_local = bundle
    if bundle_local is None:
        bundle_local, bundle_error = prepare_surrogate_bundle(active_prefix)
        if bundle_local is None:
            raise RuntimeError(bundle_error or "Could not load surrogate bundle.")

    _validate_area_length_surrogate_compatibility(bundle=bundle_local, model_prefix=active_prefix)

    _validate_surrogate_feature_availability(
        df_vertices=vertices,
        df_edges=df_edges,
        bundle=bundle_local,
    )

    stock_ids = stock["Member_ID"].tolist()
    util_matrix: np.ndarray | None = None
    feas_matrix: np.ndarray | None = None
    edge_ids: list[str] = []
    df_slots: pd.DataFrame | None = None
    df_forces_reference: pd.DataFrame | None = None

    long_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    force_rows: list[dict[str, Any]] = []

    for j, (_, stock_item) in enumerate(stock.iterrows()):
        candidate_area_m2 = (float(stock_item["Depth"]) / 1000.0) * (float(stock_item["Width"]) / 1000.0)
        df_forces_candidate, _, _ = _predict_forces_with_surrogate(
            df_vertices=vertices,
            df_edges=df_edges,
            bundle=bundle_local,
            model_prefix=active_prefix,
            edge_area_m2=float(candidate_area_m2),
            edge_feature_mode="area_length",
        )

        if util_matrix is None:
            edge_ids = df_forces_candidate["edge_id"].astype(str).tolist()
            util_matrix = np.full((len(edge_ids), len(stock_ids)), np.inf, dtype=float)
            feas_matrix = np.full((len(edge_ids), len(stock_ids)), np.inf, dtype=float)
            df_slots = df_forces_candidate[["edge_id", "length_m", "axial_force_kn"]].copy()
            df_slots["Length_Req"] = (pd.to_numeric(df_slots["length_m"], errors="coerce") * 1000.0).round().astype(int)
            df_forces_reference = df_forces_candidate.copy()

        for i, (_, slot_force) in enumerate(df_forces_candidate.iterrows()):
            req_length_m = float(slot_force["length_m"])
            req_force_kn = float(slot_force["axial_force_kn"])
            utilization = calculate_utilization_for_dataset(
                stock_item,
                req_force_kn=req_force_kn,
                req_length_m=req_length_m,
                gnn_margin=float(gnn_margin),
            )

            assert util_matrix is not None
            assert feas_matrix is not None
            util_matrix[i, j] = utilization

            geometry_reason = _classify_geometry_constraint(slot_force, stock_item)
            utilization_failed = (not np.isfinite(utilization)) or (float(utilization) > float(utilization_threshold))
            reasons = _collect_feasibility_reasons(slot_force, stock_item, utilization_failed=utilization_failed)
            feasible = reasons == ["Passed"]
            if feasible:
                feas_matrix[i, j] = 0.0
            else:
                failure_rows.append(
                    {
                        "edge_id": str(slot_force["edge_id"]),
                        "Member_ID": str(stock_item["Member_ID"]),
                        "failure_reasons": ", ".join(reasons),
                        "geometry_reason": geometry_reason,
                        "utilization": float(utilization) if np.isfinite(utilization) else np.inf,
                    }
                )

            long_rows.append(
                {
                    "edge_id": str(slot_force["edge_id"]),
                    "Member_ID": str(stock_item["Member_ID"]),
                    "length_m": req_length_m,
                    "axial_force_kn": req_force_kn,
                    "candidate_area_m2": float(candidate_area_m2),
                    "utilization": float(utilization) if np.isfinite(utilization) else np.inf,
                    "is_feasible": bool(feasible),
                    "failure_reasons": ", ".join(reasons),
                }
            )
            force_rows.append(
                {
                    "edge_id": str(slot_force["edge_id"]),
                    "Member_ID": str(stock_item["Member_ID"]),
                    "candidate_area_m2": float(candidate_area_m2),
                    "length_m": req_length_m,
                    "axial_force_kn": req_force_kn,
                }
            )

    if util_matrix is None or feas_matrix is None or df_slots is None:
        raise ValueError("No feasibility data could be generated.")

    df_utilization_long = pd.DataFrame(long_rows)
    df_failure_reasons = pd.DataFrame(failure_rows)
    df_forces_by_stock = pd.DataFrame(force_rows)
    df_utilization_matrix_display = pd.DataFrame(util_matrix, index=edge_ids, columns=stock_ids)
    df_feasibility_matrix_display = pd.DataFrame(feas_matrix, index=edge_ids, columns=stock_ids)

    df_safe_options = df_utilization_long.loc[df_utilization_long["is_feasible"]].copy()

    df_req_dims = (
        df_safe_options.sort_values(["edge_id", "utilization"], ascending=[True, False])
        .drop_duplicates(subset=["edge_id"], keep="first")
        [["edge_id", "Member_ID", "utilization"]]
        .merge(
            stock[["Member_ID", "Depth", "Width"]],
            on="Member_ID",
            how="left",
        )
        .rename(columns={"Depth": "Depth_Req", "Width": "Width_Req", "utilization": "governing_utilization"})
    )

    df_slots = df_slots.merge(
        df_req_dims[["edge_id", "Depth_Req", "Width_Req", "governing_utilization"]],
        on="edge_id",
        how="left",
    )
    df_slots["Area_Req"] = (pd.to_numeric(df_slots["Depth_Req"], errors="coerce") * pd.to_numeric(df_slots["Width_Req"], errors="coerce")) / 1_000_000.0

    return {
        "bundle": bundle_local,
        "df_vertices": vertices,
        "df_forces": df_forces_reference if df_forces_reference is not None else pd.DataFrame(),
        "df_forces_by_stock": df_forces_by_stock,
        "df_utilization_long": df_utilization_long,
        "df_utilization_matrix": util_matrix,
        "df_utilization_matrix_display": df_utilization_matrix_display,
        "df_feasibility_matrix": feas_matrix,
        "df_feasibility_matrix_display": df_feasibility_matrix_display,
        "df_safe_options": df_safe_options,
        "df_failure_reasons": df_failure_reasons,
        "df_slots": df_slots,
        "utilization_threshold": float(utilization_threshold),
    }


def compute_utilization_outputs_length_only(
    df_vertices: pd.DataFrame,
    df_edges: pd.DataFrame,
    df_input_stock: pd.DataFrame,
    bundle: dict[str, Any] | None = None,
    model_prefix: str | None = None,
    gnn_margin: float = 1.10,
    utilization_threshold: float = 1.0,
) -> dict[str, Any]:
    """Compute feasibility using a length-only surrogate once per slot.

    This mode predicts one axial force value per edge and then reuses those forces for
    every stock candidate, which removes the repeated per-stock surrogate inference.
    """
    validate_feasibility_stage_notebook_inputs(
        df_input_stock=df_input_stock,
        df_vertices=df_vertices,
        df_edges=df_edges,
    )

    vertices = df_vertices.copy()
    if "Fz" not in vertices.columns:
        vertices = assign_roof_load_fz(vertices)

    stock = df_input_stock.copy()
    stock["Member_ID"] = stock["Member_ID"].astype(str)
    for numeric_col in ("Length", "Depth", "Width", "f_c0k", "f_tk", "E_modulus_eff"):
        stock[numeric_col] = pd.to_numeric(stock[numeric_col], errors="coerce")

    active_prefix = model_prefix
    bundle_local = bundle
    if bundle_local is None:
        bundle_local, bundle_error = prepare_surrogate_bundle(active_prefix)
        if bundle_local is None:
            raise RuntimeError(bundle_error or "Could not load surrogate bundle.")

    _validate_length_only_surrogate_compatibility(bundle=bundle_local, model_prefix=active_prefix)

    _validate_surrogate_feature_availability(
        df_vertices=vertices,
        df_edges=df_edges,
        bundle=bundle_local,
    )

    # Predict once for the geometry/slot set. This is the expensive step we want to keep single-pass.
    df_forces_reference, _, _ = _predict_forces_with_surrogate(
        df_vertices=vertices,
        df_edges=df_edges,
        bundle=bundle_local,
        model_prefix=active_prefix,
        edge_area_m2=None,
        edge_feature_mode="length_only",
    )

    edge_ids = df_forces_reference["edge_id"].astype(str).tolist()
    stock_ids = stock["Member_ID"].tolist()
    util_matrix = np.full((len(edge_ids), len(stock_ids)), np.inf, dtype=float)
    feas_matrix = np.full((len(edge_ids), len(stock_ids)), np.inf, dtype=float)

    long_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    force_rows: list[dict[str, Any]] = []

    for i, (_, slot_force) in enumerate(df_forces_reference.iterrows()):
        req_length_m = float(slot_force["length_m"])
        req_force_kn = float(slot_force["axial_force_kn"])

        for j, (_, stock_item) in enumerate(stock.iterrows()):
            utilization = calculate_utilization_for_dataset(
                stock_item,
                req_force_kn=req_force_kn,
                req_length_m=req_length_m,
                gnn_margin=float(gnn_margin),
            )

            util_matrix[i, j] = utilization

            geometry_reason = _classify_geometry_constraint(slot_force, stock_item)
            utilization_failed = (not np.isfinite(utilization)) or (float(utilization) > float(utilization_threshold))
            reasons = _collect_feasibility_reasons(slot_force, stock_item, utilization_failed=utilization_failed)
            feasible = reasons == ["Passed"]
            if feasible:
                feas_matrix[i, j] = 0.0
            else:
                failure_rows.append(
                    {
                        "edge_id": str(slot_force["edge_id"]),
                        "Member_ID": str(stock_item["Member_ID"]),
                        "failure_reasons": ", ".join(reasons),
                        "geometry_reason": geometry_reason,
                        "utilization": float(utilization) if np.isfinite(utilization) else np.inf,
                    }
                )

            long_rows.append(
                {
                    "edge_id": str(slot_force["edge_id"]),
                    "Member_ID": str(stock_item["Member_ID"]),
                    "length_m": req_length_m,
                    "axial_force_kn": req_force_kn,
                    "utilization": float(utilization) if np.isfinite(utilization) else np.inf,
                    "is_feasible": bool(feasible),
                    "failure_reasons": ", ".join(reasons),
                }
            )
            force_rows.append(
                {
                    "edge_id": str(slot_force["edge_id"]),
                    "Member_ID": str(stock_item["Member_ID"]),
                    "length_m": req_length_m,
                    "axial_force_kn": req_force_kn,
                }
            )

    df_utilization_long = pd.DataFrame(long_rows)
    df_failure_reasons = pd.DataFrame(failure_rows)
    df_forces_by_stock = pd.DataFrame(force_rows)
    df_utilization_matrix_display = pd.DataFrame(util_matrix, index=edge_ids, columns=stock_ids)
    df_feasibility_matrix_display = pd.DataFrame(feas_matrix, index=edge_ids, columns=stock_ids)
    df_safe_options = df_utilization_long.loc[df_utilization_long["is_feasible"]].copy()

    df_slots = df_forces_reference[["edge_id", "length_m", "axial_force_kn"]].copy()
    df_slots["Length_Req"] = (df_slots["length_m"] * 1000.0).round().astype(int)

    df_req_dims = (
        df_safe_options.sort_values(["edge_id", "utilization"], ascending=[True, False])
        .drop_duplicates(subset=["edge_id"], keep="first")
        [["edge_id", "Member_ID", "utilization"]]
        .merge(
            stock[["Member_ID", "Depth", "Width"]],
            on="Member_ID",
            how="left",
        )
        .rename(columns={"Depth": "Depth_Req", "Width": "Width_Req", "utilization": "governing_utilization"})
    )

    df_slots = df_slots.merge(df_req_dims[["edge_id", "Depth_Req", "Width_Req", "governing_utilization"]], on="edge_id", how="left")
    df_slots["Area_Req"] = (pd.to_numeric(df_slots["Depth_Req"], errors="coerce") * pd.to_numeric(df_slots["Width_Req"], errors="coerce")) / 1_000_000.0

    return {
        "bundle": bundle_local,
        "df_vertices": vertices,
        "df_forces": df_forces_reference,
        "df_forces_by_stock": df_forces_by_stock,
        "df_utilization_long": df_utilization_long,
        "df_utilization_matrix": util_matrix,
        "df_utilization_matrix_display": df_utilization_matrix_display,
        "df_feasibility_matrix": feas_matrix,
        "df_feasibility_matrix_display": df_feasibility_matrix_display,
        "df_safe_options": df_safe_options,
        "df_failure_reasons": df_failure_reasons,
        "df_slots": df_slots,
        "utilization_threshold": float(utilization_threshold),
    }

def prepare_surrogate_bundle(model_prefix: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """Try loading surrogate bundle once for re-use in iterative runs."""
    active_prefix = model_prefix
    try:
        bundle = surrogate_io.load_surrogate_bundle(prefix_sm=active_prefix)
        return bundle, None
    except Exception as exc:
        return None, f"Failed to load surrogate bundle '{active_prefix}': {exc}"


def _validate_force_and_stock_inputs(
    df_forces: pd.DataFrame,
    df_input_stock: pd.DataFrame,
) -> None:
    required_force_cols = {"edge_id", "length_m", "axial_force_kn"}
    missing_force = [c for c in required_force_cols if c not in df_forces.columns]
    if missing_force:
        raise ValueError("df_forces missing required columns: " + ", ".join(missing_force))

    required_stock_cols = {"Member_ID", "Length", "Depth", "Width", "f_c0k", "f_tk", "E_modulus_eff"}
    missing_stock = [c for c in required_stock_cols if c not in df_input_stock.columns]
    if missing_stock:
        raise ValueError("df_input_stock missing required columns: " + ", ".join(missing_stock))
    
def calculate_utilization_for_dataset(
    row: pd.Series,
    req_force_kn: float,
    req_length_m: float,
    gnn_margin: float = 1.10,
) -> float:
    """
    - Calculates area for an element
    - Uses area and surrogate model to calculate axial force with surrogate model
    - Than uses that axial force to calculate Eurocode 5 utilization for one stock element

    Args:
        row: Row with at least `Depth`, `Width`, `f_c0k`, `f_tk`, `E_modulus_eff`.
        req_length_m: Requested member length in meters for buckling calculation.
        gnn_margin: Safety factor on predicted force (default 1.10).

    Returns:
        Utilization ratio. Values <= 1.0 are structurally acceptable.
        Returns `np.inf` if the calculated capacity is invalid or non-positive.
    """
    h = float(row["Depth"])
    b = float(row["Width"])
    l_mm = float(req_length_m) * 1000.0
    area = h * b

    required_force_kn = float(req_force_kn) * float(gnn_margin)

    f_c_k = float(row["f_c0k"])
    e_0_mean = float(row["E_modulus_eff"])
    f_t_k = float(row["f_tk"])

    gamma_m = 1.3
    k_mod = 0.8
    f_c_d = (f_c_k * k_mod) / gamma_m
    f_t_d = (f_t_k * k_mod) / gamma_m

    if required_force_kn >= 0:
        force_n = required_force_kn * 1000.0
        capaciteit_n = area * f_t_d
        if capaciteit_n <= 0:
            return float(np.inf)
        return force_n / capaciteit_n

    force_n = abs(required_force_kn * 1000.0)
    i_min = (max(h, b) * min(h, b) ** 3) / 12.0
    i_radius = math.sqrt(i_min / area)
    slenderness = l_mm / i_radius
    rel_slenderness = (slenderness / math.pi) * math.sqrt(f_c_k / e_0_mean)
    beta_c = 0.2
    k_waarde = 0.5 * (1 + beta_c * (rel_slenderness - 0.3) + rel_slenderness**2)
    k_c = 1 / (k_waarde + math.sqrt(max(0.0, k_waarde**2 - rel_slenderness**2)))

    capaciteit_n = area * k_c * f_c_d
    if capaciteit_n <= 0:
        return float(np.inf)
    return force_n / capaciteit_n


def compute_utilization_outputs(
    df_forces: pd.DataFrame,
    df_input_stock: pd.DataFrame,
    gnn_margin: float = 1.10,
    utilization_threshold: float = 1.0,
) -> dict[str, Any]:
    """Build utilization and feasibility outputs for all slot-stock combinations."""
    _validate_force_and_stock_inputs(df_forces=df_forces, df_input_stock=df_input_stock)

    forces = df_forces.copy()
    forces["edge_id"] = forces["edge_id"].astype(str)
    forces["length_m"] = pd.to_numeric(forces["length_m"], errors="coerce")
    forces["axial_force_kn"] = pd.to_numeric(forces["axial_force_kn"], errors="coerce")

    stock = df_input_stock.copy()
    stock["Member_ID"] = stock["Member_ID"].astype(str)
    for numeric_col in ("Length", "Depth", "Width", "f_c0k", "f_tk", "E_modulus_eff"):
        stock[numeric_col] = pd.to_numeric(stock[numeric_col], errors="coerce")

    edge_ids = forces["edge_id"].tolist()
    stock_ids = stock["Member_ID"].tolist()

    util_matrix = np.full((len(edge_ids), len(stock_ids)), np.inf, dtype=float)
    feas_matrix = np.full((len(edge_ids), len(stock_ids)), np.inf, dtype=float)

    long_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for i, (_, slot) in enumerate(forces.iterrows()):
        req_length_m = float(slot["length_m"])
        req_force_kn = float(slot["axial_force_kn"])

        for j, (_, stock_item) in enumerate(stock.iterrows()):
            utilization = calculate_utilization_for_dataset(
                stock_item,
                req_force_kn=req_force_kn,
                req_length_m=req_length_m,
                gnn_margin=float(gnn_margin),
            )

            util_matrix[i, j] = utilization

            geometry_reason = _classify_geometry_constraint(slot, stock_item)
            utilization_failed = (not np.isfinite(utilization)) or (float(utilization) > float(utilization_threshold))
            reasons = _collect_feasibility_reasons(slot, stock_item, utilization_failed=utilization_failed)
            feasible = reasons == ["Passed"]
            if feasible:
                feas_matrix[i, j] = 0.0
            else:
                failure_rows.append(
                    {
                        "edge_id": str(slot["edge_id"]),
                        "Member_ID": str(stock_item["Member_ID"]),
                        "failure_reasons": ", ".join(reasons),
                        "geometry_reason": geometry_reason,
                        "utilization": float(utilization) if np.isfinite(utilization) else np.inf,
                    }
                )

            long_rows.append(
                {
                    "edge_id": str(slot["edge_id"]),
                    "Member_ID": str(stock_item["Member_ID"]),
                    "length_m": req_length_m,
                    "axial_force_kn": req_force_kn,
                    "utilization": float(utilization) if np.isfinite(utilization) else np.inf,
                    "is_feasible": bool(feasible),
                    "failure_reasons": ", ".join(reasons),
                }
            )

    df_utilization_long = pd.DataFrame(long_rows)
    df_failure_reasons = pd.DataFrame(failure_rows)

    df_utilization_matrix_display = pd.DataFrame(util_matrix, index=edge_ids, columns=stock_ids)
    df_feasibility_matrix_display = pd.DataFrame(feas_matrix, index=edge_ids, columns=stock_ids)

    df_safe_options = df_utilization_long.loc[df_utilization_long["is_feasible"]].copy()

    df_slots = forces[["edge_id", "length_m", "axial_force_kn"]].copy()
    df_slots["Length_Req"] = (df_slots["length_m"] * 1000.0).round().astype(int)

    # Use the most critical feasible candidate (highest utilization <= threshold) to infer required section.
    df_req_dims = (
        df_safe_options.sort_values(["edge_id", "utilization"], ascending=[True, False])
        .drop_duplicates(subset=["edge_id"], keep="first")
        [["edge_id", "Member_ID", "utilization"]]
        .merge(
            stock[["Member_ID", "Depth", "Width"]],
            on="Member_ID",
            how="left",
        )
        .rename(columns={"Depth": "Depth_Req", "Width": "Width_Req", "utilization": "governing_utilization"})
    )

    df_slots = df_slots.merge(df_req_dims[["edge_id", "Depth_Req", "Width_Req", "governing_utilization"]], on="edge_id", how="left")
    df_slots["Area_Req"] = (pd.to_numeric(df_slots["Depth_Req"], errors="coerce") * pd.to_numeric(df_slots["Width_Req"], errors="coerce")) / 1_000_000.0

    return {
        "df_utilization_long": df_utilization_long,
        "df_utilization_matrix": util_matrix,
        "df_utilization_matrix_display": df_utilization_matrix_display,
        "df_feasibility_matrix": feas_matrix,
        "df_feasibility_matrix_display": df_feasibility_matrix_display,
        "df_safe_options": df_safe_options,
        "df_failure_reasons": df_failure_reasons,
        "df_slots": df_slots,
        "utilization_threshold": float(utilization_threshold),
    }