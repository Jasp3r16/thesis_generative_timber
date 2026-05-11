from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

import c25_feasibility_check as feasibility_check
import c21_surrogate_io as surrogate_io


def run_feasibility_stage(
    df_input_stock: pd.DataFrame,
    df_vertices: pd.DataFrame,
    df_edges: pd.DataFrame | None = None,
    model_prefix: str | None = None,
    roof_load_kn_m2: float = 2.0,
) -> dict[str, Any]:
    """
    Runs the feasibility stage of the workflow.

    For each stock item, runs full-structure surrogate inference to predict which members
    (edges) are safe or unsafe. Combines surrogate safety prediction with length constraints.

    Returns a dict with feasibility matrix and associated artifacts.

    Args:
        df_input_stock: Stock catalogue with columns [Member_ID, Length, Depth, Width, f_c0k, f_tk, E_modulus_eff, ...].
        df_vertices: Vertex/node data with columns [x, y, z, attribute, layer(?), ...].
                     attribute marks support/load/hinge conditions.
        df_edges: Edge/member data with columns [edge_id/Element_ID/Edge_ID, Length, ...].
                  Must match the trained surrogate topology exactly (e.g., 240 edges for the reference model).
        model_prefix: Surrogate model prefix. Defaults to the fixed model for this stage.
        roof_load_kn_m2: Distributed roof load in kN/m² for nodal Fz assignment (default 2.0).

    Returns:
        dict with keys:
            - 'bundle': Surrogate model bundle
            - 'df_slots': DataFrame with slot (edge) information
            - 'df_feasibility_matrix': Slot x Stock feasibility matrix (inf=infeasible, 0=feasible)
            - 'df_safe_options': For each slot, list of safe stock options
            - 'df_failure_reasons': Detailed failure reason per slot-stock combination
            - 'summary': Dict with summary statistics

    Raises:
        ValueError: If the edge count does not match the surrogate's trained topology.
    """
    # Default model for this stage
    if model_prefix is None:
        model_prefix = "ID20260510_224228_LR3e-04_EP150_BS32_FA0.50_ROC0.874"

    # Validate inputs
    feasibility_check.validate_feasibility_stage_notebook_inputs(
        df_input_stock=df_input_stock,
        df_vertices=df_vertices,
        df_edges=df_edges,
    )

    if df_edges is None or len(df_edges) == 0:
        raise ValueError("run_feasibility_stage requires non-empty df_edges for surrogate topology.")

    # Prepare node features from vertices (fixed across all stock iterations)
    df_nodes = _prepare_node_features(df_vertices, roof_load_kn_m2=roof_load_kn_m2)

    # Load surrogate bundle once
    bundle, err = feasibility_check.prepare_surrogate_bundle(model_prefix=model_prefix)
    if bundle is None:
        raise ValueError(f"Failed to load surrogate bundle: {err}")

    # Validate topology match: the edge count must match what the surrogate expects
    edge_index = bundle.get("edge_index")
    expected_edge_count = int(edge_index.shape[1])
    actual_edge_count = len(df_edges)
    
    # Handle 120→240 edge case by duplicating edges (treating 120 undirected as 240 bidirectional)
    if actual_edge_count == 120 and expected_edge_count == 240:
        df_edges_expanded = pd.concat(
            [df_edges, df_edges.copy()],
            ignore_index=True
        )
        df_edges = df_edges_expanded.reset_index(drop=True)
        print(f"INFO: Expanded {actual_edge_count} edges to {len(df_edges)} (bidirectional representation)")
    elif actual_edge_count != expected_edge_count:
        raise ValueError(
            f"Edge count mismatch with surrogate topology. "
            f"Expected {expected_edge_count} edges (from trained model), got {actual_edge_count}."
        )

    # Resolve edge ID column for indexing
    edge_id_col = feasibility_check._resolve_edge_id_column(df_edges)
    if edge_id_col is None:
        raise ValueError("df_edges must contain one of: edge_id, Edge_ID, Element_ID")

    # Initialize feasibility matrix: rows=edges (slots), cols=stock items
    slot_ids = df_edges[edge_id_col].astype(str).tolist()
    stock_ids = df_input_stock["Member_ID"].astype(str).tolist()
    feasibility_matrix = pd.DataFrame(
        data=0.0,  # Initialize as feasible; will mark inf for failures
        index=pd.Index(slot_ids, name="slot_id"),
        columns=pd.Index(stock_ids, name="Member_ID"),
    )

    # For each stock item, run full-graph surrogate inference
    for stock_idx, (stock_id, stock_row) in enumerate(df_input_stock.iterrows()):
        try:
            # Build ALL edge features for ALL edges using this stock item
            edge_features_df = _build_all_edge_features(
                df_edges=df_edges,
                stock_item=stock_row,
                bundle=bundle,
            )

            # Run surrogate inference on the full graph with this stock
            safety_predictions = _run_full_graph_inference(
                df_vertices=df_nodes,
                edge_features_df=edge_features_df,
                bundle=bundle,
            )

            # For each edge, check both surrogate safety and length constraint
            for edge_idx, (edge_id, edge_row) in enumerate(df_edges.iterrows()):
                is_unsafe = safety_predictions[edge_idx]
                length_feasible = _check_length_constraint(edge_row, stock_row)

                # Mark infeasible (inf) if either surrogate predicts unsafe OR length fails
                if is_unsafe or not length_feasible:
                    feasibility_matrix.iloc[edge_idx, stock_idx] = np.inf

        except Exception as exc:
            # If any error occurs for this stock item, mark all combinations as infeasible
            feasibility_matrix.iloc[:, stock_idx] = np.inf

    # Build output artifacts
    edge_id_col = feasibility_check._resolve_edge_id_column(df_edges)
    edge_ids = df_edges[edge_id_col].astype(str).unique()
    
    # Create df_slots (edge/slot info)
    df_slots_list = []
    for edge_id in edge_ids:
        edge_rows = df_edges[df_edges[edge_id_col].astype(str) == edge_id]
        if len(edge_rows) > 0:
            edge_row = edge_rows.iloc[0]
            df_slots_list.append({
                "edge_id": edge_id,
                "slot_index": list(edge_ids).index(edge_id) if edge_id in edge_ids else 0,
            })
    df_slots = pd.DataFrame(df_slots_list)
    
    # Create df_safe_options: for each slot, which stock items are feasible
    df_safe_options_list = []
    for slot_id in feasibility_matrix.index:
        feasible_stocks = feasibility_matrix.loc[slot_id]
        feasible_stocks = feasible_stocks[np.isfinite(feasible_stocks)].index.tolist()
        for stock_id in feasible_stocks:
            df_safe_options_list.append({
                "slot_id": slot_id,
                "Member_ID": stock_id,
                "feasible": True,
            })
    df_safe_options = pd.DataFrame(df_safe_options_list) if df_safe_options_list else pd.DataFrame(
        columns=["slot_id", "Member_ID", "feasible"]
    )
    
    # Create df_failure_reasons: why each combination failed
    df_failure_reasons_list = []
    for slot_id in feasibility_matrix.index:
        infeasible_stocks = feasibility_matrix.loc[slot_id]
        infeasible_stocks = infeasible_stocks[~np.isfinite(infeasible_stocks)].index.tolist()
        for stock_id in infeasible_stocks:
            df_failure_reasons_list.append({
                "edge_id": slot_id,
                "Member_ID": stock_id,
                "reason": "Surrogate safety or length constraint failed",
            })
    df_failure_reasons = pd.DataFrame(df_failure_reasons_list) if df_failure_reasons_list else pd.DataFrame(
        columns=["edge_id", "Member_ID", "reason"]
    )
    
    # Summary statistics
    total_pairs = len(feasibility_matrix) * len(feasibility_matrix.columns)
    feasible_pairs = int((np.isfinite(feasibility_matrix.values)).sum())
    summary = {
        "slots": len(feasibility_matrix),
        "stock_items": len(feasibility_matrix.columns),
        "total_pairs": total_pairs,
        "feasible_pairs": feasible_pairs,
        "feasibility_ratio": feasible_pairs / total_pairs if total_pairs > 0 else 0,
        "model_prefix": model_prefix,
        "roof_load_kn_m2": roof_load_kn_m2,
    }

    return {
        "bundle": bundle,
        "df_slots": df_slots,
        "df_feasibility_matrix": feasibility_matrix,
        "df_safe_options": df_safe_options,
        "df_failure_reasons": df_failure_reasons,
        "summary": summary,
    }


def _prepare_node_features(
    df_vertices: pd.DataFrame,
    roof_load_kn_m2: float = 2.0,
) -> pd.DataFrame:
    """
    Prepare node feature table with all required columns for surrogate.

    Returns df with: x, y, z, Tx, Ty, Tz, Rx, Ry, Rz, Fz
    
    DOF Assignment based on attribute column:
    - 'support': Tx=1, Ty=1, Tz=1, Rx=0, Ry=0, Rz=0
    - 'hinge': Tx=0, Ty=0, Tz=0, Rx=0, Ry=0, Rz=0, Fz=0
    - 'load': Tx=0, Ty=0, Tz=0, Rx=0, Ry=0, Rz=0 (Fz computed from roof load if top layer)
    
    Fz Calculation:
    - Computed only for 'support' and 'load' nodes in top layer
    - Always 0 for 'hinge' nodes
    """
    df = df_vertices.copy()

    # Initialize all DOF to 0
    df["Tx"] = 0
    df["Ty"] = 0
    df["Tz"] = 0
    df["Rx"] = 0
    df["Ry"] = 0
    df["Rz"] = 0
    df["Fz"] = 0.0

    if "attribute" in df.columns:
        attr_lower = df["attribute"].astype(str).str.lower()

        # Support nodes: Tx=Ty=Tz=1, Rx=Ry=Rz=0
        support_mask = attr_lower.str.contains("support", na=False)
        if support_mask.any():
            df.loc[support_mask, ["Tx", "Ty", "Tz"]] = 1

        # Hinge nodes: all DOF = 0, Fz explicitly 0
        hinge_mask = attr_lower.str.contains("hinge", na=False)
        if hinge_mask.any():
            df.loc[hinge_mask, ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz", "Fz"]] = 0

        # Load nodes: all DOF = 0, Fz computed from roof load if top layer
        load_mask = attr_lower.str.contains("load", na=False)
        # (Fz will be set below during roof load assignment)

    # Assign roof load to Fz for support and load nodes in top layer
    if "layer" in df.columns:
        df_with_roof = feasibility_check.assign_roof_load_fz(
            df_vertices=df,
            roof_load_kn_m2=roof_load_kn_m2,
            layer_column="layer",
        )
        # Only update Fz for non-hinge nodes
        if "attribute" in df.columns:
            attr_lower = df["attribute"].astype(str).str.lower()
            hinge_mask = attr_lower.str.contains("hinge", na=False)
            df.loc[~hinge_mask, "Fz"] = df_with_roof.loc[~hinge_mask, "Fz"]
            # Keep hinge Fz = 0
            df.loc[hinge_mask, "Fz"] = 0.0
        else:
            df["Fz"] = df_with_roof["Fz"]
    # If no layer column, Fz stays 0 for all (initialized above)

    return df


def _build_all_edge_features(
    df_edges: pd.DataFrame,
    stock_item: pd.Series,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    """
    Build edge feature table for ALL edges in the structure using a single stock item.

    Returns a DataFrame with one row per edge, with features matching the bundle's edge_cols schema.
    """
    scalers = bundle.get("scalers", {})
    edge_cols = scalers.get("edge_cols", ("Area", "Length", "E", "Iy", "Iz", "J", "EA/L"))

    # Build edge features for ALL edges using this stock item's properties
    edge_ids_list = []
    lengths_m_list = []

    for edge_idx, (_, edge_row) in enumerate(df_edges.iterrows()):
        # Get edge length from edge data or fallback to stock length
        edge_length_col = feasibility_check._resolve_edge_length_column(edge_row.to_frame().T)
        if edge_length_col:
            length_mm = float(edge_row[edge_length_col])
        else:
            length_mm = float(stock_item["Length"])

        length_m = length_mm / 1000.0 if length_mm > 1 else length_mm

        # Resolve edge ID
        edge_id_col = feasibility_check._resolve_edge_id_column(df_edges)
        edge_id = str(edge_row[edge_id_col]) if edge_id_col else str(edge_idx)

        edge_ids_list.append(edge_id)
        lengths_m_list.append(length_m)

    # Build the feature frame for all edges at once
    edge_df = feasibility_check._build_edge_feature_frame(
        edge_ids=edge_ids_list,
        lengths_m=np.array(lengths_m_list),
        stock_item=stock_item,
        edge_cols=edge_cols,
    )

    return edge_df


def _run_full_graph_inference(
    df_vertices: pd.DataFrame,
    edge_features_df: pd.DataFrame,
    bundle: dict[str, Any],
) -> np.ndarray:
    """
    Run surrogate model inference on the full structure graph.

    Returns a boolean array where True = unsafe (predicted failure), False = safe.
    """
    scalers = bundle.get("scalers", {})
    node_cols = scalers.get("node_cols", ("x", "y", "z", "Tx", "Ty", "Tz", "Rx", "Ry", "Rz", "Fz"))

    # Ensure node features are in the correct order and present
    df_nodes_subset = df_vertices[list(node_cols)].copy()

    # Run inference
    result_df = surrogate_io.predict_edge_failure_probabilities(
        nodes_df=df_nodes_subset,
        edges_df=edge_features_df,
        bundle=bundle,
        edge_index=bundle.get("edge_index"),
    )

    # Extract failure probabilities for all edges
    failure_probs = result_df.get("failure_prob", result_df.get("pred_prob", result_df.iloc[:, 0])).values

    # Determine safety threshold from bundle or use default
    threshold = bundle.get("safety_threshold", 0.5)

    # Return True (unsafe) if failure_prob >= threshold
    is_unsafe_array = (failure_probs >= threshold).astype(bool)

    return is_unsafe_array


def _check_length_constraint(slot_row: pd.Series, stock_item: pd.Series) -> bool:
    """
    Check whether the stock item's length satisfies the slot's required length.

    Returns True if feasible (stock length >= required length), False otherwise.
    """
    return feasibility_check._classify_geometry_constraint(slot_row, stock_item) == "Passed"


