from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class V4Schema:
    node_continuous_cols: tuple[str, ...]
    node_mask_cols: tuple[str, ...]
    node_load_cols: tuple[str, ...]
    node_virtual_cols: tuple[str, ...]
    edge_feature_cols: tuple[str, ...]
    global_feature_cols: tuple[str, ...]
    node_count: int
    edge_count: int


def _numeric_suffix(value: str) -> int:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse numeric suffix from {value!r}")
    return int(digits)


def _check_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {label}: {', '.join(missing)}")


def load_v4_sources(node_csv: Path, edge_csv: Path, global_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_node = pd.read_csv(node_csv)
    df_edge = pd.read_csv(edge_csv)
    df_global = pd.read_csv(global_csv)

    _check_columns(df_node, ["sample_id", "node_id", "x", "y", "z", "Tx", "Ty", "Tz", "Rx", "Ry", "Rz"], "node CSV")
    _check_columns(df_edge, ["Sample_ID", "Edge_ID", "Source", "Target", "Area", "Length", "E", "Iy", "Iz", "J", "EA/L", "Axial_Force"], "edge CSV")
    _check_columns(df_global, ["sample_id", "Global_Load_Sum", "Total_Structural_Volume", "Average_Connectivity"], "global CSV")

    return df_node, df_edge, df_global


def _resolve_node_load_cols(df_node: pd.DataFrame) -> tuple[str, ...]:
    """Resolve the nodal load columns used as continuous node inputs.

    The current Grasshopper export is Fz-only. Fx/Fy are accepted as optional
    future extensions, but the intended training setup still works with Fz only.
    """
    if all(column in df_node.columns for column in ("Fx", "Fy", "Fz")):
        return ("Fx", "Fy", "Fz")
    if "Fz" in df_node.columns:
        return ("Fz",)
    raise ValueError("Node CSV must contain Fz as load data, or Fx/Fy/Fz for an extended export.")


def infer_v4_schema(
    df_node: pd.DataFrame,
    df_edge: pd.DataFrame,
    df_global: pd.DataFrame,
    use_virtual_node: bool = False,
) -> V4Schema:
    node_load_cols = _resolve_node_load_cols(df_node)
    node_continuous_cols = ("x", "y", "z", *node_load_cols)
    node_mask_cols = ("Tx", "Ty", "Tz", "Rx", "Ry", "Rz")
    node_virtual_cols = ("is_virtual",) if use_virtual_node else ()
    edge_feature_cols = ("Area", "Length", "E", "Iy", "Iz", "J", "EA/L")
    global_feature_cols = ("Global_Load_Sum", "Total_Structural_Volume", "Average_Connectivity")

    node_ids = sorted(df_node["node_id"].astype(str).unique().tolist(), key=_numeric_suffix)
    edge_ids = sorted(df_edge["Edge_ID"].astype(str).unique().tolist(), key=_numeric_suffix)

    if len(node_ids) == 0:
        raise ValueError("No node IDs found in node CSV")
    if len(edge_ids) == 0:
        raise ValueError("No edge IDs found in edge CSV")

    return V4Schema(
        node_continuous_cols=node_continuous_cols,
        node_mask_cols=node_mask_cols,
        node_load_cols=node_load_cols,
        node_virtual_cols=node_virtual_cols,
        edge_feature_cols=edge_feature_cols,
        global_feature_cols=global_feature_cols,
        node_count=len(node_ids),
        edge_count=len(edge_ids),
    )


def validate_sample_coverage(df_node: pd.DataFrame, df_edge: pd.DataFrame, df_global: pd.DataFrame) -> list[int]:
    node_samples = set(df_node["sample_id"].unique().tolist())
    edge_samples = set(df_edge["Sample_ID"].unique().tolist())
    global_samples = set(df_global["sample_id"].unique().tolist())

    if node_samples != edge_samples or node_samples != global_samples:
        raise ValueError(
            "sample_id coverage mismatch across CSVs: "
            f"node={len(node_samples)}, edge={len(edge_samples)}, global={len(global_samples)}"
        )

    return sorted(node_samples)


def build_edge_index(df_edge: pd.DataFrame) -> torch.Tensor:
    first_sample_id = df_edge["Sample_ID"].iloc[0]
    edge_reference = df_edge[df_edge["Sample_ID"] == first_sample_id].sort_values(
        by="Edge_ID",
        key=lambda series: series.map(_numeric_suffix),
    )
    sources = edge_reference["Source"].astype(int).tolist()
    targets = edge_reference["Target"].astype(int).tolist()
    bidirectional_sources = sources + targets
    bidirectional_targets = targets + sources
    return torch.tensor([bidirectional_sources, bidirectional_targets], dtype=torch.long)


def build_graph_dataset(
    df_node: pd.DataFrame,
    df_edge: pd.DataFrame,
    df_global: pd.DataFrame,
    schema: V4Schema,
    node_continuous_scaled: pd.DataFrame,
    node_mask_values: pd.DataFrame,
    edge_feature_scaled: pd.DataFrame,
    edge_target_scaled: pd.DataFrame,
    global_feature_scaled: pd.DataFrame,
    edge_index: torch.Tensor,
    use_virtual_node: bool = False,
) -> list[Data]:
    sample_ids = validate_sample_coverage(df_node, df_edge, df_global)

    dataset: list[Data] = []

    for sample_id in sample_ids:
        node_sample = df_node[df_node["sample_id"] == sample_id].copy()
        edge_sample = df_edge[df_edge["Sample_ID"] == sample_id].copy()
        global_sample = df_global[df_global["sample_id"] == sample_id].copy()
        graph_edge_index = edge_index.clone()

        node_sample = node_sample.sort_values(by="node_id", key=lambda series: series.map(_numeric_suffix))
        edge_sample = edge_sample.sort_values(by="Edge_ID", key=lambda series: series.map(_numeric_suffix))

        x_continuous = torch.tensor(node_continuous_scaled.loc[node_sample.index].to_numpy(), dtype=torch.float32)
        x_mask = torch.tensor(node_mask_values.loc[node_sample.index].to_numpy(), dtype=torch.float32)
        x = torch.cat([x_continuous, x_mask], dim=1)
        node_is_virtual = torch.zeros((x.size(0), 1), dtype=torch.float32)
        if use_virtual_node:
            x = torch.cat([x, node_is_virtual], dim=1)

        edge_attr = torch.tensor(edge_feature_scaled.loc[edge_sample.index].to_numpy(), dtype=torch.float32)
        y_edge = torch.tensor(edge_target_scaled.loc[edge_sample.index].to_numpy(), dtype=torch.float32)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        y_edge = torch.cat([y_edge, y_edge], dim=0)
        u = torch.tensor(global_feature_scaled.loc[global_sample.index].to_numpy().reshape(1, -1), dtype=torch.float32)

        edge_loss_mask = torch.ones((graph_edge_index.size(1), 1), dtype=torch.float32)

        if use_virtual_node:
            virtual_node_index = x.size(0)
            virtual_node_features = torch.zeros((1, x.size(1)), dtype=torch.float32)
            virtual_node_features[0, -1] = 1.0
            x = torch.cat([x, virtual_node_features], dim=0)
            node_is_virtual = torch.cat([node_is_virtual, torch.ones((1, 1), dtype=torch.float32)], dim=0)

            physical_node_indices = torch.arange(virtual_node_index, dtype=torch.long)
            virtual_sources = torch.cat([physical_node_indices, torch.full((virtual_node_index,), virtual_node_index, dtype=torch.long)])
            virtual_targets = torch.cat([torch.full((virtual_node_index,), virtual_node_index, dtype=torch.long), physical_node_indices])
            virtual_edge_index = torch.stack([virtual_sources, virtual_targets], dim=0)

            graph_edge_index = torch.cat([graph_edge_index, virtual_edge_index], dim=1)
            virtual_edge_attr = torch.zeros((virtual_edge_index.size(1), edge_attr.size(1)), dtype=torch.float32)
            virtual_y_edge = torch.zeros((virtual_edge_index.size(1), y_edge.size(1)), dtype=torch.float32)
            virtual_edge_mask = torch.zeros((virtual_edge_index.size(1), 1), dtype=torch.float32)

            edge_attr = torch.cat([edge_attr, virtual_edge_attr], dim=0)
            y_edge = torch.cat([y_edge, virtual_y_edge], dim=0)
            edge_loss_mask = torch.cat([edge_loss_mask, virtual_edge_mask], dim=0)

        dataset.append(
            Data(
                x=x,
                edge_index=graph_edge_index,
                edge_attr=edge_attr,
                y_edge=y_edge,
                edge_loss_mask=edge_loss_mask,
                node_is_virtual=node_is_virtual,
                u=u,
                sample_id=int(sample_id),
            )
        )

    return dataset


__all__ = [
    "V4Schema",
    "build_edge_index",
    "build_graph_dataset",
    "infer_v4_schema",
    "load_v4_sources",
    "validate_sample_coverage",
]
