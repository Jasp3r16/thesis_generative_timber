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

    _check_columns(df_node, ["sample_id", "node_id", "x", "y", "z", "Tx", "Ty", "Tz", "Rx", "Ry", "Rz", "Fz"], "node CSV")
    _check_columns(df_edge, ["Sample_ID", "Edge_ID", "Source", "Target", "Area", "Length", "E", "Iy", "Iz", "J", "EA/L", "Axial_Force"], "edge CSV")
    _check_columns(df_global, ["sample_id", "Global_Load_Sum", "Total_Structural_Volume", "Average_Connectivity"], "global CSV")

    return df_node, df_edge, df_global


def infer_v4_schema(df_node: pd.DataFrame, df_edge: pd.DataFrame, df_global: pd.DataFrame) -> V4Schema:
    node_continuous_cols = ("x", "y", "z", "Fz")
    node_mask_cols = ("Tx", "Ty", "Tz", "Rx", "Ry", "Rz")
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
) -> list[Data]:
    sample_ids = validate_sample_coverage(df_node, df_edge, df_global)

    dataset: list[Data] = []

    for sample_id in sample_ids:
        node_sample = df_node[df_node["sample_id"] == sample_id].copy()
        edge_sample = df_edge[df_edge["Sample_ID"] == sample_id].copy()
        global_sample = df_global[df_global["sample_id"] == sample_id].copy()

        node_sample = node_sample.sort_values(by="node_id", key=lambda series: series.map(_numeric_suffix))
        edge_sample = edge_sample.sort_values(by="Edge_ID", key=lambda series: series.map(_numeric_suffix))

        x_continuous = torch.tensor(node_continuous_scaled.loc[node_sample.index].to_numpy(), dtype=torch.float32)
        x_mask = torch.tensor(node_mask_values.loc[node_sample.index].to_numpy(), dtype=torch.float32)
        x = torch.cat([x_continuous, x_mask], dim=1)

        edge_attr = torch.tensor(edge_feature_scaled.loc[edge_sample.index].to_numpy(), dtype=torch.float32)
        y_edge = torch.tensor(edge_target_scaled.loc[edge_sample.index].to_numpy(), dtype=torch.float32)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        y_edge = torch.cat([y_edge, y_edge], dim=0)
        u = torch.tensor(global_feature_scaled.loc[global_sample.index].to_numpy().reshape(1, -1), dtype=torch.float32)

        dataset.append(
            Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y_edge=y_edge,
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
