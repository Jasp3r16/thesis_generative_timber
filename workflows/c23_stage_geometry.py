from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

import c11_params
from c12_geometry_truss import generate_sample_vertices
from c24_reconstruction import reconstruct_edges


def sample_random_design(search_space: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Sample one random design from a search-space mapping."""
    random_params: dict[str, Any] = {}
    for var_name, rules in search_space.items():
        if rules["type"] == "continuous":
            random_params[var_name] = random.uniform(float(rules["min"]), float(rules["max"]))
        elif rules["type"] == "discrete":
            random_params[var_name] = random.choice(list(rules["options"]))
        else:
            raise ValueError(f"Unsupported search-space type for {var_name}: {rules['type']}")
    return random_params


def _to_vertex_key(v: Any) -> str:
    v_str = str(v)
    return v_str if v_str.startswith("v") else f"v{v_str}"


def _edge_length_m(vertex_lookup: dict[str, dict[str, float]], v1: Any, v2: Any) -> float:
    p1 = vertex_lookup[_to_vertex_key(v1)]
    p2 = vertex_lookup[_to_vertex_key(v2)]
    return float(
        np.linalg.norm(
            [
                p2["x"] - p1["x"],
                p2["y"] - p1["y"],
                p2["z"] - p1["z"],
            ]
        )
    )


def build_geometry_overview(df_vertices: pd.DataFrame, df_edges: pd.DataFrame) -> pd.DataFrame:
    """Build compact edge table with computed member lengths in meters."""
    vertex_lookup = df_vertices.set_index("vertex_index")[["x", "y", "z"]].to_dict("index")
    df_geometry_overview = df_edges.copy()
    df_geometry_overview["length_m"] = df_geometry_overview.apply(
        lambda r: round(_edge_length_m(vertex_lookup, r["V1"], r["V2"]), 3),
        axis=1,
    )
    return df_geometry_overview


def run_random_geometry_stage(
    json_path: Path | None = None,
    optimizer_search_space: Mapping[str, Mapping[str, Any]] | None = None,
    sample_id: int = 0,
) -> dict[str, Any]:
    """Generate random design, vertices/edges, and geometry overview in one call."""
    if optimizer_search_space is None:
        if json_path is None:
            raise ValueError("Provide optimizer_search_space or json_path")
        with open(json_path, "r", encoding="utf-8") as f:
            optimizer_search_space = json.load(f)

    my_random_design = sample_random_design(optimizer_search_space)
    vertices_list = generate_sample_vertices(sample_id=sample_id, params=my_random_design)
    df_vertices = pd.DataFrame(vertices_list)
    df_edges = reconstruct_edges(c11_params.GRID_CELLS_X, c11_params.GRID_CELLS_Y)
    df_geometry_overview = build_geometry_overview(df_vertices=df_vertices, df_edges=df_edges)

    return {
        "my_random_design": my_random_design,
        "vertices_list": vertices_list,
        "df_vertices": df_vertices,
        "df_edges": df_edges,
        "df_geometry_overview": df_geometry_overview,
    }


def run_geometry_from_design(
    design_params: Mapping[str, Any],
    sample_id: int = 0,
) -> dict[str, Any]:
    """Generate geometry tables from an explicit design parameter dictionary."""
    vertices_list = generate_sample_vertices(sample_id=sample_id, params=dict(design_params))
    df_vertices = pd.DataFrame(vertices_list)
    df_edges = reconstruct_edges(c11_params.GRID_CELLS_X, c11_params.GRID_CELLS_Y)
    df_geometry_overview = build_geometry_overview(df_vertices=df_vertices, df_edges=df_edges)

    return {
        "design_params": dict(design_params),
        "vertices_list": vertices_list,
        "df_vertices": df_vertices,
        "df_edges": df_edges,
        "df_geometry_overview": df_geometry_overview,
    }


def plot_geometry_preview(
    df_vertices: pd.DataFrame,
    df_edges: pd.DataFrame,
    figsize: tuple[float, float] = (8, 7),
):
    """Plot a 3D geometry preview for the generated truss."""
    vertex_lookup = df_vertices.set_index("vertex_index")[["x", "y", "z", "layer"]].to_dict("index")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    for _, edge in df_edges.iterrows():
        v1 = _to_vertex_key(edge["V1"])
        v2 = _to_vertex_key(edge["V2"])

        if v1 not in vertex_lookup or v2 not in vertex_lookup:
            continue

        p1 = vertex_lookup[v1]
        p2 = vertex_lookup[v2]
        ax.plot3D(
            [p1["x"], p2["x"]],
            [p1["y"], p2["y"]],
            [p1["z"], p2["z"]],
            color="0.45",
            linewidth=0.8,
            alpha=0.75,
        )

    df_top = df_vertices[df_vertices["layer"] == "top"]
    df_bottom = df_vertices[df_vertices["layer"] == "bottom"]

    ax.scatter3D(  # type: ignore
        np.asarray(df_top["x"]),
        np.asarray(df_top["y"]),
        np.asarray(df_top["z"]),
        s=35,
        c="#1f77b4",
        label="top",
    )
    ax.scatter3D(  # type: ignore
        np.asarray(df_bottom["x"]),
        np.asarray(df_bottom["y"]),
        np.asarray(df_bottom["z"]),
        s=35,
        c="#d62728",
        label="bottom",
    )

    ax.set_title("Random Design - 3D Geometry Preview")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]", labelpad=14)
    ax.legend(loc="upper right")
    ax.set_box_aspect((1, 1, 0.45))

    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.08, top=0.93)
    return fig, ax
