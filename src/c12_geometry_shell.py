from __future__ import annotations

import random
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd


DEFAULT_GRID_RADIUS = 3
DEFAULT_ROOF_RADIUS = 4.0
DEFAULT_ROOF_RISE = 3.0
DEFAULT_DOME_POWER = 1.8
DEFAULT_LATTICE_SCALE = 1.0
DEFAULT_CROWN_BULGE = 0.15
DEFAULT_RIM_DROP = 0.08


def define_search_space(grid_radius: int = DEFAULT_GRID_RADIUS) -> dict[str, dict[str, Any]]:
    """Define a compact optimization space for a one-layer dome shell."""
    return {
        "roof_radius": {"type": "continuous", "min": 4.5, "max": 8.5},
        "roof_rise": {"type": "continuous", "min": 1.8, "max": 4.8},
        "dome_power": {"type": "continuous", "min": 1.2, "max": 3.0},
        "lattice_scale": {"type": "continuous", "min": 0.92, "max": 1.08},
        "crown_bulge": {"type": "continuous", "min": 0.0, "max": 0.35},
        "rim_drop": {"type": "continuous", "min": 0.0, "max": 0.20},
        "grid_radius": {"type": "discrete", "options": [grid_radius]},
    }


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


def _normalize_vertices(vertices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    coords = np.array([[v["x"], v["y"], v["z"]] for v in vertices], dtype=np.float64)
    if coords.size == 0:
        return vertices

    centered = coords - coords.mean(axis=0, keepdims=True)
    normalized_vertices = []
    for vertex, (x, y, z) in zip(vertices, centered):
        updated_vertex = dict(vertex)
        updated_vertex["x"] = round(float(x), 3)
        updated_vertex["y"] = round(float(y), 3)
        updated_vertex["z"] = round(float(z), 3)
        normalized_vertices.append(updated_vertex)

    return normalized_vertices


def _resolve_design(
    params: Optional[Mapping[str, float]] = None,
    grid_radius: int = DEFAULT_GRID_RADIUS,
) -> dict[str, float]:
    if params is None:
        sampled = sample_random_design(define_search_space(grid_radius=grid_radius))
    else:
        sampled = dict(params)

    return {
        "roof_radius": float(sampled.get("roof_radius", DEFAULT_ROOF_RADIUS)),
        "roof_rise": float(sampled.get("roof_rise", DEFAULT_ROOF_RISE)),
        "dome_power": float(sampled.get("dome_power", DEFAULT_DOME_POWER)),
        "lattice_scale": float(sampled.get("lattice_scale", DEFAULT_LATTICE_SCALE)),
        "crown_bulge": float(sampled.get("crown_bulge", DEFAULT_CROWN_BULGE)),
        "rim_drop": float(sampled.get("rim_drop", DEFAULT_RIM_DROP)),
        "grid_radius": int(sampled.get("grid_radius", grid_radius)),
    }


def _axial_to_cartesian(q: int, r: int, scale: float) -> tuple[float, float]:
    x = scale * np.sqrt(3.0) * (q + r / 2.0)
    y = scale * 1.5 * r
    return float(x), float(y)


def _hex_distance(q: int, r: int) -> int:
    s = -q - r
    return int((abs(q) + abs(r) + abs(s)) / 2)


def _neighbor_coords(q: int, r: int) -> list[tuple[int, int]]:
    return [
        (q + 1, r),
        (q + 1, r - 1),
        (q, r - 1),
        (q - 1, r),
        (q - 1, r + 1),
        (q, r + 1),
    ]


def _generate_axial_nodes(grid_radius: int) -> list[tuple[int, int, int]]:
    nodes: list[tuple[int, int, int]] = []
    for q in range(-grid_radius, grid_radius + 1):
        for r in range(-grid_radius, grid_radius + 1):
            if _hex_distance(q, r) <= grid_radius:
                nodes.append((q, r, _hex_distance(q, r)))
    return nodes


def generate_sample_vertices(
    sample_id: int,
    params: Optional[Mapping[str, float]] = None,
    grid_radius: int = DEFAULT_GRID_RADIUS,
):
    """Generate a single-layer dome shell from a hex-dominant grid."""
    design = _resolve_design(params=params, grid_radius=grid_radius)
    radius = design["roof_radius"]
    roof_rise = design["roof_rise"]
    dome_power = design["dome_power"]
    lattice_scale = design["lattice_scale"]
    crown_bulge = design["crown_bulge"]
    rim_drop = design["rim_drop"]
    grid_radius = int(design["grid_radius"])

    nodes = _generate_axial_nodes(grid_radius)
    if not nodes:
        return []

    plane_scale = (radius / max(1.0, float(grid_radius))) * lattice_scale
    all_vertices: list[dict[str, Any]] = []

    for vertex_index, (q, r, distance) in enumerate(nodes):
        x, y = _axial_to_cartesian(q, r, plane_scale)
        radial_fraction = min(1.0, distance / max(1, grid_radius))
        crown_profile = max(0.0, 1.0 - radial_fraction**2)
        base_profile = max(0.0, 1.0 - radial_fraction**dome_power)
        z = roof_rise * max(0.0, base_profile + (crown_bulge * crown_profile) - (rim_drop * radial_fraction))
        attribute = "boundary" if distance == grid_radius else ("apex" if distance == 0 else "field")

        all_vertices.append(
            {
                "sample_id": sample_id,
                "vertex_index": f"v{vertex_index}",
                "layer": "shell",
                "attribute": attribute,
                "x": round(float(x), 3),
                "y": round(float(y), 3),
                "z": round(float(z), 3),
            }
        )

    return _normalize_vertices(all_vertices)


def generate_edges(num_samples: int, grid_radius: int = DEFAULT_GRID_RADIUS):
    """Generate a shared edge list for the one-layer shell."""
    nodes = _generate_axial_nodes(grid_radius)
    node_lookup = {(q, r): idx for idx, (q, r, _distance) in enumerate(nodes)}
    sample_edges: list[tuple[int, int]] = []

    for q, r, _distance in nodes:
        node_index = node_lookup[(q, r)]
        for nq, nr in _neighbor_coords(q, r):
            neighbor_index = node_lookup.get((nq, nr))
            if neighbor_index is None or neighbor_index <= node_index:
                continue
            sample_edges.append((node_index, neighbor_index))

    edges_data = []
    for sample_id in range(num_samples):
        for edge_counter, (u, v) in enumerate(sample_edges):
            edges_data.append(
                {
                    "sample_id": sample_id,
                    "edge_id": f"e{edge_counter}",
                    "V1": u,
                    "V2": v,
                }
            )

    return pd.DataFrame(edges_data)


def generate_vertices(num_samples: int, round_decimals: int = 2, grid_radius: int = DEFAULT_GRID_RADIUS):
    """Generate a dataset of dome-shell samples."""
    all_data = []
    seen_signatures = set()
    samples_generated = 0
    attempts = 0
    max_attempts = num_samples * 10

    while samples_generated < num_samples and attempts < max_attempts:
        vertices = generate_sample_vertices(samples_generated, params=None, grid_radius=grid_radius)
        signature = tuple(
            (round(v["x"], round_decimals), round(v["y"], round_decimals), round(v["z"], round_decimals))
            for v in vertices
        )

        if signature not in seen_signatures:
            seen_signatures.add(signature)
            all_data.extend(vertices)
            samples_generated += 1

        attempts += 1

    if attempts >= max_attempts:
        print(
            f"Warning: generation stopped early to prevent an infinite loop. "
            f"The design space may be too limited. Total generated: {samples_generated}"
        )

    return pd.DataFrame(all_data)