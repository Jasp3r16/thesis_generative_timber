import pandas as pd
import math
import json
import random
from pathlib import Path

from geometry import _build_truss_edges

def reconstruct_edges(cells_x, cells_y):
    """
    Compute the grid topology (edges) without nested if statements.
    Faster and shorter thanks to list comprehensions.
    """
    edges = _build_truss_edges(cells_x, cells_y)

    df_edges = pd.DataFrame(edges, columns=["V1", "V2"])
    
    # Add edge_id as prefix.
    df_edges.insert(0, "edge_id", ["e" + str(i) for i in range(len(df_edges))])
    
    return df_edges


def _normalize_vertex_id(vertex_id):
    """Normalize a vertex index to the format 'v{n}'."""
    return vertex_id if str(vertex_id).startswith('v') else f"v{vertex_id}"


def _validate_edge_index(edge_index):
    """Validate edge_index in the format [[sources...], [targets...]]."""
    if not isinstance(edge_index, list) or len(edge_index) != 2:
        raise ValueError("edge_index must be a list with two lists: [sources, targets]")

    sources, targets = edge_index
    if not isinstance(sources, list) or not isinstance(targets, list):
        raise ValueError("edge_index sources and targets must both be lists")
    if len(sources) != len(targets):
        raise ValueError("edge_index sources and targets must have the same length")
    if len(sources) == 0:
        raise ValueError("edge_index cannot be empty")


def calculate_average_beam_length_for_sample(df_vertices, edge_index, sample_id):
    """
    Calculate the average beam length for a single structure (sample_id).

    Returns:
        dict with average length in meters and millimeters.
    """
    required_cols = {'sample_id', 'vertex_index', 'x', 'y', 'z'}
    missing_cols = required_cols - set(df_vertices.columns)
    if missing_cols:
        raise ValueError(f"df_vertices is missing required columns: {sorted(missing_cols)}")

    _validate_edge_index(edge_index)

    df_sample = df_vertices[df_vertices['sample_id'] == sample_id]
    if df_sample.empty:
        raise ValueError(f"No vertices found for sample_id={sample_id}")

    v_dict = df_sample.set_index('vertex_index').to_dict('index')
    sources, targets = edge_index

    lengths_m = []
    for v1_raw, v2_raw in zip(sources, targets):
        v1_id = _normalize_vertex_id(v1_raw)
        v2_id = _normalize_vertex_id(v2_raw)

        if v1_id not in v_dict or v2_id not in v_dict:
            raise KeyError(
                f"Edge refers to missing vertex index: {v1_id} or {v2_id}"
            )

        pt1 = v_dict[v1_id]
        pt2 = v_dict[v2_id]

        dx = pt1['x'] - pt2['x']
        dy = pt1['y'] - pt2['y']
        dz = pt1['z'] - pt2['z']
        lengte_m = math.sqrt(dx**2 + dy**2 + dz**2)
        lengths_m.append(lengte_m)

    avg_m = sum(lengths_m) / len(lengths_m)
    return {
        'sample_id': sample_id,
        'average_length_m': avg_m,
        'average_length_mm': avg_m * 1000.0,
        'edge_count': len(lengths_m)
    }


def calculate_representative_beam_length(df_vertices, edge_index, sample_count, random_state=None):
    """
    Calculate a robust representative beam length from randomly selected samples.

    The function selects `sample_count` unique sample_ids from the dataset and takes
    the median of the per-sample average beam lengths to dampen outliers.

    Args:
        df_vertices: DataFrame met o.a. sample_id, vertex_index, x, y, z.
        edge_index: Topologie in formaat [[sources...], [targets...]].
        sample_count: Number of random samples to use.
        random_state: Optional seed for reproducible sample selection.

    Returns:
        dict with representative length in m and mm, plus traceable details.
    """
    if not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")

    available_sample_ids = sorted(df_vertices['sample_id'].unique().tolist())
    if sample_count > len(available_sample_ids):
        raise ValueError(
            f"sample_count={sample_count} is greater than the available unique samples="
            f"{len(available_sample_ids)}"
        )

    rng = random.Random(random_state)
    selected_sample_ids = rng.sample(available_sample_ids, sample_count)

    per_sample_means_m = []
    per_sample_results = []

    for sample_id in selected_sample_ids:
        result = calculate_average_beam_length_for_sample(df_vertices, edge_index, sample_id)
        per_sample_results.append(result)
        per_sample_means_m.append(result['average_length_m'])

    sorted_means = sorted(per_sample_means_m)
    n = len(sorted_means)
    if n % 2 == 1:
        median_m = sorted_means[n // 2]
    else:
        median_m = (sorted_means[n // 2 - 1] + sorted_means[n // 2]) / 2.0

    return {
        'aggregated_metric': 'median_of_sample_means',
        'sample_count': sample_count,
        'selected_sample_ids': selected_sample_ids,
        'per_sample_results': per_sample_results,
        'representative_length_m': median_m,
        'representative_length_mm': median_m * 1000.0
    }


def calculate_representative_beam_length_from_files(
    vertices_csv_path,
    edge_index_json_path,
    sample_count,
    random_state=None
):
    """
    Convenience wrapper that reads vertices CSV and edge_index JSON.
    """
    vertices_path = Path(vertices_csv_path)
    edge_path = Path(edge_index_json_path)

    df_vertices = pd.read_csv(vertices_path)
    with edge_path.open('r', encoding='utf-8') as f:
        edge_index = json.load(f)

    return calculate_representative_beam_length(
        df_vertices=df_vertices,
        edge_index=edge_index,
        sample_count=sample_count,
        random_state=random_state
    )