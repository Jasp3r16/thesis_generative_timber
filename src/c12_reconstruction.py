import pandas as pd
import math
import json
import random
from pathlib import Path
from typing import Any, Mapping
import numpy as np

from c12_geometry_truss import _build_truss_edges

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

def reconstruct_edges(cells_x, cells_y):
    """
    Compute the grid topology (edges) without nested if statements.
    Faster and shorter thanks to list comprehensions.
    """
    edges = _build_truss_edges(cells_x, cells_y)
    df_edges = pd.DataFrame(edges, columns=["V1", "V2"])
    df_edges.insert(0, "edge_id", ["e" + str(i) for i in range(len(df_edges))])
    
    return df_edges

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


def calculate_geometry_beam_statistics(df_vertices, edge_index, sample_id):
    """
    Calculate comprehensive beam length statistics for a single structure (sample_id).

    Returns:
        dict with average, min, max, total length and additional distribution statistics
        in both meters and millimeters.
    """
    required_cols = {'sample_id', 'vertex_index', 'x', 'y', 'z'}
    missing_cols = required_cols - set(df_vertices.columns)
    if missing_cols:
        raise ValueError(f"df_vertices is missing required columns: {sorted(missing_cols)}")

    _validate_edge_index(edge_index)

    df_sample = df_vertices[df_vertices['sample_id'] == sample_id].copy()
    if df_sample.empty:
        raise ValueError(f"No vertices found for sample_id={sample_id}")

    # Normalize vertex ids in the sample for robust joins
    df_sample['vertex_index_norm'] = df_sample['vertex_index'].astype(str).apply(_normalize_vertex_id)

    sources, targets = edge_index
    edges_df = pd.DataFrame({
        'V1': [_normalize_vertex_id(v) for v in sources],
        'V2': [_normalize_vertex_id(v) for v in targets]
    })

    # Merge coordinates for both ends in a vectorized manner
    left = edges_df.merge(
        df_sample[['vertex_index_norm', 'x', 'y', 'z']],
        left_on='V1',
        right_on='vertex_index_norm',
        how='left'
    ).rename(columns={'x': 'x1', 'y': 'y1', 'z': 'z1'}).drop(columns=['vertex_index_norm'])

    both = left.merge(
        df_sample[['vertex_index_norm', 'x', 'y', 'z']],
        left_on='V2',
        right_on='vertex_index_norm',
        how='left'
    ).rename(columns={'x': 'x2', 'y': 'y2', 'z': 'z2'}).drop(columns=['vertex_index_norm'])

    # Identify missing vertices (if any)
    missing_mask = both[['x1', 'y1', 'z1', 'x2', 'y2', 'z2']].isnull().any(axis=1)
    if missing_mask.any():
        missing_rows = both[missing_mask]
        missing_v1 = missing_rows.loc[missing_rows[['x1', 'y1', 'z1']].isnull().any(axis=1), 'V1'].unique().tolist()
        missing_v2 = missing_rows.loc[missing_rows[['x2', 'y2', 'z2']].isnull().any(axis=1), 'V2'].unique().tolist()
        raise KeyError(f"Edge refers to missing vertex index(es): {missing_v1 + missing_v2}")

    # Vectorized distance computation
    dx = both['x1'].to_numpy(dtype=float) - both['x2'].to_numpy(dtype=float)
    dy = both['y1'].to_numpy(dtype=float) - both['y2'].to_numpy(dtype=float)
    dz = both['z1'].to_numpy(dtype=float) - both['z2'].to_numpy(dtype=float)
    lengths_m = np.sqrt(dx * dx + dy * dy + dz * dz)
    lengths_mm = np.round(lengths_m * 1000.0, 1)

    edge_count = lengths_m.size
    avg_m = float(np.mean(lengths_m)) if edge_count > 0 else 0.0
    min_m = float(np.min(lengths_m)) if edge_count > 0 else 0.0
    max_m = float(np.max(lengths_m)) if edge_count > 0 else 0.0
    total_m = float(np.sum(lengths_m)) if edge_count > 0 else 0.0
    std_m = float(np.std(lengths_m)) if edge_count > 1 else 0.0
    median_m = float(np.median(lengths_m)) if edge_count > 0 else 0.0
    q1_m = float(np.percentile(lengths_m, 25)) if edge_count > 0 else 0.0
    q3_m = float(np.percentile(lengths_m, 75)) if edge_count > 0 else 0.0

    return {
        'sample_id': sample_id,
        'edge_count': int(edge_count),
        'average_length_m': round(avg_m, 3),
        'average_length_mm': round(avg_m * 1000.0, 1),
        'median_length_m': round(median_m, 3),
        'median_length_mm': round(median_m * 1000.0, 1),
        'min_length_m': round(min_m, 3),
        'min_length_mm': round(min_m * 1000.0, 1),
        'max_length_m': round(max_m, 3),
        'max_length_mm': round(max_m * 1000.0, 1),
        'total_length_m': round(total_m, 3),
        'total_length_mm': round(total_m * 1000.0, 1),
        'std_dev_m': round(std_m, 3),
        'std_dev_mm': round(std_m * 1000.0, 1),
        'q1_m': round(q1_m, 3),
        'q1_mm': round(q1_m * 1000.0, 1),
        'q3_m': round(q3_m, 3),
        'q3_mm': round(q3_m * 1000.0, 1),
        'lengths_mm': lengths_mm.tolist()
    }


def generate_material_statistics(df_vertices, edge_index, sample_count, random_state=None):
    """
    Generate comprehensive material statistics from randomly selected structure samples.

    The function selects `sample_count` unique sample_ids from the dataset and takes
    the median of the per-sample average beam lengths to dampen outliers.

    Args:
        df_vertices: DataFrame met o.a. sample_id, vertex_index, x, y, z.
        edge_index: Topologie in formaat [[sources...], [targets...]].
        sample_count: Number of random samples to use.
        random_state: Optional seed for reproducible sample selection.

    Returns:
        dict with representative length in m and mm, plus detailed per-sample statistics.
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
    pooled_lengths_mm = []

    for sample_id in selected_sample_ids:
        result = calculate_geometry_beam_statistics(df_vertices, edge_index, sample_id)
        per_sample_results.append(result)
        per_sample_means_m.append(result['average_length_m'])
        pooled_lengths_mm.extend(result.get('lengths_mm', []))

    sorted_means = sorted(per_sample_means_m)
    n = len(sorted_means)
    if n % 2 == 1:
        median_m = sorted_means[n // 2]
    else:
        median_m = (sorted_means[n // 2 - 1] + sorted_means[n // 2]) / 2.0

    summary_statistics = {
        'average_length_m': round(float(np.mean([r['average_length_m'] for r in per_sample_results])), 3),
        'average_length_mm': round(float(np.mean([r['average_length_mm'] for r in per_sample_results])), 1),
        'median_length_m': round(float(np.mean([r['median_length_m'] for r in per_sample_results])), 3),
        'median_length_mm': round(float(np.mean([r['median_length_mm'] for r in per_sample_results])), 1),
        'min_length_m': round(float(np.mean([r['min_length_m'] for r in per_sample_results])), 3),
        'min_length_mm': round(float(np.mean([r['min_length_mm'] for r in per_sample_results])), 1),
        'max_length_m': round(float(np.mean([r['max_length_m'] for r in per_sample_results])), 3),
        'max_length_mm': round(float(np.mean([r['max_length_mm'] for r in per_sample_results])), 1),
        'total_length_m': round(float(np.mean([r['total_length_m'] for r in per_sample_results])), 3),
        'total_length_mm': round(float(np.mean([r['total_length_mm'] for r in per_sample_results])), 1),
        'std_dev_m': round(float(np.mean([r['std_dev_m'] for r in per_sample_results])), 3),
        'std_dev_mm': round(float(np.mean([r['std_dev_mm'] for r in per_sample_results])), 1),
        'q1_m': round(float(np.mean([r['q1_m'] for r in per_sample_results])), 3),
        'q1_mm': round(float(np.mean([r['q1_mm'] for r in per_sample_results])), 1),
        'q3_m': round(float(np.mean([r['q3_m'] for r in per_sample_results])), 3),
        'q3_mm': round(float(np.mean([r['q3_mm'] for r in per_sample_results])), 1),
        'edge_count': int(round(float(np.mean([r['edge_count'] for r in per_sample_results])))),
    }

    pooled_array = np.array(pooled_lengths_mm, dtype=float)
    if pooled_array.size > 0:
        percentiles = np.percentile(pooled_array, [1, 5, 10, 25, 50, 75, 95, 99])
        pooled_length_percentiles_mm = {
            'p1_mm': round(float(percentiles[0]), 1),
            'p5_mm': round(float(percentiles[1]), 1),
            'p10_mm': round(float(percentiles[2]), 1),
            'p25_mm': round(float(percentiles[3]), 1),
            'p50_mm': round(float(percentiles[4]), 1),
            'p75_mm': round(float(percentiles[5]), 1),
            'p95_mm': round(float(percentiles[6]), 1),
            'p99_mm': round(float(percentiles[7]), 1)
        }
    else:
        pooled_length_percentiles_mm = {
            'p1_mm': 0.0,
            'p5_mm': 0.0,
            'p10_mm': 0.0,
            'p25_mm': 0.0,
            'p50_mm': 0.0,
            'p75_mm': 0.0,
            'p95_mm': 0.0,
            'p99_mm': 0.0
        }

    return {
        'aggregated_metric': 'median_of_sample_means',
        'sample_count': sample_count,
        'selected_sample_ids': selected_sample_ids,
        'summary_statistics': summary_statistics,
        'pooled_length_percentiles_mm': pooled_length_percentiles_mm,
        'p1_mm': pooled_length_percentiles_mm['p1_mm'],
        'p5_mm': pooled_length_percentiles_mm['p5_mm'],
        'p10_mm': pooled_length_percentiles_mm['p10_mm'],
        'p25_mm': pooled_length_percentiles_mm['p25_mm'],
        'p50_mm': pooled_length_percentiles_mm['p50_mm'],
        'p75_mm': pooled_length_percentiles_mm['p75_mm'],
        'p95_mm': pooled_length_percentiles_mm['p95_mm'],
        'p99_mm': pooled_length_percentiles_mm['p99_mm'],
        'representative_length_m': median_m,
        'representative_length_mm': median_m * 1000.0,
        'pooled_lengths_mm': pooled_lengths_mm,
        'per_sample_results': per_sample_results,
    }