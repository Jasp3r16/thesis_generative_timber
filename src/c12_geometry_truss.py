import pandas as pd
import random
import warnings
from typing import Mapping, Optional, Sequence

import numpy as np

import c11_params


def _build_truss_edges(cells_x, cells_y):
    nodes_x_top = cells_x + 1
    nodes_y_top = cells_y + 1
    num_top_vertices = nodes_x_top * nodes_y_top

    edges = []

    for r in range(nodes_y_top):
        for c in range(nodes_x_top):
            current = r * nodes_x_top + c

            if c < cells_x:
                edges.append((current, current + 1))

            if r < cells_y:
                edges.append((current, current + nodes_x_top))

    start_idx_bottom = num_top_vertices

    for r in range(cells_y):
        for c in range(cells_x):
            current = start_idx_bottom + r * cells_x + c

            if c < cells_x - 1:
                edges.append((current, current + 1))

            if r < cells_y - 1:
                edges.append((current, current + cells_x))

    for r in range(cells_y):
        for c in range(cells_x):
            bottom_node = start_idx_bottom + r * cells_x + c
            top_tl = r * nodes_x_top + c
            top_tr = r * nodes_x_top + (c + 1)
            top_bl = (r + 1) * nodes_x_top + c
            top_br = (r + 1) * nodes_x_top + (c + 1)

            edges.extend([
                (bottom_node, top_tl),
                (bottom_node, top_tr),
                (bottom_node, top_bl),
                (bottom_node, top_br),
            ])

    return edges

def get_valid_shifts(divisions, edge_length):
    """Calculate the allowed shifts (excluding the extremes)."""
    half_div = divisions // 2
    all_steps = list(range(-half_div, half_div + 1))
    valid_steps = all_steps[1:-1] # Verwijder eerste en laatste
    valid_shifts = [(step / divisions) * edge_length for step in valid_steps]
    return valid_shifts

def get_corner_indices(cells_x, cells_y):
    """
    Calculate the corner indices for the top layer.
    Works for any grid size (n x m).
    """
    # The number of points is always the number of cells + 1.
    nodes_x = cells_x + 1
    nodes_y = cells_y + 1
    total_nodes = nodes_x * nodes_y

    indices = {
        "bottom_left": 0,
        "bottom_right": nodes_x - 1,
        "top_left": (nodes_y - 1) * nodes_x,
        "top_right": total_nodes - 1
    }

    return indices


def define_search_space(cells_x, cells_y, divisions, edge_length):
    """
    Translate the geometric constraints into a machine-readable search space
    for a machine learning or optimization algorithm.

    Note:
    The geometry reconstruction in `generate_sample_vertices` currently uses
    grid and size values from `c11_params`. If you pass different values here,
    the generated search space may not match the geometry one-to-one.
    """
    if (
        cells_x != c11_params.GRID_CELLS_X
        or cells_y != c11_params.GRID_CELLS_Y
        or edge_length != c11_params.EDGE_LENGTH
    ):
        warnings.warn(
            "define_search_space() called with values that differ from c11_params. "
            "generate_sample_vertices currently uses c11_params directly, "
            "so this can cause a mismatch between search-space keys and geometry.",
            stacklevel=2,
        )

    valid_shifts = get_valid_shifts(divisions, edge_length)
    search_space = {}

    num_nodes_x_top = cells_x + 1
    num_nodes_y_top = cells_y + 1
    vertex_idx = 0

    corners = get_corner_indices(cells_x, cells_y).values()

    for i in range(num_nodes_y_top):
        for j in range(num_nodes_x_top):
            v_name = f"v{vertex_idx}"

            is_x_edge = (j == 0) or (j == num_nodes_x_top - 1)
            is_y_edge = (i == 0) or (i == num_nodes_y_top - 1)
            is_corner = vertex_idx in corners

            if is_corner:
                pass
            elif is_x_edge:
                search_space[f"{v_name}_shift_y"] = {"type": "discrete", "options": valid_shifts}
            elif is_y_edge:
                search_space[f"{v_name}_shift_x"] = {"type": "discrete", "options": valid_shifts}
            else:
                search_space[f"{v_name}_shift_x"] = {"type": "discrete", "options": valid_shifts}
                search_space[f"{v_name}_shift_y"] = {"type": "discrete", "options": valid_shifts}

            vertex_idx += 1

    for _r in range(cells_y):
        for _c in range(cells_x):
            v_name = f"v{vertex_idx}"

            search_space[f"{v_name}_u"] = {
                "type": "continuous",
                "min": c11_params.SCALE_UV[0],
                "max": c11_params.SCALE_UV[1],
            }
            search_space[f"{v_name}_v"] = {
                "type": "continuous",
                "min": c11_params.SCALE_UV[0],
                "max": c11_params.SCALE_UV[1],
            }

            search_space[f"{v_name}_shift_z"] = {"type": "discrete", "options": valid_shifts}

            vertex_idx += 1

    return search_space

def bilinear_interpolate(p00, p10, p01, p11, u, v):
    """
    Interpolate a point inside a quadrilateral.
    p00: Bottom-Left, p10: Bottom-Right, p01: Top-Left, p11: Top-Right (in standard Cartesian coordinates).
    In matrix indexing (row i, column j):
    (i, j) is often Top-Left in images, but Bottom-Left in Grasshopper/Cartesian coordinates if y increases upward.
    Assume: i=0 means y=0 (bottom), i=max means y=max (top).
    Then row i is below row i+1.
    p_bl = (i, j), p_br = (i, j+1)
    p_tl = (i+1, j), p_tr = (i+1, j+1)
    """
    # X interpolation
    # Bottom (row i)
    x_bot = p00['x'] * (1 - u) + p10['x'] * u
    # Top (row i+1)
    x_top = p01['x'] * (1 - u) + p11['x'] * u

    final_x = x_bot * (1 - v) + x_top * v

    # Y interpolation
    y_bot = p00['y'] * (1 - u) + p10['y'] * u
    y_top = p01['y'] * (1 - u) + p11['y'] * u

    final_y = y_bot * (1 - v) + y_top * v

    return final_x, final_y


def _normalize_vertices_pca(vertices):
    """Center a sample at its centroid and align its principal axes to XYZ."""
    coords = np.array([[v["x"], v["y"], v["z"]] for v in vertices], dtype=np.float64)
    if coords.size == 0:
        return vertices

    centered = coords - coords.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2 or np.allclose(centered, 0.0):
        normalized = centered
    else:
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        basis = eigvecs[:, order]

        # Keep the basis right-handed and deterministic.
        if np.linalg.det(basis) < 0:
            basis[:, -1] *= -1.0

        for col in range(basis.shape[1]):
            dominant_axis = int(np.argmax(np.abs(basis[:, col])))
            if basis[dominant_axis, col] < 0:
                basis[:, col] *= -1.0

        normalized = centered @ basis

    normalized_vertices = []
    for vertex, (x, y, z) in zip(vertices, normalized):
        updated_vertex = dict(vertex)
        updated_vertex["x"] = round(float(x), 3)
        updated_vertex["y"] = round(float(y), 3)
        updated_vertex["z"] = round(float(z), 3)
        normalized_vertices.append(updated_vertex)

    return normalized_vertices

def generate_edges(num_samples, cells_x, cells_y):
    """
    Generate a topological edge list for a double-layer spatial truss.

    The function builds a grid consisting of three parts:
    1. A top layer grid of (cells_x + 1) by (cells_y + 1) points.
    2. A bottom layer grid of (cells_x) by (cells_y) points, centered below the top cells.
    3. Diagonal connections (pyramid structure) between the bottom points and the four above top points.

    Args:
        num_samples (int): Number of unique samples to generate.
        cells_x (int): Number of cells in the X direction.
        cells_y (int): Number of cells in the Y direction.

    Returns:
        pd.DataFrame: A DataFrame with the columns ['sample_id', 'edge_id', 'V1', 'V2'].
            V1 and V2 are the indices of the connected vertices.
    """
    edges_data = []

    for sample_id in range(num_samples):
        sample_edges = _build_truss_edges(cells_x, cells_y)
        for edge_counter, (u, v) in enumerate(sample_edges):
            edges_data.append({
                "sample_id": sample_id,
                "edge_id": f"e{edge_counter}",
                "V1": u,
                "V2": v,
            })

    return pd.DataFrame(edges_data)

def generate_sample_vertices(
    sample_id: int,
    params: Optional[Mapping[str, float]] = None,
    valid_shifts: Optional[Sequence[float]] = None,
):
    """
    Generate the coordinates for a single spatial truss.
    
    Mode 1 (dataset generation): if 'params' is None, random shifts are applied
    based on valid_shifts.
    Mode 2 (reconstruction): if 'params' is a dictionary, the specific optimum
    values are loaded.
    """
    if params is None:
        if not valid_shifts:
            raise ValueError("valid_shifts must be provided and non-empty when params is None")
        shift_options = valid_shifts
    else:
        shift_options = []

    all_vertices = []
    num_nodes_x_top = c11_params.GRID_CELLS_X + 1
    num_nodes_y_top = c11_params.GRID_CELLS_Y + 1

    corners = get_corner_indices(c11_params.GRID_CELLS_X, c11_params.GRID_CELLS_Y).values()
    top_layer_coords = {}
    vertex_idx = 0

    # --- STEP 1: TOP LAYER ---
    for i in range(num_nodes_y_top):
        for j in range(num_nodes_x_top):
            base_x = j * c11_params.EDGE_LENGTH
            base_y = i * c11_params.EDGE_LENGTH
            base_z = 0.0

            attribute = "support" if vertex_idx in corners else "load"
            v_name = f"v{vertex_idx}"

            shift_x, shift_y = 0.0, 0.0

            # Determine the values based on the mode.
            if params is not None:
                # Mode 2: read values from the Optuna optimum.
                shift_x = params.get(f"{v_name}_shift_x", 0.0)
                shift_y = params.get(f"{v_name}_shift_y", 0.0)
            else:
                # Mode 1: generate random values for the dataset.
                is_x_edge = (j == 0) or (j == num_nodes_x_top - 1)
                is_y_edge = (i == 0) or (i == num_nodes_y_top - 1)
                is_corner = is_x_edge and is_y_edge

                if not is_corner:
                    if is_x_edge:
                        shift_y = random.choice(shift_options)
                    elif is_y_edge:
                        shift_x = random.choice(shift_options)
                    else:
                        shift_x = random.choice(shift_options)
                        shift_y = random.choice(shift_options)

            final_x = base_x + shift_x
            final_y = base_y + shift_y
            final_z = base_z

            top_layer_coords[(i, j)] = {'x': final_x, 'y': final_y, 'z': final_z}

            all_vertices.append({
                "sample_id": sample_id,
                "vertex_index": v_name,
                "layer": "top",
                "attribute": attribute,
                "x": round(final_x, 3),
                "y": round(final_y, 3),
                "z": round(final_z, 3)
            })
            vertex_idx += 1

    # --- STEP 2: BOTTOM LAYER ---
    for i in range(c11_params.GRID_CELLS_Y):
        for j in range(c11_params.GRID_CELLS_X):
            v_name = f"v{vertex_idx}"

            p00 = top_layer_coords[(i, j)]      # Bottom-Left
            p10 = top_layer_coords[(i, j+1)]    # Bottom-Right
            p01 = top_layer_coords[(i+1, j)]    # Top-Left
            p11 = top_layer_coords[(i+1, j+1)]  # Top-Right

            # Determine the values based on the mode.
            if params is not None:
                # Mode 2: read values from the Optuna optimum.
                u = params.get(f"{v_name}_u", 0.5)
                v = params.get(f"{v_name}_v", 0.5)
                z_shift = params.get(f"{v_name}_shift_z", 0.0)
            else:
                # Mode 1: generate random values for the dataset.
                u = random.uniform(*c11_params.SCALE_UV)
                v = random.uniform(*c11_params.SCALE_UV)
                z_shift = random.choice(shift_options)

            lx, ly = bilinear_interpolate(p00, p10, p01, p11, u, v)
            final_z = -c11_params.LAYER_HEIGHT + z_shift

            all_vertices.append({
                "sample_id": sample_id,
                "vertex_index": v_name,
                "layer": "bottom",
                "attribute": "hinges",
                "x": round(lx, 3),
                "y": round(ly, 3),
                "z": round(final_z, 3)
            })
            vertex_idx += 1

    return _normalize_vertices_pca(all_vertices)

def generate_vertices(num_samples, round_decimals=2):
    """
    Generate a dataset of spatial trusses and ensure geometric diversity.
    Uses spatial discretization to reject strongly similar configurations (near-duplicates).
    """
    valid_shifts = get_valid_shifts(c11_params.DIVISIONS, c11_params.EDGE_LENGTH)
    all_data = []
    
    seen_signatures = set()
    samples_generated = 0
    attempts = 0
    max_attempts = num_samples * 10  # Safety limit for the while loop.
    
    while samples_generated < num_samples and attempts < max_attempts:
        # 1. Generate a candidate configuration.
        vertices = generate_sample_vertices(samples_generated, params=None, valid_shifts=valid_shifts)
        
        # 2. Create a topological signature via rounding (discretization).
        # We only extract the (x, y, z) coordinates and round them so that
        # micromillimeter variations are treated as duplicates.
        signature = tuple(
            (round(v['x'], round_decimals), 
             round(v['y'], round_decimals), 
             round(v['z'], round_decimals)) 
            for v in vertices
        )
        
        # 3. Validate the candidate's uniqueness.
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            all_data.extend(vertices)
            samples_generated += 1
            
        attempts += 1
        
    if attempts >= max_attempts:
          print(f"Warning: generation stopped early to prevent an infinite loop. "
              f"The design space may be too limited. Total generated: {samples_generated}")
              
    return pd.DataFrame(all_data)