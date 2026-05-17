from __future__ import annotations

import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

import c00_headquarter_params as c11_params


# =============================================================================
# GEOMETRY
# =============================================================================

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
    valid_steps = all_steps[1:-1]
    valid_shifts = [(step / divisions) * edge_length for step in valid_steps]
    return valid_shifts


def get_corner_indices(cells_x, cells_y):
    """
    Calculate the corner indices for the top layer.
    Works for any grid size (n x m).
    """
    nodes_x = cells_x + 1
    nodes_y = cells_y + 1
    total_nodes = nodes_x * nodes_y

    indices = {
        "bottom_left":  0,
        "bottom_right": nodes_x - 1,
        "top_left":     (nodes_y - 1) * nodes_x,
        "top_right":    total_nodes - 1,
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
    x_bot  = p00['x'] * (1 - u) + p10['x'] * u
    x_top  = p01['x'] * (1 - u) + p11['x'] * u
    final_x = x_bot * (1 - v) + x_top * v

    y_bot  = p00['y'] * (1 - u) + p10['y'] * u
    y_top  = p01['y'] * (1 - u) + p11['y'] * u
    final_y = y_bot * (1 - v) + y_top * v

    return final_x, final_y


def _normalize_vertices_pca(vertices):
    """Center a sample at its centroid while keeping its original orientation."""
    coords = np.array([[v["x"], v["y"], v["z"]] for v in vertices], dtype=np.float64)
    if coords.size == 0:
        return vertices

    normalized = coords - coords.mean(axis=0, keepdims=True)

    normalized_vertices = []
    for vertex, (x, y, z) in zip(vertices, normalized):
        updated_vertex = dict(vertex)
        updated_vertex["x"] = round(float(x), 3)
        updated_vertex["y"] = round(float(y), 3)
        updated_vertex["z"] = round(float(z), 3)
        normalized_vertices.append(updated_vertex)

    return normalized_vertices


def get_edge_dataframe(cells_x: int, cells_y: int) -> pd.DataFrame:
    """Return the fixed edge topology (edge_id, V1, V2) for a grid as a DataFrame."""
    edges = _build_truss_edges(cells_x, cells_y)
    df = pd.DataFrame(edges, columns=["V1", "V2"])
    df.insert(0, "edge_id", [f"e{i}" for i in range(len(df))])
    return df


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
                "edge_id":   f"e{edge_counter}",
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

            if params is not None:
                shift_x = params.get(f"{v_name}_shift_x", 0.0)
                shift_y = params.get(f"{v_name}_shift_y", 0.0)
            else:
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
                "sample_id":    sample_id,
                "vertex_index": v_name,
                "layer":        "top",
                "attribute":    attribute,
                "x": round(final_x, 3),
                "y": round(final_y, 3),
                "z": round(final_z, 3),
            })
            vertex_idx += 1

    # --- STEP 2: BOTTOM LAYER ---
    for i in range(c11_params.GRID_CELLS_Y):
        for j in range(c11_params.GRID_CELLS_X):
            v_name = f"v{vertex_idx}"

            p00 = top_layer_coords[(i,   j)]
            p10 = top_layer_coords[(i,   j+1)]
            p01 = top_layer_coords[(i+1, j)]
            p11 = top_layer_coords[(i+1, j+1)]

            if params is not None:
                u      = params.get(f"{v_name}_u", 0.5)
                v      = params.get(f"{v_name}_v", 0.5)
                z_shift = params.get(f"{v_name}_shift_z", 0.0)
            else:
                u      = random.uniform(*c11_params.SCALE_UV)
                v      = random.uniform(*c11_params.SCALE_UV)
                z_shift = random.choice(shift_options)

            lx, ly = bilinear_interpolate(p00, p10, p01, p11, u, v)
            final_z = -c11_params.LAYER_HEIGHT + z_shift

            all_vertices.append({
                "sample_id":    sample_id,
                "vertex_index": v_name,
                "layer":        "bottom",
                "attribute":    "hinges",
                "x": round(lx, 3),
                "y": round(ly, 3),
                "z": round(final_z, 3),
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
    max_attempts = num_samples * 10

    while samples_generated < num_samples and attempts < max_attempts:
        vertices = generate_sample_vertices(samples_generated, params=None, valid_shifts=valid_shifts)

        signature = tuple(
            (round(v['x'], round_decimals),
             round(v['y'], round_decimals),
             round(v['z'], round_decimals))
            for v in vertices
        )

        if signature not in seen_signatures:
            seen_signatures.add(signature)
            all_data.extend(vertices)
            samples_generated += 1

        attempts += 1

    if attempts >= max_attempts:
        print(f"Warning: generation stopped early to prevent an infinite loop. "
              f"The design space may be too limited. Total generated: {samples_generated}")

    return pd.DataFrame(all_data)


# =============================================================================
# TRAINING DATA — GNN Retraining Data Generator
# =============================================================================
#
# Generates training samples for GNN surrogate model retraining.
# Produces node and edge feature CSVs to be processed by Grasshopper/Karamba
# FEA (which adds the Utilization column) before c21_surrogate_training.py.
#
# Strategy: random stock assignment across randomly sampled geometries.
# Mean-EA force estimation (c24) is retained as a fast single FEM solve per
# geometry to produce N_mean_EA — structural demand per member.
#
# Outputs (saved to config.GH_DATA_PATH):
#   training_nodes_raw.csv  — node features, one row per node per sample
#   training_edges_raw.csv  — edge features without Utilization column
#
# After Grasshopper adds Utilization, run c21_surrogate_training.run_preprocessing()
# with edge_csv="training_edges_raw" and node_csv="training_nodes_raw".

NEW_EDGE_COLS = ["Depth_m", "Width_m", "Length", "E", "Iy", "Iz", "J", "EA/L", "N_mean_EA"]
NODE_COLS     = ["x", "y", "z", "Tx", "Ty", "Tz", "Rx", "Ry", "Rz", "Fz"]


def _sort_vertices(df_vertices: pd.DataFrame) -> pd.DataFrame:
    verts = df_vertices.copy()
    verts["v_idx"] = verts["vertex_index"].str.replace("v", "", regex=False).astype(int)
    return verts.sort_values("v_idx").reset_index(drop=True)


def _derive_node_roles(
    df_vertices: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, list[int], list[int]]:
    """Extract sorted vertex table, node_positions, support_nodes, load_nodes."""
    verts = _sort_vertices(df_vertices)
    node_positions = verts[["x", "y", "z"]].values
    support_nodes  = verts[verts["attribute"] == "support"]["v_idx"].tolist()
    load_nodes     = verts[verts["attribute"] == "load"]["v_idx"].tolist()
    return verts, node_positions, support_nodes, load_nodes


def _geometry_signature(df_vertices: pd.DataFrame, decimals: int = 2) -> tuple:
    """Coordinate-based signature for near-duplicate detection."""
    verts = _sort_vertices(df_vertices)
    return tuple(map(tuple, verts[["x", "y", "z"]].round(decimals).values))


def _member_lengths_m(node_positions: np.ndarray, df_edges: pd.DataFrame) -> np.ndarray:
    """Euclidean length of each member in metres."""
    v1 = df_edges["V1"].values.astype(int)
    v2 = df_edges["V2"].values.astype(int)
    return np.linalg.norm(node_positions[v2] - node_positions[v1], axis=1)


def _random_assignment(n_slots: int, n_stock: int, rng: np.random.Generator) -> np.ndarray:
    """Assign one stock element per slot, drawn uniformly from the full stock pool."""
    return rng.integers(0, n_stock, size=n_slots)


def _build_node_rows(
    verts_sorted:  pd.DataFrame,
    support_nodes: list[int],
    fz_nodal:      np.ndarray,
    sample_id:     int,
) -> list[dict]:
    """One node feature dict per vertex, ordered by v_idx."""
    support_set = set(support_nodes)
    rows = []
    for _, row in verts_sorted.iterrows():
        idx        = int(row["v_idx"])
        is_support = idx in support_set
        rows.append({
            "sample_id":    sample_id,
            "vertex_index": row["vertex_index"],
            "layer":        row["layer"],
            "attribute":    row["attribute"],
            "x":  round(float(row["x"]), 4),
            "y":  round(float(row["y"]), 4),
            "z":  round(float(row["z"]), 4),
            "Tx": 1.0 if is_support else 0.0,
            "Ty": 1.0 if is_support else 0.0,
            "Tz": 1.0 if is_support else 0.0,
            "Rx": 0.0,
            "Ry": 0.0,
            "Rz": 0.0,
            "Fz": round(float(fz_nodal[idx]), 4),
        })
    return rows


def _build_edge_rows(
    df_edges:      pd.DataFrame,
    lengths_m:     np.ndarray,
    assignment:    np.ndarray,
    stock_gnn:     pd.DataFrame,
    member_forces: np.ndarray,
    sample_id:     int,
) -> list[dict]:
    """One edge feature dict per member (e0..eN), ordered by edge position."""
    rows = []
    for i, (_, edge) in enumerate(df_edges.iterrows()):
        s = stock_gnn.iloc[int(assignment[i])]
        rows.append({
            "sample_id":      sample_id,
            "edge_id":        edge["edge_id"],
            "V1":             int(edge["V1"]),
            "V2":             int(edge["V2"]),
            "strength_class": str(s["strength_class"]),
            "Depth_m":        round(float(s["Depth_m"]), 4),
            "Width_m":        round(float(s["Width_m"]), 4),
            "Length":         round(float(lengths_m[i]), 3),
            "E":              round(float(s["E"]), 0),
            "Iy":             round(float(s["Iy"]), 8),
            "Iz":             round(float(s["Iz"]), 9),
            "J":              round(float(s["J"]), 8),
            "EA/L":           round(float(s["E"] * s["Width_m"] * s["Depth_m"] / lengths_m[i]), 1),
            "N_mean_EA":      round(float(member_forces[i]), 2),
            "stock_length_m": round(float(s["Length"]), 3),
            "assigned_stock": str(s.get("Member_ID", "")),
        })
    return rows


def _save_checkpoint(node_rows: list[dict], edge_rows: list[dict], output_dir: Path) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(node_rows).to_csv(output_dir / f"training_nodes_checkpoint_{ts}.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(output_dir / f"training_edges_checkpoint_{ts}.csv", index=False)
    print(f"  [checkpoint: {len(node_rows)} node rows, {len(edge_rows)} edge rows]")


def generate_training_samples(
    n_samples:         int                = 10_000,
    stock_csv_path:    "Path | str | None" = None,
    output_dir:        "Path | str | None" = None,
    name_prefix:       str                = "training",
    random_seed:       int                = 42,
    verbose:           bool               = False,
    save_intermediate: int                = 0,
) -> dict[str, Any]:
    """
    Generate training samples for GNN surrogate model retraining.

    For each sample:
      1. Draw a random geometry via generate_sample_vertices (Mode 1).
      2. Estimate member forces via mean-EA FEM (c24) → N_mean_EA per member.
      3. Randomly assign one stock element per slot from the full stock pool.
      4. Build node and edge feature rows.

    Parameters
    ----------
    n_samples         : total number of samples to generate (default 10 000)
    stock_csv_path    : path to complete_timber.csv (None -> config default)
    output_dir        : directory for output CSVs   (None -> config.GH_DATA_PATH)
    random_seed       : seed for reproducibility
    verbose           : print per-sample details
    save_intermediate : save checkpoint CSVs every N samples (0 = disabled)

    Returns
    -------
    dict with keys: nodes_csv_path, edges_csv_path, n_generated, n_skipped,
                    df_nodes, df_edges_out
    """
    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    import config
    from workflows import c24_stage_feasibility as stage_feas

    stock_csv_path = Path(stock_csv_path) if stock_csv_path else config.TIMBER_STOCK_PATH / "complete_timber_v2.csv"
    output_dir     = Path(output_dir)     if output_dir     else config.GH_DATA_PATH
    output_dir.mkdir(parents=True, exist_ok=True)

    rng          = np.random.default_rng(random_seed)
    valid_shifts = get_valid_shifts(c11_params.DIVISIONS, c11_params.EDGE_LENGTH)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[c12_geometry_truss] {ts}")
    print(f"  Stock:      {stock_csv_path.name}")
    print(f"  Output dir: {output_dir}")
    print(f"  Target:     {n_samples} samples  |  seed: {random_seed}")

    # ---- Load and prepare stock ----
    df_stock  = pd.read_csv(stock_csv_path, sep=";")
    n_stock   = len(df_stock)
    print(f"\n  Stock loaded: {n_stock} elements")

    stock_gnn = df_stock.copy()
    b_m = stock_gnn["Width"].values * 1e-3
    h_m = stock_gnn["Depth"].values * 1e-3
    a_m = np.minimum(b_m, h_m)
    c_m = np.maximum(b_m, h_m)
    stock_gnn["Width_m"]       = b_m
    stock_gnn["Depth_m"]       = h_m
    stock_gnn["E"]             = stock_gnn["E_modulus_eff"].values * 1e6
    stock_gnn["Iy"]            = b_m * h_m**3 / 12
    stock_gnn["Iz"]            = h_m * b_m**3 / 12
    stock_gnn["J"]             = a_m**3 * c_m / 3 * (1 - 0.63 * a_m / c_m)
    stock_gnn["Member_ID"]     = df_stock["Member_ID"].values
    stock_gnn["strength_class"] = df_stock["f_mk"].apply(lambda v: f"c{int(v)}")
    stock_gnn = stock_gnn.reset_index(drop=True)

    # ---- Mean-EA for force estimation (stock-average, constant across geometries) ----
    mean_E_Pa  = float(df_stock["E_modulus_eff"].mean()) * 1e6
    mean_A_m2  = float((df_stock["Depth"] * df_stock["Width"]).mean()) * 1e-6
    mean_EA_SI = mean_E_Pa * mean_A_m2

    # ---- Fixed topology ----
    df_edges = get_edge_dataframe(c11_params.GRID_CELLS_X, c11_params.GRID_CELLS_Y)
    n_edges  = len(df_edges)
    edges_v1 = df_edges["V1"].values.astype(int)
    edges_v2 = df_edges["V2"].values.astype(int)
    print(f"  Topology:   {n_edges} edges, {c11_params.GRID_CELLS_X}x{c11_params.GRID_CELLS_Y} grid")

    all_node_rows: list[dict] = []
    all_edge_rows: list[dict] = []
    seen_signatures: set      = set()

    n_generated  = 0
    n_skipped    = 0
    attempts     = 0
    max_attempts = n_samples * 4

    print(f"\n--- Generating {n_samples} samples ---")

    while n_generated < n_samples and attempts < max_attempts:
        attempts  += 1
        sample_id  = n_generated

        try:
            vertices_list = generate_sample_vertices(
                sample_id=sample_id, params=None, valid_shifts=valid_shifts
            )
            df_vertices = pd.DataFrame(vertices_list)

            sig = _geometry_signature(df_vertices)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)

            verts_sorted, node_positions, support_nodes, load_nodes = _derive_node_roles(df_vertices)

            member_forces = stage_feas.estimate_member_forces(
                node_positions=node_positions,
                edges_v1=edges_v1,
                edges_v2=edges_v2,
                support_nodes=support_nodes,
                load_nodes=load_nodes,
                total_load_n=stage_feas.TOTAL_LOAD_N,
                mean_EA_SI=mean_EA_SI,
            )

            fz_nodal   = stage_feas.compute_nodal_fz(node_positions, support_nodes, load_nodes)
            assignment = _random_assignment(n_edges, n_stock, rng)
            lengths_m  = _member_lengths_m(node_positions, df_edges)

            all_node_rows.extend(_build_node_rows(verts_sorted, support_nodes, fz_nodal, sample_id))
            all_edge_rows.extend(_build_edge_rows(df_edges, lengths_m, assignment, stock_gnn, member_forces, sample_id))
            n_generated += 1

            if n_generated % 100 == 0 or verbose:
                ts_now = datetime.now().strftime("%H:%M:%S")
                print(f"  [{ts_now}]  {n_generated:>6}/{n_samples}  "
                      f"(attempts={attempts}, skipped={n_skipped})", flush=True)

            if save_intermediate > 0 and n_generated % save_intermediate == 0:
                _save_checkpoint(all_node_rows, all_edge_rows, output_dir)

        except Exception as exc:
            warnings.warn(
                f"Sample attempt {attempts}: {type(exc).__name__}: {exc}",
                stacklevel=2,
            )
            n_skipped += 1

    if n_generated < n_samples:
        warnings.warn(
            f"Generation terminated early: {n_generated}/{n_samples} samples "
            f"after {attempts} attempts. Search space may be exhausted.",
            stacklevel=2,
        )

    df_nodes     = pd.DataFrame(all_node_rows)
    df_edges_out = pd.DataFrame(all_edge_rows)

    nodes_path = output_dir / f"{name_prefix}_nodes.csv"
    edges_path = output_dir / f"{name_prefix}_edges.csv"

    df_nodes.to_csv(nodes_path, index=False)
    df_edges_out.to_csv(edges_path, index=False)

    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Samples generated: {n_generated}")
    print(f"  Attempts:          {attempts}  (skipped: {n_skipped})")
    print(f"  Node rows:         {len(df_nodes)}  "
          f"({len(df_nodes) // max(n_generated, 1)} nodes/sample)")
    print(f"  Edge rows:         {len(df_edges_out)}  "
          f"({len(df_edges_out) // max(n_generated, 1)} edges/sample)")
    print(f"\n  Nodes CSV:  {nodes_path.name}")
    print(f"  Edges CSV:  {edges_path.name}")
    print(f"\n  Edge feature columns: {NEW_EDGE_COLS}")
    print(f"\n  Next step: open Grasshopper, load both CSVs, run Karamba FEA,")
    print(f"  and export an updated edges CSV with a 'Utilization' column.")
    print(f"  Then call c21_surrogate_training.run_preprocessing().")
    print(f"{'='*60}")

    return {
        "nodes_csv_path": nodes_path,
        "edges_csv_path": edges_path,
        "n_generated":    n_generated,
        "n_skipped":      n_skipped,
        "df_nodes":       df_nodes,
        "df_edges_out":   df_edges_out,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate GNN retraining data.")
    parser.add_argument("--n-samples",  type=int,  default=10_000,
                        help="Number of samples to generate (default 10000)")
    parser.add_argument("--seed",       type=int,  default=42,
                        help="Random seed (default 42)")
    parser.add_argument("--verbose",    action="store_true",
                        help="Print per-sample details")
    parser.add_argument("--checkpoint", type=int,  default=0,
                        help="Save checkpoint CSV every N samples (0 = off)")
    args = parser.parse_args()

    generate_training_samples(
        n_samples         = args.n_samples,
        random_seed       = args.seed,
        verbose           = args.verbose,
        save_intermediate = args.checkpoint,
    )
