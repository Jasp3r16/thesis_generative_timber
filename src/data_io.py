import config
import geometry
import c11_params
import warnings

def define_search_space(cells_x, cells_y, divisions, edge_length):
    """
    Translate the geometric constraints into a machine-readable search space
    for a machine learning or optimization algorithm.

    Note:
    The geometry reconstruction in `geometry.generate_sample_vertices` currently
    uses grid and size values from `c11_params`. If you pass different values here,
    the generated search space may not match the geometry one-to-one.
    """
    if (
        cells_x != c11_params.GRID_CELLS_X
        or cells_y != c11_params.GRID_CELLS_Y
        or edge_length != c11_params.EDGE_LENGTH
    ):
        warnings.warn(
            "define_search_space() called with values that differ from c11_params. "
            "geometry.generate_sample_vertices currently uses c11_params directly, "
            "so this can cause a mismatch between search-space keys and geometry.",
            stacklevel=2,
        )

    valid_shifts = geometry.get_valid_shifts(divisions, edge_length)
    search_space = {}

    num_nodes_x_top = cells_x + 1
    num_nodes_y_top = cells_y + 1
    vertex_idx = 0

    # Helper to determine whether a point is a corner.
    corners = geometry.get_corner_indices(cells_x, cells_y).values()

    # --- 1. TOP LAYER VARIABLES ---
    for i in range(num_nodes_y_top):
        for j in range(num_nodes_x_top):
            v_name = f"v{vertex_idx}"

            is_x_edge = (j == 0) or (j == num_nodes_x_top - 1)
            is_y_edge = (i == 0) or (i == num_nodes_y_top - 1)
            is_corner = vertex_idx in corners

            if is_corner:
                # AI must not modify this node.
                pass
            elif is_x_edge:
                # Can only shift along the Y axis.
                search_space[f"{v_name}_shift_y"] = {"type": "discrete", "options": valid_shifts}
            elif is_y_edge:
                # Can only shift along the X axis.
                search_space[f"{v_name}_shift_x"] = {"type": "discrete", "options": valid_shifts}
            else:
                # Inner node: can shift in both directions.
                search_space[f"{v_name}_shift_x"] = {"type": "discrete", "options": valid_shifts}
                search_space[f"{v_name}_shift_y"] = {"type": "discrete", "options": valid_shifts}

            vertex_idx += 1

    # --- 2. BOTTOM LAYER VARIABLES ---
    for r in range(cells_y):
        for c in range(cells_x):
            v_name = f"v{vertex_idx}"

            # u and v determine the position under the top cell (continuous values between 0.25 and 0.75).
            search_space[f"{v_name}_u"] = {"type": "continuous", "min": c11_params.SCALE_UV[0], "max": c11_params.SCALE_UV[1]}
            search_space[f"{v_name}_v"] = {"type": "continuous", "min": c11_params.SCALE_UV[0], "max": c11_params.SCALE_UV[1]}

            # Z-shift is again a discrete step.
            search_space[f"{v_name}_shift_z"] = {"type": "discrete", "options": valid_shifts}

            vertex_idx += 1

    return search_space