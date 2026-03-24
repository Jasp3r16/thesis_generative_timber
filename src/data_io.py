import config
import geometry
import c11_params
import warnings

def define_search_space(cells_x, cells_y, divisions, edge_length):
    """
    Vertaalt de geometrische constraints naar een digitaal leesbare 'Search Space'
    voor een machine learning of optimalisatie algoritme.

    Let op:
    De geometrie-reconstructie in `geometry.generate_sample_vertices` gebruikt momenteel
    grid/maat-waarden uit `c11_params`. Als je hier afwijkende waarden meegeeft,
    kan de gegenereerde search space niet 1-op-1 overeenkomen met de geometrie.
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

    # Hulpfunctie om te bepalen of een punt een hoek is
    corners = geometry.get_corner_indices(cells_x, cells_y).values()

    # --- 1. TOP LAYER VARIABELEN ---
    for i in range(num_nodes_y_top):
        for j in range(num_nodes_x_top):
            v_name = f"v{vertex_idx}"

            is_x_edge = (j == 0) or (j == num_nodes_x_top - 1)
            is_y_edge = (i == 0) or (i == num_nodes_y_top - 1)
            is_corner = vertex_idx in corners

            if is_corner:
                # AI mag hier niets mee doen
                pass
            elif is_x_edge:
                # Mag alleen schuiven over de Y-as
                search_space[f"{v_name}_shift_y"] = {"type": "discrete", "options": valid_shifts}
            elif is_y_edge:
                # Mag alleen schuiven over de X-as
                search_space[f"{v_name}_shift_x"] = {"type": "discrete", "options": valid_shifts}
            else:
                # Inner node: Mag in beide richtingen schuiven
                search_space[f"{v_name}_shift_x"] = {"type": "discrete", "options": valid_shifts}
                search_space[f"{v_name}_shift_y"] = {"type": "discrete", "options": valid_shifts}

            vertex_idx += 1

    # --- 2. BOTTOM LAYER VARIABELEN ---
    for r in range(cells_y):
        for c in range(cells_x):
            v_name = f"v{vertex_idx}"

            # u en v bepalen de positie ONDER de top-cel (continue waarden tussen 0.25 en 0.75)
            search_space[f"{v_name}_u"] = {"type": "continuous", "min": c11_params.SCALE_UV[0], "max": c11_params.SCALE_UV[1]}
            search_space[f"{v_name}_v"] = {"type": "continuous", "min": c11_params.SCALE_UV[0], "max": c11_params.SCALE_UV[1]}

            # Z-shift is weer een discrete stap
            search_space[f"{v_name}_shift_z"] = {"type": "discrete", "options": valid_shifts}

            vertex_idx += 1

    return search_space