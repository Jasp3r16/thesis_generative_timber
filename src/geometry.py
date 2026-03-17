import pandas as pd

def generate_edges(num_samples, cells_x, cells_y):
    edges_data = []

    # Bereken hulpparameters
    nodes_x_top = cells_x + 1
    nodes_y_top = cells_y + 1
    num_top_vertices = nodes_x_top * nodes_y_top

    # We itereren door elke sample om de edges per sample vast te leggen
    for sample_id in range(num_samples):

        edge_counter = 0  # Reset edge counter per sample (of wil je unieke ID's over de hele file? Meestal per sample resetten: e0..e127)

        # Hulpfunctie om edge toe te voegen
        def add_edge(u, v):
            nonlocal edge_counter
            edges_data.append({
                "sample_id": sample_id,
                "edge_id": f"e{edge_counter}",
                "V1": u,
                "V2": v,
            })
            edge_counter += 1

        # --- 1. TOP LAYER GRID ---
        # Vertices 0 tot num_top_vertices-1
        for r in range(nodes_y_top):      # loop rijen
            for c in range(nodes_x_top):  # loop kolommen
                current = r * nodes_x_top + c

                # Horizontaal (naar rechts)
                if c < cells_x: # zolang niet de laatste kolom
                    add_edge(current, current + 1)

                # Verticaal (naar beneden, of 'boven' in matrix index)
                if r < cells_y: # zolang niet de laatste rij
                    add_edge(current, current + nodes_x_top)

        # --- 2. BOTTOM LAYER GRID ---
        # Start index is na de laatste top vertex
        start_idx_bottom = num_top_vertices

        # Bottom grid heeft evenveel punten als er cellen zijn (cells_x * cells_y)
        # Maar de grid verbindingen zijn er eentje minder dan het aantal punten
        # Bottom punten zijn een grid van (cells_x) breed bij (cells_y) hoog.

        for r in range(cells_y):
            for c in range(cells_x):
                current = start_idx_bottom + r * cells_x + c

                # Horizontaal (naar rechts)
                if c < cells_x - 1:
                    add_edge(current, current + 1)

                # Verticaal (naar beneden)
                if r < cells_y - 1:
                    add_edge(current, current + cells_x)

        # --- 3. DIAGONALS (Pyramid connections) ---
        # Verbind elke Bottom vertex met de 4 Top vertices erboven
        for r in range(cells_y):
            for c in range(cells_x):
                bottom_node = start_idx_bottom + r * cells_x + c

                # De 4 corresponderende punten in de Top layer
                # Top grid is (cells_x + 1) breed
                top_tl = r * nodes_x_top + c               # Top-Left (of row i)
                top_tr = r * nodes_x_top + (c + 1)         # Top-Right
                top_bl = (r + 1) * nodes_x_top + c         # Bottom-Left (row i+1)
                top_br = (r + 1) * nodes_x_top + (c + 1)   # Bottom-Right

                add_edge(bottom_node, top_tl)
                add_edge(bottom_node, top_tr)
                add_edge(bottom_node, top_bl)
                add_edge(bottom_node, top_br)

    return pd.DataFrame(edges_data)