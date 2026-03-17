import pandas as pd

def get_corner_indices(cells_x, cells_y):
    """
    Berekent de indices van de hoekpunten voor de Top Layer.
    Werkt voor elke grid grootte (n x m).
    """
    # Aantal punten is altijd aantal cellen + 1
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

def bilinear_interpolate(p00, p10, p01, p11, u, v):
    """
    Interpoleert een punt binnen een vierhoek.
    p00: Bottom-Left, p10: Bottom-Right, p01: Top-Left, p11: Top-Right (in standaard cartesiaans)
    Maar in matrix indexering (rij i, kol j):
    (i, j) is vaak Top-Left in images, maar Bottom-Left in Grasshopper/Cartesiaans als y omhoog gaat.
    Laten we aannemen: i=0 is y=0 (onder), i=max is y=max (boven).
    Dan is rij i "onder" rij i+1.
    p_bl = (i, j), p_br = (i, j+1)
    p_tl = (i+1, j), p_tr = (i+1, j+1)
    """
    # X interpolatie
    # Onderkant (row i)
    x_bot = p00['x'] * (1 - u) + p10['x'] * u
    # Bovenkant (row i+1)
    x_top = p01['x'] * (1 - u) + p11['x'] * u

    final_x = x_bot * (1 - v) + x_top * v

    # Y interpolatie
    y_bot = p00['y'] * (1 - u) + p10['y'] * u
    y_top = p01['y'] * (1 - u) + p11['y'] * u

    final_y = y_bot * (1 - v) + y_top * v

    return final_x, final_y

def generate_edges(num_samples, cells_x, cells_y):
    """
    Genereert een topologische verbindingslijst (edges) voor een dubbellaags ruimtelijk vakwerk.

    De functie bouwt een grid op bestaande uit drie onderdelen:
    1. Een Top Layer grid van (cells_x + 1) bij (cells_y + 1) punten.
    2. Een Bottom Layer grid van (cells_x) bij (cells_y) punten, gecentreerd onder de top-cellen.
    3. Diagonale verbindingen (piramide-structuur) tussen de bottom-punten en de vier bovenliggende top-punten.

    Args:
        num_samples (int): Het aantal unieke samples dat gegenereerd moet worden.
        cells_x (int): Het aantal cellen in de X-richting.
        cells_y (int): Het aantal cellen in de Y-richting.

    Returns:
        pd.DataFrame: Een DataFrame met de kolommen ['sample_id', 'edge_id', 'V1', 'V2'].
            V1 en V2 zijn de indices van de verbonden hoekpunten (vertices).
    """
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