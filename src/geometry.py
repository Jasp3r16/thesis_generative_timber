import pandas as pd
import random
from typing import Mapping, Optional, Sequence

import c11_params

def get_valid_shifts(divisions, edge_length):
    """Berekent de toegestane verschuivingen (verwijdert extremen)."""
    half_div = divisions // 2
    all_steps = list(range(-half_div, half_div + 1))
    valid_steps = all_steps[1:-1] # Verwijder eerste en laatste
    valid_shifts = [(step / divisions) * edge_length for step in valid_steps]
    return valid_shifts

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

# Zorg dat get_corner_indices, get_valid_shifts en bilinear_interpolate ook beschikbaar zijn

def generate_sample_vertices(
    sample_id: int,
    params: Optional[Mapping[str, float]] = None,
    valid_shifts: Optional[Sequence[float]] = None,
):
    """
    Genereert de coördinaten voor een enkel ruimtelijk vakwerk.
    
    Modus 1 (Dataset Generatie): Als 'params' None is, worden willekeurige 
    verschuivingen toegepast op basis van de valid_shifts.
    Modus 2 (Reconstructie): Als 'params' een dictionary is, worden de 
    specifieke optimum waarden ingeladen.
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

    # --- STAP 1: TOP LAYER ---
    for i in range(num_nodes_y_top):
        for j in range(num_nodes_x_top):
            base_x = j * c11_params.EDGE_LENGTH
            base_y = i * c11_params.EDGE_LENGTH
            base_z = 0.0

            attribute = "support" if vertex_idx in corners else "load"
            v_name = f"v{vertex_idx}"

            shift_x, shift_y = 0.0, 0.0

            # BEPAAL DE WAARDEN OP BASIS VAN DE MODUS
            if params is not None:
                # Modus 2: Haal op uit Optuna optimum
                shift_x = params.get(f"{v_name}_shift_x", 0.0)
                shift_y = params.get(f"{v_name}_shift_y", 0.0)
            else:
                # Modus 1: Genereer random voor de dataset
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

    # --- STAP 2: BOTTOM LAYER ---
    for i in range(c11_params.GRID_CELLS_Y):
        for j in range(c11_params.GRID_CELLS_X):
            v_name = f"v{vertex_idx}"

            p00 = top_layer_coords[(i, j)]      # Bottom-Left
            p10 = top_layer_coords[(i, j+1)]    # Bottom-Right
            p01 = top_layer_coords[(i+1, j)]    # Top-Left
            p11 = top_layer_coords[(i+1, j+1)]  # Top-Right

            # BEPAAL DE WAARDEN OP BASIS VAN DE MODUS
            if params is not None:
                # Modus 2: Haal op uit Optuna optimum
                u = params.get(f"{v_name}_u", 0.5)
                v = params.get(f"{v_name}_v", 0.5)
                z_shift = params.get(f"{v_name}_shift_z", 0.0)
            else:
                # Modus 1: Genereer random voor de dataset
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

    return all_vertices

def generate_full_dataset(num_samples, round_decimals=2):
    """
    Genereert een dataset van ruimtelijke vakwerken en waarborgt geometrische diversiteit.
    Maakt gebruik van ruimtelijke discretisatie om sterk op elkaar lijkende configuraties (near-duplicates) te verwerpen.
    """
    valid_shifts = get_valid_shifts(c11_params.DIVISIONS, c11_params.EDGE_LENGTH)
    all_data = []
    
    seen_signatures = set()
    samples_generated = 0
    attempts = 0
    max_attempts = num_samples * 10  # Veiligheidslimiet voor de while-loop
    
    while samples_generated < num_samples and attempts < max_attempts:
        # 1. Genereer een kandidaat-configuratie
        vertices = generate_sample_vertices(samples_generated, params=None, valid_shifts=valid_shifts)
        
        # 2. Creëer een topologische handtekening via afronding (discretisatie)
        # We extraheren alleen de (x, y, z) coördinaten en ronden ze af om
        # micromillimeter-variaties als duplicaten te identificeren.
        signature = tuple(
            (round(v['x'], round_decimals), 
             round(v['y'], round_decimals), 
             round(v['z'], round_decimals)) 
            for v in vertices
        )
        
        # 3. Valideer de uniciteit van de kandidaat
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            all_data.extend(vertices)
            samples_generated += 1
            
        attempts += 1
        
    if attempts >= max_attempts:
        print(f"Waarschuwing: Generatie voortijdig gestopt ter preventie van een oneindige loop. "
              f"De design space is mogelijk te beperkt. Totaal gegenereerd: {samples_generated}")
              
    return pd.DataFrame(all_data)