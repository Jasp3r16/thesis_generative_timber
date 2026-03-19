import pandas as pd
import math

def reconstruct_edges(cells_x, cells_y):
    """
    Berekent de topologie (Edges) van het grid zonder geneste 'if' statements.
    Sneller en korter door gebruik van list comprehensions.
    """
    nodes_x_top = cells_x + 1
    nodes_y_top = cells_y + 1
    num_top = nodes_x_top * nodes_y_top
    
    edges = []

    # --- 1. TOP LAYER EDGES ---
    # Horizontale verbindingen
    edges.extend([(r * nodes_x_top + c, r * nodes_x_top + c + 1) 
                  for r in range(nodes_y_top) for c in range(cells_x)])
    # Verticale verbindingen
    edges.extend([(r * nodes_x_top + c, (r + 1) * nodes_x_top + c) 
                  for r in range(cells_y) for c in range(nodes_x_top)])

    # --- 2. BOTTOM LAYER EDGES ---
    start_bot = num_top
    # Horizontale verbindingen
    edges.extend([(start_bot + r * cells_x + c, start_bot + r * cells_x + c + 1) 
                  for r in range(cells_y) for c in range(cells_x - 1)])
    # Verticale verbindingen
    edges.extend([(start_bot + r * cells_x + c, start_bot + (r + 1) * cells_x + c) 
                  for r in range(cells_y - 1) for c in range(cells_x)])

    # --- 3. DIAGONALS (PIRAMIDE) ---
    for r in range(cells_y):
        for c in range(cells_x):
            current_bot = start_bot + r * cells_x + c
            tl, tr = r * nodes_x_top + c, r * nodes_x_top + (c + 1)
            bl, br = (r + 1) * nodes_x_top + c, (r + 1) * nodes_x_top + (c + 1)
            
            # Voeg direct alle 4 de diagonalen toe voor dit bottom-punt
            edges.extend([(current_bot, tl), (current_bot, tr), 
                          (current_bot, bl), (current_bot, br)])

    # Bouw het DataFrame in één keer op
    df_edges = pd.DataFrame(edges, columns=["V1", "V2"])
    
    # Voeg de edge_id als prefix toe
    df_edges.insert(0, "edge_id", ["e" + str(i) for i in range(len(df_edges))])
    
    return df_edges

def extract_beam_properties(df_vertices, df_edges):
    """
    Berekent staaflengtes en kent constructieve profielen toe
    op basis van de positie in het ruimtevakwerk.
    """
    # Maak een snelle zoek-dictionary voor de coördinaten
    # (Dit is veel sneller dan elke keer door de hele tabel zoeken)
    v_dict = df_vertices.set_index('vertex_index').to_dict('index')
    beams = []

    for _, edge in df_edges.iterrows():
        # Let op de 'v' prefix die we in stap 2 hebben toegevoegd
        v1_id = f"v{edge['V1']}" if not str(edge['V1']).startswith('v') else edge['V1']
        v2_id = f"v{edge['V2']}" if not str(edge['V2']).startswith('v') else edge['V2']

        pt1 = v_dict[v1_id]
        pt2 = v_dict[v2_id]

        # Stelling van Pythagoras (3D Euclidische afstand in meters)
        dx = pt1['x'] - pt2['x']
        dy = pt1['y'] - pt2['y']
        dz = pt1['z'] - pt2['z']
        lengte_m = math.sqrt(dx**2 + dy**2 + dz**2)

        # Converteer naar millimeters voor de hout-database
        lengte_mm = lengte_m * 1000.0

        # Bepaal het constructieve type en de benodigde dwarsdoorsnede (in mm)
        if pt1['layer'] == 'top' and pt2['layer'] == 'top':
            b_type = 'Top Chord'
            w_req, d_req = 75.0, 150.0   # Aangepast naar beschikbare maten (75x150)
        elif pt1['layer'] == 'bottom' and pt2['layer'] == 'bottom':
            b_type = 'Bottom Chord'
            w_req, d_req = 75.0, 100.0   # Aangepast naar 75x100
        else:
            b_type = 'Diagonal Web'
            w_req, d_req = 50.0, 100.0   # Diagonalen mogen dunner zijn (50x100)

        beams.append({
            'edge_id': edge['edge_id'],
            'type': b_type,
            'Length_Req': round(lengte_mm, 2),
            'Width_Req': w_req,
            'Depth_Req': d_req
        })

    return pd.DataFrame(beams)