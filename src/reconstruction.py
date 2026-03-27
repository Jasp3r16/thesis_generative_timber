import pandas as pd
import math
import json
import random
from pathlib import Path

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


def _normalize_vertex_id(vertex_id):
    """Normaliseer vertex index naar het formaat 'v{n}'."""
    return vertex_id if str(vertex_id).startswith('v') else f"v{vertex_id}"


def _validate_edge_index(edge_index):
    """Valideer edge_index in formaat [[sources...], [targets...]]."""
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
    Bereken de gemiddelde beamlengte voor een enkele structuur (sample_id).

    Returns:
        dict met gemiddelde lengte in meters en millimeters.
    """
    required_cols = {'sample_id', 'vertex_index', 'x', 'y', 'z'}
    missing_cols = required_cols - set(df_vertices.columns)
    if missing_cols:
        raise ValueError(f"df_vertices mist verplichte kolommen: {sorted(missing_cols)}")

    _validate_edge_index(edge_index)

    df_sample = df_vertices[df_vertices['sample_id'] == sample_id]
    if df_sample.empty:
        raise ValueError(f"Geen vertices gevonden voor sample_id={sample_id}")

    v_dict = df_sample.set_index('vertex_index').to_dict('index')
    sources, targets = edge_index

    lengths_m = []
    for v1_raw, v2_raw in zip(sources, targets):
        v1_id = _normalize_vertex_id(v1_raw)
        v2_id = _normalize_vertex_id(v2_raw)

        if v1_id not in v_dict or v2_id not in v_dict:
            raise KeyError(
                f"Edge verwijst naar ontbrekende vertex index: {v1_id} of {v2_id}"
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
    Bereken een robuuste representatieve beamlengte op basis van random gekozen samples.

    De functie kiest `sample_count` unieke sample_id's uit de dataset en neemt de
    mediaan van de per-sample gemiddelde beamlengtes om uitschieters te dempen.

    Args:
        df_vertices: DataFrame met o.a. sample_id, vertex_index, x, y, z.
        edge_index: Topologie in formaat [[sources...], [targets...]].
        sample_count: Aantal willekeurige samples om te gebruiken.
        random_state: Optionele seed voor reproduceerbare sample-selectie.

    Returns:
        dict met representatieve lengte in m en mm, plus traceerbare details.
    """
    if not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")

    available_sample_ids = sorted(df_vertices['sample_id'].unique().tolist())
    if sample_count > len(available_sample_ids):
        raise ValueError(
            f"sample_count={sample_count} is groter dan beschikbare unieke samples="
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
    Convenience wrapper die vertices CSV en edge_index JSON inleest.
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