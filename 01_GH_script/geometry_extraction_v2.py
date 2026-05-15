import scriptcontext as sc
import csv
import Rhino.Geometry as rg
from collections import defaultdict

# ==========================================
# 0. CONNECT YOUR GRASSHOPPER VARIABLES
# ==========================================
input_file_vertices = file_path_vertices
input_file_edges    = file_path_edges
input_load_data     = load_data
input_sample_id     = num_sample

dict_vert_name = "thesis_vertices_cache"
dict_edge_name = "thesis_edges_cache"

# ==========================================
# 1. LOAD DATA WITH OPTIMIZATIONS
# ==========================================
if input_load_data or dict_vert_name not in sc.sticky or dict_edge_name not in sc.sticky:
    print("Loading datasets... Please wait.")
    
    archive_vertices = defaultdict(dict)
    archive_edges = defaultdict(list)
    
    try:
        with open(input_file_vertices, 'r') as f_vert:
            reader_v = csv.DictReader(f_vert)
            for row in reader_v:
                s_id = int(float(row['sample_id']))
                v_idx = int(row['vertex_index'].lstrip('v'))
                
                archive_vertices[s_id][v_idx] = {
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'z': float(row['z']),
                    'attribute': row['attribute'].strip().lower()
                }

        with open(input_file_edges, 'r') as f_edge:
            reader_e = csv.DictReader(f_edge)
            for row in reader_e:
                s_id = int(float(row['sample_id']))
                
                archive_edges[s_id].append({
                    'V1':             int(float(row['V1'])),
                    'V2':             int(float(row['V2'])),
                    'edge_id':        row['edge_id'],
                    'strength_class': row.get('strength_class', '').strip().lower(),
                    'Width_m':        float(row.get('Width_m', 0)),
                    'Depth_m':        float(row.get('Depth_m', 0)),
                    'Length':         float(row.get('Length', 0)),
                    'E':              float(row.get('E', 0)),
                    'Iy':             float(row.get('Iy', 0)),
                    'Iz':             float(row.get('Iz', 0)),
                    'J':              float(row.get('J', 0)),
                    'EA_over_L':      float(row.get('EA/L', 0)),
                    'N_mean_EA':      float(row.get('N_mean_EA', 0)),
                })
        
        sc.sticky[dict_vert_name] = dict(archive_vertices)
        sc.sticky[dict_edge_name] = dict(archive_edges)
        
        print("Success! Data loaded in RAM for {} samples.".format(len(archive_vertices)))
        
    except Exception as e:
        print("Error loading data: " + str(e))

# ==========================================
# 2. EFFICIENT GEOMETRY GENERATION & FILTERING
# ==========================================
dict_v = sc.sticky.get(dict_vert_name, {})
dict_e = sc.sticky.get(dict_edge_name, {})

current_verts = dict_v.get(input_sample_id, {})
current_edges = dict_e.get(input_sample_id, [])

Points = []
Lines = []
SupportPoints = []
LoadPoints = []
HingePoints = []
VertexIDs = []
EdgeIDs = []
EdgeLengths = []  # Line length per edge (same order as EdgeIDs/EdgeIndex)
LoadPointMarkers = []  # Binary marker list (1 for support, 0 for others)
EdgeIndex = []  # (source_vertex_index, target_vertex_index) tuples for readable GH output
EdgeIndexPyG = [[], []]  # [sources, targets] numeric 2xE format for PyTorch Geometric
AverageConnectivity = 0.0  # Mean node degree for the current sample graph
IsCc24 = []       # Binary: 1 = C24 timber, 0 = C18 timber (per edge, same order as Lines)
StrengthClass = []# Strength class string per edge: "c24" or "c18"
Depth = []        # Section depth in cm per edge
Width = []        # Section width in cm per edge
StockLength = []  # Stock element length per edge (m)
E_list = []       # Elastic modulus per edge (Pa), pass-through from training CSV
Iy_list = []      # Second moment of area strong axis per edge (m4)
Iz_list = []      # Second moment of area weak axis per edge (m4)
J_list = []       # Torsional constant per edge (m4)
EAL_list = []     # Axial stiffness EA/L per edge (N/m)
N_mean_EA = []    # Mean-EA axial force estimate per edge (N), for dataset export
point_lookup = {}
degree_count = dict((v_idx, 0) for v_idx in current_verts.keys())

# OPTIMIZATION 4: Define attribute constants for single-pass processing
SUPPORT_ATTR = 'support'
LOAD_ATTR = 'load'
HINGE_ATTR = 'hinges'

# OPTIMIZATION 5: Single-pass point processing with unified categorization
for v_idx, data in current_verts.items():
    x, y, z = data['x'], data['y'], data['z']
    pt = rg.Point3d(x, y, z)
    
    # Add to universal lists
    Points.append(pt)
    VertexIDs.append("v{}".format(v_idx))
    point_lookup[v_idx] = pt
    
    # OPTIMIZATION 6: Simplified attribute logic with direct categorization
    attr = data.get('attribute', '').strip().lower()
    
    is_support_point = (attr == SUPPORT_ATTR)
    is_load_point = (attr == LOAD_ATTR)
    is_hinge_point = (attr == HINGE_ATTR)
    
    # Categorize points efficiently
    if is_support_point:
        SupportPoints.append(pt)
        LoadPoints.append(pt)  # Support points also have load
        LoadPointMarkers.append(1)
    elif is_load_point:
        LoadPoints.append(pt)
        LoadPointMarkers.append(0)
    elif is_hinge_point:
        HingePoints.append(pt)
        LoadPointMarkers.append(0)
    else:
        LoadPointMarkers.append(0)

# OPTIMIZATION 7: Single-pass edge processing
for i, edge in enumerate(current_edges):
    idx1, idx2 = edge['V1'], edge['V2']
    
    if idx1 in point_lookup and idx2 in point_lookup:
        line = rg.Line(point_lookup[idx1], point_lookup[idx2])
        Lines.append(line)
        EdgeLengths.append(line.Length)
        edge_id = edge.get('edge_id', 'e_{}'.format(i))
        EdgeIDs.append(edge_id)
        EdgeIndex.append((idx1, idx2))
        EdgeIndexPyG[0].append(idx1)
        EdgeIndexPyG[1].append(idx2)
        degree_count[idx1] += 1
        degree_count[idx2] += 1
        sc_str = edge.get('strength_class', '')
        IsCc24.append(1 if sc_str == 'c24' else 0)
        StrengthClass.append(sc_str)
        Depth.append(edge.get('Depth_m', 0.0) * 100.0)
        Width.append(edge.get('Width_m', 0.0) * 100.0)
        StockLength.append(edge.get('Length', 0.0))
        E_list.append(edge.get('E', 0.0))
        Iy_list.append(edge.get('Iy', 0.0))
        Iz_list.append(edge.get('Iz', 0.0))
        J_list.append(edge.get('J', 0.0))
        EAL_list.append(edge.get('EA_over_L', 0.0))
        N_mean_EA.append(edge.get('N_mean_EA', 0.0))

if degree_count:
    AverageConnectivity = float(sum(degree_count.values())) / float(len(degree_count))

print("Processed {} points: {} support, {} load (including support), {} hinges".format(
    len(Points), len(SupportPoints), len(LoadPoints), len(HingePoints)))
print("Average connectivity (mean node degree): {:.3f}".format(AverageConnectivity))

if not current_verts:
    print("No geometry found for Sample ID: {}".format(input_sample_id))