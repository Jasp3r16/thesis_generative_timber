import scriptcontext as sc
import csv
import Rhino.Geometry as rg
from collections import defaultdict

# ==========================================
# 0. JOUW GRASSHOPPER VARIABELEN KOPPELEN
# ==========================================
input_file_vertices = file_path_vertices = 0
input_file_edges    = file_path_edges = 0
input_load_data     = load_data     = 0
input_sample_id     = num_sample = 0

dict_vert_name = "thesis_vertices_cache"
dict_edge_name = "thesis_edges_cache"

# ==========================================
# 1. DATA INLADEN MET OPTIMALISATIES
# ==========================================
if input_load_data or dict_vert_name not in sc.sticky or dict_edge_name not in sc.sticky:
    print("Datasets inladen... Een moment geduld a.u.b.")
    
    archive_vertices = defaultdict(dict)
    archive_edges = defaultdict(list)
    
    try:
        # OPTIMALISATIE 1: Voorkeur voor generator/list comprehension waar mogelijk
        with open(input_file_vertices, 'r') as f_vert:
            reader_v = csv.DictReader(f_vert)
            for row in reader_v:
                s_id = int(float(row['sample_id']))
                v_idx = int(row['vertex_index'].lstrip('v'))  # Sneller dan replace
                
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
                    'V1': int(float(row['V1'])),
                    'V2': int(float(row['V2'])),
                    'edge_id': row['edge_id']
                })
        
        # OPTIMALISATIE 2: Converteer defaultdict terug naar normale dict
        sc.sticky[dict_vert_name] = dict(archive_vertices)
        sc.sticky[dict_edge_name] = dict(archive_edges)
        
        print("Succes! Data in RAM geladen voor {} samples.".format(len(archive_vertices)))
        
    except Exception as e:
        print("Fout bij het inladen: " + str(e))

# ==========================================
# 2. SNELLE GEOMETRIE GENERATIE & FILTERING
# ==========================================
dict_v = sc.sticky.get(dict_vert_name, {})
dict_e = sc.sticky.get(dict_edge_name, {})

current_verts = dict_v.get(input_sample_id, {})
current_edges = dict_e.get(input_sample_id, [])

# OPTIMALISATIE 3: Vooraf-instantiëring in plaats van per-attribuut
Points = []
Lines = []
SupportPoints = []
LoadPoints = []
HingePoints = []
VertexIDs = []
EdgeIDs = []

ver_cord = []
point_lookup = {}

# OPTIMALISATIE 4: Categoriseer attributen eenmalig
SUPPORT_ATTR = 'support'

for v_idx, data in current_verts.items():
    x, y, z = data['x'], data['y'], data['z']
    pt = rg.Point3d(x, y, z)
    
    # Voeg toe aan universele lijsten
    Points.append(pt)
    ver_cord.extend([x, y, z])
    VertexIDs.append("v{}".format(v_idx))
    point_lookup[v_idx] = pt
    
    # OPTIMALISATIE 5: Vereenvoudigde logica met boolean checks
    attr = data.get('attribute', '').strip().lower()
    is_support = attr == SUPPORT_ATTR
    
    if is_support:
        SupportPoints.append(pt)
        LoadPoints.append(pt)
    else:
        # Alles behalve support is hinge
        HingePoints.append(pt)
        if attr == 'load':
            LoadPoints.append(pt)

# OPTIMALISATIE 6: Generator-expression voor Lines
for i, edge in enumerate(current_edges):
    idx1, idx2 = edge['V1'], edge['V2']
    
    if idx1 in point_lookup and idx2 in point_lookup:
        line = rg.Line(point_lookup[idx1], point_lookup[idx2])
        Lines.append(line)
        EdgeIDs.append(edge.get('edge_id', 'e_{}'.format(i)))

if not current_verts:
    print("Geen geometrie gevonden voor Sample ID: {}".format(input_sample_id))


import os
import csv

# ==========================================
# VARIABELEN
# ==========================================
input_file_path = file_path = 0
input_file_name = file_name = 0
input_coords_list = coords_list = 0
input_sample_id = sample_id = 0
input_label = label = 0
input_write = write = 0

# KIES ÉÉN VAN DEZE OPTIES:
OVERWRITE_MODE = False  # True = nieuw bestand telkens, False = voeg toe + voorkom duplicaten

# ==========================================
# HULPFUNCTIES
# ==========================================
def parse_datatree(data):
    """Converteer Grasshopper DataTree naar lijst met floats"""
    coords = []
    
    if isinstance(data, (list, tuple)):
        coords = [float(x) for x in data if isinstance(x, (int, float))]
    else:
        try:
            for line in str(data).strip().split('\n'):
                line = line.strip()
                if line and '{' not in line:
                    try:
                        coords.append(float(line))
                    except:
                        parts = line.split()
                        if parts:
                            try:
                                coords.append(float(parts[-1]))
                            except:
                                pass
        except:
            pass
    
    return coords

def extract_value(val):
    """Haal waarde uit lijst of enkel waarde"""
    return val[0] if isinstance(val, (list, tuple)) else val

def sample_exists(filepath, sample_id):
    """Check of sample_id al in CSV staat"""
    if not os.path.isfile(filepath):
        return False
    
    try:
        with open(filepath, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if row and row[0] == str(sample_id):
                    return True
    except:
        pass
    
    return False

# ==========================================
# MAIN
# ==========================================
def write_to_csv():
    """Schrijf data naar CSV bestand"""
    
    # Validatie pad
    if not os.path.isdir(input_file_path):
        print(f"❌ ERROR: Map bestaat niet: {input_file_path}")
        return
    
    # Bestandspad
    filepath = os.path.join(input_file_path, f"{input_file_name}.csv")
    
    # Parse coördinaten
    coords = parse_datatree(input_coords_list)
    if not coords:
        print(f"❌ ERROR: Geen coördinaten gevonden.")
        return
    
    # Haal waarden op
    sample_id = extract_value(input_sample_id)
    displacement = extract_value(input_label)
    
    # ===== OVERWRITE MODE =====
    if OVERWRITE_MODE:
        # Maak NIEUW bestand telkens (verwijdert oud bestand)
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                headers = ["sample_id"]
                num_nodes = len(coords) // 3
                for i in range(num_nodes):
                    headers.extend([f"v{i}_x", f"v{i}_y", f"v{i}_z"])
                headers.append("Max_Displacement")
                writer.writerow(headers)
                
                # Data
                row = [sample_id] + coords + [displacement]
                writer.writerow(row)
                
                print(f"✅ SUCCESS: Nieuw bestand aangemaakt met rij #{sample_id}.")
        
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    # ===== APPEND MODE MET DUPLICATE CHECK =====
    else:
        # Check voor duplicaten
        if sample_exists(filepath, sample_id):
            print(f"⚠️  WARNING: Sample ID #{sample_id} bestaat al. Niet toegevoegd.")
            return
        
        # Bestand bestaat al?
        file_exists = os.path.isfile(filepath) and os.path.getsize(filepath) > 0
        
        try:
            with open(filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                
                # Header (eerste keer)
                if not file_exists:
                    headers = ["sample_id"]
                    num_nodes = len(coords) // 3
                    for i in range(num_nodes):
                        headers.extend([f"v{i}_x", f"v{i}_y", f"v{i}_z"])
                    headers.append("Max_Displacement")
                    writer.writerow(headers)
                
                # Data
                row = [sample_id] + coords + [displacement]
                writer.writerow(row)
                
                print(f"✅ SUCCESS: Rij #{sample_id} toegevoegd (duplicaten vermeden).")
        
        except Exception as e:
            print(f"❌ ERROR: {e}")

# ==========================================
# TRIGGER
# ==========================================
if input_write:
    write_to_csv()