import scriptcontext as sc
import csv
import Rhino.Geometry as rg
from collections import defaultdict

# ==========================================
# 0. INPUTS
# ==========================================
# Normal-mode inputs (ignored when reconstruction_mode is True)
input_file_vertices = globals().get("file_path_vertices")
input_file_edges    = globals().get("file_path_edges")
input_load_data     = globals().get("load_data", False)
input_sample_id     = globals().get("num_sample", 0)

# Reconstruction-mode controls
_reconstruction_mode = bool(globals().get("reconstruction_mode", False))
_top10_vertices_path = globals().get("top10_vertices")
_top10_edges_path    = globals().get("top10_edges")
_rank_chooser        = int(globals().get("rank_chooser", 1))

# Placement options (reconstruction mode only)
_placement_plane  = globals().get("placement_plane")
_structure_height = globals().get("structure_height")

# Timber dataset (used in reconstruction mode to look up section properties)
_timber_dataset_path = globals().get("timber_dataset_v2")

# ==========================================
# HELPERS
# ==========================================

def _normalize_vertex_id(vertex_id):
    s = str(vertex_id).strip()
    return s if s.startswith("v") else "v{}".format(int(float(s)))


def _coerce_plane(plane_input):
    if plane_input is None:
        return rg.Plane.WorldXY
    if isinstance(plane_input, rg.Plane):
        return plane_input
    if hasattr(plane_input, "Origin") and hasattr(plane_input, "XAxis") and hasattr(plane_input, "YAxis"):
        return rg.Plane(plane_input.Origin, plane_input.XAxis, plane_input.YAxis)
    raise TypeError("placement_plane must be a Rhino Plane or None")


def _place_point(pt, source_origin, target_plane, z_offset):
    local_x = pt.X - source_origin.X
    local_y = pt.Y - source_origin.Y
    local_z = pt.Z + z_offset
    return (
        target_plane.Origin
        + target_plane.XAxis * local_x
        + target_plane.YAxis * local_y
        + target_plane.ZAxis * local_z
    )


# ==========================================
# LOAD TIMBER DATASET LOOKUP (shared, cached)
# ==========================================
_timber_cache_name = "thesis_timber_cache"
_timber_lookup     = {}

if _timber_dataset_path:
    if input_load_data or _timber_cache_name not in sc.sticky:
        try:
            _timber_build = {}
            with open(_timber_dataset_path, 'r', newline='') as ft:
                for row in csv.DictReader(ft, delimiter=';'):
                    mid   = str(row['Member_ID']).strip()
                    state = row.get('State', '0').strip()
                    is_ns = (state == '0')
                    _timber_build[mid] = {
                        'depth_cm':       float(row['Depth'])  / 10.0,
                        'width_cm':       float(row['Width'])  / 10.0,
                        'stock_length_m': float(row['Length']) / 1000.0,
                        'is_c24':         1 if is_ns else 0,
                        'strength_class': 'c24' if is_ns else 'c18',
                    }
            sc.sticky[_timber_cache_name] = _timber_build
            print("Timber dataset loaded: {} entries.".format(len(_timber_build)))
        except Exception as e:
            print("Error loading timber dataset: " + str(e))
    _timber_lookup = sc.sticky.get(_timber_cache_name, {})

# ==========================================
# INITIALISE ALL OUTPUT VARIABLES
# ==========================================
Points              = []
Lines               = []
SupportPoints       = []
LoadPoints          = []
HingePoints         = []
VertexIDs           = []
EdgeIDs             = []
EdgeLengths         = []
LoadPointMarkers    = []
EdgeIndex           = []
EdgeIndexPyG        = [[], []]
AverageConnectivity = 0.0

# Normal-mode structural-property outputs (empty in reconstruction mode)
IsCc24        = []
StrengthClass = []
Depth         = []
Width         = []
StockLength   = []
E_list        = []
Iy_list       = []
Iz_list       = []
J_list        = []
EAL_list      = []
N_mean_EA     = []

# Reconstruction-mode outputs (None/empty in normal mode)
EdgeAssignedTimber   = []
EdgeCO2Penalty       = []
EdgeIsRS             = []
SourcePlane          = None
TargetPlane          = None
StructureHeight      = None
HeightOffset         = 0.0
df_vertices          = []
df_edges             = []
df_geometry_overview = []
df_node_overview     = []

# ==========================================
# BRANCH A: RECONSTRUCTION MODE
# ==========================================
if _reconstruction_mode:
    if not _top10_vertices_path or not _top10_edges_path:
        print("reconstruction_mode is ON but top10_vertices or top10_edges path is not connected.")
    else:
        try:
            # Read and filter vertices by rank
            vertices_rows = []
            with open(_top10_vertices_path, 'r', newline='') as fv:
                for row in csv.DictReader(fv):
                    if int(float(row['rank'])) == _rank_chooser:
                        vertices_rows.append(row)

            # Read and filter edges by rank
            edges_rows = []
            with open(_top10_edges_path, 'r', newline='') as fe:
                for row in csv.DictReader(fe):
                    if int(float(row['rank'])) == _rank_chooser:
                        edges_rows.append(row)

            if not vertices_rows:
                print("No vertices found for rank {}.".format(_rank_chooser))
            elif not edges_rows:
                print("No edges found for rank {}.".format(_rank_chooser))
            else:
                SUPPORT_ATTR = 'support'
                LOAD_ATTR    = 'load'
                HINGE_ATTR   = 'hinges'

                target_plane = _coerce_plane(_placement_plane)

                # Build raw point lookup and annotated vertex list
                raw_lookup    = {}
                vertices_work = []
                for row in vertices_rows:
                    v_id  = _normalize_vertex_id(row['vertex_index'])
                    pt    = rg.Point3d(float(row['x']), float(row['y']), float(row['z']))
                    attr  = str(row.get('attribute', '')).strip().lower()
                    raw_lookup[v_id] = pt
                    row_copy = dict(row)
                    row_copy['vertex_index_norm'] = v_id
                    row_copy['attr_lower']        = attr
                    vertices_work.append(row_copy)

                # Source origin for placement remapping
                support_raw = [raw_lookup[r['vertex_index_norm']] for r in vertices_work if r['attr_lower'] == SUPPORT_ATTR]
                if support_raw:
                    source_origin = min(support_raw, key=lambda p: (p.X, p.Y, p.Z))
                else:
                    source_origin = min(raw_lookup.values(), key=lambda p: (p.X, p.Y, p.Z))

                max_z    = max(p.Z for p in raw_lookup.values())
                z_offset = 0.0 if _structure_height is None else float(_structure_height) - max_z

                # Remap all points into the target plane
                point_lookup = {}
                for row in vertices_work:
                    v_id     = row['vertex_index_norm']
                    remapped = _place_point(raw_lookup[v_id], source_origin, target_plane, z_offset)
                    point_lookup[v_id] = remapped
                    row['x'] = remapped.X
                    row['y'] = remapped.Y
                    row['z'] = remapped.Z

                # Build vertex outputs
                degree_count = {r['vertex_index_norm']: 0 for r in vertices_work}
                for row in vertices_work:
                    v_id = row['vertex_index_norm']
                    attr = row['attr_lower']
                    pt   = point_lookup[v_id]
                    Points.append(pt)
                    VertexIDs.append(v_id)
                    if attr == SUPPORT_ATTR:
                        SupportPoints.append(pt)
                        LoadPoints.append(pt)
                        LoadPointMarkers.append(1)
                    elif attr == LOAD_ATTR:
                        LoadPoints.append(pt)
                        LoadPointMarkers.append(0)
                    elif attr == HINGE_ATTR:
                        HingePoints.append(pt)
                        LoadPointMarkers.append(0)
                    else:
                        LoadPointMarkers.append(0)

                # Build edge outputs
                for i, row in enumerate(edges_rows):
                    v1 = _normalize_vertex_id(row['V1'])
                    v2 = _normalize_vertex_id(row['V2'])
                    if v1 not in point_lookup or v2 not in point_lookup:
                        continue
                    line = rg.Line(point_lookup[v1], point_lookup[v2])
                    Lines.append(line)
                    e_id = str(row.get('edge_id', 'e{}'.format(i)))
                    EdgeIDs.append(e_id)
                    EdgeLengths.append(float(line.Length))
                    EdgeIndex.append((v1, v2))
                    EdgeIndexPyG[0].append(int(v1.lstrip('v')))
                    EdgeIndexPyG[1].append(int(v2.lstrip('v')))
                    degree_count[v1] = degree_count.get(v1, 0) + 1
                    degree_count[v2] = degree_count.get(v2, 0) + 1

                    timber = row.get('assigned_timber')
                    co2    = row.get('CO2_Penalty')
                    EdgeAssignedTimber.append(timber)
                    EdgeCO2Penalty.append(float(co2) if co2 not in (None, '') else None)
                    EdgeIsRS.append(1 if timber and 'RS' in str(timber).upper() else 0)

                    _td = _timber_lookup.get(str(timber).strip(), {})
                    IsCc24.append(_td.get('is_c24', 0))
                    StrengthClass.append(_td.get('strength_class', ''))
                    Depth.append(_td.get('depth_cm', 0.0))
                    Width.append(_td.get('width_cm', 0.0))
                    StockLength.append(_td.get('stock_length_m', 0.0))

                if degree_count:
                    AverageConnectivity = float(sum(degree_count.values())) / float(len(degree_count))

                SourcePlane     = rg.Plane(source_origin, rg.Vector3d.XAxis, rg.Vector3d.YAxis)
                TargetPlane     = target_plane
                StructureHeight = _structure_height
                HeightOffset    = z_offset

                df_vertices = vertices_work
                df_edges    = [dict(r) for r in edges_rows]
                df_geometry_overview = [
                    {
                        'edge_id':         eid,
                        'V1':              pair[0],
                        'V2':              pair[1],
                        'length_m':        round(length, 3),
                        'assigned_timber': timber,
                        'CO2_Penalty':     penalty,
                    }
                    for eid, pair, length, timber, penalty in zip(
                        EdgeIDs, EdgeIndex, EdgeLengths, EdgeAssignedTimber, EdgeCO2Penalty
                    )
                ]
                df_node_overview = [
                    {
                        'vertex_index': r.get('vertex_index'),
                        'layer':        r.get('layer'),
                        'attribute':    r.get('attribute'),
                        'x': r.get('x'),
                        'y': r.get('y'),
                        'z': r.get('z'),
                    }
                    for r in vertices_work
                ]

                print("Reconstruction mode: rank {} loaded. {} points, {} edges.".format(
                    _rank_chooser, len(Points), len(Lines)))

        except Exception as e:
            print("Error in reconstruction mode: " + str(e))
            raise

# ==========================================
# BRANCH B: NORMAL GEOMETRY GENERATION MODE
# ==========================================
else:
    dict_vert_name = "thesis_vertices_cache"
    dict_edge_name = "thesis_edges_cache"

    if input_load_data or dict_vert_name not in sc.sticky or dict_edge_name not in sc.sticky:
        print("Loading datasets... Please wait.")

        archive_vertices = defaultdict(dict)
        archive_edges    = defaultdict(list)

        try:
            with open(input_file_vertices, 'r') as f_vert:
                reader_v = csv.DictReader(f_vert)
                for row in reader_v:
                    s_id  = int(float(row['sample_id']))
                    v_idx = int(row['vertex_index'].lstrip('v'))
                    archive_vertices[s_id][v_idx] = {
                        'x':         float(row['x']),
                        'y':         float(row['y']),
                        'z':         float(row['z']),
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

    dict_v = sc.sticky.get(dict_vert_name, {})
    dict_e = sc.sticky.get(dict_edge_name, {})

    current_verts = dict_v.get(input_sample_id, {})
    current_edges = dict_e.get(input_sample_id, [])

    point_lookup = {}
    degree_count = dict((v_idx, 0) for v_idx in current_verts.keys())

    SUPPORT_ATTR = 'support'
    LOAD_ATTR    = 'load'
    HINGE_ATTR   = 'hinges'

    for v_idx, data in current_verts.items():
        pt = rg.Point3d(data['x'], data['y'], data['z'])
        Points.append(pt)
        VertexIDs.append("v{}".format(v_idx))
        point_lookup[v_idx] = pt
        attr = data.get('attribute', '').strip().lower()
        if attr == SUPPORT_ATTR:
            SupportPoints.append(pt)
            LoadPoints.append(pt)
            LoadPointMarkers.append(1)
        elif attr == LOAD_ATTR:
            LoadPoints.append(pt)
            LoadPointMarkers.append(0)
        elif attr == HINGE_ATTR:
            HingePoints.append(pt)
            LoadPointMarkers.append(0)
        else:
            LoadPointMarkers.append(0)

    for i, edge in enumerate(current_edges):
        idx1, idx2 = edge['V1'], edge['V2']
        if idx1 in point_lookup and idx2 in point_lookup:
            line = rg.Line(point_lookup[idx1], point_lookup[idx2])
            Lines.append(line)
            EdgeLengths.append(float(line.Length))
            e_id = edge.get('edge_id', 'e_{}'.format(i))
            EdgeIDs.append(e_id)
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
