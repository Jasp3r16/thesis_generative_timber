"""Reconstruct the optimized final geometry for Grasshopper.

This script reads the final optimizer exports:
- c29_optimum_vertices.csv
- c29_optimum_edges.csv

It mirrors the geometry extraction script outputs, but it works on one final
optimized structure instead of iterating over samples.
"""

from __future__ import annotations

import csv
from pathlib import Path
import importlib

try:
	rg = importlib.import_module("Rhino.Geometry")
except ModuleNotFoundError as exc:
	rg = None
	_RHINO_IMPORT_ERROR = exc


def _resolve_input_path(default_name: str, provided_path: str | None = None) -> Path:
	if provided_path:
		return Path(provided_path)
	return Path(default_name)


def _normalize_vertex_id(vertex_id) -> str:
	vertex_text = str(vertex_id).strip()
	return vertex_text if vertex_text.startswith("v") else f"v{int(float(vertex_text))}"


def _coerce_plane(plane_input):
	if plane_input is None:
		return rg.Plane.WorldXY
	if isinstance(plane_input, rg.Plane):
		return plane_input
	if hasattr(plane_input, "Origin") and hasattr(plane_input, "XAxis") and hasattr(plane_input, "YAxis"):
		return rg.Plane(plane_input.Origin, plane_input.XAxis, plane_input.YAxis)
	if hasattr(plane_input, "origin") and hasattr(plane_input, "x_axis") and hasattr(plane_input, "y_axis"):
		return rg.Plane(plane_input.origin, plane_input.x_axis, plane_input.y_axis)
	raise TypeError("placement_plane must be a Rhino plane-like object or None")


def _make_point(row: dict[str, str]):
	return rg.Point3d(float(row["x"]), float(row["y"]), float(row["z"]))


def _place_point(point, source_origin, target_plane, z_offset):
	local_x = point.X - source_origin.X
	local_y = point.Y - source_origin.Y
	local_z = point.Z + z_offset
	return (
		target_plane.Origin
		+ target_plane.XAxis * local_x
		+ target_plane.YAxis * local_y
		+ target_plane.ZAxis * local_z
	)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
	with path.open("r", newline="", encoding="utf-8-sig") as handle:
		return list(csv.DictReader(handle))


def _load_single_geometry_tables(vertices_path: Path, edges_path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
	vertices_rows = _read_csv_rows(vertices_path)
	edges_rows = _read_csv_rows(edges_path)

	required_vertex_cols = {"vertex_index", "x", "y", "z"}
	required_edge_cols = {"edge_id", "V1", "V2"}

	missing_vertex_cols = required_vertex_cols - set(vertices_rows[0].keys() if vertices_rows else [])
	missing_edge_cols = required_edge_cols - set(edges_rows[0].keys() if edges_rows else [])

	if missing_vertex_cols:
		raise ValueError(f"Vertices CSV is missing required columns: {sorted(missing_vertex_cols)}")
	if missing_edge_cols:
		raise ValueError(f"Edges CSV is missing required columns: {sorted(missing_edge_cols)}")

	if any("sample_id" in row for row in vertices_rows):
		unique_sample_ids = sorted({row.get("sample_id") for row in vertices_rows if row.get("sample_id") not in (None, "")})
		if len(unique_sample_ids) > 1:
			raise ValueError(
				"Expected one optimized geometry, but vertices CSV contains multiple sample_id values: "
				f"{unique_sample_ids}"
			)

	if any("sample_id" in row for row in edges_rows):
		unique_edge_sample_ids = sorted({row.get("sample_id") for row in edges_rows if row.get("sample_id") not in (None, "")})
		if len(unique_edge_sample_ids) > 1:
			raise ValueError(
				"Expected one optimized geometry, but edges CSV contains multiple sample_id values: "
				f"{unique_edge_sample_ids}"
			)

	return vertices_rows, edges_rows


def _build_geometry_outputs(
	vertices_rows: list[dict[str, str]],
	edges_rows: list[dict[str, str]],
	placement_plane=None,
	structure_height: float | None = None,
) -> dict[str, object]:
	if rg is None:
		raise ImportError("Rhino.Geometry is required to reconstruct the optimized geometry.") from _RHINO_IMPORT_ERROR

	target_plane = _coerce_plane(placement_plane)

	points = []
	lines = []
	support_points = []
	load_points = []
	hinge_points = []
	vertex_ids = []
	edge_ids = []
	edge_lengths = []
	load_point_markers = []
	edge_index = []
	edge_index_pyg = [[], []]
	edge_assigned_timber = []
	edge_co2_penalty = []
	edge_is_rs = []

	point_lookup = {}
	degree_count = {}
	raw_points = []

	support_attr = "support"
	load_attr = "load"
	hinge_attr = "hinges"

	vertices_work = []
	for row in vertices_rows:
		row_copy = dict(row)
		row_copy["vertex_index_norm"] = _normalize_vertex_id(row_copy["vertex_index"])
		vertices_work.append(row_copy)

	for row in vertices_work:
		vertex_index = row["vertex_index_norm"]
		attr = str(row.get("attribute", "")).strip().lower()

		point = _make_point(row)
		raw_points.append(point)
		vertex_ids.append(vertex_index)
		point_lookup[vertex_index] = point
		degree_count[vertex_index] = 0

		if attr == support_attr:
			support_points.append(point)
			load_points.append(point)
			load_point_markers.append(1)
		elif attr == load_attr:
			load_points.append(point)
			load_point_markers.append(0)
		elif attr == hinge_attr:
			hinge_points.append(point)
			load_point_markers.append(0)
		else:
			load_point_markers.append(0)

	if not raw_points:
		raise ValueError("No vertices were loaded from the optimum vertices CSV")

	source_support_points = [
		point_lookup[row["vertex_index_norm"]]
		for row in vertices_work
		if str(row.get("attribute", "")).strip().lower() == support_attr
	]
	if source_support_points:
		source_origin = min(source_support_points, key=lambda pt: (pt.X, pt.Y, pt.Z))
	else:
		source_origin = min(raw_points, key=lambda pt: (pt.X, pt.Y, pt.Z))

	max_z = max(point.Z for point in raw_points)
	if structure_height is None:
		z_offset = 0.0
	else:
		z_offset = float(structure_height) - max_z

	remapped_lookup = {}
	for row in vertices_work:
		vertex_index = row["vertex_index_norm"]
		remapped_point = _place_point(point_lookup[vertex_index], source_origin, target_plane, z_offset)
		remapped_lookup[vertex_index] = remapped_point
		row["x"] = remapped_point.X
		row["y"] = remapped_point.Y
		row["z"] = remapped_point.Z

	point_lookup = remapped_lookup
	points = [point_lookup[row["vertex_index_norm"]] for row in vertices_work]
	support_points = []
	load_points = []
	hinge_points = []
	for row in vertices_work:
		remapped_point = point_lookup[row["vertex_index_norm"]]
		attr = str(row.get("attribute", "")).strip().lower()
		if attr == support_attr:
			support_points.append(remapped_point)
			load_points.append(remapped_point)
		elif attr == load_attr:
			load_points.append(remapped_point)
		elif attr == hinge_attr:
			hinge_points.append(remapped_point)

	edges_work = []
	for row in edges_rows:
		row_copy = dict(row)
		row_copy["V1_norm"] = _normalize_vertex_id(row_copy["V1"])
		row_copy["V2_norm"] = _normalize_vertex_id(row_copy["V2"])
		edges_work.append(row_copy)

	for row in edges_work:
		v1 = row["V1_norm"]
		v2 = row["V2_norm"]

		if v1 not in point_lookup or v2 not in point_lookup:
			continue

		line = rg.Line(point_lookup[v1], point_lookup[v2])
		lines.append(line)

		edge_id = str(row.get("edge_id", f"e{len(edge_ids)}"))
		edge_ids.append(edge_id)
		edge_lengths.append(float(line.Length))
		edge_index.append((v1, v2))
		edge_index_pyg[0].append(int(v1.lstrip("v")))
		edge_index_pyg[1].append(int(v2.lstrip("v")))
		degree_count[v1] = degree_count.get(v1, 0) + 1
		degree_count[v2] = degree_count.get(v2, 0) + 1

		if "assigned_timber" in row:
			edge_assigned_timber.append(row.get("assigned_timber"))
		else:
			edge_assigned_timber.append(None)

		# Binary flag: 1 for RS members, 0 for NS members (aligned with edge order)
		assigned_val = row.get("assigned_timber")
		if assigned_val is None:
			edge_is_rs.append(0)
		else:
			# treat any value containing 'RS' (case-insensitive) as RS
			edge_is_rs.append(1 if "RS" in str(assigned_val).upper() else 0)

		if "CO2_Penalty" in row:
			edge_co2_penalty.append(float(row.get("CO2_Penalty", 0.0)))
		else:
			edge_co2_penalty.append(None)

	geometry_overview = [
		{
			"edge_id": edge_id,
			"V1": pair[0],
			"V2": pair[1],
			"length_m": round(length, 3),
			"assigned_timber": timber,
			"CO2_Penalty": penalty,
		}
		for edge_id, pair, length, timber, penalty in zip(
			edge_ids, edge_index, edge_lengths, edge_assigned_timber, edge_co2_penalty
		)
	]

	node_overview = [
		{
			"vertex_index": row.get("vertex_index"),
			"layer": row.get("layer"),
			"attribute": row.get("attribute"),
			"x": row.get("x"),
			"y": row.get("y"),
			"z": row.get("z"),
		}
		for row in vertices_work
	]

	return {
		"Points": points,
		"Lines": lines,
		"SupportPoints": support_points,
		"LoadPoints": load_points,
		"HingePoints": hinge_points,
		"VertexIDs": vertex_ids,
		"EdgeIDs": edge_ids,
		"EdgeLengths": edge_lengths,
		"LoadPointMarkers": load_point_markers,
		"EdgeIndex": edge_index,
		"EdgeIndexPyG": edge_index_pyg,
		"SourcePlane": rg.Plane(source_origin, rg.Vector3d.XAxis, rg.Vector3d.YAxis),
		"TargetPlane": target_plane,
		"StructureHeight": structure_height,
		"HeightOffset": z_offset,
		"EdgeAssignedTimber": edge_assigned_timber,
		"EdgeCO2Penalty": edge_co2_penalty,
		"df_vertices": vertices_work,
		"df_edges": edges_work,
		"df_geometry_overview": geometry_overview,
		"df_node_overview": node_overview,
		"EdgeIsRS": edge_is_rs,
	}


def reconstruct_optimum_geometry(
	vertices_csv: str | None = None,
	edges_csv: str | None = None,
	placement_plane=None,
	structure_height: float | None = None,
) -> dict[str, object]:
	vertices_path = _resolve_input_path("c29_optimum_vertices.csv", vertices_csv)
	edges_path = _resolve_input_path("c29_optimum_edges.csv", edges_csv)

	vertices_df, edges_df = _load_single_geometry_tables(vertices_path, edges_path)
	return _build_geometry_outputs(
		vertices_df,
		edges_df,
		placement_plane=placement_plane,
		structure_height=structure_height,
	)


_gh_vertices_csv = globals().get("file_path_vertices")
_gh_edges_csv = globals().get("file_path_edges")
_gh_placement_plane = globals().get("placement_plane")
_gh_structure_height = globals().get("structure_height")

OUTPUTS = reconstruct_optimum_geometry(
	vertices_csv=_gh_vertices_csv,
	edges_csv=_gh_edges_csv,
	placement_plane=_gh_placement_plane,
	structure_height=_gh_structure_height,
)

Points = OUTPUTS["Points"]
Lines = OUTPUTS["Lines"]
SupportPoints = OUTPUTS["SupportPoints"]
LoadPoints = OUTPUTS["LoadPoints"]
HingePoints = OUTPUTS["HingePoints"]
VertexIDs = OUTPUTS["VertexIDs"]
EdgeIDs = OUTPUTS["EdgeIDs"]
EdgeLengths = OUTPUTS["EdgeLengths"]
LoadPointMarkers = OUTPUTS["LoadPointMarkers"]
EdgeIndex = OUTPUTS["EdgeIndex"]
EdgeIndexPyG = OUTPUTS["EdgeIndexPyG"]
SourcePlane = OUTPUTS["SourcePlane"]
TargetPlane = OUTPUTS["TargetPlane"]
StructureHeight = OUTPUTS["StructureHeight"]
HeightOffset = OUTPUTS["HeightOffset"]
EdgeAssignedTimber = OUTPUTS["EdgeAssignedTimber"]
EdgeCO2Penalty = OUTPUTS["EdgeCO2Penalty"]
EdgeIsRS = OUTPUTS["EdgeIsRS"]
df_vertices = OUTPUTS["df_vertices"]
df_edges = OUTPUTS["df_edges"]
df_geometry_overview = OUTPUTS["df_geometry_overview"]
df_node_overview = OUTPUTS["df_node_overview"]