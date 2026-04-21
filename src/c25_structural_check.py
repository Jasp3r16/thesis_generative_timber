from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd
import c21_surrogate_io as surrogate_io


def assign_roof_load_fz(
	df_vertices: pd.DataFrame,
	roof_load_kn_m2: float = 2.0,
	layer_column: str = "layer",
	top_layer_value: str = "top",
	bottom_layer_value: str = "bottom",
) -> pd.DataFrame:
	"""Assign nodal Fz from distributed roof load over the top-node roof footprint.

	The roof load is projected on the XY plane and distributed to top vertices using
	a triangulation-based tributary area method:
	- Triangulate the top-node XY points with `matplotlib.tri.Triangulation`
	- For each triangle, distribute one-third of its area to each triangle node
	- Nodal force is `Fz = -roof_load_kn_m2 * tributary_area_m2`
	- Bottom nodes are explicitly set to `Fz = 0`

	Args:
		df_vertices: Vertex table containing at least `x`, `y`, `z` and `layer`.
		roof_load_kn_m2: Uniform roof surface load in kN/m2.
		layer_column: Column that stores top/bottom layer labels.
		top_layer_value: Value that marks upper (roof) vertices.
		bottom_layer_value: Value that marks lower vertices.

	Returns:
		A copy of `df_vertices` with columns:
		- `Fz`: nodal vertical load in kN (negative downward)
		- `roof_tributary_area_m2`: tributary roof area per node in m2

	Raises:
		ValueError: If required columns are missing or top-node geometry is invalid.
	"""
	required_cols = {"x", "y", "z", layer_column}
	missing = [c for c in required_cols if c not in df_vertices.columns]
	if missing:
		raise ValueError("assign_roof_load_fz missing columns: " + ", ".join(missing))

	df_out = df_vertices.copy()
	df_out["roof_tributary_area_m2"] = 0.0
	df_out["Fz"] = 0.0

	top_mask = df_out[layer_column].astype(str).str.lower() == str(top_layer_value).lower()
	bottom_mask = df_out[layer_column].astype(str).str.lower() == str(bottom_layer_value).lower()

	df_top = df_out.loc[top_mask, ["x", "y"]].astype(float)
	if len(df_top) < 3:
		raise ValueError("Need at least 3 top-layer vertices to distribute roof load.")

	xy = df_top.to_numpy(dtype=float)
	if np.linalg.matrix_rank(xy - xy.mean(axis=0, keepdims=True)) < 2:
		raise ValueError("Top-layer vertices are collinear in XY; cannot define roof area.")

	# Local import keeps this utility lightweight unless structural stage uses it.
	import matplotlib.tri as mtri

	tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
	if tri.triangles is None or len(tri.triangles) == 0:
		raise ValueError("Could not triangulate top-layer roof footprint.")

	tributary_area = np.zeros(len(df_top), dtype=float)
	for a, b, c in tri.triangles:
		pa = xy[a]
		pb = xy[b]
		pc = xy[c]
		area = 0.5 * abs(np.cross(pb - pa, pc - pa))
		share = area / 3.0
		tributary_area[a] += share
		tributary_area[b] += share
		tributary_area[c] += share

	top_indices = df_out.index[top_mask].to_numpy()
	df_out.loc[top_indices, "roof_tributary_area_m2"] = tributary_area
	df_out.loc[top_indices, "Fz"] = -float(roof_load_kn_m2) * tributary_area

	# Preserve training convention: lower vertices carry no vertical roof load.
	df_out.loc[bottom_mask, "Fz"] = 0.0

	return df_out


def geometry_df_to_design_row(
	df_geometry: pd.DataFrame,
	df_edges: pd.DataFrame | None = None,
) -> pd.Series:
	"""Convert geometry and optional edge table to surrogate design-row format."""
	required = ["x", "y", "z"]
	missing = [c for c in required if c not in df_geometry.columns]
	if missing:
		raise ValueError(f"Geometry dataframe misses required columns: {missing}")

	numeric_columns = list(df_geometry.select_dtypes(include=[np.number]).columns)
	if not all(column in numeric_columns for column in required):
		raise ValueError("Geometry dataframe must store x, y and z as numeric columns.")

	coords = df_geometry[numeric_columns].reset_index(drop=True).astype(float)
	payload: dict[str, float] = {}
	for idx, row in coords.iterrows():
		for column_name in coords.columns:
			payload[f"v{idx}_{column_name}"] = float(row[column_name])
	if df_edges is not None and len(df_edges) > 0:
		edge_table = df_edges.reset_index(drop=True)
		columns_by_lower = {str(col).strip().lower(): col for col in edge_table.columns}
		edge_id_col = columns_by_lower.get("edge_id")
		area_col = None
		for candidate in ("area", "cross_section_area", "a"):
			if candidate in columns_by_lower:
				area_col = columns_by_lower[candidate]
				break

		if area_col is not None:
			edge_table[area_col] = pd.to_numeric(edge_table[area_col], errors="coerce")

		for idx, edge_row in edge_table.iterrows():
			edge_key_raw = str(edge_row[edge_id_col]).strip() if edge_id_col is not None else f"e{idx}"
			edge_key_raw_lower = edge_key_raw.lower()
			if edge_key_raw_lower.startswith("e"):
				edge_key = edge_key_raw_lower
			elif edge_key_raw.isdigit():
				edge_key = f"e{edge_key_raw}"
			else:
				edge_key = f"e{idx}"
			if not edge_key.startswith("e"):
				edge_key = f"e{idx}"
			if area_col is not None and pd.notna(edge_row[area_col]):
				payload[f"{edge_key}_Area"] = float(edge_row[area_col])

	return pd.Series(payload, dtype=np.float32)

def _predict_forces_with_surrogate(
    df_vertices: pd.DataFrame,
    df_edges: pd.DataFrame | None,
    bundle: dict[str, Any] | None,
    model_prefix: str | None,
) -> tuple[pd.DataFrame, dict[str, Any] | None, str]:
    """Predict forces via the surrogate model."""
    df_geometry = df_vertices.copy().reset_index(drop=True)

    # Apply distributed roof load as nodal Fz before surrogate inference.
    # By convention, top-layer nodes receive tributary load and bottom-layer nodes remain zero.
    df_geometry = assign_roof_load_fz(df_geometry, roof_load_kn_m2=2.0)

    active_bundle = bundle if bundle is not None else surrogate_io.load_surrogate_bundle(prefix_sm=model_prefix)

    design_row = geometry_df_to_design_row(
        df_geometry=df_geometry,
        df_edges=df_edges,
    )
    df_forces = surrogate_io.predict_edge_forces_kn(design_row, active_bundle).copy()
    df_forces["V1"] = df_forces["V1"].astype(str)
    df_forces["V2"] = df_forces["V2"].astype(str)
    df_forces["length_m"] = df_forces["length_m"].round(3)
    df_forces["axial_force_kn"] = df_forces["axial_force_kn"].round(2)
    return df_forces, active_bundle, "surrogate"

def prepare_surrogate_bundle(model_prefix: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """Try loading surrogate bundle once for re-use in iterative runs."""
    try:
        return surrogate_io.load_surrogate_bundle(prefix_sm=model_prefix), None
    except Exception as exc:
        return None, str(exc)
	
def calculate_utilization_for_dataset(
	row: pd.Series,
	req_length_m: float,
	gnn_margin: float = 1.10,
) -> float:
	"""
	- Calculates area for an element
	- Uses area and surrogate model to calculate axial force with surrogate model
	- Than uses that axial force to calculate Eurocode 5 utilization for one stock element

	Args:
		row: Row with at least `Depth`, `Width`, `f_c0k`, `f_tk`, `E_modulus_eff`.
		req_length_m: Requested member length in meters for buckling calculation.
		gnn_margin: Safety factor on predicted force (default 1.10).

	Returns:
		Utilization ratio. Values <= 1.0 are structurally acceptable.
		Returns `np.inf` if the calculated capacity is invalid or non-positive.
	"""
	h = float(row["Depth"])
	b = float(row["Width"])
	l_mm = float(req_length_m) * 1000.0
	area = h * b

	#Surrogate model predicts required axial force in kN
	req_force_kn = 0 # Output of surrogate model, placeholder for now

	required_force_kn = float(req_force_kn) * float(gnn_margin)

	f_c_k = float(row["f_c0k"])
	e_0_mean = float(row["E_modulus_eff"])
	f_t_k = float(row["f_tk"])

	gamma_m = 1.3
	k_mod = 0.8
	f_c_d = (f_c_k * k_mod) / gamma_m
	f_t_d = (f_t_k * k_mod) / gamma_m

	if required_force_kn >= 0:
		force_n = required_force_kn * 1000.0
		capaciteit_n = area * f_t_d
		if capaciteit_n <= 0:
			return float(np.inf)
		return force_n / capaciteit_n

	force_n = abs(required_force_kn * 1000.0)
	i_min = (max(h, b) * min(h, b) ** 3) / 12.0
	i_radius = math.sqrt(i_min / area)
	slenderness = l_mm / i_radius
	rel_slenderness = (slenderness / math.pi) * math.sqrt(f_c_k / e_0_mean)
	beta_c = 0.2
	k_waarde = 0.5 * (1 + beta_c * (rel_slenderness - 0.3) + rel_slenderness**2)
	k_c = 1 / (k_waarde + math.sqrt(max(0.0, k_waarde**2 - rel_slenderness**2)))

	capaciteit_n = area * k_c * f_c_d
	if capaciteit_n <= 0:
		return float(np.inf)
	return force_n / capaciteit_n

def compute_utilization_outputs(
	df_forces: pd.DataFrame,
	df_input_stock: pd.DataFrame,
	gnn_margin: float = 1.10,
) -> dict[str, pd.DataFrame]:
	"""Genereer alle utilization-tabellen voor notebook- en cost-matrix workflow.

	Args:
		df_forces: DataFrame met minimaal `edge_id` (of `beam_id`), `length_m`, `axial_force_kn`.
		df_input_stock: Stock-dataset met geometrie- en sterktekolommen.
		gnn_margin: Safety factor on the predicted force.

	Returns:
		Dictionary met:
		- `df_inventory`: opgeschoonde stock-data incl. `f_tk`
		- `df_forces_local`: genormaliseerde force-tabel
		- `df_utilization_long`: long-format combinatie-tabel (edge x stock)
		- `df_utilization_matrix`: raw utilization matrix
		- `df_utilization_matrix_display`: matrix met `inf` voor utilization > 1.0
		- `veilige_opties`: combinaties met 0 < utilization <= 1.0
		- `df_slots`: inputtabel voor cost matrix (`edge_id`, `length_m`, `axial_force_kn`, `Length_Req`)

	Raises:
		ValueError: Raised when required columns are missing in the force or stock data.
	"""
	df_forces_local = df_forces.copy()
	df_inventory = df_input_stock.copy()

	if "edge_id" not in df_forces_local.columns and "beam_id" in df_forces_local.columns:
		df_forces_local = df_forces_local.rename(columns={"beam_id": "edge_id"})

	required_stock_cols = ["Member_ID", "Length", "Depth", "Width", "f_c0k", "f_tk", "E_modulus_eff"]
	required_force_cols = ["edge_id", "length_m", "axial_force_kn"]

	missing_stock_cols = [c for c in required_stock_cols if c not in df_inventory.columns]
	missing_force_cols = [c for c in required_force_cols if c not in df_forces_local.columns]
	if missing_stock_cols:
		raise ValueError("Missing columns in df_input_stock: " + ", ".join(missing_stock_cols))
	if missing_force_cols:
		raise ValueError("Missing columns in df_forces: " + ", ".join(missing_force_cols))

	numeric_stock_cols = ["Length", "Depth", "Width", "f_c0k", "f_tk", "E_modulus_eff"]
	for col in numeric_stock_cols:
		df_inventory[col] = pd.to_numeric(df_inventory[col], errors="coerce")

	df_inventory = df_inventory.dropna(subset=numeric_stock_cols).copy()

	records: list[pd.DataFrame] = []
	for _, force_row in df_forces_local.iterrows():
		edge_id = str(force_row["edge_id"])
		req_force_kn = float(force_row["axial_force_kn"])
		req_length_m = float(force_row["length_m"])

		util_col = f"Utilization_{edge_id}"
		df_inventory[util_col] = df_inventory.apply(
			lambda stock_row: calculate_utilization_for_dataset(
				stock_row,
				req_force_kn=req_force_kn,
				req_length_m=req_length_m,
				gnn_margin=gnn_margin,
			),
			axis=1,
		)

		df_edge = df_inventory[["Member_ID", "Length", "Width", "Depth", util_col]].copy()
		df_edge = df_edge.rename(columns={util_col: "utilization"})
		df_edge["edge_id"] = edge_id
		df_edge["axial_force_kn"] = req_force_kn
		df_edge["length_m"] = req_length_m
		records.append(df_edge)

	df_utilization_long = pd.concat(records, ignore_index=True)

	df_utilization_matrix = df_utilization_long.pivot_table(
		index="edge_id",
		columns="Member_ID",
		values="utilization",
		aggfunc="first",
	)
	df_utilization_matrix = df_utilization_matrix.sort_index(axis=0).sort_index(axis=1)
	df_utilization_matrix_display = df_utilization_matrix.where(df_utilization_matrix <= 1.0, np.inf)

	safe_options = df_utilization_long[
		(df_utilization_long["utilization"] <= 1.0)
		& (df_utilization_long["utilization"] > 0.0)
		& np.isfinite(df_utilization_long["utilization"])
	].copy()
	safe_options = safe_options.sort_values(by=["edge_id", "utilization"], ascending=[True, False])

	df_slots = df_forces_local[["edge_id", "length_m", "axial_force_kn"]].copy()
	df_slots["Length_Req"] = (df_slots["length_m"] * 1000.0).round(0)

	# Derive required slot section (Width_Req, Depth_Req) from the most efficient
	# structurally safe option per edge (highest utilization <= 1.0).
	best_safe_per_edge = (
		safe_options
		.sort_values(by=["edge_id", "utilization"], ascending=[True, False])
		.drop_duplicates(subset=["edge_id"], keep="first")
		[["edge_id", "Depth", "Width", "utilization"]]
		.rename(
			columns={
				"Depth": "Depth_Req",
				"Width": "Width_Req",
				"utilization": "Utilization_Req",
			}
		)
	)

	df_slots = df_slots.merge(best_safe_per_edge, on="edge_id", how="left")

	return {
		"df_inventory": df_inventory,
		"df_forces_local": df_forces_local,
		"df_utilization_long": df_utilization_long,
		"df_utilization_matrix": df_utilization_matrix,
		"df_utilization_matrix_display": df_utilization_matrix_display,
		"safe_options": safe_options,
		"df_slots": df_slots,
	}


def validate_structural_stage_notebook_inputs(
	df_input_stock: pd.DataFrame | None,
	df_vertices: pd.DataFrame | None,
) -> None:
	"""Validate required notebook inputs for the structural stage wrapper."""
	missing: list[str] = []
	if df_input_stock is None:
		missing.append("df_input_stock")
	if df_vertices is None:
		missing.append("df_vertices")
	if missing:
		raise ValueError("Missing required structural inputs: " + ", ".join(missing))


def package_structural_outputs_for_notebook(
	structural_out: dict[str, Any],
	bundle_error: str | None = None,
) -> dict[str, Any]:
	"""Package structural stage outputs into notebook-friendly variable names."""
	summary = structural_out["summary"]
	return {
		"SURROGATE_BUNDLE": structural_out["bundle"],
		"SURROGATE_BUNDLE_ERROR": bundle_error,
		"structural_out": structural_out,
		"df_forces": structural_out["df_forces"],
		"df_inventory": structural_out["df_inventory"],
		"df_forces_local": structural_out["df_forces_local"],
		"df_utilization_long": structural_out["df_utilization_long"],
		"df_utilization_matrix": structural_out["df_utilization_matrix"],
		"df_utilization_matrix_display": structural_out["df_utilization_matrix_display"],
		"safe_options": structural_out["safe_options"],
		"df_slots": structural_out["df_slots"],
		"summary": summary,
		"forces_source": structural_out["forces_source"],
	}

__all__ = [
	"assign_roof_load_fz",
	"calculate_utilization_for_dataset",
	"geometry_df_to_design_row",
	"compute_utilization_outputs",
	"validate_structural_stage_notebook_inputs",
	"package_structural_outputs_for_notebook",
]
