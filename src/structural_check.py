from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd


def bereken_utilization_voor_dataset(
	row: pd.Series,
	req_force_kn: float,
	req_length_m: float,
	gnn_marge: float = 1.10,
) -> float:
	"""Bereken Eurocode 5 utilization voor een stock element en gevraagde kracht."""
	reken_kracht_kn = float(req_force_kn) * float(gnn_marge)

	f_c_k = float(row["f_c0k"])
	e_0_mean = float(row["E_modulus_eff"])
	f_t_k = float(row["f_tk"])

	gamma_m = 1.3
	k_mod = 0.8
	f_c_d = (f_c_k * k_mod) / gamma_m
	f_t_d = (f_t_k * k_mod) / gamma_m

	b = float(row["Width"])
	h = float(row["Depth"])
	l_mm = float(req_length_m) * 1000.0
	area = b * h

	if reken_kracht_kn >= 0:
		force_n = reken_kracht_kn * 1000.0
		capaciteit_n = area * f_t_d
		if capaciteit_n <= 0:
			return float(np.inf)
		return force_n / capaciteit_n

	force_n = abs(reken_kracht_kn * 1000.0)
	i_min = (max(b, h) * min(b, h) ** 3) / 12.0
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


def _resolve_ftk_column(df_stock: pd.DataFrame) -> pd.DataFrame:
	"""Maak/normaliseer kolom f_tk op basis van aanwezige datasetkolommen."""
	df_resolved = df_stock.copy()
	if "f_tk" in df_resolved.columns:
		return df_resolved

	tensile_aliases = ["f_t0k", "f_t_0_k", "f_t_0k", "ftk", "f_t,k"]
	found_alias = next((col for col in tensile_aliases if col in df_resolved.columns), None)

	if found_alias is not None:
		df_resolved["f_tk"] = df_resolved[found_alias]
	elif "f_mk" in df_resolved.columns:
		df_resolved["f_tk"] = 0.58 * pd.to_numeric(df_resolved["f_mk"], errors="coerce")
	else:
		df_resolved["f_tk"] = 14.0

	return df_resolved


def compute_utilization_outputs(
	df_forces: pd.DataFrame,
	df_input_stock: pd.DataFrame,
	gnn_marge: float = 1.10,
) -> dict[str, pd.DataFrame]:
	"""Genereer utilization long/matrix, veilige opties en slots voor de workflow."""
	df_forces_local = df_forces.copy()
	df_inventory = _resolve_ftk_column(df_input_stock.copy())

	if "edge_id" not in df_forces_local.columns and "beam_id" in df_forces_local.columns:
		df_forces_local = df_forces_local.rename(columns={"beam_id": "edge_id"})

	required_stock_cols = ["Member_ID", "Length", "Width", "Depth", "f_c0k", "f_tk", "E_modulus_eff"]
	required_force_cols = ["edge_id", "length_m", "axial_force_kn"]

	missing_stock_cols = [c for c in required_stock_cols if c not in df_inventory.columns]
	missing_force_cols = [c for c in required_force_cols if c not in df_forces_local.columns]
	if missing_stock_cols:
		raise ValueError("Ontbrekende kolommen in df_input_stock: " + ", ".join(missing_stock_cols))
	if missing_force_cols:
		raise ValueError("Ontbrekende kolommen in df_forces: " + ", ".join(missing_force_cols))

	numeric_stock_cols = ["Length", "Width", "Depth", "f_c0k", "f_tk", "E_modulus_eff"]
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
			lambda stock_row: bereken_utilization_voor_dataset(
				stock_row,
				req_force_kn=req_force_kn,
				req_length_m=req_length_m,
				gnn_marge=gnn_marge,
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

	veilige_opties = df_utilization_long[
		(df_utilization_long["utilization"] <= 1.0)
		& (df_utilization_long["utilization"] > 0.0)
		& np.isfinite(df_utilization_long["utilization"])
	].copy()
	veilige_opties = veilige_opties.sort_values(by=["edge_id", "utilization"], ascending=[True, False])

	df_slots = df_forces_local[["edge_id", "length_m", "axial_force_kn"]].copy()
	df_slots["Length_Req"] = (df_slots["length_m"] * 1000.0).round(0)

	return {
		"df_inventory": df_inventory,
		"df_forces_local": df_forces_local,
		"df_utilization_long": df_utilization_long,
		"df_utilization_matrix": df_utilization_matrix,
		"df_utilization_matrix_display": df_utilization_matrix_display,
		"veilige_opties": veilige_opties,
		"df_slots": df_slots,
	}


__all__ = [
	"bereken_utilization_voor_dataset",
	"compute_utilization_outputs",
]
