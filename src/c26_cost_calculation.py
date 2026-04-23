from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import c26_params

M_A1_A3 = float(c26_params.IMPACT_FACTOR_A1_A3)
M_RECOVER = float(c26_params.IMPACT_FACTOR_RECOVERED_C1)
E_PREP_SAW = float(c26_params.ENERGY_PREP_SAW_A5)
E_OFFCUT = float(c26_params.ENERGY_OFFCUT_FACTOR_C3_C4)
SCARCITY_PENALTY = float(c26_params.SCARCITY_PENALTY)

_SLOT_ID_CANDIDATES = ("edge_id", "Edge_ID", "Slot_ID", "slot_id")
_STOCK_ID_CANDIDATES = ("Member_ID", "member_id", "Stock_ID", "stock_id")
_LENGTH_CANDIDATES = ("Length", "length_mm", "Length_mm")
_WIDTH_CANDIDATES = ("Width", "width_mm", "Width_mm")
_DEPTH_CANDIDATES = ("Depth", "depth_mm", "Depth_mm")
_STATE_CANDIDATES = ("State", "state", "State_Resolved")
_DENSITY_CANDIDATES = ("mean_density", "Mean_Density", "density", "Density")
_DISTANCE_CANDIDATES = ("Transport_Dist", "transport_dist", "Distance", "distance_km")
_TRANSPORT_FACTOR_CANDIDATES = ("EmissionFactor", "emission_factor", "TransportFactor", "Transport_Factor")


def _pick_column(df: pd.DataFrame, candidates: Sequence[str], *, required: bool = True) -> str | None:
    columns_by_lower = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        column = columns_by_lower.get(candidate.strip().lower())
        if column is not None:
            return column
    if required:
        raise ValueError(f"Missing required column. Expected one of: {list(candidates)}")
    return None


def _coerce_numeric(series: pd.Series, *, label: str) -> pd.Series:
    coerced = pd.to_numeric(series, errors="coerce")
    if coerced.isna().any():
        raise ValueError(f"Column '{label}' contains empty or non-numeric values.")
    return coerced.astype(float)


def _resolve_state(df_stock: pd.DataFrame, member_id: pd.Series) -> pd.Series:
    state_col = _pick_column(df_stock, _STATE_CANDIDATES, required=False)
    if state_col is not None:
        return _coerce_numeric(df_stock[state_col], label=state_col).clip(lower=0.0, upper=1.0)

    member_text = member_id.astype(str).str.strip().str.upper()
    return pd.Series(np.where(member_text.str.startswith("RS"), 1.0, 0.0), index=df_stock.index, dtype=float)


def _resolve_stock_branch(stock_item: pd.Series) -> str:
    state = float(stock_item.get("State_Resolved", 0.0))
    return "reclaimed" if state >= 0.5 else "new"


def _slot_requirements(slot: pd.Series) -> dict[str, float]:
    length_m = slot.get("length_m", np.nan)
    if not np.isfinite(length_m):
        length_req = slot.get("Length_Req", np.nan)
        if np.isfinite(length_req):
            length_m = float(length_req) / 1000.0
    width_mm = slot.get("Width_Req", np.nan)
    depth_mm = slot.get("Depth_Req", np.nan)

    if not np.isfinite(length_m) or not np.isfinite(width_mm) or not np.isfinite(depth_mm):
        raise ValueError("Slot row is missing required length/width/depth requirements.")

    return {
        "length_m": float(length_m),
        "width_mm": float(width_mm),
        "depth_mm": float(depth_mm),
        "area_m2": float(width_mm * depth_mm / 1_000_000.0),
    }


def _stock_geometry(stock_item: pd.Series) -> dict[str, float]:
    length_mm = float(stock_item["Length_Resolved"])
    width_mm = float(stock_item["Width_Resolved"])
    depth_mm = float(stock_item["Depth_Resolved"])
    return {
        "length_m": length_mm / 1000.0,
        "width_mm": width_mm,
        "depth_mm": depth_mm,
        "area_m2": float(width_mm * depth_mm / 1_000_000.0),
    }


def _validate_utilization_matrix(
    df_utilization_matrix: pd.DataFrame | np.ndarray,
    slot_ids: Sequence[str],
    stock_ids: Sequence[str],
) -> pd.DataFrame:
    if isinstance(df_utilization_matrix, pd.DataFrame):
        util = df_utilization_matrix.copy()
    else:
        util = pd.DataFrame(df_utilization_matrix, index=list(slot_ids), columns=list(stock_ids))

    util.index = util.index.astype(str)
    util.columns = util.columns.astype(str)
    util = util.apply(pd.to_numeric, errors="coerce")
    util = util.reindex(index=list(slot_ids), columns=list(stock_ids))
    return util


def prepare_stock_cost_inputs(df_stock_raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize stock data used by the v2 cost calculation."""
    df_stock = df_stock_raw.copy()

    member_id_col = _pick_column(df_stock, _STOCK_ID_CANDIDATES)
    length_col = _pick_column(df_stock, _LENGTH_CANDIDATES)
    width_col = _pick_column(df_stock, _WIDTH_CANDIDATES)
    depth_col = _pick_column(df_stock, _DEPTH_CANDIDATES)
    density_col = _pick_column(df_stock, _DENSITY_CANDIDATES)
    distance_col = _pick_column(df_stock, _DISTANCE_CANDIDATES)
    factor_col = _pick_column(df_stock, _TRANSPORT_FACTOR_CANDIDATES)

    df_stock["Member_ID"] = df_stock[member_id_col].astype(str)
    df_stock["Length_Resolved"] = _coerce_numeric(df_stock[length_col], label=length_col)
    df_stock["Width_Resolved"] = _coerce_numeric(df_stock[width_col], label=width_col)
    df_stock["Depth_Resolved"] = _coerce_numeric(df_stock[depth_col], label=depth_col)
    df_stock["Density_Resolved"] = _coerce_numeric(df_stock[density_col], label=density_col)
    df_stock["Distance_Resolved"] = _coerce_numeric(df_stock[distance_col], label=distance_col)
    df_stock["TransportFactor_Resolved"] = _coerce_numeric(df_stock[factor_col], label=factor_col)
    df_stock["State_Resolved"] = _resolve_state(df_stock, df_stock["Member_ID"])
    df_stock["Stock_Area_m2"] = (df_stock["Width_Resolved"] * df_stock["Depth_Resolved"]) / 1_000_000.0
    df_stock["Stock_Volume_m3"] = df_stock["Stock_Area_m2"] * (df_stock["Length_Resolved"] / 1000.0)

    if df_stock["Member_ID"].isna().any() or (df_stock["Member_ID"].astype(str).str.strip() == "").any():
        raise ValueError("Stock table contains empty Member_ID values.")

    return df_stock


def calculate_cost_formula_v2(slot: pd.Series, stock_item: pd.Series) -> tuple[float, dict[str, float | str]]:
    """Compute the c26 v2 cost value for one slot-stock pair."""
    requirements = _slot_requirements(slot)
    geometry = _stock_geometry(stock_item)

    density = float(stock_item["Density_Resolved"])
    distance_km = float(stock_item["Distance_Resolved"])
    transport_factor = float(stock_item["TransportFactor_Resolved"])
    state = float(stock_item["State_Resolved"])

    req_length_m = requirements["length_m"]
    req_area_m2 = requirements["area_m2"]
    stock_length_m = geometry["length_m"]
    stock_area_m2 = geometry["area_m2"]

    v_req = req_area_m2 * req_length_m
    v_stock = stock_area_m2 * stock_length_m
    v_waste = max(0.0, stock_area_m2 * (stock_length_m - req_length_m))
    v_over = max(0.0, stock_area_m2 * req_length_m - v_req)

    mass_req = v_req * density
    mass_stock = v_stock * density
    mass_waste = v_waste * density

    trans_factor_km = transport_factor / 1000.0

    branch = "reclaimed" if state >= 0.5 else "new"

    e_embodied = mass_req * M_A1_A3 if branch == "new" else 0.0
    e_transport = (mass_req * distance_km * trans_factor_km if branch == "new" 
                   else mass_stock * distance_km * trans_factor_km)
    e_recovered = mass_stock * M_RECOVER if branch == "reclaimed" else 0.0
    e_prep = mass_stock * E_PREP_SAW if branch == "reclaimed" else 0.0
    e_waste = mass_waste * E_OFFCUT if branch == "reclaimed" else 0.0
    e_scarcity = SCARCITY_PENALTY * v_waste if branch == "reclaimed" else 0.0

    e_new = e_embodied + e_transport
    e_reclaimed = e_recovered + e_transport + e_prep + e_waste + e_scarcity

    total_cost = e_reclaimed if branch == "reclaimed" else e_new

    components: dict[str, float | str] = {
        "branch": branch,
        "V_req": float(v_req),
        "V_over": float(v_over),
        "V_waste": float(v_waste),
        "V_stock": float(v_stock),
        "Mass_req": float(mass_req),
        "Mass_stock": float(mass_stock),
        "Mass_waste": float(mass_waste),
        "E_embodied": float(e_embodied),
        "E_recovered": float(e_recovered),
        "E_transport": float(e_transport),
        "E_prep": float(e_prep),
        "E_waste": float(e_waste),
        "E_scarcity": float(e_scarcity),
    }
    return float(total_cost), components


def calculate_cost_formula_v1(slot: pd.Series, stock_item: pd.Series) -> tuple[float, dict[str, float | str]]:
    """Compute the legacy v1 cost value for one slot-stock pair."""
    requirements = _slot_requirements(slot)
    geometry = _stock_geometry(stock_item)

    density = float(stock_item["Density_Resolved"])
    distance_km = float(stock_item["Distance_Resolved"])
    transport_factor = float(stock_item["TransportFactor_Resolved"])

    req_length_m = requirements["length_m"]
    req_area_m2 = requirements["area_m2"]
    stock_length_m = geometry["length_m"]
    stock_area_m2 = geometry["area_m2"]

    v_req = req_area_m2 * req_length_m
    v_stock = stock_area_m2 * stock_length_m
    v_waste = max(0.0, stock_area_m2 * (stock_length_m - req_length_m))
    v_over = max(0.0, v_stock - v_req)

    mass_stock = v_stock * density
    mass_waste = v_waste * density
    trans_factor_km = transport_factor / 1000.0

    e_embodied = v_stock * M_A1_A3
    e_prep = v_stock * E_PREP_SAW
    e_trans = mass_stock * distance_km * trans_factor_km
    e_waste = mass_waste * E_OFFCUT
    e_saw = 0.0 if np.isclose(stock_length_m, req_length_m) else E_PREP_SAW
    e_opp = SCARCITY_PENALTY * (v_over + v_waste)
    total_cost = e_embodied + e_prep + e_trans + e_waste + e_saw + e_opp

    components: dict[str, float | str] = {
        "branch": "legacy_v1",
        "V_req": float(v_req),
        "V_over": float(v_over),
        "V_waste": float(v_waste),
        "V_stock": float(v_stock),
        "Mass_req": float(v_req * density),
        "Mass_stock": float(mass_stock),
        "Mass_waste": float(mass_waste),
        "E_embodied": float(e_embodied),
        "E_recovered": 0.0,
        "E_transport": float(e_trans),
        "E_prep": float(e_prep),
        "E_waste": float(e_waste),
        "E_scarcity": float(e_opp),
        "TOTAL_Score": float(total_cost),
    }
    return float(total_cost), components

def _calculate_cost_pair(
    slot: pd.Series,
    stock_item: pd.Series,
    *,
    cost_formula_version: str,
) -> tuple[float, dict[str, float | str]]:
    version = str(cost_formula_version).strip().lower()
    if version == "v1":
        return calculate_cost_formula_v1(slot, stock_item)
    if version == "v2":
        return calculate_cost_formula_v2(slot, stock_item)
    raise ValueError("cost_formula_version must be 'v1' or 'v2'.")


def build_cost_matrix(
    df_slots: pd.DataFrame,
    df_stock_raw: pd.DataFrame,
    df_utilization_matrix: pd.DataFrame | np.ndarray | None = None,
    max_utilization_threshold: float = 1.0,
    target_stock_ids: Sequence[str] | None = None,
    cost_formula_version: str = "v2",
    **_: Any,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Build a feasibility-filtered cost matrix and detailed pair log."""
    if df_utilization_matrix is None:
        raise ValueError("df_utilization_matrix is required for c26 cost calculation.")

    if "edge_id" not in df_slots.columns:
        raise ValueError("df_slots must contain an edge_id column.")

    stock = prepare_stock_cost_inputs(df_stock_raw)
    slot_ids = df_slots["edge_id"].astype(str).tolist()
    stock_ids = stock["Member_ID"].astype(str).tolist()
    utilization = _validate_utilization_matrix(df_utilization_matrix, slot_ids, stock_ids)

    cost_matrix = np.full((len(slot_ids), len(stock_ids)), np.inf, dtype=float)
    logs: list[dict[str, Any]] = []
    target_stock_set = {str(item) for item in target_stock_ids} if target_stock_ids is not None else None

    for i, slot_id in enumerate(slot_ids):
        slot = df_slots.iloc[i]
        for j, stock_id in enumerate(stock_ids):
            stock_item = stock.iloc[j]
            util_value = utilization.iloc[i, j]
            util_float = float(util_value) if pd.notna(util_value) else float("nan")
            feasible = bool(np.isfinite(util_float) and util_float <= float(max_utilization_threshold))

            log_row: dict[str, Any] = {
                "Slot_ID": str(slot_id),
                "Stock_ID": str(stock_id),
                "Utilization": util_float,
                "Utilization_Threshold": float(max_utilization_threshold),
                "Feasible": feasible,
                "Branch": _resolve_stock_branch(stock_item),
                "Formula_Version": str(cost_formula_version),
            }

            if feasible:
                total_cost, components = _calculate_cost_pair(
                    slot,
                    stock_item,
                    cost_formula_version=cost_formula_version,
                )
                cost_matrix[i, j] = float(total_cost)
                log_row.update(
                    {
                        "Status": "feasible",
                        "Total cost": float(total_cost),
                        "V_req_m3": float(components["V_req"]),
                        "V_over_m3": float(components["V_over"]),
                        "V_waste_m3": float(components["V_waste"]),
                        "V_stock_m3": float(components["V_stock"]),
                        "Mass_req_kg": float(components["Mass_req"]),
                        "Mass_stock_kg": float(components["Mass_stock"]),
                        "E_embodied": float(components["E_embodied"]),
                        "E_recovered": float(components["E_recovered"]),
                        "E_transport": float(components["E_transport"]),
                        "E_prep": float(components["E_prep"]),
                        "E_waste": float(components["E_waste"]),
                        "E_scarcity": float(components["E_scarcity"]),
                    }
                )
            else:
                log_row.update(
                    {
                        "Status": "infeasible",
                        "Total cost": float("inf"),
                        "V_req_m3": np.nan,
                        "V_over_m3": np.nan,
                        "V_waste_m3": np.nan,
                        "V_stock_m3": np.nan,
                        "Mass_req_kg": np.nan,
                        "Mass_stock_kg": np.nan,
                        "E_embodied": np.nan,
                        "E_recovered": np.nan,
                        "E_transport": np.nan,
                        "E_prep": np.nan,
                        "E_waste": np.nan,
                        "E_scarcity": np.nan,
                    }
                )

            if target_stock_set is None or stock_id in target_stock_set:
                logs.append(log_row)

    df_logs = pd.DataFrame(logs)
    df_logs.attrs["utilization_mode"] = "c25_feasibility_matrix"
    df_logs.attrs["utilization_threshold"] = float(max_utilization_threshold)
    df_logs.attrs["cost_formula_version"] = str(cost_formula_version)
    return cost_matrix, stock, df_logs


def analyze_and_export_slot_logs(
    df_logs: pd.DataFrame,
    target_slot_for_analysis: str,
    all_stock_ids: Sequence[str],
    export_dir: Path,
    display_fn: Any | None = None,
    max_full_list_rows: int | None = None,
    show_full_list: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Prepare a compact per-slot analysis table and return the export path."""
    analysis_export_path = export_dir / f"c26_depth_analysis_{target_slot_for_analysis}.csv"

    if df_logs.empty:
        empty = pd.DataFrame(columns=df_logs.columns)
        return empty, empty, analysis_export_path

    slot_mask = df_logs["Slot_ID"].astype(str).str.strip().str.lower() == str(target_slot_for_analysis).strip().lower()
    df_slot = df_logs.loc[slot_mask].copy().sort_values(["Stock_ID"]).reset_index(drop=True)
    df_slot_rs = df_slot[df_slot["Stock_ID"].astype(str).str.upper().str.startswith("RS")].copy().reset_index(drop=True)

    if not show_full_list and max_full_list_rows is not None:
        df_slot = df_slot.head(int(max_full_list_rows)).copy()

    if display_fn is not None:
        display_fn(df_slot)

    return df_slot, df_slot_rs, analysis_export_path
