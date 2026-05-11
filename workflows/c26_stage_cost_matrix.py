from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import c00_headquarter_params as c26_params

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


def calculate_cost_formula(slot: pd.Series, stock_item: pd.Series) -> tuple[float, dict[str, float | str]]:
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

def build_cost_matrix(
    df_slots: pd.DataFrame,
    df_input_stock: pd.DataFrame,
    feasibility_mask: np.ndarray,
    **_: Any,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Build a feasibility-filtered cost matrix and detailed pair log.

    Parameters
    ----------
    df_slots : pd.DataFrame
        Slot table from c25_stage_feasibility.build_df_slots() with columns
        edge_id, length_m, Length_Req, Width_Req, Depth_Req.
    df_input_stock : pd.DataFrame
        Raw stock inventory (e.g. complete_timber.csv).
    feasibility_mask : np.ndarray bool [n_slots, n_stock]
        Boolean mask from c25_stage_feasibility.build_cost_filter().
        True = combination is feasible; False = set to inf in cost matrix.
    """
    if "edge_id" not in df_slots.columns:
        raise ValueError("df_slots must contain an edge_id column.")

    stock = prepare_stock_cost_inputs(df_input_stock)
    slot_ids = df_slots["edge_id"].astype(str).tolist()
    stock_ids = stock["Member_ID"].astype(str).tolist()

    n_slots = len(slot_ids)
    n_stock = len(stock_ids)

    if feasibility_mask.shape != (n_slots, n_stock):
        raise ValueError(
            f"feasibility_mask shape {feasibility_mask.shape} does not match "
            f"({n_slots}, {n_stock}) slots × stock."
        )

    cost_matrix = np.full((n_slots, n_stock), np.inf, dtype=float)
    logs: list[dict[str, Any]] = []

    for i, slot_id in enumerate(slot_ids):
        slot = df_slots.iloc[i]
        for j, stock_id in enumerate(stock_ids):
            stock_item = stock.iloc[j]
            feasible = bool(feasibility_mask[i, j])

            log_row: dict[str, Any] = {
                "Slot_ID": str(slot_id),
                "Stock_ID": str(stock_id),
                "Feasible": feasible,
                "Branch": _resolve_stock_branch(stock_item),
            }

            if feasible:
                total_cost, components = calculate_cost_formula(slot, stock_item)
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

            logs.append(log_row)

    df_logs = pd.DataFrame(logs)
    return cost_matrix, stock, df_logs