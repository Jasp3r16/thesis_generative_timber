from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import c00_headquarter_params as c26_params

# =============================================================================
# LCA impact constants (from headquarter params)
# =============================================================================

M_A1_A3        = float(c26_params.IMPACT_FACTOR_A1_A3)
M_RECOVER      = float(c26_params.IMPACT_FACTOR_RECOVERED_C1)
E_PREP_SAW     = float(c26_params.ENERGY_PREP_SAW_A5)
E_OFFCUT       = float(c26_params.ENERGY_OFFCUT_FACTOR_C3_C4)
SCARCITY_PENALTY = float(c26_params.SCARCITY_PENALTY)

# =============================================================================
# Column name candidates (case-insensitive lookup)
# =============================================================================

_SLOT_ID_CANDIDATES           = ("edge_id", "Edge_ID", "Slot_ID", "slot_id")
_STOCK_ID_CANDIDATES          = ("Member_ID", "member_id", "Stock_ID", "stock_id")
_LENGTH_CANDIDATES            = ("Length", "length_mm", "Length_mm")
_WIDTH_CANDIDATES             = ("Width", "width_mm", "Width_mm")
_DEPTH_CANDIDATES             = ("Depth", "depth_mm", "Depth_mm")
_STATE_CANDIDATES             = ("State", "state", "State_Resolved")
_DENSITY_CANDIDATES           = ("mean_density", "Mean_Density", "density", "Density")
_DISTANCE_CANDIDATES          = ("Transport_Dist", "transport_dist", "Distance", "distance_km")
_TRANSPORT_FACTOR_CANDIDATES  = ("EmissionFactor", "emission_factor", "TransportFactor", "Transport_Factor")


# =============================================================================
# Column helpers
# =============================================================================

def _pick_column(df: pd.DataFrame, candidates: Sequence[str], *,
                 required: bool = True) -> str | None:
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
        return _coerce_numeric(
            df_stock[state_col], label=state_col
        ).clip(lower=0.0, upper=1.0)
    member_text = member_id.astype(str).str.strip().str.upper()
    return pd.Series(
        np.where(member_text.str.startswith("RS"), 1.0, 0.0),
        index=df_stock.index, dtype=float,
    )


def _resolve_stock_branch(stock_item: pd.Series) -> str:
    return "reclaimed" if float(stock_item.get("State_Resolved", 0.0)) >= 0.5 else "new"


# =============================================================================
# Slot / stock geometry helpers (used by per-pair log when build_logs=True)
# =============================================================================

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
        "length_m":  float(length_m),
        "width_mm":  float(width_mm),
        "depth_mm":  float(depth_mm),
        "area_m2":   float(width_mm * depth_mm / 1_000_000.0),
    }


def _stock_geometry(stock_item: pd.Series) -> dict[str, float]:
    length_mm = float(stock_item["Length_Resolved"])
    width_mm  = float(stock_item["Width_Resolved"])
    depth_mm  = float(stock_item["Depth_Resolved"])
    return {
        "length_m": length_mm / 1000.0,
        "width_mm": width_mm,
        "depth_mm": depth_mm,
        "area_m2":  float(width_mm * depth_mm / 1_000_000.0),
    }


# =============================================================================
# Stock preparation
# =============================================================================

def prepare_stock_cost_inputs(df_stock_raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise stock data for cost calculation."""
    df_stock = df_stock_raw.copy()

    member_id_col = _pick_column(df_stock, _STOCK_ID_CANDIDATES)
    length_col    = _pick_column(df_stock, _LENGTH_CANDIDATES)
    width_col     = _pick_column(df_stock, _WIDTH_CANDIDATES)
    depth_col     = _pick_column(df_stock, _DEPTH_CANDIDATES)
    density_col   = _pick_column(df_stock, _DENSITY_CANDIDATES)
    distance_col  = _pick_column(df_stock, _DISTANCE_CANDIDATES)
    factor_col    = _pick_column(df_stock, _TRANSPORT_FACTOR_CANDIDATES)

    df_stock["Member_ID"]              = df_stock[member_id_col].astype(str)
    df_stock["Length_Resolved"]        = _coerce_numeric(df_stock[length_col],   label=length_col)
    df_stock["Width_Resolved"]         = _coerce_numeric(df_stock[width_col],    label=width_col)
    df_stock["Depth_Resolved"]         = _coerce_numeric(df_stock[depth_col],    label=depth_col)
    df_stock["Density_Resolved"]       = _coerce_numeric(df_stock[density_col],  label=density_col)
    df_stock["Distance_Resolved"]      = _coerce_numeric(df_stock[distance_col], label=distance_col)
    df_stock["TransportFactor_Resolved"] = _coerce_numeric(df_stock[factor_col], label=factor_col)
    df_stock["State_Resolved"]         = _resolve_state(df_stock, df_stock["Member_ID"])
    df_stock["Stock_Area_m2"]          = (
        df_stock["Width_Resolved"] * df_stock["Depth_Resolved"]
    ) / 1_000_000.0
    df_stock["Stock_Volume_m3"]        = (
        df_stock["Stock_Area_m2"] * (df_stock["Length_Resolved"] / 1000.0)
    )

    if df_stock["Member_ID"].isna().any() or \
       (df_stock["Member_ID"].astype(str).str.strip() == "").any():
        raise ValueError("Stock table contains empty Member_ID values.")

    return df_stock


# =============================================================================
# Vectorised cost calculation (core)
# =============================================================================

def _calculate_costs_vectorised(
    feasible_i:   np.ndarray,   # [n_feasible] slot indices
    feasible_j:   np.ndarray,   # [n_feasible] stock indices
    df_slots:     pd.DataFrame,
    stock:        pd.DataFrame,
) -> np.ndarray:
    """
    Compute LCA costs for all feasible (slot, stock) pairs in one NumPy pass.

    Replaces the Python loop over calculate_cost_formula() calls.
    ~50-100× faster than the per-pair loop for a typical 120 × 506 matrix.

    Returns
    -------
    total_costs : np.ndarray [n_feasible], float
        LCA cost per feasible pair (kg CO2e or equivalent).
    """
    slot_rows  = df_slots.iloc[feasible_i]
    stock_rows = stock.iloc[feasible_j]

    # Slot geometry
    req_length_m = slot_rows["length_m"].values.astype(float)
    req_width_mm = slot_rows["Width_Req"].values.astype(float)
    req_depth_mm = slot_rows["Depth_Req"].values.astype(float)
    req_area_m2  = (req_width_mm * req_depth_mm) / 1_000_000.0

    # Stock geometry
    stk_length_m = stock_rows["Length_Resolved"].values.astype(float) / 1000.0
    stk_area_m2  = (
        stock_rows["Width_Resolved"].values.astype(float) *
        stock_rows["Depth_Resolved"].values.astype(float)
    ) / 1_000_000.0

    # Material & transport properties
    density      = stock_rows["Density_Resolved"].values.astype(float)
    distance_km  = stock_rows["Distance_Resolved"].values.astype(float)
    trans_factor = stock_rows["TransportFactor_Resolved"].values.astype(float) / 1000.0
    state        = stock_rows["State_Resolved"].values.astype(float)
    is_reclaimed = state >= 0.5

    # Volumes
    v_req   = req_area_m2  * req_length_m
    v_stock = stk_area_m2  * stk_length_m
    v_waste = np.maximum(0.0, stk_area_m2 * (stk_length_m - req_length_m))

    # Masses
    mass_req   = v_req   * density
    mass_stock = v_stock * density
    mass_waste = v_waste * density

    # LCA components
    e_embodied  = np.where(is_reclaimed, 0.0,        mass_req   * M_A1_A3)
    e_transport = np.where(is_reclaimed, mass_stock,  mass_req)  * distance_km * trans_factor
    e_recovered = np.where(is_reclaimed, mass_stock  * M_RECOVER,      0.0)
    e_prep      = np.where(is_reclaimed, mass_stock  * E_PREP_SAW,     0.0)
    e_waste     = np.where(is_reclaimed, mass_waste  * E_OFFCUT,       0.0)
    e_scarcity  = np.where(is_reclaimed, v_waste     * SCARCITY_PENALTY, 0.0)

    total_costs = np.where(
        is_reclaimed,
        e_recovered + e_transport + e_prep + e_waste + e_scarcity,
        e_embodied  + e_transport,
    )

    return total_costs


# =============================================================================
# Main — build_cost_matrix
# =============================================================================

def build_cost_matrix(
    df_slots:        pd.DataFrame,
    df_input_stock:  pd.DataFrame,
    feasibility_mask: np.ndarray,
    build_logs:      bool = False,
    **_: Any,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame | None]:
    """
    Build a feasibility-filtered LCA cost matrix.

    Parameters
    ----------
    df_slots : pd.DataFrame
        Slot table from c25_stage_feasibility.build_cost_filter() with columns:
        edge_id, length_m, Length_Req, Width_Req, Depth_Req.

    df_input_stock : pd.DataFrame
        Raw stock inventory (e.g. complete_timber.csv).

    feasibility_mask : np.ndarray bool [n_slots, n_stock]
        Boolean mask from c25_stage_feasibility.build_cost_filter().
        True  = feasible  → cost calculated.
        False = infeasible → cost set to inf (MILP will not select).

    build_logs : bool, default False
        Whether to build the per-pair detail log DataFrame.
        Set True only when inspecting a specific iteration — building 60,720
        log rows every GA iteration adds meaningful overhead.
        When False, the third return value is None.

    **_ : Any
        Silently absorbs unexpected keyword arguments for API compatibility.
        Note: unexpected arguments are ignored, not validated — pass carefully.

    Returns
    -------
    cost_matrix : np.ndarray float [n_slots, n_stock]
        LCA cost per slot/stock pair. inf where infeasible.

    stock : pd.DataFrame
        Prepared stock table (output of prepare_stock_cost_inputs).
        Contains resolved columns: Member_ID, Length_Resolved, etc.

    df_logs : pd.DataFrame | None
        Per-pair detail log if build_logs=True, else None.
        Columns: Slot_ID, Stock_ID, Feasible, Branch, Status, Total cost,
                 V_req_m3, V_waste_m3, V_stock_m3, Mass_req_kg, E_embodied,
                 E_recovered, E_transport, E_prep, E_waste, E_scarcity.
    """
    if "edge_id" not in df_slots.columns:
        raise ValueError("df_slots must contain an edge_id column.")

    stock    = prepare_stock_cost_inputs(df_input_stock)
    slot_ids = df_slots["edge_id"].astype(str).tolist()
    n_slots  = len(slot_ids)
    n_stock  = len(stock)

    if feasibility_mask.shape != (n_slots, n_stock):
        raise ValueError(
            f"feasibility_mask shape {feasibility_mask.shape} does not match "
            f"({n_slots}, {n_stock}) slots × stock."
        )

    # Initialise cost matrix to inf — infeasible entries stay at inf
    cost_matrix = np.full((n_slots, n_stock), np.inf, dtype=float)

    # ---- Vectorised cost calculation ----
    feasible_i, feasible_j = np.where(feasibility_mask)   # indices of feasible pairs
    n_feasible = len(feasible_i)

    if n_feasible > 0:
        total_costs = _calculate_costs_vectorised(
            feasible_i, feasible_j, df_slots, stock
        )
        cost_matrix[feasible_i, feasible_j] = total_costs

    # ---- Optional per-pair log ----
    df_logs = None
    if build_logs:
        df_logs = _build_logs(
            df_slots, stock, feasibility_mask,
            cost_matrix, slot_ids, n_slots, n_stock,
        )

    return cost_matrix, stock, df_logs


# =============================================================================
# Log builder (only called when build_logs=True)
# =============================================================================

def _build_logs(
    df_slots:        pd.DataFrame,
    stock:           pd.DataFrame,
    feasibility_mask: np.ndarray,
    cost_matrix:     np.ndarray,
    slot_ids:        list[str],
    n_slots:         int,
    n_stock:         int,
) -> pd.DataFrame:
    """
    Build a detailed per-pair log DataFrame.
    Only called when build_logs=True — not part of the hot GA loop.
    """
    stock_ids    = stock["Member_ID"].astype(str).tolist()
    feasible_i, feasible_j = np.where(feasibility_mask)
    infeasible_i, infeasible_j = np.where(~feasibility_mask)

    # ---- Feasible rows ----
    feasible_slot_rows  = df_slots.iloc[feasible_i]
    feasible_stock_rows = stock.iloc[feasible_j]
    feasible_costs      = cost_matrix[feasible_i, feasible_j]

    # Recompute components for log (vectorised)
    req_length_m = feasible_slot_rows["length_m"].values.astype(float)
    req_width_mm = feasible_slot_rows["Width_Req"].values.astype(float)
    req_depth_mm = feasible_slot_rows["Depth_Req"].values.astype(float)
    req_area_m2  = req_width_mm * req_depth_mm / 1_000_000.0

    stk_length_m = feasible_stock_rows["Length_Resolved"].values.astype(float) / 1000.0
    stk_area_m2  = (
        feasible_stock_rows["Width_Resolved"].values.astype(float) *
        feasible_stock_rows["Depth_Resolved"].values.astype(float)
    ) / 1_000_000.0

    density      = feasible_stock_rows["Density_Resolved"].values.astype(float)
    distance_km  = feasible_stock_rows["Distance_Resolved"].values.astype(float)
    trans_factor = feasible_stock_rows["TransportFactor_Resolved"].values.astype(float) / 1000.0
    state        = feasible_stock_rows["State_Resolved"].values.astype(float)
    is_reclaimed = state >= 0.5

    v_req   = req_area_m2  * req_length_m
    v_stock = stk_area_m2  * stk_length_m
    v_waste = np.maximum(0.0, stk_area_m2 * (stk_length_m - req_length_m))
    v_over  = np.maximum(0.0, stk_area_m2 * req_length_m - v_req)

    mass_req   = v_req   * density
    mass_stock = v_stock * density
    mass_waste = v_waste * density

    e_embodied  = np.where(is_reclaimed, 0.0,        mass_req   * M_A1_A3)
    e_transport = np.where(is_reclaimed, mass_stock,  mass_req)  * distance_km * trans_factor
    e_recovered = np.where(is_reclaimed, mass_stock  * M_RECOVER,      0.0)
    e_prep      = np.where(is_reclaimed, mass_stock  * E_PREP_SAW,     0.0)
    e_waste_v   = np.where(is_reclaimed, mass_waste  * E_OFFCUT,       0.0)
    e_scarcity  = np.where(is_reclaimed, v_waste     * SCARCITY_PENALTY, 0.0)

    df_feasible = pd.DataFrame({
        "Slot_ID":      [slot_ids[i] for i in feasible_i],
        "Stock_ID":     [stock_ids[j] for j in feasible_j],
        "Feasible":     True,
        "Status":       "feasible",
        "Branch":       np.where(is_reclaimed, "reclaimed", "new"),
        "Total cost":   feasible_costs,
        "V_req_m3":     v_req,
        "V_over_m3":    v_over,
        "V_waste_m3":   v_waste,
        "V_stock_m3":   v_stock,
        "Mass_req_kg":  mass_req,
        "Mass_stock_kg": mass_stock,
        "E_embodied":   e_embodied,
        "E_recovered":  e_recovered,
        "E_transport":  e_transport,
        "E_prep":       e_prep,
        "E_waste":      e_waste_v,
        "E_scarcity":   e_scarcity,
    })

    # ---- Infeasible rows ----
    df_infeasible = pd.DataFrame({
        "Slot_ID":      [slot_ids[i] for i in infeasible_i],
        "Stock_ID":     [stock_ids[j] for j in infeasible_j],
        "Feasible":     False,
        "Status":       "infeasible",
        "Branch":       [
            _resolve_stock_branch(stock.iloc[j]) for j in infeasible_j
        ],
        "Total cost":   np.inf,
        "V_req_m3":     np.nan,
        "V_over_m3":    np.nan,
        "V_waste_m3":   np.nan,
        "V_stock_m3":   np.nan,
        "Mass_req_kg":  np.nan,
        "Mass_stock_kg": np.nan,
        "E_embodied":   np.nan,
        "E_recovered":  np.nan,
        "E_transport":  np.nan,
        "E_prep":       np.nan,
        "E_waste":      np.nan,
        "E_scarcity":   np.nan,
    })

    # Combine and restore original row order (slot_i * n_stock + stock_j)
    df_logs = pd.concat([df_feasible, df_infeasible], ignore_index=True)
    return df_logs