from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
import pandas as pd

import c00_headquarter_params as _params

# LCA impact constants (from headquarter params)

M_A1_A3          = float(_params.IMPACT_FACTOR_A1_A3)
M_RECOVER        = float(_params.IMPACT_FACTOR_RECOVERED_C1)
E_PREP           = float(_params.ENERGY_PREP_A5)
E_SAW            = float(_params.ENERGY_SAW_A5)
E_OFFCUT         = float(_params.ENERGY_OFFCUT_FACTOR_C3_C4)
WASTE_DIST_KM    = float(_params.WASTE_TRANSPORT_DIST_KM)
SCARCITY_PENALTY = float(_params.SCARCITY_PENALTY)

# TDUK 2026 sensitivity (thesis §6.4.2). When the env var GA_A1A3_PER_M3 is set,
# the new-timber A1–A3 embodied term switches from the per-MASS basis
# (mass_req × M_A1_A3, kg CO2e/kg) to a per-VOLUME basis (v_req × factor,
# kg CO2e/m³). The 2026 TDUK UK average is 47 kg CO2e/m³. Per-volume is the
# faithful unit for that figure (it decouples from per-element density), and the
# override leaves baseline behaviour unchanged when the var is unset.
_A1A3_PER_M3_OVERRIDE = os.environ.get("GA_A1A3_PER_M3")
_A1A3_PER_M3 = float(_A1A3_PER_M3_OVERRIDE) if _A1A3_PER_M3_OVERRIDE else None

# Column name candidates (case-insensitive lookup)

_SLOT_ID_CANDIDATES           = ("edge_id", "Edge_ID", "Slot_ID", "slot_id")
_STOCK_ID_CANDIDATES          = ("Member_ID", "member_id", "Stock_ID", "stock_id")
_LENGTH_CANDIDATES            = ("Length", "length_mm", "Length_mm")
_WIDTH_CANDIDATES             = ("Width", "width_mm", "Width_mm")
_DEPTH_CANDIDATES             = ("Depth", "depth_mm", "Depth_mm")
_STATE_CANDIDATES             = ("State", "state", "State_Resolved")
_DENSITY_CANDIDATES           = ("mean_density", "Mean_Density", "density", "Density")
_DISTANCE_CANDIDATES          = ("Transport_Dist", "transport_dist", "Distance", "distance_km")
_TRANSPORT_FACTOR_CANDIDATES  = ("EmissionFactor", "emission_factor", "TransportFactor", "Transport_Factor")

# Column helpers

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

# Stock preparation

def prepare_stock_cost_inputs(df_stock_raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise stock data for cost calculation.

    Call once before the GA loop and pass the result as prepared_stock to
    build_cost_matrix() to avoid repeating this work every iteration.
    """
    df_stock = df_stock_raw.copy()

    member_id_col = _pick_column(df_stock, _STOCK_ID_CANDIDATES)
    length_col    = _pick_column(df_stock, _LENGTH_CANDIDATES)
    width_col     = _pick_column(df_stock, _WIDTH_CANDIDATES)
    depth_col     = _pick_column(df_stock, _DEPTH_CANDIDATES)
    density_col   = _pick_column(df_stock, _DENSITY_CANDIDATES)
    distance_col  = _pick_column(df_stock, _DISTANCE_CANDIDATES)
    factor_col    = _pick_column(df_stock, _TRANSPORT_FACTOR_CANDIDATES)

    df_stock["Member_ID"]                = df_stock[member_id_col].astype(str)
    df_stock["Length_Resolved"]          = _coerce_numeric(df_stock[length_col],   label=length_col)
    df_stock["Width_Resolved"]           = _coerce_numeric(df_stock[width_col],    label=width_col)
    df_stock["Depth_Resolved"]           = _coerce_numeric(df_stock[depth_col],    label=depth_col)
    df_stock["Density_Resolved"]         = _coerce_numeric(df_stock[density_col],  label=density_col)
    df_stock["Distance_Resolved"]        = _coerce_numeric(df_stock[distance_col], label=distance_col)
    df_stock["TransportFactor_Resolved"] = _coerce_numeric(df_stock[factor_col],   label=factor_col)
    df_stock["State_Resolved"]           = _resolve_state(df_stock, df_stock["Member_ID"])
    df_stock["Stock_Area_m2"]            = (
        df_stock["Width_Resolved"] * df_stock["Depth_Resolved"]
    ) / 1_000_000.0
    df_stock["Stock_Volume_m3"]          = (
        df_stock["Stock_Area_m2"] * (df_stock["Length_Resolved"] / 1000.0)
    )

    if df_stock["Member_ID"].isna().any() or \
       (df_stock["Member_ID"].astype(str).str.strip() == "").any():
        raise ValueError("Stock table contains empty Member_ID values.")

    return df_stock

# LCA vector computation — single source of truth for cost matrix and logs

def _compute_lca_vectors(
    slot_rows:  pd.DataFrame,
    stock_rows: pd.DataFrame,
) -> dict:
    """Compute all LCA quantities for a set of aligned (slot, stock) pairs.

    Both DataFrames must have the same length; row i in slot_rows corresponds
    to row i in stock_rows. stock_rows must be a prepared stock table
    (output of prepare_stock_cost_inputs).

    Returns a dict of NumPy arrays, all of length n_pairs, covering
    individual LCA components and the summed total_costs.

    Transport basis:
        New stock:      required mass (bought cut-to-size; waste not shipped)
        Reclaimed stock: full stock mass (whole physical element moved to site)

    A5 split:
        e_prep — cleaning, de-nailing, structural testing; applied to all reclaimed elements
        e_saw  — secondary cross-cut resizing; applied only when stk_length > req_length

    Offcut waste split:
        e_waste_c2   — module C2: transport of offcut to waste facility (WASTE_DIST_KM)
        e_waste_c3c4 — modules C3+C4: disposal/incineration of offcut
        e_waste      — sum of C2 + C3+C4
    """
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
    # TransportFactor stored as kg CO2e per tonne-km → divide by 1000 for per-kg-km
    trans_factor = stock_rows["TransportFactor_Resolved"].values.astype(float) / 1000.0
    is_reclaimed = stock_rows["State_Resolved"].values.astype(float) >= 0.5

    # Volumes
    v_req        = req_area_m2 * req_length_m
    v_stock      = stk_area_m2 * stk_length_m
    v_waste      = np.maximum(0.0, stk_area_m2 * (stk_length_m - req_length_m))
    needs_sawing = v_waste > 0.0

    # Masses
    mass_req   = v_req   * density
    mass_stock = v_stock * density
    mass_waste = v_waste * density

    # LCA components
    # New-timber A1–A3: per-volume (TDUK 2026 override) or per-mass (baseline).
    if _A1A3_PER_M3 is not None:
        e_embodied = np.where(is_reclaimed, 0.0, v_req    * _A1A3_PER_M3)
    else:
        e_embodied = np.where(is_reclaimed, 0.0, mass_req * M_A1_A3)
    e_transport  = np.where(is_reclaimed, mass_stock, mass_req)  * distance_km * trans_factor
    e_recovered  = np.where(is_reclaimed, mass_stock * M_RECOVER, 0.0)
    # A5 prep: cleaning/de-nailing/testing — always applies to reclaimed elements
    e_prep       = np.where(is_reclaimed, mass_stock * E_PREP, 0.0)
    # A5 saw: cross-cut resizing — only when the stock element is longer than required
    e_saw        = np.where(is_reclaimed & needs_sawing, mass_stock * E_SAW, 0.0)
    # C2: transport of offcut to waste facility (fixed distance WASTE_DIST_KM)
    e_waste_c2   = np.where(is_reclaimed, mass_waste * WASTE_DIST_KM * trans_factor, 0.0)
    # C3+C4: disposal/incineration of offcut
    e_waste_c3c4 = np.where(is_reclaimed, mass_waste * E_OFFCUT, 0.0)
    e_waste      = e_waste_c2 + e_waste_c3c4
    e_scarcity   = np.where(is_reclaimed, v_waste * SCARCITY_PENALTY, 0.0)

    total_costs = np.where(
        is_reclaimed,
        e_recovered + e_transport + e_prep + e_saw + e_waste + e_scarcity,
        e_embodied  + e_transport,
    )

    return {
        "req_length_m": req_length_m,
        "req_area_m2":  req_area_m2,
        "stk_length_m": stk_length_m,
        "stk_area_m2":  stk_area_m2,
        "v_req":        v_req,
        "v_stock":      v_stock,
        "v_waste":      v_waste,
        "mass_req":     mass_req,
        "mass_stock":   mass_stock,
        "mass_waste":   mass_waste,
        "e_embodied":   e_embodied,
        "e_transport":  e_transport,
        "e_recovered":  e_recovered,
        "e_prep":       e_prep,
        "e_saw":        e_saw,
        "e_waste_c2":   e_waste_c2,
        "e_waste_c3c4": e_waste_c3c4,
        "e_waste":      e_waste,
        "e_scarcity":   e_scarcity,
        "is_reclaimed": is_reclaimed,
        "total_costs":  total_costs,
    }

# Vectorised cost calculation (core — hot GA path)

def _calculate_costs_vectorised(
    feasible_i: np.ndarray,
    feasible_j: np.ndarray,
    df_slots:   pd.DataFrame,
    stock:      pd.DataFrame,
) -> np.ndarray:
    """Return LCA total_costs for all feasible (slot, stock) pairs."""
    v = _compute_lca_vectors(df_slots.iloc[feasible_i], stock.iloc[feasible_j])
    return v["total_costs"]

# Main — build_cost_matrix

def build_cost_matrix(
    df_slots:         pd.DataFrame,
    df_input_stock:   pd.DataFrame,
    feasibility_mask: np.ndarray,
    build_logs:       bool = False,
    prepared_stock:   pd.DataFrame | None = None,
    **_: Any,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame | None]:
    """
    Build a feasibility-filtered LCA cost matrix.

    Parameters
    ----------
    df_slots : pd.DataFrame
        Slot table from c24_stage_feasibility.build_cost_filter() with columns:
        edge_id, length_m, Length_Req, Width_Req, Depth_Req.

    df_input_stock : pd.DataFrame
        Raw stock inventory (e.g. complete_timber.csv).
        Ignored when prepared_stock is provided.

    feasibility_mask : np.ndarray bool [n_slots, n_stock]
        Boolean mask from c24_stage_feasibility.build_cost_filter().
        True  = feasible  → cost calculated.
        False = infeasible → cost set to inf (MILP will not select).

    build_logs : bool, default False
        Whether to build the per-pair detail log DataFrame.
        Set True only when inspecting a specific iteration — building 60,720
        log rows every GA iteration adds meaningful overhead.
        When False, the third return value is None.

    prepared_stock : pd.DataFrame | None, default None
        Pre-prepared stock table (output of prepare_stock_cost_inputs).
        Pass this to avoid repeating stock preparation on every GA iteration.
        When None, prepare_stock_cost_inputs(df_input_stock) is called internally.

    **_ : Any
        Silently absorbs unexpected keyword arguments for API compatibility.

    Returns
    -------
    cost_matrix : np.ndarray float [n_slots, n_stock]
        LCA cost per slot/stock pair. inf where infeasible.

    stock : pd.DataFrame
        Prepared stock table. Cache and pass back as prepared_stock on
        subsequent calls to avoid re-preparation every GA iteration.

    df_logs : pd.DataFrame | None
        Per-pair detail log if build_logs=True, else None.
        Columns: Slot_ID, Stock_ID, Feasible, Status, Branch, Total cost,
                 V_req_m3, V_waste_m3, V_stock_m3, Mass_req_kg, Mass_stock_kg,
                 E_embodied, E_recovered, E_transport, E_prep, E_saw,
                 E_waste_C2, E_waste_C3C4, E_waste, E_scarcity.
    """
    if "edge_id" not in df_slots.columns:
        raise ValueError("df_slots must contain an edge_id column.")

    stock   = prepared_stock if prepared_stock is not None \
              else prepare_stock_cost_inputs(df_input_stock)
    slot_ids = df_slots["edge_id"].astype(str).tolist()
    n_slots  = len(slot_ids)
    n_stock  = len(stock)

    if feasibility_mask.shape != (n_slots, n_stock):
        raise ValueError(
            f"feasibility_mask shape {feasibility_mask.shape} does not match "
            f"({n_slots}, {n_stock}) slots × stock."
        )

    cost_matrix = np.full((n_slots, n_stock), np.inf, dtype=float)

    feasible_i, feasible_j = np.where(feasibility_mask)
    if len(feasible_i) > 0:
        cost_matrix[feasible_i, feasible_j] = _calculate_costs_vectorised(
            feasible_i, feasible_j, df_slots, stock
        )

    df_logs = None
    if build_logs:
        df_logs = _build_logs(df_slots, stock, feasibility_mask, slot_ids, n_slots, n_stock)

    return cost_matrix, stock, df_logs

# Log builder (only called when build_logs=True — not part of the hot GA loop)

def _build_logs(
    df_slots:         pd.DataFrame,
    stock:            pd.DataFrame,
    feasibility_mask: np.ndarray,
    slot_ids:         list[str],
    n_slots:          int,
    n_stock:          int,
) -> pd.DataFrame:
    stock_ids                  = stock["Member_ID"].astype(str).tolist()
    feasible_i,   feasible_j   = np.where(feasibility_mask)
    infeasible_i, infeasible_j = np.where(~feasibility_mask)

    # ---- Feasible rows — all components from shared LCA helper ----
    v = _compute_lca_vectors(df_slots.iloc[feasible_i], stock.iloc[feasible_j])

    df_feasible = pd.DataFrame({
        "Slot_ID":        [slot_ids[i] for i in feasible_i],
        "Stock_ID":       [stock_ids[j] for j in feasible_j],
        "Feasible":       True,
        "Status":         "feasible",
        "Branch":         np.where(v["is_reclaimed"], "reclaimed", "new"),
        "Total cost":     v["total_costs"],
        "V_req_m3":       v["v_req"],
        "V_waste_m3":     v["v_waste"],
        "V_stock_m3":     v["v_stock"],
        "Mass_req_kg":    v["mass_req"],
        "Mass_stock_kg":  v["mass_stock"],
        "E_embodied":     v["e_embodied"],
        "E_recovered":    v["e_recovered"],
        "E_transport":    v["e_transport"],
        "E_prep":         v["e_prep"],
        "E_saw":          v["e_saw"],
        "E_waste_C2":     v["e_waste_c2"],
        "E_waste_C3C4":   v["e_waste_c3c4"],
        "E_waste":        v["e_waste"],
        "E_scarcity":     v["e_scarcity"],
    })

    # ---- Infeasible rows — vectorised branch resolution ----
    inf_state = stock["State_Resolved"].values[infeasible_j]
    nan_col   = np.full(len(infeasible_i), np.nan)

    df_infeasible = pd.DataFrame({
        "Slot_ID":        [slot_ids[i] for i in infeasible_i],
        "Stock_ID":       [stock_ids[j] for j in infeasible_j],
        "Feasible":       False,
        "Status":         "infeasible",
        "Branch":         np.where(inf_state >= 0.5, "reclaimed", "new"),
        "Total cost":     np.inf,
        "V_req_m3":       nan_col,
        "V_waste_m3":     nan_col,
        "V_stock_m3":     nan_col,
        "Mass_req_kg":    nan_col,
        "Mass_stock_kg":  nan_col,
        "E_embodied":     nan_col,
        "E_recovered":    nan_col,
        "E_transport":    nan_col,
        "E_prep":         nan_col,
        "E_saw":          nan_col,
        "E_waste_C2":     nan_col,
        "E_waste_C3C4":   nan_col,
        "E_waste":        nan_col,
        "E_scarcity":     nan_col,
    })

    return pd.concat([df_feasible, df_infeasible], ignore_index=True)