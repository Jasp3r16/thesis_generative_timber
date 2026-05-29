from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pulp

# Helpers

def _resolve_stock_state(enriched_stock: pd.DataFrame) -> pd.Series:
    columns_by_lower = {str(col).strip().lower(): col for col in enriched_stock.columns}
    for candidate in ("state_resolved", "state"):
        column = columns_by_lower.get(candidate)
        if column is not None:
            values = pd.to_numeric(enriched_stock[column], errors="coerce")
            if not values.isna().any():
                return (values >= 0.5).astype(int)

    member_ids = enriched_stock["Member_ID"].astype(str).str.strip().str.upper()
    return pd.Series(
        np.where(member_ids.str.startswith("RS"), 1, 0),
        index=enriched_stock.index, dtype=int,
    )

def _identify_stock_groups(
    enriched_stock: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    stock_state     = _resolve_stock_state(enriched_stock)
    reclaimed_items = enriched_stock.loc[stock_state == 1, "Member_ID"].astype(str).tolist()
    new_items       = enriched_stock.loc[stock_state == 0, "Member_ID"].astype(str).tolist()
    return reclaimed_items, new_items

def _build_milp_assignment(
    df_results:   pd.DataFrame,
    df_slots:     pd.DataFrame,
    stock_df_raw: pd.DataFrame,
) -> np.ndarray:
    """
    Build a [n_slots] integer array mapping each slot to a stock row index.

    This is the format required by gnn_feasibility() in step 5. The array
    index corresponds to the slot position in df_slots; the value is the
    integer position of the assigned stock element in stock_df_raw.

    Slots with no assignment (should not occur after a successful MILP solve)
    are set to -1.

    Parameters
    ----------
    df_results   : MILP result DataFrame with columns edge_id, assigned_timber
    df_slots     : slot table with column edge_id (order defines slot positions)
    stock_df_raw : raw stock inventory (Member_ID order defines stock positions)

    Returns
    -------
    milp_assignment : np.ndarray int [n_slots]
        milp_assignment[i] = row index into stock_df_raw for slot i
    """
    n_slots = len(df_slots)

    slot_id_to_idx  = {
        str(eid): int(i)
        for i, eid in enumerate(df_slots["edge_id"].astype(str))
    }
    stock_id_to_idx = {
        str(mid): int(i)
        for i, mid in enumerate(stock_df_raw["Member_ID"].astype(str))
    }

    milp_assignment = np.full(n_slots, -1, dtype=int)
    for _, row in df_results.iterrows():
        slot_idx  = slot_id_to_idx.get(str(row["edge_id"]), -1)
        stock_idx = stock_id_to_idx.get(str(row["assigned_timber"]), -1)
        if slot_idx >= 0 and stock_idx >= 0:
            milp_assignment[slot_idx] = stock_idx

    unassigned = int((milp_assignment == -1).sum())
    if unassigned > 0:
        import warnings
        warnings.warn(
            f"_build_milp_assignment: {unassigned} slot(s) have no assignment "
            f"(milp_assignment[i] = -1). GNN step will fail for these slots.",
            stacklevel=2,
        )

    return milp_assignment

# Main

def run_milp_stage(
    cost_matrix:               np.ndarray,
    enriched_stock:            pd.DataFrame,
    df_slots:                  pd.DataFrame,
    stock_df_raw:              pd.DataFrame | None = None,
    reclaimed_marker:          str = "RS",
    new_marker:                str = "NS",
    new_stock_max_uses:        int | None = None,
    min_reuse_fraction:        float | None = None,
    solver_msg:                bool = False,
    solver_time_limit:         int = 30,
    raise_on_infeasible_slots: bool = True,
) -> dict[str, Any]:
    """
    Run MILP timber assignment for one cost matrix.

    Parameters
    ----------
    cost_matrix : np.ndarray float [n_slots, n_stock]
        LCA cost matrix from c25_stage_cost_matrix.build_cost_matrix().
        inf entries are infeasible and will not be selected.

    enriched_stock : pd.DataFrame
        Prepared stock table (output of prepare_stock_cost_inputs).
        Must contain Member_ID and State_Resolved columns.

    df_slots : pd.DataFrame
        Slot table with column edge_id.

    stock_df_raw : pd.DataFrame | None
        Raw stock inventory (complete_timber.csv).
        Required to build milp_assignment for the GNN step.
        If None, milp_assignment will not be included in the return dict.

    reclaimed_marker : str
        Prefix identifying reclaimed stock items. Default "RS".

    new_marker : str
        Prefix identifying new stock items. Default "NS".

    new_stock_max_uses : int | None
        Maximum times a new stock element can be assigned across all slots.
        None = unlimited (default) — new elements are orderable in any quantity.
        Set to 1 to treat new elements as single-use (same as reclaimed).
        Reclaimed elements are always limited to one use regardless of this setting.

    min_reuse_fraction : float | None
        Minimum fraction of slots that must be assigned a reclaimed (RS) element.
        0.30 means at least 30 % of slots must use RS stock.
        The target count is automatically capped by what is feasibly achievable:
          cap = min(slots_with_any_RS_option, n_reclaimed_pieces_available)
        so the constraint is never stronger than the stock pool allows.
        None = no minimum reuse constraint (default).

    solver_msg : bool
        Whether to print CBC solver output. Default False.

    solver_time_limit : int
        Maximum CBC solve time in seconds. Default 30.
        Timeout returns status "Not Solved"; the evaluator treats this as a
        penalty candidate.

    raise_on_infeasible_slots : bool
        If True, raise ValueError when any slot has no feasible stock.
        If False, return InfeasibleByMatrix result dict. Default True.

    Returns
    -------
    dict with keys:
        status             — str: "Optimal", "Infeasible", "InfeasibleByMatrix", etc.
        total_cost         — float: MILP objective value (inf if not optimal)
        df_results         — pd.DataFrame: edge_id, assigned_timber, CO2_Penalty
        milp_assignment    — np.ndarray int [n_slots] or None if stock_df_raw not provided
                             Row index into stock_df_raw per slot. Pass to gnn_feasibility().
        infeasible_slots   — list[str]: slot IDs with no feasible stock option
        infeasible_slot_count — int
        summary            — dict: slots, reclaimed_items, new_items, assignments, etc.
    """
    stock_items        = enriched_stock["Member_ID"].astype(str).tolist()
    construction_slots = df_slots["edge_id"].astype(str).tolist()

    reclaimed_items, new_items = _identify_stock_groups(enriched_stock)

    # ---- Build match lists from finite cost matrix entries ----
    finite_positions = np.argwhere(np.isfinite(cost_matrix))
    valid_matches:   list[tuple[str, str]] = []
    costs:           dict[tuple[str, str], float] = {}
    slot_to_stocks:  dict[str, list[str]] = {sid: [] for sid in construction_slots}
    stock_to_slots:  dict[str, list[str]] = {sid: [] for sid in stock_items}

    for i, j in finite_positions:
        slot_id  = construction_slots[int(i)]
        stock_id = stock_items[int(j)]
        match    = (stock_id, slot_id)
        valid_matches.append(match)
        costs[match] = float(cost_matrix[int(i), int(j)])
        slot_to_stocks[slot_id].append(stock_id)
        stock_to_slots[stock_id].append(slot_id)

    # ---- Early exit: slots with no feasible stock ----
    infeasible_slots = [sid for sid, opts in slot_to_stocks.items() if not opts]
    if infeasible_slots:
        message = (
            f"MILP aborted: {len(infeasible_slots)} slot(s) have no feasible "
            f"stock option: {infeasible_slots[:5]}{'...' if len(infeasible_slots) > 5 else ''}"
        )
        if raise_on_infeasible_slots:
            raise ValueError(message)
        return {
            "status":               "InfeasibleByMatrix",
            "total_cost":           float("inf"),
            "df_results":           pd.DataFrame(columns=["edge_id", "assigned_timber", "CO2_Penalty"]),
            "milp_assignment":      None,
            "infeasible_slots":     infeasible_slots,
            "infeasible_slot_count": int(len(infeasible_slots)),
            "summary": {
                "slots":              int(len(construction_slots)),
                "reclaimed_items":    int(len(reclaimed_items)),
                "new_items":          int(len(new_items)),
                "new_stock_max_uses": new_stock_max_uses,
                "min_reuse_fraction": min_reuse_fraction,
                "min_reuse_target":   0,
                "assignments":        0,
            },
        }

    # ---- Build PuLP problem ----
    problem = pulp.LpProblem("Timber_Matching", pulp.LpMinimize)
    x = pulp.LpVariable.dicts(
        "Match", valid_matches, lowBound=0, upBound=1, cat=pulp.LpBinary
    )

    # Objective: minimise total LCA cost
    problem += pulp.lpSum(x[m] * costs[m] for m in valid_matches)

    # Each slot must be assigned exactly one stock element
    for slot_id, options in slot_to_stocks.items():
        problem += pulp.lpSum(x[(stock_id, slot_id)] for stock_id in options) == 1

    # Reclaimed elements: each physical piece can only be used once
    for stock_id in reclaimed_items:
        if stock_id in stock_to_slots and stock_to_slots[stock_id]:
            problem += (
                pulp.lpSum(x[(stock_id, slot_id)] for slot_id in stock_to_slots[stock_id])
                <= 1
            )

    # New elements: add use-limit constraint only when explicitly restricted
    # new_stock_max_uses=None means unlimited — no constraint needed
    if new_stock_max_uses is not None:
        for stock_id in new_items:
            if stock_id in stock_to_slots and stock_to_slots[stock_id]:
                problem += (
                    pulp.lpSum(x[(stock_id, slot_id)] for slot_id in stock_to_slots[stock_id])
                    <= int(new_stock_max_uses)
                )

    # Minimum reclaimed reuse constraint — capped by what the stock pool allows.
    # Target = floor(fraction * n_slots), but never more than:
    #   • number of slots that have ≥1 feasible RS option (cost matrix determines this)
    #   • number of unique RS pieces available (each piece usable at most once)
    min_reuse_target = 0
    if min_reuse_fraction is not None and min_reuse_fraction > 0.0:
        reclaimed_set = set(reclaimed_items)
        n_slots_with_rs = sum(
            1 for slot_id in construction_slots
            if any(sid in reclaimed_set for sid in slot_to_stocks[slot_id])
        )
        max_feasible_rs = min(n_slots_with_rs, len(reclaimed_items))
        min_reuse_target = min(
            int(min_reuse_fraction * len(construction_slots)),
            max_feasible_rs,
        )
        if min_reuse_target > 0:
            problem += (
                pulp.lpSum(
                    x[(sid, slot_id)]
                    for sid in reclaimed_items
                    for slot_id in stock_to_slots.get(sid, [])
                ) >= min_reuse_target,
                "min_reuse_constraint",
            )

    # ---- Solve ----
    solver = pulp.PULP_CBC_CMD(
        msg=solver_msg,
        timeLimit=solver_time_limit,
        options=["RandomSeed 0"],
    )
    solver.solve(problem)
    status = pulp.LpStatus[problem.status]

    # ---- Extract results ----
    if status == "Optimal":
        total_cost = float(pulp.value(problem.objective))
        rows = [
            {
                "edge_id":         slot_id,
                "assigned_timber": stock_id,
                "CO2_Penalty":     float(costs[(stock_id, slot_id)]),
            }
            for stock_id, slot_id in valid_matches
            if x[(stock_id, slot_id)].varValue is not None
            and x[(stock_id, slot_id)].varValue > 0.5
        ]
        df_results = pd.DataFrame(rows)

    elif status in ("Infeasible", "Undefined", "Not Solved"):
        print(
            f"  [MILP] Solver returned '{status}' — "
            f"check stock pool coverage and new_stock_max_uses={new_stock_max_uses}."
        )
        total_cost = float("inf")
        df_results = pd.DataFrame(columns=["edge_id", "assigned_timber", "CO2_Penalty"])

    else:
        total_cost = float("inf")
        df_results = pd.DataFrame(columns=["edge_id", "assigned_timber", "CO2_Penalty"])

    # ---- Build milp_assignment for GNN step ----
    milp_assignment = None
    if stock_df_raw is not None and status == "Optimal" and not df_results.empty:
        milp_assignment = _build_milp_assignment(df_results, df_slots, stock_df_raw)

    return {
        "status":                status,
        "total_cost":            total_cost,
        "df_results":            df_results,
        "milp_assignment":       milp_assignment,
        "infeasible_slots":      infeasible_slots,
        "infeasible_slot_count": int(len(infeasible_slots)),
        "summary": {
            "slots":               int(len(construction_slots)),
            "reclaimed_items":     int(len(reclaimed_items)),
            "new_items":           int(len(new_items)),
            "new_stock_max_uses":  new_stock_max_uses,
            "min_reuse_fraction":  min_reuse_fraction,
            "min_reuse_target":    min_reuse_target,
            "assignments":         int(len(df_results)),
        },
    }
