from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pulp


def run_milp_stage(
    cost_matrix: np.ndarray,
    enriched_stock: pd.DataFrame,
    df_slots: pd.DataFrame,
    reclaimed_marker: str = "RS",
    new_marker: str = "NS",
    solver_msg: bool = False,
    raise_on_infeasible_slots: bool = True,
) -> dict[str, Any]:
    
    """Run MILP assignment stage for one cost matrix."""
    stock_items = enriched_stock["Member_ID"].astype(str).tolist()
    construction_slots = df_slots["edge_id"].astype(str).tolist()

    new_items = [item for item in stock_items if new_marker in item]
    reclaimed_items = [item for item in stock_items if reclaimed_marker in item]

    finite_positions = np.argwhere(np.isfinite(cost_matrix))
    valid_matches: list[tuple[str, str]] = []
    costs: dict[tuple[str, str], float] = {}
    slot_to_stocks: dict[str, list[str]] = {sid: [] for sid in construction_slots}
    stock_to_slots: dict[str, list[str]] = {sid: [] for sid in stock_items}

    for i, j in finite_positions:
        slot_id = construction_slots[int(i)]
        stock_id = stock_items[int(j)]
        match = (stock_id, slot_id)
        valid_matches.append(match)
        costs[match] = float(cost_matrix[int(i), int(j)])
        slot_to_stocks[slot_id].append(stock_id)
        stock_to_slots[stock_id].append(slot_id)

    infeasible_slots = [sid for sid, options in slot_to_stocks.items() if not options]
    if infeasible_slots:
        message = f"MILP aborted: {len(infeasible_slots)} infeasible slots"
        if raise_on_infeasible_slots:
            raise ValueError(message)
        return {
            "status": "InfeasibleByMatrix",
            "total_cost": float("inf"),
            "df_results": pd.DataFrame(columns=["edge_id", "assigned_timber", "CO2_Penalty"]),
            "infeasible_slots": infeasible_slots,
            "summary": {
                "slots": int(len(construction_slots)),
                "reclaimed_items": int(len(reclaimed_items)),
                "new_items": int(len(new_items)),
            },
        }

    problem = pulp.LpProblem("Timber_Matching", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("Match", valid_matches, lowBound=0, upBound=1, cat=pulp.LpBinary)
    problem += pulp.lpSum(x[m] * costs[m] for m in valid_matches)

    for slot_id, options in slot_to_stocks.items():
        problem += pulp.lpSum(x[(stock_id, slot_id)] for stock_id in options) == 1

    for stock_id in reclaimed_items:
        if stock_id in stock_to_slots:
            problem += pulp.lpSum(x[(stock_id, slot_id)] for slot_id in stock_to_slots[stock_id]) <= 1

    for stock_id in new_items:
        if stock_id in stock_to_slots:
            problem += pulp.lpSum(x[(stock_id, slot_id)] for slot_id in stock_to_slots[stock_id]) <= len(construction_slots)

    pulp.PULP_CBC_CMD(msg=solver_msg).solve(problem)
    status = pulp.LpStatus[problem.status]

    if status == "Optimal":
        total_cost = float(pulp.value(problem.objective))
        rows = [
            {
                "edge_id": slot_id,
                "assigned_timber": stock_id,
                "CO2_Penalty": round(costs[(stock_id, slot_id)], 2),
            }
            for stock_id, slot_id in valid_matches
            if x[(stock_id, slot_id)].varValue == 1
        ]
        df_results = pd.DataFrame(rows)
    else:
        total_cost = float("inf")
        df_results = pd.DataFrame(columns=["edge_id", "assigned_timber", "CO2_Penalty"])

    return {
        "status": status,
        "total_cost": total_cost,
        "df_results": df_results,
        "infeasible_slots": infeasible_slots,
        "valid_matches": valid_matches,
        "summary": {
            "slots": int(len(construction_slots)),
            "reclaimed_items": int(len(reclaimed_items)),
            "new_items": int(len(new_items)),
            "assignments": int(len(df_results)),
        },
    }
