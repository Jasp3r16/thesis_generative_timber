"""Compute normalization upper bounds with auxiliary MILP solves.

This module computes design-specific maxima for:
- MILP assignment cost (C_max)
- assignment waste (W_max)
- achievable reclaimed reuse rate (R_max)

It reuses the assignment constraints from the c27 stage so bounds are
compatible with the same feasible search space.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pulp


def _resolve_stock_state(enriched_stock: pd.DataFrame) -> pd.Series:
    columns_by_lower = {str(col).strip().lower(): col for col in enriched_stock.columns}
    for candidate in ("state_resolved", "state"):
        column = columns_by_lower.get(candidate)
        if column is not None:
            values = pd.to_numeric(enriched_stock[column], errors="coerce")
            if not values.isna().any():
                return values.clip(lower=0.0, upper=1.0).astype(int)

    member_ids = enriched_stock["Member_ID"].astype(str).str.strip().str.upper()
    return pd.Series(np.where(member_ids.str.startswith("RS"), 1, 0), index=enriched_stock.index, dtype=int)


def _identify_stock_groups(
    enriched_stock: pd.DataFrame,
    reclaimed_marker: str,
    new_marker: str,
) -> tuple[list[str], list[str]]:
    stock_items = enriched_stock["Member_ID"].astype(str).tolist()
    stock_state = _resolve_stock_state(enriched_stock)

    reclaimed_items = enriched_stock.loc[stock_state == 1, "Member_ID"].astype(str).tolist()
    new_items = enriched_stock.loc[stock_state == 0, "Member_ID"].astype(str).tolist()

    if not reclaimed_items:
        reclaimed_items = [item for item in stock_items if reclaimed_marker in item]
    if not new_items:
        new_items = [item for item in stock_items if new_marker in item]

    return reclaimed_items, new_items


def _extract_valid_matches(
    cost_matrix: np.ndarray,
    stock_items: list[str],
    construction_slots: list[str],
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], float]]:
    finite_positions = np.argwhere(np.isfinite(cost_matrix))
    valid_matches: list[tuple[str, str]] = []
    costs: dict[tuple[str, str], float] = {}

    for slot_idx, stock_idx in finite_positions:
        slot_id = construction_slots[int(slot_idx)]
        stock_id = stock_items[int(stock_idx)]
        key = (stock_id, slot_id)
        valid_matches.append(key)
        costs[key] = float(cost_matrix[int(slot_idx), int(stock_idx)])

    return valid_matches, costs


def _build_connectivity(
    valid_matches: list[tuple[str, str]],
    construction_slots: list[str],
    stock_items: list[str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    slot_to_stock = {
        slot_id: [stock_id for stock_id, matched_slot in valid_matches if matched_slot == slot_id]
        for slot_id in construction_slots
    }
    stock_to_slots = {
        stock_id: [slot_id for matched_stock, slot_id in valid_matches if matched_stock == stock_id]
        for stock_id in stock_items
    }
    return slot_to_stock, stock_to_slots


def _solve_extreme_assignment(
    *,
    objective_name: str,
    objective_coeffs: dict[tuple[str, str], float],
    valid_matches: list[tuple[str, str]],
    construction_slots: list[str],
    stock_items: list[str],
    reclaimed_items: list[str],
    new_items: list[str],
    slot_to_stock: dict[str, list[str]],
    stock_to_slots: dict[str, list[str]],
    maximize: bool,
    new_stock_max_uses: int | None,
    solver_msg: bool,
) -> dict[str, Any]:
    sense = pulp.LpMaximize if maximize else pulp.LpMinimize
    problem = pulp.LpProblem(objective_name, sense)

    x = pulp.LpVariable.dicts("x", valid_matches, lowBound=0, upBound=1, cat="Binary")

    problem += pulp.lpSum(objective_coeffs[(stock_id, slot_id)] * x[(stock_id, slot_id)] for stock_id, slot_id in valid_matches)

    for slot_id in construction_slots:
        options = slot_to_stock.get(slot_id, [])
        if options:
            problem += pulp.lpSum(x[(stock_id, slot_id)] for stock_id in options) == 1

    for stock_id in reclaimed_items:
        slots = stock_to_slots.get(stock_id, [])
        if slots:
            problem += pulp.lpSum(x[(stock_id, slot_id)] for slot_id in slots) <= 1

    for stock_id in new_items:
        slots = stock_to_slots.get(stock_id, [])
        if slots:
            if new_stock_max_uses is None:
                problem += pulp.lpSum(x[(stock_id, slot_id)] for slot_id in slots) <= len(construction_slots)
            else:
                problem += pulp.lpSum(x[(stock_id, slot_id)] for slot_id in slots) <= int(new_stock_max_uses)

    pulp.PULP_CBC_CMD(msg=solver_msg).solve(problem)
    status = pulp.LpStatus[problem.status]

    selected_pairs = [
        (stock_id, slot_id)
        for stock_id, slot_id in valid_matches
        if x[(stock_id, slot_id)].varValue == 1
    ]

    objective_value = float(pulp.value(problem.objective)) if status == "Optimal" else float("nan")

    return {
        "status": status,
        "objective_value": objective_value,
        "selected_pairs": selected_pairs,
    }


def _build_waste_coefficients(
    df_logs: pd.DataFrame,
    valid_matches: list[tuple[str, str]],
) -> tuple[dict[tuple[str, str], float], int]:
    required = {"Slot_ID", "Stock_ID", "V_waste_m3"}
    missing_cols = sorted(required - set(df_logs.columns))
    if missing_cols:
        raise ValueError(f"df_logs is missing required columns: {', '.join(missing_cols)}")

    df = df_logs.copy()
    if "Status" in df.columns:
        df = df[df["Status"].astype(str).str.lower() == "feasible"].copy()

    df["Slot_ID"] = df["Slot_ID"].astype(str)
    df["Stock_ID"] = df["Stock_ID"].astype(str)
    df["V_waste_m3"] = pd.to_numeric(df["V_waste_m3"], errors="coerce").fillna(0.0)

    lookup = {
        (stock_id, slot_id): float(waste)
        for slot_id, stock_id, waste in zip(df["Slot_ID"], df["Stock_ID"], df["V_waste_m3"])
    }

    waste_coeffs: dict[tuple[str, str], float] = {}
    missing_count = 0
    for key in valid_matches:
        if key in lookup:
            waste_coeffs[key] = lookup[key]
        else:
            waste_coeffs[key] = 0.0
            missing_count += 1

    return waste_coeffs, missing_count


def compute_normalization_bounds(
    *,
    cost_matrix: np.ndarray,
    df_logs: pd.DataFrame,
    enriched_stock: pd.DataFrame,
    df_slots: pd.DataFrame,
    reclaimed_marker: str = "RS",
    new_marker: str = "NS",
    new_stock_max_uses: int | None = 1,
    solver_msg: bool = False,
) -> dict[str, Any]:
    """Compute C_max, W_max, and achievable R_max using auxiliary MILP objectives."""
    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be 2D")

    stock_items = enriched_stock["Member_ID"].astype(str).tolist()
    construction_slots = df_slots["edge_id"].astype(str).tolist()

    if cost_matrix.shape != (len(construction_slots), len(stock_items)):
        raise ValueError(
            "cost_matrix shape does not match df_slots x enriched_stock dimensions: "
            f"{cost_matrix.shape} vs ({len(construction_slots)}, {len(stock_items)})"
        )

    reclaimed_items, new_items = _identify_stock_groups(
        enriched_stock,
        reclaimed_marker,
        new_marker,
    )

    valid_matches, cost_coeffs = _extract_valid_matches(
        cost_matrix,
        stock_items,
        construction_slots,
    )

    slot_to_stock, stock_to_slots = _build_connectivity(
        valid_matches,
        construction_slots,
        stock_items,
    )

    infeasible_slots = [slot_id for slot_id, options in slot_to_stock.items() if len(options) == 0]
    if infeasible_slots:
        return {
            "status": "Infeasible",
            "infeasible_slots": infeasible_slots,
            "normalization_constants": {
                "C_max": float("nan"),
                "R_max": float("nan"),
                "W_max": float("nan"),
            },
            "bounds": {},
            "metadata": {
                "slots": int(len(construction_slots)),
                "stock_items": int(len(stock_items)),
                "reclaimed_items": int(len(reclaimed_items)),
                "new_items": int(len(new_items)),
                "new_stock_max_uses": None if new_stock_max_uses is None else int(new_stock_max_uses),
                "valid_pairs": int(len(valid_matches)),
            },
        }

    waste_coeffs, missing_waste_pairs = _build_waste_coefficients(df_logs, valid_matches)

    max_cost = _solve_extreme_assignment(
        objective_name="maximize_cost",
        objective_coeffs=cost_coeffs,
        valid_matches=valid_matches,
        construction_slots=construction_slots,
        stock_items=stock_items,
        reclaimed_items=reclaimed_items,
        new_items=new_items,
        slot_to_stock=slot_to_stock,
        stock_to_slots=stock_to_slots,
        maximize=True,
        new_stock_max_uses=new_stock_max_uses,
        solver_msg=solver_msg,
    )

    max_waste = _solve_extreme_assignment(
        objective_name="maximize_waste",
        objective_coeffs=waste_coeffs,
        valid_matches=valid_matches,
        construction_slots=construction_slots,
        stock_items=stock_items,
        reclaimed_items=reclaimed_items,
        new_items=new_items,
        slot_to_stock=slot_to_stock,
        stock_to_slots=stock_to_slots,
        maximize=True,
        new_stock_max_uses=new_stock_max_uses,
        solver_msg=solver_msg,
    )

    reuse_coeffs = {
        (stock_id, slot_id): float(1.0 if stock_id in reclaimed_items else 0.0)
        for stock_id, slot_id in valid_matches
    }
    max_reuse_count = _solve_extreme_assignment(
        objective_name="maximize_reuse_count",
        objective_coeffs=reuse_coeffs,
        valid_matches=valid_matches,
        construction_slots=construction_slots,
        stock_items=stock_items,
        reclaimed_items=reclaimed_items,
        new_items=new_items,
        slot_to_stock=slot_to_stock,
        stock_to_slots=stock_to_slots,
        maximize=True,
        new_stock_max_uses=new_stock_max_uses,
        solver_msg=solver_msg,
    )

    slot_count = max(int(len(construction_slots)), 1)
    reuse_max_pct = float(max_reuse_count["objective_value"] / slot_count * 100.0) if max_reuse_count["status"] == "Optimal" else float("nan")

    c_max = float(max_cost["objective_value"]) if max_cost["status"] == "Optimal" else float("nan")
    w_max = float(max_waste["objective_value"]) if max_waste["status"] == "Optimal" else float("nan")

    return {
        "status": "Optimal" if all(result["status"] == "Optimal" for result in (max_cost, max_waste, max_reuse_count)) else "Partial",
        "normalization_constants": {
            "C_max": c_max,
            "R_max": reuse_max_pct,
            "W_max": w_max,
        },
        "bounds": {
            "max_cost": max_cost,
            "max_waste": max_waste,
            "max_reuse_count": max_reuse_count,
            "max_reuse_rate_pct": reuse_max_pct,
        },
        "metadata": {
            "slots": int(len(construction_slots)),
            "stock_items": int(len(stock_items)),
            "reclaimed_items": int(len(reclaimed_items)),
            "new_items": int(len(new_items)),
            "new_stock_max_uses": None if new_stock_max_uses is None else int(new_stock_max_uses),
            "valid_pairs": int(len(valid_matches)),
            "missing_waste_pairs_in_logs": int(missing_waste_pairs),
        },
    }
