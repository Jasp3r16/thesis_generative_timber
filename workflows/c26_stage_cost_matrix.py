from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import contextlib
import io
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from c26_cost_calculation import analyze_and_export_slot_logs, build_cost_matrix


def _build_threshold_sweep(
    cost_matrix: np.ndarray,
    df_utilization_matrix: pd.DataFrame,
    thresholds: Sequence[float],
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    utilization_values = df_utilization_matrix.to_numpy(dtype=float, copy=True)

    for threshold in thresholds:
        feasible_mask = np.isfinite(utilization_values) & (utilization_values <= float(threshold))
        valid_count = int(feasible_mask.sum())
        total_count = int(feasible_mask.size)
        finite_cost_mask = feasible_mask & np.isfinite(cost_matrix)
        if int(finite_cost_mask.sum()) > 0:
            mean_cost = float(np.nanmean(np.where(finite_cost_mask, cost_matrix, np.nan)))
        else:
            mean_cost = float("nan")

        rows.append(
            {
                "threshold": float(threshold),
                "valid": valid_count,
                "total": total_count,
                "ratio": float(valid_count / total_count) if total_count > 0 else 0.0,
                "mean_cost": mean_cost,
            }
        )

    return pd.DataFrame(rows)


def _format_cost_display(cost_matrix: np.ndarray, df_slots: pd.DataFrame, enriched_stock: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        cost_matrix,
        index=df_slots["edge_id"].astype(str).tolist(),
        columns=enriched_stock["Member_ID"].astype(str).tolist(),
    )


def run_cost_matrix_stage(
    df_slots: pd.DataFrame,
    df_input_stock: pd.DataFrame,
    df_utilization_matrix: pd.DataFrame | np.ndarray | None = None,
    utilization_threshold: float = 1.0,
    cost_formula_version: str = "v2",
    target_stock_ids: Sequence[str] | None = None,
    include_threshold_sweep: bool = False,
    utilization_threshold_sweep: Sequence[float] = (1.00, 1.25, 1.50),
    export_slot_analysis: bool = False,
    target_slot_for_analysis: str = "e24",
    export_dir: Path | None = None,
    quiet: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the c26 cost-matrix stage and return notebook-friendly artifacts."""
    if df_utilization_matrix is None:
        raise ValueError("df_utilization_matrix is required for c26.")

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            cost_matrix, enriched_stock, df_logs = build_cost_matrix(
                df_slots=df_slots,
                df_stock_raw=df_input_stock,
                df_utilization_matrix=df_utilization_matrix,
                max_utilization_threshold=float(utilization_threshold),
                target_stock_ids=target_stock_ids,
                cost_formula_version=cost_formula_version,
                **kwargs,
            )
    else:
        cost_matrix, enriched_stock, df_logs = build_cost_matrix(
            df_slots=df_slots,
            df_stock_raw=df_input_stock,
            df_utilization_matrix=df_utilization_matrix,
            max_utilization_threshold=float(utilization_threshold),
            target_stock_ids=target_stock_ids,
            cost_formula_version=cost_formula_version,
            **kwargs,
        )

    df_cost_matrix_display = _format_cost_display(cost_matrix, df_slots, enriched_stock)

    df_threshold_sweep = None
    if include_threshold_sweep:
        if not isinstance(df_utilization_matrix, pd.DataFrame):
            df_utilization_matrix = pd.DataFrame(
                df_utilization_matrix,
                index=df_slots["edge_id"].astype(str).tolist(),
                columns=enriched_stock["Member_ID"].astype(str).tolist(),
            )
        df_threshold_sweep = _build_threshold_sweep(
            cost_matrix=cost_matrix,
            df_utilization_matrix=df_utilization_matrix,
            thresholds=utilization_threshold_sweep,
        )

    slot_analysis = None
    if export_slot_analysis:
        all_stock_ids = enriched_stock["Member_ID"].astype(str).tolist()
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                df_logs_slot, df_logs_slot_rs, analysis_export_path = analyze_and_export_slot_logs(
                    df_logs=df_logs,
                    target_slot_for_analysis=target_slot_for_analysis,
                    all_stock_ids=all_stock_ids,
                    export_dir=export_dir,
                    display_fn=None,
                    max_full_list_rows=None,
                    show_full_list=False,
                )
        else:
            df_logs_slot, df_logs_slot_rs, analysis_export_path = analyze_and_export_slot_logs(
                df_logs=df_logs,
                target_slot_for_analysis=target_slot_for_analysis,
                all_stock_ids=all_stock_ids,
                export_dir=export_dir,
                display_fn=None,
                max_full_list_rows=None,
                show_full_list=False,
            )

        slot_analysis = {
            "df_logs_slot": df_logs_slot,
            "df_logs_slot_rs": df_logs_slot_rs,
            "analysis_export_path": analysis_export_path,
        }

    finite_mask = np.isfinite(cost_matrix)
    valid_pairs = int(finite_mask.sum())
    total_pairs = int(cost_matrix.size)
    pruned_pairs = int(total_pairs - valid_pairs)
    utilization_mode = str(df_logs.attrs.get("utilization_mode", "c25_feasibility_matrix")) if hasattr(df_logs, "attrs") else "c25_feasibility_matrix"
    logged_formula_version = str(df_logs.attrs.get("cost_formula_version", cost_formula_version)) if hasattr(df_logs, "attrs") else str(cost_formula_version)

    return {
        "cost_matrix": cost_matrix,
        "enriched_stock": enriched_stock,
        "df_logs": df_logs,
        "df_cost_matrix_display": df_cost_matrix_display,
        "df_threshold_sweep": df_threshold_sweep,
        "slot_analysis": slot_analysis,
        "summary": {
            "slots": int(cost_matrix.shape[0]),
            "stock_items": int(cost_matrix.shape[1]),
            "valid_pairs": valid_pairs,
            "pruned_pairs": pruned_pairs,
            "total_pairs": total_pairs,
            "valid_ratio": float(valid_pairs / total_pairs) if total_pairs > 0 else 0.0,
            "utilization_mode": utilization_mode,
            "cost_formula_version": logged_formula_version,
            "utilization_threshold": float(utilization_threshold),
        },
    }
