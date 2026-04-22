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


def _validate_surrogate_context_for_cost_stage(surrogate_context: dict[str, Any] | None) -> None:
    """Validate strict surrogate-context requirements for candidate-level evaluation.

    The candidate-surrogate path requires real baseline edge areas for all edges.
    """
    if surrogate_context is None:
        return

    df_edges = surrogate_context.get("df_edges")
    if df_edges is None:
        raise ValueError("surrogate_context must include df_edges for candidate-level surrogate evaluation.")

    columns_by_lower = {str(col).strip().lower(): col for col in df_edges.columns}
    area_col = None
    for candidate in ("area", "cross_section_area", "a"):
        if candidate in columns_by_lower:
            area_col = columns_by_lower[candidate]
            break

    if area_col is None:
        raise ValueError(
            "df_edges must include a real baseline area column (Area/cross_section_area/A) "
            "when surrogate_context is provided."
        )

    area_values = pd.to_numeric(df_edges[area_col], errors="coerce")
    if area_values.isna().any():
        raise ValueError(
            "df_edges area column contains empty or non-numeric values; "
            "candidate-level surrogate evaluation requires complete real baseline areas."
        )

def _build_threshold_sweep(
    df_slots: pd.DataFrame,
    df_input_stock: pd.DataFrame,
    df_utilization_matrix: pd.DataFrame | None,
    thresholds: Sequence[float],
    quiet: bool,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    for thr in thresholds:
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                cm_s, _, _ = build_cost_matrix(
                    df_slots,
                    df_input_stock,
                    target_stock_ids=None,
                    df_utilization_matrix=df_utilization_matrix,
                    max_utilization_threshold=float(thr),
                )
        else:
            cm_s, _, _ = build_cost_matrix(
                df_slots,
                df_input_stock,
                target_stock_ids=None,
                df_utilization_matrix=df_utilization_matrix,
                max_utilization_threshold=float(thr),
            )

        finite_mask = np.isfinite(cm_s)
        valid_count = int(finite_mask.sum())
        total_count = int(cm_s.size)
        mean_cost = float(np.nanmean(np.where(finite_mask, cm_s, np.nan))) if valid_count > 0 else np.nan

        rows.append(
            {
                "threshold": float(thr),
                "valid": valid_count,
                "total": total_count,
                "ratio": float(round(valid_count / total_count if total_count > 0 else 0.0, 4)),
                "mean_cost": float(round(mean_cost, 2)) if np.isfinite(mean_cost) else np.nan,
            }
        )

    return pd.DataFrame(rows)

def run_cost_matrix_stage(
    df_slots: pd.DataFrame,
    df_input_stock: pd.DataFrame,
    df_utilization_matrix: pd.DataFrame | None = None,
    utilization_threshold: float = 1.25,
    target_stock_ids: Sequence[str] | None = None,
    export_cost_matrix_path: Path | None = None,
    include_threshold_sweep: bool = False,
    utilization_threshold_sweep: Sequence[float] = (1.00, 1.25, 1.50),
    export_slot_analysis: bool = False,
    target_slot_for_analysis: str = "e24",
    export_dir: Path | None = None,
    quiet: bool = True,
    surrogate_context: dict[str, Any] | None = None,
    require_structural_constraints: bool = True,
    require_surrogate_when_context: bool = True,
    weights: dict[str, float] | None = None,
    normalize: bool = False,
) -> dict[str, Any]:
    """Run cost-matrix stage and return matrix plus diagnostics."""
    _validate_surrogate_context_for_cost_stage(surrogate_context)

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            cost_matrix, enriched_stock, df_logs = build_cost_matrix(
                df_slots,
                df_input_stock,
                target_stock_ids=list(target_stock_ids) if target_stock_ids is not None else None,
                df_utilization_matrix=df_utilization_matrix,
                max_utilization_threshold=float(utilization_threshold),
                surrogate_context=surrogate_context,
                require_structural_constraints=bool(require_structural_constraints),
                require_surrogate_when_context=bool(require_surrogate_when_context),
                weights=weights,
                normalize=normalize,
            )
    else:
        cost_matrix, enriched_stock, df_logs = build_cost_matrix(
            df_slots,
            df_input_stock,
            target_stock_ids=list(target_stock_ids) if target_stock_ids is not None else None,
            df_utilization_matrix=df_utilization_matrix,
            max_utilization_threshold=float(utilization_threshold),
            surrogate_context=surrogate_context,
            require_structural_constraints=bool(require_structural_constraints),
            require_surrogate_when_context=bool(require_surrogate_when_context),
            weights=weights,
            normalize=normalize,
        )

    df_cost_matrix_display = pd.DataFrame(
        cost_matrix,
        index=[f"{row['edge_id']}" for _, row in df_slots.iterrows()],
        columns=enriched_stock["Member_ID"].tolist(),
    )

    if export_cost_matrix_path is not None:
        export_cost_matrix_path.parent.mkdir(parents=True, exist_ok=True)
        df_cost_matrix_display.to_csv(export_cost_matrix_path, index=True)

    df_threshold_sweep = None
    if include_threshold_sweep:
        df_threshold_sweep = _build_threshold_sweep(
            df_slots=df_slots,
            df_input_stock=df_input_stock,
            df_utilization_matrix=df_utilization_matrix,
            thresholds=utilization_threshold_sweep,
            quiet=quiet,
        )

    slot_analysis = None
    if export_slot_analysis:
        if export_dir is None:
            raise ValueError("export_dir is required when export_slot_analysis=True")

        all_stock_ids = df_input_stock["Member_ID"].dropna().astype(str).tolist()
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
    utilization_mode = "unknown"
    if hasattr(df_logs, "attrs"):
        utilization_mode = str(df_logs.attrs.get("utilization_mode", "unknown"))

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
            "pruned_pairs": int(total_pairs - valid_pairs),
            "total_pairs": total_pairs,
            "valid_ratio": float(round(valid_pairs / total_pairs, 4)) if total_pairs > 0 else 0.0,
            "utilization_mode": utilization_mode,
            "utilization_threshold": float(utilization_threshold),
        },
    }
