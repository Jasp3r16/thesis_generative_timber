from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from c28_normalization_bounds import compute_normalization_bounds


def run_normalization_bounds_stage(
    *,
    cost_matrix: np.ndarray,
    df_logs: pd.DataFrame,
    enriched_stock: pd.DataFrame,
    df_slots: pd.DataFrame,
    reclaimed_marker: str = "RS",
    new_marker: str = "NS",
    new_stock_max_uses: int | None = 1,
    solver_msg: bool = False,
    print_summary: bool = True,
) -> dict[str, Any]:
    """Notebook-facing wrapper for computing exact normalization bounds."""
    out = compute_normalization_bounds(
        cost_matrix=cost_matrix,
        df_logs=df_logs,
        enriched_stock=enriched_stock,
        df_slots=df_slots,
        reclaimed_marker=reclaimed_marker,
        new_marker=new_marker,
        new_stock_max_uses=new_stock_max_uses,
        solver_msg=solver_msg,
    )

    if print_summary:
        status = out.get("status", "unknown")
        constants = out.get("normalization_constants", {})
        print(f"Bounds status: {status}")
        print(
            "Normalization constants "
            f"C_max={constants.get('C_max')}, "
            f"R_max={constants.get('R_max')}, "
            f"W_max={constants.get('W_max')}"
        )

    return out
