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

import c21_surrogate_io as surrogate_io
import c25_feasibility_check as feasibility_check

def _validate_feasibility_stage_notebook_inputs(
    df_input_stock: pd.DataFrame | None,
    df_vertices: pd.DataFrame | None,
) -> None:
    helper = getattr(feasibility_check, "validate_feasibility_stage_notebook_inputs", None)
    if callable(helper):
        helper(df_input_stock=df_input_stock, df_vertices=df_vertices)
        return

    missing: list[str] = []
    if df_input_stock is None:
        missing.append("df_input_stock")
    if df_vertices is None:
        missing.append("df_vertices")
    if missing:
        raise ValueError("Missing required feasibility inputs: " + ", ".join(missing))

def run_feasibility_stage():
    '''
    Runs the feasibility stage of the workflow, which includes, running the surrogate model to predict which combinations of stock/ slot are fialing with binary output. where 0 = safe , 1 = unsafe
    then using this together with length checking, to see which combinations of stock/slot are feasible, then returing a matrix of slot x stock where infeasible combinations have inf as a value and feasible combinations have no value
    this is done so the cost matrix formula in the next stage can easily skip infeasible combinations by checking for inf values.
    '''


