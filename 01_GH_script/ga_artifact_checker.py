"""
GHPython component: select GA dataset by stock type and validate all three files.

Add these inputs in the GH component editor:
  stock_type        : int — 0 = new stock, 1 = stock A, 2 = stock B
  path_new_edges    : str — edges CSV for 'new' run
  path_new_vertices : str — vertices CSV for 'new' run
  path_new_material : str — material CSV for 'new' run
  path_A_edges      : str — edges CSV for run 'A'
  path_A_vertices   : str — vertices CSV for run 'A'
  path_A_material   : str — material CSV for run 'A'
  path_B_edges      : str — edges CSV for run 'B'
  path_B_vertices   : str — vertices CSV for run 'B'
  path_B_material   : str — material CSV for run 'B'

Outputs:
  ok           : bool — True only when every check passes
  report       : str  — human-readable summary of all checks
  out_edges    : str  — edges path for the chosen stock type
  out_vertices : str  — vertices path for the chosen stock type
  out_material : str  — material path for the chosen stock type
"""

import re
import csv
import os

# ---------------------------------------------------------------------------
# Expected column signatures for each file type
# ---------------------------------------------------------------------------
_EDGES_REQUIRED    = {"edge_id", "V1", "V2"}
_VERTICES_REQUIRED = {"vertex_index", "x", "y", "z"}
_MATERIAL_REQUIRED = {"Member_ID"}

# Matches e.g. "20260518_141615_RUN1_GEN250_EVAL7500_F-2_2315" or without RUN segment
_GA_ID_RUN = re.compile(r"(\d{8}_\d{6}(?:_RUN\d+)?_GEN\d+_EVAL\d+_F-?\d+_\d+)")

# Matches the date+time stamp alone, e.g. "20260518_141615"
_TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")

# Matches "GA_new_", "GA_A_", or "GA_B_" in a folder name
_STOCK_TYPE_RE = re.compile(r"GA_(new|A|B)_")

_STOCK_LABELS = {0: "new", 1: "A", 2: "B"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_ga_run_id(path):
    m = _GA_ID_RUN.search(path.replace("\\", "/"))
    return m.group(1) if m else None

def _extract_timestamp(path):
    m = _TIMESTAMP_RE.search(path.replace("\\", "/"))
    return m.group(1) if m else None


def _extract_stock_label(path):
    m = _STOCK_TYPE_RE.search(path.replace("\\", "/"))
    return m.group(1) if m else None


def _read_header(path):
    try:
        with open(path, "r") as fh:
            first_line = fh.readline()
        delim = ";" if first_line.count(";") > first_line.count(",") else ","
        reader = csv.reader([first_line], delimiter=delim)
        return set(next(reader))
    except Exception:
        return set()


def _check_columns(path, required, label):
    cols = _read_header(path)
    missing = required - cols
    if missing:
        return False, "{} column check FAILED — missing: {}".format(label, sorted(missing))
    return True, "{} column check OK ({})".format(label, sorted(required))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

lines        = []
passed       = []
inputs_valid = True

# --- Resolve stock type ---
_stock_int   = int(stock_type) if stock_type is not None else None  # noqa: F821
_stock_label = _STOCK_LABELS.get(_stock_int)

if _stock_label is None:
    lines.append("INPUT ERROR: stock_type must be 0, 1, or 2 (got {})".format(stock_type))  # noqa: F821
    inputs_valid = False

# --- Select paths for chosen stock type ---
if _stock_label == "new":
    path_edges    = path_new_edges      # noqa: F821
    path_vertices = path_new_vertices   # noqa: F821
    path_material = path_new_material   # noqa: F821
elif _stock_label == "A":
    path_edges    = path_A_edges        # noqa: F821
    path_vertices = path_A_vertices     # noqa: F821
    path_material = path_A_material     # noqa: F821
elif _stock_label == "B":
    path_edges    = path_B_edges        # noqa: F821
    path_vertices = path_B_vertices     # noqa: F821
    path_material = path_B_material     # noqa: F821
else:
    path_edges = path_vertices = path_material = None

# Guard against missing / not-found files
if inputs_valid:
    for _lbl, _val in [("path_edges", path_edges),
                       ("path_vertices", path_vertices),
                       ("path_material", path_material)]:
        if not _val:
            lines.append("INPUT MISSING: {} is not connected".format(_lbl))
            inputs_valid = False
        elif not os.path.isfile(_val):
            lines.append("FILE NOT FOUND: {}".format(_val))
            inputs_valid = False

if not inputs_valid:
    ok           = False
    out_edges    = None
    out_vertices = None
    out_material = None
    report       = "\n".join(lines)
else:
    # --- 1. Stock-type match check ---
    for _lbl, _val in [("edges", path_edges),
                       ("vertices", path_vertices),
                       ("material", path_material)]:
        found = _extract_stock_label(_val)
        if found is None:
            lines.append("Stock-type check FAILED for {} — GA_new/A/B not found in path".format(_lbl))
            passed.append(False)
        elif found == _stock_label:
            lines.append("Stock-type check OK for {} — matches '{}'".format(_lbl, _stock_label))
            passed.append(True)
        else:
            lines.append("Stock-type check FAILED for {} — expected '{}', found '{}' in path".format(
                _lbl, _stock_label, found))
            passed.append(False)

    # --- 2. Timestamp consistency check (date + time) ---
    ts_edges    = _extract_timestamp(path_edges)
    ts_vertices = _extract_timestamp(path_vertices)
    ts_material = _extract_timestamp(path_material)

    if None in (ts_edges, ts_vertices, ts_material):
        lines.append("Timestamp check FAILED — YYYYMMDD_HHMMSS not found in one or more paths")
        lines.append("  edges    : {}".format(ts_edges))
        lines.append("  vertices : {}".format(ts_vertices))
        lines.append("  material : {}".format(ts_material))
        passed.append(False)
    elif ts_edges == ts_vertices == ts_material:
        lines.append("Timestamp check OK — all share timestamp: {}".format(ts_edges))
        passed.append(True)
    else:
        lines.append("Timestamp check FAILED — timestamps do not match across files:")
        lines.append("  edges    : {}".format(ts_edges))
        lines.append("  vertices : {}".format(ts_vertices))
        lines.append("  material : {}".format(ts_material))
        passed.append(False)

    # --- 3. Same GA artifact ---
    id_edges    = _extract_ga_run_id(path_edges)
    id_vertices = _extract_ga_run_id(path_vertices)
    id_material = _extract_ga_run_id(path_material)

    if None in (id_edges, id_vertices, id_material):
        lines.append("GA ID check FAILED — could not parse GA run ID from one or more paths")
        lines.append("  edges    : {}".format(id_edges))
        lines.append("  vertices : {}".format(id_vertices))
        lines.append("  material : {}".format(id_material))
        passed.append(False)
    elif id_edges == id_vertices == id_material:
        lines.append("GA artifact check OK — all share run ID: {}".format(id_edges))
        passed.append(True)
    else:
        lines.append("GA artifact check FAILED — run IDs do not match:")
        lines.append("  edges    : {}".format(id_edges))
        lines.append("  vertices : {}".format(id_vertices))
        lines.append("  material : {}".format(id_material))
        passed.append(False)

    # --- 4. File-type column checks ---
    for (_path, _required, _lbl) in [
        (path_edges,    _EDGES_REQUIRED,    "edges"),
        (path_vertices, _VERTICES_REQUIRED, "vertices"),
        (path_material, _MATERIAL_REQUIRED, "material"),
    ]:
        ok_check, msg = _check_columns(_path, _required, _lbl)
        lines.append(msg)
        passed.append(ok_check)

    ok           = all(passed)
    out_edges    = path_edges    if ok else None
    out_vertices = path_vertices if ok else None
    out_material = path_material if ok else None
    status       = "ALL CHECKS PASSED" if ok else "CHECKS FAILED"
    report       = "Stock type: {} ({})\n{}\n{}".format(_stock_int, _stock_label, status, "\n".join(lines))