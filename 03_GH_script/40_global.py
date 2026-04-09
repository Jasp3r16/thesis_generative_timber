"""
GHPython component script: export global graph parameters (U matrix) to CSV.

Expected inputs:
- sample_id: unique sample identifier from Colibri (int/str)
- file_path: folder path OR full path to output CSV file
- file_name: CSV file name, for example "global_u.csv"
- write: bool, when True writes current sample
- reset: bool, when True clears dedup memory and recreates CSV file
- write_header: optional bool, default True
- global_load_sum: precomputed Global_Load_Sum
- total_structural_volume: precomputed Total_Structural_Volume
- average_connectivity: precomputed Average_Connectivity

Outputs:
- status: human-readable status message
- row_written: 0 or 1
"""

EXPECTED_INPUTS = (
    "sample_id",
    "file_path",
    "file_name",
    "write",
    "reset",
    "write_header",
    "global_load_sum",
    "total_structural_volume",
    "average_connectivity",
)

# Decimal precision for exported global values. Set None to disable rounding.
U_DECIMALS = 6

import csv
import os

try:
    import scriptcontext as sc  # type: ignore
except ImportError:
    class _FallbackScriptContext(object):
        sticky = {}
    sc = _FallbackScriptContext()


def _to_float(value, default_value=0.0):
    try:
        return float(value)
    except Exception:
        return default_value


def _round_if_number(value, decimals=None):
    if decimals is None:
        return value
    try:
        return round(float(value), int(decimals))
    except Exception:
        return value


def _sample_scalar(value):
    if value is None:
        return None
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except Exception:
        return str(value)


def _resolve_csv_file(file_path_value, file_name_value):
    path_val = str(file_path_value).strip() if file_path_value else ""
    name_val = str(file_name_value).strip() if file_name_value else ""

    if name_val and not name_val.lower().endswith(".csv"):
        name_val += ".csv"

    if name_val:
        if path_val:
            if path_val.lower().endswith(".csv"):
                base_dir = os.path.dirname(path_val)
            else:
                base_dir = path_val
            return os.path.join(base_dir, name_val) if base_dir else name_val
        return name_val

    return path_val if path_val else None


def _ensure_parent_dir(path):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


def _sticky_keys():
    comp_guid = "local"
    gh = globals().get("ghenv")
    if gh is not None and hasattr(gh, "Component"):
        comp_guid = str(gh.Component.InstanceGuid)
    return (
        "global_u_written_samples_{}".format(comp_guid),
        "global_u_header_written_{}".format(comp_guid),
    )


row_written = 0
status = "Idle"

_in = globals()
sample = _sample_scalar(_in.get("sample_id"))

global_load_sum = _to_float(_in.get("global_load_sum"), 0.0)
total_structural_volume = _to_float(_in.get("total_structural_volume"), 0.0)
average_connectivity = _to_float(_in.get("average_connectivity"), 0.0)

global_load_sum = _round_if_number(global_load_sum, U_DECIMALS)
total_structural_volume = _round_if_number(total_structural_volume, U_DECIMALS)
average_connectivity = _round_if_number(average_connectivity, U_DECIMALS)

csv_file = _resolve_csv_file(_in.get("file_path"), _in.get("file_name"))
header_enabled = True if _in.get("write_header") is None else bool(_in.get("write_header"))
run_flag = bool(_in.get("write"))
reset_flag = bool(_in.get("reset"))

samples_key, header_key = _sticky_keys()
written_samples = sc.sticky.get(samples_key, set())

if reset_flag:
    written_samples = set()
    sc.sticky[samples_key] = written_samples
    sc.sticky[header_key] = False

    if csv_file:
        _ensure_parent_dir(csv_file)
        with open(csv_file, "w", newline="") as f:
            pass

if not run_flag:
    status = "write=False, nothing written"

elif not csv_file:
    status = "Missing output path: provide file_path and file_name"

elif sample is None:
    status = "Missing sample_id (required to avoid duplicate writes in Colibri)"

elif sample in written_samples:
    status = "Sample {} already written; skipped".format(sample)

else:
    _ensure_parent_dir(csv_file)
    file_exists = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0
    header_written = bool(sc.sticky.get(header_key, False))
    should_write_header = header_enabled and (not file_exists) and (not header_written)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)

        if should_write_header:
            writer.writerow([
                "sample_id",
                "Global_Load_Sum",
                "Total_Structural_Volume",
                "Average_Connectivity",
            ])
            sc.sticky[header_key] = True

        writer.writerow([
            sample,
            global_load_sum,
            total_structural_volume,
            average_connectivity,
        ])

    written_samples.add(sample)
    sc.sticky[samples_key] = written_samples
    row_written = 1
    status = "Wrote 1 global row for sample {}".format(sample)
