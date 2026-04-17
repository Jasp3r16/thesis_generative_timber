"""
GHPython component script: export edge data to CSV during Colibri iteration.

Expected inputs:
- edge_pairs: list/tree of source-target pairs (for example [(0,1), (1,2)])
- sample_id: unique sample identifier from Colibri (int/str)
- file_path: folder path OR full path to output CSV file
- file_name: CSV file name, for example "edges.csv"
- write: bool, when True writes current sample
- reset: bool, when True clears dedup memory and recreates CSV file
- edge_id: optional list/tree of edge ids (same length as edge_pairs)
- length: optional list/tree of edge lengths L (same length as edge_pairs)
- axial_force: optional list/tree of axial force values per edge (Axial_Force)
- A_list: list/tree of cross-sectional areas per edge
- E_list: optional list/tree of Young's modulus per edge
- Iy_list: optional list/tree of Iy per edge
- Iz_list: optional list/tree of Iz per edge
- J_list: optional list/tree of J per edge
- write_header: optional bool, default True

Mechanical-property columns are only exported when their inputs are provided.

Outputs:
- status: human-readable status message
- rows_written: number of rows written in this solve
"""

EXPECTED_INPUTS = (
	"edge_pairs",
	"sample_id",
	"file_path",
	"file_name",
	"write",
	"reset",
	"edge_id",
	"length",
	"axial_force",
	"A_list",
	"E_list",
	"Iy_list",
	"Iz_list",
	"J_list",
	"write_header",
)

# Decimal precision for exported mechanical-property values.
# Small values below 0.0001 keep at least two significant digits.
MECH_PROP_DECIMALS = 4
# Decimal precision for exported E values.
E_DECIMALS = 0
# Decimal precision for exported Length and Axial_Force values.
# Set to None to disable rounding.
LENGTH_DECIMALS = 4
AXIAL_FORCE_DECIMALS = 4

import csv
import math
import os
import re

try:
	import scriptcontext as sc  # type: ignore
except ImportError:
	class _FallbackScriptContext(object):
		sticky = {}
	sc = _FallbackScriptContext()


def _as_list(data):
	"""Flatten common GH list/tree input shapes to a plain Python list."""
	if data is None:
		return []

	if hasattr(data, "BranchCount") and hasattr(data, "Branch"):
		out = []
		for i in range(data.BranchCount):
			out.extend(list(data.Branch(i)))
		return out

	if isinstance(data, (list, tuple)):
		return list(data)

	return [data]


def _sample_scalar(value):
	"""Reduce list/tree sample inputs to a stable scalar id string."""
	vals = _as_list(value)
	if not vals:
		return None

	v = vals[0]
	try:
		f = float(v)
		if f.is_integer():
			return str(int(f))
		return str(f)
	except Exception:
		return str(v)


def _to_index(value):
	"""Convert common source/target formats to integer node indices."""
	if value is None:
		return None

	if isinstance(value, str):
		v = value.strip().lower()
		if v.startswith("v"):
			v = v[1:]
		try:
			return int(float(v))
		except Exception:
			return None

	try:
		return int(float(value))
	except Exception:
		return None


def _parse_edge_pair(item):
	"""Parse edge pair from tuple/list/dict/string into (source, target)."""
	if item is None:
		return (None, None)

	if isinstance(item, dict):
		s = item.get("source", item.get("Source", item.get("V1")))
		t = item.get("target", item.get("Target", item.get("V2")))
		return (_to_index(s), _to_index(t))

	if isinstance(item, (list, tuple)) and len(item) >= 2:
		return (_to_index(item[0]), _to_index(item[1]))

	if isinstance(item, str):
		m = re.findall(r"-?\d+\.?\d*", item)
		if len(m) >= 2:
			return (_to_index(m[0]), _to_index(m[1]))

	return (None, None)


def _value_by_index_or_scalar(values, idx, default_value=0.0):
	"""Return indexed value when list-like, otherwise scalar fallback."""
	if isinstance(values, list):
		if not values:
			return default_value
		if len(values) == 1:
			return values[0]
		if idx < len(values):
			return values[idx]
		return default_value

	if values is None:
		return default_value

	return values


def _to_float(value, default_value=0.0):
	try:
		return float(value)
	except Exception:
		return default_value


def _round_if_number(value, decimals=None):
	"""Round numeric values for cleaner CSV output; keep non-numeric unchanged."""
	if decimals is None:
		return value
	try:
		return round(float(value), int(decimals))
	except Exception:
		return value


def _format_fixed_number(value, decimals=None, min_significant_digits=2):
	"""Format numeric values without scientific notation; keep non-numeric unchanged."""
	try:
		number = float(value)
	except Exception:
		return value

	if decimals is None:
		return format(number, "f")

	decimals = int(decimals)
	display_decimals = decimals

	if number != 0.0 and min_significant_digits is not None and abs(number) < 1.0:
		required_decimals = int(math.floor(-math.log10(abs(number)))) + int(min_significant_digits)
		display_decimals = max(display_decimals, required_decimals)

	return format(number, ".{}f".format(display_decimals))


def _sticky_keys():
	# Namespaced by component guid so multiple components do not clash.
	comp_guid = "local"
	gh = globals().get("ghenv")
	if gh is not None and hasattr(gh, "Component"):
		comp_guid = str(gh.Component.InstanceGuid)
	return (
		"edge_csv_written_samples_{}".format(comp_guid),
		"edge_csv_header_written_{}".format(comp_guid),
	)


def _ensure_parent_dir(path):
	folder = os.path.dirname(path)
	if folder and not os.path.exists(folder):
		os.makedirs(folder)


def _resolve_csv_file(file_path_value, file_name_value):
	"""Resolve output CSV from optional folder/full-path and optional file name."""
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


rows_written = 0
status = "Idle"

_in = globals()
sample = _sample_scalar(_in.get("sample_id"))
edge_pair_list = _as_list(_in.get("edge_pairs"))
edge_id_list = _as_list(_in.get("edge_id"))
length_list = _as_list(_in.get("length"))
axial_list = _as_list(_in.get("axial_force"))

A_list = _as_list(_in.get("A_list"))
E_list = _as_list(_in.get("E_list"))
Iy_list = _as_list(_in.get("Iy_list"))
Iz_list = _as_list(_in.get("Iz_list"))
J_list = _as_list(_in.get("J_list"))

has_E_input = _in.get("E_list") is not None and len(E_list) > 0
has_Iy_input = _in.get("Iy_list") is not None and len(Iy_list) > 0
has_Iz_input = _in.get("Iz_list") is not None and len(Iz_list) > 0
has_J_input = _in.get("J_list") is not None and len(J_list) > 0

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

elif not edge_pair_list:
	status = "No edge_pairs received for sample {}".format(sample)

else:
	_ensure_parent_dir(csv_file)

	file_exists = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0
	header_written = bool(sc.sticky.get(header_key, False))
	should_write_header = header_enabled and (not file_exists) and (not header_written)

	with open(csv_file, "a", newline="") as f:
		writer = csv.writer(f)

		if should_write_header:
			header = [
				"Sample_ID",
				"Edge_ID",
				"Source",
				"Target",
				"Area",
				"Length",
				"Axial_Force",
			]

			insert_idx = header.index("Axial_Force")
			if has_E_input:
				header.insert(insert_idx, "E")
				insert_idx += 1
			if has_Iy_input:
				header.insert(insert_idx, "Iy")
				insert_idx += 1
			if has_Iz_input:
				header.insert(insert_idx, "Iz")
				insert_idx += 1
			if has_J_input:
				header.insert(insert_idx, "J")
				insert_idx += 1
			if has_E_input:
				header.insert(insert_idx, "EA/L")

			writer.writerow(header)
			sc.sticky[header_key] = True

		for i, edge_pair in enumerate(edge_pair_list):
			source, target = _parse_edge_pair(edge_pair)
			if source is None or target is None:
				continue

			edge_id = _value_by_index_or_scalar(edge_id_list, i, default_value=i)
			A_value = _to_float(_value_by_index_or_scalar(A_list, i, default_value=0.0), default_value=0.0)
			L_value = _to_float(_value_by_index_or_scalar(length_list, i, default_value=0.0), default_value=0.0)
			E_value = _to_float(_value_by_index_or_scalar(E_list, i, default_value=0.0), default_value=0.0)
			axial_force_value = _to_float(_value_by_index_or_scalar(axial_list, i, default_value=0.0), default_value=0.0)
			L_display_value = _round_if_number(L_value, LENGTH_DECIMALS)
			axial_force_display_value = _round_if_number(axial_force_value, AXIAL_FORCE_DECIMALS)
			ea_over_l = 0.0
			if L_value != 0.0:
				ea_over_l = (E_value * A_value) / L_value
			A = _format_fixed_number(A_value, MECH_PROP_DECIMALS)
			E = _format_fixed_number(E_value, E_DECIMALS, min_significant_digits=None)
			L = _format_fixed_number(L_display_value, LENGTH_DECIMALS, min_significant_digits=None)
			axial_force_value = _format_fixed_number(axial_force_display_value, AXIAL_FORCE_DECIMALS, min_significant_digits=None)
			ea_over_l = _format_fixed_number(ea_over_l, MECH_PROP_DECIMALS)
			Iy = _to_float(_value_by_index_or_scalar(Iy_list, i, default_value=0.0), default_value=0.0)
			Iz = _to_float(_value_by_index_or_scalar(Iz_list, i, default_value=0.0), default_value=0.0)
			J = _to_float(_value_by_index_or_scalar(J_list, i, default_value=0.0), default_value=0.0)
			Iy = _format_fixed_number(Iy, MECH_PROP_DECIMALS)
			Iz = _format_fixed_number(Iz, MECH_PROP_DECIMALS)
			J = _format_fixed_number(J, MECH_PROP_DECIMALS)

			row = [
				sample,
				edge_id,
				source,
				target,
				A,
				L,
				axial_force_value,
			]

			insert_idx = len(row) - 1
			if has_E_input:
				row.insert(insert_idx, E)
				insert_idx += 1
			if has_Iy_input:
				row.insert(insert_idx, Iy)
				insert_idx += 1
			if has_Iz_input:
				row.insert(insert_idx, Iz)
				insert_idx += 1
			if has_J_input:
				row.insert(insert_idx, J)
				insert_idx += 1
			if has_E_input:
				row.insert(insert_idx, ea_over_l)

			writer.writerow(row)

			rows_written += 1

	written_samples.add(sample)
	sc.sticky[samples_key] = written_samples
	status = "Wrote {} edge rows for sample {}".format(rows_written, sample)

