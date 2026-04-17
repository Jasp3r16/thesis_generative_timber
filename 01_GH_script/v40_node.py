"""
GHPython component script: export node data to CSV during Colibri iteration.

Expected inputs:
- coords_list: list/tree of node points (Point3d-like objects)
- sample_id: unique sample identifier from Colibri (int/str)
- file_path: folder path OR full path to output CSV file
- file_name: CSV file name, for example "nodes.csv"
- write: bool, when True writes current sample
- reset: bool, when True clears dedup memory and recreates CSV file
- node_ids: optional list/tree of node ids (same length as coords_list)
- is_support_list: optional list/tree of 0/1 values per node
- load_z_list: optional scalar or list/tree of Fz loads per node (kN)
- load_x_list: optional scalar or list/tree of Fx loads (used only if ENABLE_FX_FY = True)
- load_y_list: optional scalar or list/tree of Fy loads (used only if ENABLE_FX_FY = True)
- write_header: optional bool, default True

Outputs:
- status: human-readable status message
- rows_written: number of rows written in this solve
"""

EXPECTED_INPUTS = (
	"coords_list",
	"sample_id",
	"file_path",
	"file_name",
	"write",
	"reset",
	"node_ids",
	"is_support_list",
	"load_x_list",
	"load_y_list",
	"load_z_list",
	"write_header",
)

# Toggle Fx/Fy export. When False, Fx and Fy are omitted from CSV.
ENABLE_FX_FY = False
# Number of decimals for exported coordinates. Set to None to disable rounding.
COORD_DECIMALS = 4
# Enable runtime debug prints/status details in Grasshopper.
DEBUG_LOG = True

import csv
import os
import re

try:
	import rhinoscriptsyntax as rs  # type: ignore
except ImportError:
	rs = None

try:
	import System  # type: ignore
except ImportError:
	System = None

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

	# Grasshopper DataTree-like object
	if hasattr(data, "BranchCount") and hasattr(data, "Branch"):
		out = []
		for i in range(data.BranchCount):
			out.extend(list(data.Branch(i)))
		return out

	# Already a Python iterable list/tuple
	if isinstance(data, (list, tuple)):
		return list(data)

	# Single item
	return [data]


def _point_xyz(pt):
	"""Extract xyz from Rhino/Grasshopper point-like values, dicts, tuples, or strings."""
	if pt is None:
		return (None, None, None)

	def _guid_to_xyz(guid_like):
		"""Resolve Rhino object GUIDs to point coordinates when possible."""
		# First try rhinoscriptsyntax coercion.
		if rs is not None:
			try:
				coerced = rs.coerce3dpoint(guid_like, raise_if_missing=False)
				if coerced is not None and hasattr(coerced, "X") and hasattr(coerced, "Y") and hasattr(coerced, "Z"):
					return (coerced.X, coerced.Y, coerced.Z)
			except Exception:
				pass

		# Fallback: find object in current Rhino document and inspect geometry.
		if hasattr(sc, "doc") and getattr(sc, "doc", None) is not None and hasattr(sc.doc, "Objects"):
			try:
				guid_value = guid_like
				if isinstance(guid_like, str) and System is not None:
					guid_value = System.Guid(guid_like)
				obj = sc.doc.Objects.FindId(guid_value)
				if obj is not None:
					geo = obj.Geometry
					if hasattr(geo, "Location"):
						loc = geo.Location
						if hasattr(loc, "X") and hasattr(loc, "Y") and hasattr(loc, "Z"):
							return (loc.X, loc.Y, loc.Z)
			except Exception:
				pass

		return (None, None, None)

	if hasattr(pt, "X") and hasattr(pt, "Y") and hasattr(pt, "Z"):
		return (pt.X, pt.Y, pt.Z)

	# Grasshopper may pass Rhino object references as GUIDs.
	if type(pt).__name__ == "Guid":
		return _guid_to_xyz(pt)

	# Some GH wrappers expose a Location Point3d.
	if hasattr(pt, "Location"):
		loc = pt.Location
		if hasattr(loc, "X") and hasattr(loc, "Y") and hasattr(loc, "Z"):
			return (loc.X, loc.Y, loc.Z)

	# Accept dictionary-like payloads from intermediate scripts.
	if isinstance(pt, dict):
		x = pt.get("x", pt.get("X"))
		y = pt.get("y", pt.get("Y"))
		z = pt.get("z", pt.get("Z"))
		if x is not None and y is not None and z is not None:
			try:
				return (float(x), float(y), float(z))
			except Exception:
				return (None, None, None)

	if isinstance(pt, (list, tuple)) and len(pt) >= 3:
		return (pt[0], pt[1], pt[2])

	# Accept GH panel-style strings like:
	# "0. {-3,-3,0.9375}" or "{-3,-3,0.9375}"
	if isinstance(pt, str):
		# Resolve GUID-like strings to point coordinates.
		if re.match(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$", pt.strip()):
			gx, gy, gz = _guid_to_xyz(pt.strip())
			if gx is not None and gy is not None and gz is not None:
				return (gx, gy, gz)

		# Prefer the content inside braces when present.
		match = re.search(r"\{([^{}]+)\}", pt)
		coord_text = match.group(1) if match else pt
		parts = [p.strip() for p in coord_text.split(",")]
		if len(parts) >= 3:
			try:
				return (float(parts[0]), float(parts[1]), float(parts[2]))
			except Exception:
				pass

		# Fallback for formats like "Point3d(1,2,3)" or indexed panel text.
		nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", pt)
		if len(nums) >= 3:
			try:
				return (float(nums[-3]), float(nums[-2]), float(nums[-1]))
			except Exception:
				return (None, None, None)

	return (None, None, None)


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


def _to_bool01(value):
	"""Convert common Grasshopper numeric/bool/string values to 0/1."""
	if isinstance(value, bool):
		return 1 if value else 0

	try:
		return 1 if float(value) != 0.0 else 0
	except Exception:
		pass

	if isinstance(value, str):
		v = value.strip().lower()
		if v in ("true", "yes", "y", "on"):
			return 1

	return 0


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


def _sticky_keys():
	# Namespaced by component guid so multiple components do not clash.
	comp_guid = "local"
	gh = globals().get("ghenv")
	if gh is not None and hasattr(gh, "Component"):
		comp_guid = str(gh.Component.InstanceGuid)
	return (
		"node_csv_written_samples_{}".format(comp_guid),
		"node_csv_header_written_{}".format(comp_guid),
	)


def _ensure_parent_dir(path):
	folder = os.path.dirname(path)
	if folder and not os.path.exists(folder):
		os.makedirs(folder)


def _resolve_csv_file(csv_path_value, file_name_value):
	"""Resolve output CSV from optional folder/full-path and optional file name."""
	path_val = str(csv_path_value).strip() if csv_path_value else ""
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


def _round_if_number(value, decimals=None):
	"""Round numeric values for cleaner CSV output; keep non-numeric values unchanged."""
	if decimals is None:
		return value
	try:
		return round(float(value), int(decimals))
	except Exception:
		return value


def _short_text(value, max_len=160):
	"""Short, safe string representation for debug output."""
	try:
		text = str(value)
	except Exception:
		text = "<unprintable>"
	if len(text) > max_len:
		return text[: max_len - 3] + "..."
	return text


rows_written = 0
status = "Idle"

_in = globals()
sample = _sample_scalar(_in.get("sample_id"))
node_list = _as_list(_in.get("coords_list"))
id_list = _as_list(_in.get("node_ids"))
support_list = _as_list(_in.get("is_support_list"))
has_support_input = _in.get("is_support_list") is not None and len(support_list) > 0
fx_input = _in.get("load_x_list")
fy_input = _in.get("load_y_list")
fz_input = _in.get("load_z_list")
fx_list = _as_list(fx_input)
fy_list = _as_list(fy_input)
fz_list = _as_list(fz_input)
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
	status = "run=False, nothing written"

elif not csv_file:
	status = "Missing output path: provide file_path and file_name"

elif sample is None:
	status = "Missing sample_id (required to avoid duplicate writes in Colibri)"

elif sample in written_samples:
	status = "Sample {} already written; skipped".format(sample)

elif not node_list:
	status = "No nodes received for sample {}".format(sample)

else:
	_ensure_parent_dir(csv_file)
	skipped_invalid_coords = 0
	invalid_examples = []

	file_exists = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0
	header_written = bool(sc.sticky.get(header_key, False))
	should_write_header = header_enabled and (not file_exists) and (not header_written)

	if DEBUG_LOG:
		print(
			"[v40_node] sample={} nodes={} coords_type={} first_item_type={}".format(
				sample,
				len(node_list),
				type(_in.get("coords_list")).__name__ if _in.get("coords_list") is not None else "None",
				type(node_list[0]).__name__ if node_list else "None",
			)
		)

	with open(csv_file, "a", newline="") as f:
		writer = csv.writer(f)

		if should_write_header:
			header = [
				"sample_id",
				"node_id",
				"x",
				"y",
				"z",
				"Fz",
			]
			if has_support_input:
				header[5:5] = ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"]
			if ENABLE_FX_FY:
				header.insert(-1, "Fx")
				header.insert(-1, "Fy")
			writer.writerow(header)
			sc.sticky[header_key] = True

		for i, node in enumerate(node_list):
			node_id = id_list[i] if i < len(id_list) else i
			x, y, z = _point_xyz(node)
			if x is None or y is None or z is None:
				skipped_invalid_coords += 1
				if len(invalid_examples) < 5:
					invalid_examples.append(
						"idx={} type={} value={}".format(
							i,
							type(node).__name__,
							_short_text(node),
						)
					)
				continue
			x = _round_if_number(x, COORD_DECIMALS)
			y = _round_if_number(y, COORD_DECIMALS)
			z = _round_if_number(z, COORD_DECIMALS)

			if has_support_input:
				support_val = _value_by_index_or_scalar(support_list, i, default_value=0)
				is_support = _to_bool01(support_val)

				if is_support == 1:
					tx, ty, tz = 1, 1, 1
					rx, ry, rz = 0, 0, 0
				else:
					tx, ty, tz = 0, 0, 0
					rx, ry, rz = 0, 0, 0

			fz = _value_by_index_or_scalar(fz_list, i, default_value=0.0)
			try:
				fz = float(fz)
			except Exception:
				fz = 0.0

			row = [
				sample,
				node_id,
				x,
				y,
				z,
				fz,
			]
			if has_support_input:
				row[5:5] = [tx, ty, tz, rx, ry, rz]

			if ENABLE_FX_FY:
				fx = _value_by_index_or_scalar(fx_list, i, default_value=0.0)
				fy = _value_by_index_or_scalar(fy_list, i, default_value=0.0)
				try:
					fx = float(fx)
				except Exception:
					fx = 0.0
				try:
					fy = float(fy)
				except Exception:
					fy = 0.0
				row.insert(-1, fx)
				row.insert(-1, fy)

			writer.writerow(row)
			rows_written += 1

	if rows_written > 0:
		written_samples.add(sample)
		sc.sticky[samples_key] = written_samples

	if rows_written == 0:
		status = "Wrote 0 node rows for sample {} (all {} coords failed parsing)".format(
			sample,
			skipped_invalid_coords,
		)
	else:
		status = "Wrote {} node rows for sample {} (skipped {} invalid coords)".format(
			rows_written,
			sample,
			skipped_invalid_coords,
		)

	if invalid_examples:
		detail = " | parse_fail_examples: " + " || ".join(invalid_examples)
		status = status + detail

	if DEBUG_LOG and invalid_examples:
		print("[v40_node] parse failures:")
		for item in invalid_examples:
			print("  - " + item)

