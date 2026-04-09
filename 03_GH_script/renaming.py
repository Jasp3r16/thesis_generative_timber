"""Rename `v4_*.csv` files by appending a computed suffix before the extension.

Expected inputs:
- folder_path: folder containing the `v4_` CSV files
- sample_amount: sample count used for the `S` part of the suffix
- write: bool/button, when True performs the rename
- add_time: optional bool, when True appends time to the `D` stamp

The suffix is built as:
	_C{column_count}_S{sample_amount}_D{yyyymmdd}

Where:
- `C` = number of columns in the CSV file
- `S` = provided sample amount
- `D` = today's date in `YYYYMMDD` format
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


TARGET_FILENAMES = (
	"v4_edge.csv",
	"v4_global.csv",
	"v4_node.csv",
)


def _as_list(data):
	if data is None:
		return []
	if isinstance(data, (list, tuple)):
		return list(data)
	return [data]


def _scalar(value):
	values = _as_list(value)
	if not values:
		return None
	return values[0]


def _coerce_int(value):
	if value is None:
		return None
	try:
		return int(float(value))
	except Exception:
		return None


def _resolve_path(value):
	if value is None:
		return None
	text = str(value).strip()
	if not text:
		return None
	return Path(text).expanduser()


def _column_count(csv_path: Path) -> int:
	with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
		lines = [line for line in handle if line.strip()]

	if not lines:
		return 0

	sample = "".join(lines[:5])
	try:
		dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
		reader = csv.reader(lines, dialect)
	except Exception:
		reader = None

	if reader is not None:
		for row in reader:
			if row:
				return len(row)

	first_line = lines[0].strip()
	for delimiter in (",", ";", "\t", "|"):
		if delimiter in first_line:
			return len([part for part in first_line.split(delimiter)])

	return 1


def _build_suffix_with_time(column_count: int, sample_amount: int, add_time: bool) -> str:
	stamp = datetime.now().strftime("%Y%m%d_%H%M%S") if add_time else datetime.now().strftime("%Y%m%d")
	return "_C{}_S{}_D{}".format(column_count, sample_amount, stamp)


def rename_csv_files(folder_path: Path, sample_amount: int, add_time: bool = False, dry_run: bool = False):
	changes = []

	for name in TARGET_FILENAMES:
		source = folder_path / name
		if not source.exists() or not source.is_file():
			continue

		column_count = _column_count(source)
		suffix = _build_suffix_with_time(column_count, sample_amount, add_time)
		target = source.with_name(f"{source.stem}{suffix}{source.suffix}")

		if target.exists():
			changes.append((source, target, "skipped: target already exists"))
			continue

		changes.append((source, target, "dry-run" if dry_run else "renamed"))
		if not dry_run:
			source.rename(target)

	return changes


_in = globals()
folder_path = _resolve_path(_in.get("folder_path"))
sample_amount = _coerce_int(_scalar(_in.get("sample_amount")))
run_flag = bool(_in.get("write") if _in.get("write") is not None else _in.get("run"))
dry_run = bool(_in.get("dry_run", False))
add_time = bool(_in.get("add_time", False))

renamed_files = []
status = "Idle"

if not run_flag:
	status = "write=False, nothing renamed"

elif folder_path is None:
	status = "Missing folder_path"
elif not folder_path.exists():
	status = "Folder does not exist: {}".format(folder_path)
elif sample_amount is None:
	status = "Missing sample_amount"
else:
	changes = rename_csv_files(folder_path, sample_amount, add_time=add_time, dry_run=dry_run)
	renamed_files = ["{} -> {} ({})".format(src.name, dst.name, note) for src, dst, note in changes]

	if not changes:
		status = "No matching v4_ CSV files found"
	else:
		status = "Processed {} file(s)".format(len(changes))
