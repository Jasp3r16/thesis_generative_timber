"""Shared naming helpers for datasets, model artifacts, and run folders."""

from __future__ import annotations

from datetime import datetime


def _format_timestamp(dt: datetime | None = None, fmt: str = "%Y%m%d_%H%M%S") -> str:
    return (dt or datetime.now()).strftime(fmt)


def compact_timestamp(dt: datetime | None = None) -> str:
    """Return a sortable timestamp token like 20260402_124110."""
    return _format_timestamp(dt, "%Y%m%d_%H%M%S")


def human_timestamp(dt: datetime | None = None) -> str:
    """Return a human-readable timestamp token like 2026-04-02_124110."""
    return _format_timestamp(dt, "%Y-%m-%d_%H%M%S")


def format_learning_rate(learning_rate: float) -> str:
    """Format learning rate without scientific notation or trailing zeros."""
    return f"{learning_rate:.6f}".rstrip("0").rstrip(".")


def format_r2_score(r2_score: float | None) -> str:
    """Format R² with two decimals and a safe fallback."""
    if r2_score is None:
        return "NA"
    return f"{r2_score:.2f}"


def infer_dataset_version(dataset_filename: str) -> str:
    """Infer a dataset version token from legacy or strict filenames."""
    stem = dataset_filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].replace(".csv", "")

    if stem.startswith("v") and "_F" in stem and "_T" in stem and "_S" in stem:
        return stem.split("_", 1)[0][1:]

    if stem.startswith("data_"):
        return stem.split("data_", 1)[1]

    if stem.startswith("v"):
        return stem[1:]

    return stem


def build_dataset_label(
    dataset_version: str,
    feature_count: int,
    target_count: int,
    sample_count: int,
) -> str:
    """Build the strict raw-dataset label used for metadata and future exports."""
    return f"v{dataset_version}_F{feature_count}_T{target_count}_S{sample_count}"


def build_dataset_filename(
    dataset_version: str,
    feature_count: int,
    target_count: int,
    sample_count: int,
) -> str:
    """Build the strict raw-dataset filename."""
    return f"{build_dataset_label(dataset_version, feature_count, target_count, sample_count)}.csv"


def build_run_id(dt: datetime | None = None) -> str:
    """Build a unique run identifier suitable as a filename prefix."""
    return f"ID{compact_timestamp(dt)}"

def build_run_folder_name(
    run_id: str,
    feature_count: int | None = None,
) -> str:
    """Build the output folder name for a training/evaluation run."""
    folder_name = run_id
    if feature_count is not None:
        folder_name = f"{folder_name}_F{feature_count}"
    return folder_name


def build_evaluation_folder_name(base_name: str, feature_count: int | None = None) -> str:
    """Build the output folder name for exported surrogate evaluation data."""
    return build_run_folder_name(base_name, feature_count)


__all__ = [
    "build_dataset_filename",
    "build_dataset_label",
    "build_evaluation_folder_name",
    "build_run_folder_name",
    "build_run_id",
    "compact_timestamp",
    "format_learning_rate",
    "format_r2_score",
    "human_timestamp",
    "infer_dataset_version",
]