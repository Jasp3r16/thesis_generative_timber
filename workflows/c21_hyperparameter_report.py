#!/usr/bin/env python
"""
Aggregate c21 run manifests into a detailed hyperparameter report.

Outputs:
- CSV table with one row per run/hyperparameter combination.
- Markdown leaderboard and summary statistics.

Usage:
  python workflows/c21_hyperparameter_report.py
"""

from __future__ import annotations

import getpass
import json
import os
from pathlib import Path

import pandas as pd


def _configure_delftblue_default() -> None:
    # Ensure config.py resolves DelftBlue paths when script is run outside SLURM.
    os.environ.setdefault("DELFTBLUE_DATA_BASE", f"/scratch/{getpass.getuser()}")


def _load_manifests(manifest_dir: Path) -> list[dict]:
    records: list[dict] = []
    for manifest_path in sorted(manifest_dir.rglob("*_run_manifest.json")):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                run = json.load(f)
            run["manifest_file"] = str(manifest_path)
            records.append(run)
        except Exception as exc:
            print(f"Skipping invalid manifest {manifest_path.name}: {exc}")
    return records


def _build_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append(
            {
                "run_id": r.get("run_id"),
                "artifact_stem": r.get("artifact_stem"),
                "learning_rate": r.get("learning_rate"),
                "epochs": r.get("epochs"),
                "epochs_completed": r.get("epochs_completed"),
                "batch_size": r.get("batch_size"),
                "hidden_dim": r.get("hidden_dim"),
                "weight_decay": r.get("weight_decay"),
                "train_split_ratio": r.get("train_split_ratio"),
                "random_seed": r.get("random_seed"),
                "test_r2": r.get("test_r2", r.get("final_val_r2")),
                "train_r2": r.get("train_r2"),
                "test_rmse": r.get("test_rmse"),
                "train_rmse": r.get("train_rmse"),
                "test_mae": r.get("test_mae"),
                "train_mae": r.get("train_mae"),
                "best_train_loss": r.get("best_train_loss"),
                "training_time_seconds": r.get("training_time_seconds"),
                "node_csv": (r.get("dataset_sources") or {}).get("node"),
                "edge_csv": (r.get("dataset_sources") or {}).get("edge"),
                "global_csv": (r.get("dataset_sources") or {}).get("global"),
                "slurm_job_id": r.get("slurm_job_id"),
                "slurm_array_task_id": r.get("slurm_array_task_id"),
                "timestamp_utc": r.get("timestamp_utc"),
                "model_path": r.get("model_path"),
                "manifest_file": r.get("manifest_file"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    numeric_cols = [
        "learning_rate",
        "epochs",
        "epochs_completed",
        "batch_size",
        "hidden_dim",
        "weight_decay",
        "train_split_ratio",
        "random_seed",
        "test_r2",
        "train_r2",
        "test_rmse",
        "train_rmse",
        "test_mae",
        "train_mae",
        "best_train_loss",
        "training_time_seconds",
        "slurm_job_id",
        "slurm_array_task_id",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(by=["test_r2", "test_rmse"], ascending=[False, True], na_position="last")


def _write_markdown_summary(df: pd.DataFrame, out_md: Path) -> None:
    if df.empty:
        out_md.write_text("# C21 Hyperparameter Report\n\nNo run manifests were found.\n", encoding="utf-8")
        return

    top_n = min(20, len(df))
    top = df.head(top_n)

    lines = [
        "# C21 Hyperparameter Report",
        "",
        f"Total runs: {len(df)}",
        f"Best test R2: {df['test_r2'].max():.6f}",
        f"Median test R2: {df['test_r2'].median():.6f}",
        f"Best (lowest) test RMSE: {df['test_rmse'].min():.6f}",
        "",
        f"## Top {top_n} Runs",
        "",
        "| Rank | run_id | test_r2 | test_rmse | test_mae | lr | epochs | batch | hidden | weight_decay |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for idx, row in enumerate(top.itertuples(index=False), start=1):
        lines.append(
            "| "
            + f"{idx} | {row.run_id} | {row.test_r2:.6f} | {row.test_rmse:.6f} | {row.test_mae:.6f} | "
            + f"{row.learning_rate:g} | {int(row.epochs) if pd.notna(row.epochs) else ''} | "
            + f"{int(row.batch_size) if pd.notna(row.batch_size) else ''} | "
            + f"{int(row.hidden_dim) if pd.notna(row.hidden_dim) else ''} | {row.weight_decay:g} |"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _configure_delftblue_default()
    import config  # Imported after env var setup

    manifest_dir = Path(config.SM_EXPORT_PATH)
    out_csv = Path(config.EXPORT_PATH) / "c21_hyperparameter_report.csv"
    out_md = Path(config.EXPORT_PATH) / "c21_hyperparameter_report.md"

    records = _load_manifests(manifest_dir)
    df = _build_table(records)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    _write_markdown_summary(df, out_md)

    print(f"Manifests scanned: {len(records)}")
    print(f"CSV report: {out_csv}")
    print(f"Markdown summary: {out_md}")
    if not df.empty:
        best = df.iloc[0]
        print(
            "Best run: "
            + f"run_id={best['run_id']} test_r2={best['test_r2']:.6f} "
            + f"test_rmse={best['test_rmse']:.6f}"
        )


if __name__ == "__main__":
    main()
