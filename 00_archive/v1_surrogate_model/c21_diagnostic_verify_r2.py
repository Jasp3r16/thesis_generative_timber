"""c21 diagnostics utilities and R2 legitimacy report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

import config
from c21_model_evaluation import compute_split_metrics, drop_non_finite_pairs
from src.c21_data_pipeline import load_v4_sources


def collect_preds_trues_original(loader, model, device, edge_target_scaler) -> tuple[np.ndarray, np.ndarray]:
    """Collect predictions/targets from a loader on original target scale.

    Applies edge_loss_mask when present so results match training/evaluation logic.
    Returns (preds_original, trues_original) as flat arrays.
    """
    model.eval()
    pred_batches: list[np.ndarray] = []
    true_batches: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            out = model(
                batch.x,
                batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                u=batch.u,
            )
            mask = getattr(batch, "edge_loss_mask", None)
            if mask is None:
                mask = torch.ones_like(batch.y_edge)
            keep = (mask.view(-1) > 0.5).detach().cpu()
            out_cpu = out.detach().cpu()
            y_cpu = batch.y_edge.detach().cpu()
            pred_batches.append(out_cpu[keep].numpy())
            true_batches.append(y_cpu[keep].numpy())

    preds_scaled = np.concatenate(pred_batches, axis=0)
    trues_scaled = np.concatenate(true_batches, axis=0)

    preds_original = edge_target_scaler.inverse_transform(preds_scaled).reshape(-1)
    trues_original = edge_target_scaler.inverse_transform(trues_scaled).reshape(-1)
    return preds_original, trues_original


def summarize_edge_masking(loader) -> dict[str, Any]:
    """Summarize masked/kept edges for a dataloader."""
    total_edges = 0
    total_masked = 0
    total_kept = 0
    first_batch: dict[str, int] | None = None

    for batch_idx, batch in enumerate(loader):
        mask = getattr(batch, "edge_loss_mask", None)
        if mask is None:
            continue

        mask_values = mask.view(-1)
        kept = int((mask_values > 0.5).sum().item())
        masked = int((mask_values <= 0.5).sum().item())

        total_edges += int(mask_values.numel())
        total_kept += kept
        total_masked += masked

        if batch_idx == 0:
            first_batch = {"kept": kept, "masked": masked, "total": kept + masked}

    if total_edges == 0:
        return {
            "has_mask": False,
            "total_edges": 0,
            "total_kept": 0,
            "total_masked": 0,
            "pct_kept": 0.0,
            "pct_masked": 0.0,
            "first_batch": first_batch,
        }

    return {
        "has_mask": True,
        "total_edges": total_edges,
        "total_kept": total_kept,
        "total_masked": total_masked,
        "pct_kept": 100.0 * total_kept / total_edges,
        "pct_masked": 100.0 * total_masked / total_edges,
        "first_batch": first_batch,
    }


def run_r2_legitimacy_diagnostic(
    node_csv: str = "v4_node_C12_S19999_D20260416.csv",
    edge_csv: str = "v4_edge_C12_S19999_D20260416.csv",
    global_csv: str = "v4_global_C4_S19999_D20260416.csv",
    reported_test_r2: float = 0.9933,
    reported_test_mae: float = 1.1415,
) -> dict[str, Any]:
    """Compute sanity checks that help interpret a very high R2."""
    node_path = config.GH_DATA_PATH / node_csv
    edge_path = config.GH_DATA_PATH / edge_csv
    global_path = config.GH_DATA_PATH / global_csv

    _, df_edge, _ = load_v4_sources(node_path, edge_path, global_path)

    force = df_edge["Axial_Force"].astype(float)
    y_true = force.to_numpy()
    y_pred_mean = np.full_like(y_true, fill_value=y_true.mean(), dtype=float)

    baseline_r2 = float(r2_score(y_true, y_pred_mean))
    baseline_mae = float(mean_absolute_error(y_true, y_pred_mean))

    edge_cols = ["Area", "Length", "E", "Iy", "Iz", "J", "EA/L"]
    available = [col for col in edge_cols if col in df_edge.columns]
    correlations = {
        col: float(df_edge[col].corr(df_edge["Axial_Force"])) for col in available
    }

    x_edge = df_edge[available].to_numpy(dtype=float)
    y_edge = df_edge["Axial_Force"].to_numpy(dtype=float)
    linear = LinearRegression().fit(x_edge, y_edge)
    linear_r2 = float(linear.score(x_edge, y_edge))

    n_samples = int(df_edge["Sample_ID"].nunique())
    n_edges = int(len(df_edge))

    std = float(force.std())
    rng = float(force.max() - force.min())

    return {
        "target": {
            "count": int(len(force)),
            "min": float(force.min()),
            "q1": float(force.quantile(0.25)),
            "median": float(force.quantile(0.50)),
            "q3": float(force.quantile(0.75)),
            "max": float(force.max()),
            "std": std,
            "range": rng,
        },
        "baseline": {
            "r2": baseline_r2,
            "mae": baseline_mae,
        },
        "linear_signal": {
            "correlations": correlations,
            "linear_r2": linear_r2,
        },
        "split": {
            "n_samples": n_samples,
            "n_edges": n_edges,
            "avg_edges_per_sample": float(n_edges / max(n_samples, 1)),
        },
        "reported": {
            "test_r2": float(reported_test_r2),
            "test_mae": float(reported_test_mae),
            "mae_over_std": float(reported_test_mae / std),
            "mae_pct_of_range": float(100.0 * reported_test_mae / rng),
        },
    }


def print_r2_legitimacy_report(summary: dict[str, Any]) -> None:
    """Pretty-print the legitimacy diagnostic summary."""
    print("=" * 70)
    print("DIAGNOSTIC: Is R2 around 0.99 plausible?")
    print("=" * 70)

    target = summary["target"]
    print("\n1) Target spread")
    print("-" * 70)
    print(f"Count:   {target['count']:,}")
    print(f"Min:     {target['min']:.3f}")
    print(f"Max:     {target['max']:.3f}")
    print(f"Std:     {target['std']:.3f}")
    print(f"Range:   {target['range']:.3f}")

    baseline = summary["baseline"]
    print("\n2) Baseline check (predict mean)")
    print("-" * 70)
    print(f"Baseline R2:  {baseline['r2']:.4f}")
    print(f"Baseline MAE: {baseline['mae']:.4f}")

    linear_signal = summary["linear_signal"]
    print("\n3) Linear signal strength in edge features")
    print("-" * 70)
    for col, corr in linear_signal["correlations"].items():
        print(f"{col:>8}: r={corr:+.4f} |r|={abs(corr):.4f}")
    print(f"Linear model R2 on full dataset: {linear_signal['linear_r2']:.4f}")

    split = summary["split"]
    print("\n4) Split sanity")
    print("-" * 70)
    print(f"Unique samples: {split['n_samples']:,}")
    print(f"Total edge rows: {split['n_edges']:,}")
    print(f"Edges per sample (avg): {split['avg_edges_per_sample']:.2f}")
    print("Expected training split is graph-level (80/20), which avoids edge-level leakage.")

    reported = summary["reported"]
    print("\n5) Context for your reported result")
    print("-" * 70)
    print(f"Reported test R2: {reported['test_r2']:.4f}")
    print(f"Reported test MAE: {reported['test_mae']:.4f}")
    print(f"MAE / std: {reported['mae_over_std']:.3f}")
    print(f"MAE / range: {reported['mae_pct_of_range']:.2f}%")

    print("\nConclusion")
    print("-" * 70)
    print("If train and test R2 are close and split is graph-level, a high R2 can be valid.")
    print("Use this report plus a holdout dataset from a different generation batch for final trust.")


def main() -> None:
    summary = run_r2_legitimacy_diagnostic()
    print_r2_legitimacy_report(summary)


if __name__ == "__main__":
    main()
