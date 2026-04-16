#!/usr/bin/env python
"""
Headless evaluation export for SLURM c21 runs.
Generates the same diagnostics as the training notebook and writes them to SM_DATA_PATH.
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config
from c21_model_evaluation import (
    build_error_distribution_figure,
    build_pred_residual_figure,
    build_training_visuals_figure,
    save_evaluation,
)
from naming import build_model_artifact_stem


def _collect_scaled_preds_trues(model, loader, device):
    model.eval()
    pred_batches = []
    true_batches = []

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
            pred_batches.append(out.detach().cpu().numpy())
            true_batches.append(batch.y_edge.detach().cpu().numpy())

    preds_scaled = np.concatenate(pred_batches, axis=0)
    trues_scaled = np.concatenate(true_batches, axis=0)
    return preds_scaled, trues_scaled


def export_slurm_evaluation(results: dict) -> dict:
    """Generate and save notebook-equivalent evaluation artifacts for SLURM runs."""
    model = results["model"]
    device = results["device"]
    epoch_history = results["epoch_history"]
    train_loss_history = results["train_loss_history"]
    final_val_r2 = results["final_val_r2"]
    edge_target_scaler = results["scalers"]["edge_target"]
    train_loader = results["train_loader"]
    test_loader = results["test_loader"]
    schema = results["schema"]
    params = results["params"]

    train_preds_scaled, train_trues_scaled = _collect_scaled_preds_trues(model, train_loader, device)
    test_preds_scaled, test_trues_scaled = _collect_scaled_preds_trues(model, test_loader, device)

    train_preds = edge_target_scaler.inverse_transform(train_preds_scaled).reshape(-1)
    train_trues = edge_target_scaler.inverse_transform(train_trues_scaled).reshape(-1)
    test_preds = edge_target_scaler.inverse_transform(test_preds_scaled).reshape(-1)
    test_trues = edge_target_scaler.inverse_transform(test_trues_scaled).reshape(-1)

    train_residuals = train_trues - train_preds
    test_residuals = test_trues - test_preds

    train_r2 = float(r2_score(train_trues, train_preds))
    test_r2 = float(r2_score(test_trues, test_preds))
    train_mae = float(mean_absolute_error(train_trues, train_preds))
    test_mae = float(mean_absolute_error(test_trues, test_preds))
    train_rmse = float(np.sqrt(mean_squared_error(train_trues, train_preds)))
    test_rmse = float(np.sqrt(mean_squared_error(test_trues, test_preds)))

    r2_gap = train_r2 - test_r2
    if train_r2 < 0.7 and test_r2 < 0.7:
        status = "underfitting"
    elif r2_gap > 0.05:
        status = "overfitting"
    else:
        status = "good_fit"

    metrics = {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "r2_gap": float(r2_gap),
    }

    artifact_stem = build_model_artifact_stem(
        params["run_id"],
        params["learning_rate"],
        params["epochs"],
        final_val_r2,
    )

    architecture_summary = {
        "model_class": "TrussEdgeNNConv",
        "node_in_dim": len(schema.node_continuous_cols) + len(schema.node_mask_cols),
        "edge_in_dim": len(schema.edge_feature_cols),
        "global_in_dim": len(schema.global_feature_cols),
        "hidden_dim": params["hidden_dim"],
        "edge_count": schema.edge_count,
        "device": str(device),
        "dataset_sources": {
            "node": params["node_csv"],
            "edge": params["edge_csv"],
            "global": params["global_csv"],
        },
    }

    experiment_notes = (
        f"SLURM_EVAL=true; USE_PRETRAINED={params['use_pretrained']}; "
        f"lr={params['learning_rate']}; epochs={params['epochs']}; "
        f"batch_size={params['batch_size']}; hidden_dim={params['hidden_dim']}; "
        f"weight_decay={params['weight_decay']}"
    )

    training_visuals_fig = build_training_visuals_figure(
        epoch_history,
        train_loss_history,
        test_trues_scaled,
        test_preds_scaled,
    )
    pred_residuals_fig = build_pred_residual_figure(
        train_trues,
        train_preds,
        test_trues,
        test_preds,
        train_r2,
        test_r2,
    )
    error_dist_fig = build_error_distribution_figure(
        train_residuals,
        test_residuals,
        train_mae,
        test_mae,
    )

    saved_files = save_evaluation(
        model_prefix=artifact_stem,
        dataset_name=f"{params['node_csv']} | {params['edge_csv']} | {params['global_csv']}",
        metrics=metrics,
        pred_residuals_fig=pred_residuals_fig,
        error_dist_fig=error_dist_fig,
        training_visuals_fig=training_visuals_fig,
        node_count=schema.node_count,
        edge_count=schema.edge_count,
        export_path=config.SM_DATA_PATH,
        status=status,
        run_id=params["run_id"],
        artifact_stem=artifact_stem,
        learning_rate=params["learning_rate"],
        epochs=params["epochs"],
        final_val_r2=final_val_r2,
        strict_dataset_label=f"{params['node_csv']} | {params['edge_csv']} | {params['global_csv']}",
        source_dataset_path=str(config.GH_DATA_PATH / params["edge_csv"]),
        architecture_summary=architecture_summary,
        experiment_notes=experiment_notes,
        train_split_ratio=params["train_split_ratio"],
        random_seed=params["random_seed"],
        source_notebook="c21_slurm_train.py",
    )

    plt.close(training_visuals_fig)
    plt.close(pred_residuals_fig)
    plt.close(error_dist_fig)

    print("\n✅ SLURM evaluation export completed.")
    print(f"Run ID: {params['run_id']}")
    print(f"Artifact stem (shared with 01_surrogate_models): {artifact_stem}")
    print(f"Evaluation folder root: {config.SM_DATA_PATH}")

    return saved_files


if __name__ == "__main__":
    raise SystemExit("This module is intended to be imported by workflows/c21_slurm_train.py")
