"""
c21_train.py - Core training module for surrogate model
Encapsulates all training logic for use in both notebook and SLURM workflows.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch_geometric.loader import DataLoader

import config
from naming import build_run_id, build_model_artifact_stem
from c21_data_pipeline import (
    build_edge_index,
    infer_v4_schema,
    load_v4_sources,
    validate_sample_coverage,
    build_graph_dataset,
)
from c21_surrogate_model import TrussEdgeNNConv


def load_parameters(device: torch.device | None = None):
    """Load and validate all hyperparameters from environment variables."""
    fast_mode_env = os.getenv("C21_FAST_MODE")
    if fast_mode_env is None:
        fast_mode = bool(device is not None and device.type == "cpu")
        fast_mode_source = "auto_cpu" if fast_mode else "default"
    else:
        fast_mode = fast_mode_env.lower() == "true"
        fast_mode_source = "env"

    params = {
        # Data
        "node_csv": os.getenv("C21_NODE_CSV", "v4_node_C12_S9999_D20260409.csv"),
        "edge_csv": os.getenv("C21_EDGE_CSV", "v4_edge_C12_S9999_D20260409.csv"),
        "global_csv": os.getenv("C21_GLOBAL_CSV", "v4_global_C4_S9999_D20260409.csv"),
        # Workflow
        "use_pretrained": os.getenv("C21_USE_PRETRAINED", "false").lower() == "true",
        "pretrained_model_prefix": os.getenv("C21_PRETRAINED_MODEL_PREFIX", "data_4_0000"),
        # Training control
        "use_training_time_limit": os.getenv("C21_USE_TRAINING_TIME_LIMIT", "false").lower() == "true",
        "training_time_limit_seconds": int(os.getenv("C21_TRAINING_TIME_LIMIT_SECONDS", "60")),
        # Hyperparameters
        "learning_rate": float(os.getenv("C21_LEARNING_RATE", "0.0005")),
        "epochs": int(os.getenv("C21_EPOCHS", "100")),
        "batch_size": int(os.getenv("C21_BATCH_SIZE", "16")),
        "hidden_dim": int(os.getenv("C21_HIDDEN_DIM", "128")),
        "weight_decay": float(os.getenv("C21_WEIGHT_DECAY", "0.0")),
        "train_split_ratio": float(os.getenv("C21_TRAIN_SPLIT_RATIO", "0.8")),
        "random_seed": int(os.getenv("C21_RANDOM_SEED", "42")),
        "num_workers": int(
            os.getenv(
                "C21_NUM_WORKERS",
                "0" if os.name == "nt" else str(min(4, os.cpu_count() or 1)),
            )
        ),
        "eval_every": int(os.getenv("C21_EVAL_EVERY", "10")),
        "fast_mode": fast_mode,
        "fast_mode_source": fast_mode_source,
        # Run identity
        "run_id": os.getenv("C21_RUN_ID", build_run_id()),
    }

    # Fast-mode presets for local CPU iteration. Explicit env vars always win.
    if fast_mode:
        if "C21_EPOCHS" not in os.environ:
            params["epochs"] = 20
        if "C21_BATCH_SIZE" not in os.environ:
            params["batch_size"] = 64
        if "C21_HIDDEN_DIM" not in os.environ:
            params["hidden_dim"] = 64
        if "C21_EVAL_EVERY" not in os.environ:
            params["eval_every"] = 20
        if "C21_WEIGHT_DECAY" not in os.environ:
            params["weight_decay"] = 1e-4
        if "C21_USE_TRAINING_TIME_LIMIT" not in os.environ:
            params["use_training_time_limit"] = True
        if "C21_TRAINING_TIME_LIMIT_SECONDS" not in os.environ:
            params["training_time_limit_seconds"] = 1800

    return params


def load_data(params):
    """Load and validate multi-source dataset."""
    print("1. Loading multi-source dataset...")
    
    node_path = config.GH_DATA_PATH / params["node_csv"]
    edge_path = config.GH_DATA_PATH / params["edge_csv"]
    global_path = config.GH_DATA_PATH / params["global_csv"]

    df_node, df_edge, df_global = load_v4_sources(node_path, edge_path, global_path)
    schema = infer_v4_schema(df_node, df_edge, df_global)
    sample_ids = validate_sample_coverage(df_node, df_edge, df_global)
    edge_index = build_edge_index(df_edge)

    print(f"\n--- DATA VALIDATION ---")
    print(f"Node rows:         {len(df_node)}")
    print(f"Edge rows:         {len(df_edge)}")
    print(f"Global rows:       {len(df_global)}")
    print(f"Samples:           {len(sample_ids)}")
    print(f"Node count:        {schema.node_count}")
    print(f"Edge count:        {schema.edge_count}")
    print(f"edge_index shape:  {tuple(edge_index.shape)}")
    print("Validation successful. Multi-source data loaded correctly.\n")

    return df_node, df_edge, df_global, schema, sample_ids, edge_index


def process_data(df_node, df_edge, df_global, schema, sample_ids, edge_index, params):
    """Process, normalize, and split data."""
    print("Processing and normalizing data...")
    
    NODE_CSV = params["node_csv"]
    NODE_CONTINUOUS_COLS = list(schema.node_continuous_cols)
    NODE_MASK_COLS = list(schema.node_mask_cols)
    EDGE_FEATURE_COLS = list(schema.edge_feature_cols)
    GLOBAL_FEATURE_COLS = list(schema.global_feature_cols)
    TARGET_COL = "Axial_Force"

    # Split at graph level
    rng = np.random.default_rng(params["random_seed"])
    shuffled_sample_ids = np.array(rng.permutation(sample_ids), dtype=int)
    train_size = int(params["train_split_ratio"] * len(shuffled_sample_ids))
    train_sample_ids = shuffled_sample_ids[:train_size]
    test_sample_ids = shuffled_sample_ids[train_size:]

    train_node_df = df_node[df_node["sample_id"].isin(train_sample_ids)].copy()
    train_edge_df = df_edge[df_edge["Sample_ID"].isin(train_sample_ids)].copy()
    train_global_df = df_global[df_global["sample_id"].isin(train_sample_ids)].copy()

    # Fit scalers only on training split
    node_continuous_scaler = StandardScaler().fit(train_node_df[NODE_CONTINUOUS_COLS])
    edge_feature_scaler = StandardScaler().fit(train_edge_df[EDGE_FEATURE_COLS])
    edge_target_scaler = StandardScaler().fit(train_edge_df[[TARGET_COL]])
    global_feature_scaler = StandardScaler().fit(train_global_df[GLOBAL_FEATURE_COLS])

    # Transform full dataset
    node_continuous_scaled = pd.DataFrame(
        node_continuous_scaler.transform(df_node[NODE_CONTINUOUS_COLS]),
        index=df_node.index,
        columns=NODE_CONTINUOUS_COLS,
    )
    node_mask_values = df_node[NODE_MASK_COLS].astype(float).copy()
    edge_feature_scaled = pd.DataFrame(
        edge_feature_scaler.transform(df_edge[EDGE_FEATURE_COLS]),
        index=df_edge.index,
        columns=EDGE_FEATURE_COLS,
    )
    edge_target_scaled = pd.DataFrame(
        edge_target_scaler.transform(df_edge[[TARGET_COL]]),
        index=df_edge.index,
        columns=[TARGET_COL],
    )
    global_feature_scaled = pd.DataFrame(
        global_feature_scaler.transform(df_global[GLOBAL_FEATURE_COLS]),
        index=df_global.index,
        columns=GLOBAL_FEATURE_COLS,
    )

    # Build graph dataset
    graph_dataset = build_graph_dataset(
        df_node=df_node,
        df_edge=df_edge,
        df_global=df_global,
        schema=schema,
        node_continuous_scaled=node_continuous_scaled,
        node_mask_values=node_mask_values,
        edge_feature_scaled=edge_feature_scaled,
        edge_target_scaled=edge_target_scaled,
        global_feature_scaled=global_feature_scaled,
        edge_index=edge_index,
    )

    sample_to_graph = {int(data.sample_id): data for data in graph_dataset}
    train_dataset = [sample_to_graph[int(sample_id)] for sample_id in train_sample_ids]
    test_dataset = [sample_to_graph[int(sample_id)] for sample_id in test_sample_ids]

    # Create data loaders
    loader_kwargs = {
        "batch_size": params["batch_size"],
        "num_workers": max(0, int(params.get("num_workers", 0))),
        "pin_memory": torch.cuda.is_available(),
    }
    if loader_kwargs["num_workers"] > 0 and os.name != "nt":
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    print(f"Dataset ready! Train: {len(train_dataset)} graphs. Test: {len(test_dataset)} graphs.\n")

    return (
        train_loader,
        test_loader,
        node_continuous_scaler,
        edge_feature_scaler,
        edge_target_scaler,
        global_feature_scaler,
        schema,
    )


def setup_model(params, schema, device):
    """Initialize or load pretrained model."""
    NODE_FEATURE_DIM = len(schema.node_continuous_cols) + len(schema.node_mask_cols)
    EDGE_FEATURE_DIM = len(schema.edge_feature_cols)
    GLOBAL_FEATURE_DIM = len(schema.global_feature_cols)

    if params["use_pretrained"]:
        print(f"🔄 Loading pretrained model: {params['pretrained_model_prefix']}")
        checkpoint_path = config.SM_EXPORT_PATH / f"{params['pretrained_model_prefix']}_surrogate_model.pt"
        if not checkpoint_path.exists():
            checkpoint_path = config.SM_EXPORT_PATH / f"truss_edge_gnn_{params['pretrained_model_prefix']}.pt"
        
        checkpoint = torch.load(checkpoint_path, map_location=device)

        node_in_dim = checkpoint.get("node_in_dim", NODE_FEATURE_DIM) if isinstance(checkpoint, dict) else NODE_FEATURE_DIM
        edge_in_dim = checkpoint.get("edge_in_dim", EDGE_FEATURE_DIM) if isinstance(checkpoint, dict) else EDGE_FEATURE_DIM
        global_in_dim = checkpoint.get("global_in_dim", GLOBAL_FEATURE_DIM) if isinstance(checkpoint, dict) else GLOBAL_FEATURE_DIM
        hidden_dim = checkpoint.get("hidden_dim", params["hidden_dim"]) if isinstance(checkpoint, dict) else params["hidden_dim"]

        model = TrussEdgeNNConv(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            global_in_dim=global_in_dim,
            hidden_dim=hidden_dim,
        ).to(device)
        
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print("Pretrained model loaded.\n")
    else:
        print("Training from scratch.")
        model = TrussEdgeNNConv(
            node_in_dim=NODE_FEATURE_DIM,
            edge_in_dim=EDGE_FEATURE_DIM,
            global_in_dim=GLOBAL_FEATURE_DIM,
            hidden_dim=params["hidden_dim"],
        ).to(device)

    return model


def _collect_eval_metrics(model, loader, edge_target_scaler, device):
    """Collect R2/MAE/RMSE on original target scale for a dataloader."""
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
    preds_original = edge_target_scaler.inverse_transform(preds_scaled).reshape(-1)
    trues_original = edge_target_scaler.inverse_transform(trues_scaled).reshape(-1)

    r2 = float(r2_score(trues_original, preds_original))
    mae = float(mean_absolute_error(trues_original, preds_original))
    rmse = float(np.sqrt(mean_squared_error(trues_original, preds_original)))
    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "n_edges": int(trues_original.shape[0]),
    }


def train_model(model, train_loader, test_loader, edge_target_scaler, schema, params, device):
    """Execute training loop."""
    if params["use_pretrained"]:
        print("⏭️  Skipping training (using pretrained model)\n")
        train_metrics = _collect_eval_metrics(model, train_loader, edge_target_scaler, device)
        test_metrics = _collect_eval_metrics(model, test_loader, edge_target_scaler, device)
        run_metrics = {
            "final_val_r2": test_metrics["r2"],
            "train_r2": train_metrics["r2"],
            "test_r2": test_metrics["r2"],
            "train_mae": train_metrics["mae"],
            "test_mae": test_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "test_rmse": test_metrics["rmse"],
            "epochs_completed": 0,
            "training_time_seconds": 0.0,
            "best_train_loss": None,
            "n_train_edges": train_metrics["n_edges"],
            "n_test_edges": test_metrics["n_edges"],
        }
        return [], [], run_metrics

    print("🚀 Starting training...\n")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    criterion = torch.nn.MSELoss()

    EVAL_EVERY = max(1, int(params.get("eval_every", 10)))
    epoch_history = []
    train_loss_history = []
    final_val_r2 = None
    train_start_time = time.time()

    for epoch in range(params["epochs"]):
        if params["use_training_time_limit"]:
            elapsed = time.time() - train_start_time
            if elapsed >= params["training_time_limit_seconds"]:
                print(f"Stopped training after {elapsed:.1f}s (time limit reached).")
                break

        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            out = model(
                batch.x,
                batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                u=batch.u,
            )
            loss = criterion(out, batch.y_edge)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        avg_train_loss = total_loss / len(train_loader.dataset)
        epoch_history.append(epoch + 1)
        train_loss_history.append(avg_train_loss)

        # Evaluation
        if (epoch + 1) % EVAL_EVERY == 0:
            model.eval()
            pred_batches = []
            true_batches = []

            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device, non_blocking=True)
                    out = model(
                        batch.x,
                        batch.edge_index,
                        edge_attr=batch.edge_attr,
                        batch=batch.batch,
                        u=batch.u,
                    )
                    pred_batches.append(out.detach().cpu())
                    true_batches.append(batch.y_edge.detach().cpu())

            preds_scaled = torch.cat(pred_batches, dim=0).numpy()
            trues_scaled = torch.cat(true_batches, dim=0).numpy()

            preds_original = edge_target_scaler.inverse_transform(preds_scaled)
            trues_original = edge_target_scaler.inverse_transform(trues_scaled)

            r2 = r2_score(trues_original, preds_original)
            final_val_r2 = float(r2)
            print(f"Epoch {epoch+1:03d}/{params['epochs']} | Train Loss: {avg_train_loss:.4f} | Test R2: {r2:.4f}")

    train_metrics = _collect_eval_metrics(model, train_loader, edge_target_scaler, device)
    test_metrics = _collect_eval_metrics(model, test_loader, edge_target_scaler, device)
    training_time_seconds = float(time.time() - train_start_time)

    if final_val_r2 is None:
        final_val_r2 = test_metrics["r2"]

    run_metrics = {
        "final_val_r2": float(final_val_r2),
        "train_r2": train_metrics["r2"],
        "test_r2": test_metrics["r2"],
        "train_mae": train_metrics["mae"],
        "test_mae": test_metrics["mae"],
        "train_rmse": train_metrics["rmse"],
        "test_rmse": test_metrics["rmse"],
        "epochs_completed": int(len(epoch_history)),
        "training_time_seconds": training_time_seconds,
        "best_train_loss": float(min(train_loss_history)) if train_loss_history else None,
        "n_train_edges": train_metrics["n_edges"],
        "n_test_edges": test_metrics["n_edges"],
    }

    print(f"\nTraining completed! Test R2: {run_metrics['test_r2']:.4f} | Test RMSE: {run_metrics['test_rmse']:.4f} | Test MAE: {run_metrics['test_mae']:.4f}\n")
    return epoch_history, train_loss_history, run_metrics


def export_model(model, scalers, schema, params, run_metrics):
    """Save trained model and scalars."""
    if params["use_pretrained"]:
        print("⏭️  Skipping export (using pretrained model)\n")
        return

    print("💾 Exporting model and scalers...\n")

    artifact_stem = build_model_artifact_stem(
        params["run_id"],
        params["learning_rate"],
        params["epochs"],
        run_metrics["final_val_r2"],
    )

    # Save prefix for downstream
    with open(config.SM_EXPORT_PATH / "prefix_sm.txt", "w", encoding="utf-8") as f:
        f.write(artifact_stem)

    run_manifest = {
        "run_id": params["run_id"],
        "artifact_stem": artifact_stem,
        "dataset_sources": {
            "node": params["node_csv"],
            "edge": params["edge_csv"],
            "global": params["global_csv"],
        },
        "learning_rate": params["learning_rate"],
        "epochs": params["epochs"],
        "batch_size": params["batch_size"],
        "hidden_dim": params["hidden_dim"],
        "weight_decay": params["weight_decay"],
        "final_val_r2": run_metrics["final_val_r2"],
        "test_r2": run_metrics["test_r2"],
        "train_r2": run_metrics["train_r2"],
        "test_mae": run_metrics["test_mae"],
        "train_mae": run_metrics["train_mae"],
        "test_rmse": run_metrics["test_rmse"],
        "train_rmse": run_metrics["train_rmse"],
        "epochs_completed": run_metrics["epochs_completed"],
        "training_time_seconds": run_metrics["training_time_seconds"],
        "best_train_loss": run_metrics["best_train_loss"],
        "n_train_edges": run_metrics["n_train_edges"],
        "n_test_edges": run_metrics["n_test_edges"],
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "node_count": schema.node_count,
        "edge_count": schema.edge_count,
        "node_feature_dim": len(schema.node_continuous_cols) + len(schema.node_mask_cols),
        "edge_feature_dim": len(schema.edge_feature_cols),
        "global_feature_dim": len(schema.global_feature_cols),
        "target_col": "Axial_Force",
    }

    # Export scalers
    node_scaler_path = config.SM_EXPORT_PATH / f"{artifact_stem}_node_scaler.pkl"
    edge_feature_scaler_path = config.SM_EXPORT_PATH / f"{artifact_stem}_edge_feature_scaler.pkl"
    edge_target_scaler_path = config.SM_EXPORT_PATH / f"{artifact_stem}_edge_target_scaler.pkl"
    global_feature_scaler_path = config.SM_EXPORT_PATH / f"{artifact_stem}_global_feature_scaler.pkl"

    joblib.dump(scalers["node"], node_scaler_path)
    joblib.dump(scalers["edge_feature"], edge_feature_scaler_path)
    joblib.dump(scalers["edge_target"], edge_target_scaler_path)
    joblib.dump(scalers["global_feature"], global_feature_scaler_path)

    print(f"Scalers saved to:\n- {node_scaler_path}\n- {edge_feature_scaler_path}\n- {edge_target_scaler_path}\n- {global_feature_scaler_path}")

    # Export model checkpoint
    model_path = config.SM_EXPORT_PATH / f"{artifact_stem}_surrogate_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "node_in_dim": run_manifest["node_feature_dim"],
            "edge_in_dim": run_manifest["edge_feature_dim"],
            "global_in_dim": run_manifest["global_feature_dim"],
            "hidden_dim": params["hidden_dim"],
            "edge_count": schema.edge_count,
            "checkpoint_prefix": artifact_stem,
            "run_id": params["run_id"],
            "dataset_sources": run_manifest["dataset_sources"],
            "learning_rate": params["learning_rate"],
            "epochs": params["epochs"],
            "batch_size": params["batch_size"],
            "weight_decay": params["weight_decay"],
            "final_val_r2": run_metrics["final_val_r2"],
            "test_r2": run_metrics["test_r2"],
            "train_r2": run_metrics["train_r2"],
            "test_mae": run_metrics["test_mae"],
            "test_rmse": run_metrics["test_rmse"],
        },
        model_path
    )

    manifest_path = config.SM_EXPORT_PATH / f"{artifact_stem}_run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            run_manifest | {
                "model_path": model_path.name,
                "node_scaler_path": node_scaler_path.name,
                "edge_feature_scaler_path": edge_feature_scaler_path.name,
                "edge_target_scaler_path": edge_target_scaler_path.name,
                "global_feature_scaler_path": global_feature_scaler_path.name,
            },
            f,
            indent=2
        )

    print(f"Model checkpoint saved:\n- {model_path}")
    print(f"Run manifest saved:\n- {manifest_path}\n")


def main():
    """Main entry point for training workflow."""
    print("=" * 60)
    print("c21 SURROGATE MODEL TRAINING")
    print("=" * 60 + "\n")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load parameters
    params = load_parameters(device=device)
    print("Parameters loaded:")
    print(f"- fast_mode={params['fast_mode']} ({params['fast_mode_source']})")
    print(f"- lr={params['learning_rate']}, epochs={params['epochs']}, batch={params['batch_size']}")
    print(f"- hidden_dim={params['hidden_dim']}, weight_decay={params['weight_decay']}")
    print(f"- eval_every={params['eval_every']}, num_workers={params['num_workers']}")
    if params["use_training_time_limit"]:
        print(f"- training_time_limit_seconds={params['training_time_limit_seconds']}")
    print(f"- Run ID: {params['run_id']}\n")

    # Data loading
    df_node, df_edge, df_global, schema, sample_ids, edge_index = load_data(params)

    # Data processing
    train_loader, test_loader, node_scaler, edge_feature_scaler, edge_target_scaler, global_feature_scaler, schema = process_data(
        df_node, df_edge, df_global, schema, sample_ids, edge_index, params
    )

    # Model setup
    model = setup_model(params, schema, device)
    print(f"Model: TrussEdgeNNConv on {device}\n")

    # Training
    epoch_history, train_loss_history, run_metrics = train_model(
        model, train_loader, test_loader, edge_target_scaler, schema, params, device
    )

    # Export
    scalers = {
        "node": node_scaler,
        "edge_feature": edge_feature_scaler,
        "edge_target": edge_target_scaler,
        "global_feature": global_feature_scaler,
    }
    export_model(model, scalers, schema, params, run_metrics)

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    return {
        "model": model,
        "device": device,
        "epoch_history": epoch_history,
        "train_loss_history": train_loss_history,
        "final_val_r2": run_metrics["final_val_r2"],
        "run_metrics": run_metrics,
        "scalers": scalers,
        "schema": schema,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "params": params,
    }


if __name__ == "__main__":
    main()
