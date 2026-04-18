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
from c00_naming import build_run_id, build_model_artifact_stem
from c21_data_pipeline import (
    build_edge_index,
    infer_v4_schema,
    load_v4_sources,
    validate_sample_coverage,
    build_graph_dataset,
)
from c21_surrogate_model import TrussEdgeNNConv, TrussEdgeNNConvV2


def _select_model_class(model_variant: str):
    variant = str(model_variant).strip().lower()
    if variant in {"v1", "baseline", "trussedgennconv"}:
        return TrussEdgeNNConv, "v1"
    if variant in {"v2", "enhanced", "trussedgennconvv2"}:
        return TrussEdgeNNConvV2, "v2"
    raise ValueError(f"Unsupported C21_MODEL_VARIANT='{model_variant}'. Use 'v1' or 'v2'.")


class EdgeWiseStandardScaler:
    """Per-edge standardization using a fixed edge order for inverse transforms."""

    def __init__(self, edge_order, edge_mean_map, edge_std_map):
        self.edge_order = list(edge_order)
        self.edge_count = len(self.edge_order)
        self.mean_map = {str(k): float(v) for k, v in edge_mean_map.items()}
        self.std_map = {str(k): float(max(v, 1e-8)) for k, v in edge_std_map.items()}

        self._mean_vector = np.array([self.mean_map[eid] for eid in self.edge_order], dtype=np.float32).reshape(-1, 1)
        self._std_vector = np.array([self.std_map[eid] for eid in self.edge_order], dtype=np.float32).reshape(-1, 1)

    def transform_df(self, edge_df: pd.DataFrame, target_col: str, edge_id_col: str = "Edge_ID") -> pd.DataFrame:
        edge_ids = edge_df[edge_id_col].astype(str)
        means = edge_ids.map(self.mean_map).to_numpy(dtype=np.float32).reshape(-1, 1)
        stds = edge_ids.map(self.std_map).to_numpy(dtype=np.float32).reshape(-1, 1)
        values = edge_df[[target_col]].to_numpy(dtype=np.float32)
        scaled = (values - means) / np.maximum(stds, 1e-8)
        return pd.DataFrame(scaled, index=edge_df.index, columns=[target_col])

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] != 1:
            raise ValueError(f"Expected shape (N, 1), got {arr.shape}")
        if arr.shape[0] % self.edge_count != 0:
            raise ValueError(
                f"Input length {arr.shape[0]} is not divisible by edge_count={self.edge_count}. "
                "Cannot apply edge-wise inverse transform safely."
            )

        repeats = arr.shape[0] // self.edge_count
        means = np.tile(self._mean_vector, (repeats, 1))
        stds = np.tile(self._std_vector, (repeats, 1))
        return arr * stds + means


def _resolve_artifact_dir(artifact_prefix: str) -> Path:
    """Resolve the artifact directory for a given prefix, supporting flat legacy layout."""
    nested_dir = config.SM_EXPORT_PATH / artifact_prefix
    if nested_dir.exists() and nested_dir.is_dir():
        return nested_dir
    return config.SM_EXPORT_PATH


def _resolve_checkpoint_path(artifact_prefix: str) -> Path:
    """Resolve checkpoint path from nested or legacy flat layout."""
    artifact_dir = _resolve_artifact_dir(artifact_prefix)

    nested_candidate = artifact_dir / f"{artifact_prefix}_surrogate_model.pt"
    if nested_candidate.exists():
        return nested_candidate

    nested_legacy = artifact_dir / f"truss_edge_gnn_{artifact_prefix}.pt"
    if nested_legacy.exists():
        return nested_legacy

    flat_candidate = config.SM_EXPORT_PATH / f"{artifact_prefix}_surrogate_model.pt"
    if flat_candidate.exists():
        return flat_candidate

    return config.SM_EXPORT_PATH / f"truss_edge_gnn_{artifact_prefix}.pt"


def _parse_csv_columns_env(var_name: str) -> tuple[str, ...] | None:
    """Parse comma-separated env var into a tuple, or None when unset/auto."""
    raw = os.getenv(var_name)
    if raw is None:
        return None
    cleaned = [item.strip() for item in raw.split(",") if item.strip()]
    if not cleaned:
        return None
    return tuple(cleaned)


def load_parameters(device: torch.device | None = None):
    """Load and validate all hyperparameters from environment variables."""
    fast_mode_env = os.getenv("C21_FAST_MODE")
    if fast_mode_env is None:
        fast_mode = bool(device is not None and device.type == "cpu")
        fast_mode_source = "auto_cpu" if fast_mode else "default"
    else:
        fast_mode = fast_mode_env.lower() == "true"
        fast_mode_source = "env"

    data_prefix = os.getenv("C21_DATA_PREFIX", "S19999_D20260418")

    params = {
        # Data
        "data_prefix": data_prefix,
        "node_csv": os.getenv("C21_NODE_CSV", f"v4_node_C6_{data_prefix}.csv"),
        "edge_csv": os.getenv("C21_EDGE_CSV", f"v4_edge_C7_{data_prefix}.csv"),
        "global_csv": os.getenv("C21_GLOBAL_CSV", f"v4_global_C4_{data_prefix}.csv"),
        "use_global_csv": os.getenv("C21_USE_GLOBAL_CSV", "false").lower() == "true",
        "selected_node_continuous_cols": _parse_csv_columns_env("C21_NODE_CONTINUOUS_COLS"),
        "selected_node_mask_cols": _parse_csv_columns_env("C21_NODE_MASK_COLS"),
        "selected_edge_feature_cols": _parse_csv_columns_env("C21_EDGE_FEATURE_COLS"),
        "selected_global_feature_cols": _parse_csv_columns_env("C21_GLOBAL_FEATURE_COLS"),
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
        "model_variant": os.getenv("C21_MODEL_VARIANT", "v1").lower(),
        "dropout_p": float(os.getenv("C21_DROPOUT_P", "0.1")),
        "weight_decay": float(os.getenv("C21_WEIGHT_DECAY", "0.0")),
        "loss_type": os.getenv("C21_LOSS_TYPE", "mse").lower(),
        "huber_beta": float(os.getenv("C21_HUBER_BETA", "1.0")),
        "use_lr_scheduler": os.getenv("C21_USE_LR_SCHEDULER", "false").lower() == "true",
        "lr_scheduler_factor": float(os.getenv("C21_LR_SCHEDULER_FACTOR", "0.5")),
        "lr_scheduler_patience": int(os.getenv("C21_LR_SCHEDULER_PATIENCE", "2")),
        "lr_scheduler_metric": os.getenv("C21_LR_SCHEDULER_METRIC", "r2").lower(),
        "lr_scheduler_min_lr": float(os.getenv("C21_LR_SCHEDULER_MIN_LR", "1e-6")),
        "edge_wise_target_scaling": os.getenv("C21_EDGE_WISE_TARGET_SCALING", "false").lower() == "true",
        "use_virtual_node": os.getenv("C21_USE_VIRTUAL_NODE", "true").lower() == "true",
        "train_split_ratio": float(os.getenv("C21_TRAIN_SPLIT_RATIO", "0.8")),
        "random_seed": int(os.getenv("C21_RANDOM_SEED", "42")),
        "num_workers": int(
            os.getenv(
                "C21_NUM_WORKERS",
                "0" if os.name == "nt" else str(min(4, os.cpu_count() or 1)),
            )
        ),
        "eval_every": int(os.getenv("C21_EVAL_EVERY", "10")),
        "use_early_stopping": os.getenv("C21_USE_EARLY_STOPPING", "false").lower() == "true",
        "early_stopping_patience": int(os.getenv("C21_EARLY_STOPPING_PATIENCE", "5")),
        "early_stopping_metric": os.getenv("C21_EARLY_STOPPING_METRIC", "r2"),
        "skip_export": os.getenv("C21_SKIP_EXPORT", "false").lower() == "true",
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
    use_global = params.get("use_global_csv", True)
    selected_node_continuous_cols = params.get("selected_node_continuous_cols")
    selected_node_mask_cols = params.get("selected_node_mask_cols")
    selected_edge_feature_cols = params.get("selected_edge_feature_cols")
    selected_global_feature_cols = params.get("selected_global_feature_cols")

    schema_kwargs = {
        "use_virtual_node": params.get("use_virtual_node", False),
        "use_global_csv": use_global,
        "selected_node_continuous_cols": selected_node_continuous_cols,
        "selected_node_mask_cols": selected_node_mask_cols,
        "selected_edge_feature_cols": selected_edge_feature_cols,
        "selected_global_feature_cols": selected_global_feature_cols,
    }

    if use_global:
        global_path = config.GH_DATA_PATH / params["global_csv"]
        df_node, df_edge, df_global = load_v4_sources(node_path, edge_path, global_path)
        schema = infer_v4_schema(df_node, df_edge, df_global, **schema_kwargs)
        sample_ids = validate_sample_coverage(df_node, df_edge, df_global)
    else:
        import pandas as pd
        df_node = pd.read_csv(node_path)
        df_edge = pd.read_csv(edge_path)
        df_global = None
        schema = infer_v4_schema(df_node, df_edge, None, **schema_kwargs)
        # Only check node/edge sample coverage
        node_samples = set(df_node["sample_id"].unique().tolist())
        edge_samples = set(df_edge["Sample_ID"].unique().tolist())
        if node_samples != edge_samples:
            raise ValueError(f"sample_id coverage mismatch: node={len(node_samples)}, edge={len(edge_samples)}")
        sample_ids = sorted(node_samples)
    edge_index = build_edge_index(df_edge)

    print(f"\n--- DATA VALIDATION ---")
    print(f"Node rows:         {len(df_node)}")
    print(f"Edge rows:         {len(df_edge)}")
    if use_global:
        print(f"Global rows:       {len(df_global)}")
    print(f"Samples:           {len(sample_ids)}")
    print(f"Node count:        {schema.node_count}")
    print(f"Edge count:        {schema.edge_count}")
    print(f"Node features:     {schema.node_continuous_cols + schema.node_mask_cols + schema.node_virtual_cols}")
    print(f"Edge features:     {schema.edge_feature_cols}")
    print(f"Global features:   {schema.global_feature_cols}")
    print(f"edge_index shape:  {tuple(edge_index.shape)}")
    print("Validation successful. Multi-source data loaded correctly.\n")

    return df_node, df_edge, df_global, schema, sample_ids, edge_index


def process_data(df_node, df_edge, df_global, schema, sample_ids, edge_index, params):
    """Process, normalize, and split data."""
    print("Processing and normalizing data...")
    
    NODE_CSV = params["node_csv"]
    NODE_CONTINUOUS_COLS = list(schema.node_continuous_cols)
    NODE_MASK_COLS = list(schema.node_mask_cols)
    NODE_LOAD_COLS = list(schema.node_load_cols)
    NODE_VIRTUAL_COLS = list(schema.node_virtual_cols)
    EDGE_FEATURE_COLS = list(schema.edge_feature_cols)
    GLOBAL_FEATURE_COLS = list(schema.global_feature_cols) if schema.global_feature_cols else []
    TARGET_COL = "Axial_Force"

    for load_col in NODE_LOAD_COLS:
        if load_col not in df_node.columns:
            df_node[load_col] = 0.0

    # Split at graph level
    rng = np.random.default_rng(params["random_seed"])
    shuffled_sample_ids = np.array(rng.permutation(sample_ids), dtype=int)
    train_size = int(params["train_split_ratio"] * len(shuffled_sample_ids))
    train_sample_ids = shuffled_sample_ids[:train_size]
    test_sample_ids = shuffled_sample_ids[train_size:]

    train_node_df = df_node[df_node["sample_id"].isin(train_sample_ids)].copy()
    train_edge_df = df_edge[df_edge["Sample_ID"].isin(train_sample_ids)].copy()
    train_global_df = df_global[df_global["sample_id"].isin(train_sample_ids)].copy() if (df_global is not None and hasattr(df_global, '__getitem__')) else None

    # Fit scalers only on training split
    node_continuous_scaler = StandardScaler().fit(train_node_df[NODE_CONTINUOUS_COLS])
    edge_feature_scaler = StandardScaler().fit(train_edge_df[EDGE_FEATURE_COLS])
    if params.get("edge_wise_target_scaling", False):
        edge_order = sorted(df_edge["Edge_ID"].astype(str).unique().tolist(), key=lambda v: int("".join(ch for ch in v if ch.isdigit()) or "0"))
        grouped = train_edge_df.groupby(train_edge_df["Edge_ID"].astype(str))[TARGET_COL]
        edge_mean_map = grouped.mean().to_dict()
        edge_std_map = grouped.std(ddof=0).fillna(1.0).replace(0.0, 1.0).to_dict()
        edge_target_scaler = EdgeWiseStandardScaler(edge_order=edge_order, edge_mean_map=edge_mean_map, edge_std_map=edge_std_map)
    else:
        edge_target_scaler = StandardScaler().fit(train_edge_df[[TARGET_COL]])
    if train_global_df is not None and GLOBAL_FEATURE_COLS:
        global_feature_scaler = StandardScaler().fit(train_global_df[GLOBAL_FEATURE_COLS])
    else:
        global_feature_scaler = None

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
    if params.get("edge_wise_target_scaling", False):
        edge_target_scaled = edge_target_scaler.transform_df(df_edge, TARGET_COL, edge_id_col="Edge_ID")
    else:
        edge_target_scaled = pd.DataFrame(
            edge_target_scaler.transform(df_edge[[TARGET_COL]]),
            index=df_edge.index,
            columns=[TARGET_COL],
        )
    if global_feature_scaler is not None and df_global is not None and GLOBAL_FEATURE_COLS:
        global_feature_scaled = pd.DataFrame(
            global_feature_scaler.transform(df_global[GLOBAL_FEATURE_COLS]),
            index=df_global.index,
            columns=GLOBAL_FEATURE_COLS,
        )
    else:
        global_feature_scaled = None

    use_virtual_node = params.get("use_virtual_node", False)

    # Build graph dataset
    graph_dataset = build_graph_dataset(
        df_node=df_node,
        df_edge=df_edge,
        df_global=df_global if df_global is not None else None,
        schema=schema,
        node_continuous_scaled=node_continuous_scaled,
        node_mask_values=node_mask_values,
        edge_feature_scaled=edge_feature_scaled,
        edge_target_scaled=edge_target_scaled,
        global_feature_scaled=global_feature_scaled if global_feature_scaled is not None else None,
        edge_index=edge_index,
        use_virtual_node=use_virtual_node,
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
    NODE_FEATURE_DIM = len(schema.node_continuous_cols) + len(schema.node_mask_cols) + len(schema.node_virtual_cols)
    EDGE_FEATURE_DIM = len(schema.edge_feature_cols)
    GLOBAL_FEATURE_DIM = len(schema.global_feature_cols)

    if params["use_pretrained"]:
        print(f"🔄 Loading pretrained model: {params['pretrained_model_prefix']}")
        checkpoint_path = _resolve_checkpoint_path(params["pretrained_model_prefix"])
        
        checkpoint = torch.load(checkpoint_path, map_location=device)

        node_in_dim = checkpoint.get("node_in_dim", NODE_FEATURE_DIM) if isinstance(checkpoint, dict) else NODE_FEATURE_DIM
        edge_in_dim = checkpoint.get("edge_in_dim", EDGE_FEATURE_DIM) if isinstance(checkpoint, dict) else EDGE_FEATURE_DIM
        global_in_dim = checkpoint.get("global_in_dim", GLOBAL_FEATURE_DIM) if isinstance(checkpoint, dict) else GLOBAL_FEATURE_DIM
        hidden_dim = checkpoint.get("hidden_dim", params["hidden_dim"]) if isinstance(checkpoint, dict) else params["hidden_dim"]
        checkpoint_variant = checkpoint.get("model_variant", params.get("model_variant", "v1")) if isinstance(checkpoint, dict) else params.get("model_variant", "v1")
        model_class, resolved_variant = _select_model_class(checkpoint_variant)
        dropout_p = checkpoint.get("dropout_p", params.get("dropout_p", 0.1)) if isinstance(checkpoint, dict) else params.get("dropout_p", 0.1)

        if node_in_dim != NODE_FEATURE_DIM or edge_in_dim != EDGE_FEATURE_DIM or global_in_dim != GLOBAL_FEATURE_DIM:
            raise ValueError(
                "Pretrained checkpoint dimensions do not match the current schema. "
                f"checkpoint=({node_in_dim}, {edge_in_dim}, {global_in_dim}), "
                f"current=({NODE_FEATURE_DIM}, {EDGE_FEATURE_DIM}, {GLOBAL_FEATURE_DIM})"
            )

        model_kwargs = {
            "node_in_dim": node_in_dim,
            "edge_in_dim": edge_in_dim,
            "global_in_dim": global_in_dim,
            "hidden_dim": hidden_dim,
        }
        if resolved_variant == "v2":
            model_kwargs["dropout_p"] = float(dropout_p)
        model = model_class(**model_kwargs).to(device)
        
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        params["model_variant"] = resolved_variant
        params["dropout_p"] = float(dropout_p)
        print("Pretrained model loaded.\n")
    else:
        print("Training from scratch.")
        model_class, resolved_variant = _select_model_class(params.get("model_variant", "v1"))
        model_kwargs = {
            "node_in_dim": NODE_FEATURE_DIM,
            "edge_in_dim": EDGE_FEATURE_DIM,
            "global_in_dim": GLOBAL_FEATURE_DIM,
            "hidden_dim": params["hidden_dim"],
        }
        if resolved_variant == "v2":
            model_kwargs["dropout_p"] = float(params.get("dropout_p", 0.1))
        model = model_class(**model_kwargs).to(device)
        params["model_variant"] = resolved_variant

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

    finite_mask = np.isfinite(trues_original) & np.isfinite(preds_original)
    trues_original = trues_original[finite_mask]
    preds_original = preds_original[finite_mask]

    if trues_original.size == 0:
        return {
            "r2": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "n_edges": 0,
        }

    r2 = float(r2_score(trues_original, preds_original))
    mae = float(mean_absolute_error(trues_original, preds_original))
    rmse = float(np.sqrt(mean_squared_error(trues_original, preds_original)))
    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "n_edges": int(trues_original.shape[0]),
    }


def _masked_edge_loss(predictions, targets, mask, criterion):
    if mask is None:
        mask = torch.ones_like(targets)
    per_edge_loss = criterion(predictions, targets)
    if per_edge_loss.dim() > 1:
        per_edge_loss = per_edge_loss.view(per_edge_loss.size(0), -1).mean(dim=1)
    mask_values = mask.view(-1).to(per_edge_loss.dtype)
    denom = mask_values.sum().clamp_min(1.0)
    return (per_edge_loss * mask_values).sum() / denom


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
    if params.get("loss_type", "mse") == "huber":
        criterion = torch.nn.SmoothL1Loss(beta=float(params.get("huber_beta", 1.0)), reduction="none")
    else:
        criterion = torch.nn.MSELoss(reduction="none")

    scheduler = None
    if params.get("use_lr_scheduler", False):
        scheduler_metric = params.get("lr_scheduler_metric", "r2")
        scheduler_mode = "max" if scheduler_metric == "r2" else "min"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=float(params.get("lr_scheduler_factor", 0.5)),
            patience=int(params.get("lr_scheduler_patience", 2)),
            min_lr=float(params.get("lr_scheduler_min_lr", 1e-6)),
        )

    EVAL_EVERY = max(1, int(params.get("eval_every", 10)))
    epoch_history = []
    train_loss_history = []
    epoch_metrics_history = []  # Collect per-epoch eval metrics for export
    final_val_r2 = None
    train_start_time = time.time()
    
    # Early stopping tracking
    best_val_r2 = -float('inf')
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_metric = params.get("early_stopping_metric", "r2")
    early_stopping_patience = params.get("early_stopping_patience", 5)
    use_early_stopping = params.get("use_early_stopping", False)

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
            loss = _masked_edge_loss(out, batch.y_edge, getattr(batch, "edge_loss_mask", None), criterion)
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
                mask_batches = []
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
                    mask_batches.append(getattr(batch, "edge_loss_mask", torch.ones_like(batch.y_edge)).detach().cpu())

            preds_tensor = torch.cat(pred_batches, dim=0)
            trues_tensor = torch.cat(true_batches, dim=0)
            mask_tensor = torch.cat(mask_batches, dim=0)

            eval_loss = float(_masked_edge_loss(preds_tensor, trues_tensor, mask_tensor, criterion).item())
            keep = mask_tensor.view(-1) > 0.5
            preds_scaled = preds_tensor[keep].numpy()
            trues_scaled = trues_tensor[keep].numpy()

            preds_original = edge_target_scaler.inverse_transform(preds_scaled)
            trues_original = edge_target_scaler.inverse_transform(trues_scaled)

            preds_flat = preds_original.reshape(-1)
            trues_flat = trues_original.reshape(-1)
            finite_mask = np.isfinite(trues_flat) & np.isfinite(preds_flat)

            if not np.any(finite_mask):
                print(
                    f"Epoch {epoch+1:03d}/{params['epochs']} | "
                    f"Train Loss: {avg_train_loss:.4f} | Val Loss: {eval_loss:.4f} | "
                    "Test R2: nan (no finite eval pairs)"
                )
                final_val_r2 = float("nan")
                continue

            preds_eval = preds_flat[finite_mask]
            trues_eval = trues_flat[finite_mask]

            dropped_non_finite = int(finite_mask.size - finite_mask.sum())
            if dropped_non_finite > 0:
                print(
                    f"Filtered {dropped_non_finite} non-finite validation pairs before metric computation."
                )

            r2 = r2_score(trues_eval, preds_eval)
            final_val_r2 = float(r2)
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1:03d}/{params['epochs']} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {eval_loss:.4f} | "
                f"Test R2: {r2:.4f} | LR: {current_lr:.6g}"
            )
            # Collect epoch metrics for export
            epoch_metrics_history.append({
                "epoch": epoch + 1,
                "train_loss": float(avg_train_loss),
                "val_loss": float(eval_loss),
                "test_r2": float(r2),
                "learning_rate": float(current_lr),
            })

            if scheduler is not None:
                if params.get("lr_scheduler_metric", "r2") == "r2":
                    scheduler.step(float(r2))
                else:
                    scheduler.step(float(eval_loss))
            
            # Early stopping check
            if use_early_stopping:
                if early_stopping_metric == "r2":
                    if r2 > best_val_r2:
                        best_val_r2 = r2
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                else:  # val_loss
                    if eval_loss < best_val_loss:
                        best_val_loss = eval_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                
                if epochs_no_improve >= early_stopping_patience:
                    print(f"\n⏸️  Early stopping: no improvement for {early_stopping_patience} evaluations.")
                    break

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
    return epoch_history, train_loss_history, run_metrics, epoch_metrics_history


def export_model(model, scalers, schema, params, run_metrics):
    """Save trained model and scalars."""
    if params["use_pretrained"]:
        print("⏭️  Skipping export (using pretrained model)\n")
        return
    if params.get("skip_export", False):
        print("⏭️  Skipping export (C21_SKIP_EXPORT=true)\n")
        return

    print("💾 Exporting model and scalers...\n")

    artifact_stem = build_model_artifact_stem(
        params["run_id"],
        params["learning_rate"],
        params["epochs"],
        run_metrics["final_val_r2"],
    )
    artifact_dir = config.SM_EXPORT_PATH / artifact_stem
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Save prefix for downstream
    with open(config.SM_EXPORT_PATH / "prefix_sm.txt", "w", encoding="utf-8") as f:
        f.write(artifact_stem)

    run_manifest = {
        "run_id": params["run_id"],
        "artifact_stem": artifact_stem,
        "dataset_sources": {
            "node": params["node_csv"],
            "edge": params["edge_csv"],
            "global": params["global_csv"] if params.get("use_global_csv", True) else None,
        },
        "use_global_csv": params.get("use_global_csv", True),
        "selected_node_continuous_cols": list(schema.node_continuous_cols),
        "selected_node_mask_cols": list(schema.node_mask_cols),
        "selected_node_virtual_cols": list(getattr(schema, "node_virtual_cols", ())),
        "selected_edge_feature_cols": list(schema.edge_feature_cols),
        "selected_global_feature_cols": list(schema.global_feature_cols),
        "learning_rate": params["learning_rate"],
        "epochs": params["epochs"],
        "batch_size": params["batch_size"],
        "hidden_dim": params["hidden_dim"],
        "model_variant": params.get("model_variant", "v1"),
        "dropout_p": params.get("dropout_p", 0.1),
        "weight_decay": params["weight_decay"],
        "loss_type": params.get("loss_type", "mse"),
        "huber_beta": params.get("huber_beta", 1.0),
        "use_lr_scheduler": params.get("use_lr_scheduler", False),
        "lr_scheduler_factor": params.get("lr_scheduler_factor", 0.5),
        "lr_scheduler_patience": params.get("lr_scheduler_patience", 2),
        "lr_scheduler_metric": params.get("lr_scheduler_metric", "r2"),
        "lr_scheduler_min_lr": params.get("lr_scheduler_min_lr", 1e-6),
        "edge_wise_target_scaling": params.get("edge_wise_target_scaling", False),
        "use_virtual_node": params.get("use_virtual_node", False),
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
    node_scaler_path = artifact_dir / f"{artifact_stem}_node_scaler.pkl"
    edge_feature_scaler_path = artifact_dir / f"{artifact_stem}_edge_feature_scaler.pkl"
    edge_target_scaler_path = artifact_dir / f"{artifact_stem}_edge_target_scaler.pkl"
    global_feature_scaler_path = artifact_dir / f"{artifact_stem}_global_feature_scaler.pkl"

    joblib.dump(scalers["node"], node_scaler_path)
    joblib.dump(scalers["edge_feature"], edge_feature_scaler_path)
    joblib.dump(scalers["edge_target"], edge_target_scaler_path)
    joblib.dump(scalers["global_feature"], global_feature_scaler_path)

    print(f"Scalers saved to:\n- {node_scaler_path}\n- {edge_feature_scaler_path}\n- {edge_target_scaler_path}\n- {global_feature_scaler_path}")

    # Export model checkpoint
    model_path = artifact_dir / f"{artifact_stem}_surrogate_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "node_in_dim": run_manifest["node_feature_dim"],
            "edge_in_dim": run_manifest["edge_feature_dim"],
            "global_in_dim": run_manifest["global_feature_dim"],
            "hidden_dim": params["hidden_dim"],
            "model_variant": params.get("model_variant", "v1"),
            "dropout_p": params.get("dropout_p", 0.1),
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

    manifest_path = artifact_dir / f"{artifact_stem}_run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            run_manifest | {
                "artifact_dir": artifact_dir.name,
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
    print(
        f"- data: node={params['node_csv']}, edge={params['edge_csv']}, "
        f"global={params['global_csv']} (enabled={params['use_global_csv']})"
    )
    print(f"- hidden_dim={params['hidden_dim']}, weight_decay={params['weight_decay']}")
    print(f"- model_variant={params['model_variant']}, dropout_p={params['dropout_p']}")
    print(f"- loss_type={params['loss_type']}, huber_beta={params['huber_beta']}")
    print(f"- eval_every={params['eval_every']}, num_workers={params['num_workers']}")
    if params["use_lr_scheduler"]:
        print(
            f"- lr_scheduler: metric={params['lr_scheduler_metric']}, factor={params['lr_scheduler_factor']}, "
            f"patience={params['lr_scheduler_patience']}, min_lr={params['lr_scheduler_min_lr']}"
        )
    if params["edge_wise_target_scaling"]:
        print("- edge_wise_target_scaling=True")
    if params["use_training_time_limit"]:
        print(f"- training_time_limit_seconds={params['training_time_limit_seconds']}")
    if params["use_early_stopping"]:
        print(f"- early_stopping: patience={params['early_stopping_patience']}, metric={params['early_stopping_metric']}")
    if params["skip_export"]:
        print("- skip_export=True")
    print(f"- Run ID: {params['run_id']}\n")

    # Data loading
    df_node, df_edge, df_global, schema, sample_ids, edge_index = load_data(params)

    # Data processing
    train_loader, test_loader, node_scaler, edge_feature_scaler, edge_target_scaler, global_feature_scaler, schema = process_data(
        df_node, df_edge, df_global, schema, sample_ids, edge_index, params
    )

    # Model setup
    model = setup_model(params, schema, device)
    print(f"Model: {model.__class__.__name__} on {device}\n")

    # Training
    epoch_history, train_loss_history, run_metrics, epoch_metrics_history = train_model(
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
        "epoch_metrics_history": epoch_metrics_history,
        "scalers": scalers,
        "schema": schema,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "params": params,
    }


if __name__ == "__main__":
    main()
