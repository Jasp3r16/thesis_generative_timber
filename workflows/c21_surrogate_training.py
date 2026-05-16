"""GNN surrogate model training pipeline for structural timber feasibility prediction.

Four stages callable from a notebook:
    pre   = run_preprocessing(edge_csv, node_csv, ...)
    train = run_training(pre, ...)
    eval_ = run_evaluation(train, pre)
    exp   = run_export(pre, train, eval_)

Each function accepts keyword arguments for its important parameters and returns
a dict that the next stage (and notebook inspection) can read from.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, auc, brier_score_loss, confusion_matrix,
    f1_score, matthews_corrcoef, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
    classification_report,
)
from sklearn.calibration import calibration_curve
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

import config
from c21_surrogate_model_v4 import create_model, FocalLoss, WeightedBCELoss


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _collect_preds(dataloader, model, device) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over a DataLoader. Returns (probs, targets) as flat arrays."""
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            probs = model(batch.x, batch.edge_index, batch.edge_attr)
            all_probs.append(probs.cpu())
            all_targets.append(batch.y.cpu())
    probs   = torch.cat(all_probs,   dim=0).view(-1).numpy()
    targets = torch.cat(all_targets, dim=0).view(-1).numpy()
    return probs, targets


def _scores_at(threshold: float, probs: np.ndarray, targets: np.ndarray) -> dict:
    pred               = (probs >= threshold).astype(int)
    cm                 = confusion_matrix(targets, pred, labels=[0, 1])
    tn_, fp_, fn_, tp_ = cm.ravel()
    return dict(
        threshold=threshold, pred=pred, cm=cm,
        tn=int(tn_), fp=int(fp_), fn=int(fn_), tp=int(tp_),
        accuracy  = accuracy_score(targets, pred),
        precision = precision_score(targets, pred, zero_division=0),
        recall    = recall_score(targets, pred, zero_division=0),
        f1        = f1_score(targets, pred, zero_division=0),
        mcc       = matthews_corrcoef(targets, pred),
    )


def _classification_report_at_threshold(probs, targets, threshold, label=""):
    from sklearn.metrics import confusion_matrix, classification_report as cr
    preds_bin     = (probs >= threshold).astype(int)
    cm            = confusion_matrix(targets.astype(int), preds_bin)
    rec  = recall_score(targets, preds_bin, pos_label=1, zero_division=0)
    prec = precision_score(targets, preds_bin, pos_label=1, zero_division=0)
    f1   = f1_score(targets, preds_bin, pos_label=1, zero_division=0)
    print(f"\n{'='*60}")
    print(f"{label}  (threshold={threshold:.2f})")
    print(f"{'='*60}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"              Pred Safe  Pred Unsafe")
    print(f"  Act Safe    {cm[0,0]:9d}  {cm[0,1]:11d}")
    print(f"  Act Unsafe  {cm[1,0]:9d}  {cm[1,1]:11d}")
    print()
    print(cr(targets.astype(int), preds_bin,
             target_names=["Safe (0)", "Unsafe (1)"], digits=4))
    print(f"Unsafe class -> Recall: {rec:.4f}  Precision: {prec:.4f}  F1: {f1:.4f}")
    return rec, prec, f1


# =============================================================================
# STAGE 1 — PREPROCESSING
# =============================================================================

def run_preprocessing(
    edge_csv:        str,
    node_csv:        str,
    *,
    bidirectional:   bool = False,
    batch_size:      int  = 32,
    data_inspection: bool = False,
    hidden_dim:      int  = 64,
    num_layers:      int  = 3,
    dropout_p:       float = 0.3,
) -> dict[str, Any]:
    """Load CSVs, build PyG Dataset with train/val/test split, create model.

    Parameters
    ----------
    edge_csv        : stem of the edge feature CSV in config.GH_DATA_PATH
    node_csv        : stem of the node feature CSV in config.GH_DATA_PATH
    bidirectional   : use undirected (240 edges) or directed (120) topology
    batch_size      : DataLoader batch size
    data_inspection : print per-sample utilisation statistics

    Returns
    -------
    dict with keys:
        model, loss_fn, focal_alpha, train_pos_rate
        train_dataloader, val_dataloader, test_dataloader
        train_dataset,    val_dataset,    test_dataset
        train_samples,    val_samples,    test_samples
        node_cols, edge_cols
        node_feature_means, node_feature_stds
        edge_feature_means, edge_feature_stds
        device, batch_size
        num_edges, num_edges_raw, bidirectional
        node_csv_path, edge_csv_path, edge_index_path
        nodes_df, edges_df
    """
    import pandas as pd

    node_csv_path = config.GH_DATA_PATH / f"{node_csv}.csv"
    edge_csv_path = config.GH_DATA_PATH / f"{edge_csv}.csv"

    nodes_df = pd.read_csv(node_csv_path)
    edges_df = pd.read_csv(edge_csv_path)

    node_cols = ["x", "y", "z", "Tx", "Ty", "Tz", "Rx", "Ry", "Rz", "Fz"]
    edge_cols = ["Width_m", "Depth_m", "Length", "E", "Iy", "Iz", "J", "EA/L", "N_mean_EA"]

    # ---- Topology ----
    edge_index_path = Path(config.DATA_IO_PATH) / "edge_index.json"
    if not edge_index_path.exists():
        raise FileNotFoundError(
            f"edge_index.json not found at {edge_index_path}."
        )
    with open(edge_index_path) as f:
        edge_index_list = json.load(f)
    edge_index = torch.tensor(edge_index_list, dtype=torch.long)

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape [2, num_edges], got {tuple(edge_index.shape)}"
        )

    expected_num_nodes = int(edge_index.max().item()) + 1

    # ---- Sample ID detection ----
    sample_id_col = None
    for col in ("sample_id", "Sample_ID", "SampleId"):
        if col in nodes_df.columns and col in edges_df.columns:
            sample_id_col = col
            break
    if sample_id_col is None:
        raise KeyError(
            "No sample ID column found. Expected: 'sample_id', 'Sample_ID', 'SampleId'."
        )
    print(f"Sample ID column: '{sample_id_col}'")

    # num_edges_raw from CSV rows per sample (always 120, regardless of JSON)
    num_edges_raw = int(edges_df.groupby(sample_id_col).size().iloc[0])
    print(f"Topology: {edge_index.shape[1]} edges in JSON | "
          f"{num_edges_raw} rows/sample in CSV | {expected_num_nodes} nodes/sample.")

    # ---- Bidirectionality ----
    src, dst    = edge_index[0], edge_index[1]
    forward     = set(zip(src.tolist(), dst.tolist()))
    backward    = set(zip(dst.tolist(), src.tolist()))
    is_undirected = len(forward - backward) == 0

    if bidirectional:
        if not is_undirected:
            print(f"Converting directed → undirected ({edge_index.shape[1]} → 2× edges).")
            edge_index = to_undirected(edge_index)
            with open(edge_index_path, "w") as f:
                json.dump(edge_index.tolist(), f)
            print(f"Saved bidirectional edge_index: {edge_index.shape[1]} edges.")
        else:
            print(f"Graph already undirected: {edge_index.shape[1]} edges.")
    else:
        if is_undirected and edge_index.shape[1] == 2 * num_edges_raw:
            print(f"BIDIRECTIONAL=False but JSON has {edge_index.shape[1]} edges. Stripping back.")
            mask = edge_index[0] < edge_index[1]
            edge_index = edge_index[:, mask]
            with open(edge_index_path, "w") as f:
                json.dump(edge_index.tolist(), f)
            print(f"Directed edge_index restored: {edge_index.shape[1]} edges.")
        else:
            print(f"Using directed graph: {edge_index.shape[1]} edges.")

    num_edges = int(edge_index.shape[1])
    print(f"Final edge count: {num_edges} ({'bidirectional' if bidirectional else 'unidirectional'}).")

    # ---- Column validation ----
    for cols, csv_path, label in [
        (node_cols, node_csv_path, "node"),
        (edge_cols, edge_csv_path, "edge"),
    ]:
        missing = [c for c in cols if c not in (nodes_df if label == "node" else edges_df).columns]
        if missing:
            raise KeyError(f"Missing required {label} columns: {missing} in {csv_path}.")
    if "Utilization" not in edges_df.columns:
        raise KeyError(f"Missing required target column 'Utilization' in {edge_csv_path}.")

    # ---- Sample validation ----
    node_groups = nodes_df.groupby(sample_id_col)
    edge_groups = edges_df.groupby(sample_id_col)
    samples = sorted(set(node_groups.groups.keys()) & set(edge_groups.groups.keys()))
    if not samples:
        raise ValueError("No matching sample IDs between node and edge CSVs.")
    print(f"Found {len(samples)} matching samples.")

    for s in samples:
        nc = len(node_groups.get_group(s))
        ec = len(edge_groups.get_group(s))
        if nc != expected_num_nodes:
            raise ValueError(f"Sample {s}: node count {nc} != {expected_num_nodes}")
        if ec != num_edges_raw:
            raise ValueError(f"Sample {s}: edge count {ec} != {num_edges_raw}")

    # ---- Train / Val / Test split ----
    torch.manual_seed(42)
    shuffled    = torch.randperm(len(samples)).tolist()
    train_size  = int(0.8 * len(samples))
    val_size    = int(0.1 * len(samples))
    train_samples = [samples[i] for i in shuffled[:train_size]]
    val_samples   = [samples[i] for i in shuffled[train_size:train_size + val_size]]
    test_samples  = [samples[i] for i in shuffled[train_size + val_size:]]
    print(f"Split: Train={len(train_samples)} | Val={len(val_samples)} | Test={len(test_samples)}")

    # ---- Normalisation (from training data only) ----
    train_nodes = nodes_df[nodes_df[sample_id_col].isin(train_samples)]
    train_edges = edges_df[edges_df[sample_id_col].isin(train_samples)]

    node_means = train_nodes[node_cols].mean()
    node_stds  = train_nodes[node_cols].std(ddof=0).replace(0, 1.0)
    edge_means = train_edges[edge_cols].mean()
    edge_stds  = train_edges[edge_cols].std(ddof=0).replace(0, 1.0)

    norm_stats = {
        "node_means": node_means.to_dict(), "node_stds":  node_stds.to_dict(),
        "edge_means": edge_means.to_dict(), "edge_stds":  edge_stds.to_dict(),
        "node_cols": node_cols,             "edge_cols":  edge_cols,
    }
    norm_stats_path = Path(config.DATA_IO_PATH) / "norm_stats.pt"
    torch.save(norm_stats, norm_stats_path)
    print(f"Norm stats saved: {norm_stats_path}")

    # ---- Class balance ----
    train_pos_rate = float((train_edges["Utilization"] > 1).mean())
    pos_weight     = float((1.0 - train_pos_rate) / max(train_pos_rate, 1e-6))
    print(f"Train positive rate: {train_pos_rate:.4f} → pos_weight={pos_weight:.4f}")

    # ---- Dataset construction ----
    def build_sample(s):
        n_df = node_groups.get_group(s)
        e_df = edge_groups.get_group(s)
        x = torch.tensor(
            ((n_df[node_cols] - node_means) / node_stds).clip(-5, 5).values,
            dtype=torch.float32,
        )
        ea = torch.tensor(
            ((e_df[edge_cols] - edge_means) / edge_stds).clip(-5, 5).values,
            dtype=torch.float32,
        )
        y = torch.tensor((e_df["Utilization"] > 1).astype(int).values, dtype=torch.float32)
        if bidirectional and num_edges == 2 * num_edges_raw:
            ea = torch.cat([ea, ea], dim=0)
            y  = torch.cat([y,  y ], dim=0)
        return Data(x=x, edge_index=edge_index.clone(), edge_attr=ea, y=y.view(-1, 1))

    train_dataset = [build_sample(s) for s in train_samples]
    val_dataset   = [build_sample(s) for s in val_samples]
    test_dataset  = [build_sample(s) for s in test_samples]
    print(f"Dataset: {len(train_dataset)} train | {len(val_dataset)} val | {len(test_dataset)} test. "
          f"Each: {expected_num_nodes} nodes, {num_edges} edges.")

    # ---- Model ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = create_model(
        node_features_dim=len(node_cols),
        edge_features_dim=len(edge_cols),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_p=dropout_p,
        device=device,
    ).to(device)
    print(f"Model on {device} | node_dim={len(node_cols)} | edge_dim={len(edge_cols)} | "
          f"hidden_dim={hidden_dim} | num_layers={num_layers} | dropout_p={dropout_p}")

    # ---- DataLoaders ----
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    print(f"DataLoaders: {len(train_dataloader)} train | {len(val_dataloader)} val | "
          f"{len(test_dataloader)} test batches (batch_size={batch_size})")

    # ---- Loss ----
    loss_fn = WeightedBCELoss(pos_weight=pos_weight)
    print(f"WeightedBCELoss(pos_weight={pos_weight:.4f})")

    # ---- Optional inspection ----
    if data_inspection:
        unsafe_per_sample = edges_df.groupby(sample_id_col).apply(
            lambda g: (g["Utilization"] > 1).sum()
        )
        print(unsafe_per_sample.describe())
        print(unsafe_per_sample.value_counts().sort_index().head(20))
        zero_ids = unsafe_per_sample[unsafe_per_sample == 0].index
        if len(zero_ids):
            print(f"\nEdge feature stats for {len(zero_ids)} all-safe samples:")
            print(edges_df[edges_df[sample_id_col].isin(zero_ids)][edge_cols].describe())

    return {
        "model": model, "loss_fn": loss_fn, "pos_weight": pos_weight,
        "train_pos_rate": train_pos_rate,
        "train_dataloader": train_dataloader,
        "val_dataloader":   val_dataloader,
        "test_dataloader":  test_dataloader,
        "train_dataset": train_dataset, "val_dataset": val_dataset, "test_dataset": test_dataset,
        "train_samples": train_samples, "val_samples": val_samples,  "test_samples": test_samples,
        "node_cols": node_cols,  "edge_cols": edge_cols,
        "node_feature_means": node_means, "node_feature_stds": node_stds,
        "edge_feature_means": edge_means, "edge_feature_stds": edge_stds,
        "device": device, "batch_size": batch_size,
        "num_edges": num_edges, "num_edges_raw": num_edges_raw, "bidirectional": bidirectional,
        "node_csv_path": node_csv_path, "edge_csv_path": edge_csv_path,
        "edge_index_path": edge_index_path,
        "nodes_df": nodes_df, "edges_df": edges_df,
    }


# =============================================================================
# STAGE 2 — TRAINING
# =============================================================================

def run_training(
    pre: dict,
    *,
    epochs:            int   = 150,
    lr:                float = 3e-4,
    patience:          int   = 40,
    lr_factor:         float = 0.5,
    lr_patience:       int   = 10,
    lr_min:            float = 1e-6,
    grad_clip:         float = 1.0,
    weight_decay:      float = 1e-3,
    pos_weight:        float | None = None,
    default_threshold: float = 0.35,
    min_precision:     float = 0.40,
) -> dict[str, Any]:
    """Train the GNN and run threshold sweep on the validation set.

    Parameters
    ----------
    pre              : output of run_preprocessing()
    epochs           : maximum training epochs
    lr               : initial learning rate (AdamW)
    patience         : early stopping patience (epochs without val improvement)
    lr_factor        : ReduceLROnPlateau reduction factor
    lr_patience      : ReduceLROnPlateau patience (epochs)
    lr_min           : minimum learning rate floor
    grad_clip        : gradient clipping max-norm (None = disabled)
    focal_alpha      : FocalLoss alpha override (0.5 empirically better than 1-pos_rate)
    default_threshold: fallback decision threshold when no val sweep finds one
    min_precision    : minimum unsafe-class precision required for threshold selection

    Returns
    -------
    dict with keys:
        model, loss_fn, focal_alpha
        best_val_loss, best_epoch
        train_losses, val_losses, test_loss
        val_probs, val_targets
        test_probs, test_targets
        best_threshold, val_auc, test_auc
        EPOCHS, LR, PATIENCE, GRAD_CLIP, CKPT_PATH, batch_size
        min_precision
    """
    model           = pre["model"]
    device          = pre["device"]
    train_dataloader = pre["train_dataloader"]
    val_dataloader   = pre["val_dataloader"]
    test_dataloader  = pre["test_dataloader"]
    train_dataset   = pre["train_dataset"]
    val_dataset     = pre["val_dataset"]
    test_dataset    = pre["test_dataset"]

    CKPT_PATH = config.DATA_IO_PATH / "surrogate_v4_checkpoint.pth"

    # Use pos_weight from caller if given, otherwise inherit from preprocessing.
    _pw = pos_weight if pos_weight is not None else float(pre.get("pos_weight", 4.0))
    loss_fn = WeightedBCELoss(pos_weight=_pw)
    print(f"WeightedBCELoss(pos_weight={_pw:.4f})")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=lr_min,
    )

    print(f"\nHyperparameters: epochs={epochs}, lr={lr:.0e}, patience={patience}, "
          f"grad_clip={grad_clip}, weight_decay={weight_decay:.0e}, pos_weight={_pw:.4f}, "
          f"default_threshold={default_threshold}, min_precision={min_precision}")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"\nStarting training: {epochs} epochs, early stopping patience={patience}")
    print("-" * 70)

    train_losses: list[float] = []
    val_losses:   list[float] = []
    best_val_loss    = float("inf")
    best_state       = None
    best_epoch       = -1
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_train_loss = 0.0
        for batch in train_dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            preds = model(batch.x, batch.edge_index, batch.edge_attr)
            loss  = loss_fn(preds, batch.y)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            epoch_train_loss += loss.item() * batch.num_graphs
        epoch_train_loss /= len(train_dataset)
        train_losses.append(epoch_train_loss)

        # Validate
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = batch.to(device)
                preds = model(batch.x, batch.edge_index, batch.edge_attr)
                loss  = loss_fn(preds, batch.y)
                epoch_val_loss += loss.item() * batch.num_graphs
        epoch_val_loss /= len(val_dataset)
        val_losses.append(epoch_val_loss)

        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if epoch_val_loss < best_val_loss:
            best_val_loss     = float(epoch_val_loss)
            best_epoch        = int(epoch)
            best_state        = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:03d}  train={epoch_train_loss:.6f}  "
                  f"val={epoch_val_loss:.6f}  lr={current_lr:.2e}  "
                  f"no_improve={epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
            break

    print("-" * 70)

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best checkpoint from epoch {best_epoch+1}  val_loss={best_val_loss:.6f}")
    else:
        print("Warning: best_state not set; using last epoch weights.")

    torch.save({
        "model_state_dict": model.state_dict(),
        "best_val_loss": best_val_loss, "best_epoch": best_epoch,
        "pos_weight": _pw,
        "train_losses": train_losses, "val_losses": val_losses,
    }, CKPT_PATH)
    print(f"Checkpoint saved: {CKPT_PATH}")

    # ---- Threshold sweep on validation set ----
    print("\n--- Threshold sweep on validation set ---")
    val_probs, val_targets = _collect_preds(val_dataloader, model, device)

    try:
        val_auc = float(roc_auc_score(val_targets, val_probs))
        print(f"Val AUC-ROC: {val_auc:.4f}")
    except ValueError:
        print("Val AUC-ROC: n/a (only one class in val targets)")
        val_auc = None

    thresholds     = np.arange(0.10, 0.65, 0.05)
    best_threshold = default_threshold
    best_recall    = -1.0
    sweep_results  = []

    for t in thresholds:
        preds_bin   = (val_probs >= t).astype(int)
        rec  = recall_score(val_targets, preds_bin, pos_label=1, zero_division=0)
        prec = precision_score(val_targets, preds_bin, pos_label=1, zero_division=0)
        f1   = f1_score(val_targets, preds_bin, pos_label=1, zero_division=0)
        sweep_results.append((t, rec, prec, f1))
        if rec > best_recall and prec >= min_precision:
            best_recall    = rec
            best_threshold = t

    print(f"\n{'Threshold':>10}  {'Recall(unsafe)':>15}  {'Precision(unsafe)':>18}  {'F1':>10}")
    print("-" * 58)
    for t, r, p, f in sweep_results:
        marker = " <-- selected" if abs(t - best_threshold) < 1e-6 else ""
        print(f"{t:10.2f}  {r:15.4f}  {p:18.4f}  {f:10.4f}{marker}")

    print(f"\nSelected threshold: {best_threshold:.2f} "
          f"(max recall >= {min_precision:.0%} precision constraint)")

    _classification_report_at_threshold(val_probs, val_targets, best_threshold, "VALIDATION SET")

    # ---- Test set evaluation ----
    print("\n--- Test set evaluation ---")
    test_probs, test_targets = _collect_preds(test_dataloader, model, device)

    try:
        test_auc = float(roc_auc_score(test_targets, test_probs))
        print(f"Test AUC-ROC: {test_auc:.4f}")
    except ValueError:
        print("Test AUC-ROC: n/a")
        test_auc = None

    _classification_report_at_threshold(test_probs, test_targets, best_threshold, "TEST SET")
    _classification_report_at_threshold(test_probs, test_targets, 0.5, "TEST SET (thr=0.50, reference)")

    # Test loss
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(device)
            preds = model(batch.x, batch.edge_index, batch.edge_attr)
            test_loss += loss_fn(preds, batch.y).item() * batch.num_graphs
    test_loss /= len(test_dataset)

    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"  Best epoch:      {best_epoch+1}")
    print(f"  Best val loss:   {best_val_loss:.6f}")
    print(f"  Test focal loss: {test_loss:.6f}")
    if val_auc:  print(f"  Val  AUC-ROC:    {val_auc:.4f}")
    if test_auc: print(f"  Test AUC-ROC:    {test_auc:.4f}")
    print(f"  Decision threshold (val-tuned): {best_threshold:.2f}")
    print(f"  Checkpoint: {CKPT_PATH}")
    print(f"{'='*70}")

    return {
        "model": model, "loss_fn": loss_fn, "pos_weight": _pw,
        "best_val_loss": best_val_loss, "best_epoch": best_epoch,
        "train_losses": train_losses, "val_losses": val_losses, "test_loss": test_loss,
        "val_probs": val_probs, "val_targets": val_targets,
        "test_probs": test_probs, "test_targets": test_targets,
        "best_threshold": best_threshold, "val_auc": val_auc, "test_auc": test_auc,
        "EPOCHS": epochs, "LR": lr, "PATIENCE": patience, "GRAD_CLIP": grad_clip,
        "CKPT_PATH": CKPT_PATH, "batch_size": pre["batch_size"],
        "min_precision": min_precision,
    }


# =============================================================================
# STAGE 3 — EVALUATION
# =============================================================================

def run_evaluation(
    train_out: dict,
    pre:       dict,
    *,
    save_path: Path | None = None,
) -> dict[str, Any]:
    """Compute metrics and produce 5 publication-quality figures.

    Parameters
    ----------
    train_out  : output of run_training()
    pre        : output of run_preprocessing()
    save_path  : directory for intermediate figure PNGs (default: config.SM_EXPORT_PATH)

    Returns
    -------
    dict with keys:
        metrics
        fig1, fig2, fig3, fig4, fig5
        thr_primary, threshold_f1, threshold_safety
        roc_auc, pr_auc, brier
        test_probs_arr, test_true
        s_primary, s_f1, s_safety, s_05
        per_member_recall, per_member_fpr, per_member_unsafe_rate
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    C = config.PLOT_COLORS
    S = config.PLOT_STYLE

    if save_path is None:
        save_path = config.SM_EXPORT_PATH
    save_path = Path(save_path)

    train_losses   = train_out["train_losses"]
    val_losses     = train_out["val_losses"]
    test_probs     = train_out["test_probs"]
    test_targets   = train_out["test_targets"]
    best_threshold = train_out["best_threshold"]
    best_epoch     = train_out["best_epoch"]
    min_precision  = train_out.get("min_precision", 0.40)

    num_edges          = pre["num_edges"]
    num_edges_physical = pre["num_edges_raw"]

    # ---- Global style ----
    plt.rcParams.update({
        "figure.dpi": S["dpi"], "axes.grid": True, "grid.alpha": S["grid_alpha"],
        "grid.color": C["neutral"], "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": C["black"], "axes.labelcolor": C["black"],
        "xtick.color": C["black"], "ytick.color": C["black"], "text.color": C["black"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "lines.linewidth": S["line_width"], "lines.markersize": S["marker_size"],
    })

    def save_fig(fig, stem):
        out = save_path / f"{stem}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C["white"])
        print(f"  Saved: {out.name}")

    # ---- Core metrics ----
    test_probs_arr = np.asarray(test_probs).flatten()
    test_true      = np.asarray(test_targets).flatten().astype(int)
    epochs_range   = np.arange(1, len(train_losses) + 1)

    fpr, tpr, _                           = roc_curve(test_true, test_probs_arr)
    roc_auc_val                           = roc_auc_score(test_true, test_probs_arr)
    precision_curve, recall_curve, pr_thr = precision_recall_curve(test_true, test_probs_arr)
    pr_auc_val                            = auc(recall_curve, precision_curve)

    f1_curve     = (2 * precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-12)
    idx_f1       = int(np.argmax(f1_curve[:-1]))
    threshold_f1 = float(pr_thr[idx_f1]) if len(pr_thr) > 0 else 0.5

    viable_mask = precision_curve[:-1] >= min_precision
    if viable_mask.any():
        idx_safety       = np.where(viable_mask)[0][int(np.argmax(recall_curve[:-1][viable_mask]))]
        threshold_safety = float(pr_thr[idx_safety])
    else:
        threshold_safety = threshold_f1
        print(f"Warning: no threshold achieves precision >= {min_precision:.0%}; "
              f"falling back to max-F1 threshold ({threshold_f1:.3f}).")

    thr_primary = float(best_threshold)
    brier       = brier_score_loss(test_true, test_probs_arr)

    s_05      = _scores_at(0.5,            test_probs_arr, test_true)
    s_primary = _scores_at(thr_primary,    test_probs_arr, test_true)
    s_f1      = _scores_at(threshold_f1,   test_probs_arr, test_true)
    s_safety  = _scores_at(threshold_safety, test_probs_arr, test_true)

    print(f"Thresholds — val-tuned: {thr_primary:.3f} | max-F1: {threshold_f1:.3f} | "
          f"safety (P>={min_precision:.0%}): {threshold_safety:.3f}")

    # ---- Figure 1 — Training Dynamics ----
    fig1, axes = plt.subplots(1, 2, figsize=S["figsize_medium"])
    fig1.suptitle("Figure 1 — Training Dynamics", fontweight="bold", fontsize=13)
    ax = axes[0]
    ax.plot(epochs_range, train_losses, color=C["primary"],   lw=S["line_width"], marker="o", ms=3, label="Train Loss")
    ax.plot(epochs_range, val_losses,   color=C["accent"],    lw=S["line_width"], marker="s", ms=3, linestyle="--", label="Val Loss")
    ax.axvline(best_epoch+1, color=C["danger"], linestyle=":", lw=1.5, label=f"Best epoch ({best_epoch+1})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Focal Loss"); ax.set_title("Train vs Validation Loss"); ax.legend()
    ax = axes[1]
    gap = np.array(val_losses) - np.array(train_losses)
    ax.plot(epochs_range, gap, color=C["secondary"], lw=S["line_width"])
    ax.axhline(0, color=C["black"], linestyle="--", lw=1)
    ax.axvline(best_epoch+1, color=C["danger"], linestyle=":", lw=1.5, label=f"Best epoch ({best_epoch+1})")
    ax.fill_between(epochs_range, gap, 0, where=(gap > 0), alpha=0.2, color=C["danger"],  label="Overfitting")
    ax.fill_between(epochs_range, gap, 0, where=(gap < 0), alpha=0.2, color=C["primary"], label="Underfitting")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val − Train Loss"); ax.set_title("Generalisation Gap"); ax.legend(fontsize=9)
    plt.tight_layout(); save_fig(fig1, "eval_fig1_training_dynamics"); plt.show()

    # ---- Figure 2 — Threshold Analysis ----
    fig2, axes = plt.subplots(1, 3, figsize=(S["figsize_large"][0], 5))
    fig2.suptitle("Figure 2 — Threshold Analysis", fontweight="bold", fontsize=13)
    ax = axes[0]
    ax.plot(fpr, tpr, color=C["primary"], lw=S["line_width"], label=f"AUC = {roc_auc_val:.3f}")
    ax.plot([0,1], [0,1], color=C["secondary"], lw=1.5, linestyle="--", label="Random")
    for s, label, col in [
        (s_primary, f"val-tuned ({thr_primary:.2f})",   C["primary"]),
        (s_f1,      f"max-F1 ({threshold_f1:.2f})",     C["secondary"]),
        (s_safety,  f"safety ({threshold_safety:.2f})", C["accent"]),
    ]:
        ax.scatter(s["fp"]/max(s["fp"]+s["tn"],1), s["tp"]/max(s["tp"]+s["fn"],1),
                   color=col, zorder=5, s=80, label=label, edgecolors=C["black"], linewidths=0.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve"); ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    ax.plot(recall_curve, precision_curve, color=C["primary"], lw=S["line_width"], label=f"PR AUC = {pr_auc_val:.3f}")
    ax.axhline(test_true.mean(), color=C["neutral"], linestyle="--", lw=1.2, label=f"Baseline")
    ax.axhline(min_precision, color=C["accent"], linestyle=":", lw=1.2, label=f"Min prec={min_precision:.0%}")
    for s, label, col in [(s_primary,"val-tuned",C["primary"]),(s_f1,"max-F1",C["secondary"]),(s_safety,"safety",C["accent"])]:
        ax.scatter(s["recall"], s["precision"], color=col, zorder=5, s=80, label=label, edgecolors=C["black"], linewidths=0.5)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall Curve"); ax.legend(fontsize=8)
    ax = axes[2]
    sw_thr = np.linspace(0.05, 0.95, 100)
    sw_rec, sw_prec, sw_f1, sw_mcc = [], [], [], []
    for t in sw_thr:
        p = (test_probs_arr >= t).astype(int)
        sw_rec.append(recall_score(test_true, p, zero_division=0))
        sw_prec.append(precision_score(test_true, p, zero_division=0))
        sw_f1.append(f1_score(test_true, p, zero_division=0))
        sw_mcc.append(matthews_corrcoef(test_true, p))
    ax.plot(sw_thr, sw_rec,  color=C["danger"],    lw=S["line_width"], label="Recall")
    ax.plot(sw_thr, sw_prec, color=C["primary"],   lw=S["line_width"], label="Precision")
    ax.plot(sw_thr, sw_f1,   color=C["secondary"], lw=S["line_width"], label="F1")
    ax.plot(sw_thr, sw_mcc,  color=C["accent"],    lw=S["line_width"], linestyle="--", label="MCC")
    for thr, col, lbl in [(thr_primary,C["primary"],"val-tuned"),(threshold_f1,C["secondary"],"max-F1"),(threshold_safety,C["accent"],"safety")]:
        ax.axvline(thr, color=col, lw=1.5, linestyle=":", label=f"{lbl}({thr:.2f})")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score"); ax.set_title("Metrics vs Threshold"); ax.legend(fontsize=8)
    plt.tight_layout(); save_fig(fig2, "eval_fig2_threshold_analysis"); plt.show()

    # ---- Figure 3 — Prediction Quality ----
    fig3, axes = plt.subplots(1, 3, figsize=(S["figsize_large"][0], 5))
    fig3.suptitle("Figure 3 — Prediction Quality & Calibration", fontweight="bold", fontsize=13)
    ax = axes[0]
    ax.hist(test_probs_arr[test_true==0], bins=40, alpha=0.65, label=f"Safe (n={(test_true==0).sum()})",   color=C["primary"], edgecolor=C["white"])
    ax.hist(test_probs_arr[test_true==1], bins=40, alpha=0.65, label=f"Unsafe (n={(test_true==1).sum()})", color=C["danger"],  edgecolor=C["white"])
    ax.axvline(thr_primary,      color=C["primary"], lw=2,   linestyle="--", label=f"val-tuned ({thr_primary:.2f})")
    ax.axvline(threshold_safety, color=C["accent"],  lw=1.5, linestyle=":",  label=f"safety ({threshold_safety:.2f})")
    ax.axvline(0.5,              color=C["black"],   lw=1.2, linestyle="--", label="thr=0.50")
    ax.set_xlabel("P(unsafe)"); ax.set_ylabel("Count"); ax.set_title("Score Distribution by True Class"); ax.legend(fontsize=8)
    ax = axes[1]
    bins = np.linspace(0,1,41)
    ax.hist(test_probs_arr[test_true==0], bins=bins, alpha=0.65, label="Safe",   color=C["primary"], edgecolor=C["white"])
    ax.hist(test_probs_arr[test_true==1], bins=bins, alpha=0.65, label="Unsafe", color=C["danger"],  edgecolor=C["white"])
    ax.axvline(thr_primary, color=C["primary"], lw=2, linestyle="--", label=f"val-tuned ({thr_primary:.2f})")
    ax.set_yscale("log"); ax.set_xlabel("P(unsafe)"); ax.set_ylabel("Count (log)"); ax.set_title("Score Distribution (log scale)"); ax.legend(fontsize=8)
    ax = axes[2]
    try:
        frac_pos, mean_pred = calibration_curve(test_true, test_probs_arr, n_bins=10, strategy="uniform")
        ax.plot(mean_pred, frac_pos, color=C["primary"], lw=S["line_width"], marker="o", ms=6, label=f"GNN (Brier={brier:.3f})")
    except Exception as e:
        ax.text(0.5, 0.5, f"Calibration n/a\n({e})", ha="center", va="center", fontsize=9)
    ax.plot([0,1],[0,1], color=C["black"], linestyle="--", lw=1.5, label="Perfect")
    ax.set_xlabel("Mean Predicted Prob"); ax.set_ylabel("Fraction Positive"); ax.set_title("Calibration Curve"); ax.legend(fontsize=9)
    plt.tight_layout(); save_fig(fig3, "eval_fig3_prediction_quality"); plt.show()

    # ---- Figure 4 — Confusion Matrices ----
    fig4, axes = plt.subplots(1, 3, figsize=S["figsize_large"])
    fig4.suptitle("Figure 4 — Confusion Matrices", fontweight="bold", fontsize=13)
    cm_cmap = LinearSegmentedColormap.from_list("cm", [C["white"], C["primary"]], N=256)
    for ax, (s, title) in zip(axes, [
        (s_05,      "Default (thr=0.50)"),
        (s_primary, f"Val-tuned (thr={thr_primary:.2f})"),
        (s_safety,  f"Safety (thr={threshold_safety:.2f})"),
    ]):
        cm = s["cm"]
        ax.imshow(cm, interpolation="nearest", cmap=cm_cmap)
        ax.set_title(title); ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Safe","Unsafe"]); ax.set_yticklabels(["Safe","Unsafe"])
        for i in range(2):
            for j in range(2):
                val = cm[i,j]
                ax.text(j, i, f"{val}", ha="center", va="center", fontsize=13, fontweight="bold",
                        color=C["white"] if val > cm.max()*0.6 else C["black"])
        ax.set_xlabel(f"Predicted\nRecall={s['recall']:.3f}  Prec={s['precision']:.3f}\nF1={s['f1']:.3f}  MCC={s['mcc']:.3f}  FN={s['fn']}", fontsize=9)
        ax.set_ylabel("True")
    plt.tight_layout(); save_fig(fig4, "eval_fig4_confusion_matrices"); plt.show()

    # ---- Figure 5 — Per-Member Analysis ----
    fig5 = None
    per_member_recall = per_member_fpr = per_member_unsafe_rate = None

    n_test_samples = len(test_probs_arr) // num_edges
    if len(test_probs_arr) % num_edges == 0 and n_test_samples > 0:
        preds_mat   = (test_probs_arr >= thr_primary).astype(int).reshape(n_test_samples, num_edges)
        targets_mat = test_true.reshape(n_test_samples, num_edges)
        probs_mat   = test_probs_arr.reshape(n_test_samples, num_edges)
        preds_mat   = preds_mat[:, :num_edges_physical]
        targets_mat = targets_mat[:, :num_edges_physical]
        probs_mat   = probs_mat[:, :num_edges_physical]

        per_member_unsafe_rate = targets_mat.mean(axis=0)
        per_member_recall = np.where(
            targets_mat.sum(axis=0) > 0,
            ((preds_mat==1) & (targets_mat==1)).sum(axis=0) / targets_mat.sum(axis=0).clip(1),
            np.nan,
        )
        per_member_fpr = np.where(
            (1-targets_mat).sum(axis=0) > 0,
            ((preds_mat==1) & (targets_mat==0)).sum(axis=0) / (1-targets_mat).sum(axis=0).clip(1),
            np.nan,
        )
        per_member_mean_prob = probs_mat.mean(axis=0)
        edge_ids = np.arange(num_edges_physical)

        fig5, axes = plt.subplots(3, 1, figsize=(S["figsize_large"][0], 12))
        fig5.suptitle(f"Figure 5 — Per-Member Analysis ({num_edges_physical} physical members, fixed topology)",
                      fontweight="bold", fontsize=13)
        ax = axes[0]
        colors_rate = [C["danger"] if r > 0.3 else C["primary"] for r in per_member_unsafe_rate]
        ax.bar(edge_ids, per_member_unsafe_rate, color=colors_rate, edgecolor="none", width=0.8)
        ax.axhline(per_member_unsafe_rate.mean(), color=C["black"], linestyle="--", lw=1.5,
                   label=f"Mean = {per_member_unsafe_rate.mean():.3f}")
        ax.set_xlabel("Member ID"); ax.set_ylabel("Fraction unsafe"); ax.set_title("A — Unsafe Rate per Member"); ax.legend(fontsize=9); ax.set_xlim(-1, num_edges_physical)
        ax = axes[1]
        recall_vals = np.nan_to_num(per_member_recall, nan=-0.05)
        colors_recall = [C["neutral"] if per_member_unsafe_rate[i]==0 else
                         C["danger"] if recall_vals[i]<0.5 else
                         C["accent"] if recall_vals[i]<0.8 else C["primary"]
                         for i in range(num_edges_physical)]
        ax.bar(edge_ids, recall_vals, color=colors_recall, edgecolor="none", width=0.8)
        ax.axhline(np.nanmean(per_member_recall), color=C["black"], linestyle="--", lw=1.5,
                   label=f"Mean recall = {np.nanmean(per_member_recall):.3f}")
        ax.axhline(0.8, color=C["primary"], linestyle=":", lw=1.2, label="Target = 0.80")
        ax.set_xlabel("Member ID"); ax.set_ylabel("Recall"); ax.set_title("B — Per-Member Recall"); ax.set_ylim(-0.1, 1.05); ax.set_xlim(-1, num_edges_physical); ax.legend(fontsize=9)
        ax = axes[2]
        ax.bar(edge_ids, per_member_mean_prob, color=C["secondary"], edgecolor="none", width=0.8, label="Mean P(unsafe)")
        ax.plot(edge_ids, per_member_unsafe_rate, color=C["danger"], marker="o", ms=3, lw=1.2, label="True unsafe rate")
        ax.axhline(thr_primary, color=C["primary"], linestyle="--", lw=1.5, label=f"Threshold ({thr_primary:.2f})")
        ax.set_xlabel("Member ID"); ax.set_ylabel("Prob / Rate"); ax.set_title("C — Mean Predicted Probability vs True Unsafe Rate"); ax.set_xlim(-1, num_edges_physical); ax.legend(fontsize=9)
        plt.tight_layout(); save_fig(fig5, "eval_fig5_per_member"); plt.show()

        members_that_fail = np.where(per_member_unsafe_rate > 0)[0]
        sorted_by_recall  = members_that_fail[np.argsort(per_member_recall[members_that_fail])]
        print(f"\nTop 10 hardest members (lowest recall):")
        print(f"  {'MemberID':>8}  {'UnsafeRate':>10}  {'Recall':>8}  {'FPR':>8}")
        for mid in sorted_by_recall[:10]:
            print(f"  {mid:>8d}  {per_member_unsafe_rate[mid]:>10.3f}  "
                  f"{per_member_recall[mid]:>8.3f}  {per_member_fpr[mid]:>8.3f}")
    else:
        print(f"[Per-member analysis skipped] len(test_probs_arr)={len(test_probs_arr)} "
              f"not divisible by num_edges={num_edges}.")

    # ---- Metrics summary ----
    print(f"\n{'='*65}")
    print("EVALUATION SUMMARY — TrussEdgeSafetyGNN")
    print(f"{'='*65}")
    print(f"  Epochs trained: {len(train_losses)}  (best: {best_epoch+1})")
    print(f"  ROC AUC: {roc_auc_val:.4f}  |  PR AUC: {pr_auc_val:.4f}  |  Brier: {brier:.4f}")
    for label, s in [
        ("Default (thr=0.50)", s_05),
        (f"Val-tuned (thr={thr_primary:.2f})", s_primary),
        (f"Safety (thr={threshold_safety:.2f})", s_safety),
    ]:
        print(f"\n  @ {label}:")
        print(f"    Precision={s['precision']:.4f}  Recall={s['recall']:.4f}  "
              f"F1={s['f1']:.4f}  MCC={s['mcc']:.4f}  FN={s['fn']}")
    print(f"{'='*65}")

    metrics = {
        "roc_auc": float(roc_auc_val), "pr_auc": float(pr_auc_val), "brier_score": float(brier),
        "acc_0.50": float(s_05["accuracy"]), "precision_0.50": float(s_05["precision"]),
        "recall_0.50": float(s_05["recall"]), "f1_0.50": float(s_05["f1"]), "mcc_0.50": float(s_05["mcc"]),
        "threshold_primary": float(thr_primary),
        "acc_primary": float(s_primary["accuracy"]), "precision_primary": float(s_primary["precision"]),
        "recall_primary": float(s_primary["recall"]), "f1_primary": float(s_primary["f1"]),
        "mcc_primary": float(s_primary["mcc"]),
        "tp_primary": int(s_primary["tp"]), "tn_primary": int(s_primary["tn"]),
        "fp_primary": int(s_primary["fp"]), "fn_primary": int(s_primary["fn"]),
        "threshold_safety": float(threshold_safety),
        "acc_safety": float(s_safety["accuracy"]), "precision_safety": float(s_safety["precision"]),
        "recall_safety": float(s_safety["recall"]), "f1_safety": float(s_safety["f1"]),
        "mcc_safety": float(s_safety["mcc"]),
        "tp_safety": int(s_safety["tp"]), "tn_safety": int(s_safety["tn"]),
        "fp_safety": int(s_safety["fp"]), "fn_safety": int(s_safety["fn"]),
        "false_negative_rate": float(s_safety["fn"] / max(s_safety["tp"] + s_safety["fn"], 1)),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(train_losses[-1]), "final_val_loss": float(val_losses[-1]),
    }
    if per_member_recall is not None:
        metrics["per_member_recall_mean"] = float(np.nanmean(per_member_recall))
        metrics["per_member_recall_min"]  = float(np.nanmin(per_member_recall))
        metrics["per_member_fpr_mean"]    = float(np.nanmean(per_member_fpr))

    print("\nEvaluation complete. Figures: fig1-fig5  |  Metrics dict: eval_out['metrics']")

    return {
        "metrics": metrics,
        "fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5,
        "thr_primary": thr_primary, "threshold_f1": threshold_f1, "threshold_safety": threshold_safety,
        "roc_auc": roc_auc_val, "pr_auc": pr_auc_val, "brier": brier,
        "test_probs_arr": test_probs_arr, "test_true": test_true,
        "s_primary": s_primary, "s_f1": s_f1, "s_safety": s_safety, "s_05": s_05,
        "per_member_recall": per_member_recall,
        "per_member_fpr": per_member_fpr,
        "per_member_unsafe_rate": per_member_unsafe_rate,
    }


# =============================================================================
# STAGE 4 — EXPORT
# =============================================================================

def run_export(
    pre:       dict,
    train_out: dict,
    eval_out:  dict,
) -> dict[str, Any]:
    """Save all artifacts to SM_EXPORT_PATH and SM_DATA_PATH.

    Returns
    -------
    dict with keys:
        artifact_stem, models_dir, data_dir, all_files
    """
    model         = train_out["model"]
    loss_fn       = train_out["loss_fn"]
    pos_weight    = train_out.get("pos_weight", train_out.get("focal_alpha", 4.0))
    best_val_loss = train_out["best_val_loss"]
    best_epoch    = train_out["best_epoch"]
    train_losses  = train_out["train_losses"]
    val_losses    = train_out["val_losses"]
    LR            = train_out["LR"]
    EPOCHS        = train_out["EPOCHS"]
    PATIENCE      = train_out["PATIENCE"]
    CKPT_PATH     = train_out["CKPT_PATH"]
    batch_size    = train_out["batch_size"]
    test_probs    = train_out["test_probs"]
    test_targets  = train_out["test_targets"]

    metrics          = eval_out["metrics"]
    thr_primary      = eval_out["thr_primary"]
    threshold_f1     = eval_out["threshold_f1"]
    threshold_safety = eval_out["threshold_safety"]
    roc_auc_val      = eval_out["roc_auc"]
    pr_auc_val       = eval_out["pr_auc"]
    brier            = eval_out["brier"]
    test_true        = eval_out["test_true"]

    node_cols         = pre["node_cols"]
    edge_cols         = pre["edge_cols"]
    node_feature_means = pre["node_feature_means"]
    node_feature_stds  = pre["node_feature_stds"]
    edge_feature_means = pre["edge_feature_means"]
    edge_feature_stds  = pre["edge_feature_stds"]
    train_pos_rate    = pre["train_pos_rate"]
    train_dataset     = pre["train_dataset"]
    val_dataset       = pre["val_dataset"]
    test_dataset      = pre["test_dataset"]
    nodes_df          = pre["nodes_df"]
    edges_df          = pre["edges_df"]
    node_csv_path     = pre["node_csv_path"]
    edge_csv_path     = pre["edge_csv_path"]
    edge_index_path   = pre["edge_index_path"]

    # ---- Artifact stem ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_stem = (
        f"ID{ts}"
        f"_LR{LR:.0e}"
        f"_EP{len(train_losses)}"
        f"_BS{batch_size}"
        f"_PW{pos_weight:.1f}"
        f"_ROC{roc_auc_val:.3f}"
    )
    print(f"Artifact stem: {artifact_stem}")

    models_dir = Path(config.SM_EXPORT_PATH) / artifact_stem
    data_dir   = Path(config.SM_DATA_PATH)   / artifact_stem
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model checkpoint
    ckpt_src    = Path(CKPT_PATH)
    ckpt_target = models_dir / f"{artifact_stem}.pth"
    if ckpt_src.exists():
        shutil.copy2(ckpt_src, ckpt_target)
        print(f"Checkpoint copied:   {ckpt_target.name}")
    else:
        torch.save({"model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss, "best_epoch": best_epoch}, ckpt_target)
        print(f"Warning: {CKPT_PATH} not found — saved current state to {ckpt_target.name}")

    # 2. Norm stats
    norm_stats_src    = Path(config.DATA_IO_PATH) / "norm_stats.pt"
    norm_stats_target = models_dir / f"{artifact_stem}_norm_stats.pt"
    if norm_stats_src.exists():
        shutil.copy2(norm_stats_src, norm_stats_target)
        print(f"Norm stats copied:   {norm_stats_target.name}")
    else:
        torch.save({"node_means": node_feature_means.to_dict(), "node_stds": node_feature_stds.to_dict(),
                    "edge_means": edge_feature_means.to_dict(), "edge_stds": edge_feature_stds.to_dict(),
                    "node_cols": list(node_cols), "edge_cols": list(edge_cols)}, norm_stats_target)
        print(f"Norm stats rebuilt:  {norm_stats_target.name}")

    # 3. Topology
    edge_index_target = models_dir / f"{artifact_stem}_edge_index.json"
    shutil.copy2(edge_index_path, edge_index_target)
    print(f"Topology copied:     {edge_index_target.name}")

    # 4. Scalers JSON
    scalers_path = models_dir / f"{artifact_stem}_scalers.json"
    with open(scalers_path, "w") as f:
        json.dump({"node_cols": list(node_cols), "edge_cols": list(edge_cols),
                   "node_mean": node_feature_means.to_dict(), "node_std": node_feature_stds.to_dict(),
                   "edge_mean": edge_feature_means.to_dict(), "edge_std": edge_feature_stds.to_dict()}, f, indent=2)
    print(f"Scalers JSON saved:  {scalers_path.name}")

    # 5. Inference config
    inference_config_path = models_dir / f"{artifact_stem}_inference_config.json"
    with open(inference_config_path, "w") as f:
        json.dump({
            "model_class":           type(model).__name__,
            "node_features_dim":     len(node_cols),
            "edge_features_dim":     len(edge_cols),
            "hidden_dim":            getattr(model, "hidden_dim", "n/a"),
            "num_layers":            getattr(model, "num_layers", "n/a"),
            "use_batch_norm":        getattr(model, "use_batch_norm", "n/a"),
            "use_residuals":         getattr(model, "use_residuals", "n/a"),
            "dropout_p":             getattr(model, "dropout_p", "n/a"),
            "node_cols":             list(node_cols), "edge_cols": list(edge_cols),
            "clip_sigma":            5.0,
            "threshold_primary":     float(thr_primary),
            "threshold_f1":          float(threshold_f1),
            "threshold_safety":      float(threshold_safety),
            "recommended_threshold": float(thr_primary),
        }, f, indent=2)
    print(f"Inference config:    {inference_config_path.name}")

    # 6. Metrics JSON
    metrics_path = data_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics JSON saved:  {metrics_path.name}")

    # 7. Training report
    report_lines = [
        "SURROGATE MODEL TRAINING REPORT", "=" * 80,
        f"Artifact:      {artifact_stem}", f"Generated:     {ts}", "",
        "DATA SOURCES", "-" * 80,
        f"Node CSV:      {node_csv_path}", f"Edge CSV:      {edge_csv_path}",
        f"Edge index:    {edge_index_path}",
        f"Total samples: {len(train_dataset)+len(val_dataset)+len(test_dataset)}",
        f"Train/Val/Test:{len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}",
        f"Positive rate (train): {train_pos_rate:.4f}",
        f"Positive labels (full): {int((edges_df['Utilization']>1).sum())}",
        f"Negative labels (full): {int((edges_df['Utilization']<=1).sum())}", "",
        "MODEL CONFIGURATION", "-" * 80,
        f"Class:      {type(model).__name__}",
        f"Device:     {next(model.parameters()).device}",
        f"Hidden dim: {getattr(model,'hidden_dim','n/a')}",
        f"Num layers: {getattr(model,'num_layers','n/a')}",
        f"Batch norm: {getattr(model,'use_batch_norm','n/a')}",
        f"Residuals:  {getattr(model,'use_residuals','n/a')}",
        f"Dropout p:  {getattr(model,'dropout_p','n/a')}",
        f"Node features: {', '.join(node_cols)}", f"Edge features: {', '.join(edge_cols)}", "",
        "TRAINING HYPERPARAMETERS", "-" * 80,
        f"Learning rate:     {LR}", f"Max epochs:        {EPOCHS}",
        f"Early stop pat.:   {PATIENCE}", f"Actual epochs run: {len(train_losses)}",
        f"Best epoch:        {best_epoch+1}", f"Batch size:        {batch_size}",
        f"Loss:              {type(loss_fn).__name__}",
        f"Pos weight:        {pos_weight:.6f}",
        f"Best val loss:     {best_val_loss:.6f}", "",
        "EVALUATION SUMMARY", "-" * 80,
        f"ROC AUC: {roc_auc_val:.6f}  |  PR AUC: {pr_auc_val:.6f}  |  Brier: {brier:.6f}", "",
        f"{'Metric':<22} {'thr=0.50':>10} {'val-tuned':>10} {'safety':>10}", "-"*55,
        f"{'Accuracy':<22} {metrics['acc_0.50']:>10.4f} {metrics['acc_primary']:>10.4f} {metrics['acc_safety']:>10.4f}",
        f"{'Precision':<22} {metrics['precision_0.50']:>10.4f} {metrics['precision_primary']:>10.4f} {metrics['precision_safety']:>10.4f}",
        f"{'Recall (unsafe)':<22} {metrics['recall_0.50']:>10.4f} {metrics['recall_primary']:>10.4f} {metrics['recall_safety']:>10.4f}",
        f"{'F1':<22} {metrics['f1_0.50']:>10.4f} {metrics['f1_primary']:>10.4f} {metrics['f1_safety']:>10.4f}",
        f"{'MCC':<22} {metrics['mcc_0.50']:>10.4f} {metrics['mcc_primary']:>10.4f} {metrics['mcc_safety']:>10.4f}", "",
        f"Threshold (val-tuned): {thr_primary:.4f}",
        f"Threshold (max-F1):    {threshold_f1:.4f}",
        f"Threshold (safety):    {threshold_safety:.4f}",
        f"False negative rate:   {metrics['false_negative_rate']:.4f}",
        f"TP={metrics['tp_safety']}  TN={metrics['tn_safety']}  FP={metrics['fp_safety']}  FN={metrics['fn_safety']}", "",
        "FILES", "-" * 80,
        f"Checkpoint:       {ckpt_target.name}",
        f"Norm stats (.pt): {norm_stats_target.name}",
        f"Topology:         {edge_index_target.name}",
        f"Scalers JSON:     {scalers_path.name}",
        f"Inference config: {inference_config_path.name}",
        f"Metrics JSON:     {metrics_path.name}", "",
        "TRAINING HISTORY", "-" * 80, "Epoch,TrainLoss,ValLoss",
    ] + [f"{i},{tl:.10f},{vl:.10f}" for i, (tl, vl) in enumerate(zip(train_losses, val_losses), 1)]

    report_path = models_dir / f"{artifact_stem}_training_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Training report:     {report_path.name}")

    # 8. Figures
    fig_map = {
        "fig1": "eval_fig1_training_dynamics",
        "fig2": "eval_fig2_threshold_analysis",
        "fig3": "eval_fig3_prediction_quality",
        "fig4": "eval_fig4_confusion_matrices",
        "fig5": "eval_fig5_per_member",
    }
    for var_name, stem in fig_map.items():
        fig = eval_out.get(var_name)
        if fig is not None:
            out = data_dir / f"{artifact_stem}_{stem}.png"
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"Figure saved:        {out.name}")
        else:
            print(f"Figure '{var_name}' not found — skipped.")

    # 9. Raw predictions
    np.savetxt(data_dir / "test_probs.csv",   test_probs,   delimiter=",")
    np.savetxt(data_dir / "test_targets.csv", test_targets, delimiter=",")
    print("Raw predictions and targets saved.")

    all_files = sorted(set(
        [ckpt_target, norm_stats_target, edge_index_target,
         scalers_path, inference_config_path, report_path, metrics_path]
        + list(data_dir.glob("*"))
    ))

    print(f"\n{'='*65}")
    print("EXPORT COMPLETE")
    print(f"{'='*65}")
    print(f"  Models dir: {models_dir}")
    print(f"  Data dir:   {data_dir}")
    print(f"\n  Files saved:")
    for fp in all_files:
        size_kb = fp.stat().st_size / 1024
        print(f"    {fp.name:<58} {size_kb:6.1f} KB")

    return {
        "artifact_stem": artifact_stem,
        "models_dir":    models_dir,
        "data_dir":      data_dir,
        "all_files":     all_files,
    }
