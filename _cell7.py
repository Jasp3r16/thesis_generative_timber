# --- Final evaluation export (follows c21_evaluation_v3.py) ---
#
# Requires from training:      model, loss_fn, focal_alpha, best_val_loss,
#                              best_epoch, train_losses, val_losses,
#                              LR, EPOCHS, PATIENCE, batch_size, CKPT_PATH
# Requires from preprocessing: node_cols, edge_cols, node_feature_means,
#                              node_feature_stds, edge_feature_means,
#                              edge_feature_stds, train_pos_rate,
#                              train_dataset, val_dataset, test_dataset,
#                              nodes_df, edges_df, node_csv_path,
#                              edge_csv_path, edge_index_path
# Requires from evaluation:    metrics, test_probs, test_true,
#                              thr_primary, threshold_f1, threshold_safety,
#                              roc_auc, pr_auc, brier,
#                              fig1, fig2, fig3, fig4, fig5
#
# Artifact folder structure:
#   SM_EXPORT_PATH / {stem} /   <- checkpoint, norm_stats, topology,
#                                  scalers, inference_config, report
#   SM_DATA_PATH   / {stem} /   <- metrics.json, figures, raw predictions

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import config

# ============================================================================
# GUARD
# ============================================================================

required = [
    # from preprocessing
    "node_cols", "edge_cols",
    "node_feature_means", "node_feature_stds",
    "edge_feature_means", "edge_feature_stds",
    "train_pos_rate", "train_dataset", "val_dataset", "test_dataset",
    "nodes_df", "edges_df", "node_csv_path", "edge_csv_path", "edge_index_path",
    "batch_size",
    # from training
    "model", "loss_fn", "focal_alpha", "best_val_loss", "best_epoch",
    "train_losses", "val_losses", "LR", "EPOCHS", "PATIENCE",
    "CKPT_PATH", "test_probs", "test_targets",
    # from evaluation
    "metrics", "test_true",
    "thr_primary", "threshold_f1", "threshold_safety",
    "roc_auc", "pr_auc", "brier",
]
missing = [v for v in required if v not in globals()]
if missing:
    raise RuntimeError(
        "Missing variables — run preprocessing, training, and evaluation cells first.\n"
        "Missing: " + ", ".join(missing)
    )

# ============================================================================
# ARTIFACT STEM
# ============================================================================

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
artifact_stem = (
    f"ID{ts}"
    f"_LR{LR:.0e}"
    f"_EP{len(train_losses)}"
    f"_BS{batch_size}"
    f"_FA{focal_alpha:.2f}"
    f"_ROC{roc_auc:.3f}"
)
print(f"Artifact stem: {artifact_stem}")

# ============================================================================
# DIRECTORIES
# ============================================================================

models_dir = config.SM_EXPORT_PATH / artifact_stem
data_dir   = config.SM_DATA_PATH   / artifact_stem
models_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. MODEL CHECKPOINT
# ============================================================================

ckpt_src    = Path(CKPT_PATH)
ckpt_target = models_dir / f"{artifact_stem}.pth"

if ckpt_src.exists():
    shutil.copy2(ckpt_src, ckpt_target)
    print(f"Checkpoint copied:   {ckpt_target.name}")
else:
    torch.save(
        {"model_state_dict": model.state_dict(),
         "best_val_loss": best_val_loss, "best_epoch": best_epoch},
        ckpt_target,
    )
    print(f"Warning: {CKPT_PATH} not found — saved current state to {ckpt_target.name}")

# ============================================================================
# 2. NORMALISATION STATS
# ============================================================================

norm_stats_src    = Path(config.DATA_IO_PATH) / "norm_stats.pt"
norm_stats_target = models_dir / f"{artifact_stem}_norm_stats.pt"

if norm_stats_src.exists():
    shutil.copy2(norm_stats_src, norm_stats_target)
    print(f"Norm stats copied:   {norm_stats_target.name}")
else:
    torch.save(
        {"node_means": node_feature_means.to_dict(),
         "node_stds":  node_feature_stds.to_dict(),
         "edge_means": edge_feature_means.to_dict(),
         "edge_stds":  edge_feature_stds.to_dict(),
         "node_cols":  list(node_cols),
         "edge_cols":  list(edge_cols)},
        norm_stats_target,
    )
    print(f"Norm stats rebuilt:  {norm_stats_target.name}")

# ============================================================================
# 3. TOPOLOGY
# ============================================================================

edge_index_target = models_dir / f"{artifact_stem}_edge_index.json"
shutil.copy2(edge_index_path, edge_index_target)
print(f"Topology copied:     {edge_index_target.name}")

# ============================================================================
# 4. SCALERS JSON (human-readable, for GH/Rhino inference)
# ============================================================================

scalers_path = models_dir / f"{artifact_stem}_scalers.json"
with open(scalers_path, "w") as f:
    json.dump({
        "node_cols":  list(node_cols),
        "edge_cols":  list(edge_cols),
        "node_mean":  node_feature_means.to_dict(),
        "node_std":   node_feature_stds.to_dict(),
        "edge_mean":  edge_feature_means.to_dict(),
        "edge_std":   edge_feature_stds.to_dict(),
    }, f, indent=2)
print(f"Scalers JSON saved:  {scalers_path.name}")

# ============================================================================
# 5. INFERENCE CONFIG
# ============================================================================

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
        "node_cols":             list(node_cols),
        "edge_cols":             list(edge_cols),
        "clip_sigma":            5.0,
        "threshold_primary":     float(thr_primary),
        "threshold_f1":          float(threshold_f1),
        "threshold_safety":      float(threshold_safety),
        "recommended_threshold": float(thr_primary),
        # output is P(unsafe) — predict unsafe if prob >= recommended_threshold
    }, f, indent=2)
print(f"Inference config:    {inference_config_path.name}")

# ============================================================================
# 6. METRICS JSON
# ============================================================================

metrics_path = data_dir / "metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics JSON saved:  {metrics_path.name}")

# ============================================================================
# 7. TRAINING REPORT
# ============================================================================

focal_gamma = float(getattr(loss_fn, "gamma", 2.0))
s_primary   = metrics  # shorthand for readability below

report_lines = [
    "SURROGATE MODEL TRAINING REPORT",
    "=" * 80,
    f"Artifact:      {artifact_stem}",
    f"Generated:     {ts}",
    "",
    "DATA SOURCES",
    "-" * 80,
    f"Node CSV:      {node_csv_path}",
    f"Edge CSV:      {edge_csv_path}",
    f"Edge index:    {edge_index_path}",
    f"Total samples: {len(train_dataset) + len(val_dataset) + len(test_dataset)}",
    f"Train/Val/Test:{len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}",
    f"Positive rate (train): {train_pos_rate:.4f}",
    f"Positive labels (full): {int((edges_df['Utilization'] > 1).sum())}",
    f"Negative labels (full): {int((edges_df['Utilization'] <= 1).sum())}",
    "",
    "MODEL CONFIGURATION",
    "-" * 80,
    f"Class:         {type(model).__name__}",
    f"Device:        {next(model.parameters()).device}",
    f"Hidden dim:    {getattr(model, 'hidden_dim', 'n/a')}",
    f"Num layers:    {getattr(model, 'num_layers', 'n/a')}",
    f"Batch norm:    {getattr(model, 'use_batch_norm', 'n/a')}",
    f"Residuals:     {getattr(model, 'use_residuals', 'n/a')}",
    f"Dropout p:     {getattr(model, 'dropout_p', 'n/a')}",
    f"Node features: {', '.join(node_cols)}",
    f"Edge features: {', '.join(edge_cols)}",
    "",
    "TRAINING HYPERPARAMETERS",
    "-" * 80,
    f"Learning rate:     {LR}",
    f"Max epochs:        {EPOCHS}",
    f"Early stop pat.:   {PATIENCE}",
    f"Actual epochs run: {len(train_losses)}",
    f"Best epoch:        {best_epoch + 1}",
    f"Batch size:        {batch_size}",
    f"Loss:              {type(loss_fn).__name__}",
    f"Focal alpha:       {focal_alpha:.6f}",
    f"Focal gamma:       {focal_gamma:.6f}",
    f"Best val loss:     {best_val_loss:.6f}",
    "",
    "EVALUATION SUMMARY",
    "-" * 80,
    f"ROC AUC:           {roc_auc:.6f}",
    f"PR  AUC:           {pr_auc:.6f}",
    f"Brier score:       {brier:.6f}",
    "",
    f"{'Metric':<22} {'thr=0.50':>10} {'val-tuned':>10} {'safety':>10}",
    "-" * 55,
    f"{'Accuracy':<22} {metrics['acc_0.50']:>10.4f} {metrics['acc_primary']:>10.4f} {metrics['acc_safety']:>10.4f}",
    f"{'Precision':<22} {metrics['precision_0.50']:>10.4f} {metrics['precision_primary']:>10.4f} {metrics['precision_safety']:>10.4f}",
    f"{'Recall (unsafe)':<22} {metrics['recall_0.50']:>10.4f} {metrics['recall_primary']:>10.4f} {metrics['recall_safety']:>10.4f}",
    f"{'F1':<22} {metrics['f1_0.50']:>10.4f} {metrics['f1_primary']:>10.4f} {metrics['f1_safety']:>10.4f}",
    f"{'MCC':<22} {metrics['mcc_0.50']:>10.4f} {metrics['mcc_primary']:>10.4f} {metrics['mcc_safety']:>10.4f}",
    "",
    f"Threshold (val-tuned):  {thr_primary:.4f}",
    f"Threshold (max-F1):     {threshold_f1:.4f}",
    f"Threshold (safety):     {threshold_safety:.4f}",
    f"False negative rate:    {metrics['false_negative_rate']:.4f}",
    f"TP={metrics['tp_safety']}  TN={metrics['tn_safety']}  "
    f"FP={metrics['fp_safety']}  FN={metrics['fn_safety']}",
    "",
    "FILES",
    "-" * 80,
    f"Checkpoint:       {ckpt_target.name}",
    f"Norm stats (.pt): {norm_stats_target.name}",
    f"Topology:         {edge_index_target.name}",
    f"Scalers JSON:     {scalers_path.name}",
    f"Inference config: {inference_config_path.name}",
    f"Metrics JSON:     {metrics_path.name}",
    "",
    "TRAINING HISTORY",
    "-" * 80,
    "Epoch,TrainLoss,ValLoss",
] + [f"{i},{tl:.10f},{vl:.10f}"
     for i, (tl, vl) in enumerate(zip(train_losses, val_losses), 1)]

report_path = models_dir / f"{artifact_stem}_training_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"Training report:     {report_path.name}")

# ============================================================================
# 8. FIGURES  (fig1-fig5 from evaluation v3)
# ============================================================================

fig_map = {
    "fig1": "eval_fig1_training_dynamics",
    "fig2": "eval_fig2_threshold_analysis",
    "fig3": "eval_fig3_prediction_quality",
    "fig4": "eval_fig4_confusion_matrices",
    "fig5": "eval_fig5_per_member",
}
for var_name, stem in fig_map.items():
    fig = globals().get(var_name)
    if fig is not None:
        out = data_dir / f"{artifact_stem}_{stem}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Figure saved:        {out.name}")
    else:
        print(f"Figure '{var_name}' not found — skipped.")

# ============================================================================
# 9. RAW PREDICTIONS + TARGETS
# ============================================================================

np.savetxt(data_dir / "test_probs.csv",   test_probs,   delimiter=",")
np.savetxt(data_dir / "test_targets.csv", test_targets, delimiter=",")
print("Raw predictions and targets saved.")

# ============================================================================
# SUMMARY
# ============================================================================

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