# --- Training loop (follows c21_preprocessing_v4.py) ---
#
# Requires from preprocessing: model, loss_fn, train_dataloader, val_dataloader,
#                              test_dataloader, train_dataset, val_dataset,
#                              test_dataset, device, focal_alpha
#
# Changes vs v2:
#   1. focal_alpha overridden to 0.5 — the preprocessing formula (1 - pos_rate)
#      gives 0.81 at 19% positive rate which is too aggressive. 0.5 is the
#      empirically better value from training runs.
#   2. experiment_mode block cleaned up — loss_functions dict was defined but
#      never used; removed to avoid confusion.
#   3. CKPT_PATH uses config.DATA_IO_PATH so checkpoint lands next to the other
#      data files, not in the working directory.
#   4. Hyperparameters block prints CKPT_PATH correctly (was printing string
#      literal "surrogate_v4_checkpoint.pth" regardless of actual path).

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import config
from c21_surrogate_model_v4 import FocalLoss

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

EPOCHS        = 150
LR            = 3e-4
PATIENCE      = 40
LR_FACTOR     = 0.5
LR_PATIENCE   = 10
LR_MIN        = 1e-6
GRAD_CLIP     = 1.0
CKPT_PATH     = config.DATA_IO_PATH / "surrogate_v4_checkpoint.pth"

# Override preprocessing focal_alpha — the 1-pos_rate formula overcorrects
# at 19% positive rate. 0.5 is empirically better (see training run history).
focal_alpha       = 0.5
DEFAULT_THRESHOLD = 0.35
min_precision     = 0.40   # used in threshold sweep and evaluation script

print("Hyperparameters:")
for k, v in {
    "EPOCHS": EPOCHS, "LR": LR, "PATIENCE": PATIENCE,
    "LR_FACTOR": LR_FACTOR, "LR_PATIENCE": LR_PATIENCE,
    "LR_MIN": LR_MIN, "GRAD_CLIP": GRAD_CLIP,
    "CKPT_PATH": CKPT_PATH, "focal_alpha": focal_alpha,
    "DEFAULT_THRESHOLD": DEFAULT_THRESHOLD,
    "min_precision": min_precision,
}.items():
    print(f"  {k}: {v}")

# ============================================================================
# OPTIMIZER, SCHEDULER, LOSS
# ============================================================================

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=LR_FACTOR,
    patience=LR_PATIENCE,
    min_lr=LR_MIN,
)

loss_fn = FocalLoss(alpha=focal_alpha, gamma=2.0)
print(f"FocalLoss(alpha={focal_alpha:.4f}, gamma=2.0)")

# ============================================================================
# TRAINING LOOP
# ============================================================================

train_losses      = []
val_losses        = []
epoch_history     = []
best_val_loss     = float("inf")
best_state        = None
best_epoch        = -1
epochs_no_improve = 0

print(f"\nStarting training: {EPOCHS} epochs, early stopping patience={PATIENCE}")
print("-" * 70)

for epoch in range(EPOCHS):

    # ---- TRAIN ----
    model.train()
    epoch_train_loss = 0.0
    for batch in train_dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        preds = model(batch.x, batch.edge_index, batch.edge_attr)
        loss  = loss_fn(preds, batch.y)
        loss.backward()
        if GRAD_CLIP is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        epoch_train_loss += loss.item() * batch.num_graphs
    epoch_train_loss /= len(train_dataset)
    train_losses.append(epoch_train_loss)

    # ---- VALIDATE ----
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
    current_lr = optimizer.param_groups[0]['lr']

    # ---- CHECKPOINT ----
    if epoch_val_loss < best_val_loss:
        best_val_loss     = float(epoch_val_loss)
        best_epoch        = int(epoch)
        best_state        = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    # ---- LOGGING ----
    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch+1:03d}  "
            f"train={epoch_train_loss:.6f}  "
            f"val={epoch_val_loss:.6f}  "
            f"lr={current_lr:.2e}  "
            f"no_improve={epochs_no_improve}/{PATIENCE}"
        )
        epoch_history.append((epoch + 1, epoch_train_loss, epoch_val_loss, current_lr))

    # ---- EARLY STOPPING ----
    if epochs_no_improve >= PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch + 1} "
              f"(no improvement for {PATIENCE} epochs).")
        break

print("-" * 70)

# ============================================================================
# RESTORE BEST CHECKPOINT
# ============================================================================

if best_state is not None:
    model.load_state_dict(best_state)
    print(f"Restored best checkpoint from epoch {best_epoch + 1}  "
          f"val_loss={best_val_loss:.6f}")
else:
    print("Warning: best_state was not set; using last epoch weights.")

# ============================================================================
# SAVE CHECKPOINT
# ============================================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "best_val_loss":    best_val_loss,
        "best_epoch":       best_epoch,
        "focal_alpha":      focal_alpha,
        "train_losses":     train_losses,
        "val_losses":       val_losses,
    },
    CKPT_PATH,
)
print(f"Checkpoint saved: {CKPT_PATH}")

# ============================================================================
# HELPERS
# ============================================================================

def collect_preds(dataloader, model, device):
    """Returns (probs, targets) as flat numpy arrays."""
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


def classification_report_at_threshold(probs, targets, threshold, label=""):
    preds_binary     = (probs >= threshold).astype(int)
    cm               = confusion_matrix(targets.astype(int), preds_binary)
    recall_unsafe    = recall_score(targets, preds_binary, pos_label=1, zero_division=0)
    precision_unsafe = precision_score(targets, preds_binary, pos_label=1, zero_division=0)
    f1_unsafe        = f1_score(targets, preds_binary, pos_label=1, zero_division=0)
    print(f"\n{'='*60}")
    print(f"{label}  (threshold={threshold:.2f})")
    print(f"{'='*60}")
    print(f"Confusion matrix (rows=actual, cols=predicted):")
    print(f"              Pred Safe  Pred Unsafe")
    print(f"  Act Safe    {cm[0,0]:9d}  {cm[0,1]:11d}")
    print(f"  Act Unsafe  {cm[1,0]:9d}  {cm[1,1]:11d}")
    print()
    print(classification_report(
        targets.astype(int), preds_binary,
        target_names=["Safe (0)", "Unsafe (1)"], digits=4,
    ))
    print(f"Unsafe class  ->  Recall: {recall_unsafe:.4f}  "
          f"Precision: {precision_unsafe:.4f}  F1: {f1_unsafe:.4f}")
    return recall_unsafe, precision_unsafe, f1_unsafe

# ============================================================================
# THRESHOLD SWEEP ON VALIDATION SET
# ============================================================================

print("\n--- Threshold sweep on validation set ---")
val_probs, val_targets = collect_preds(val_dataloader, model, device)

try:
    val_auc = roc_auc_score(val_targets, val_probs)
    print(f"Val AUC-ROC: {val_auc:.4f}")
except ValueError:
    print("Val AUC-ROC: n/a (only one class present in val targets)")
    val_auc = None

thresholds     = np.arange(0.10, 0.65, 0.05)
best_threshold = DEFAULT_THRESHOLD
best_recall    = -1.0
sweep_results  = []

for t in thresholds:
    preds_bin   = (val_probs >= t).astype(int)
    recall_u    = recall_score(val_targets, preds_bin, pos_label=1, zero_division=0)
    precision_u = precision_score(val_targets, preds_bin, pos_label=1, zero_division=0)
    f1_u        = f1_score(val_targets, preds_bin, pos_label=1, zero_division=0)
    sweep_results.append((t, recall_u, precision_u, f1_u))
    if recall_u > best_recall and precision_u >= min_precision:
        best_recall    = recall_u
        best_threshold = t

print(f"\n{'Threshold':>10}  {'Recall(unsafe)':>15}  "
      f"{'Precision(unsafe)':>18}  {'F1(unsafe)':>12}")
print("-" * 62)
for t, r, p, f in sweep_results:
    marker = " <-- selected" if abs(t - best_threshold) < 1e-6 else ""
    print(f"{t:10.2f}  {r:15.4f}  {p:18.4f}  {f:12.4f}{marker}")

print(f"\nSelected threshold: {best_threshold:.2f}  "
      f"(max recall >= {min_precision:.0%} precision constraint)")

classification_report_at_threshold(
    val_probs, val_targets, best_threshold, label="VALIDATION SET"
)

# ============================================================================
# TEST SET EVALUATION
# ============================================================================

print("\n--- Test set evaluation ---")
test_probs, test_targets = collect_preds(test_dataloader, model, device)

try:
    test_auc = roc_auc_score(test_targets, test_probs)
    print(f"Test AUC-ROC: {test_auc:.4f}")
except ValueError:
    print("Test AUC-ROC: n/a")
    test_auc = None

classification_report_at_threshold(
    test_probs, test_targets, best_threshold, label="TEST SET"
)
classification_report_at_threshold(
    test_probs, test_targets, 0.5,
    label="TEST SET (default threshold=0.50, for reference)"
)

# ============================================================================
# TEST LOSS
# ============================================================================

model.eval()
test_loss = 0.0
with torch.no_grad():
    for batch in test_dataloader:
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index, batch.edge_attr)
        loss  = loss_fn(preds, batch.y)
        test_loss += loss.item() * batch.num_graphs
test_loss /= len(test_dataset)
print(f"\nTest focal loss: {test_loss:.6f}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)
print(f"  Best epoch:       {best_epoch + 1}")
print(f"  Best val loss:    {best_val_loss:.6f}")
print(f"  Test focal loss:  {test_loss:.6f}")
if val_auc:  print(f"  Val  AUC-ROC:     {val_auc:.4f}")
if test_auc: print(f"  Test AUC-ROC:     {test_auc:.4f}")
print(f"  Decision threshold (val-tuned): {best_threshold:.2f}")
print(f"  Checkpoint: {CKPT_PATH}")
print("=" * 70)