# Plot: binary-classification diagnostics (loss, distributions, ROC/PR, confusion)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

required_vars = ["train_losses", "val_losses", "test_preds", "test_targets"]
missing = [name for name in required_vars if name not in globals()]
if missing:
    raise RuntimeError(
        "Missing required variables from the training cell: "
        + ", ".join(missing)
        + ". Run Cell 2 first, then run this cell."
    )

# Create epoch history
epochs_range = np.arange(1, len(train_losses) + 1)

test_probs = test_preds.flatten()
test_true = test_targets.flatten().astype(int)

fpr, tpr, roc_thresholds = roc_curve(test_true, test_probs)
roc_auc = auc(fpr, tpr)

precision, recall, pr_thresholds = precision_recall_curve(test_true, test_probs)
pr_auc = auc(recall, precision)

# Threshold from max F1 on PR curve
f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
best_idx = int(np.argmax(f1_scores))
best_threshold = float(pr_thresholds[max(0, min(best_idx, len(pr_thresholds) - 1))]) if len(pr_thresholds) > 0 else 0.5

pred_05 = (test_probs >= 0.5).astype(int)
pred_best = (test_probs >= best_threshold).astype(int)

acc_05 = accuracy_score(test_true, pred_05)
acc_best = accuracy_score(test_true, pred_best)

cm = confusion_matrix(test_true, pred_best)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Training and validation loss curves
ax = axes[0, 0]
ax.plot(epochs_range, train_losses, "b-", label="Train Loss", linewidth=2, marker="o", markersize=4)
ax.plot(epochs_range, val_losses, color="orange", label="Val Loss", linewidth=2, marker="s", markersize=4)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("BCE Loss", fontsize=11)
ax.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: Prediction probability distribution by class
ax = axes[0, 1]
ax.hist(test_probs[test_true == 0], bins=25, alpha=0.65, label="Safe (target=0)", color="green", edgecolor="black")
ax.hist(test_probs[test_true == 1], bins=25, alpha=0.65, label="Unsafe (target=1)", color="red", edgecolor="black")
ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="thr=0.50")
ax.axvline(best_threshold, color="purple", linestyle=":", linewidth=1.8, label=f"thr*= {best_threshold:.3f}")
ax.set_xlabel("Predicted Probability", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.set_title("Probability Distribution by Class", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

# Plot 3: ROC + PR summary
ax = axes[1, 0]
ax.plot(fpr, tpr, color="darkorange", linewidth=2, label=f"ROC AUC = {roc_auc:.3f}")
ax.plot([0, 1], [0, 1], color="navy", linewidth=1.5, linestyle="--", label="Random")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curve", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)

# Plot 4: Confusion matrix at best threshold
ax = axes[1, 1]
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.set_title(f"Confusion Matrix (thr*={best_threshold:.3f})", fontsize=12, fontweight="bold")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Safe", "Unsafe"])
ax.set_yticklabels(["Safe", "Unsafe"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black", fontsize=11)

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# Expose this figure for downstream reporting cell.
training_visuals_fig = fig

tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

print("\nTraining complete (binary classification setup)")
print(f"Final Train Loss: {train_losses[-1]:.6f}")
print(f"Final Val Loss:   {val_losses[-1]:.6f}")
print("\nTest Set Metrics")
print(f"  ROC AUC:      {roc_auc:.4f}")
print(f"  PR AUC:       {pr_auc:.4f}")
print(f"  Accuracy@0.5: {acc_05:.4f}")
print(f"  Accuracy@thr*: {acc_best:.4f}")
print(f"  TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")

# Binary evaluation graphs + compatibility variables for export cell
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

print("Environment ready for binary evaluation diagnostics.")

required_vars = ["test_preds", "test_targets", "epochs", "batch_size"]
missing = [name for name in required_vars if name not in globals()]
if missing:
    raise RuntimeError(
        "Missing required variables from earlier cells: "
        + ", ".join(missing)
        + ". Run Cell 2 first, then run this cell."
    )

# Use predictions already computed in Cell 2 (fast; avoids another full forward pass).
test_probs = test_preds.flatten()
test_true = test_targets.flatten().astype(int)

# Curves
fpr, tpr, roc_thresholds = roc_curve(test_true, test_probs)
roc_auc = auc(fpr, tpr)
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(test_true, test_probs)
pr_auc = auc(recall_curve, precision_curve)

# Select threshold by max F1 on PR curve
f1_curve = (2 * precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-12)
best_idx = int(np.argmax(f1_curve))
best_threshold = float(pr_thresholds[max(0, min(best_idx, len(pr_thresholds) - 1))]) if len(pr_thresholds) > 0 else 0.5

# Predictions at thresholds
pred_05 = (test_probs >= 0.5).astype(int)
pred_best = (test_probs >= best_threshold).astype(int)

# Basic scores
acc_05 = accuracy_score(test_true, pred_05)
precision_05 = precision_score(test_true, pred_05, zero_division=0)
recall_05 = recall_score(test_true, pred_05, zero_division=0)
f1_05 = f1_score(test_true, pred_05, zero_division=0)

acc_best = accuracy_score(test_true, pred_best)
precision_best = precision_score(test_true, pred_best, zero_division=0)
recall_best = recall_score(test_true, pred_best, zero_division=0)
f1_best = f1_score(test_true, pred_best, zero_division=0)

cm_best = confusion_matrix(test_true, pred_best)
if cm_best.size == 4:
    tn, fp, fn, tp = cm_best.ravel()
else:
    tn = fp = fn = tp = 0

brier = brier_score_loss(test_true, test_probs)

# Figure A: ROC + PR
pred_residuals_fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(fpr, tpr, color="darkorange", linewidth=2, label=f"ROC AUC = {roc_auc:.3f}")
ax.plot([0, 1], [0, 1], color="navy", linewidth=1.5, linestyle="--", label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right")

ax = axes[1]
ax.plot(recall_curve, precision_curve, color="teal", linewidth=2, label=f"PR AUC = {pr_auc:.3f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower left")

plt.tight_layout()

# Figure B: probability distribution + confusion matrix
error_dist_fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(test_probs[test_true == 0], bins=25, alpha=0.65, label="Safe (0)", color="green", edgecolor="black")
ax.hist(test_probs[test_true == 1], bins=25, alpha=0.65, label="Unsafe (1)", color="red", edgecolor="black")
ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="thr=0.50")
ax.axvline(best_threshold, color="purple", linestyle=":", linewidth=1.8, label=f"thr*= {best_threshold:.3f}")
ax.set_xlabel("Predicted Probability")
ax.set_ylabel("Frequency")
ax.set_title("Probability Distribution by Class")
ax.grid(True, alpha=0.3, axis="y")
ax.legend(fontsize=9)

ax = axes[1]
im = ax.imshow(cm_best, interpolation="nearest", cmap="Blues")
ax.set_title(f"Confusion Matrix (thr*={best_threshold:.3f})")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Safe", "Unsafe"])
ax.set_yticklabels(["Safe", "Unsafe"])

for i in range(cm_best.shape[0]):
    for j in range(cm_best.shape[1]):
        ax.text(j, i, f"{cm_best[i, j]}", ha="center", va="center", color="black", fontsize=11)

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()

print("\nBinary evaluation summary (test set):")
print(f"  ROC AUC:      {roc_auc:.4f}")
print(f"  PR AUC:       {pr_auc:.4f}")
print(f"  Brier score:  {brier:.4f}")
print(f"  Acc@0.50:     {acc_05:.4f} | P/R/F1: {precision_05:.4f}/{recall_05:.4f}/{f1_05:.4f}")
print(f"  Acc@thr*:     {acc_best:.4f} | P/R/F1: {precision_best:.4f}/{recall_best:.4f}/{f1_best:.4f}")
print(f"  thr*:         {best_threshold:.4f}")
print(f"  TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")

# Explicit metrics dictionary for export and reporting
metrics = {
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "brier_score": float(brier),
    "acc_0.5": float(acc_05),
    "precision_0.5": float(precision_05),
    "recall_0.5": float(recall_05),
    "f1_0.5": float(f1_05),
    "acc_best": float(acc_best),
    "precision_best": float(precision_best),
    "recall_best": float(recall_best),
    "f1_best": float(f1_best),
    "best_threshold": float(best_threshold),
    "tp": int(tp),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
}

# Compatibility aliases (kept for downstream code expecting these names)
train_roc_auc = roc_auc
test_roc_auc = roc_auc
train_r2 = float(roc_auc)  # heuristic mapping for downstream compatibility
test_r2 = float(roc_auc)   # heuristic mapping for downstream compatibility
train_mae = 1.0 - acc_05
test_mae = 1.0 - acc_best
train_rmse = float(np.sqrt(brier))
test_rmse = float(np.sqrt(brier))
final_val_r2 = float(roc_auc)  # kept for compatibility with downstream exports

