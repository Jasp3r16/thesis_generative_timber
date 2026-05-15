# =============================================================================
# Binary Classification Evaluation — TrussEdgeSafetyGNN
# =============================================================================
#
# Requires from training:
#   train_losses, val_losses      — per-epoch loss lists
#   test_probs, test_targets      — flat numpy arrays from collect_preds()
#   best_threshold                — threshold chosen on val set (from train script)
#   best_epoch                    — epoch at which best checkpoint was saved
#
# Produces:
#   fig1  — Training dynamics  (loss curves + generalisation gap)
#   fig2  — Threshold analysis (ROC, PR, metrics vs threshold sweep)
#   fig3  — Prediction quality (probability distributions + calibration)
#   fig4  — Confusion matrices (default | val-tuned | safety threshold)
#   fig5  — Per-member analysis (unsafe rate, recall, mean prob — 120 physical members)
#   metrics — dict of all scalar metrics for export cell
#
# Fixes vs v3:
#   - Per-member slicing moved to AFTER computation (was before, causing NameError)
#   - NUM_EDGES_PHYSICAL = 120 used for per-member plots (was 240, showing duplicates)
#   - NUM_EDGES = 240 retained for test set reshaping (bidirectional)
#   - save_fig routes to config.SM_EXPORT_PATH

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score, auc, brier_score_loss, confusion_matrix,
    f1_score, matthews_corrcoef, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.calibration import calibration_curve
import config
from config import PLOT_COLORS as C, PLOT_STYLE as S

# =============================================================================
# 0. SETTINGS
# =============================================================================

NUM_EDGES          = 240   # total edges per sample after bidirectional conversion
NUM_EDGES_PHYSICAL = 120   # physical members — first half of bidirectional edge list
MIN_PRECISION      = 0.40  # minimum precision for safety-oriented threshold
SAVE_PATH          = config.SM_EXPORT_PATH

# =============================================================================
# 1. GUARD
# =============================================================================

required_vars = [
    "train_losses", "val_losses",
    "test_probs", "test_targets",
    "best_threshold", "best_epoch",
]
missing = [v for v in required_vars if v not in globals()]
if missing:
    raise RuntimeError(
        "Missing variables from training script: "
        + ", ".join(missing)
        + ". Run c21_train_v2.py first."
    )

# =============================================================================
# 2. GLOBAL STYLE
# =============================================================================

plt.rcParams.update({
    "figure.dpi":        S["dpi"],
    "axes.grid":         True,
    "grid.alpha":        S["grid_alpha"],
    "grid.color":        C["neutral"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.edgecolor":    C["black"],
    "axes.labelcolor":   C["black"],
    "xtick.color":       C["black"],
    "ytick.color":       C["black"],
    "text.color":        C["black"],
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "lines.linewidth":   S["line_width"],
    "lines.markersize":  S["marker_size"],
})

def save_fig(fig, stem):
    out = SAVE_PATH / f"{stem}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C["white"])
    print(f"  Saved: {out.name}")

# =============================================================================
# 3. CORE METRICS
# =============================================================================

test_probs_arr = np.asarray(test_probs).flatten()
test_true      = np.asarray(test_targets).flatten().astype(int)
epochs_range   = np.arange(1, len(train_losses) + 1)

# Curves
fpr, tpr, _                           = roc_curve(test_true, test_probs_arr)
roc_auc                               = roc_auc_score(test_true, test_probs_arr)
precision_curve, recall_curve, pr_thr = precision_recall_curve(test_true, test_probs_arr)
pr_auc                                = auc(recall_curve, precision_curve)

# Threshold A: maximise F1
f1_curve     = (2 * precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-12)
idx_f1       = int(np.argmax(f1_curve[:-1]))
threshold_f1 = float(pr_thr[idx_f1]) if len(pr_thr) > 0 else 0.5

# Threshold B: maximise recall subject to precision >= MIN_PRECISION
viable_mask = precision_curve[:-1] >= MIN_PRECISION
if viable_mask.any():
    idx_safety       = np.where(viable_mask)[0][int(np.argmax(recall_curve[:-1][viable_mask]))]
    threshold_safety = float(pr_thr[idx_safety])
else:
    threshold_safety = threshold_f1
    print(f"Warning: no threshold achieves precision >= {MIN_PRECISION:.0%}; "
          f"falling back to max-F1 threshold ({threshold_f1:.3f}).")

thr_primary = float(best_threshold)

def scores_at(thr):
    pred               = (test_probs_arr >= thr).astype(int)
    cm                 = confusion_matrix(test_true, pred, labels=[0, 1])
    tn_, fp_, fn_, tp_ = cm.ravel()
    return dict(
        threshold=thr, pred=pred, cm=cm,
        tn=int(tn_), fp=int(fp_), fn=int(fn_), tp=int(tp_),
        accuracy  = accuracy_score(test_true, pred),
        precision = precision_score(test_true, pred, zero_division=0),
        recall    = recall_score(test_true, pred, zero_division=0),
        f1        = f1_score(test_true, pred, zero_division=0),
        mcc       = matthews_corrcoef(test_true, pred),
    )

s_05      = scores_at(0.5)
s_primary = scores_at(thr_primary)
s_f1      = scores_at(threshold_f1)
s_safety  = scores_at(threshold_safety)
brier     = brier_score_loss(test_true, test_probs_arr)

print(f"Thresholds — val-tuned: {thr_primary:.3f} | max-F1: {threshold_f1:.3f} | "
      f"safety (P>={MIN_PRECISION:.0%}): {threshold_safety:.3f}")

# =============================================================================
# 4. FIGURE 1 — Training Dynamics
# =============================================================================

fig1, axes = plt.subplots(1, 2, figsize=S["figsize_medium"])
fig1.suptitle("Figure 1 — Training Dynamics", fontweight="bold", fontsize=13)

ax = axes[0]
ax.plot(epochs_range, train_losses, color=C["primary"],
        lw=S["line_width"], marker="o", ms=3, label="Train Loss")
ax.plot(epochs_range, val_losses,   color=C["accent"],
        lw=S["line_width"], marker="s", ms=3, linestyle="--", label="Val Loss")
ax.axvline(best_epoch + 1, color=C["danger"], linestyle=":", lw=1.5,
           label=f"Best epoch ({best_epoch + 1})")
ax.set_xlabel("Epoch")
ax.set_ylabel("Focal Loss")
ax.set_title("Train vs Validation Loss")
ax.legend()

ax = axes[1]
gap = np.array(val_losses) - np.array(train_losses)
ax.plot(epochs_range, gap, color=C["secondary"], lw=S["line_width"])
ax.axhline(0, color=C["black"], linestyle="--", lw=1)
ax.axvline(best_epoch + 1, color=C["danger"], linestyle=":", lw=1.5,
           label=f"Best epoch ({best_epoch + 1})")
ax.fill_between(epochs_range, gap, 0, where=(gap > 0),
                alpha=0.2, color=C["danger"],  label="Overfitting region")
ax.fill_between(epochs_range, gap, 0, where=(gap < 0),
                alpha=0.2, color=C["primary"], label="Underfitting region")
ax.set_xlabel("Epoch")
ax.set_ylabel("Val Loss − Train Loss")
ax.set_title("Generalisation Gap")
ax.legend(fontsize=9)

plt.tight_layout()
save_fig(fig1, "eval_fig1_training_dynamics")
plt.show()

# =============================================================================
# 5. FIGURE 2 — Threshold Analysis
# =============================================================================

fig2, axes = plt.subplots(1, 3, figsize=(S["figsize_large"][0], 5))
fig2.suptitle("Figure 2 — Threshold Analysis", fontweight="bold", fontsize=13)

ax = axes[0]
ax.plot(fpr, tpr, color=C["primary"], lw=S["line_width"], label=f"AUC = {roc_auc:.3f}")
ax.plot([0, 1], [0, 1], color=C["secondary"], lw=1.5, linestyle="--", label="Random")
for s, label, col in [
    (s_primary, f"val-tuned ({thr_primary:.2f})",   C["primary"]),
    (s_f1,      f"max-F1 ({threshold_f1:.2f})",     C["secondary"]),
    (s_safety,  f"safety ({threshold_safety:.2f})", C["accent"]),
]:
    ax.scatter(s["fp"] / max(s["fp"] + s["tn"], 1),
               s["tp"] / max(s["tp"] + s["fn"], 1),
               color=col, zorder=5, s=80, label=label,
               edgecolors=C["black"], linewidths=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend(fontsize=8, loc="lower right")

ax = axes[1]
ax.plot(recall_curve, precision_curve, color=C["primary"],
        lw=S["line_width"], label=f"PR AUC = {pr_auc:.3f}")
baseline = test_true.mean()
ax.axhline(baseline, color=C["neutral"], linestyle="--", lw=1.2,
           label=f"Baseline (pos rate={baseline:.2f})")
ax.axhline(MIN_PRECISION, color=C["accent"], linestyle=":", lw=1.2,
           label=f"Min precision = {MIN_PRECISION:.0%}")
for s, label, col in [
    (s_primary, "val-tuned", C["primary"]),
    (s_f1,      "max-F1",    C["secondary"]),
    (s_safety,  "safety",    C["accent"]),
]:
    ax.scatter(s["recall"], s["precision"], color=col, zorder=5, s=80,
               label=label, edgecolors=C["black"], linewidths=0.5)
ax.set_xlabel("Recall (unsafe class)")
ax.set_ylabel("Precision (unsafe class)")
ax.set_title("Precision-Recall Curve")
ax.legend(fontsize=8)

ax = axes[2]
sweep_thresholds                              = np.linspace(0.05, 0.95, 100)
sweep_recall, sweep_precision, sweep_f1, sweep_mcc = [], [], [], []
for t in sweep_thresholds:
    pred = (test_probs_arr >= t).astype(int)
    sweep_recall.append(recall_score(test_true, pred, zero_division=0))
    sweep_precision.append(precision_score(test_true, pred, zero_division=0))
    sweep_f1.append(f1_score(test_true, pred, zero_division=0))
    sweep_mcc.append(matthews_corrcoef(test_true, pred))

ax.plot(sweep_thresholds, sweep_recall,    color=C["danger"],    lw=S["line_width"], label="Recall (unsafe)")
ax.plot(sweep_thresholds, sweep_precision, color=C["primary"],   lw=S["line_width"], label="Precision (unsafe)")
ax.plot(sweep_thresholds, sweep_f1,        color=C["secondary"], lw=S["line_width"], label="F1 (unsafe)")
ax.plot(sweep_thresholds, sweep_mcc,       color=C["accent"],    lw=S["line_width"], linestyle="--", label="MCC")
ax.axvline(thr_primary,      color=C["primary"],   lw=1.5, linestyle=":", label=f"val-tuned ({thr_primary:.2f})")
ax.axvline(threshold_f1,     color=C["secondary"], lw=1.5, linestyle=":", label=f"max-F1 ({threshold_f1:.2f})")
ax.axvline(threshold_safety, color=C["accent"],    lw=1.5, linestyle=":", label=f"safety ({threshold_safety:.2f})")
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Score")
ax.set_title("Metrics vs Threshold")
ax.legend(fontsize=8)

plt.tight_layout()
save_fig(fig2, "eval_fig2_threshold_analysis")
plt.show()

# =============================================================================
# 6. FIGURE 3 — Prediction Quality & Calibration
# =============================================================================

fig3, axes = plt.subplots(1, 3, figsize=(S["figsize_large"][0], 5))
fig3.suptitle("Figure 3 — Prediction Quality & Calibration", fontweight="bold", fontsize=13)

ax = axes[0]
ax.hist(test_probs_arr[test_true == 0], bins=40, alpha=0.65,
        label=f"Safe (n={int((test_true==0).sum())})",
        color=C["primary"], edgecolor=C["white"])
ax.hist(test_probs_arr[test_true == 1], bins=40, alpha=0.65,
        label=f"Unsafe (n={int((test_true==1).sum())})",
        color=C["danger"], edgecolor=C["white"])
ax.axvline(thr_primary,      color=C["primary"], lw=2,   linestyle="--", label=f"val-tuned ({thr_primary:.2f})")
ax.axvline(threshold_safety, color=C["accent"],  lw=1.5, linestyle=":",  label=f"safety ({threshold_safety:.2f})")
ax.axvline(0.5,              color=C["black"],   lw=1.2, linestyle="--", label="thr=0.50")
ax.set_xlabel("Predicted Probability  P(unsafe)")
ax.set_ylabel("Count")
ax.set_title("Score Distribution by True Class")
ax.legend(fontsize=8)

ax = axes[1]
bins = np.linspace(0, 1, 41)
ax.hist(test_probs_arr[test_true == 0], bins=bins, alpha=0.65,
        label="Safe",   color=C["primary"], edgecolor=C["white"])
ax.hist(test_probs_arr[test_true == 1], bins=bins, alpha=0.65,
        label="Unsafe", color=C["danger"],  edgecolor=C["white"])
ax.axvline(thr_primary, color=C["primary"], lw=2, linestyle="--",
           label=f"val-tuned ({thr_primary:.2f})")
ax.set_yscale("log")
ax.set_xlabel("Predicted Probability  P(unsafe)")
ax.set_ylabel("Count (log scale)")
ax.set_title("Score Distribution (log scale)")
ax.legend(fontsize=8)

ax = axes[2]
try:
    frac_pos, mean_pred = calibration_curve(
        test_true, test_probs_arr, n_bins=10, strategy="uniform"
    )
    ax.plot(mean_pred, frac_pos, color=C["primary"],
            lw=S["line_width"], marker="o", ms=6, label=f"GNN (Brier={brier:.3f})")
except Exception as e:
    ax.text(0.5, 0.5, f"Calibration n/a\n({e})", ha="center", va="center", fontsize=9)
ax.plot([0, 1], [0, 1], color=C["black"], linestyle="--", lw=1.5,
        label="Perfect calibration")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction Positive (actual unsafe rate)")
ax.set_title("Calibration Curve (Reliability Diagram)")
ax.legend(fontsize=9)

plt.tight_layout()
save_fig(fig3, "eval_fig3_prediction_quality")
plt.show()

# =============================================================================
# 7. FIGURE 4 — Confusion Matrices
# =============================================================================

fig4, axes = plt.subplots(1, 3, figsize=S["figsize_large"])
fig4.suptitle("Figure 4 — Confusion Matrices", fontweight="bold", fontsize=13)

cm_cmap = LinearSegmentedColormap.from_list("cm", [C["white"], C["primary"]], N=256)

for ax, (s, title) in zip(axes, [
    (s_05,      "Default (thr=0.50)"),
    (s_primary, f"Val-tuned (thr={thr_primary:.2f})"),
    (s_safety,  f"Safety-oriented (thr={threshold_safety:.2f})"),
]):
    cm = s["cm"]
    im = ax.imshow(cm, interpolation="nearest", cmap=cm_cmap)
    ax.set_title(title)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Safe", "Unsafe"])
    ax.set_yticklabels(["Safe", "Unsafe"])
    for i in range(2):
        for j in range(2):
            val   = cm[i, j]
            color = C["white"] if val > cm.max() * 0.6 else C["black"]
            ax.text(j, i, f"{val}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel(
        f"Predicted\n"
        f"Recall={s['recall']:.3f}  Precision={s['precision']:.3f}\n"
        f"F1={s['f1']:.3f}  MCC={s['mcc']:.3f}  FN={s['fn']}",
        fontsize=9,
    )
    ax.set_ylabel("True")

plt.tight_layout()
save_fig(fig4, "eval_fig4_confusion_matrices")
plt.show()

# =============================================================================
# 8. FIGURE 5 — Per-Member Analysis
# Reshape test predictions to [n_samples, NUM_EDGES], then slice to first
# NUM_EDGES_PHYSICAL columns — the second half are reverse-edge duplicates
# with identical statistics and don't add information.
# =============================================================================

fig5                   = None
per_member_recall      = None
per_member_fpr         = None
per_member_unsafe_rate = None

n_test_samples = len(test_probs_arr) // NUM_EDGES

if len(test_probs_arr) % NUM_EDGES == 0 and n_test_samples > 0:

    # Reshape to [n_samples, 240] then slice to [n_samples, 120]
    preds_mat   = (test_probs_arr >= thr_primary).astype(int).reshape(n_test_samples, NUM_EDGES)
    targets_mat = test_true.reshape(n_test_samples, NUM_EDGES)
    probs_mat   = test_probs_arr.reshape(n_test_samples, NUM_EDGES)

    # Slice to physical members only — reverse edges are statistical duplicates
    preds_mat   = preds_mat[:, :NUM_EDGES_PHYSICAL]
    targets_mat = targets_mat[:, :NUM_EDGES_PHYSICAL]
    probs_mat   = probs_mat[:, :NUM_EDGES_PHYSICAL]

    # Per-member statistics (now over 120 physical members)
    per_member_unsafe_rate = targets_mat.mean(axis=0)
    per_member_recall      = np.where(
        targets_mat.sum(axis=0) > 0,
        ((preds_mat == 1) & (targets_mat == 1)).sum(axis=0) / targets_mat.sum(axis=0).clip(1),
        np.nan,
    )
    per_member_fpr = np.where(
        (1 - targets_mat).sum(axis=0) > 0,
        ((preds_mat == 1) & (targets_mat == 0)).sum(axis=0) / (1 - targets_mat).sum(axis=0).clip(1),
        np.nan,
    )
    per_member_mean_prob = probs_mat.mean(axis=0)
    edge_ids = np.arange(NUM_EDGES_PHYSICAL)

    fig5, axes = plt.subplots(3, 1, figsize=(S["figsize_large"][0], 12))
    fig5.suptitle(
        f"Figure 5 — Per-Member Analysis ({NUM_EDGES_PHYSICAL} physical members, fixed topology)",
        fontweight="bold", fontsize=13,
    )

    # A: unsafe rate per member
    ax = axes[0]
    colors_rate = [C["danger"] if r > 0.3 else C["primary"] for r in per_member_unsafe_rate]
    ax.bar(edge_ids, per_member_unsafe_rate, color=colors_rate, edgecolor="none", width=0.8)
    ax.axhline(per_member_unsafe_rate.mean(), color=C["black"], linestyle="--", lw=1.5,
               label=f"Mean unsafe rate = {per_member_unsafe_rate.mean():.3f}")
    ax.set_xlabel("Member ID")
    ax.set_ylabel("Fraction unsafe across samples")
    ax.set_title("A — Unsafe Rate per Member  (red = >30% of samples)")
    ax.legend(fontsize=9)
    ax.set_xlim(-1, NUM_EDGES_PHYSICAL)

    # B: recall per member
    ax = axes[1]
    recall_vals   = np.nan_to_num(per_member_recall, nan=-0.05)
    colors_recall = []
    for r, rate in zip(recall_vals, per_member_unsafe_rate):
        if rate == 0:
            colors_recall.append(C["neutral"])
        elif r < 0.5:
            colors_recall.append(C["danger"])
        elif r < 0.8:
            colors_recall.append(C["accent"])
        else:
            colors_recall.append(C["primary"])
    ax.bar(edge_ids, recall_vals, color=colors_recall, edgecolor="none", width=0.8)
    ax.axhline(np.nanmean(per_member_recall), color=C["black"], linestyle="--", lw=1.5,
               label=f"Mean recall = {np.nanmean(per_member_recall):.3f}")
    ax.axhline(0.8, color=C["primary"], linestyle=":", lw=1.2, label="Target recall = 0.80")
    ax.set_xlabel("Member ID")
    ax.set_ylabel("Recall (unsafe class)")
    ax.set_title("B — Per-Member Recall  (red<0.5 | orange<0.8 | blue≥0.8 | gray=never fails)")
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlim(-1, NUM_EDGES_PHYSICAL)
    ax.legend(fontsize=9)

    # C: mean predicted probability vs true unsafe rate
    ax = axes[2]
    ax.bar(edge_ids, per_member_mean_prob, color=C["secondary"],
           edgecolor="none", width=0.8, label="Mean P(unsafe) predicted")
    ax.plot(edge_ids, per_member_unsafe_rate, color=C["danger"],
            marker="o", ms=3, lw=1.2, label="True unsafe rate")
    ax.axhline(thr_primary, color=C["primary"], linestyle="--", lw=1.5,
               label=f"Decision threshold ({thr_primary:.2f})")
    ax.set_xlabel("Member ID")
    ax.set_ylabel("Mean predicted probability / unsafe rate")
    ax.set_title("C — Mean Predicted Probability vs True Unsafe Rate per Member")
    ax.set_xlim(-1, NUM_EDGES_PHYSICAL)
    ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig(fig5, "eval_fig5_per_member")
    plt.show()

    # Top 10 hardest members
    members_that_fail = np.where(per_member_unsafe_rate > 0)[0]
    sorted_by_recall  = members_that_fail[np.argsort(per_member_recall[members_that_fail])]
    print(f"\nTop 10 hardest members (lowest recall, physical members 0-{NUM_EDGES_PHYSICAL-1}):")
    print(f"  {'MemberID':>8}  {'UnsafeRate':>10}  {'Recall':>8}  {'FPR':>8}")
    for mid in sorted_by_recall[:10]:
        print(f"  {mid:>8d}  {per_member_unsafe_rate[mid]:>10.3f}  "
              f"{per_member_recall[mid]:>8.3f}  {per_member_fpr[mid]:>8.3f}")

else:
    print(f"\n[Per-member analysis skipped] "
          f"len(test_probs_arr)={len(test_probs_arr)} is not divisible by "
          f"NUM_EDGES={NUM_EDGES}.")

# =============================================================================
# 9. METRICS SUMMARY
# =============================================================================

print("\n" + "=" * 65)
print("EVALUATION SUMMARY — TrussEdgeSafetyGNN")
print("=" * 65)
print(f"  Epochs trained:      {len(train_losses)}  (best: {best_epoch + 1})")
print(f"  Final train loss:    {train_losses[-1]:.6f}")
print(f"  Final val loss:      {val_losses[-1]:.6f}")
print()
print("  Threshold-independent:")
print(f"    ROC AUC:    {roc_auc:.4f}   (>0.90 = excellent, >0.80 = good)")
print(f"    PR  AUC:    {pr_auc:.4f}   (baseline = {test_true.mean():.3f})")
print(f"    Brier:      {brier:.4f}   (0 = perfect calibration)")
print()
for label, s in [
    ("Default (thr=0.50)",                  s_05),
    (f"Val-tuned  (thr={thr_primary:.2f})", s_primary),
    (f"Safety     (thr={threshold_safety:.2f})", s_safety),
]:
    print(f"  @ {label}:")
    print(f"    Accuracy:        {s['accuracy']:.4f}")
    print(f"    Precision:       {s['precision']:.4f}")
    print(f"    Recall (unsafe): {s['recall']:.4f}   <- most important for structural safety")
    print(f"    F1:              {s['f1']:.4f}")
    print(f"    MCC:             {s['mcc']:.4f}   (>0.5 = good for imbalanced data)")
    print(f"    TP={s['tp']}  TN={s['tn']}  FP={s['fp']}  FN={s['fn']}  (FN = missed failures)")
    print()
print("=" * 65)

# =============================================================================
# 10. METRICS DICT
# =============================================================================

metrics = {
    "roc_auc":              float(roc_auc),
    "pr_auc":               float(pr_auc),
    "brier_score":          float(brier),
    "acc_0.50":             float(s_05["accuracy"]),
    "precision_0.50":       float(s_05["precision"]),
    "recall_0.50":          float(s_05["recall"]),
    "f1_0.50":              float(s_05["f1"]),
    "mcc_0.50":             float(s_05["mcc"]),
    "threshold_primary":    float(thr_primary),
    "acc_primary":          float(s_primary["accuracy"]),
    "precision_primary":    float(s_primary["precision"]),
    "recall_primary":       float(s_primary["recall"]),
    "f1_primary":           float(s_primary["f1"]),
    "mcc_primary":          float(s_primary["mcc"]),
    "tp_primary":           int(s_primary["tp"]),
    "tn_primary":           int(s_primary["tn"]),
    "fp_primary":           int(s_primary["fp"]),
    "fn_primary":           int(s_primary["fn"]),
    "threshold_safety":     float(threshold_safety),
    "acc_safety":           float(s_safety["accuracy"]),
    "precision_safety":     float(s_safety["precision"]),
    "recall_safety":        float(s_safety["recall"]),
    "f1_safety":            float(s_safety["f1"]),
    "mcc_safety":           float(s_safety["mcc"]),
    "tp_safety":            int(s_safety["tp"]),
    "tn_safety":            int(s_safety["tn"]),
    "fp_safety":            int(s_safety["fp"]),
    "fn_safety":            int(s_safety["fn"]),
    "false_negative_rate":  float(s_safety["fn"] / max(s_safety["tp"] + s_safety["fn"], 1)),
    "best_epoch":           int(best_epoch),
    "final_train_loss":     float(train_losses[-1]),
    "final_val_loss":       float(val_losses[-1]),
}

if per_member_recall is not None:
    metrics["per_member_recall_mean"] = float(np.nanmean(per_member_recall))
    metrics["per_member_recall_min"]  = float(np.nanmin(per_member_recall))
    metrics["per_member_fpr_mean"]    = float(np.nanmean(per_member_fpr))

print("\nEvaluation complete.")
print("Figures: fig1, fig2, fig3, fig4, fig5")
print("Metrics dict: metrics")