"""
Utility for model evaluation: printing metrics, interpretation, and saving results.
Includes formatted output and organized folder structure for tracking model performance.
"""

import json
import platform
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from pprint import pformat
from importlib.metadata import PackageNotFoundError, version as package_version
import matplotlib.pyplot as plt
import numpy as np

from c00_naming import build_evaluation_folder_name

try:
    from config import PLOT_COLORS, PLOT_STYLE
except ImportError:
    PLOT_COLORS = {
        "primary": "#61788C",
        "secondary": "#9CA5A6",
        "accent": "#F2994B",
        "danger": "#D9653B",
        "neutral": "#D7D9D9",
        "black": "#000000",
        "white": "#FFFFFF",
    }
    PLOT_STYLE = {
        "figsize_small": (8, 5),
        "figsize_medium": (12, 7),
        "figsize_large": (16, 10),
        "dpi": 100,
        "grid_alpha": 0.3,
        "line_width": 2.0,
        "marker_size": 5,
    }


def _safe_package_version(package_name: str) -> str | None:
    try:
        return package_version(package_name)
    except PackageNotFoundError:
        return None


def _format_feature_line(label: str, values) -> str:
    if values is None:
        return f"- {label}: n/a"
    if isinstance(values, tuple):
        return f"- {label}: {values}"
    if isinstance(values, list):
        return f"- {label}: {tuple(values)}"
    return f"- {label}: {values}"


def collect_environment_snapshot() -> dict:
    """Collect a compact environment snapshot for experiment records."""
    repo_root = Path(__file__).resolve().parents[1]
    git_commit = None

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            git_commit = completed.stdout.strip() or None
    except OSError:
        git_commit = None

    package_names = ["numpy", "pandas", "torch", "torch-geometric", "scikit-learn", "joblib", "matplotlib"]
    package_versions = {}
    for package_name in package_names:
        package_version_value = _safe_package_version(package_name)
        if package_version_value is not None:
            package_versions[package_name] = package_version_value

    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "git_commit": git_commit,
        "packages": package_versions,
    }

def compute_split_metrics(trues: np.ndarray, preds: np.ndarray, tail_target_quantile: float = 0.90) -> dict[str, float]:
    """Compute regression and tail-behavior metrics for one split."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    signed_errors = preds - trues
    abs_errors = np.abs(signed_errors)

    abs_target = np.abs(trues)
    tail_threshold = float(np.quantile(abs_target, tail_target_quantile))
    tail_mask = abs_target >= tail_threshold
    if not np.any(tail_mask):
        tail_mask = np.ones_like(abs_target, dtype=bool)

    true_min = float(np.min(trues))
    true_max = float(np.max(trues))
    outside_range_mask = (preds < true_min) | (preds > true_max)

    if trues.size >= 2:
        slope, intercept = np.polyfit(trues, preds, 1)
        slope = float(slope)
        intercept = float(intercept)
    else:
        slope = float("nan")
        intercept = float("nan")

    return {
        "r2": float(r2_score(trues, preds)),
        "mae": float(mean_absolute_error(trues, preds)),
        "rmse": float(np.sqrt(mean_squared_error(trues, preds))),
        "bias": float(np.mean(signed_errors)),
        "median_abs_error": float(np.median(abs_errors)),
        "p90_abs_error": float(np.quantile(abs_errors, 0.90)),
        "p95_abs_error": float(np.quantile(abs_errors, 0.95)),
        "p99_abs_error": float(np.quantile(abs_errors, 0.99)),
        "max_abs_error": float(np.max(abs_errors)),
        "tail_target_quantile": float(tail_target_quantile),
        "tail_target_threshold_abs": tail_threshold,
        "tail_count": float(np.sum(tail_mask)),
        "tail_fraction": float(np.mean(tail_mask)),
        "tail_mae": float(np.mean(abs_errors[tail_mask])),
        "tail_rmse": float(np.sqrt(np.mean(np.square(signed_errors[tail_mask])))),
        "tail_bias": float(np.mean(signed_errors[tail_mask])),
        "outside_true_range_count": float(np.sum(outside_range_mask)),
        "outside_true_range_fraction": float(np.mean(outside_range_mask)),
        "pred_slope_vs_true": slope,
        "pred_intercept_vs_true": intercept,
    }


def print_evaluation_metrics(metrics: dict, status: str = "unknown"):
    """
    Print formatted performance metrics and interpretation.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary with keys: train_r2, test_r2, train_mae, test_mae, train_rmse, test_rmse, r2_gap
    status : str
        Model status ("good_fit", "overfitting", "underfitting")
    
    Returns:
    --------
    None (prints to stdout)
    """
    
    print("=" * 70)
    print("MODEL EVALUATION METRICS")
    print("=" * 70)
    print(f"\nRegression performance:")
    print(f"   Train R²:  {metrics['train_r2']:.4f}")
    print(f"   Test R²:   {metrics['test_r2']:.4f}")
    print(f"   R² Gap:    {metrics['r2_gap']:.4f}  {'🟢 GOOD' if metrics['r2_gap'] < 0.05 else '🟡 CAUTION' if metrics['r2_gap'] < 0.10 else '🔴 HIGH'}")
    
    print(f"\nError metrics (kN):")
    print(f"   Train MAE:  {metrics['train_mae']:.4f}")
    print(f"   Test MAE:   {metrics['test_mae']:.4f}")
    print(f"   Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"   Test RMSE:  {metrics['test_rmse']:.4f}")

    if all(key in metrics for key in ("test_p95_abs_error", "test_max_abs_error", "test_tail_mae")):
        print(f"\nExtreme-behavior metrics (Test, kN):")
        print(f"   Median |error|: {metrics.get('test_median_abs_error', float('nan')):.4f}")
        print(f"   P95 |error|:    {metrics['test_p95_abs_error']:.4f}")
        print(f"   P99 |error|:    {metrics.get('test_p99_abs_error', float('nan')):.4f}")
        print(f"   Max |error|:    {metrics['test_max_abs_error']:.4f}")
        print(f"   Tail MAE:       {metrics['test_tail_mae']:.4f}")
        if "test_outside_true_range_count" in metrics:
            print(
                f"   Pred outside true range: {int(metrics['test_outside_true_range_count'])} "
                f"({100.0 * metrics.get('test_outside_true_range_fraction', 0.0):.2f}%)"
            )
    
    print(f"\nInterpretation:")
    if status == "good_fit":
        print(f"   Good fit - model generalizes well.")
        print(f"      • Train and test R² are close (gap: {metrics['r2_gap']:.4f} < 0.05)")
        print(f"      • Predictions are accurate (Test R²: {metrics['test_r2']:.4f})")
        print(f"      • Model ready for deployment")
    elif status == "overfitting":
        print(f"   Overfitting - model memorizes training data")
        print(f"      • Large gap between train and test R² (gap: {metrics['r2_gap']:.4f} > 0.05)")
        print(f"      • Train R² ({metrics['train_r2']:.4f}) >> Test R² ({metrics['test_r2']:.4f})")
        print(f"      • Recommendation: collect more diverse training data or add regularization")
    elif status == "underfitting":
        print(f"   Underfitting - model lacks capacity")
        print(f"      • Both R² scores are low (Train: {metrics['train_r2']:.4f}, Test: {metrics['test_r2']:.4f} < 0.7)")
        print(f"      • Model cannot capture data complexity")
        print(f"      • Recommendation: increase model capacity or train longer")
    else:
        print(f"   ❓ Status: {status}")
    
    print("=" * 70 + "\n")


def save_evaluation(
    model_prefix: str,
    dataset_name: str,
    metrics: dict,
    pred_residuals_fig: plt.Figure,
    error_dist_fig: plt.Figure,
    node_count: int,
    edge_count: int,
    export_path: Path,
    training_visuals_fig: plt.Figure = None,
    extremes_diagnostics_fig: plt.Figure | None = None,
    status: str = "✅ GOOD FIT",
    run_id: str | None = None,
    artifact_stem: str | None = None,
    learning_rate: float | None = None,
    epochs: int | None = None,
    eval_every: int | None = None,
    final_val_r2: float | None = None,
    strict_dataset_label: str | None = None,
    source_dataset_path: str | None = None,
    architecture_summary: dict | str | None = None,
    feature_count: int | None = None,
    experiment_notes: str | None = None,
    train_split_ratio: float | None = None,
    random_seed: int | None = None,
    source_notebook: str | None = None,
    environment_snapshot: dict | None = None,
    epoch_metrics_history: list | None = None,
):
    """
    Save all evaluation results (metrics + plots) to organized folder structure.
    
    Parameters:
    -----------
    model_prefix : str
        Model identifier (e.g., "data_3_1_0000")
    dataset_name : str
        Dataset filename (e.g., "data_3.1.csv")
    metrics : dict
        Dictionary with keys: train_r2, test_r2, train_mae, test_mae, train_rmse, test_rmse, r2_gap
    pred_residuals_fig : plt.Figure
        2x2 figure with predictions vs actual + residuals plots
    error_dist_fig : plt.Figure
        Figure with error distribution histograms
    training_visuals_fig : plt.Figure, optional
        Figure with training diagnostics (loss curve + normalized target distribution)
    node_count : int
        Number of nodes in the graph
    edge_count : int
        Number of edges in the graph
    export_path : Path
        Path to export directory (typically from config.SM_DATA_PATH)
    status : str
        Model status (e.g., "✅ GOOD FIT", "⚠️ OVERFITTING")
    
    Returns:
    --------
    dict : paths to all saved files
    """
    
    # Create directory structure
    base_folder_name = artifact_stem or model_prefix
    resolved_feature_count = feature_count
    if resolved_feature_count is None and isinstance(architecture_summary, dict):
        node_features = architecture_summary.get("node_in_dim")
        edge_features = architecture_summary.get("edge_in_dim")
        global_features = architecture_summary.get("global_in_dim")
        if all(isinstance(value, (int, float)) for value in (node_features, edge_features, global_features)):
            resolved_feature_count = int(node_features) + int(edge_features) + int(global_features)

    eval_dir_name = build_evaluation_folder_name(base_folder_name, resolved_feature_count)
    eval_dir = export_path / eval_dir_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}
    
    print(f"\n{'='*60}")
    print(f"Saving evaluation results for: {model_prefix}")
    print(f"{'='*60}\n")

    architecture_text = pformat(architecture_summary, sort_dicts=False) if architecture_summary is not None else "Not provided"
    if isinstance(architecture_summary, dict):
        feature_lines = []
        if any(key in architecture_summary for key in ("node_features", "edge_features", "global_features")):
            feature_lines.append("Feature selection:")
            feature_lines.append(_format_feature_line("Node features", architecture_summary.get("node_features")))
            feature_lines.append(_format_feature_line("Edge features", architecture_summary.get("edge_features")))
            feature_lines.append(_format_feature_line("Global features", architecture_summary.get("global_features")))
            architecture_text = architecture_text + "\n\n" + "\n".join(feature_lines)
    environment_data = environment_snapshot or collect_environment_snapshot()
    
    # 1. Save predictions vs actual + residuals plot
    pred_plot_path = eval_dir / f"01_predictions_residuals_{timestamp}.png"
    pred_residuals_fig.savefig(pred_plot_path, dpi=150, bbox_inches='tight')
    saved_files['pred_residuals_plot'] = pred_plot_path
    print(f"Predictions plot saved: 01_predictions_residuals_{timestamp}.png")
    
    # 2. Save error distribution plot
    error_plot_path = eval_dir / f"02_error_distribution_{timestamp}.png"
    error_dist_fig.savefig(error_plot_path, dpi=150, bbox_inches='tight')
    saved_files['error_dist_plot'] = error_plot_path
    print(f"Error distribution saved: 02_error_distribution_{timestamp}.png")

    # 3. Save training diagnostics plot (optional)
    if training_visuals_fig is not None:
        training_plot_path = eval_dir / f"03_training_diagnostics_{timestamp}.png"
        training_visuals_fig.savefig(training_plot_path, dpi=150, bbox_inches='tight')
        saved_files['training_diagnostics_plot'] = training_plot_path
        print(f"Training diagnostics saved: 03_training_diagnostics_{timestamp}.png")

    # 4. Save extreme-behavior diagnostics plot (optional)
    if extremes_diagnostics_fig is not None:
        extremes_plot_path = eval_dir / f"04_extremes_diagnostics_{timestamp}.png"
        extremes_diagnostics_fig.savefig(extremes_plot_path, dpi=150, bbox_inches='tight')
        saved_files['extremes_diagnostics_plot'] = extremes_plot_path
        print(f"Extremes diagnostics saved: 04_extremes_diagnostics_{timestamp}.png")
    
    # 4b. Save epoch-by-epoch metrics as CSV (for reviewing training progress)
    if epoch_metrics_history:
        import pandas as pd
        epoch_csv_path = eval_dir / f"epoch_history_{timestamp}.csv"
        df_epochs = pd.DataFrame(epoch_metrics_history)
        df_epochs.to_csv(epoch_csv_path, index=False)
        saved_files['epoch_history_csv'] = epoch_csv_path
        print(f"Epoch history saved: epoch_history_{timestamp}.csv")
    
    # 5. Save architecture summary
    architecture_path = eval_dir / f"model_architecture_{timestamp}.txt"
    architecture_text_block = f"""# Model Architecture Summary

Run ID: {run_id or model_prefix}
Artifact stem: {artifact_stem or model_prefix}
Dataset: {dataset_name}
Strict dataset label: {strict_dataset_label or 'n/a'}
Learning rate: {learning_rate if learning_rate is not None else 'n/a'}
Epochs: {epochs if epochs is not None else 'n/a'}
Eval every: {eval_every if eval_every is not None else 'n/a'}
Final validation R²: {final_val_r2 if final_val_r2 is not None else 'n/a'}
Graph: {node_count} nodes x {edge_count} edges

{architecture_text}
"""
    with open(architecture_path, 'w', encoding='utf-8') as f:
        f.write(architecture_text_block)
    saved_files['architecture_summary'] = architecture_path
    print(f"Architecture summary saved: {architecture_path.name}")

    # 6. Save combined manifest + metrics for downstream tooling
    train_metric_bundle = {}
    test_metric_bundle = {}
    for key, value in metrics.items():
        if key.startswith("train_") and isinstance(value, (int, float)):
            train_metric_bundle[key.replace("train_", "", 1)] = float(value)
        if key.startswith("test_") and isinstance(value, (int, float)):
            test_metric_bundle[key.replace("test_", "", 1)] = float(value)

    manifest_path = eval_dir / f"run_manifest_{timestamp}.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                "run_id": run_id or model_prefix,
                "artifact_stem": artifact_stem or model_prefix,
                "dataset": dataset_name,
                "dataset_sources": architecture_summary.get("dataset_sources") if isinstance(architecture_summary, dict) else None,
                "strict_dataset_label": strict_dataset_label,
                "source_dataset_path": source_dataset_path,
                "node_features": architecture_summary.get("node_features") if isinstance(architecture_summary, dict) else None,
                "edge_features": architecture_summary.get("edge_features") if isinstance(architecture_summary, dict) else None,
                "global_features": architecture_summary.get("global_features") if isinstance(architecture_summary, dict) else None,
                "selected_node_continuous_cols": architecture_summary.get("selected_node_continuous_cols") if isinstance(architecture_summary, dict) else None,
                "selected_node_mask_cols": architecture_summary.get("selected_node_mask_cols") if isinstance(architecture_summary, dict) else None,
                "selected_node_virtual_cols": architecture_summary.get("selected_node_virtual_cols") if isinstance(architecture_summary, dict) else None,
                "selected_edge_feature_cols": architecture_summary.get("selected_edge_feature_cols") if isinstance(architecture_summary, dict) else None,
                "selected_global_feature_cols": architecture_summary.get("selected_global_feature_cols") if isinstance(architecture_summary, dict) else None,
                "feature_count": resolved_feature_count,
                "metrics": {
                    "train": train_metric_bundle,
                    "test": test_metric_bundle,
                    "r2_gap": float(metrics['r2_gap']),
                    "status": status
                },
                "learning_rate": learning_rate,
                "epochs": epochs,
                "eval_every": eval_every,
                "final_val_r2": final_val_r2,
                "status": status,
                "node_count": node_count,
                "edge_count": edge_count,
                "files": {key: str(path.name) for key, path in saved_files.items()},
            },
            f,
            indent=2,
        )
    saved_files['manifest'] = manifest_path
    print(f"Manifest saved: {manifest_path.name}")

    # 7. Save experiment card for reproducibility audits
    experiment_card_path = eval_dir / f"experiment_card_{timestamp}.md"
    experiment_card = f"""# Experiment Card

## Identity

- Run ID: {run_id or model_prefix}
- Artifact stem: {artifact_stem or model_prefix}
- Validation folder: {eval_dir.name}
- Source notebook: {source_notebook or 'n/a'}

## Data

- Dataset file: {dataset_name}
- Source dataset path: {source_dataset_path or 'n/a'}
- Strict dataset label: {strict_dataset_label or 'n/a'}
- Train split ratio: {train_split_ratio if train_split_ratio is not None else 'n/a'}
- Random seed: {random_seed if random_seed is not None else 'n/a'}

## Training

- Learning rate: {learning_rate if learning_rate is not None else 'n/a'}
- Epochs: {epochs if epochs is not None else 'n/a'}
- Final validation R²: {final_val_r2 if final_val_r2 is not None else 'n/a'}
- Status: {status}

## Model

- Node count: {node_count}
- Edge count: {edge_count}
- Architecture summary:

```text
{architecture_text}
```

## Environment

- Python: {environment_data.get('python_version', 'n/a')}
- Implementation: {environment_data.get('python_implementation', 'n/a')}
- Platform: {environment_data.get('platform', 'n/a')}
- Executable: {environment_data.get('executable', 'n/a')}
- Git commit: {environment_data.get('git_commit', 'n/a')}

## Package Versions

```json
{json.dumps(environment_data.get('packages', {}), indent=2)}
```

## Notes

{experiment_notes or 'n/a'}
"""
    with open(experiment_card_path, 'w', encoding='utf-8') as f:
        f.write(experiment_card)
    saved_files['experiment_card'] = experiment_card_path
    print(f"Experiment card saved: {experiment_card_path.name}")

    # 8. Create summary README
    extreme_summary = ""
    if "test_p95_abs_error" in metrics:
        extreme_summary = f"""
## Extreme-Error Diagnostics

- Test median |error|: {metrics.get('test_median_abs_error', float('nan')):.4f} kN
- Test p95 |error|: {metrics['test_p95_abs_error']:.4f} kN
- Test p99 |error|: {metrics.get('test_p99_abs_error', float('nan')):.4f} kN
- Test max |error|: {metrics.get('test_max_abs_error', float('nan')):.4f} kN
- Test tail MAE (largest-force region): {metrics.get('test_tail_mae', float('nan')):.4f} kN
- Pred outside true range: {int(metrics.get('test_outside_true_range_count', 0.0))} ({100.0 * metrics.get('test_outside_true_range_fraction', 0.0):.2f}%)
"""

    contextual_notes = f"## Contextual Notes\n\n{experiment_notes}\n" if experiment_notes else ""
    summary_text = f"""# Evaluation Results: {artifact_stem or model_prefix}

**Run ID**: {run_id or model_prefix}  
**Artifact stem**: {artifact_stem or model_prefix}  
**Date**: {timestamp}  
**Dataset**: {dataset_name}  
**Strict dataset label**: {strict_dataset_label or 'n/a'}  
**Learning rate**: {learning_rate if learning_rate is not None else 'n/a'}  
**Epochs**: {epochs if epochs is not None else 'n/a'}  
**Final validation R²**: {final_val_r2 if final_val_r2 is not None else 'n/a'}  
**Graph**: {node_count} nodes × {edge_count} edges

## Performance Metrics

| Metric | Train | Test |
|--------|-------|------|
| **R² Score** | {metrics['train_r2']:.4f} | {metrics['test_r2']:.4f} |
| **MAE (kN)** | {metrics['train_mae']:.4f} | {metrics['test_mae']:.4f} |
| **RMSE (kN)** | {metrics['train_rmse']:.4f} | {metrics['test_rmse']:.4f} |

## Interpretation

- **R² Gap (Train - Test)**: {metrics['r2_gap']:.4f}
- **Status**: {status}

**Meaning**: The model explains {metrics['test_r2']*100:.1f}% of variance in test data with {metrics['test_mae']:.2f} kN average error.

{extreme_summary}

## Architecture Summary

```text
{architecture_text}
```

## Experimental Setup

- Source dataset: {source_dataset_path or dataset_name}
- Validation folder: {eval_dir.name}
- Hyperparameter provenance is captured in the filename stem and manifest.

{contextual_notes}

## Files in This Folder

1. `01_predictions_residuals_{timestamp}.png` — 4-panel: predictions vs actual + residual plots (train & test)
2. `02_error_distribution_{timestamp}.png` — Error histograms for train & test
3. `03_training_diagnostics_{timestamp}.png` — Training loss curve + normalized target profile (if available)
4. `04_extremes_diagnostics_{timestamp}.png` — Sorted profile, tail error scatter, residual trend, and top-k worst errors (if available)
5. `model_architecture_{timestamp}.txt` — Text summary of the trained model
6. `run_manifest_{timestamp}.json` — Combined manifest, metrics, feature selection, and environment snapshot
7. `experiment_card_{timestamp}.md` — Reproducibility snapshot for the run
8. `README.md` — This summary

## Model Files

Located in `SM_EXPORT_PATH/01_surrogate_models/`:
- `{artifact_stem or model_prefix}_surrogate_model.pt` — Model weights
- `{artifact_stem or model_prefix}_node_scaler.pkl` — Node feature scaler
- `{artifact_stem or model_prefix}_edge_feature_scaler.pkl` — Edge feature scaler
- `{artifact_stem or model_prefix}_edge_target_scaler.pkl` — Edge target scaler
- `{artifact_stem or model_prefix}_global_feature_scaler.pkl` — Global feature scaler

## Comparison Tips

- Compare `r2_gap` across runs to see which model generalizes best
- Look at test plots to identify systematic errors or outliers
- Check if test MAE is acceptable for your use case
- Use the manifest to trace the exact dataset and hyperparameters for each run
"""
    
    readme_path = eval_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(summary_text)
    saved_files['readme'] = readme_path
    print(f"README saved: README.md")
    
    print(f"\nAll evaluation files saved to:")
    print(f"   {eval_dir}\n")
    
    return saved_files
