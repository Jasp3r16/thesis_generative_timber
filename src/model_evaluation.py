"""
Utility for model evaluation: printing metrics, interpretation, and saving results.
Includes formatted output and organized folder structure for tracking model performance.
"""

import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


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
    print(f"\n📊 REGRESSION PERFORMANCE:")
    print(f"   Train R²:  {metrics['train_r2']:.4f}")
    print(f"   Test R²:   {metrics['test_r2']:.4f}")
    print(f"   R² Gap:    {metrics['r2_gap']:.4f}  {'🟢 GOOD' if metrics['r2_gap'] < 0.05 else '🟡 CAUTION' if metrics['r2_gap'] < 0.10 else '🔴 HIGH'}")
    
    print(f"\n📏 ERROR METRICS (kN):")
    print(f"   Train MAE:  {metrics['train_mae']:.4f}")
    print(f"   Test MAE:   {metrics['test_mae']:.4f}")
    print(f"   Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"   Test RMSE:  {metrics['test_rmse']:.4f}")
    
    print(f"\n🎯 INTERPRETATION:")
    if status == "good_fit":
        print(f"   ✅ GOOD FIT - Model generalizes well!")
        print(f"      • Train and Test R² are close (gap: {metrics['r2_gap']:.4f} < 0.05)")
        print(f"      • Predictions are accurate (Test R²: {metrics['test_r2']:.4f})")
        print(f"      • Model ready for deployment")
    elif status == "overfitting":
        print(f"   ⚠️ OVERFITTING - Model memorizes training data")
        print(f"      • Large gap between Train and Test R² (gap: {metrics['r2_gap']:.4f} > 0.05)")
        print(f"      • Train R² ({metrics['train_r2']:.4f}) >> Test R² ({metrics['test_r2']:.4f})")
        print(f"      • Recommendation: Collect more diverse training data or add regularization")
    elif status == "underfitting":
        print(f"   ⚠️ UNDERFITTING - Model lacks capacity")
        print(f"      • Both R² scores low (Train: {metrics['train_r2']:.4f}, Test: {metrics['test_r2']:.4f} < 0.7)")
        print(f"      • Model cannot capture data complexity")
        print(f"      • Recommendation: Increase model capacity or train longer")
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
    status: str = "✅ GOOD FIT"
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
    date_today = datetime.now().strftime("%Y-%m-%d")
    eval_dir = export_path / f"{model_prefix}_{date_today}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}
    
    print(f"\n{'='*60}")
    print(f"SAVING EVALUATION RESULTS FOR: {model_prefix}")
    print(f"{'='*60}\n")
    
    # 1. Save metrics as JSON
    metrics_data = {
        "model_prefix": model_prefix,
        "dataset": dataset_name,
        "timestamp": timestamp,
        "node_count": node_count,
        "edge_count": edge_count,
        "metrics": {
            "train": {
                "r2": float(metrics['train_r2']),
                "mae": float(metrics['train_mae']),
                "rmse": float(metrics['train_rmse'])
            },
            "test": {
                "r2": float(metrics['test_r2']),
                "mae": float(metrics['test_mae']),
                "rmse": float(metrics['test_rmse'])
            },
            "r2_gap": float(metrics['r2_gap']),
            "status": status
        }
    }
    
    metrics_path = eval_dir / f"metrics_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    saved_files['metrics'] = metrics_path
    print(f"✅ Metrics saved: metrics_{timestamp}.json")
    
    # 2. Save predictions vs actual + residuals plot
    pred_plot_path = eval_dir / f"01_predictions_residuals_{timestamp}.png"
    pred_residuals_fig.savefig(pred_plot_path, dpi=150, bbox_inches='tight')
    saved_files['pred_residuals_plot'] = pred_plot_path
    print(f"✅ Predictions plot saved: 01_predictions_residuals_{timestamp}.png")
    
    # 3. Save error distribution plot
    error_plot_path = eval_dir / f"02_error_distribution_{timestamp}.png"
    error_dist_fig.savefig(error_plot_path, dpi=150, bbox_inches='tight')
    saved_files['error_dist_plot'] = error_plot_path
    print(f"✅ Error distribution saved: 02_error_distribution_{timestamp}.png")
    
    # 4. Create summary README
    summary_text = f"""# Evaluation Results: {model_prefix}

**Date**: {timestamp}  
**Dataset**: {dataset_name}  
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

## Files in This Folder

1. `metrics_{timestamp}.json` — Raw metrics (machine-readable)
2. `01_predictions_residuals_{timestamp}.png` — 4-panel: predictions vs actual + residual plots (train & test)
3. `02_error_distribution_{timestamp}.png` — Error histograms for train & test
4. `README.md` — This summary

## Model Files

Located in `SM_EXPORT_PATH/01_surrogate_models/`:
- `truss_edge_gnn_{model_prefix}.pt` — Model weights
- `node_scaler_{model_prefix}.pkl` — Node feature scaler
- `edge_scaler_{model_prefix}.pkl` — Edge label scaler

## Comparison Tips

- Compare `r2_gap` across runs to see which model generalizes best
- Look at test plots to identify systematic errors or outliers
- Check if test MAE is acceptable for your use case
"""
    
    readme_path = eval_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(summary_text)
    saved_files['readme'] = readme_path
    print(f"✅ README saved: README.md")
    
    print(f"\n📁 All evaluation files saved to:")
    print(f"   {eval_dir}\n")
    
    return saved_files
