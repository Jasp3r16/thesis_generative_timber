from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.c21_data_pipeline import V4Schema


@dataclass(frozen=True)
class FeatureImportanceResult:
	feature_name: str
	feature_group: str
	pearson_r_target_mean_abs: float
	spearman_r_target_mean_abs: float
	mean_abs_prediction_change_kn: float
	mean_prediction_difference_kn: float
	mean_abs_prediction_change_pct: float
	mean_target_mean_abs_kn: float
	mean_pred_baseline_kn: float
	mean_pred_ablated_kn: float
	n_samples: int


def _pearson_r(values_a: np.ndarray, values_b: np.ndarray) -> float:
	if values_a.size < 2 or values_b.size < 2:
		return float("nan")
	if np.allclose(values_a, values_a[0]) or np.allclose(values_b, values_b[0]):
		return float("nan")
	return float(np.corrcoef(values_a, values_b)[0, 1])


def _spearman_r(values_a: np.ndarray, values_b: np.ndarray) -> float:
	series_a = pd.Series(values_a)
	series_b = pd.Series(values_b)
	if series_a.nunique(dropna=True) < 2 or series_b.nunique(dropna=True) < 2:
		return float("nan")
	return float(series_a.corr(series_b, method="spearman"))


def _sample_graphs(graphs: list[Any], sample_count: int, seed: int) -> list[Any]:
	if sample_count <= 0 or len(graphs) <= sample_count:
		return list(graphs)
	rng = np.random.default_rng(seed)
	indices = np.asarray(rng.choice(len(graphs), size=sample_count, replace=False), dtype=int)
	return [graphs[idx] for idx in indices]


def _physical_masks(graph: Any) -> tuple[torch.Tensor, torch.Tensor]:
	if hasattr(graph, "edge_loss_mask"):
		edge_mask = graph.edge_loss_mask.view(-1) > 0.5
	else:
		edge_mask = torch.ones(graph.edge_attr.size(0), dtype=torch.bool)

	if hasattr(graph, "node_is_virtual"):
		node_mask = graph.node_is_virtual.view(-1) < 0.5
	else:
		node_mask = torch.ones(graph.x.size(0), dtype=torch.bool)

	return node_mask, edge_mask


def _inverse_transform_targets(edge_target_scaler: Any, scaled_values: np.ndarray) -> np.ndarray:
	arr = np.asarray(scaled_values, dtype=np.float32)
	if arr.ndim == 1:
		arr = arr.reshape(-1, 1)
	return np.asarray(edge_target_scaler.inverse_transform(arr), dtype=np.float32).reshape(-1)


def _predict_graph(model: torch.nn.Module, graph: Any, device: torch.device, edge_target_scaler: Any) -> tuple[np.ndarray, np.ndarray]:
	graph_device = graph.to(device)
	with torch.no_grad():
		pred_scaled = model(
			graph_device.x,
			graph_device.edge_index,
			edge_attr=graph_device.edge_attr,
			batch=getattr(graph_device, "batch", torch.zeros(graph_device.x.size(0), dtype=torch.long, device=device)),
			u=graph_device.u,
		).detach().cpu()

	mask = getattr(graph, "edge_loss_mask", None)
	if mask is None:
		keep = torch.ones(pred_scaled.shape[0], dtype=torch.bool)
	else:
		keep = (mask.view(-1) > 0.5).detach().cpu()

	pred_original = _inverse_transform_targets(edge_target_scaler, pred_scaled[keep].numpy())
	true_original = _inverse_transform_targets(edge_target_scaler, graph.y_edge.detach().cpu()[keep].numpy())
	return pred_original, true_original


def _ablate_feature_in_graph(graph: Any, feature_group: str, feature_index: int) -> Any:
	graph_copy = graph.clone()
	node_mask, edge_mask = _physical_masks(graph_copy)

	if feature_group == "node_continuous" or feature_group == "node_mask":
		graph_copy.x[node_mask, feature_index] = 0.0
	elif feature_group == "edge":
		graph_copy.edge_attr[edge_mask, feature_index] = 0.0
	elif feature_group == "global":
		graph_copy.u[:, feature_index] = 0.0
	else:
		raise ValueError(f"Unsupported feature group: {feature_group}")

	return graph_copy


def _graph_feature_summary(graph: Any, feature_group: str, feature_index: int) -> float:
	node_mask, edge_mask = _physical_masks(graph)
	if feature_group == "node_continuous" or feature_group == "node_mask":
		values = graph.x[node_mask, feature_index].detach().cpu().numpy()
	elif feature_group == "edge":
		values = graph.edge_attr[edge_mask, feature_index].detach().cpu().numpy()
	elif feature_group == "global":
		values = graph.u[:, feature_index].detach().cpu().numpy().reshape(-1)
	else:
		raise ValueError(f"Unsupported feature group: {feature_group}")

	if values.size == 0:
		return float("nan")
	return float(np.mean(values))


def _resolve_feature_groups(schema: V4Schema) -> list[tuple[str, str, int]]:
	feature_groups: list[tuple[str, str, int]] = []
	for index, feature_name in enumerate(schema.node_continuous_cols):
		feature_groups.append(("node_continuous", feature_name, index))
	for index, feature_name in enumerate(schema.node_mask_cols, start=len(schema.node_continuous_cols)):
		feature_groups.append(("node_mask", feature_name, index))
	for index, feature_name in enumerate(schema.edge_feature_cols):
		feature_groups.append(("edge", feature_name, index))
	for index, feature_name in enumerate(schema.global_feature_cols):
		feature_groups.append(("global", feature_name, index))
	return feature_groups


def analyze_feature_ablation(
	model: torch.nn.Module,
	test_graphs: list[Any],
	schema: V4Schema,
	edge_target_scaler: Any,
	device: torch.device,
	sample_count: int = 100,
	seed: int = 42,
	output_csv_path: Path | None = None,
) -> pd.DataFrame:
	"""Run leave-one-feature-out sensitivity on a sample of graphs.

	The correlations are computed on the model-input scale: feature values are taken
	from the graph tensors used by the trained model, and the output sensitivity is
	measured as the mean absolute prediction change after zeroing one feature at a time.
	"""
	selected_graphs = _sample_graphs(test_graphs, sample_count=sample_count, seed=seed)
	feature_groups = _resolve_feature_groups(schema)

	rows: list[FeatureImportanceResult] = []
	model.eval()
	for feature_group, feature_name, feature_index in feature_groups:
		feature_values: list[float] = []
		target_means_abs: list[float] = []
		baseline_means: list[float] = []
		ablated_means: list[float] = []
		absolute_deltas: list[float] = []

		for graph in selected_graphs:
			baseline_pred, true_values = _predict_graph(model, graph, device, edge_target_scaler)
			ablated_graph = _ablate_feature_in_graph(graph, feature_group, feature_index)
			ablated_pred, _ = _predict_graph(model, ablated_graph, device, edge_target_scaler)

			feature_values.append(_graph_feature_summary(graph, feature_group, feature_index))
			target_means_abs.append(float(np.mean(np.abs(true_values))))
			baseline_means.append(float(np.mean(baseline_pred)))
			ablated_means.append(float(np.mean(ablated_pred)))
			absolute_deltas.append(float(np.mean(np.abs(ablated_pred - baseline_pred))))

		feature_array = np.asarray(feature_values, dtype=np.float32)
		target_array = np.asarray(target_means_abs, dtype=np.float32)
		baseline_array = np.asarray(baseline_means, dtype=np.float32)
		ablated_array = np.asarray(ablated_means, dtype=np.float32)
		delta_array = np.asarray(absolute_deltas, dtype=np.float32)

		baseline_scale = float(np.mean(np.abs(baseline_array))) if baseline_array.size else float("nan")
		if np.isfinite(baseline_scale) and baseline_scale > 0.0:
			mean_abs_prediction_change_pct = float((np.mean(delta_array) / baseline_scale) * 100.0)
		else:
			mean_abs_prediction_change_pct = float("nan")

		rows.append(
			FeatureImportanceResult(
				feature_name=feature_name,
				feature_group=feature_group,
				pearson_r_target_mean_abs=_pearson_r(feature_array, target_array),
				spearman_r_target_mean_abs=_spearman_r(feature_array, target_array),
				mean_abs_prediction_change_kn=float(np.mean(delta_array)),
				mean_prediction_difference_kn=float(np.mean(ablated_array) - np.mean(baseline_array)),
				mean_abs_prediction_change_pct=mean_abs_prediction_change_pct,
				mean_target_mean_abs_kn=float(np.mean(target_array)),
				mean_pred_baseline_kn=float(np.mean(baseline_array)),
				mean_pred_ablated_kn=float(np.mean(ablated_array)),
				n_samples=int(len(selected_graphs)),
			)
		)

	report = pd.DataFrame([row.__dict__ for row in rows])
	report = report.sort_values(
		by=["mean_abs_prediction_change_kn", "pearson_r_target_mean_abs"],
		ascending=[False, False],
		na_position="last",
	).reset_index(drop=True)
	report["rank"] = np.arange(1, len(report) + 1)

	if output_csv_path is not None:
		output_csv_path.parent.mkdir(parents=True, exist_ok=True)
		report.to_csv(output_csv_path, index=False)

	return report


def run_from_training_results(
	results: dict[str, Any],
	sample_count: int = 100,
	seed: int = 42,
	output_csv_path: Path | None = None,
) -> pd.DataFrame:
	"""Convenience wrapper for notebook outputs from c21_training."""
	model = results["model"]
	test_loader = results["test_loader"]
	device = results["device"]
	schema = results["schema"]
	edge_target_scaler = results["scalers"]["edge_target"] if "scalers" in results else results["edge_target_scaler"]
	test_graphs = list(test_loader.dataset)
	return analyze_feature_ablation(
		model=model,
		test_graphs=test_graphs,
		schema=schema,
		edge_target_scaler=edge_target_scaler,
		device=device,
		sample_count=sample_count,
		seed=seed,
		output_csv_path=output_csv_path,
	)


__all__ = [
	"FeatureImportanceResult",
	"analyze_feature_ablation",
	"run_from_training_results",
]