from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import config
from c21_surrogate_model_v4 import create_model


def _resolve_prefix(prefix_sm: str | None) -> str:
	if prefix_sm:
		return str(prefix_sm).strip()

	prefix_path = config.SM_EXPORT_PATH / "prefix_sm.txt"
	if prefix_path.exists():
		candidate = prefix_path.read_text(encoding="utf-8").strip()
		if candidate:
			return candidate

	checkpoint_candidates = sorted(
		config.SM_EXPORT_PATH.rglob("*.pth"),
		key=lambda path: path.stat().st_mtime,
		reverse=True,
	)
	if not checkpoint_candidates:
		raise FileNotFoundError("No surrogate model checkpoint (.pth) found in SM_EXPORT_PATH.")
	return checkpoint_candidates[0].stem


def _resolve_artifact_dir(prefix_sm: str) -> Path:
	candidate_dir = config.SM_EXPORT_PATH / prefix_sm
	if candidate_dir.exists() and candidate_dir.is_dir():
		return candidate_dir

	hits = sorted(
		config.SM_EXPORT_PATH.rglob(f"{prefix_sm}.pth"),
		key=lambda path: path.stat().st_mtime,
		reverse=True,
	)
	if hits:
		return hits[0].parent

	return config.SM_EXPORT_PATH


def _resolve_model_path(prefix_sm: str, artifact_dir: Path) -> Path:
	candidate = artifact_dir / f"{prefix_sm}.pth"
	if candidate.exists():
		return candidate

	hits = sorted(
		config.SM_EXPORT_PATH.rglob(f"{prefix_sm}.pth"),
		key=lambda path: path.stat().st_mtime,
		reverse=True,
	)
	if hits:
		return hits[0]

	raise FileNotFoundError(f"No checkpoint found for prefix '{prefix_sm}'.")


def _resolve_scalers_path(prefix_sm: str, artifact_dir: Path) -> Path:
	candidate = artifact_dir / f"{prefix_sm}_scalers.json"
	if candidate.exists():
		return candidate

	hits = sorted(
		artifact_dir.glob("*_scalers.json"),
		key=lambda path: path.stat().st_mtime,
		reverse=True,
	)
	if hits:
		return hits[0]

	raise FileNotFoundError(f"No scaler metadata found for prefix '{prefix_sm}'.")


def _resolve_calibration_path(prefix_sm: str, artifact_dir: Path) -> Path | None:
	candidate = artifact_dir / f"{prefix_sm}_calibration.json"
	if candidate.exists():
		return candidate

	hits = sorted(
		artifact_dir.glob("*_calibration.json"),
		key=lambda path: path.stat().st_mtime,
		reverse=True,
	)
	if hits:
		return hits[0]

	return None


def load_edge_index(edge_index_path: Path) -> torch.Tensor:
	"""Load edge topology from JSON and return tensor of shape [2, num_edges]."""
	with open(edge_index_path, "r", encoding="utf-8") as f:
		payload = json.load(f)

	if isinstance(payload, dict):
		starts = payload.get("start") or payload.get("starts") or payload.get("source")
		ends = payload.get("end") or payload.get("ends") or payload.get("target")
		if starts is None or ends is None:
			raise ValueError("edge_index.json dictionary misses start/end keys.")
		edge_index = torch.tensor([starts, ends], dtype=torch.long)
	elif isinstance(payload, list) and len(payload) == 2:
		edge_index = torch.tensor(payload, dtype=torch.long)
	else:
		raise ValueError("Unsupported edge_index.json format.")

	if edge_index.shape[0] != 2:
		raise ValueError("edge_index must have shape [2, num_edges].")
	return edge_index


def _load_scalers(scalers_path: Path) -> dict[str, Any]:
	with open(scalers_path, "r", encoding="utf-8") as f:
		payload = json.load(f)

	node_cols = payload.get("node_cols") or []
	edge_cols = payload.get("edge_cols") or []
	node_mean = payload.get("node_mean") or {}
	node_std = payload.get("node_std") or {}
	edge_mean = payload.get("edge_mean") or {}
	edge_std = payload.get("edge_std") or {}

	if not node_cols or not edge_cols:
		raise ValueError("Scaler metadata is missing node_cols or edge_cols.")

	return {
		"node_cols": tuple(node_cols),
		"edge_cols": tuple(edge_cols),
		"node_mean": node_mean,
		"node_std": node_std,
		"edge_mean": edge_mean,
		"edge_std": edge_std,
	}


def load_surrogate_bundle(prefix_sm: str | None = None, device: str | None = None) -> dict[str, Any]:
	"""Load model checkpoint, scalers, topology, and calibration metadata."""
	device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

	prefix_sm = _resolve_prefix(prefix_sm)
	artifact_dir = _resolve_artifact_dir(prefix_sm)
	model_path = _resolve_model_path(prefix_sm, artifact_dir)
	scalers_path = _resolve_scalers_path(prefix_sm, artifact_dir)
	calibration_path = _resolve_calibration_path(prefix_sm, artifact_dir)

	checkpoint = torch.load(model_path, map_location=device_obj)
	state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

	scalers = _load_scalers(scalers_path)
	node_in_dim = int(len(scalers["node_cols"]))
	edge_in_dim = int(len(scalers["edge_cols"]))

	model = create_model(
		node_features_dim=node_in_dim,
		edge_features_dim=edge_in_dim,
		device=device_obj,
	)
	model.load_state_dict(state_dict)
	model.eval()

	edge_index = load_edge_index(config.DATA_IO_PATH / "edge_index.json")

	calibration = None
	if calibration_path is not None:
		with open(calibration_path, "r", encoding="utf-8") as f:
			calibration = json.load(f)

	return {
		"prefix_sm": prefix_sm,
		"device": device_obj,
		"model": model,
		"edge_index": edge_index,
		"scalers": scalers,
		"calibration": calibration,
		"artifact_dir": artifact_dir,
		"model_path": model_path,
	}


def apply_psi_gamma(probabilities: np.ndarray, gamma: float = 2.0, eps: float = 1e-6) -> np.ndarray:
	"""Apply the closed-form Psi_gamma calibration for focal-loss probabilities."""
	probs = np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1.0 - eps)
	one_minus = 1.0 - probs

	h_q = probs * (one_minus ** gamma) - gamma * (one_minus ** (gamma - 1.0)) * probs * np.log(probs)
	h_1_q = one_minus * (probs ** gamma) - gamma * (probs ** (gamma - 1.0)) * one_minus * np.log(one_minus)

	denominator = h_q + h_1_q
	return h_q / denominator


def calibrate_failure_probabilities(
	probabilities: np.ndarray,
	calibration: dict[str, Any] | None,
	fallback_gamma: float = 2.0,
) -> np.ndarray:
	"""Apply stored Psi_gamma calibration (or fallback gamma) to failure probabilities."""
	if calibration is None:
		gamma = fallback_gamma
	else:
		gamma = float(calibration.get("gamma", fallback_gamma))
	return apply_psi_gamma(probabilities, gamma=gamma)


def compute_failure_threshold(
	calibration: dict[str, Any] | None,
	*,
	safety_margin: float = 0.8,
	override_threshold: float | None = None,
	fallback_prevalence: float = 0.5,
) -> float:
	"""Compute the failure threshold from prevalence and safety margin."""
	if override_threshold is not None:
		threshold = float(override_threshold)
		return float(min(max(threshold, 0.0), 1.0))

	prevalence = float(fallback_prevalence)
	if calibration is not None:
		prevalence = float(calibration.get("prevalence", prevalence))

	threshold = float(prevalence) * float(safety_margin)
	return float(min(max(threshold, 0.0), 1.0))


def _resolve_sample_id_column(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> str | None:
	for candidate in ("sample_id", "Sample_ID", "SampleId"):
		if candidate in nodes_df.columns and candidate in edges_df.columns:
			return candidate
	return None


def _select_sample_frame(
	nodes_df: pd.DataFrame,
	edges_df: pd.DataFrame,
	sample_id: Any | None,
	sample_id_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	if sample_id_col is None:
		return nodes_df, edges_df

	nodes_ids = nodes_df[sample_id_col].unique()
	edges_ids = edges_df[sample_id_col].unique()
	shared_ids = sorted(set(nodes_ids).intersection(edges_ids))
	if not shared_ids:
		raise ValueError("No matching sample IDs between node and edge tables.")

	if sample_id is None:
		if len(shared_ids) > 1:
			raise ValueError("Multiple sample IDs found; pass sample_id to select one.")
		sample_id = shared_ids[0]

	nodes_filtered = nodes_df.loc[nodes_df[sample_id_col] == sample_id].copy()
	edges_filtered = edges_df.loc[edges_df[sample_id_col] == sample_id].copy()
	return nodes_filtered, edges_filtered


def predict_edge_failure_probabilities(
	nodes_df: pd.DataFrame,
	edges_df: pd.DataFrame,
	bundle: dict[str, Any],
	edge_index: torch.Tensor | None = None,
	sample_id: Any | None = None,
	apply_calibration: bool = True,
) -> pd.DataFrame:
	"""Predict per-edge failure probabilities and apply Psi_gamma calibration."""
	scalers = bundle["scalers"]
	node_cols = scalers["node_cols"]
	edge_cols = scalers["edge_cols"]

	sample_id_col = _resolve_sample_id_column(nodes_df, edges_df)
	nodes_df, edges_df = _select_sample_frame(nodes_df, edges_df, sample_id, sample_id_col)

	missing_nodes = [c for c in node_cols if c not in nodes_df.columns]
	if missing_nodes:
		raise KeyError(f"Missing required node columns: {missing_nodes}")

	missing_edges = [c for c in edge_cols if c not in edges_df.columns]
	if missing_edges:
		raise KeyError(f"Missing required edge columns: {missing_edges}")

	edge_index_local = edge_index if edge_index is not None else bundle["edge_index"]
	expected_nodes = int(edge_index_local.max().item()) + 1
	if len(nodes_df) != expected_nodes:
		raise ValueError(
			"Node count does not match topology. "
			f"expected={expected_nodes}, got={len(nodes_df)}"
		)
	if len(edges_df) != edge_index_local.shape[1]:
		raise ValueError(
			"Edge count does not match topology. "
			f"expected={edge_index_local.shape[1]}, got={len(edges_df)}"
		)

	node_means = pd.Series(scalers["node_mean"], dtype=float)
	node_stds = pd.Series(scalers["node_std"], dtype=float).replace(0, 1.0)
	edge_means = pd.Series(scalers["edge_mean"], dtype=float)
	edge_stds = pd.Series(scalers["edge_std"], dtype=float).replace(0, 1.0)

	node_features = (nodes_df.loc[:, node_cols] - node_means) / node_stds
	edge_features = (edges_df.loc[:, edge_cols] - edge_means) / edge_stds

	node_features = node_features.clip(-5, 5)
	edge_features = edge_features.clip(-5, 5)

	x = torch.tensor(node_features.values, dtype=torch.float32, device=bundle["device"])
	edge_attr = torch.tensor(edge_features.values, dtype=torch.float32, device=bundle["device"])
	edge_index_local = edge_index_local.to(bundle["device"])

	with torch.no_grad():
		raw_probs = bundle["model"](x, edge_index_local, edge_attr).detach().cpu().numpy().reshape(-1)

	if apply_calibration:
		calibrated = calibrate_failure_probabilities(raw_probs, bundle.get("calibration"))
	else:
		calibrated = raw_probs.copy()

	edge_id_col = None
	for candidate in ("edge_id", "Edge_ID", "Element_ID"):
		if candidate in edges_df.columns:
			edge_id_col = candidate
			break

	if edge_id_col is not None:
		edge_ids = edges_df[edge_id_col].astype(str).tolist()
	else:
		edge_ids = [f"e{i}" for i in range(len(raw_probs))]

	return pd.DataFrame(
		{
			"edge_id": edge_ids,
			"failure_prob_raw": raw_probs,
			"failure_prob_calibrated": calibrated,
		}
	)


__all__ = [
	"load_edge_index",
	"load_surrogate_bundle",
	"apply_psi_gamma",
	"calibrate_failure_probabilities",
	"compute_failure_threshold",
	"predict_edge_failure_probabilities",
]
