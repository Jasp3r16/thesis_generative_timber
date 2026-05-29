from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import config
from c21_surrogate_model_v4 import create_model

NUM_EDGES_PHYSICAL = 120   # physical members — constant regardless of bi/uni

# Feature column order — must match c28_stage_GNN.py exactly
_NODE_COLS = ["x", "y", "z", "Tx", "Ty", "Tz", "Rx", "Ry", "Rz", "Fz"]
_EDGE_COLS = ["Width_m", "Depth_m", "Length", "E", "Iy", "Iz", "J", "EA/L", "N_mean_EA"]


# =============================================================================
# INTERNAL RESOLVERS
# =============================================================================

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
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not checkpoint_candidates:
        raise FileNotFoundError(
            "No surrogate model checkpoint (.pth) found in SM_EXPORT_PATH."
        )
    import warnings
    chosen = checkpoint_candidates[0]
    if len(checkpoint_candidates) > 1:
        warnings.warn(
            f"prefix_sm not set and prefix_sm.txt not found. Multiple checkpoints "
            f"exist — loading most recent: '{chosen.name}'. Pass prefix_sm "
            "explicitly to avoid loading the wrong model.",
            stacklevel=3,
        )
    else:
        warnings.warn(
            f"prefix_sm not set and prefix_sm.txt not found. Derived prefix from "
            f"'{chosen.name}' — artifact lookup may fail if the filename stem does "
            "not match the export prefix. Create prefix_sm.txt to avoid this.",
            stacklevel=3,
        )
    return chosen.stem


def _resolve_artifact_dir(prefix_sm: str) -> Path:
    candidate_dir = config.SM_EXPORT_PATH / prefix_sm
    if candidate_dir.exists() and candidate_dir.is_dir():
        return candidate_dir

    hits = sorted(
        config.SM_EXPORT_PATH.rglob(f"{prefix_sm}.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if hits:
        return hits[0].parent

    import warnings
    warnings.warn(
        f"No artifact directory found for prefix '{prefix_sm}'. "
        "Falling back to SM_EXPORT_PATH root — scalers and edge_index may not "
        "match the checkpoint if multiple models exist there. "
        "Create a named subdirectory or ensure prefix_sm.txt is current.",
        stacklevel=3,
    )
    return config.SM_EXPORT_PATH


def _resolve_model_path(prefix_sm: str, artifact_dir: Path) -> Path:
    candidate = artifact_dir / f"{prefix_sm}.pth"
    if candidate.exists():
        return candidate

    hits = sorted(
        config.SM_EXPORT_PATH.rglob(f"{prefix_sm}.pth"),
        key=lambda p: p.stat().st_mtime,
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
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if hits:
        return hits[0]

    raise FileNotFoundError(
        f"No scaler metadata found for prefix '{prefix_sm}'."
    )


def _resolve_inference_config_path(prefix_sm: str, artifact_dir: Path) -> Path | None:
    candidate = artifact_dir / f"{prefix_sm}_inference_config.json"
    if candidate.exists():
        return candidate

    hits = sorted(
        artifact_dir.glob("*_inference_config.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return hits[0] if hits else None


def _resolve_edge_index_path(prefix_sm: str, artifact_dir: Path) -> Path:
    """
    Load edge_index from the artifact folder — NOT from DATA_IO_PATH.
    This guarantees the topology matches the checkpoint regardless of what
    the working directory's edge_index.json currently contains.
    """
    candidate = artifact_dir / f"{prefix_sm}_edge_index.json"
    if candidate.exists():
        return candidate

    hits = sorted(
        artifact_dir.glob("*_edge_index.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if hits:
        return hits[0]

    raise FileNotFoundError(
        f"No edge_index.json found in artifact folder '{artifact_dir}'. "
        "Re-export the model using c21_export_v3.py to include it."
    )


# =============================================================================
# LOADERS
# =============================================================================

def load_edge_index(edge_index_path: Path) -> torch.Tensor:
    """Load edge topology from JSON and return tensor [2, num_edges]."""
    with open(edge_index_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        starts = payload.get("start") or payload.get("starts") or payload.get("source")
        ends   = payload.get("end")   or payload.get("ends")   or payload.get("target")
        if starts is None or ends is None:
            raise ValueError("edge_index.json dict is missing start/end keys.")
        edge_index = torch.tensor([starts, ends], dtype=torch.long)
    elif isinstance(payload, list) and len(payload) == 2:
        edge_index = torch.tensor(payload, dtype=torch.long)
    else:
        raise ValueError(
            f"Unsupported edge_index.json format "
            f"(type={type(payload)}, len={len(payload) if isinstance(payload, list) else 'n/a'})."
        )

    if edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape [2, num_edges], got {tuple(edge_index.shape)}."
        )
    return edge_index


def _load_scalers(scalers_path: Path) -> dict[str, Any]:
    with open(scalers_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    node_cols = payload.get("node_cols") or []
    edge_cols = payload.get("edge_cols") or []
    node_mean = payload.get("node_mean") or {}
    node_std  = payload.get("node_std")  or {}
    edge_mean = payload.get("edge_mean") or {}
    edge_std  = payload.get("edge_std")  or {}

    if not node_cols or not edge_cols:
        raise ValueError("Scaler metadata is missing node_cols or edge_cols.")

    return {
        "node_cols": tuple(node_cols),
        "edge_cols": tuple(edge_cols),
        "node_mean": node_mean,
        "node_std":  node_std,
        "edge_mean": edge_mean,
        "edge_std":  edge_std,
    }


# =============================================================================
# MAIN LOADER
# =============================================================================

def load_surrogate_bundle(
    prefix_sm: str | None = None,
    device:    str | None = None,
) -> dict[str, Any]:
    """
    Load model checkpoint, scalers, topology, and architecture config.

    edge_index is loaded from the artifact folder (not DATA_IO_PATH) to
    guarantee topology consistency with the checkpoint.

    num_edges and bidirectional are derived from the loaded edge_index and
    stored in the bundle — required by c21_stage_gnn_v3.

    Returns
    -------
    bundle : dict with keys:
        prefix_sm, device, model, edge_index, num_edges, bidirectional,
        scalers, artifact_dir, model_path
    """
    device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_obj = torch.device(device_str)

    prefix_sm    = _resolve_prefix(prefix_sm)
    artifact_dir = _resolve_artifact_dir(prefix_sm)
    model_path   = _resolve_model_path(prefix_sm, artifact_dir)
    scalers_path = _resolve_scalers_path(prefix_sm, artifact_dir)
    ei_path      = _resolve_edge_index_path(prefix_sm, artifact_dir)
    inf_cfg_path = _resolve_inference_config_path(prefix_sm, artifact_dir)

    # Load inference config for architecture params (explicit > defaults)
    inf_config: dict[str, Any] = {}
    if inf_cfg_path is not None:
        with open(inf_cfg_path, "r", encoding="utf-8") as f:
            inf_config = json.load(f)
        print(f"[IO] Inference config loaded from {inf_cfg_path.name}")
    else:
        print("[IO] Warning: no inference_config.json found; "
              "using create_model() defaults. Re-export with c21_export_v3.py.")

    scalers     = _load_scalers(scalers_path)
    node_in_dim = len(scalers["node_cols"])
    edge_in_dim = len(scalers["edge_cols"])

    model = create_model(
        node_features_dim = node_in_dim,
        edge_features_dim = edge_in_dim,
        hidden_dim        = int(inf_config.get("hidden_dim",  128)),
        num_layers        = int(inf_config.get("num_layers",  4)),
        dropout_p         = float(inf_config.get("dropout_p", 0.1)),
        device            = device_str,
    )

    checkpoint = torch.load(model_path, map_location=device_obj, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.eval()

    best_epoch = checkpoint.get("best_epoch", "?")
    best_loss  = checkpoint.get("best_val_loss", float("nan"))
    epoch_str  = f"{best_epoch + 1}" if isinstance(best_epoch, int) else str(best_epoch)
    print(f"[IO] Checkpoint loaded — epoch {epoch_str}  val_loss={best_loss:.6f}")

    # Load edge_index from artifact folder
    edge_index    = load_edge_index(ei_path).to(device_obj)
    num_edges     = int(edge_index.shape[1])
    bidirectional = num_edges == 2 * NUM_EDGES_PHYSICAL

    model.cache_topology(edge_index)

    print(
        f"[IO] Model ready on {device_str}  |  "
        f"edge_index: {tuple(edge_index.shape)}  |  "
        f"{'bidirectional' if bidirectional else 'unidirectional'}"
    )

    # Use the scalers' own column lists — authoritative order from training,
    # avoids KeyError when loading an older model with different feature set.
    _nc = scalers["node_cols"]
    _ec = scalers["edge_cols"]
    norm_stats = {
        "node_means": np.array([scalers["node_mean"][c] for c in _nc]),
        "node_stds":  np.array([scalers["node_std"][c]  for c in _nc]),
        "edge_means": np.array([scalers["edge_mean"][c] for c in _ec]),
        "edge_stds":  np.array([scalers["edge_std"][c]  for c in _ec]),
    }

    return {
        "prefix_sm":     prefix_sm,
        "device":        device_obj,
        "model":         model,
        "edge_index":    edge_index,
        "num_edges":     num_edges,
        "bidirectional": bidirectional,
        "scalers":       scalers,
        "norm_stats":    norm_stats,
        "config":        inf_config,
        "artifact_dir":  artifact_dir,
        "model_path":    model_path,
    }


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _resolve_sample_id_column(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> str | None:
    for candidate in ("sample_id", "Sample_ID", "SampleId"):
        if candidate in nodes_df.columns and candidate in edges_df.columns:
            return candidate
    return None


def _select_sample_frame(
    nodes_df:      pd.DataFrame,
    edges_df:      pd.DataFrame,
    sample_id:     Any | None,
    sample_id_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sample_id_col is None:
        return nodes_df, edges_df

    nodes_ids  = nodes_df[sample_id_col].unique()
    edges_ids  = edges_df[sample_id_col].unique()
    shared_ids = sorted(set(nodes_ids).intersection(edges_ids))
    if not shared_ids:
        raise ValueError("No matching sample IDs between node and edge tables.")

    if sample_id is None:
        if len(shared_ids) > 1:
            raise ValueError(
                f"Multiple sample IDs found ({len(shared_ids)}); "
                "pass sample_id to select one."
            )
        sample_id = shared_ids[0]

    return (
        nodes_df.loc[nodes_df[sample_id_col] == sample_id].copy(),
        edges_df.loc[edges_df[sample_id_col] == sample_id].copy(),
    )


# =============================================================================
# INFERENCE
# =============================================================================

def predict_edge_failure_probabilities(
    nodes_df:   pd.DataFrame,
    edges_df:   pd.DataFrame,
    bundle:     dict[str, Any],
    edge_index: torch.Tensor | None = None,
    sample_id:  Any | None = None,
) -> pd.DataFrame:
    """
    Predict per-edge failure probabilities for a single sample.

    edges_df should contain the original CSV rows (120 rows per sample).
    Bidirectional duplication is handled internally based on bundle["bidirectional"].
    No calibration is applied — raw model probabilities are returned.
    Use bundle inference_config["recommended_threshold"] (or 0.35) for decisions.

    Returns
    -------
    DataFrame with columns:
        edge_id              — member identifier
        failure_prob_raw     — P(unsafe) from model, range [0, 1]
        predicted_unsafe     — bool, True if failure_prob_raw >= threshold
    """
    scalers   = bundle["scalers"]
    node_cols = scalers["node_cols"]
    edge_cols = scalers["edge_cols"]

    sample_id_col = _resolve_sample_id_column(nodes_df, edges_df)
    nodes_df, edges_df = _select_sample_frame(
        nodes_df, edges_df, sample_id, sample_id_col
    )

    missing_nodes = [c for c in node_cols if c not in nodes_df.columns]
    if missing_nodes:
        raise KeyError(f"Missing required node columns: {missing_nodes}")

    missing_edges = [c for c in edge_cols if c not in edges_df.columns]
    if missing_edges:
        raise KeyError(f"Missing required edge columns: {missing_edges}")

    # Validate node count against topology
    edge_index_local = (edge_index if edge_index is not None
                        else bundle["edge_index"])
    expected_nodes = int(edge_index_local.max().item()) + 1
    if len(nodes_df) != expected_nodes:
        raise ValueError(
            f"Node count mismatch: expected {expected_nodes}, got {len(nodes_df)}."
        )

    # Validate edge count against physical members (CSV rows), not edge_index size
    if len(edges_df) != NUM_EDGES_PHYSICAL:
        raise ValueError(
            f"Edge count mismatch: expected {NUM_EDGES_PHYSICAL} CSV rows, "
            f"got {len(edges_df)}."
        )

    # Build normalised features
    node_means = pd.Series(scalers["node_mean"], dtype=float)
    node_stds  = pd.Series(scalers["node_std"],  dtype=float).replace(0, 1.0)
    edge_means = pd.Series(scalers["edge_mean"], dtype=float)
    edge_stds  = pd.Series(scalers["edge_std"],  dtype=float).replace(0, 1.0)

    node_feat = ((nodes_df[list(node_cols)] - node_means) / node_stds).clip(-5, 5)
    edge_feat = ((edges_df[list(edge_cols)] - edge_means) / edge_stds).clip(-5, 5)

    x         = torch.tensor(node_feat.values, dtype=torch.float32,
                              device=bundle["device"])
    edge_attr = torch.tensor(edge_feat.values, dtype=torch.float32,
                              device=bundle["device"])

    # Duplicate edge features for bidirectional graph
    if bundle["bidirectional"]:
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

    edge_index_local = edge_index_local.to(bundle["device"])

    with torch.no_grad():
        raw_preds = bundle["model"](x, edge_index_local, edge_attr)

    # Take first NUM_EDGES_PHYSICAL predictions (physical members only)
    raw_probs = raw_preds[:NUM_EDGES_PHYSICAL, 0].cpu().numpy()

    # Threshold from inference_config if available, else default
    inf_config = bundle.get("config", {})
    threshold  = float(inf_config.get("recommended_threshold", 0.35))

    # Build edge IDs
    edge_id_col = next(
        (c for c in ("edge_id", "Edge_ID", "Element_ID") if c in edges_df.columns),
        None,
    )
    edge_ids = (
        edges_df[edge_id_col].astype(str).tolist()
        if edge_id_col is not None
        else [f"e{i}" for i in range(len(raw_probs))]
    )

    return pd.DataFrame({
        "edge_id":          edge_ids,
        "failure_prob_raw": raw_probs,
        "predicted_unsafe": raw_probs >= threshold,
    })


__all__ = [
    "load_edge_index",
    "load_surrogate_bundle",
    "predict_edge_failure_probabilities",
]