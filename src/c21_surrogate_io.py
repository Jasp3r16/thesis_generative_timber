import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

import config
from c21_surrogate_model import TrussEdgeGNN, TrussEdgeGNNV2


def _select_model_class(model_variant: str):
    variant = str(model_variant).strip().lower()
    if variant in {"v1", "baseline", "trussedgennconv"}:
        return TrussEdgeGNN, "v1"
    if variant in {"v2", "enhanced", "trussedgennconvv2"}:
        return TrussEdgeGNNV2, "v2"
    return TrussEdgeGNN, "v1"


def _resolve_surrogate_artifact_dir(prefix_sm: str) -> Path:
    """Resolve nested artifact directory if present, else use legacy root."""
    nested = config.SM_EXPORT_PATH / prefix_sm
    if nested.exists() and nested.is_dir():
        return nested
    return config.SM_EXPORT_PATH


def _resolve_prefix(prefix_sm: str | None) -> str:
    def _prefix_has_checkpoint(prefix: str) -> bool:
        artifact_dir = _resolve_surrogate_artifact_dir(prefix)
        nested_candidate = artifact_dir / f"{prefix}_surrogate_model.pt"
        legacy_candidate = config.SM_EXPORT_PATH / f"truss_edge_gnn_{prefix}.pt"
        return nested_candidate.exists() or legacy_candidate.exists()

    if prefix_sm is not None:
        if not _prefix_has_checkpoint(prefix_sm):
            raise FileNotFoundError(f"No checkpoint found for prefix '{prefix_sm}'.")
        return prefix_sm

    prefix_path = config.SM_EXPORT_PATH / "prefix_sm.txt"
    if prefix_path.exists():
        file_prefix = prefix_path.read_text(encoding="utf-8").strip()
        if _prefix_has_checkpoint(file_prefix):
            return file_prefix

    checkpoint_candidates = sorted(
        config.SM_EXPORT_PATH.rglob("*_surrogate_model.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not checkpoint_candidates:
        raise FileNotFoundError("No surrogate model checkpoint found in SM_EXPORT_PATH.")
    return checkpoint_candidates[0].stem.removesuffix("_surrogate_model")


def _resolve_scaler_paths(artifact_dir: Path, prefix_sm: str) -> tuple[Path, Path, Path | None]:
    node_scaler_path = artifact_dir / f"{prefix_sm}_node_scaler.pkl"
    edge_target_scaler_path = artifact_dir / f"{prefix_sm}_edge_target_scaler.pkl"
    global_feature_scaler_path = artifact_dir / f"{prefix_sm}_global_feature_scaler.pkl"

    if not edge_target_scaler_path.exists():
        edge_target_scaler_path = artifact_dir / f"{prefix_sm}_edge_scaler.pkl"

    if not node_scaler_path.exists() or not edge_target_scaler_path.exists():
        legacy_node_scaler_path = config.SM_EXPORT_PATH / f"node_scaler_{prefix_sm}.pkl"
        legacy_edge_scaler_path = config.SM_EXPORT_PATH / f"edge_scaler_{prefix_sm}.pkl"
        if legacy_node_scaler_path.exists() and legacy_edge_scaler_path.exists():
            node_scaler_path = legacy_node_scaler_path
            edge_target_scaler_path = legacy_edge_scaler_path
            global_feature_scaler_path = None

    return node_scaler_path, edge_target_scaler_path, global_feature_scaler_path if global_feature_scaler_path.exists() else None


def _ensure_bidirectional(edge_index: torch.Tensor) -> torch.Tensor:
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    pairs = set(zip(src, dst))
    if all((v, u) in pairs for (u, v) in pairs):
        return edge_index
    src_bi = src + dst
    dst_bi = dst + src
    return torch.tensor([src_bi, dst_bi], dtype=torch.long)


def _extract_node_count_and_coords(design_row: pd.Series) -> np.ndarray:
    pattern = re.compile(r"^v(\d+)_(x|y|z)$")
    grouped: dict[int, dict[str, float]] = {}
    for column in design_row.index:
        match = pattern.match(str(column))
        if not match:
            continue
        idx = int(match.group(1))
        axis = match.group(2)
        grouped.setdefault(idx, {})[axis] = float(design_row[column])

    if not grouped:
        raise ValueError("No node coordinate columns found. Expected columns like v0_x, v0_y, v0_z.")

    sorted_ids = sorted(grouped)
    rows = []
    for idx in sorted_ids:
        axes = grouped[idx]
        if not all(axis in axes for axis in ("x", "y", "z")):
            raise ValueError(f"Incomplete coordinate triplet for node v{idx}.")
        rows.append([axes["x"], axes["y"], axes["z"]])
    return np.asarray(rows, dtype=np.float32)


def _build_node_features(node_raw_xyz: np.ndarray, bundle: dict) -> tuple[torch.Tensor, bool]:
    node_scaler = bundle["node_scaler"]
    node_in_dim = int(bundle["node_in_dim"])
    continuous_dim = int(getattr(node_scaler, "n_features_in_", 3))

    continuous_raw = np.zeros((node_raw_xyz.shape[0], continuous_dim), dtype=np.float32)
    continuous_raw[:, : min(3, continuous_dim)] = node_raw_xyz[:, : min(3, continuous_dim)]
    scaler_feature_names = list(getattr(node_scaler, "feature_names_in_", []))
    if len(scaler_feature_names) == continuous_dim:
        continuous_input = pd.DataFrame(continuous_raw, columns=scaler_feature_names)
    else:
        continuous_input = continuous_raw
    continuous_scaled = node_scaler.transform(continuous_input)

    extra_dim = max(0, node_in_dim - continuous_dim)
    x = np.concatenate([continuous_scaled, np.zeros((node_raw_xyz.shape[0], extra_dim), dtype=np.float32)], axis=1)

    use_virtual_node = bool(bundle.get("use_virtual_node", False))
    if use_virtual_node:
        if extra_dim < 1:
            raise ValueError("Model expects virtual-node mode but node feature dimension has no spare indicator column.")
        virtual = np.zeros((1, node_in_dim), dtype=np.float32)
        virtual[0, -1] = 1.0
        x = np.vstack([x, virtual])

    return torch.from_numpy(x.astype(np.float32)), use_virtual_node


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


def load_surrogate_bundle(prefix_sm: str | None = None, device: str | None = None) -> dict:
    """Load model, scalers and edge topology for surrogate inference."""
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    prefix_sm = _resolve_prefix(prefix_sm)

    artifact_dir = _resolve_surrogate_artifact_dir(prefix_sm)
    model_path = artifact_dir / f"{prefix_sm}_surrogate_model.pt"
    node_scaler_path, edge_target_scaler_path, global_feature_scaler_path = _resolve_scaler_paths(artifact_dir, prefix_sm)
    run_manifest_path = artifact_dir / f"{prefix_sm}_run_manifest.json"
    run_manifest = None
    if run_manifest_path.exists():
        with open(run_manifest_path, "r", encoding="utf-8") as f:
            run_manifest = json.load(f)

    if not model_path.exists():
        legacy_model_path = config.SM_EXPORT_PATH / f"truss_edge_gnn_{prefix_sm}.pt"
        if legacy_model_path.exists():
            model_path = legacy_model_path

    checkpoint = torch.load(model_path, map_location=device_obj)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    node_scaler = joblib.load(node_scaler_path)
    edge_target_scaler = joblib.load(edge_target_scaler_path)
    global_feature_scaler = joblib.load(global_feature_scaler_path) if global_feature_scaler_path is not None else None

    # Prefer inferring dimensions from state_dict to avoid metadata drift between
    # checkpoint/run-manifest and architecture changes.
    inferred_node_in_dim = None
    inferred_edge_in_dim = None
    inferred_global_in_dim = None
    inferred_hidden_dim = None

    conv1_lin_weight = state_dict.get("conv1.lin.weight")
    if conv1_lin_weight is not None and conv1_lin_weight.ndim == 2:
        inferred_hidden_dim = int(conv1_lin_weight.shape[0])
        inferred_node_in_dim = int(conv1_lin_weight.shape[1])

    edge_attr_encoder_weight = state_dict.get("edge_attr_encoder.0.weight")
    if edge_attr_encoder_weight is not None and edge_attr_encoder_weight.ndim == 2:
        inferred_edge_in_dim = int(edge_attr_encoder_weight.shape[1])

    global_encoder_weight = state_dict.get("global_encoder.0.weight")
    if global_encoder_weight is not None and global_encoder_weight.ndim == 2:
        inferred_global_in_dim = int(global_encoder_weight.shape[1])

    node_in_dim = (
        inferred_node_in_dim
        or (checkpoint.get("node_in_dim") if isinstance(checkpoint, dict) else None)
        or int(getattr(node_scaler, "n_features_in_", 3))
    )
    edge_in_dim = (
        inferred_edge_in_dim
        or (checkpoint.get("edge_in_dim") if isinstance(checkpoint, dict) else None)
        or 7
    )
    global_in_dim = (
        inferred_global_in_dim
        or (checkpoint.get("global_in_dim") if isinstance(checkpoint, dict) else None)
        or 3
    )
    hidden_dim = (
        inferred_hidden_dim
        or (checkpoint.get("hidden_dim") if isinstance(checkpoint, dict) else None)
        or 128
    )
    model_variant = checkpoint.get("model_variant", "v1") if isinstance(checkpoint, dict) else "v1"
    dropout_p = checkpoint.get("dropout_p", 0.1) if isinstance(checkpoint, dict) else 0.1
    if run_manifest is not None:
        model_variant = run_manifest.get("model_variant", model_variant)
        dropout_p = run_manifest.get("dropout_p", dropout_p)

    continuous_dim = int(getattr(node_scaler, "n_features_in_", 3))
    extra_node_dim = int(node_in_dim) - continuous_dim
    use_virtual_node = extra_node_dim >= 7
    if run_manifest is not None:
        use_virtual_node = bool(run_manifest.get("use_virtual_node", use_virtual_node))

    model_class, resolved_variant = _select_model_class(model_variant)
    model_kwargs = {
        "node_in_dim": node_in_dim,
        "edge_in_dim": edge_in_dim,
        "global_in_dim": global_in_dim,
        "hidden_dim": hidden_dim,
    }
    
    if resolved_variant == "v2":
        model_kwargs["dropout_p"] = float(dropout_p)

    model = model_class(**model_kwargs).to(device_obj)
    model.load_state_dict(state_dict)
    model.eval()

    edge_index = _ensure_bidirectional(load_edge_index(config.DATA_IO_PATH / "edge_index.json"))

    print(f"Loaded surrogate prefix: {prefix_sm}")
    print(f"Device: {device_obj}")
    print(f"Model: {model_path.name}")
    print(f"Model variant: {resolved_variant}")

    return {
        "prefix_sm": prefix_sm,
        "device": device_obj,
        "model": model,
        "node_scaler": node_scaler,
        "edge_target_scaler": edge_target_scaler,
        "edge_scaler": edge_target_scaler,
        "global_feature_scaler": global_feature_scaler,
        "node_in_dim": int(node_in_dim),
        "edge_in_dim": int(edge_in_dim),
        "global_in_dim": int(global_in_dim),
        "model_variant": resolved_variant,
        "dropout_p": float(dropout_p),
        "use_virtual_node": use_virtual_node,
        "run_manifest": run_manifest,
        "edge_index": edge_index,
    }


def load_and_prepare_stock(stock_csv: Path) -> pd.DataFrame:
    """Load stock robustly and ensure structural columns exist for checks."""
    df = pd.read_csv(stock_csv, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    required_base = ["Member_ID", "Length", "Width", "Depth", "E_modulus_eff"]
    missing_base = [c for c in required_base if c not in df.columns]
    if missing_base:
        raise ValueError(f"Stock file misses required columns: {missing_base}")

    if "f_tk" not in df.columns or "f_c0k" not in df.columns:
        if "f_mk" not in df.columns:
            raise ValueError("Stock file misses f_tk/f_c0k and also has no f_mk for backfill.")

        mk_to_props = {
            18.0: {"f_tk": 11.0, "f_c0k": 18.0},
            24.0: {"f_tk": 14.0, "f_c0k": 21.0},
        }

        f_mk_rounded = pd.to_numeric(df["f_mk"], errors="coerce").round(0)
        if "f_tk" not in df.columns:
            df["f_tk"] = f_mk_rounded.map(lambda v: mk_to_props.get(float(v), {}).get("f_tk"))
        if "f_c0k" not in df.columns:
            df["f_c0k"] = f_mk_rounded.map(lambda v: mk_to_props.get(float(v), {}).get("f_c0k"))

    required_full = ["Member_ID", "Length", "Width", "Depth", "f_c0k", "f_tk", "E_modulus_eff"]
    missing_full = [c for c in required_full if c not in df.columns]
    if missing_full:
        raise ValueError(f"Stock preparation failed. Missing columns: {missing_full}")

    return df


def predict_edge_forces_kn(design_row: pd.Series, bundle: dict) -> pd.DataFrame:
    """Predict per-edge axial force in kN for a single design row."""
    node_raw = _extract_node_count_and_coords(design_row)
    x, use_virtual_node = _build_node_features(node_raw, bundle)

    edge_index = bundle["edge_index"].clone()
    num_physical_edges = edge_index.size(1) // 2
    physical_edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
    physical_edge_mask[:num_physical_edges] = True

    if use_virtual_node:
        virtual_node_idx = x.size(0) - 1
        physical_indices = torch.arange(virtual_node_idx, dtype=torch.long)
        virtual_sources = torch.cat([physical_indices, torch.full((virtual_node_idx,), virtual_node_idx, dtype=torch.long)])
        virtual_targets = torch.cat([torch.full((virtual_node_idx,), virtual_node_idx, dtype=torch.long), physical_indices])
        virtual_edge_index = torch.stack([virtual_sources, virtual_targets], dim=0)
        edge_index = torch.cat([edge_index, virtual_edge_index], dim=1)
        physical_edge_mask = torch.cat([physical_edge_mask, torch.zeros(virtual_edge_index.size(1), dtype=torch.bool)], dim=0)

    edge_attr = torch.zeros((edge_index.size(1), int(bundle.get("edge_in_dim", 7))), dtype=torch.float32)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    u = torch.zeros((1, int(bundle.get("global_in_dim", 3))), dtype=torch.float32)

    x = x.to(bundle["device"])
    edge_index = edge_index.to(bundle["device"])
    edge_attr = edge_attr.to(bundle["device"])
    batch = batch.to(bundle["device"])
    u = u.to(bundle["device"])

    with torch.no_grad():
        pred_scaled = bundle["model"](
            x,
            edge_index,
            edge_attr=edge_attr,
            batch=batch,
            u=u,
        ).detach().cpu().numpy()

    pred_kn_full = bundle["edge_target_scaler"].inverse_transform(pred_scaled).reshape(-1)
    pred_kn = pred_kn_full[physical_edge_mask.numpy()]

    starts = bundle["edge_index"][0][:num_physical_edges].detach().cpu().numpy()
    ends = bundle["edge_index"][1][:num_physical_edges].detach().cpu().numpy()
    lengths_m = np.linalg.norm(node_raw[starts] - node_raw[ends], axis=1)

    return pd.DataFrame(
        {
            "edge_id": [f"e{i}" for i in range(len(pred_kn))],
            "V1": starts,
            "V2": ends,
            "length_m": lengths_m,
            "axial_force_kn": pred_kn,
        }
    )


__all__ = [
    "load_edge_index",
    "load_surrogate_bundle",
    "load_and_prepare_stock",
    "predict_edge_forces_kn",
]
