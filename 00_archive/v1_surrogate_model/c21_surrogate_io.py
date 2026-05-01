import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

import config
from c21_surrogate_model_v1 import TrussEdgeGNN, TrussEdgeGNNV2


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

    # Also support artifacts stored in suffixed folders where filenames use the
    # unsuffixed stem (e.g., folder *_F6, file <stem>_surrogate_model.pt).
    base_prefix = _normalize_prefix(prefix_sm)
    model_hits = sorted(
        config.SM_EXPORT_PATH.rglob(f"{base_prefix}_surrogate_model.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if model_hits:
        return model_hits[0].parent

    return config.SM_EXPORT_PATH


def _normalize_prefix(prefix_sm: str) -> str:
    """Drop optional feature-count suffix from folder-style prefixes."""
    return re.sub(r"_F\d+$", "", str(prefix_sm).strip())


def _resolve_model_path(prefix_sm: str) -> Path:
    """Resolve model checkpoint path across nested and legacy layouts."""
    candidates: list[str] = []
    normalized = _normalize_prefix(prefix_sm)
    candidates.append(prefix_sm)
    if normalized != prefix_sm:
        candidates.append(normalized)

    for candidate in candidates:
        artifact_dir = _resolve_surrogate_artifact_dir(candidate)
        model_path = artifact_dir / f"{candidate}_surrogate_model.pt"
        if model_path.exists():
            return model_path

        # Fallback: locate by filename anywhere below export root.
        hits = sorted(
            config.SM_EXPORT_PATH.rglob(f"{candidate}_surrogate_model.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if hits:
            return hits[0]

        legacy_model_path = config.SM_EXPORT_PATH / f"truss_edge_gnn_{candidate}.pt"
        if legacy_model_path.exists():
            return legacy_model_path

    raise FileNotFoundError(f"No checkpoint found for prefix '{prefix_sm}'.")


def _resolve_prefix(prefix_sm: str | None) -> str:
    def _prefix_has_checkpoint(prefix: str) -> bool:
        try:
            _resolve_model_path(prefix)
            return True
        except FileNotFoundError:
            return False

    if prefix_sm is not None:
        normalized = _normalize_prefix(prefix_sm)
        if _prefix_has_checkpoint(prefix_sm):
            return prefix_sm
        if _prefix_has_checkpoint(normalized):
            return normalized
        raise FileNotFoundError(f"No checkpoint found for prefix '{prefix_sm}'.")

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
    prefix_candidates = [prefix_sm]
    normalized_prefix = _normalize_prefix(prefix_sm)
    if normalized_prefix != prefix_sm:
        prefix_candidates.append(normalized_prefix)

    node_scaler_path: Path | None = None
    edge_target_scaler_path: Path | None = None
    global_feature_scaler_path: Path | None = None

    for candidate in prefix_candidates:
        candidate_node = artifact_dir / f"{candidate}_node_scaler.pkl"
        candidate_edge_target = artifact_dir / f"{candidate}_edge_target_scaler.pkl"
        candidate_edge_legacy = artifact_dir / f"{candidate}_edge_scaler.pkl"
        candidate_global = artifact_dir / f"{candidate}_global_feature_scaler.pkl"

        if node_scaler_path is None and candidate_node.exists():
            node_scaler_path = candidate_node
        if edge_target_scaler_path is None:
            if candidate_edge_target.exists():
                edge_target_scaler_path = candidate_edge_target
            elif candidate_edge_legacy.exists():
                edge_target_scaler_path = candidate_edge_legacy
        if global_feature_scaler_path is None and candidate_global.exists():
            global_feature_scaler_path = candidate_global

        if node_scaler_path is not None and edge_target_scaler_path is not None:
            break

    if node_scaler_path is None or edge_target_scaler_path is None:
        for candidate in prefix_candidates:
            legacy_node_scaler_path = config.SM_EXPORT_PATH / f"node_scaler_{candidate}.pkl"
            legacy_edge_scaler_path = config.SM_EXPORT_PATH / f"edge_scaler_{candidate}.pkl"
            if legacy_node_scaler_path.exists() and legacy_edge_scaler_path.exists():
                node_scaler_path = legacy_node_scaler_path
                edge_target_scaler_path = legacy_edge_scaler_path
                global_feature_scaler_path = None
                break

    if node_scaler_path is None or edge_target_scaler_path is None:
        raise FileNotFoundError(
            f"Could not resolve scaler files for prefix '{prefix_sm}' in {artifact_dir}"
        )

    return node_scaler_path, edge_target_scaler_path, global_feature_scaler_path


def _ensure_bidirectional(edge_index: torch.Tensor) -> torch.Tensor:
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    pairs = set(zip(src, dst))
    if all((v, u) in pairs for (u, v) in pairs):
        return edge_index
    src_bi = src + dst
    dst_bi = dst + src
    return torch.tensor([src_bi, dst_bi], dtype=torch.long)


def _extract_node_feature_groups(design_row: pd.Series) -> dict[int, dict[str, float]]:
    pattern = re.compile(r"^v(\d+)_(.+)$")
    grouped: dict[int, dict[str, float]] = {}
    for column in design_row.index:
        match = pattern.match(str(column))
        if not match:
            continue
        idx = int(match.group(1))
        feature_name = match.group(2)
        grouped.setdefault(idx, {})[feature_name] = float(design_row[column])

    if not grouped:
        raise ValueError("No node columns found. Expected columns like v0_x, v0_y, v0_z.")

    return grouped


def _resolve_node_feature_layout(bundle: dict) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    run_manifest = bundle.get("run_manifest") or {}
    node_continuous_cols = tuple(run_manifest.get("selected_node_continuous_cols") or ())
    node_mask_cols = tuple(run_manifest.get("selected_node_mask_cols") or ())
    use_virtual_node = bool(run_manifest.get("use_virtual_node", bundle.get("use_virtual_node", False)))

    if not node_continuous_cols:
        scaler_feature_names = tuple(getattr(bundle["node_scaler"], "feature_names_in_", ()))
        if scaler_feature_names:
            node_continuous_cols = scaler_feature_names

    if not node_continuous_cols:
        raise ValueError("Surrogate bundle does not define node continuous feature columns.")

    return node_continuous_cols, node_mask_cols, use_virtual_node


def _resolve_edge_feature_layout(bundle: dict) -> tuple[str, ...]:
    run_manifest = bundle.get("run_manifest") or {}
    edge_feature_cols = tuple(run_manifest.get("selected_edge_feature_cols") or ())
    if not edge_feature_cols:
        raise ValueError("Surrogate bundle does not define edge feature columns.")
    return edge_feature_cols


def _build_node_features(design_row: pd.Series, bundle: dict) -> tuple[torch.Tensor, bool]:
    node_scaler = bundle["node_scaler"]
    node_in_dim = int(bundle["node_in_dim"])
    node_feature_groups = _extract_node_feature_groups(design_row)
    sorted_node_ids = sorted(node_feature_groups)

    node_continuous_cols, node_mask_cols, use_virtual_node = _resolve_node_feature_layout(bundle)

    continuous_dim = int(getattr(node_scaler, "n_features_in_", len(node_continuous_cols)))
    if continuous_dim != len(node_continuous_cols):
        raise ValueError(
            "Node scaler feature count does not match the trained node continuous columns: "
            f"scaler={continuous_dim}, manifest={len(node_continuous_cols)}"
        )

    missing_required: list[str] = []
    continuous_rows: list[list[float]] = []
    mask_rows: list[list[float]] = []

    for node_id in sorted_node_ids:
        node_features = node_feature_groups[node_id]

        continuous_row: list[float] = []
        for feature_name in node_continuous_cols:
            column_name = f"v{node_id}_{feature_name}"
            if feature_name not in node_features:
                missing_required.append(column_name)
                continuous_row.append(np.nan)
            else:
                continuous_row.append(float(node_features[feature_name]))
        continuous_rows.append(continuous_row)

        mask_row: list[float] = []
        for feature_name in node_mask_cols:
            column_name = f"v{node_id}_{feature_name}"
            if feature_name not in node_features:
                missing_required.append(column_name)
                mask_row.append(np.nan)
            else:
                mask_row.append(float(node_features[feature_name]))
        mask_rows.append(mask_row)

    if missing_required:
        missing_display = ", ".join(missing_required[:12])
        suffix = "..." if len(missing_required) > 12 else ""
        raise ValueError(
            "Structural surrogate inputs are incomplete for the trained feature schema. "
            f"Missing columns: {missing_display}{suffix}"
        )

    continuous_raw = np.asarray(continuous_rows, dtype=np.float32)
    scaler_feature_names = list(getattr(node_scaler, "feature_names_in_", []))
    if len(scaler_feature_names) == continuous_dim:
        continuous_input = pd.DataFrame(continuous_raw, columns=scaler_feature_names)
    else:
        continuous_input = continuous_raw
    continuous_scaled = node_scaler.transform(continuous_input)

    x = continuous_scaled
    if node_mask_cols:
        x = np.concatenate([x, np.asarray(mask_rows, dtype=np.float32)], axis=1)

    extra_dim = int(node_in_dim) - int(x.shape[1])
    if extra_dim < 0:
        raise ValueError(
            "Surrogate node feature width exceeds the trained model input width: "
            f"built={x.shape[1]}, model={node_in_dim}"
        )
    if use_virtual_node:
        if extra_dim == 1:
            # Add virtual-indicator feature column for all physical nodes (always 0.0).
            x = np.concatenate([x, np.zeros((x.shape[0], 1), dtype=np.float32)], axis=1)
        elif extra_dim != 0:
            raise ValueError("Model expects virtual-node mode but node feature dimension has no spare indicator column.")

        virtual = np.zeros((1, node_in_dim), dtype=np.float32)
        virtual[0, -1] = 1.0
        x = np.vstack([x, virtual])
    elif extra_dim != 0:
        raise ValueError(
            "Surrogate node feature width does not match the trained model input width: "
            f"built={x.shape[1]}, model={node_in_dim}"
        )

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
    model_path = _resolve_model_path(prefix_sm)
    node_scaler_path, edge_target_scaler_path, global_feature_scaler_path = _resolve_scaler_paths(artifact_dir, prefix_sm)
    run_manifest_path = artifact_dir / f"{prefix_sm}_run_manifest.json"
    if not run_manifest_path.exists():
        normalized_prefix = _normalize_prefix(prefix_sm)
        if normalized_prefix != prefix_sm:
            fallback_manifest_path = artifact_dir / f"{normalized_prefix}_run_manifest.json"
            if fallback_manifest_path.exists():
                run_manifest_path = fallback_manifest_path
    run_manifest = None
    if run_manifest_path.exists():
        with open(run_manifest_path, "r", encoding="utf-8") as f:
            run_manifest = json.load(f)

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

    # Infer whether the trained head expected global context from predictor width.
    # For these models: predictor_in = hidden_dim * (4 if has_global else 3).
    predictor0_weight = state_dict.get("edge_predictor.0.weight")
    if predictor0_weight is not None and predictor0_weight.ndim == 2 and inferred_hidden_dim is not None:
        predictor_in_features = int(predictor0_weight.shape[1])
        if predictor_in_features == int(inferred_hidden_dim) * 3 and inferred_global_in_dim is None:
            inferred_global_in_dim = 0

    def _coalesce_not_none(*values):
        for value in values:
            if value is not None:
                return value
        return None

    node_in_dim = _coalesce_not_none(
        inferred_node_in_dim,
        (checkpoint.get("node_in_dim") if isinstance(checkpoint, dict) else None),
        int(getattr(node_scaler, "n_features_in_", 3)),
    )
    edge_in_dim = _coalesce_not_none(
        inferred_edge_in_dim,
        (checkpoint.get("edge_in_dim") if isinstance(checkpoint, dict) else None),
        7,
    )
    global_in_dim_manifest = None
    if run_manifest is not None:
        if "global_feature_dim" in run_manifest:
            try:
                global_in_dim_manifest = int(run_manifest.get("global_feature_dim", 0))
            except Exception:
                global_in_dim_manifest = None
        if global_in_dim_manifest is None:
            selected_global_cols = run_manifest.get("selected_global_feature_cols")
            if isinstance(selected_global_cols, (list, tuple)):
                global_in_dim_manifest = int(len(selected_global_cols))
        if global_in_dim_manifest is None and run_manifest.get("use_global_csv") is False:
            global_in_dim_manifest = 0

    global_in_dim = _coalesce_not_none(
        inferred_global_in_dim,
        global_in_dim_manifest,
        (checkpoint.get("global_in_dim") if isinstance(checkpoint, dict) else None),
        3,
    )
    hidden_dim = _coalesce_not_none(
        inferred_hidden_dim,
        (checkpoint.get("hidden_dim") if isinstance(checkpoint, dict) else None),
        128,
    )
    model_variant = checkpoint.get("model_variant", "v1") if isinstance(checkpoint, dict) else "v1"
    dropout_p = checkpoint.get("dropout_p", 0.1) if isinstance(checkpoint, dict) else 0.1
    if run_manifest is not None:
        model_variant = run_manifest.get("model_variant", model_variant)
        dropout_p = run_manifest.get("dropout_p", dropout_p)

    continuous_dim = int(getattr(node_scaler, "n_features_in_", 3))
    use_virtual_node = bool(run_manifest.get("use_virtual_node", False)) if run_manifest is not None else False

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
    x, use_virtual_node = _build_node_features(design_row, bundle)

    node_feature_groups = _extract_node_feature_groups(design_row)
    node_coords = []
    for node_id in sorted(node_feature_groups):
        node_features = node_feature_groups[node_id]
        if not all(axis in node_features for axis in ("x", "y", "z")):
            raise ValueError(f"Incomplete coordinate triplet for node v{node_id}.")
        node_coords.append([node_features["x"], node_features["y"], node_features["z"]])
    node_coords_array = np.asarray(node_coords, dtype=np.float32)

    edge_feature_names = _resolve_edge_feature_layout(bundle)
    edge_rows: list[list[float]] = []
    missing_edge_columns: list[str] = []
    physical_edge_count = bundle["edge_index"].size(1) // 2
    for edge_id in range(physical_edge_count):
        row_values: list[float] = []
        for feature_name in edge_feature_names:
            if feature_name == "Length":
                start_idx = int(bundle["edge_index"][0][edge_id].item())
                end_idx = int(bundle["edge_index"][1][edge_id].item())
                row_values.append(float(np.linalg.norm(node_coords_array[start_idx] - node_coords_array[end_idx])))
                continue

            candidate_column = f"e{edge_id}_{feature_name}"
            if candidate_column not in design_row.index:
                missing_edge_columns.append(candidate_column)
                row_values.append(np.nan)
            else:
                row_values.append(float(design_row[candidate_column]))
        edge_rows.append(row_values)

    if missing_edge_columns:
        missing_display = ", ".join(missing_edge_columns[:12])
        suffix = "..." if len(missing_edge_columns) > 12 else ""
        raise ValueError(
            "Structural surrogate inputs are incomplete for the trained edge feature schema. "
            f"Missing columns: {missing_display}{suffix}"
        )

    edge_index = bundle["edge_index"].clone()
    num_physical_edges = physical_edge_count
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

    edge_attr = np.asarray(edge_rows, dtype=np.float32)
    edge_in_dim = int(bundle.get("edge_in_dim", edge_attr.shape[1]))
    if edge_attr.shape[1] != edge_in_dim:
        raise ValueError(
            "Surrogate edge feature width does not match the trained model input width: "
            f"built={edge_attr.shape[1]}, model={edge_in_dim}"
        )
    edge_attr = np.concatenate([edge_attr, edge_attr], axis=0)
    if use_virtual_node:
        virtual_edge_attr = np.zeros((edge_index.size(1) - edge_attr.shape[0], edge_attr.shape[1]), dtype=np.float32)
        edge_attr = np.vstack([edge_attr, virtual_edge_attr])

    if edge_attr.shape[0] != edge_index.size(1):
        raise ValueError(
            "Surrogate edge feature rows do not match the edge topology: "
            f"features={edge_attr.shape[0]}, edges={edge_index.size(1)}"
        )

    edge_attr = torch.from_numpy(edge_attr.astype(np.float32))
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
    lengths_m = np.linalg.norm(node_coords_array[starts] - node_coords_array[ends], axis=1)

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
