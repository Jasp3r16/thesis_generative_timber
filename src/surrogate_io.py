import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

import config
from c21_surrogate_model import TrussEdgeGNN


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

    if prefix_sm is None:
        prefix_path = config.SM_EXPORT_PATH / "prefix_sm.txt"
        if prefix_path.exists():
            prefix_sm = prefix_path.read_text(encoding="utf-8").strip()
        else:
            checkpoint_candidates = sorted(
                config.SM_EXPORT_PATH.glob("*_surrogate_model.pt"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if not checkpoint_candidates:
                raise FileNotFoundError("No surrogate model checkpoint found in SM_EXPORT_PATH.")
            prefix_sm = checkpoint_candidates[0].stem.removesuffix("_surrogate_model")

    model_path = config.SM_EXPORT_PATH / f"{prefix_sm}_surrogate_model.pt"
    node_scaler_path = config.SM_EXPORT_PATH / f"{prefix_sm}_node_scaler.pkl"
    edge_scaler_path = config.SM_EXPORT_PATH / f"{prefix_sm}_edge_scaler.pkl"

    if not model_path.exists():
        legacy_model_path = config.SM_EXPORT_PATH / f"truss_edge_gnn_{prefix_sm}.pt"
        legacy_node_scaler_path = config.SM_EXPORT_PATH / f"node_scaler_{prefix_sm}.pkl"
        legacy_edge_scaler_path = config.SM_EXPORT_PATH / f"edge_scaler_{prefix_sm}.pkl"
        if legacy_model_path.exists():
            model_path = legacy_model_path
            node_scaler_path = legacy_node_scaler_path
            edge_scaler_path = legacy_edge_scaler_path

    checkpoint = torch.load(model_path, map_location=device_obj)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    node_in_dim = checkpoint.get("node_in_dim", 3) if isinstance(checkpoint, dict) else 3
    hidden_dim = checkpoint.get("hidden_dim", 128) if isinstance(checkpoint, dict) else 128
    model = TrussEdgeGNN(node_in_dim=node_in_dim, hidden_dim=hidden_dim).to(device_obj)
    model.load_state_dict(state_dict)
    model.eval()

    node_scaler = joblib.load(node_scaler_path)
    edge_scaler = joblib.load(edge_scaler_path)
    edge_index = load_edge_index(config.DATA_IO_PATH / "edge_index.json")

    print(f"Loaded surrogate prefix: {prefix_sm}")
    print(f"Device: {device_obj}")
    print(f"Model: {model_path.name}")

    return {
        "prefix_sm": prefix_sm,
        "device": device_obj,
        "model": model,
        "node_scaler": node_scaler,
        "edge_scaler": edge_scaler,
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
    node_cols = [f"v{i}_{axis}" for i in range(13) for axis in ("x", "y", "z")]
    missing = [c for c in node_cols if c not in design_row.index]
    if missing:
        raise ValueError(f"Design row misses required columns: {missing[:5]}...")

    node_raw = design_row[node_cols].to_numpy(dtype=np.float32).reshape(13, 3)
    node_scaled = bundle["node_scaler"].transform(node_raw)

    x = torch.from_numpy(node_scaled).to(bundle["device"])
    edge_index = bundle["edge_index"].to(bundle["device"])

    with torch.no_grad():
        pred_scaled = bundle["model"](x, edge_index).detach().cpu().numpy()

    pred_kn = bundle["edge_scaler"].inverse_transform(pred_scaled).reshape(-1)

    starts = edge_index[0].detach().cpu().numpy()
    ends = edge_index[1].detach().cpu().numpy()
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
