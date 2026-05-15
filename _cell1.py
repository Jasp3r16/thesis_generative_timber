# --- Prepare tensors and model
# (loads CSVs, builds per-sample PyG Dataset with correct train/val/test split)
#
# Changes vs v3:
#   10. num_edges_raw derived from CSV row count per sample, not edge_index.json
#       — fixes ValueError when edge_index.json already has 240 edges from a
#       previous bidirectional run.
#   11. BIDIRECTIONAL flag added — set False to use unidirectional graph.
#       Bidirectional edges hurt performance for this dataset (AUC 0.887 uni
#       vs 0.874 bi) because identical duplicate edge features provide no new
#       structural information and inflate NNConv add-aggregation magnitudes.
#       Default is False (unidirectional = best known configuration).

import json
import torch
import config
import pandas as pd
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
from c21_surrogate_model_v4 import create_model, FocalLoss

edge_csv = "v5_edge_C12_S19999_D20260430"
node_csv = "v5_node_C12_S19999_D20260430"

node_csv_path = config.GH_DATA_PATH / f"{node_csv}.csv"
edge_csv_path = config.GH_DATA_PATH / f"{edge_csv}.csv"

nodes_df = pd.read_csv(node_csv_path)
edges_df = pd.read_csv(edge_csv_path)

data_inspection = False
BIDIRECTIONAL   = False   # True = 240 edges, False = 120 edges (best known config)

# ============================================================================
# TOPOLOGY
# ============================================================================

edge_index_path = Path(config.DATA_IO_PATH) / "edge_index.json"
if not edge_index_path.exists():
    raise FileNotFoundError(
        f"edge_index.json not found at {edge_index_path}. Provide a valid topology file."
    )
with open(edge_index_path, "r") as f:
    edge_index_list = json.load(f)
edge_index = torch.tensor(edge_index_list, dtype=torch.long)

if edge_index.ndim != 2 or edge_index.shape[0] != 2:
    raise ValueError(
        f"edge_index must have shape [2, num_edges], got {tuple(edge_index.shape)}"
    )

expected_num_nodes = int(edge_index.max().item()) + 1

# ============================================================================
# SAMPLE ID DETECTION
# Done early so num_edges_raw can be derived from CSV before validation.
# ============================================================================

node_cols = ["x", "y", "z", "Tx", "Ty", "Tz", "Rx", "Ry", "Rz", "Fz"]
edge_cols = ["Area", "Length", "E", "Iy", "Iz", "J", "EA/L"]

sample_id_col = None
for col in ("sample_id", "Sample_ID", "SampleId"):
    if col in nodes_df.columns and col in edges_df.columns:
        sample_id_col = col
        break

if sample_id_col is None:
    raise KeyError(
        "No sample ID column found in CSVs. "
        "Expected one of: 'sample_id', 'Sample_ID', 'SampleId'. "
        "Ensure both node and edge CSVs contain the same sample ID column."
    )

print(f"Sample ID column detected: '{sample_id_col}'")

# num_edges_raw = CSV rows per sample — always the original 120, regardless
# of what edge_index.json currently contains.
num_edges_raw = int(edges_df.groupby(sample_id_col).size().iloc[0])
print(f"Topology: {edge_index.shape[1]} edges in JSON | "
      f"{num_edges_raw} rows per sample in CSV | "
      f"{expected_num_nodes} nodes per sample.")

# ============================================================================
# BIDIRECTIONALITY
# If BIDIRECTIONAL=True and the graph is currently directed, convert and save.
# If BIDIRECTIONAL=False and the graph is currently directed, leave it as-is.
# If BIDIRECTIONAL=False and the graph is already undirected (240 edges),
#   strip reverse edges back to 120 by keeping only edges where src < dst,
#   then save the corrected topology.
# ============================================================================

src, dst  = edge_index[0], edge_index[1]
forward   = set(zip(src.tolist(), dst.tolist()))
backward  = set(zip(dst.tolist(), src.tolist()))
is_undirected = len(forward - backward) == 0

if BIDIRECTIONAL:
    if not is_undirected:
        print(f"Converting directed graph ({edge_index.shape[1]} edges) to undirected.")
        edge_index = to_undirected(edge_index)
        with open(edge_index_path, "w") as f:
            json.dump(edge_index.tolist(), f)
        print(f"Bidirectional edge_index saved: {edge_index.shape[1]} edges.")
    else:
        print(f"Graph already undirected: {edge_index.shape[1]} edges.")
else:
    if is_undirected and edge_index.shape[1] == 2 * num_edges_raw:
        # edge_index.json was previously converted — strip back to directed
        print(f"BIDIRECTIONAL=False but JSON has {edge_index.shape[1]} edges "
              f"(was previously converted). Stripping to {num_edges_raw} directed edges.")
        mask       = edge_index[0] < edge_index[1]
        edge_index = edge_index[:, mask]
        with open(edge_index_path, "w") as f:
            json.dump(edge_index.tolist(), f)
        print(f"Directed edge_index restored and saved: {edge_index.shape[1]} edges.")
    elif not is_undirected:
        print(f"Using directed graph: {edge_index.shape[1]} edges (BIDIRECTIONAL=False).")
    else:
        print(f"Graph undirected with {edge_index.shape[1]} edges but count doesn't "
              f"match 2×{num_edges_raw} — using as-is.")

num_edges = int(edge_index.shape[1])
print(f"Final edge count: {num_edges} ({'bidirectional' if BIDIRECTIONAL else 'unidirectional'}).")

# ============================================================================
# COLUMN VALIDATION
# ============================================================================

missing_node_cols = [c for c in node_cols if c not in nodes_df.columns]
if missing_node_cols:
    raise KeyError(
        f"Missing required node columns: {missing_node_cols}. "
        f"Please provide these columns in {node_csv_path}."
    )

missing_edge_cols = [c for c in edge_cols if c not in edges_df.columns]
if missing_edge_cols:
    raise KeyError(
        f"Missing required edge columns: {missing_edge_cols}. "
        f"Please provide these columns in {edge_csv_path}."
    )

if "Utilization" not in edges_df.columns:
    raise KeyError(
        f"Missing required target column 'Utilization' in {edge_csv_path}."
    )

# ============================================================================
# SAMPLE VALIDATION
# Always compare against num_edges_raw (CSV rows = 120).
# ============================================================================

node_groups = nodes_df.groupby(sample_id_col)
edge_groups = edges_df.groupby(sample_id_col)
samples = sorted(
    set(node_groups.groups.keys()).intersection(edge_groups.groups.keys())
)
if not samples:
    raise ValueError("No matching sample IDs between node and edge CSVs.")

print(f"Found {len(samples)} matching samples.")

for s in samples:
    n_count = len(node_groups.get_group(s))
    e_count = len(edge_groups.get_group(s))
    if n_count != expected_num_nodes:
        raise ValueError(
            f"Sample {s}: node count {n_count} != expected {expected_num_nodes}"
        )
    if e_count != num_edges_raw:
        raise ValueError(
            f"Sample {s}: edge count {e_count} != expected {num_edges_raw} "
            f"(CSV row count per sample)"
        )

# ============================================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================================

torch.manual_seed(42)
shuffled   = torch.randperm(len(samples)).tolist()
train_size = int(0.8 * len(samples))
val_size   = int(0.1 * len(samples))

train_indices = shuffled[:train_size]
val_indices   = shuffled[train_size:train_size + val_size]
test_indices  = shuffled[train_size + val_size:]

train_samples = [samples[i] for i in train_indices]
val_samples   = [samples[i] for i in val_indices]
test_samples  = [samples[i] for i in test_indices]

print(f"Split: Train={len(train_samples)} | Val={len(val_samples)} | Test={len(test_samples)}")

# ============================================================================
# NORMALISATION
# ============================================================================

train_nodes = nodes_df[nodes_df[sample_id_col].isin(train_samples)]
train_edges = edges_df[edges_df[sample_id_col].isin(train_samples)]

node_feature_means = train_nodes[node_cols].mean()
node_feature_stds  = train_nodes[node_cols].std(ddof=0).replace(0, 1.0)
edge_feature_means = train_edges[edge_cols].mean()
edge_feature_stds  = train_edges[edge_cols].std(ddof=0).replace(0, 1.0)

print("Normalisation statistics computed from training data only (z-score, clipped to ±5 sigma).")

norm_stats = {
    "node_means": node_feature_means.to_dict(),
    "node_stds":  node_feature_stds.to_dict(),
    "edge_means": edge_feature_means.to_dict(),
    "edge_stds":  edge_feature_stds.to_dict(),
    "node_cols":  node_cols,
    "edge_cols":  edge_cols,
}
norm_stats_path = Path(config.DATA_IO_PATH) / "norm_stats.pt"
torch.save(norm_stats, norm_stats_path)
print(f"Normalisation stats saved to {norm_stats_path}")

# ============================================================================
# CLASS BALANCE
# ============================================================================

train_pos_rate = float((train_edges["Utilization"] > 1).mean())
focal_alpha    = float(max(0.05, min(0.95, 1.0 - train_pos_rate)))
print(f"Train positive rate (Utilization>1): {train_pos_rate:.4f} -> focal_alpha={focal_alpha:.4f}")

# ============================================================================
# DATASET CONSTRUCTION
# ============================================================================

def build_sample(s, node_groups, edge_groups):
    """
    Build a single PyG Data object for sample s.
    edge_attr and y are duplicated for reverse edges only if BIDIRECTIONAL=True.
    """
    n_df = node_groups.get_group(s)
    e_df = edge_groups.get_group(s)

    x = torch.tensor(
        ((n_df[node_cols] - node_feature_means) / node_feature_stds)
        .clip(-5, 5).values,
        dtype=torch.float32,
    )

    edge_attr_norm = torch.tensor(
        ((e_df[edge_cols] - edge_feature_means) / edge_feature_stds)
        .clip(-5, 5).values,
        dtype=torch.float32,
    )

    y_vals = torch.tensor(
        (e_df["Utilization"] > 1).astype(int).values,
        dtype=torch.float32,
    )

    if BIDIRECTIONAL and num_edges == 2 * num_edges_raw:
        edge_attr_norm = torch.cat([edge_attr_norm, edge_attr_norm], dim=0)
        y_vals         = torch.cat([y_vals, y_vals], dim=0)

    return Data(
        x=x,
        edge_index=edge_index.clone(),
        edge_attr=edge_attr_norm,
        y=y_vals.view(-1, 1),
    )


train_dataset = [build_sample(s, node_groups, edge_groups) for s in train_samples]
val_dataset   = [build_sample(s, node_groups, edge_groups) for s in val_samples]
test_dataset  = [build_sample(s, node_groups, edge_groups) for s in test_samples]

print(
    f"Dataset constructed: {len(train_dataset)} train | "
    f"{len(val_dataset)} val | {len(test_dataset)} test samples. "
    f"Each sample: {expected_num_nodes} nodes, {num_edges} edges "
    f"({'bidirectional' if BIDIRECTIONAL else 'unidirectional'})."
)

# ============================================================================
# MODEL
# ============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = create_model(
    node_features_dim=len(node_cols),
    edge_features_dim=len(edge_cols),
    device=device,
)
model.to(device)
print(
    f"Model on {device} | "
    f"node_features_dim={len(node_cols)} | edge_features_dim={len(edge_cols)}"
)

# ============================================================================
# DATALOADERS
# ============================================================================

batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

print(
    f"DataLoaders ready — "
    f"Train: {len(train_dataloader)} batches | "
    f"Val: {len(val_dataloader)} batches | "
    f"Test: {len(test_dataloader)} batches "
    f"(batch_size={batch_size})"
)

# ============================================================================
# LOSS
# ============================================================================

loss_fn = FocalLoss(alpha=focal_alpha, gamma=2.0)
print(f"FocalLoss(alpha={focal_alpha:.4f}, gamma=2.0)")

# ============================================================================
# DATA INSPECTION (optional)
# ============================================================================

if data_inspection:
    unsafe_per_sample = edges_df.groupby(sample_id_col).apply(
        lambda g: (g["Utilization"] > 1).sum()
    )
    print(unsafe_per_sample.describe())
    print(unsafe_per_sample.value_counts().sort_index().head(20))

    zero_unsafe = edges_df.groupby(sample_id_col).apply(
        lambda g: (g["Utilization"] > 1).sum() == 0
    )
    zero_ids = zero_unsafe[zero_unsafe].index
    print(edges_df[edges_df[sample_id_col].isin(zero_ids)][edge_cols].describe())

# ============================================================================
# READY
#   model, loss_fn, train_dataloader, val_dataloader, test_dataloader, device
#   num_edges_raw  (120 — physical members, for per-member analysis)
#   num_edges      (120 unidirectional or 240 bidirectional)
#   BIDIRECTIONAL  (flag — read by evaluation script)
#
# evaluation: NUM_EDGES = num_edges, NUM_EDGES_PHYSICAL = num_edges_raw
# ============================================================================