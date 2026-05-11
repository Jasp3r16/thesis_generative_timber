# c25_stage_feasibility: Surrogate-Based Feasibility Assessment

## Overview

This module implements the feasibility stage of the timber optimization workflow. It assesses which stock items are feasible for each structural member (edge) by:

1. **Running surrogate model inference** on the full structure with each stock item's properties
2. **Predicting binary safety** (safe/unsafe) for each member using the trained GNN surrogate
3. **Combining with length constraints** to create a complete feasibility assessment
4. **Returning a slot×stock matrix** where infeasible combinations are marked as `inf`

## Function: `run_feasibility_stage()`

### Signature

```python
def run_feasibility_stage(
    df_input_stock: pd.DataFrame,
    df_vertices: pd.DataFrame,
    df_edges: pd.DataFrame | None = None,
    model_prefix: str | None = None,
    roof_load_kn_m2: float = 2.0,
) -> pd.DataFrame:
```

### Input Requirements

#### `df_input_stock` (Required)
Stock item catalogue with columns:
- `Member_ID` (str): Unique stock identifier
- `Length` (float): Member length in millimeters
- `Depth` (float): Cross-section depth in millimeters
- `Width` (float): Cross-section width in millimeters
- `f_c0k` (float): Characteristic compression strength (MPa)
- `f_tk` (float): Characteristic tension strength (MPa)
- `E_modulus_eff` (float): Effective elastic modulus (MPa)

Example:
```python
df_stock = pd.DataFrame({
    'Member_ID': ['S1', 'S2', 'S3'],
    'Length': [5000.0, 6000.0, 7000.0],
    'Depth': [200.0, 200.0, 250.0],
    'Width': [150.0, 150.0, 150.0],
    'f_c0k': [30.0, 30.0, 30.0],
    'f_tk': [0.0, 0.0, 0.0],
    'E_modulus_eff': [12000.0, 12000.0, 12000.0],
})
```

#### `df_vertices` (Required)
Node/vertex data with columns:
- `vertex_index` or `node_id` (int): Unique node identifier
- `x`, `y`, `z` (float): Spatial coordinates in meters
- `attribute` (str): Node type marker, typically one of: `'support'`, `'load'`, `'hinge'`
- `layer` (str, optional): Layer classification (e.g., `'top'`, `'bottom'`) for roof load distribution
- `Fz` (float, optional): Nodal vertical load in kN (computed from roof load if absent)

Note: Support vertices (identified by `attribute` containing `'support'`) will have all DOF fixed (Tx=Ty=Tz=Rx=Ry=Rz=1). Other vertices are assumed free.

Example:
```python
df_vertices = pd.DataFrame({
    'vertex_index': [0, 1, 2, 3, ...],  # Must have at least 39 nodes for default model
    'x': [0.0, 5.0, ...],
    'y': [0.0, 0.0, ...],
    'z': [0.0, 0.0, ...],
    'attribute': ['support', 'support', 'load', 'load', ...],
    'layer': ['bottom', 'bottom', 'top', 'top', ...],
})
```

#### `df_edges` (Required)
Edge/member data with columns:
- `edge_id`, `Edge_ID`, or `Element_ID` (str/int): Unique edge identifier
- `Length` (float): Member length in millimeters

**CRITICAL:** The edge count must exactly match the surrogate model's topology. For the default model `ID20260510_224228_LR3e-04_EP150_BS32_FA0.50_ROC0.874`:
- **Required: 240 edges** (bidirectional graph representation)
- **Required: 39 nodes** (matching edge_index topology)

Example:
```python
df_edges = pd.DataFrame({
    'edge_id': ['E1', 'E2', ...],  # 240 edges
    'Length': [5000.0, 4500.0, ...],
})
```

#### `model_prefix` (Optional)
Surrogate model identifier. Default: `"ID20260510_224228_LR3e-04_EP150_BS32_FA0.50_ROC0.874"`

All available models are in `60_Research_Exports/01_surrogate_models/`.

#### `roof_load_kn_m2` (Optional)
Distributed roof load in kN/m². Default: `2.0`

Used to compute nodal Fz loads for top-layer vertices via tributary area triangulation.

### Output

Returns: `pd.DataFrame`

**Shape:** `(num_edges, num_stock_items)`

- **Index:** Edge identifiers from `df_edges`
- **Columns:** Stock Member_ID values
- **Values:**
  - `0.0`: Feasible (safe + length constraint satisfied)
  - `inf`: Infeasible (unsafe OR length too short)

Example:
```
          S1    S2    S3
E1      0.0   inf   0.0
E2      0.0   0.0   inf
E3      inf   0.0   0.0
...
```

### Errors & Validation

The function validates inputs and will raise descriptive errors for:

- **Missing required columns** in input dataframes
- **Topology mismatch:** Edge count doesn't match surrogate's trained topology
- **Node/edge mismatch:** Graph structure incompatible with topology
- **Feature unavailability:** Missing required node/edge features (x, y, z, Tx-Rz, Fz)
- **Support identification failure:** `attribute` column missing or invalid
- **Roof load calculation:** Fewer than 3 top-layer vertices for triangulation

## Implementation Details

### Binary Safety Output

The surrogate model predicts a **binary safety classification** (not a continuous utilization score):
- `0 = Safe` (no predicted failure)
- `1 = Unsafe` (predicted failure)

The safety threshold is fixed at **0.5** (configurable per model in the artifact metadata).

### Node Feature Synthesis

The stage automatically prepares all required node features:

| Feature | Source | Default | Support Behavior |
|---------|--------|---------|------------------|
| `x, y, z` | From input `df_vertices` | Required | Used as-is |
| `Tx, Ty, Tz` | Derived from `attribute='support'` | 0 (free) | 1 (fixed) if support |
| `Rx, Ry, Rz` | Derived from `attribute='support'` | 0 (free) | 1 (fixed) if support |
| `Fz` | From `layer`-based roof load or input | Computed | Roof load → tributary area → nodal forces |

### Edge Feature Synthesis

Edge features are computed from **stock item properties** and **geometry**:

| Feature | Computation | Units |
|---------|-----------|-------|
| `Area` | Width × Depth | m² |
| `Length` | From `df_edges` or `df_stock` | m |
| `E` | From `E_modulus_eff` in stock | MPa |
| `Iy`, `Iz`, `J` | From cross-section (rectangular) | m⁴ |
| `EA/L` | (E × Area) / Length | MPa |

### Full-Graph Inference Loop

For each stock item:
1. Build ALL 240 edge features using that stock's properties
2. Run the full graph through the surrogate with fixed node features
3. Extract per-edge failure probabilities
4. Apply safety threshold (≥0.5 = unsafe)
5. Combine with length constraint check
6. Mark `inf` for any infeasible combination

This approach respects the GNN's trained topology and context-awareness across the structure.

## Usage Example

```python
from workflows.c25_stage_feasibility import run_feasibility_stage
import pandas as pd

# Load or prepare your data
df_stock = pd.read_csv("stock_catalog.csv")
df_vertices = pd.read_csv("structure_vertices.csv")
df_edges = pd.read_csv("structure_edges.csv")

# Run feasibility assessment
feasibility_matrix = run_feasibility_stage(
    df_input_stock=df_stock,
    df_vertices=df_vertices,
    df_edges=df_edges,
    model_prefix="ID20260510_224228_LR3e-04_EP150_BS32_FA0.50_ROC0.874",
    roof_load_kn_m2=2.0,
)

# Filter feasible combinations
feasible_only = feasibility_matrix[feasibility_matrix != np.inf]
print(f"Feasible combinations: {feasible_only.count().sum()}")

# Pass to next stage (cost matrix)
# cost_matrix = c26_cost_calculation.run_cost_stage(feasibility_matrix, ...)
```

## Data Format Requirements

### Topology Compatibility

The default surrogate model (`ID20260510_224228...`) was trained on:
- **39 nodes** with specific connectivity
- **240 directed edges** representing full bidirectional relationships
- **Node features:** x, y, z, Tx, Ty, Tz, Rx, Ry, Rz, Fz (10 dims)
- **Edge features:** Area, Length, E, Iy, Iz, J, EA/L (7 dims)

### Column Naming Flexibility

The stage uses flexible column resolution:
- Edge ID: Tries `edge_id`, `Edge_ID`, `Element_ID` in order
- Edge length: Tries `Length`, `length_m`, `length`, `Length_m`
- Edge area: Tries `Area`, `area`, `cross_section_area`, `A`

### Support Identification

The `attribute` column should use lowercase values:
- `'support'` → Fixed DOF (Tx=Ty=Tz=Rx=Ry=Rz=1)
- Other values → Free DOF (Tx=Ty=Tz=Rx=Ry=Rz=0)

Use `.str.lower()` if your data has mixed case.

## Performance Notes

- **Runtime:** ~2-5 minutes per stock item (depends on GPU availability)
- **Memory:** ~500 MB for full inference
- **Bottleneck:** Surrogate inference (runs on GPU if available)

## Next Steps

The feasibility matrix output is typically fed to the cost-matrix stage (`c26_cost_calculation`) which computes material and cutting costs for feasible combinations. Infeasible combinations (marked as `inf`) are automatically skipped during optimization.

## References

- Surrogate model training: `c21_surrogate_model_v4.py`
- Cost matrix calculation: `c26_cost_calculation.py`
- Timber properties: `c22_params.py`
- Configuration: `config.py`
