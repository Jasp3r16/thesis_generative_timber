# Deep Generative Design for Reclaimed Timber Structures
**MSc Thesis - Building Technology | TU Delft**

## Project Overview
This research investigates a hybrid computational workflow for the integration of **non-standardized reclaimed timber** into **deep generative design** processes. The objective is to bridge the gap between continuous geometric optimization and the discrete, heterogeneous constraints of a finite reclaimed material stock. 

By replacing computationally heavy Finite Element Analysis (FEA) with a high-speed surrogate model, this workflow enables the rapid, iterative evaluation of spatial topologies (such as 3D trusses and reciprocal frames). Simultaneously, an assignment algorithm matches specific reclaimed timber elements to the structure, navigating the complex trade-off between resource efficiency (minimizing volumetric waste) and circularity (maximizing 'avoided burden' in terms of $kg\ CO_2\ eq$).

### Key Research Focus:
* **Synthetic Data Generation:** Building automated parametric pipelines (Grasshopper/Python) to generate large, varied datasets of spatial timber structures.
* **Graph Neural Networks (GNN):** Training surrogate models on topological edge indices to instantaneously predict local structural performance (e.g., element utilization and maximum deflection).
* **Discrete Material Matching:** Utilizing Integer Linear Programming (ILP) and Genetic Algorithms (GA) to assign discrete reclaimed elements to structural slots, minimizing a Compound Cost Function (sawing waste, over-dimensioning, and embodied carbon).

---

## The Computational Pipeline
The workflow is divided into three primary phases:

### Phase I: Data Acquisition and Synthetic Generation
1. **Parametric Geometry Generation:** Generating diverse structural variations (spatial trusses) bounded by a defined multi-dimensional search space.
2. **Structural Evaluation (Ground Truth):** Running automated FEA (Karamba3D) via an in-memory Python integration to calculate exact member forces and global stability.
3. **Data Serialization:** Exporting flattened nodal coordinates, topological graphs (`edge_index.json`), and structural labels (Multi-Output utilization scores) into machine-learning-ready datasets.

### Phase II: The Generative Optimization Loop
4. **Surrogate Model Training:** Training a GNN to map spatial topologies to local member utilization, bypassing the $O(N)$ computational bottleneck of traditional FEA.
5. **The Assignment Matrix:** Calculating the environmental and geometric penalties of placing specific reclaimed timber stock into the generated geometry (filtering out Hard Constraints like insufficient length or structural failure).
6. **Multi-Objective Optimization (MOO):** Adjusting vertex positions to find the "Least-Carbon Path" by actively minimizing the combined system cost using gradient-based or evolutionary solvers.

### Phase III: Reconstruction and Verification
7. **Model Reconstruction:** Translating the optimized parameter vector ($U, V$ coordinates and $Z$-shifts) back into explicit Rhino/Grasshopper geometry.
8. **Final Validation:** Running a conclusive Karamba3D analysis to verify the GNN predictions and outputting the final Life Cycle Assessment (LCA) and Bill of Materials (BOM).

---

## Code Organization (Reusable Modules)

To reduce code duplication across notebooks, the surrogate model and inference utilities are now centralized in reusable Python modules.

### Shared Surrogate Model Class
- `c21_surrogate_model.py`
- Contains `TrussEdgeGNN`.
- Used by training notebooks (e.g., `c21_surrogate_model_training.ipynb`) and inference notebooks.

### Shared Surrogate I/O + Inference Utilities
- `src/surrogate_io.py`
- `load_edge_index(...)`: loads `edge_index.json` to a tensor with shape `[2, num_edges]`.
- `load_surrogate_bundle(...)`: loads model checkpoint, scalers, and edge topology.
- `predict_edge_forces_kn(...)`: predicts per-edge axial forces (kN) for one design row.
- `load_and_prepare_stock(...)`: robust stock CSV loader with delimiter auto-detection and strength-column backfill for older exports.

### Structural Utilization Checks
- `src/structural_check.py`
- `compute_utilization_outputs(...)` calculates Eurocode-based utilization tables and matrices.

### Project Tree (Core Files)

```text
thesis_generative_timber/
  c21_surrogate_model.py
  c21_surrogate_model_training.ipynb
  c25_structural_check.ipynb
  src/
    surrogate_io.py
    structural_check.py
```

---

## Quick Usage

### 1) In Training Notebook (c21)

```python
from c21_surrogate_model import TrussEdgeGNN

model = TrussEdgeGNN(node_in_dim=3, hidden_dim=128).to(device)
```

### 2) In Structural Check / Inference Notebook (c25)

```python
from src.surrogate_io import (
    load_surrogate_bundle,
    predict_edge_forces_kn,
    load_and_prepare_stock,
)
from src.structural_check import compute_utilization_outputs

bundle = load_surrogate_bundle(prefix_sm=None)
df_forces = predict_edge_forces_kn(design_row, bundle)
df_input_stock = load_and_prepare_stock(stock_csv_path)

outputs = compute_utilization_outputs(
    df_forces=df_forces[["edge_id", "length_m", "axial_force_kn"]],
    df_input_stock=df_input_stock,
    gnn_marge=1.10,
)
```

---

## Data Note (Timber Inventory CSV)

Some generated stock files are semicolon-delimited and may contain `f_mk` without explicit `f_tk` / `f_c0k` columns. The helper `load_and_prepare_stock(...)` handles this by:
- auto-detecting delimiter,
- validating required base columns,
- deriving `f_tk` and `f_c0k` from `f_mk` for C18/C24 datasets.
