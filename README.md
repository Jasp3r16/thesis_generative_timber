# Deep Generative Design for Reclaimed Timber Structures
**MSc Thesis — Building Technology | TU Delft**

## Project Overview

This research investigates a hybrid computational workflow for integrating **non-standardized reclaimed timber** into **deep generative design** processes. The goal is to bridge the gap between continuous geometric optimization and the discrete, heterogeneous constraints of a finite reclaimed material stock.

A Graph Neural Network (GNN) surrogate replaces computationally expensive Finite Element Analysis (FEA), enabling rapid iterative evaluation of spatial topologies (3D trusses). A downstream assignment pipeline then matches specific reclaimed timber elements to structural slots, navigating the trade-off between resource efficiency (minimizing volumetric waste) and circularity (maximizing avoided burden in kg CO₂ eq).

### Key Research Areas

- **Synthetic Data Generation** — Automated parametric pipelines (Grasshopper + Python) producing large, varied datasets of spatial timber structures.
- **Graph Neural Networks (GNN)** — `TrussEdgeSafetyGNN` trained on topological edge indices to predict per-member structural failure probability under Eurocode load combinations.
- **Discrete Material Matching** — MILP and Genetic Algorithm (GA) solvers assign reclaimed timber elements to structural slots, minimising a compound cost function (sawing waste, over-dimensioning, embodied carbon).

---

## Computational Pipeline

### Phase I — Data Acquisition and Synthetic Generation
1. **Parametric Geometry Generation** — Diverse structural variations (spatial trusses) bounded by a defined multi-dimensional search space, scripted in `01_GH_script/`.
2. **Structural Evaluation (Ground Truth)** — Automated FEA via Karamba3D calculates exact member forces and global stability.
3. **Data Serialisation** — Nodal coordinates, topological graphs, and structural labels exported to `30_Data_Inventory/` (OneDrive).

### Phase II — The Generative Optimisation Loop
4. **Surrogate Model Training** — `TrussEdgeSafetyGNN` (v4) maps spatial topologies to per-edge binary failure probabilities, bypassing the FEA bottleneck. Trained locally or on DelftBlue HPC.
5. **Feasibility Filtering** — `c24_stage_feasibility.py` filters the design space against hard geometric and material constraints.
6. **Cost Matrix** — `c25_stage_cost_matrix.py` computes the assignment penalty matrix for all (element, stock-piece) pairs.
7. **MILP Assignment** — `c26_stage_MILP.py` finds the globally optimal reclaimed timber assignment.
8. **GNN Inference** — `c27_stage_GNN.py` re-evaluates the assigned structure using the surrogate.
9. **Fitness Scoring** — `c28_stage_fitness_score.py` returns the compound objective value to the GA.

### Phase III — Reconstruction and Verification
10. **GA Optimisation** — `c23_ga_algorithm.py` / `c23_ga_evaluator.py` search the design space, calling the Phase II pipeline as an oracle.
11. **Model Reconstruction** — `src/c12_reconstruction.py` translates the optimised parameter vector back into Rhino/Grasshopper geometry.
12. **Final Validation** — Conclusive Karamba3D analysis verifies GNN predictions and produces the final LCA / Bill of Materials.

---

## Repository Structure

```
thesis_generative_timber/
│
├── config.py                          # Path config: local vs. DelftBlue detection
├── config_delftblue.py                # DelftBlue-specific paths
├── c00_headquarter_params.py          # Global design-space parameter definitions
├── c16_params.py                      # Timber stock generation parameters
│
├── c21_surrogate_model_v4.py          # TrussEdgeSafetyGNN model definition
│
├── src/
│   ├── c00_naming.py                  # Shared naming conventions
│   ├── c12_geometry_truss.py          # Truss geometry primitives
│   ├── c12_reconstruction.py          # Parameter-vector → Rhino geometry
│   ├── c16_generation_timber.py       # Reclaimed timber stock generation
│   └── c21_surrogate_io.py            # Surrogate model I/O and inference utilities
│
├── workflows/
│   ├── c21_surrogate_training.py      # Training loop (also used by DelftBlue)
│   ├── c22_stage_geometry.py          # Stage: geometry feature extraction
│   ├── c23_ga_algorithm.py            # GA driver
│   ├── c23_ga_evaluator.py            # GA fitness oracle (calls c24–c28)
│   ├── c23_ga_analysis_export.py      # GA result export utilities
│   ├── c24_stage_feasibility.py       # Stage: hard-constraint filtering
│   ├── c25_stage_cost_matrix.py       # Stage: assignment cost matrix
│   ├── c26_stage_MILP.py              # Stage: MILP timber assignment
│   ├── c27_stage_GNN.py               # Stage: GNN failure-probability inference
│   └── c28_stage_fitness_score.py     # Stage: compound fitness scoring
│       c28_stage_normalization_bounds.py
│
├── 01_GH_script/                      # Grasshopper Python scripts
│   ├── geometry_extraction_v2.py      # Node/edge feature extraction
│   ├── v60_node.py / v60_edge.py      # Latest GH export scripts
│   └── util_violations_writer.py      # Constraint violation logging
│
├── delftblue_scripts/                 # HPC (SLURM) submission scripts
│   ├── README.md                      # Run-order guide
│   ├── DB_c21_slurm_train.py          # Training worker
│   ├── DB_c21_slurm_evaluate.py       # Evaluation worker
│   └── DB_c21_hyperparameter_report.py
│
└── *.ipynb                            # Exploratory / analysis notebooks
    ├── c12_15_main_geometry_generation.ipynb
    ├── c16_timber_stock_dataset_generation.ipynb
    ├── c21_surrogate_model_training_v4.ipynb
    ├── c23_(24-28)_design_space_optimizer_v2.ipynb
    └── c24-c28_tester.ipynb
```

**Data lives on OneDrive** (`30_Data_Inventory/`) and research exports on (`60_Research_Exports/`). `config.py` resolves these paths automatically on any machine.

---

## Model: TrussEdgeSafetyGNN (v4)

The surrogate is defined in `c21_surrogate_model_v4.py` and operates on **39 nodes / 120 physical edges** (bidirectional edge index: 240 directed edges).

| Component | Details |
|---|---|
| Node features | 10D: `x, y, z, Tx, Ty, Tz, Rx, Ry, Rz, Fz` |
| Edge features | 9D: `Width_m, Depth_m, Length, E, Iy, Iz, J, EA/L, N_mean_EA` |
| Architecture | NodeEncoder (2-layer MLP) → NNConv message-passing stack → EdgeDecoder (symmetric interactions: `\|h_i − h_j\|`, `h_i ⊙ h_j`) |
| Loss | Focal Loss (numerically stable) |
| Output | Per-edge binary failure probability `[num_edges, 1]` |

Architecture improvements in v4 over v3: deeper node encoder, residual skip connections from layer 0, symmetric edge embeddings, dropout in processor and decoder, third hidden layer in `EdgeFeatureMLPFilter`.

---

## Quick Start

### Local inference

```python
import config
from src.c21_surrogate_io import load_surrogate_bundle, predict_edge_failure_probabilities

bundle = load_surrogate_bundle()          # resolves latest checkpoint automatically
probs  = predict_edge_failure_probabilities(design_row, bundle)
```

### Training (local)

Open `c21_surrogate_model_training_v4.ipynb` or run:

```python
from workflows.c21_surrogate_training import run_training
run_training(config)
```

### Training (DelftBlue HPC)

See `delftblue_scripts/README.md` for the full run order. Short version:

```bash
# 1. Set up environment (once)
bash delftblue_scripts/01_setup_environment.sh

# 2. Optional CPU smoke test
bash delftblue_scripts/02_submit_cpu_smoke.sh

# 3. Phase 1 hyperparameter sweep (GPU)
bash delftblue_scripts/10_submit_phase1.sh

# 4. Phase 2 sweep (GPU)
bash delftblue_scripts/20_submit_phase2.sh

# 5. Report generation (auto-queued by phase scripts)
bash delftblue_scripts/30_generate_c21_report.sh
```

Monitor with `squeue -u $USER`. Default array concurrency is `2` (education-account safe); override with `ARRAY_MAX_CONCURRENT=1 bash ...`.

---

## Data Note

Timber stock CSVs in `03_timber_data/` may be semicolon-delimited and some older exports lack explicit `f_tk` / `f_c0k` columns. `load_surrogate_bundle()` / `load_and_prepare_stock()` in `src/c21_surrogate_io.py` handle this automatically via delimiter auto-detection and `f_mk`-based backfill for C18/C24 grade datasets.
