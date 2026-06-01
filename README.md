# Deep Generative Design for Reclaimed Timber Structures
**MSc Thesis - Building Technology | TU Delft**

---

## Overview

This research develops a hybrid computational workflow for integrating **reclaimed timber** into **generative structural design**. The pipeline co-optimises the three-dimensional geometry of a space frame and the assignment of specific salvaged elements to structural slots, minimising embodied carbon while maximising material reuse.

The core challenge is bridging continuous geometric optimisation - which requires thousands of fast fitness evaluations - and discrete material matching against a finite, heterogeneous reclaimed stock. The solution is a three-component architecture:

- **CMA-ES** searches the continuous geometry parameter space (node positions)
- **MILP** finds the globally optimal stock-to-slot assignment for each candidate geometry
- **TrussEdgeSafetyGNN** replaces computationally expensive Karamba3D FEA with a millisecond-scale structural proxy during the search loop

Final designs from the top-k shortlist are verified with full Karamba3D FEA and exported as a complete fabrication package (bill of materials, node coordinates, cross-section schedule).

---

## Computational Pipeline

### Phase I - Data generation

1. **Parametric geometry**: Diverse spatial truss variations generated within a bounded 73-dimensional search space, scripted in `01_GH_script/` and driven by Grasshopper + Karamba3D.
2. **Structural ground truth**: Automated FEA computes member forces and utilisation ratios (UC) per Eurocode 5. Labels: UC > 1.0 = unsafe.
3. **Data serialisation**: Node coordinates, graph topology, cross-section features, and binary safety labels exported to `30_Data_Inventory/` (OneDrive). Training set: 20,000 labelled configurations (16k train / 2k val / 2k test).

### Phase II - Surrogate training

1. **GNN training**: `TrussEdgeSafetyGNN` (v4) trained to predict per-member failure probability from node coordinates, boundary conditions, and cross-section edge features. See model details below.

### Phase III - Generative optimisation loop

1. **Feasibility filter** (`c24_stage_feasibility.py`) - Applies hard geometric and EC5 force constraints to eliminate infeasible slot/stock pairs before assignment.
2. **LCA cost matrix** (`c25_stage_cost_matrix.py`) - Computes embodied carbon cost for all feasible (slot, stock) pairs, accounting for A1–A3 embodied carbon, A4 transport, A5 preparation/sawing, and C2/C3–C4 waste costs.
3. **MILP assignment** (`c26_stage_MILP.py`) - Finds the globally optimal stock assignment minimising total LCA cost subject to uniqueness, coverage, new-use cap, and optional reuse floor constraints.
4. **GNN inference** (`c27_stage_GNN.py`) - Re-evaluates the assigned structure; returns per-member failure probabilities and a scalar feasibility score.
5. **Fitness scoring** (`c28_stage_fitness_score.py`) - Computes the compound objective F = ω₁·Ĉ − ω₂·R̂ + ω₄·S (minimised), where Ĉ is normalised LCA cost, R̂ is normalised reuse fraction, and S is structural infeasibility. ω₄ is annealed 2.0 → 0.8 over 250 generations.

The CMA-ES driver (`c23_ga_algorithm.py` / `c23_ga_evaluator.py`) calls steps 1–5 above as a black-box oracle on every candidate. Budget: **250 generations × 30 population = 7,500 evaluations per run**.

### Phase IV - Verification and export

1. **Top-k reconstruction**: The best-ranked designs are translated back to 3D geometry via `src/c12_reconstruction.py`.
2. **Final FEA validation**: Full Karamba3D analysis on the top-k shortlist verifies GNN predictions and flags any residual utilisation violations.
3. **Export**: Bill of materials (`_bom.csv`), node coordinates (`_vertices.csv`), cross-section schedule (`_crosssections.csv`), and assignment table (`_edges.csv`) written to `60_Research_Exports/`.

---

## Repository Structure

```
thesis_generative_timber/
│
├── config.py                              # Path resolution: local / OneDrive auto-detection
├── c00_headquarter_params.py              # Global LCA constants and design-space parameters
├── c16_params.py                          # Timber stock generation parameters
├── c21_surrogate_model_v4.py              # TrussEdgeSafetyGNN model definition
├── c23_run_ga_batch.py                    # Batch runner for multiple GA experiments
├── requirements.txt / requirements-lock.txt
│
├── 02_data_io/                            # Fixed reference data
│   ├── search_space_5x3.json              # CMA-ES parameter bounds (node position ranges)
│   ├── edge_index_5x3.json                # Fixed graph topology in COO format
│   ├── df_edges.csv / df_vertices.csv     # Reference geometry for the 5×3 truss
│   ├── norm_stats.pt                      # Feature normalisation statistics
│   └── representative_beam_statistics.json
│
├── src/
│   ├── c00_naming.py                      # Shared naming conventions
│   ├── c12_geometry_truss.py              # Truss geometry primitives
│   ├── c12_reconstruction.py              # Parameter vector → 3D geometry
│   ├── c16_generation_timber.py           # Synthetic timber stock generation
│   └── c21_surrogate_io.py                # Surrogate model I/O and inference utilities
│
├── workflows/
│   ├── c21_surrogate_training.py          # GNN training loop
│   ├── c22_stage_geometry.py              # Stage 1: geometry feature extraction
│   ├── c23_ga_algorithm.py                # CMA-ES driver
│   ├── c23_ga_evaluator.py                # Fitness oracle - calls stages 5–9
│   ├── c23_ga_analysis_export.py          # GA result export utilities
│   ├── c24_stage_feasibility.py           # Stage 2: hard-constraint filtering
│   ├── c25_stage_cost_matrix.py           # Stage 3: LCA assignment cost matrix
│   ├── c26_stage_MILP.py                  # Stage 4: MILP timber assignment
│   ├── c27_stage_GNN.py                   # Stage 5: GNN structural inference
│   ├── c28_stage_fitness_score.py         # Stage 6: compound fitness scoring
│   └── c28_stage_normalization_bounds.py  # One-time normalisation bounds derivation
│
├── figures/                               # Thesis figure scripts and exports
│   ├── fig_*.py                           # Python scripts generating all thesis figures
│   ├── pdf/                               # PDF exports
│   └── png/                               # PNG exports
│
├── 01_GH_script/                          # Grasshopper Python scripts
│   ├── v60_node.py / v60_edge.py          # Latest GH export scripts (node + edge features)
│   ├── geometry_extraction_v2.py          # Node/edge feature extraction
│   ├── ga_artifact_checker.py             # GA run artifact validation
│   └── util_violations_writer.py          # Constraint violation logging
│
├── appendix_code_highlights.py            # Thesis appendix: four key functions
│
└── *.ipynb
    ├── c12_15_main_geometry_generation.ipynb
    ├── c16_timber_stock_dataset_generation.ipynb
    ├── c21_surrogate_model_training_v4.ipynb
    ├── c23_(24-28)_design_space_optimizer_v2.ipynb
    ├── c24-c28_tester.ipynb
    └── c30_final_batch_analysis.ipynb     # Final batch results (thesis ch. 6)
```

**Data lives on OneDrive**: `30_Data_Inventory/` (stock, FEA datasets, model checkpoints) and `60_Research_Exports/` (GA outputs, figures, BOM exports). `config.py` resolves these paths automatically on any machine by detecting the local OneDrive root.

---

## Model: TrussEdgeSafetyGNN (v4)

Defined in `c21_surrogate_model_v4.py`. Operates on the fixed **5×3 Pratt truss**: 39 nodes, 120 physical edges (240 directed edges in COO format).

| Component | Details |
| --- | --- |
| Node features | 10D: `x, y, z` + boundary condition flags `Tx, Ty, Tz, Rx, Ry, Rz` + applied load `Fz` |
| Edge features | 9D: `Width_m, Depth_m, Length, E, Iy, Iz, J, EA/L, N_mean_EA` |
| Architecture | `NodeEncoder` (2-layer MLP) → 4× `NNConv` with adaptive edge weights + residuals + dropout → `EdgeDecoder` (symmetric: `\|h_i − h_j\|`, `h_i ⊙ h_j`) |
| Hidden dim | 64 |
| Layers | 4 NNConv layers |
| Dropout | 0.3 (processor + decoder) |
| Loss | Weighted BCE (`pos_weight=2.5`) |
| Output | Per-edge P(unsafe) = P(UC > 1.0) in [0, 1] |

**Trained model** (`ID20260516_182257_LR1e-04_EP200_BS64_PW2.5_ROC0.863`):

| Metric | Value |
| --- | --- |
| ROC-AUC | 0.863 |
| PR-AUC | 0.615 |
| Classification threshold θ | 0.30 (tuned for recall on validation set) |
| Recall (at θ=0.30) | 86.6% |
| False negative rate | 11.4% |
| Precision | 41.7% |
| Training set | 16,000 Karamba3D-labelled configurations |

Architecture improvements in v4 over v3: deeper node encoder (2-layer vs. bare Linear), residual skip connections applied from layer 0, symmetric edge embeddings in decoder (direction-invariant), dropout added to processor and decoder, third hidden layer in `EdgeFeatureMLPFilter`.

---

## GA Experiment Configuration

Four stock conditions were tested across 21 total runs (3 seeds × 3 conditions + 12 additional):

| Condition | Stock | RS pieces | NS pieces | new_stock_max_uses |
| --- | --- | --- | --- | --- |
| Stock A | 524 elements | 103 RS | 421 NS | unlimited |
| Stock B | 576 elements | 155 RS | 421 NS | unlimited |
| New max=10 | 421 elements | 0 RS | 421 NS | 10 |
| New max=120 | 421 elements | 0 RS | 421 NS | 120 |

**CMA-ES settings**: σ_init=0.25, σ_min=1e-8, tolfun=1e-11, μ=15, λ=30, 250 generations (7,500 evaluations/run). Stagnation limit: 30 generations. Best fitness logged with `c30_final_batch_analysis.ipynb`.

---

## Quick Start

### Requirements

Python 3.14, PyTorch 2.11, PyTorch Geometric 2.7.

```bash
pip install -r requirements.txt
```

### GNN inference on a single design

```python
import config
from src.c21_surrogate_io import load_surrogate_bundle, predict_edge_failure_probabilities

bundle = load_surrogate_bundle()   # resolves latest checkpoint from 30_Data_Inventory/
probs  = predict_edge_failure_probabilities(design_row, bundle)
# probs: np.ndarray [120,] - P(unsafe) per member
```

### Run a single GA optimisation

```python
from workflows.c23_ga_algorithm import run_ga
from workflows.c23_ga_evaluator import evaluate_design_candidate
import config

result = run_ga(config=config, stock_condition="A")
```

### Run the full final batch

```bash
python c23_run_ga_batch.py
```

Results are written to `60_Research_Exports/03_ga_data/` with one subdirectory per run, each containing `_run_config.json`, `_history.csv`, `_best_design.json`, `_bom.csv`, `_vertices.csv`, `_edges.csv`, and `_crosssections.csv`.

### Analyse results

Open `c30_final_batch_analysis.ipynb`. Set `BATCH_DIR` to the run output directory and execute all cells. Figures are saved to `60_Research_Exports/03_ga_data/c30_output/`.

### Train the GNN

Open `c21_surrogate_model_training_v4.ipynb` or run:

```python
from workflows.c21_surrogate_training import run_training
run_training(config)
```

Training data must be present in `30_Data_Inventory/02_GNN_data/`. Checkpoints are saved to `30_Data_Inventory/04_models/`.

---

## LCA Cost Model

Defined in `workflows/c25_stage_cost_matrix.py`. All constants are set in `c00_headquarter_params.py`.

| Constant | Value | Module | Description |
| --- | --- | --- | --- |
| `IMPACT_FACTOR_A1_A3` | 0.25 kg CO₂e/kg | A1–A3 | Embodied carbon, new softwood |
| `IMPACT_FACTOR_RECOVERED_C1` | 0.0085 kg CO₂e/kg | C1 | Deconstruction energy, reclaimed |
| `ENERGY_PREP_A5` | 0.010 kg CO₂e/kg | A5 | Preparation (cleaning, de-nailing) |
| `ENERGY_SAW_A5` | 0.004 kg CO₂e/kg | A5 | Cross-cut sawing (if needed) |
| `ENERGY_OFFCUT_FACTOR_C3_C4` | 0.031 kg CO₂e/kg | C3–C4 | Offcut disposal/incineration |
| `WASTE_TRANSPORT_DIST_KM` | 50 km | C2 | Offcut transport to waste facility |
| `SCARCITY_PENALTY` | 0.0 | - | Disabled in all reported runs |

Transport for new stock is calculated on required mass only; transport for reclaimed stock is calculated on full physical element mass.

---

## Data Notes

- **Timber stock CSVs** in `03_timber_data/` may be semicolon-delimited. `load_and_prepare_stock()` in `src/c21_surrogate_io.py` handles delimiter auto-detection automatically.
- **Older exports** may lack explicit `f_tk` / `f_c0k` columns; the loader backfills these from `f_mk` for C18/C24 grade datasets.
- **Model checkpoints** are identified by prefix (e.g., `ID20260516_182257_LR1e-04_EP200_BS64_PW2.5_ROC0.863`). `load_surrogate_bundle()` resolves the latest checkpoint automatically, or accepts an explicit prefix via `prefix_sm=`.
- **Normalisation constants** (C_max, R_max) are derived once before each GA run via 40 random probe designs and stored in `_run_config.json`. They are condition-specific and not comparable across stock conditions.
