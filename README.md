# Deep Generative Design for Reclaimed Timber Structures
**MSc Thesis - Building Technology | TU Delft**

## Project Overview
This research investigates a hybrid computational workflow for the integration of **non-standardized reclaimed timber** into **deep generative design** processes. The goal is to bridge the gap between continuous geometric optimization and the discrete constraints of a finite material stock.

### Key Research Focus:
* **Material-Aware Tessellation:** Decomposing complex geometries into buildable components.
* **Discrete Material Matching:** Developing multi-objective algorithms (MOGA/GNN) to assign specific timber members to tessellae based on length, cross-section, and strength class.

---

##The Computational Pipeline
The workflow is divided into six primary stages:
1. **Geometry Generation:** Deep generative models for initial shell structures.
2. **Structural Evaluation:** Material-agnostic FEA.
3. **Material-Aware Tessellation:** Geometry discretization.
4. **Material Matching & Assignment:** The core algorithmic intervention.
5. **Integrated Structural Re-analysis:** FEA with heterogeneous material data.
6. **Iterative Optimization:** Feedback loops between inventory and design.

---

##Repository Structure
```text
├── notebooks/          # Google Colab / Jupyter Notebooks for experiments
├── scripts/            # Core Python modules (.py) for logic and math
├── data/               # (Optional) Sample inventory data (CSV/Excel)
├── rhino_gh/           # Grasshopper definitions and Rhino base files
└── exports/            # Sample outputs, plots, and evaluation results
