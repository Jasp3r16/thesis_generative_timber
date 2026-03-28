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
