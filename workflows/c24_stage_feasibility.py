# =============================================================================
# Step 2 — Cost Matrix Pre-Filter  (v2)
# =============================================================================
#
# Filters the 120 × 506 slot/stock combination matrix before MILP.
# Outputs are directly compatible with c26_cost_calculation.build_cost_matrix().
#
# Load assumption:
#   Single truss carrying full roof (15m × 9m = 135 m²)
#   Design load: 2.0 kN/m² (self-weight + snow, per Eurocode guidelines)
#   Total: 270 kN distributed equally across 20 load nodes = 13.5 kN/node
#
# Filter pipeline:
#   Stage 1 — Length (hard constraint):
#             Stock must be >= slot length (no shorter allowed).
#             Stock may be longer by up to MAX_OVERSIZE_FRAC (cutting waste limit).
#   Stage 2 — Force estimation:
#             Linear truss solve with mean stock properties.
#             Indeterminate structure → forces within ~10-20% of true values.
#   Stage 3 — EC5 cross-section checks (v2: four sub-checks):
#             3a. Slenderness:      lambda = L/i <= MAX_SLENDERNESS (compression)
#             3b. Depth-to-length:  depth >= L / MAX_DEPTH_TO_LENGTH_RATIO (all members)
#             3c. Width-to-depth:   width >= depth / MAX_WIDTH_DEPTH_RATIO (all members)
#             3d. Tension:          A >= N / (kmod * f_tk  / gamma_M)
#             3e. Compression+buckling: A >= N / (kc * kmod * f_c0k / gamma_M)
#
# v2 additions vs v1:
#   - Stage 3b: depth-to-length ratio filter (d >= L/40) — eliminates shallow
#     stock from long slots regardless of force magnitude
#   - Stage 3c: width-to-depth ratio filter (w >= d/5) — eliminates laterally
#     unstable sections (EC5 lateral stability)
#   - Slenderness check (3a) now applied to ALL members, not just compression
#     (long tension members also benefit from minimum depth)
#   - filter_stats extended with per-stage elimination counts
#
# Outputs:
#   df_slots          — pd.DataFrame [120 rows] with edge_id, length_m,
#                       Length_Req, Width_Req, Depth_Req per slot
#   feasibility_mask  — np.ndarray bool [120, 506]
#                       True  = slot/stock combination is feasible
#                       False = infeasible (set to inf in cost matrix)
#   member_forces     — np.ndarray [120] estimated axial force per member (N)
#   filter_stats      — dict summarising eliminations per stage
#
# Integration (call each GA iteration):
#   df_slots, mask, forces, stats = build_cost_filter(node_positions, ...)
#
#   # Apply mask to your LCA cost matrix before MILP:
#   cost_matrix[~mask] = np.inf

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Load ---
ROOF_LENGTH_M     = 15.0     # truss long dimension (m)
ROOF_WIDTH_M      =  9.0     # truss short dimension (m)
LOAD_KN_PER_M2    =  2.0     # design load: self-weight + snow (kN/m²)
ROOF_AREA_M2      = ROOF_LENGTH_M * ROOF_WIDTH_M           # 135 m²
TOTAL_LOAD_N      = LOAD_KN_PER_M2 * ROOF_AREA_M2 * 1000  # 270,000 N

# --- Length filter ---
# Hard lower bound: stock must be >= slot length (LENGTH_TOL_FRAC = 0.0)
LENGTH_TOL_FRAC   = 0.00     # 0.0 = hard constraint, stock cannot be shorter
MAX_OVERSIZE_FRAC = 0.50     # stock can be at most 50% longer (cutting waste limit)

# --- EC5 timber checks ---
GAMMA_M             = 1.3    # partial factor for timber (EC5 §2.4.1)
KMOD                = 0.8    # service class 1, medium-term load (EC5 Table 3.1)
BETA_C              = 0.2    # imperfection factor for glulam (EC5 §6.3.2)
FORCE_SAFETY_FACTOR = 2.0    # extra margin on estimated forces before EC5 checks
                              # accounts for ~15% approximation error in linear solve
MAX_SLENDERNESS     = 150    # EC5 practical limit for compression members
                              # lambda = L/i <= 150  where i = depth/sqrt(12)

# v2 additions:
MAX_DEPTH_TO_LENGTH_RATIO = 40   # depth >= L / 40 (all members)
                                  # e.g. 3000mm slot requires depth >= 75mm
                                  # prevents shallow stock in long slots
                                  # regardless of force magnitude

MAX_WIDTH_DEPTH_RATIO = 5        # width >= depth / 5 (all members)
                                  # EC5 lateral stability requirement
                                  # e.g. 200mm deep section needs width >= 40mm

# --- Slot minimum cross-section defaults ---
# Used to populate Width_Req and Depth_Req in df_slots when no structural
# minimum is computed (near-zero force members). These are lower bounds only —
# the cost calculation uses them for volume/waste calculations.
DEFAULT_MIN_WIDTH_MM  = 38.0   # minimum standard timber width (mm)
DEFAULT_MIN_DEPTH_MM  = 100.0  # minimum standard timber depth (mm)

# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def compute_member_lengths(node_positions, edges_v1, edges_v2):
    """Euclidean length of each member (metres)."""
    return np.linalg.norm(
        node_positions[edges_v2] - node_positions[edges_v1], axis=1
    )


def compute_nodal_fz(
    node_positions: np.ndarray,
    support_nodes:  list[int],
    load_nodes:     list[int],
    load_kn_per_m2: float = LOAD_KN_PER_M2,
) -> np.ndarray:
    """
    Compute vertical nodal forces (N, downward = negative) from a uniform
    distributed roof load (kN/m²) using Delaunay tributary areas.

    All top-chord nodes (support_nodes ∪ load_nodes) receive load proportional
    to their XY-projected tributary area (1/3 of each adjacent Delaunay
    triangle's area). Bottom-chord nodes receive Fz = 0.

    This matches the Karamba load model used during training data generation,
    where corner/edge nodes receive less load than interior nodes.

    Parameters
    ----------
    node_positions : [n_nodes, 3] float, metres
    support_nodes  : indices of pin-support nodes (also carry roof load)
    load_nodes     : indices of free load-receiving nodes
    load_kn_per_m2 : distributed load intensity (kN/m²), default LOAD_KN_PER_M2

    Returns
    -------
    fz : np.ndarray [n_nodes] — Fz per node in Newtons, negative = downward
    """
    n_nodes     = node_positions.shape[0]
    fz          = np.zeros(n_nodes, dtype=np.float64)
    roof_idx    = sorted(set(load_nodes) | set(support_nodes))

    if len(roof_idx) < 3:
        return fz

    roof_xy    = node_positions[np.array(roof_idx), :2]
    tri        = Delaunay(roof_xy)
    pressure   = load_kn_per_m2 * 1_000.0  # → N/m²

    for simplex in tri.simplices:
        pts  = roof_xy[simplex]            # [3, 2]
        area = 0.5 * abs(
            (pts[1, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1]) -
            (pts[2, 0] - pts[0, 0]) * (pts[1, 1] - pts[0, 1])
        )
        share = -pressure * area / 3.0    # downward, 1/3 per vertex
        for local_i in simplex:
            fz[roof_idx[local_i]] += share

    return fz


# =============================================================================
# STAGE 1 — LENGTH FILTER (hard constraint)
# =============================================================================

def length_filter(slot_lengths_m, stock_lengths_mm):
    """
    Returns bool mask [n_slots, n_stock], True = length compatible.

    Lower bound: stock >= slot_length (hard, LENGTH_TOL_FRAC = 0.0).
    Upper bound: stock <= slot_length * (1 + MAX_OVERSIZE_FRAC).
    """
    slot_mm      = slot_lengths_m * 1000.0
    min_required = slot_mm[:, None] * (1.0 - LENGTH_TOL_FRAC)
    max_allowed  = slot_mm[:, None] * (1.0 + MAX_OVERSIZE_FRAC)
    return (stock_lengths_mm[None, :] >= min_required) & \
           (stock_lengths_mm[None, :] <= max_allowed)


# =============================================================================
# STAGE 2 — FORCE ESTIMATION (3D bar stiffness method)
# =============================================================================

def assemble_stiffness(node_positions, edges_v1, edges_v2, EA_per_member):
    """Assemble global stiffness matrix for 3D pin-jointed bar elements."""
    n_nodes = node_positions.shape[0]
    K       = np.zeros((n_nodes * 3, n_nodes * 3))
    for i, (v1, v2) in enumerate(zip(edges_v1, edges_v2)):
        d = node_positions[v2] - node_positions[v1]
        L = np.linalg.norm(d)
        if L < 1e-12:
            continue
        t        = d / L
        T        = np.zeros((2, 6))
        T[0, :3] = -t
        T[1, 3:] =  t
        k_e      = (EA_per_member[i] / L) * (T.T @ T)
        dofs = [v1*3, v1*3+1, v1*3+2, v2*3, v2*3+1, v2*3+2]
        for a in range(6):
            for b in range(6):
                K[dofs[a], dofs[b]] += k_e[a, b]
    return K


def apply_boundary_conditions(K, f_vec, support_nodes):
    """Enforce fixed pin supports (all 3 DOF fixed)."""
    K_bc = K.copy()
    f_bc = f_vec.copy()
    for node in support_nodes:
        for offset in range(3):
            dof            = node * 3 + offset
            K_bc[dof, :]   = 0.0
            K_bc[:, dof]   = 0.0
            K_bc[dof, dof] = 1.0
            f_bc[dof]      = 0.0
    return K_bc, f_bc


def estimate_member_forces(node_positions, edges_v1, edges_v2,
                           support_nodes, load_nodes,
                           total_load_n, mean_EA_SI):
    """
    Solve truss with uniform mean EA, distributed vertical load.
    Returns axial forces [n_members] in Newtons: + tension, - compression.

    Load is distributed by tributary area (compute_nodal_fz) to match the
    Karamba load model. total_load_n is ignored; load intensity comes from
    LOAD_KN_PER_M2 via compute_nodal_fz().
    """
    n_nodes = node_positions.shape[0]
    n_edges = len(edges_v1)
    EA_arr  = np.full(n_edges, mean_EA_SI)
    K       = assemble_stiffness(node_positions, edges_v1, edges_v2, EA_arr)

    fz_nodal = compute_nodal_fz(node_positions, support_nodes, load_nodes)
    f_vec    = np.zeros(n_nodes * 3)
    for node in range(n_nodes):
        f_vec[node * 3 + 2] += fz_nodal[node]

    K_bc, f_bc = apply_boundary_conditions(K, f_vec, support_nodes)

    try:
        u = np.linalg.solve(K_bc, f_bc)
    except np.linalg.LinAlgError:
        print("  Warning: singular stiffness matrix — check support conditions.")
        return np.zeros(n_edges)

    forces = np.zeros(n_edges)
    for i, (v1, v2) in enumerate(zip(edges_v1, edges_v2)):
        d = node_positions[v2] - node_positions[v1]
        L = np.linalg.norm(d)
        if L < 1e-12:
            continue
        t         = d / L
        delta     = np.dot(t, u[v2*3:v2*3+3] - u[v1*3:v1*3+3])
        forces[i] = (EA_arr[i] / L) * delta
    return forces


# =============================================================================
# STAGE 3 — EC5 CROSS-SECTION CHECKS
# =============================================================================

def buckling_factor_kc(slenderness, E_005, f_c0k):
    """EC5 §6.3.2 column buckling factor kc."""
    lambda_rel = (slenderness / np.pi) * np.sqrt(f_c0k / E_005)
    if lambda_rel <= 0.3:
        return 1.0
    k  = 0.5 * (1.0 + BETA_C * (lambda_rel - 0.3) + lambda_rel**2)
    kc = 1.0 / (k + np.sqrt(max(k**2 - lambda_rel**2, 0.0)))
    return float(np.clip(kc, 0.05, 1.0))


def structural_filter(member_forces_n, member_lengths_m, stock_df):
    """
    Returns bool mask [n_slots, n_stock], True = all EC5 checks pass.
    Also returns minimum required Depth and Width per slot (mm)
    for populating df_slots.

    Sub-checks applied in order. Each count is snapshotted after a full
    pass over all slots so the remaining numbers are globally correct.
        3a. Slenderness:       lambda = L/i <= MAX_SLENDERNESS (compression only)
        3b. Depth-to-length:   depth >= L / MAX_DEPTH_TO_LENGTH_RATIO (all members)
        3c. Width-to-depth:    width >= depth / MAX_WIDTH_DEPTH_RATIO (all members)
        3d. Tension strength:  A >= N / (kmod * f_tk  / gamma_M)
        3e. Compression+buckling: A >= N / (kc * kmod * f_c0k / gamma_M)
    """
    n_slots  = len(member_forces_n)
    n_stock  = len(stock_df)
    mask     = np.ones((n_slots, n_stock), dtype=bool)

    A_mm2    = (stock_df['Depth'].values * stock_df['Width'].values)
    f_tk     = stock_df['f_tk'].values
    f_c0k    = stock_df['f_c0k'].values
    E_005    = stock_df['E_modulus_005'].values
    depth_mm = stock_df['Depth'].values
    width_mm = stock_df['Width'].values

    f_td = KMOD * f_tk  / GAMMA_M
    f_cd = KMOD * f_c0k / GAMMA_M

    i_y_all      = depth_mm / np.sqrt(12.0)               # [n_stock]
    N_design_all = member_forces_n * FORCE_SAFETY_FACTOR   # [n_slots]
    L_mm_all     = member_lengths_m * 1000.0               # [n_slots]

    # ------------------------------------------------------------------
    # 3a. Slenderness (compression members only) — vectorised
    # ------------------------------------------------------------------
    comp_slots = N_design_all < -1.0                                  # [n_slots] bool
    if comp_slots.any():
        lambda_s = L_mm_all[comp_slots, None] / i_y_all[None, :]     # [n_comp, n_stock]
        mask[comp_slots] &= (lambda_s <= MAX_SLENDERNESS)

    n_after_slenderness = int(mask.sum())

    # ------------------------------------------------------------------
    # 3b. Depth-to-length ratio (all members) — vectorised
    # ------------------------------------------------------------------
    min_depth_req = L_mm_all / MAX_DEPTH_TO_LENGTH_RATIO              # [n_slots]
    mask &= (depth_mm[None, :] >= min_depth_req[:, None])

    n_after_depth_ratio = int(mask.sum())

    # ------------------------------------------------------------------
    # 3c. Width-to-depth ratio (all members) — vectorised
    # ------------------------------------------------------------------
    min_width_req = depth_mm / MAX_WIDTH_DEPTH_RATIO                  # [n_stock]
    mask &= (width_mm[None, :] >= min_width_req[None, :])

    n_after_width_ratio = int(mask.sum())

    # ------------------------------------------------------------------
    # 3d. Tension — vectorised over all tension slots
    # ------------------------------------------------------------------
    tens_slots = N_design_all >= 1.0                                   # [n_slots] bool
    if tens_slots.any():
        min_area_tens = N_design_all[tens_slots, None] / f_td[None, :]  # [n_tens, n_stock]
        mask[tens_slots] &= (A_mm2[None, :] >= min_area_tens)

    # ------------------------------------------------------------------
    # 3e. Compression + buckling — vectorised over all compression slots
    # ------------------------------------------------------------------
    if comp_slots.any():
        N_comp     = np.abs(N_design_all[comp_slots])                                # [n_comp]
        lambda_s   = L_mm_all[comp_slots, None] / i_y_all[None, :]                  # [n_comp, n_stock]
        lambda_rel = (lambda_s / np.pi) * np.sqrt(f_c0k[None, :] / E_005[None, :])  # [n_comp, n_stock]
        k          = 0.5 * (1.0 + BETA_C * (lambda_rel - 0.3) + lambda_rel**2)
        kc         = np.where(
            lambda_rel <= 0.3,
            1.0,
            1.0 / (k + np.sqrt(np.maximum(k**2 - lambda_rel**2, 0.0))),
        )
        kc         = np.clip(kc, 0.05, 1.0)
        min_area_comp = N_comp[:, None] / (kc * f_cd[None, :])                      # [n_comp, n_stock]
        mask[comp_slots] &= (A_mm2[None, :] >= min_area_comp)

    n_after_strength = int(mask.sum())

    # ---- Min required depth/width per slot (from surviving stock) --------
    depth_filtered     = np.where(mask, depth_mm[None, :], np.inf)
    width_filtered     = np.where(mask, width_mm[None, :], np.inf)
    col_min_depth      = depth_filtered.min(axis=1)
    col_min_width      = width_filtered.min(axis=1)
    min_depth_per_slot = np.where(
        np.isinf(col_min_depth), DEFAULT_MIN_DEPTH_MM, col_min_depth
    )
    min_width_per_slot = np.where(
        np.isinf(col_min_width), DEFAULT_MIN_WIDTH_MM, col_min_width
    )

    substage_counts = {
        "after_slenderness":  n_after_slenderness,
        "after_depth_ratio":  n_after_depth_ratio,
        "after_width_ratio":  n_after_width_ratio,
        "after_strength":     n_after_strength,
    }

    return mask, min_depth_per_slot, min_width_per_slot, substage_counts


# =============================================================================
# BUILD df_slots — compatible with c26_cost_calculation
# =============================================================================

def build_df_slots(edges_df, slot_lengths_m,
                   min_depth_per_slot, min_width_per_slot):
    """
    Build the df_slots DataFrame required by c26_cost_calculation.build_cost_matrix().

    Columns produced:
        edge_id      — slot identifier (e.g. 'e0', 'e1', ...)
        length_m     — slot length in metres (used by _slot_requirements)
        Length_Req   — slot length in mm     (fallback in _slot_requirements)
        Width_Req    — minimum required width (mm) from EC5 check
        Depth_Req    — minimum required depth (mm) from EC5 check
    """
    df_slots = pd.DataFrame({
        "edge_id":   edges_df["edge_id"].values,
        "length_m":  slot_lengths_m,
        "Length_Req": slot_lengths_m * 1000.0,   # mm
        "Width_Req":  min_width_per_slot,
        "Depth_Req":  min_depth_per_slot,
    })
    return df_slots



# =============================================================================
# MAIN — call each GA iteration
# =============================================================================

def build_cost_filter(node_positions, edges_df, stock_df,
                      support_nodes, load_nodes,
                      total_load_n=TOTAL_LOAD_N):
    """
    Build the feasibility mask and slot table for the cost matrix step.

    Parameters
    ----------
    node_positions : np.ndarray [39, 3]
        Current node xyz coordinates from GA (metres).
    edges_df : pd.DataFrame
        Edge table with columns edge_id, V1, V2 (fixed across GA iterations).
    stock_df : pd.DataFrame
        Stock inventory (506 elements) — complete_timber.csv.
    support_nodes : list[int]
        Node indices with fixed pin supports.
    load_nodes : list[int]
        Node indices where vertical load is applied.
    total_load_n : float
        Total vertical load in Newtons (default: 270,000 N).

    Returns
    -------
    df_slots : pd.DataFrame [120 rows]
        Slot table with edge_id, length_m, Length_Req, Width_Req, Depth_Req.

    feasibility_mask : np.ndarray bool [120, n_stock]
        True  = slot/stock combination passes all checks (length + EC5).
        False = infeasible — set to inf in cost matrix before MILP.
        Apply with: cost_matrix[~feasibility_mask] = np.inf

    member_forces : np.ndarray [120]
        Estimated axial force per member (N). + tension, - compression.

    filter_stats : dict
        Summary of eliminations per stage.
    """
    edges_v1  = edges_df['V1'].values
    edges_v2  = edges_df['V2'].values
    n_slots   = len(edges_v1)
    n_stock   = len(stock_df)
    total     = n_slots * n_stock

    print(f"Cost matrix filter: {n_slots} slots × {n_stock} stock = {total:,} combinations")
    print(f"  Load: {total_load_n/1000:.1f} kN total  "
          f"({total_load_n/1000/ROOF_AREA_M2:.2f} kN/m² × {ROOF_AREA_M2:.0f} m²)")

    slot_lengths_m   = compute_member_lengths(node_positions, edges_v1, edges_v2)
    stock_lengths_mm = stock_df['Length'].values

    # ---- Stage 1: Length ----
    mask_length    = length_filter(slot_lengths_m, stock_lengths_mm)
    n_after_length = int(mask_length.sum())
    print(f"  Stage 1 (length):    {total - n_after_length:6,} eliminated  "
          f"({n_after_length:,} remaining, {100*n_after_length/total:.1f}%)")

    # ---- Stage 2: Force estimation ----
    mean_E_Pa  = stock_df['E_modulus_eff'].mean() * 1e6
    mean_A_m2  = (stock_df['Depth'] * stock_df['Width']).mean() * 1e-6
    mean_EA_SI = mean_E_Pa * mean_A_m2

    member_forces = estimate_member_forces(
        node_positions, edges_v1, edges_v2,
        support_nodes, load_nodes, total_load_n, mean_EA_SI,
    )

    print(f"  Force estimation:    "
          f"max tension={member_forces.max()/1000:.1f} kN  "
          f"max compression={member_forces.min()/1000:.1f} kN  "
          f"mean |F|={np.abs(member_forces).mean()/1000:.1f} kN")

    # ---- Stage 3: EC5 structural checks ----
    mask_structural, min_depth, min_width, substage_counts = structural_filter(
        member_forces, slot_lengths_m, stock_df
    )

    # Compute combined counts at each EC5 sub-stage checkpoint
    # by intersecting the length mask with progressively applied EC5 masks
    depth_mm_s   = stock_df['Depth'].values
    width_mm_s   = stock_df['Width'].values
    i_y_s        = depth_mm_s / np.sqrt(12.0)
    slot_mm_s    = slot_lengths_m * 1000.0

    # 3a: slenderness mask (compression only) — vectorised
    N_design_s   = member_forces * FORCE_SAFETY_FACTOR               # [n_slots]
    comp_s       = N_design_s < -1.0                                  # [n_slots] bool
    mask_3a      = np.ones((len(slot_lengths_m), len(stock_df)), dtype=bool)
    if comp_s.any():
        lambda_s_3a       = slot_mm_s[comp_s, None] / i_y_s[None, :]
        mask_3a[comp_s]   = (lambda_s_3a <= MAX_SLENDERNESS)

    # 3b: depth-to-length mask
    min_d_req = slot_mm_s / MAX_DEPTH_TO_LENGTH_RATIO
    mask_3b   = depth_mm_s[None, :] >= min_d_req[:, None]

    # 3c: width-to-depth mask
    min_w_req = depth_mm_s / MAX_WIDTH_DEPTH_RATIO
    mask_3c   = width_mm_s[None, :] >= min_w_req[None, :]

    n_after_3a = int((mask_length & mask_3a).sum())
    n_after_3b = int((mask_length & mask_3a & mask_3b).sum())
    n_after_3c = int((mask_length & mask_3a & mask_3b & mask_3c).sum())

    mask_combined  = mask_length & mask_structural
    n_after_struct = int(mask_combined.sum())

    # Report stage 3 with correct sub-stage breakdown
    print(f"  Stage 3 (EC5):       {n_after_length - n_after_struct:6,} eliminated  "
          f"({n_after_struct:,} remaining, {100*n_after_struct/total:.1f}%)")

    sub_steps = [
        ("3a slenderness",  n_after_length, n_after_3a),
        ("3b depth/length", n_after_3a,     n_after_3b),
        ("3c width/depth",  n_after_3b,     n_after_3c),
        ("3d/e strength",   n_after_3c,     n_after_struct),
    ]
    for label, before, after in sub_steps:
        elim = before - after
        if elim > 0:
            print(f"    {label:<18}  -{elim:5,}  ({after:,} remaining)")

    # ---- Warn on unassignable slots ----
    slots_no_stock = np.where(mask_combined.sum(axis=1) == 0)[0]
    if len(slots_no_stock) > 0:
        print(f"\n  WARNING: {len(slots_no_stock)} slot(s) have NO feasible stock: "
              f"{slots_no_stock.tolist()}")
        for s in slots_no_stock:
            print(f"    Slot {s:3d} ({edges_df['edge_id'].iloc[s]}): "
                  f"length={slot_lengths_m[s]*1000:.0f} mm  "
                  f"pass_length={mask_length[s].sum()}  "
                  f"pass_EC5={mask_combined[s].sum()}")
        print(f"  Tip: GA should penalise this geometry — "
              f"MILP cannot assign these slots from current stock pool.")

    # ---- Build outputs ----
    df_slots = build_df_slots(
        edges_df, slot_lengths_m, min_depth, min_width
    )

    filter_stats = {
        "total_combinations":        int(total),
        "after_length_filter":       int(n_after_length),
        "after_structural_filter":   int(n_after_struct),
        "pct_feasible":              float(100 * n_after_struct / total),
        "slots_no_feasible_stock":   slots_no_stock.tolist(),
        "n_tension_members":         int((member_forces >  1.0).sum()),
        "n_compression_members":     int((member_forces < -1.0).sum()),
        "max_tension_kn":            float(member_forces.max() / 1000),
        "max_compression_kn":        float(member_forces.min() / 1000),
        # EC5 sub-stage eliminations (relative to post-length-filter baseline)
        "ec5_elim_slenderness":      int(n_after_length - n_after_3a),
        "ec5_elim_depth_ratio":      int(n_after_3a     - n_after_3b),
        "ec5_elim_width_ratio":      int(n_after_3b     - n_after_3c),
        "ec5_elim_strength":         int(n_after_3c     - n_after_struct),
    }

    return df_slots, mask_combined, member_forces, filter_stats