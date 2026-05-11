# =============================================================================
# Step 2 — Cost Matrix Pre-Filter
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
#   Stage 3 — EC5 cross-section checks:
#             Tension:     A >= N / (kmod * f_tk  / gamma_M)
#             Compression: A >= N / (kc * kmod * f_c0k / gamma_M)  [with buckling]
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
MAX_OVERSIZE_FRAC = 0.25     # stock can be at most 25% longer (cutting waste limit)

# --- EC5 timber checks ---
GAMMA_M             = 1.3    # partial factor for timber (EC5 §2.4.1)
KMOD                = 0.8    # service class 1, medium-term load (EC5 Table 3.1)
BETA_C              = 0.2    # imperfection factor for glulam (EC5 §6.3.2)
FORCE_SAFETY_FACTOR = 1.30   # extra margin on estimated forces before EC5 checks
                              # accounts for ~15% approximation error in linear solve

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
    """
    n_nodes = node_positions.shape[0]
    n_edges = len(edges_v1)
    EA_arr  = np.full(n_edges, mean_EA_SI)
    K       = assemble_stiffness(node_positions, edges_v1, edges_v2, EA_arr)

    f_vec = np.zeros(n_nodes * 3)
    load_per_node = -total_load_n / len(load_nodes)
    for node in load_nodes:
        f_vec[node * 3 + 2] += load_per_node

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
    Returns bool mask [n_slots, n_stock], True = EC5 checks pass.
    Also returns minimum required Depth and Width per slot (mm)
    for populating df_slots.
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

    # Minimum required depth and width per slot (for df_slots output)
    min_depth_per_slot = np.full(n_slots, DEFAULT_MIN_DEPTH_MM)
    min_width_per_slot = np.full(n_slots, DEFAULT_MIN_WIDTH_MM)

    for slot_idx in range(n_slots):
        N_design = member_forces_n[slot_idx] * FORCE_SAFETY_FACTOR
        L_mm     = member_lengths_m[slot_idx] * 1000.0

        if abs(N_design) < 1.0:
            # Near-zero force — use defaults, no structural filter
            continue

        if N_design >= 0:
            # Tension: minimum area per stock element
            min_area                = N_design / f_td          # [n_stock]
            mask[slot_idx, :]      &= (A_mm2 >= min_area)
            # Record minimum depth/width from weakest passing stock
            passing                 = A_mm2 >= min_area
            if passing.any():
                min_depth_per_slot[slot_idx] = depth_mm[passing].min()
                min_width_per_slot[slot_idx] = width_mm[passing].min()
        else:
            # Compression + buckling
            N_comp = abs(N_design)
            pass_flags = np.ones(n_stock, dtype=bool)
            for s_idx in range(n_stock):
                i_y      = depth_mm[s_idx] / np.sqrt(12.0)
                lambda_s = L_mm / i_y
                kc       = buckling_factor_kc(lambda_s, E_005[s_idx], f_c0k[s_idx])
                min_area = N_comp / (kc * f_cd[s_idx])
                if A_mm2[s_idx] < min_area:
                    pass_flags[s_idx] = False
                    mask[slot_idx, s_idx] = False
            if pass_flags.any():
                min_depth_per_slot[slot_idx] = depth_mm[pass_flags].min()
                min_width_per_slot[slot_idx] = width_mm[pass_flags].min()

    return mask, min_depth_per_slot, min_width_per_slot


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
    mask_structural, min_depth, min_width = structural_filter(
        member_forces, slot_lengths_m, stock_df
    )
    mask_combined  = mask_length & mask_structural
    n_after_struct = int(mask_combined.sum())
    print(f"  Stage 3 (EC5):       {n_after_length - n_after_struct:6,} eliminated  "
          f"({n_after_struct:,} remaining, {100*n_after_struct/total:.1f}%)")

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
    }

    return df_slots, mask_combined, member_forces, filter_stats


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":

    stock_df    = pd.read_csv("complete_timber.csv", sep=";")
    edges_df    = pd.read_csv("df_edges.csv")
    vertices_df = pd.read_csv("df_vertices.csv")

    # Parse initial geometry (sample 0)
    verts = vertices_df[vertices_df['sample_id'] == 0].copy()
    verts['v_idx'] = verts['vertex_index'].str.replace('v', '').astype(int)
    verts = verts.sort_values('v_idx').reset_index(drop=True)
    node_positions = verts[['x', 'y', 'z']].values

    support_nodes = verts[verts['attribute'] == 'support']['v_idx'].tolist()
    load_nodes    = verts[verts['attribute'] == 'load']['v_idx'].tolist()

    print(f"Roof: {ROOF_LENGTH_M}m × {ROOF_WIDTH_M}m = {ROOF_AREA_M2:.0f} m²")
    print(f"Load: {LOAD_KN_PER_M2} kN/m² × {ROOF_AREA_M2:.0f} m² "
          f"= {TOTAL_LOAD_N/1000:.0f} kN total")
    print(f"      {TOTAL_LOAD_N/len(load_nodes)/1000:.2f} kN per load node "
          f"({len(load_nodes)} load nodes)")
    print(f"Length: hard lower bound (stock >= slot), "
          f"upper bound <= slot × {1+MAX_OVERSIZE_FRAC:.0%}")
    print()

    df_slots, feasibility_mask, member_forces, stats = build_cost_filter(
        node_positions = node_positions,
        edges_df       = edges_df,
        stock_df       = stock_df,
        support_nodes  = support_nodes,
        load_nodes     = load_nodes,
    )

    print()
    print("=" * 60)
    print("FILTER SUMMARY")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:<35} {v}")

    print()
    print("df_slots (first 5 rows):")
    print(df_slots.head().to_string(index=False))

    print()
    print(f"feasibility_mask shape: {feasibility_mask.shape}  dtype: {feasibility_mask.dtype}")
    print(f"  Feasible   (True):  {feasibility_mask.sum():,}")
    print(f"  Infeasible (False): {(~feasibility_mask).sum():,}")

    print()
    print("Usage in GA loop:")
    print("  cost_matrix[~feasibility_mask] = np.inf")