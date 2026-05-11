# =============================================================================
# Step 2 — Cost Matrix Pre-Filter
# =============================================================================
#
# Filters the 120 × 506 slot/stock combination matrix before MILP.
# Sets cost matrix entries to inf where a stock element is geometrically or
# structurally incompatible with a slot.
#
# Load assumption:
#   Single truss carrying full roof (15m × 9m = 135 m²)
#   Design load: 2.0 kN/m² (self-weight + snow, per Eurocode guidelines)
#   Total: 270 kN distributed equally across 20 load nodes = 13.5 kN/node
#
# Filter pipeline:
#   Stage 1 — Length (hard constraint): stock must be >= slot length.
#             Stock may be longer by up to MAX_OVERSIZE_FRAC (cutting waste limit).
#   Stage 2 — Force estimation: linear truss solve with mean stock properties.
#             Indeterminate structure → forces within ~10-20% of true values.
#   Stage 3 — EC5 cross-section checks:
#             Tension:     A >= N / (kmod * f_tk / gamma_M)
#             Compression: A >= N / (kc * kmod * f_c0k / gamma_M)  [with buckling]
#
# Output:
#   feasibility_mask  — bool [120, n_stock], True = combination is feasible
#   member_forces     — float [120], estimated axial force per member (N)
#   filter_stats      — dict summarising eliminations per stage
#
# Integration (call each GA iteration):
#   mask, forces, stats = build_cost_filter(node_positions, ...)
#   filtered_cost = apply_feasibility_mask(lca_cost_matrix, mask)

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Load ---
ROOF_LENGTH_M     = 15.0    # truss long dimension (m)
ROOF_WIDTH_M      =  9.0    # truss short dimension (m)
LOAD_KN_PER_M2    =  2.0    # design load: self-weight + snow (kN/m²)
ROOF_AREA_M2      = ROOF_LENGTH_M * ROOF_WIDTH_M          # 135 m²
TOTAL_LOAD_N      = LOAD_KN_PER_M2 * ROOF_AREA_M2 * 1000  # 270,000 N

# --- Length filter ---
# Hard lower bound: stock must be >= slot length (LENGTH_TOL_FRAC = 0.0)
# Stock may be longer by up to MAX_OVERSIZE_FRAC to allow cutting
LENGTH_TOL_FRAC   = 0.00    # 0.0 = hard constraint, no shorter than slot
MAX_OVERSIZE_FRAC = 0.50    # stock can be at most 50% longer than slot

# --- EC5 timber checks ---
GAMMA_M           = 1.3     # partial factor for timber (EC5 §2.4.1)
KMOD              = 0.8     # service class 1, medium-term load (EC5 Table 3.1)
BETA_C            = 0.2     # imperfection factor for glulam (EC5 §6.3.2)
FORCE_SAFETY_FACTOR = 1.30  # extra margin on estimated forces before EC5 checks
                             # accounts for ~15% error in approximate solve

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

    Hard lower bound: stock >= slot length (LENGTH_TOL_FRAC = 0.0 means exact).
    Upper bound: stock <= slot * (1 + MAX_OVERSIZE_FRAC) to limit cutting waste.

    Units: slot lengths in metres, stock lengths in mm (as stored in CSV).
    """
    slot_mm  = slot_lengths_m * 1000.0      # convert m → mm for comparison

    # Broadcasting: [n_slots, 1] vs [1, n_stock]
    min_required = slot_mm[:, None] * (1.0 - LENGTH_TOL_FRAC)
    max_allowed  = slot_mm[:, None] * (1.0 + MAX_OVERSIZE_FRAC)

    return (stock_lengths_mm[None, :] >= min_required) & \
           (stock_lengths_mm[None, :] <= max_allowed)


# =============================================================================
# STAGE 2 — FORCE ESTIMATION (3D bar stiffness method)
# =============================================================================

def assemble_stiffness(node_positions, edges_v1, edges_v2, EA_per_member):
    """
    Assemble global stiffness matrix for 3D bar (pin-jointed truss) elements.
    DOF layout: [ux, uy, uz] per node → total DOF = n_nodes × 3.
    """
    n_nodes = node_positions.shape[0]
    K       = np.zeros((n_nodes * 3, n_nodes * 3))

    for i, (v1, v2) in enumerate(zip(edges_v1, edges_v2)):
        d = node_positions[v2] - node_positions[v1]
        L = np.linalg.norm(d)
        if L < 1e-12:
            continue
        t = d / L   # direction cosines

        # 6×6 element stiffness (global coords): k = (EA/L) * T^T * T
        T          = np.zeros((2, 6))
        T[0, :3]   = -t
        T[1, 3:]   =  t
        k_e        = (EA_per_member[i] / L) * (T.T @ T)

        dofs = [v1*3, v1*3+1, v1*3+2, v2*3, v2*3+1, v2*3+2]
        for a in range(6):
            for b in range(6):
                K[dofs[a], dofs[b]] += k_e[a, b]

    return K


def apply_boundary_conditions(K, f_vec, support_nodes):
    """
    Enforce fixed supports by zeroing rows/cols and setting diagonal = 1.
    All 3 DOF fixed per support node (pin support).
    """
    K_bc = K.copy()
    f_bc = f_vec.copy()
    for node in support_nodes:
        for offset in range(3):
            dof = node * 3 + offset
            K_bc[dof, :]   = 0.0
            K_bc[:, dof]   = 0.0
            K_bc[dof, dof] = 1.0
            f_bc[dof]      = 0.0
    return K_bc, f_bc


def estimate_member_forces(node_positions, edges_v1, edges_v2,
                           support_nodes, load_nodes, total_load_n, mean_EA_SI):
    """
    Solve truss with uniform mean EA to get approximate member forces.
    Load is distributed equally in the -z direction across all load nodes.

    Returns: axial forces [n_members] in Newtons, + = tension, - = compression.
    """
    n_nodes    = node_positions.shape[0]
    n_edges    = len(edges_v1)
    EA_arr     = np.full(n_edges, mean_EA_SI)

    K   = assemble_stiffness(node_positions, edges_v1, edges_v2, EA_arr)

    # Build load vector: equal downward (-z) load per load node
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

    # Recover axial force from nodal displacements
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
# STAGE 3 — EC5 CROSS-SECTION ADEQUACY FILTER
# =============================================================================

def buckling_factor_kc(slenderness, E_005, f_c0k):
    """
    EC5 §6.3.2 column buckling factor kc.
    slenderness: lambda = L_ef / i  where i = sqrt(I/A) = depth/sqrt(12) for rect.
    """
    lambda_rel = (slenderness / np.pi) * np.sqrt(f_c0k / E_005)
    if lambda_rel <= 0.3:
        return 1.0
    k  = 0.5 * (1.0 + BETA_C * (lambda_rel - 0.3) + lambda_rel**2)
    kc = 1.0 / (k + np.sqrt(max(k**2 - lambda_rel**2, 0.0)))
    return float(np.clip(kc, 0.05, 1.0))


def structural_filter(member_forces_n, member_lengths_m, stock_df):
    """
    Returns bool mask [n_slots, n_stock], True = EC5 checks pass.

    Tension:     A  >= N / (kmod * f_tk  / gamma_M)
    Compression: A  >= N / (kc * kmod * f_c0k / gamma_M)
                 where kc accounts for Euler column buckling (EC5 §6.3.2)

    Forces are scaled by FORCE_SAFETY_FACTOR before checking.
    """
    n_slots = len(member_forces_n)
    n_stock = len(stock_df)
    mask    = np.ones((n_slots, n_stock), dtype=bool)

    # Pre-compute stock arrays (mm and N/mm²)
    A_mm2    = (stock_df['Depth'].values * stock_df['Width'].values)   # mm²
    f_tk     = stock_df['f_tk'].values        # N/mm²
    f_c0k    = stock_df['f_c0k'].values       # N/mm²
    E_005    = stock_df['E_modulus_005'].values  # N/mm²
    depth_mm = stock_df['Depth'].values       # mm — governs buckling axis

    # Design strengths [n_stock]
    f_td = KMOD * f_tk  / GAMMA_M
    f_cd = KMOD * f_c0k / GAMMA_M

    for slot_idx in range(n_slots):
        N_design = member_forces_n[slot_idx] * FORCE_SAFETY_FACTOR
        L_mm     = member_lengths_m[slot_idx] * 1000.0

        if N_design >= 0:
            # Tension: minimum area check [vectorised over stock]
            min_area            = N_design / f_td          # [n_stock]
            mask[slot_idx, :]  &= (A_mm2 >= min_area)
        else:
            # Compression + buckling: loop over stock (kc is stock-dependent)
            N_comp = abs(N_design)
            for s_idx in range(n_stock):
                i_y      = depth_mm[s_idx] / np.sqrt(12.0)  # radius of gyration
                lambda_s = L_mm / i_y                        # slenderness
                kc       = buckling_factor_kc(lambda_s, E_005[s_idx], f_c0k[s_idx])
                min_area = N_comp / (kc * f_cd[s_idx])
                if A_mm2[s_idx] < min_area:
                    mask[slot_idx, s_idx] = False

    return mask


# =============================================================================
# MAIN — call each GA iteration
# =============================================================================

def build_cost_filter(node_positions, edges_v1, edges_v2,
                      support_nodes, load_nodes, stock_df,
                      total_load_n=TOTAL_LOAD_N):
    """
    Build feasibility mask for the cost matrix.

    Parameters
    ----------
    node_positions : np.ndarray [39, 3]
        Current node xyz coordinates from GA (metres).
    edges_v1, edges_v2 : np.ndarray [120]
        Member connectivity (fixed across GA iterations).
    support_nodes : list[int]
        Nodes with fixed pin supports.
    load_nodes : list[int]
        Nodes where vertical load is applied.
    stock_df : pd.DataFrame
        Stock inventory (506 elements).
    total_load_n : float
        Total vertical load in Newtons (default: 270,000 N = 2 kN/m² × 135 m²).

    Returns
    -------
    feasibility_mask : np.ndarray bool [120, n_stock]
        True = slot/stock combination is feasible for cost matrix.
    member_forces : np.ndarray [120]
        Estimated axial force per member (N). + tension, - compression.
    filter_stats : dict
    """
    n_slots = len(edges_v1)
    n_stock = len(stock_df)
    total   = n_slots * n_stock

    print(f"Cost matrix filter: {n_slots} slots × {n_stock} stock = {total:,} combinations")
    print(f"  Load: {total_load_n/1000:.1f} kN total  "
          f"({total_load_n/1000/ROOF_AREA_M2:.2f} kN/m² × {ROOF_AREA_M2:.0f} m²)")

    # Member lengths
    slot_lengths_m   = compute_member_lengths(node_positions, edges_v1, edges_v2)
    stock_lengths_mm = stock_df['Length'].values

    # ---- Stage 1: Length ----
    mask_length    = length_filter(slot_lengths_m, stock_lengths_mm)
    n_after_length = int(mask_length.sum())
    print(f"  Stage 1 (length):    {total - n_after_length:6,} eliminated  "
          f"({n_after_length:,} remaining, {100*n_after_length/total:.1f}%)")

    # ---- Stage 2: Force estimation ----
    # Convert mean stock properties to SI (Pa and m²) for stiffness solve
    mean_E_Pa  = stock_df['E_modulus_eff'].mean() * 1e6   # N/mm² → Pa (N/m²)
    mean_A_m2  = (stock_df['Depth'] * stock_df['Width']).mean() * 1e-6  # mm² → m²
    mean_EA_SI = mean_E_Pa * mean_A_m2                    # N

    member_forces = estimate_member_forces(
        node_positions, edges_v1, edges_v2,
        support_nodes, load_nodes, total_load_n, mean_EA_SI,
    )

    print(f"  Force estimation:    "
          f"max tension={member_forces.max()/1000:.1f} kN  "
          f"max compression={member_forces.min()/1000:.1f} kN  "
          f"mean |F|={np.abs(member_forces).mean()/1000:.1f} kN")

    # ---- Stage 3: EC5 structural checks ----
    mask_structural = structural_filter(member_forces, slot_lengths_m, stock_df)
    mask_combined   = mask_length & mask_structural

    n_after_struct = int(mask_combined.sum())
    print(f"  Stage 3 (EC5):       {n_after_length - n_after_struct:6,} eliminated  "
          f"({n_after_struct:,} remaining, {100*n_after_struct/total:.1f}%)")

    # ---- Warn on slots with no feasible stock ----
    slots_no_stock = np.where(mask_combined.sum(axis=1) == 0)[0]
    if len(slots_no_stock) > 0:
        print(f"\n  WARNING: {len(slots_no_stock)} slot(s) have NO feasible stock "
              f"after filtering: {slots_no_stock.tolist()}")
        for s in slots_no_stock:
            L_mm = slot_lengths_m[s] * 1000
            n_pass_length = mask_length[s].sum()
            print(f"    Slot {s:3d}: length={L_mm:.0f} mm  "
                  f"({n_pass_length} pass length, "
                  f"{mask_combined[s].sum()} pass EC5)")
        print(f"  Consider: relaxing MAX_OVERSIZE_FRAC (currently {MAX_OVERSIZE_FRAC:.0%}), "
              f"or adding shorter/stronger stock elements.")

    filter_stats = {
        "total_combinations":        int(total),
        "after_length_filter":       int(n_after_length),
        "after_structural_filter":   int(n_after_struct),
        "pct_feasible":              float(100 * n_after_struct / total),
        "slots_no_feasible_stock":   slots_no_stock.tolist(),
        "n_tension_members":         int((member_forces > 1.0).sum()),
        "n_compression_members":     int((member_forces < -1.0).sum()),
        "max_tension_kn":            float(member_forces.max() / 1000),
        "max_compression_kn":        float(member_forces.min() / 1000),
    }

    return mask_combined, member_forces, filter_stats


def apply_feasibility_mask(cost_matrix, feasibility_mask):
    """
    Set infeasible entries to inf in the cost matrix.

    cost_matrix:      np.ndarray float [n_slots, n_stock]
    feasibility_mask: np.ndarray bool  [n_slots, n_stock]
    Returns: cost matrix with inf at infeasible entries (copy).
    """
    masked = cost_matrix.astype(float).copy()
    masked[~feasibility_mask] = np.inf
    return masked


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
    node_positions = verts[['x', 'y', 'z']].values   # [39, 3] metres

    edges_v1      = edges_df['V1'].values
    edges_v2      = edges_df['V2'].values
    support_nodes = verts[verts['attribute'] == 'support']['v_idx'].tolist()
    load_nodes    = verts[verts['attribute'] == 'load']['v_idx'].tolist()

    print(f"Roof: {ROOF_LENGTH_M}m × {ROOF_WIDTH_M}m = {ROOF_AREA_M2:.0f} m²")
    print(f"Load: {LOAD_KN_PER_M2} kN/m² → {TOTAL_LOAD_N/1000:.0f} kN total")
    print(f"      distributed across {len(load_nodes)} load nodes "
          f"= {TOTAL_LOAD_N/len(load_nodes)/1000:.2f} kN/node")
    print(f"Length filter: stock >= slot length (hard), "
          f"<= slot × {1+MAX_OVERSIZE_FRAC:.0%} (waste limit)")
    print()

    feasibility_mask, member_forces, stats = build_cost_filter(
        node_positions = node_positions,
        edges_v1       = edges_v1,
        edges_v2       = edges_v2,
        support_nodes  = support_nodes,
        load_nodes     = load_nodes,
        stock_df       = stock_df,
    )

    # Dummy LCA cost matrix to demonstrate masking
    dummy_cost   = np.random.uniform(0.5, 5.0, size=(len(edges_df), len(stock_df)))
    masked_cost  = apply_feasibility_mask(dummy_cost, feasibility_mask)

    print()
    print("=" * 60)
    print("FILTER SUMMARY")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:<35} {v}")
    print()
    print(f"Cost matrix entries excluded: "
          f"{np.isinf(masked_cost).sum():,} / {masked_cost.size:,} "
          f"({100*np.isinf(masked_cost).mean():.1f}%)")
    print()
    print("Member force distribution:")
    print(f"  Tension     (N > 0): {(member_forces >  1.0).sum()} members")
    print(f"  Compression (N < 0): {(member_forces < -1.0).sum()} members")
    print(f"  Near-zero:           {(np.abs(member_forces) <= 1.0).sum()} members")
    print(f"  Max tension:         {member_forces.max()/1000:.2f} kN")
    print(f"  Max compression:     {member_forces.min()/1000:.2f} kN")