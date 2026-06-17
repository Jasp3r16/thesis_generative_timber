"""Direct FEM structural evaluator — drop-in alternative to the GNN surrogate.

This module is a *toggleable* replacement for `c27_stage_GNN.run_gnn_stage`. It
solves the pin-jointed space truss directly (correct direct-stiffness method) with
the actual per-member section properties from the MILP assignment, then applies the
project's EC5 axial+buckling utilisation check per member. It returns the **same
contract** as the GNN stage so the optimiser cannot tell which backend it called:

    feasibility_score : float in [0,1]   (higher = safer)   ← drives the optimiser
    unsafe_member_ids : list[int]                            ← stored / reporting
    preds_physical    : np.ndarray[n_members]               ← stored / reporting

Why a new solver instead of reusing c24:
    `c24_stage_feasibility.assemble_stiffness` is NOT a correct truss FEM — its
    element matrix `(EA/L)·(Tᵀ T)` with the 2×6 `T` is block-diagonal, dropping the
    node-to-node coupling, so member forces are wrong whenever a member spans two
    free nodes (verified by hand-calc). c24's force estimate is only an internal
    pre-filter / GNN feature, not a structural ground truth. Here we assemble the
    correct element matrix `(EA/L)·[[tt^T,-tt^T],[-tt^T,tt^T]]`.

We DO reuse c24's correct, non-solver pieces: the tributary-area load model
(`compute_nodal_fz`), the boundary-condition application, and the EC5 constants.

Design decisions (confirmed with the human, see SCAN.md):
    D1  feasibility_score = 1 − n_unsafe / n_members   (fraction of members safe)
    D2  force_safety_factor = 1.0 (exact per-member forces; drop c24's pre-filter ×2)

Units: solve in SI (m, Pa, N); EC5 capacity in mm / N·mm⁻² (MPa) / N — matching c24
and the Karamba training labels. The module has no torch dependency.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import c24_stage_feasibility as c24
from c24_stage_feasibility import (
    compute_nodal_fz, LOAD_KN_PER_M2,
    GAMMA_M, KMOD, BETA_C,
)

# Default boundary conditions for the 5×3 grid topology (mirror c27_stage_GNN).
# Pass support_nodes / load_nodes explicitly when the geometry differs.
_DEFAULT_SUPPORT_NODES = [0, 5, 18, 23]
_DEFAULT_LOAD_NODES    = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                          16, 17, 19, 20, 21, 22]


# SECTION-PROPERTY RESOLUTION ------------------------------------------------

def _resolve_si_section(
    milp_assignment: np.ndarray,
    df_input_stock:  pd.DataFrame,
    stock_df:        "pd.DataFrame | None",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (E_pa[n], A_m2[n]) for the assigned members, in SI units.

    Prefers the pre-prepared SI stock table (output of prepare_stock_for_gnn) so the
    section properties match the GNN path byte-for-byte. Falls back to converting the
    raw stock (mm, N/mm²) if no prepared table is supplied.
    """
    if stock_df is not None and {"E", "Width_m", "Depth_m"}.issubset(stock_df.columns):
        s = stock_df.iloc[milp_assignment]
        E_pa = s["E"].to_numpy(dtype=np.float64)
        A_m2 = (s["Width_m"].to_numpy(dtype=np.float64) *
                s["Depth_m"].to_numpy(dtype=np.float64))
    else:
        s = df_input_stock.iloc[milp_assignment]
        E_pa = s["E_modulus_eff"].to_numpy(dtype=np.float64) * 1e6        # N/mm² → Pa
        A_m2 = (s["Width"].to_numpy(dtype=np.float64) *
                s["Depth"].to_numpy(dtype=np.float64)) * 1e-6             # mm² → m²
    return E_pa, A_m2


# CORRECT SPACE-TRUSS DIRECT-STIFFNESS SOLVER --------------------------------
#
# Factored as: assemble_truss_K (geometry+EA → K) · solve_truss (arbitrary fixed
# DOFs + arbitrary load vector) · solve_truss_axial_forces (roof-load wrapper used
# by the optimiser). The general solve_truss core is what the Phase-4 unit tests
# call with textbook trusses (known closed-form forces).

def assemble_truss_K(
    node_positions: np.ndarray,
    edges_v1:       np.ndarray,
    edges_v2:       np.ndarray,
    EA:             np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble the global stiffness matrix for a 3D pin-jointed truss.

    Correct element matrix:  k_e = (EA/L) [[ tt^T, -tt^T], [-tt^T, tt^T]]
    (NOT c24's block-diagonal (EA/L)·TᵀT, which drops the node coupling.)

    Returns (K [n_dof, n_dof], L [n_mem], t [n_mem, 3], valid [n_mem]).
    """
    n_dof = node_positions.shape[0] * 3
    n_mem = len(edges_v1)

    d     = node_positions[edges_v2] - node_positions[edges_v1]       # [n_mem, 3]
    L     = np.linalg.norm(d, axis=1)                                 # [n_mem]
    valid = L > 1e-12
    t     = np.zeros_like(d)
    t[valid] = d[valid] / L[valid, None]                             # unit vectors

    K = np.zeros((n_dof, n_dof))
    for i in range(n_mem):
        if not valid[i]:
            continue
        ti   = t[i]
        ktt  = (EA[i] / L[i]) * np.outer(ti, ti)                      # 3×3
        a, b = int(edges_v1[i]) * 3, int(edges_v2[i]) * 3
        K[a:a+3, a:a+3] += ktt
        K[b:b+3, b:b+3] += ktt
        K[a:a+3, b:b+3] -= ktt
        K[b:b+3, a:a+3] -= ktt
    return K, L, t, valid


def solve_truss(
    node_positions: np.ndarray,
    edges_v1:       np.ndarray,
    edges_v2:       np.ndarray,
    EA:             np.ndarray,
    fixed_dofs:     np.ndarray,
    f_ext:          np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    General truss solve for arbitrary fixed DOFs and an arbitrary load vector.

    Parameters
    ----------
    fixed_dofs : iterable[int]   global DOF indices (node*3 + {0,1,2}) held at zero
    f_ext      : np.ndarray[n_dof]   external nodal load vector

    Returns
    -------
    forces : np.ndarray[n_members]  axial force (N), + tension / − compression
    u      : np.ndarray[n_dof]      nodal displacements (0 on failure)
    ok     : bool                   False if K is singular (mechanism / unstable)
    """
    n_nodes = node_positions.shape[0]
    n_dof   = n_nodes * 3
    n_mem   = len(edges_v1)

    K, L, t, valid = assemble_truss_K(node_positions, edges_v1, edges_v2, EA)

    # ---- apply boundary conditions (zero the fixed rows/cols, unit on diagonal)
    K_bc = K.copy()
    f_bc = np.asarray(f_ext, dtype=np.float64).copy()
    for dof in fixed_dofs:
        K_bc[dof, :] = 0.0
        K_bc[:, dof] = 0.0
        K_bc[dof, dof] = 1.0
        f_bc[dof] = 0.0

    try:
        u = np.linalg.solve(K_bc, f_bc)
    except np.linalg.LinAlgError:
        return np.zeros(n_mem), np.zeros(n_dof), False
    if not np.all(np.isfinite(u)):
        return np.zeros(n_mem), np.zeros(n_dof), False

    # ---- recover member axial forces:  N = (EA/L) · t·(u2 − u1) ---------------
    u_nodes = u.reshape(n_nodes, 3)
    du      = u_nodes[edges_v2] - u_nodes[edges_v1]                   # [n_mem, 3]
    forces  = np.zeros(n_mem)
    forces[valid] = (EA[valid] / L[valid]) * np.einsum("ij,ij->i", t[valid], du[valid])
    return forces, u, True


def solve_truss_axial_forces(
    node_positions: np.ndarray,
    edges_v1:       np.ndarray,
    edges_v2:       np.ndarray,
    EA:             np.ndarray,
    support_nodes:  list[int],
    load_nodes:     list[int],
    load_kn_per_m2: float = LOAD_KN_PER_M2,
) -> tuple[np.ndarray, bool]:
    """
    Roof-truss wrapper: pin supports (all 3 translations fixed) + vertical nodal
    loads from Delaunay tributary areas (compute_nodal_fz). Used in the optimiser.

    Returns (forces [n_members] N, ok bool).
    """
    n_dof = node_positions.shape[0] * 3

    fz    = compute_nodal_fz(node_positions, support_nodes, load_nodes, load_kn_per_m2)
    f_ext = np.zeros(n_dof)
    f_ext[2::3] = fz                                                  # z-DOF per node

    fixed_dofs = np.array(
        [n * 3 + k for n in support_nodes for k in range(3)], dtype=int
    )

    forces, _u, ok = solve_truss(
        node_positions, edges_v1, edges_v2, EA, fixed_dofs, f_ext,
    )
    return forces, ok


# EC5 PER-MEMBER UTILISATION -------------------------------------------------

def ec5_member_utilisation(
    member_forces_n:  np.ndarray,
    member_lengths_m: np.ndarray,
    width_mm:         np.ndarray,
    depth_mm:         np.ndarray,
    f_tk:             np.ndarray,
    f_c0k:            np.ndarray,
    E_005:            np.ndarray,
    force_safety_factor: float = 1.0,
) -> np.ndarray:
    """
    Per-member EC5 utilisation ratio UC (UC ≥ 1 ⇒ unsafe). Pure axial strength +
    compression buckling — the same checks c24 uses, recast as a continuous UC for
    the single assigned section. Mirrors the Karamba "Utilization" label semantics.

    Tension      : UC = N / (A · f_td),               f_td = kmod·f_tk / γ_M
    Compression  : UC = |N| / (A · kc · f_cd),         f_cd = kmod·f_c0k / γ_M
                   kc per EC5 §6.3.2, weak-axis radius of gyration i = min(w,d)/√12
    """
    A_mm2 = width_mm * depth_mm
    f_td  = KMOD * f_tk  / GAMMA_M
    f_cd  = KMOD * f_c0k / GAMMA_M

    N      = member_forces_n * force_safety_factor
    L_mm   = member_lengths_m * 1000.0
    i_z    = np.minimum(width_mm, depth_mm) / np.sqrt(12.0)           # weak-axis governs
    lam    = np.divide(L_mm, i_z, out=np.zeros_like(L_mm), where=i_z > 0)

    UC = np.zeros_like(N, dtype=np.float64)

    tens = N >= 0.0
    comp = ~tens

    # tension
    denom_t = A_mm2 * f_td
    UC[tens] = np.divide(N[tens], denom_t[tens],
                         out=np.zeros_like(N[tens]), where=denom_t[tens] > 0)

    # compression + buckling
    if np.any(comp):
        lam_rel = (lam[comp] / np.pi) * np.sqrt(f_c0k[comp] / E_005[comp])
        k       = 0.5 * (1.0 + BETA_C * (lam_rel - 0.3) + lam_rel ** 2)
        kc      = np.where(
            lam_rel <= 0.3,
            1.0,
            1.0 / (k + np.sqrt(np.maximum(k ** 2 - lam_rel ** 2, 0.0))),
        )
        kc      = np.clip(kc, 0.05, 1.0)
        denom_c = A_mm2[comp] * kc * f_cd[comp]
        UC[comp] = np.divide(np.abs(N[comp]), denom_c,
                             out=np.zeros_like(N[comp]), where=denom_c > 0)

    return UC


# FEM FEASIBILITY — call every optimiser iteration ---------------------------

def fem_feasibility(
    node_positions:  np.ndarray,
    milp_assignment: np.ndarray,
    df_input_stock:  pd.DataFrame,
    edges_v1:        np.ndarray,
    edges_v2:        np.ndarray,
    stock_df:        "pd.DataFrame | None" = None,
    support_nodes:   list[int] | None = None,
    load_nodes:      list[int] | None = None,
    force_safety_factor: float = 1.0,
) -> tuple[float, list[int], np.ndarray, dict]:
    """
    Single direct-FEM evaluation of structural feasibility for a MILP assignment.

    Returns
    -------
    feasibility_score : float [0,1] = 1 − n_unsafe / n_members   (D1)
    unsafe_member_ids : list[int]   members with UC ≥ 1
    member_uc         : np.ndarray[n_members]  per-member utilisation (the
                        "preds_physical" analogue)
    extra             : dict  {member_forces, n_unsafe, n_safe, solver_ok}
    """
    support_nodes = support_nodes if support_nodes is not None else _DEFAULT_SUPPORT_NODES
    load_nodes    = load_nodes    if load_nodes    is not None else _DEFAULT_LOAD_NODES

    n_mem = len(edges_v1)

    # section properties
    E_pa, A_m2 = _resolve_si_section(milp_assignment, df_input_stock, stock_df)
    raw = df_input_stock.iloc[milp_assignment]
    width_mm = raw["Width"].to_numpy(dtype=np.float64)
    depth_mm = raw["Depth"].to_numpy(dtype=np.float64)
    f_tk     = raw["f_tk"].to_numpy(dtype=np.float64)
    f_c0k    = raw["f_c0k"].to_numpy(dtype=np.float64)
    E_005    = raw["E_modulus_005"].to_numpy(dtype=np.float64)

    member_lengths_m = np.linalg.norm(
        node_positions[edges_v2] - node_positions[edges_v1], axis=1
    )
    EA = E_pa * A_m2

    forces, ok = solve_truss_axial_forces(
        node_positions, edges_v1, edges_v2, EA, support_nodes, load_nodes,
    )

    if not ok:
        # Unstable geometry / mechanism → maximally infeasible. The evaluator's
        # hard floor (structural_infeasibility > max_structural_infeas) penalises it.
        member_uc = np.full(n_mem, np.inf)
        unsafe_ids = list(range(n_mem))
        extra = {"member_forces": forces, "n_unsafe": n_mem, "n_safe": 0,
                 "solver_ok": False}
        return 0.0, unsafe_ids, member_uc, extra

    member_uc = ec5_member_utilisation(
        member_forces_n     = forces,
        member_lengths_m    = member_lengths_m,
        width_mm            = width_mm,
        depth_mm            = depth_mm,
        f_tk                = f_tk,
        f_c0k               = f_c0k,
        E_005               = E_005,
        force_safety_factor = force_safety_factor,
    )

    unsafe_flags      = member_uc >= 1.0
    unsafe_member_ids = np.where(unsafe_flags)[0].tolist()
    n_unsafe          = int(unsafe_flags.sum())
    feasibility_score = float(1.0 - n_unsafe / n_mem)                 # D1

    extra = {"member_forces": forces, "n_unsafe": n_unsafe,
             "n_safe": n_mem - n_unsafe, "solver_ok": True}
    return feasibility_score, unsafe_member_ids, member_uc, extra


# ORCHESTRATION — same return contract as run_gnn_stage ----------------------

def run_fem_stage(
    node_positions:  np.ndarray,
    milp_assignment: np.ndarray,
    df_input_stock:  pd.DataFrame,
    edges_df:        pd.DataFrame,
    stock_df:        "pd.DataFrame | None" = None,
    support_nodes:   list[int] | None = None,
    load_nodes:      list[int] | None = None,
    force_safety_factor: float = 1.0,
    print_summary:   bool = False,
) -> dict[str, Any]:
    """
    Drop-in structural feasibility stage using direct FEM.

    Returns a dict with the SAME keys consumed downstream as run_gnn_stage:
        feasibility_score, structural_penalty, unsafe_member_ids, preds_physical,
        n_unsafe, n_safe   (+ member_forces, member_uc, solver_ok for analysis).

    `edges_df` supplies connectivity (columns V1, V2) in the same row order as
    `milp_assignment` / df_slots. No GNN bundle is required.
    """
    edges_v1 = edges_df["V1"].to_numpy()
    edges_v2 = edges_df["V2"].to_numpy()

    feasibility_score, unsafe_member_ids, member_uc, extra = fem_feasibility(
        node_positions      = node_positions,
        milp_assignment     = milp_assignment,
        df_input_stock      = df_input_stock,
        edges_v1            = edges_v1,
        edges_v2            = edges_v2,
        stock_df            = stock_df,
        support_nodes       = support_nodes,
        load_nodes          = load_nodes,
        force_safety_factor = force_safety_factor,
    )

    n_unsafe = extra["n_unsafe"]
    n_safe   = extra["n_safe"]

    if print_summary:
        n_mem = len(edges_v1)
        print(f"\n[FEM] Feasibility Results ({n_mem} members, "
              f"{'OK' if extra['solver_ok'] else 'SINGULAR'} solve):")
        print(f"  Feasibility score:  {feasibility_score:.4f}  (1.0 = all safe)")
        print(f"  Safe members:       {n_safe} / {n_mem}")
        print(f"  Unsafe members:     {n_unsafe} / {n_mem}")
        if unsafe_member_ids:
            preview = unsafe_member_ids[:20]
            suffix  = "..." if n_unsafe > 20 else ""
            print(f"  Unsafe member IDs:  {preview}{suffix}")

    return {
        "feasibility_score":  feasibility_score,
        "structural_penalty": 0.0,            # parity with run_gnn_stage signature
        "unsafe_member_ids":  unsafe_member_ids,
        "preds_physical":     member_uc,      # per-member UC (analogue of P_unsafe)
        "member_uc":          member_uc,
        "member_forces":      extra["member_forces"],
        "n_unsafe":           n_unsafe,
        "n_safe":             n_safe,
        "solver_ok":          extra["solver_ok"],
    }
