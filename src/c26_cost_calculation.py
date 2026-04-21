import c11_params
import numpy as np
import pandas as pd
from c25_structural_check import calculate_utilization_for_dataset
from c25_structural_check import assign_roof_load_fz, geometry_df_to_design_row
from c21_surrogate_io import load_surrogate_bundle, predict_edge_forces_kn

# ==========================================
# LCA COST MATRIX PARAMETERS
# ==========================================
PREPARATION_FACTOR = float(c11_params.PREPARATION_EMISSION_FACTOR)
END_OF_LIFE_FACTOR = float(c11_params.END_OF_LIFE_EMISSION_FACTOR)
SAW_CUT_PENALTY = float(c11_params.SAW_CUT_PENALTY)


def _resolve_utilization_value(df_utilization_matrix, slot_id, stock_id):
    """Fetch utilization from the matrix; return np.nan if the combination is missing."""
    if df_utilization_matrix is None:
        return np.nan
    if slot_id not in df_utilization_matrix.index:
        return np.nan
    if stock_id not in df_utilization_matrix.columns:
        return np.nan
    return df_utilization_matrix.loc[slot_id, stock_id]


def _resolve_edge_columns(df_edges):
    columns_by_lower = {str(col).strip().lower(): col for col in df_edges.columns}
    edge_id_col = columns_by_lower.get('edge_id')
    area_col = None
    for candidate in ('area', 'cross_section_area', 'a'):
        if candidate in columns_by_lower:
            area_col = columns_by_lower[candidate]
            break
    return edge_id_col, area_col


def _prepare_surrogate_candidate_context(surrogate_context):
    if surrogate_context is None:
        return None

    df_vertices = surrogate_context.get('df_vertices')
    df_edges = surrogate_context.get('df_edges')
    bundle = surrogate_context.get('bundle')
    model_prefix = surrogate_context.get('model_prefix')

    if df_vertices is None or df_edges is None:
        raise ValueError('surrogate_context requires df_vertices and df_edges.')

    edge_id_col, area_col = _resolve_edge_columns(df_edges)
    if area_col is None:
        raise ValueError(
            'surrogate_context df_edges must include a real area column (Area/cross_section_area/A). '
            'No synthetic area fallback is allowed.'
        )

    df_edges_base = df_edges.copy().reset_index(drop=True)
    if edge_id_col is None:
        df_edges_base['edge_id'] = [f'e{i}' for i in range(len(df_edges_base))]
        edge_id_col = 'edge_id'
    else:
        df_edges_base[edge_id_col] = df_edges_base[edge_id_col].astype(str)

    df_edges_base[area_col] = pd.to_numeric(df_edges_base[area_col], errors='coerce')
    if df_edges_base[area_col].isna().any():
        raise ValueError('surrogate_context df_edges contains empty area values; cannot run strict surrogate inference.')

    if bundle is None:
        bundle = load_surrogate_bundle(prefix_sm=model_prefix)

    df_geometry = assign_roof_load_fz(df_vertices.copy().reset_index(drop=True), roof_load_kn_m2=2.0)

    return {
        'bundle': bundle,
        'df_geometry': df_geometry,
        'df_edges_base': df_edges_base,
        'edge_id_col': edge_id_col,
        'area_col': area_col,
        'force_cache': {},
    }


def _predict_candidate_force_kn(prepared_context, slot_edge_id, candidate_area_m2):
    cache_key = (str(slot_edge_id).strip().lower(), float(round(candidate_area_m2, 10)))
    force_cache = prepared_context['force_cache']
    if cache_key in force_cache:
        return force_cache[cache_key]

    edge_id_col = prepared_context['edge_id_col']
    area_col = prepared_context['area_col']
    df_edges_candidate = prepared_context['df_edges_base'].copy()
    normalized_edge_id = str(slot_edge_id).strip().lower()

    edge_ids_normalized = df_edges_candidate[edge_id_col].astype(str).str.strip().str.lower()
    mask = edge_ids_normalized == normalized_edge_id
    if not mask.any():
        raise ValueError(f"Edge '{slot_edge_id}' not found in surrogate_context df_edges.")

    df_edges_candidate.loc[mask, area_col] = float(candidate_area_m2)

    design_row = geometry_df_to_design_row(
        df_geometry=prepared_context['df_geometry'],
        df_edges=df_edges_candidate,
    )
    df_forces = predict_edge_forces_kn(design_row, prepared_context['bundle'])
    forces_edge_norm = df_forces['edge_id'].astype(str).str.strip().str.lower()
    force_match = df_forces.loc[forces_edge_norm == normalized_edge_id, 'axial_force_kn']
    if force_match.empty:
        raise ValueError(f"Surrogate output did not contain force for edge '{slot_edge_id}'.")

    predicted_force = float(force_match.iloc[0])
    force_cache[cache_key] = predicted_force
    return predicted_force


def _build_utilization_matrix_from_slot_forces(df_design, df_stock, gnn_margin=1.10):
    """Build a utilization matrix directly in c26 from slot force demand.

    Requires `edge_id`, `axial_force_kn`, and `length_m` columns in `df_design`, and
    stock strength/geometry columns required by `calculate_utilization_for_dataset`.
    """
    required_slot_cols = ['edge_id', 'axial_force_kn', 'length_m']
    required_stock_cols = ['Member_ID', 'Width', 'Depth', 'f_c0k', 'f_tk', 'E_modulus_eff']

    if not all(col in df_design.columns for col in required_slot_cols):
        return None
    if not all(col in df_stock.columns for col in required_stock_cols):
        return None

    df_slots_local = df_design.copy()
    df_slots_local['axial_force_kn'] = pd.to_numeric(df_slots_local['axial_force_kn'], errors='coerce')
    df_slots_local['length_m'] = pd.to_numeric(df_slots_local['length_m'], errors='coerce')

    if df_slots_local[['axial_force_kn', 'length_m']].dropna().empty:
        return None

    records = []
    for _, slot_row in df_slots_local.iterrows():
        slot_id = str(slot_row['edge_id'])
        req_force = slot_row['axial_force_kn']
        req_length = slot_row['length_m']

        if not np.isfinite(req_force) or not np.isfinite(req_length):
            continue

        for _, stock_row in df_stock.iterrows():
            stock_id = stock_row['Member_ID']
            util = calculate_utilization_for_dataset(
                stock_row,
                req_force_kn=float(req_force),
                req_length_m=float(req_length),
                gnn_margin=float(gnn_margin),
            )
            records.append({
                'edge_id': slot_id,
                'Member_ID': stock_id,
                'utilization': float(util),
            })

    if not records:
        return None

    matrix = pd.DataFrame.from_records(records).pivot_table(
        index='edge_id',
        columns='Member_ID',
        values='utilization',
        aggfunc='first',
    )
    return matrix


def prepare_stock_cost_inputs(df_stock_raw):
    """
    Prepare a strict input table for cost calculation.
    Required data must be present in the stock CSV; no numeric fallbacks are used.
    """
    df_stock = df_stock_raw.copy()

    required_columns = ['mean_density', 'Transport_Dist', 'EmissionFactor', 'ProcessingFactor', 'ECC']
    missing_columns = [col for col in required_columns if col not in df_stock.columns]
    if missing_columns:
        raise ValueError(f"Missing required stock columns: {missing_columns}")

    null_columns = [col for col in required_columns if df_stock[col].isna().any()]
    if null_columns:
        raise ValueError(f"Empty values found in required stock columns: {null_columns}")

    df_stock['Density_Resolved'] = df_stock['mean_density'].astype(float)
    df_stock['Distance_Resolved'] = df_stock['Transport_Dist'].astype(float)
    df_stock['TransportFactor_Resolved'] = df_stock['EmissionFactor'].astype(float)
    df_stock['ProcessingFactor_Resolved'] = df_stock['ProcessingFactor'].astype(float)
    df_stock['ECC_Resolved'] = df_stock['ECC'].astype(float)
    df_stock['PreparationFactor_Resolved'] = df_stock['ProcessingFactor_Resolved'] * PREPARATION_FACTOR

    return df_stock


def _get_required_area(slot, a_stock):
    """Return required cross-section area in m2, falling back to stock area when unavailable."""
    if (
        'Width_Req' in slot and 'Depth_Req' in slot
        and pd.notna(slot['Width_Req']) and pd.notna(slot['Depth_Req'])
    ):
        return (float(slot['Width_Req']) / 1000.0) * (float(slot['Depth_Req']) / 1000.0)
    return a_stock


def _classify_geometry_constraint(slot, stock_item):
    """Classify infeasibility cause as length, dimensions, both, or passed."""
    l_stock = float(stock_item['Length']) / 1000.0
    a_stock = (float(stock_item['Width']) / 1000.0) * (float(stock_item['Depth']) / 1000.0)
    l_req = float(slot['Length_Req']) / 1000.0
    a_req = _get_required_area(slot, a_stock)

    length_failed = l_stock < l_req
    dimensions_failed = a_stock < a_req

    if length_failed and dimensions_failed:
        return 'Length+Dimensions'
    if length_failed:
        return 'Length'
    if dimensions_failed:
        return 'Dimensions'
    return 'Passed'


def _collect_feasibility_reasons(slot, stock_item, utilization_failed):
    """Collect all active feasibility constraints for this slot-stock combination."""
    reasons = []
    if utilization_failed:
        reasons.append('Utilization')

    geometry_reason = _classify_geometry_constraint(slot, stock_item)
    if geometry_reason == 'Length+Dimensions':
        reasons.extend(['Length', 'Dimensions'])
    elif geometry_reason in ('Length', 'Dimensions'):
        reasons.append(geometry_reason)

    return reasons if reasons else ['Passed']


def _evaluate_candidate_feasibility(slot, stock_item, utilization_value, max_utilization_threshold, utilization_active):
    """Evaluate the three feasibility checks for one slot-stock candidate.

    Returns a compact summary that can be used to prune infeasible candidates
    before the more expensive cost formula is evaluated.
    """
    l_stock = float(stock_item['Length']) / 1000.0
    a_stock = (float(stock_item['Width']) / 1000.0) * (float(stock_item['Depth']) / 1000.0)

    l_req = float(slot['Length_Req']) / 1000.0
    a_req = _get_required_area(slot, a_stock)

    length_feasible = l_stock >= l_req
    dimensions_feasible = a_stock >= a_req

    utilization_feasible = True
    if utilization_active:
        utilization_feasible = np.isfinite(utilization_value) and float(utilization_value) <= float(max_utilization_threshold)
    geometry_reason = _classify_geometry_constraint(slot, stock_item)
    reasons = _collect_feasibility_reasons(slot, stock_item, utilization_active and (not utilization_feasible))

    return {
        'length_feasible': bool(length_feasible),
        'dimensions_feasible': bool(dimensions_feasible),
        'utilization_feasible': bool(utilization_feasible),
        'structural_feasible': bool(length_feasible and dimensions_feasible and utilization_feasible),
        'geometry_reason': geometry_reason,
        'reasons': reasons,
    }


def calculate_cost_formula(slot, stock_item):
    """
    Step 1: calculate C_{i,j} according to the LCA logic.
    C = E_embodied + E_prep + E_trans + E_waste + E_saw + E_opp.
    Returns (np.inf, None) if the match is physically infeasible.
    """
    l_stock = stock_item['Length'] / 1000.0
    a_stock = (stock_item['Width'] / 1000.0) * (stock_item['Depth'] / 1000.0)

    l_req = slot['Length_Req'] / 1000.0
    a_req = _get_required_area(slot, a_stock)

    # Hard feasibility rule: insufficient length or cross-section area -> impossible match.
    if l_stock < l_req or a_stock < a_req:
        return np.inf, None

    density = float(stock_item['Density_Resolved'])
    distance_km = float(stock_item['Distance_Resolved'])
    transport_factor = float(stock_item['TransportFactor_Resolved'])
    embodied_factor = float(stock_item['ECC_Resolved'])
    preparation_factor = float(stock_item['PreparationFactor_Resolved'])

    # Explicit volume decomposition:
    # V_stock = V_req + V_over + V_waste
    v_req = a_req * l_req
    v_profile_target = a_stock * l_req
    v_over = max(0.0, v_profile_target - v_req)
    v_waste = max(0.0, a_stock * (l_stock - l_req))
    v_stock = a_stock * l_stock

    e_embodied = v_stock * embodied_factor
    e_prep = v_stock * preparation_factor
    e_trans = (((v_req + v_over) * density) / 1000.0) * distance_km * transport_factor
    e_waste = v_waste * END_OF_LIFE_FACTOR
    e_saw = 0.0 if stock_item['Length'] == slot['Length_Req'] else SAW_CUT_PENALTY
    e_opp = 0.0  # Opportunity cost is currently not implemented; can be added here if needed.

    total_cost = e_embodied + e_prep + e_trans + e_waste + e_saw + e_opp

    return total_cost, {
        'V_req': v_req,
        'V_over': v_over,
        'V_waste': v_waste,
        'V_stock': v_stock,
        'E_embodied': e_embodied,
        'E_prep': e_prep,
        'E_trans': e_trans,
        'E_waste': e_waste,
        'E_saw': e_saw,
        'E_opp': e_opp,
        'TOTAL_Score': total_cost
    }

def build_cost_matrix(
    df_design,
    df_stock_raw,
    target_stock_ids=None,
    df_utilization_matrix=None,
    max_utilization_threshold=1.0,
    gnn_margin=1.10,
    surrogate_context=None,
    require_structural_constraints=True,
    require_surrogate_when_context=True,
):
    """
    Step 2: build the cost matrix with the assignment-cost function.
    """
    print("Starting generation of the integrated CO2 cost matrix (new LCA logic)...")

    df_stock = prepare_stock_cost_inputs(df_stock_raw)

    n_slots = len(df_design)
    n_stock = len(df_stock)

    cost_matrix = np.full((n_slots, n_stock), np.inf)
    successful_matches = 0

    # Store the detailed calculations here.
    detailed_logs = []

    utilization_mode = 'external_matrix'
    utilization_matrix_active = df_utilization_matrix
    surrogate_candidate_context = None
    if utilization_matrix_active is None:
        utilization_matrix_active = _build_utilization_matrix_from_slot_forces(
            df_design=df_design,
            df_stock=df_stock,
            gnn_margin=float(gnn_margin),
        )
        if utilization_matrix_active is not None:
            utilization_mode = 'slot_force_derived'
        elif surrogate_context is not None:
            surrogate_candidate_context = _prepare_surrogate_candidate_context(surrogate_context)
            utilization_mode = 'candidate_surrogate'
        else:
            utilization_mode = 'inactive'

    if require_surrogate_when_context and surrogate_context is not None and utilization_mode != 'candidate_surrogate':
        raise RuntimeError(
            "surrogate_context was provided but candidate-level surrogate mode was not activated. "
            f"Resolved utilization_mode='{utilization_mode}'."
        )

    if require_structural_constraints and utilization_mode == 'inactive':
        raise RuntimeError(
            "Structural constraints are inactive in cost-matrix generation. "
            "Provide either df_utilization_matrix, slot force demand, or surrogate_context."
        )

    util_constraint_active = utilization_matrix_active is not None

    for i in range(n_slots):
        slot = df_design.iloc[i]
        slot_id = slot['edge_id'] # of Element_ID, afhankelijk van je kolomnaam

        for j in range(n_stock):
            stock_item = df_stock.iloc[j]
            stock_id = stock_item['Member_ID']
            utilization_value = _resolve_utilization_value(utilization_matrix_active, slot_id, stock_id)

            if surrogate_candidate_context is not None:
                candidate_area_m2 = (float(stock_item['Width']) / 1000.0) * (float(stock_item['Depth']) / 1000.0)
                predicted_force_kn = _predict_candidate_force_kn(
                    prepared_context=surrogate_candidate_context,
                    slot_edge_id=slot_id,
                    candidate_area_m2=candidate_area_m2,
                )
                utilization_value = calculate_utilization_for_dataset(
                    stock_item,
                    req_force_kn=float(predicted_force_kn),
                    req_length_m=float(slot['length_m']) if 'length_m' in slot else float(slot['Length_Req']) / 1000.0,
                    gnn_margin=float(gnn_margin),
                )

            candidate_check = _evaluate_candidate_feasibility(
                slot=slot,
                stock_item=stock_item,
                utilization_value=utilization_value,
                max_utilization_threshold=max_utilization_threshold,
                utilization_active=util_constraint_active,
            )

            all_reasons = candidate_check['reasons']

            if not candidate_check['structural_feasible']:
                total_match_score, components = np.inf, None
            else:
                total_match_score, components = calculate_cost_formula(slot, stock_item)

            if np.isfinite(total_match_score):
                cost_matrix[i, j] = total_match_score
                successful_matches += 1

                # Extensive logging for the detailed analysis.
                if target_stock_ids and stock_id in target_stock_ids:
                    assert components is not None
                    detailed_logs.append({
                        'Slot_ID': slot_id,
                        'Stock_ID': stock_id,
                        'Status': '✅',
                        'Feasibility_Reasons': ', '.join(all_reasons),
                        'Utilization_Value': round(float(utilization_value), 4) if np.isfinite(utilization_value) else '-',
                        'V_req_m3': round(components['V_req'], 6),
                        'V_over_m3': round(components['V_over'], 6),
                        'V_waste_m3': round(components['V_waste'], 6),
                        'V_stock_m3': round(components['V_stock'], 6),
                        'Embodied_CO2': round(components['E_embodied'], 3),
                        'Prep_CO2': round(components['E_prep'], 3),
                        'Trans_CO2': round(components['E_trans'], 3),
                        'Waste_CO2': round(components['E_waste'], 3),
                        'Saw_CO2': round(components['E_saw'], 4),
                        'Opportunity_CO2': round(components['E_opp'], 3),
                        'TOTAL_Score': round(components['TOTAL_Score'], 3)
                    })
            else:
                if target_stock_ids and stock_id in target_stock_ids:
                    detailed_logs.append({
                        'Slot_ID': slot_id, 'Stock_ID': stock_id, 'Status': '❌',
                        'Feasibility_Reasons': ', '.join(all_reasons),
                        'Utilization_Value': round(float(utilization_value), 4) if np.isfinite(utilization_value) else '-',
                        'V_req_m3': '-', 'V_over_m3': '-', 'V_waste_m3': '-', 'V_stock_m3': '-',
                        'Embodied_CO2': '-', 'Prep_CO2': '-', 'Trans_CO2': '-', 'Waste_CO2': '-', 'Saw_CO2': '-',
                        'Opportunity_CO2': '-', 'TOTAL_Score': np.inf
                    })

    print(f"Matrix generated! Dimensions: {n_slots} required members x {n_stock} inventory beams.")
    print(f"Physical valid combinations found: {successful_matches}")
    if util_constraint_active:
        print(f"Utilization constraint active with threshold <= {float(max_utilization_threshold):.3f}")
        print(f"Utilization source: {utilization_mode}")
    else:
        print("Utilization constraint inactive (no external matrix and no slot-force-derived matrix).")

    logs_df = pd.DataFrame(detailed_logs)
    logs_df.attrs['utilization_mode'] = utilization_mode
    return cost_matrix, df_stock, logs_df


def analyze_and_export_slot_logs(
    df_logs,
    target_slot_for_analysis,
    all_stock_ids,
    export_dir,
    display_fn=None,
    max_full_list_rows=None,
    show_full_list=True,
):
    """
    Prepare the detailed analysis for a specific slot ID, display tables, and export CSV.

    Returns:
        (df_logs_slot, df_logs_slot_rs, analysis_export_path)
    """
    print("\n" + "=" * 80)
    print(f"DETAILED ANALYSIS: ALL FACTORS FOR SLOT {target_slot_for_analysis}")
    print("=" * 80)

    analysis_export_path = export_dir / f"depth_analysis_{target_slot_for_analysis}.csv"

    if df_logs.empty:
        print("No log data found.")
        print("=" * 80)
        return pd.DataFrame(), pd.DataFrame(), analysis_export_path

    df_logs = df_logs.sort_values(by=['Slot_ID', 'Stock_ID']).reset_index(drop=True)
    slot_mask = df_logs['Slot_ID'].astype(str).str.strip().str.lower() == str(target_slot_for_analysis).lower()
    df_logs_slot = df_logs.loc[slot_mask].copy()

    # Ensure all stock items are visible for this slot, even if they are missing from the logs.
    df_all_stock = pd.DataFrame({'Stock_ID': all_stock_ids}).drop_duplicates()
    df_logs_slot = df_all_stock.merge(df_logs_slot, on='Stock_ID', how='left')

    if 'Slot_ID' in df_logs_slot.columns:
        df_logs_slot['Slot_ID'] = df_logs_slot['Slot_ID'].fillna(target_slot_for_analysis)
    if 'Status' in df_logs_slot.columns:
        df_logs_slot['Status'] = df_logs_slot['Status'].fillna('is not in logs')

    rs_mask = df_logs_slot['Stock_ID'].astype(str).str.contains('RS', case=False, na=False)
    df_logs_slot_rs = df_logs_slot.loc[rs_mask].copy()

    print(f"\nAmount of RS-items for {target_slot_for_analysis}: {len(df_logs_slot_rs)}")
    if not df_logs_slot_rs.empty:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
            if display_fn is not None:
                display_fn(df_logs_slot_rs.fillna('-').reset_index(drop=True))
            else:
                print(df_logs_slot_rs.fillna('-').reset_index(drop=True).to_string(index=False))
    else:
        print("No RS items found in the input stock list.")

    if show_full_list:
        df_logs_slot_full = df_logs_slot.fillna('-').reset_index(drop=True)
        if max_full_list_rows is not None:
            max_rows = int(max_full_list_rows)
            if max_rows < 0:
                max_rows = 0
            df_logs_slot_display = df_logs_slot_full.head(max_rows)
            print(f"\nFull list (NS + RS) - showing first {max_rows} rows:")
        else:
            df_logs_slot_display = df_logs_slot_full
            print("\nFull list (NS + RS):")

        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
            if display_fn is not None:
                display_fn(df_logs_slot_display)
            else:
                print(df_logs_slot_display.to_string(index=False))
    else:
        print("\nFull list (NS + RS) skipped (testing mode is off).")

    df_logs_slot.to_csv(analysis_export_path, index=False)
    print(f"\nDetailed analysis exported to: {analysis_export_path}")
    print("=" * 80)

    return df_logs_slot, df_logs_slot_rs, analysis_export_path