import c11_params
import numpy as np
import pandas as pd
from c25_feasibility_check import calculate_utilization_for_dataset
from c25_feasibility_check import assign_roof_load_fz, geometry_df_to_design_row
from c21_surrogate_io import load_surrogate_bundle, predict_edge_forces_kn

# ==========================================
# LCA COST MATRIX PARAMETERS
# ==========================================
PREPARATION_FACTOR = float(c11_params.PREPARATION_EMISSION_FACTOR)
END_OF_LIFE_FACTOR = float(c11_params.END_OF_LIFE_EMISSION_FACTOR)
SAW_CUT_PENALTY = float(c11_params.SAW_CUT_PENALTY)

def _resolve_edge_columns(df_edges):
    columns_by_lower = {str(col).strip().lower(): col for col in df_edges.columns}
    edge_id_col = columns_by_lower.get('edge_id')
    area_col = None
    for candidate in ('area', 'cross_section_area', 'a'):
        if candidate in columns_by_lower:
            area_col = columns_by_lower[candidate]
            break
    return edge_id_col, area_col

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

# Cost formula components and normalization
def normalize_cost_formula_values(cost_components):
    """Each independent penalty is calculated across all possible assignments to generate separate, raw matrices 
    (e.g., $C^{saw}$, $C^{opp}$, $C^{waste}$). Before aggregation, each matrix is individually normalized using Min-Max scaling, 
    mapping all values to a bounded interval of $[0, 1]$:$C^{norm}_{ij} = \frac{C_{ij} - \min(C)}{\max(C) - \min(C)}$
    Through this mathematical translation, the worst possible sawing penalty in the matrix becomes exactly $1.0$, 
    and the worst possible scarcity penalty also becomes exactly $1.0$. The values are now dimensionally uniform, 
    preventing the macro-penalties from mathematically eclipsing the micro-penalties."""
    normalized_components = {}
    for key, matrix in cost_components.items():
        min_val = np.nanmin(matrix)
        max_val = np.nanmax(matrix)
        if np.isfinite(min_val) and np.isfinite(max_val) and max_val > min_val:
            normalized_matrix = (matrix - min_val) / (max_val - min_val)
            normalized_components[key] = normalized_matrix
    return normalized_components    

def calculate_scarcity_weight(df_stock, stock_item):
    """Compute a scarcity ratio for a stock item based on availability by length category.

    Stock elements are grouped into 500 mm length bins. The scarcity ratio is:

        1 - (count_in_same_length_bin / total_stock_count)

    Higher values indicate that the item's length category is less common in the
    inventory (more scarce). If the stock table is empty, the function returns 0.0.
    """
    length_bin_size_mm = 500
    length_bin = int(float(stock_item['Length']) // length_bin_size_mm) * length_bin_size_mm
    category_count = sum(
        1 for _, item in df_stock.iterrows()
        if int(float(item['Length']) // length_bin_size_mm) * length_bin_size_mm == length_bin
    )
    total_count = len(df_stock)
    scarcity_ratio = 1.0 - (category_count / total_count) if total_count > 0 else 0.0
    return scarcity_ratio

def calculate_cost_formula_v1(slot, stock_item, df_stock, weights=None):
    """
    Step 1: calculate C_{i,j} according to the LCA logic.
    C = E_embodied + E_prep + E_trans + E_waste + E_saw + E_opp.
    Returns (np.inf, None) if the match is physically infeasible.
    """
    l_stock = stock_item['Length'] / 1000.0
    a_stock = (stock_item['Width'] / 1000.0) * (stock_item['Depth'] / 1000.0)

    l_req = slot['Length_Req'] / 1000.0
    d_req = 

    # Hard feasibility rule: insufficient length or cross-section area -> impossible match.
    if l_stock < l_req or a_stock < a_req:
        return np.inf, None

    density = float(stock_item['Density_Resolved'])
    distance_km = float(stock_item['Distance_Resolved'])
    transport_factor = float(stock_item['TransportFactor_Resolved'])
    embodied_factor = float(stock_item['ECC_Resolved'])
    preparation_factor = float(stock_item['PreparationFactor_Resolved'])
    scarcity_ratio = calculate_scarcity_weight(df_stock, stock_item)

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
    e_opp = scarcity_ratio*(v_over + v_waste) * embodied_factor

    norm_comp = normalize_cost_formula_values(
        cost_components={
            'embodied_energy': e_embodied,
            'preparation_energy': e_prep,
            'transportation_energy': e_trans,
            'waste_energy': e_waste,
            'sawing_energy': e_saw,
            'opportunity_cost': e_opp,
        })
    
    norm_e_embodied = norm_comp.get('embodied_energy', e_embodied)
    norm_e_prep = norm_comp.get('preparation_energy', e_prep)
    norm_e_trans = norm_comp.get('transportation_energy', e_trans)
    norm_e_waste = norm_comp.get('waste_energy', e_waste)
    norm_e_saw = norm_comp.get('sawing_energy', e_saw)
    norm_e_opp = norm_comp.get('opportunity_cost', e_opp)

    if weights is not None:
        w_embodied = weights.get('embodied_energy', 0.0)
        w_prep = weights.get('preparation_energy', 0.0)
        w_trans = weights.get('transportation_energy', 0.0)
        w_waste = weights.get('waste_energy', 0.0)
        w_saw = weights.get('sawing_energy', 0.0)
        w_opp = weights.get('opportunity_cost', 0.0)

    total_cost = (w_embodied * norm_e_embodied + w_prep * norm_e_prep + w_trans * norm_e_trans +
                  w_waste * norm_e_waste + w_saw * norm_e_saw + w_opp * norm_e_opp)

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

# Main stage function
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
    weights=None,
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
                total_match_score, components = calculate_cost_formula_v1(slot, stock_item, df_stock, weights)

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