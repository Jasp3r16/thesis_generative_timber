import c26_params
import numpy as np
import pandas as pd
from c25_feasibility_check import calculate_utilization_for_dataset

# ==========================================
# LCA COST MATRIX PARAMETERS
# ==========================================
M_A1_A3 = float(c26_params.IMPACT_FACTOR_A1_A3)
M_RECOVER = float(c26_params.IMPACT_FACTOR_RECOVERED_C1)
E_PREP_SAW = float(c26_params.ENERGY_PREP_SAW_A5)
E_OFFCUT = float(c26_params.ENERGY_OFFCUT_FACTOR_C3_C4)

SCARCITY_PENALTY = float(c26_params.SCARCITY_PENALTY)

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

    return df_stock

def calculate_cost_formula_v1(slot, stock_item, df_stock, weights=None, normalize=False):
    """
    Step 1: calculate C_{i,j} according to the LCA logic.
    C = E_embodied + E_prep + E_trans + E_waste + E_saw + E_opp.
    Returns (np.inf, None) if the match is physically infeasible.
    """
    l_stock = stock_item['Length'] / 1000.0
    a_stock = (stock_item['Width'] / 1000.0) * (stock_item['Depth'] / 1000.0)

    l_req = slot['Length_Req'] / 1000.0
    d_req = slot['Depth_Req'] / 1000.0
    w_req = slot['Width_Req'] / 1000.0
    a_req = w_req * d_req

    density = float(stock_item['Density_Resolved'])
    distance_km = float(stock_item['Distance_Resolved'])
    transport_factor = float(stock_item['TransportFactor_Resolved'])
    embodied_factor = float(stock_item['ECC_Resolved'])
    preparation_factor = float(stock_item['PreparationFacto_Resolved'])
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

    if normalize:
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

def calculate_cost_formula_v2(slot, stock_item):
    """
    Step 1: calculate C_{i,j} according to the LCA logic.
    state binary is used to distinguish between new and reclaimed element
    new and reclaimed elements have different calculation logic  
    """
    l_stock = stock_item['Length'] / 1000.0
    a_stock = (stock_item['Width'] / 1000.0) * (stock_item['Depth'] / 1000.0)

    l_req = slot['Length_Req'] / 1000.0
    d_req = slot['Depth_Req'] / 1000.0
    w_req = slot['Width_Req'] / 1000.0
    a_req = w_req * d_req

    density = float(stock_item['Density_Resolved'])
    distance_km = float(stock_item['Distance_Resolved'])
    transport_factor = float(stock_item['TransportFactor_Resolved'])
    trans_factor_km = transport_factor / 1000.0
    state = float(stock_item['State_Resolved'])

    # Explicit volume decomposition:
    # V_stock = V_req + V_over + V_waste
    v_req = a_req * l_req
    v_stock = a_stock * l_stock
    v_waste = max(0.0, a_stock * (l_stock - l_req))
    v_profile_target = a_stock * l_req
    v_over = max(0.0, v_profile_target - v_req)

    mass_req = v_req * density
    mass_stock = v_stock * density
    mass_waste = v_waste * density

    m_a1_a3 = M_A1_A3
    m_recover = M_RECOVER
    e_prep_saw = E_PREP_SAW
    e_offcut = E_OFFCUT
    scar_p = SCARCITY_PENALTY

    e_new = (mass_req * m_a1_a3) + (mass_req * distance_km * trans_factor_km)

    e_reclaimed = ((mass_stock * m_recover) + (mass_stock * distance_km * trans_factor_km) + 
                   (mass_stock * e_prep_saw) + (mass_waste * e_offcut) + (scar_p * v_waste))

    total_cost = (1-state) * e_new + state * e_reclaimed

    return total_cost, {
        'V_req': v_req,
        'V_over': v_over,
        'V_waste': v_waste,
        'V_stock': v_stock,
        'E_prep_saw': e_prep_saw,
        'E_offcut': e_offcut,
        'TOTAL_Score': total_cost
    }

# Main stage function
def build_cost_matrix(
    df_design,
    df_stock_raw,
    target_stock_ids=None,
):
    """
    Step 2: build the cost matrix with the assignment-cost function.
    """
    print("Starting generation of the integrated CO2 cost matrix (new LCA logic)...")

    df_stock = prepare_stock_cost_inputs(df_stock_raw)

    n_slots = len(df_design)
    n_stock = len(df_stock)

    cost_matrix = np.full((n_slots, n_stock), np.inf)

    # Store the detailed calculations here.
    detailed_logs = []

    for i in range(n_slots):
        slot = df_design.iloc[i]
        slot_id = slot['edge_id'] # of Element_ID, afhankelijk van je kolomnaam

        for j in range(n_stock):
            stock_item = df_stock.iloc[j]
            stock_id = stock_item['Member_ID']

                total_match_score, components = calculate_cost_formula_v2(slot, stock_item))

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
                        'V_req_m3': round(components['V_req'], 6),
                        'V_over_m3': round(components['V_over'], 6),
                        'V_waste_m3': round(components['V_waste'], 6),
                        'V_stock_m3': round(components['V_stock'], 6),
                        'Prep_CO2': round(components['E_prep'], 3),
                        'Trans_CO2': round(components['E_trans'], 3),
                        'Waste_CO2': round(components['E_waste'], 3),
                        'Saw_CO2': round(components['E_saw'], 4),
                        'TOTAL_Score': round(components['TOTAL_Score'], 3)
                    })
            else:
                if target_stock_ids and stock_id in target_stock_ids:
                    detailed_logs.append({
                        'Slot_ID': slot_id, 'Stock_ID': stock_id, 'Status': '❌',
                        'V_req_m3': '-', 'V_over_m3': '-', 'V_waste_m3': '-', 'V_stock_m3': '-',
                        'Embodied_CO2': '-', 'Prep_CO2': '-', 'Trans_CO2': '-', 'Waste_CO2': '-', 'Saw_CO2': '-',
                        'Opportunity_CO2': '-', 'TOTAL_Score': np.inf
                    })

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