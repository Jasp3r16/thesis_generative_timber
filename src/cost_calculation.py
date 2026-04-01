import c11_params
import numpy as np
import pandas as pd

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


def prepare_stock_cost_inputs(df_stock_raw):
    """
    Prepare a strict input table for cost calculation.
    Required data must be present in the stock CSV; no numeric fallbacks are used.
    """
    df_stock = df_stock_raw.copy()

    required_columns = ['mean_density', 'Transport_Dist', 'Emmisiefactor', 'Bewerkingsfactor', 'ECC']
    missing_columns = [col for col in required_columns if col not in df_stock.columns]
    if missing_columns:
        raise ValueError(f"Missing required stock columns: {missing_columns}")

    null_columns = [col for col in required_columns if df_stock[col].isna().any()]
    if null_columns:
        raise ValueError(f"Empty values found in required stock columns: {null_columns}")

    df_stock['Density_Resolved'] = df_stock['mean_density'].astype(float)
    df_stock['Distance_Resolved'] = df_stock['Transport_Dist'].astype(float)
    df_stock['TransportFactor_Resolved'] = df_stock['Emmisiefactor'].astype(float)
    df_stock['Bewerkingsfactor_Resolved'] = df_stock['Bewerkingsfactor'].astype(float)
    df_stock['ECC_Resolved'] = df_stock['ECC'].astype(float)
    df_stock['PreparationFactor_Resolved'] = df_stock['Bewerkingsfactor_Resolved'] * PREPARATION_FACTOR

    return df_stock


def calculate_lca_formula(slot, stock_item):
    """
    Step 1: calculate C_{i,j} according to the LCA logic.
    C = E_embodied + E_prep + E_trans + E_waste + E_saw.
    Returns (np.inf, None) if the match is physically infeasible.
    """
    l_stock = stock_item['Length'] / 1000.0
    a_stock = (stock_item['Width'] / 1000.0) * (stock_item['Depth'] / 1000.0)

    l_req = slot['Length_Req'] / 1000.0
    # Backward-compatible fallback for workflows where section demand is not modeled per slot.
    if 'Width_Req' in slot and 'Depth_Req' in slot:
        a_req = (slot['Width_Req'] / 1000.0) * (slot['Depth_Req'] / 1000.0)
    else:
        a_req = a_stock

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

    total_cost = e_embodied + e_prep + e_trans + e_waste + e_saw

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
        'TOTAL_Score': total_cost
    }


def calculate_assignment_cost(slot, stock_item):
    """Backward-compatible alias voor de formulefunctie."""
    return calculate_lca_formula(slot, stock_item)

print("Calculation modules defined successfully.")

def build_cost_matrix(
    df_design,
    df_stock_raw,
    target_stock_ids=None,
    df_utilization_matrix=None,
    max_utilization_threshold=1.0,
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

    util_constraint_active = df_utilization_matrix is not None

    for i in range(n_slots):
        slot = df_design.iloc[i]
        slot_id = slot['edge_id'] # of Element_ID, afhankelijk van je kolomnaam

        for j in range(n_stock):
            stock_item = df_stock.iloc[j]
            stock_id = stock_item['Member_ID']
            utilization_value = _resolve_utilization_value(df_utilization_matrix, slot_id, stock_id)

            utilization_failed = False
            if util_constraint_active:
                if not np.isfinite(utilization_value):
                    utilization_failed = True
                elif float(utilization_value) > float(max_utilization_threshold):
                    utilization_failed = True

            if utilization_failed:
                total_match_score, components = np.inf, None
                feasibility_reason = 'Utilization'
            else:
                total_match_score, components = calculate_lca_formula(slot, stock_item)
                feasibility_reason = 'Passed' if np.isfinite(total_match_score) else 'GeometryOrLCA'

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
                        'Feasibility_Check': feasibility_reason,
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
                        'TOTAL_Score': round(components['TOTAL_Score'], 3)
                    })
            else:
                if target_stock_ids and stock_id in target_stock_ids:
                    detailed_logs.append({
                        'Slot_ID': slot_id, 'Stock_ID': stock_id, 'Status': '❌',
                        'Feasibility_Check': feasibility_reason,
                        'Utilization_Value': round(float(utilization_value), 4) if np.isfinite(utilization_value) else '-',
                        'V_req_m3': '-', 'V_over_m3': '-', 'V_waste_m3': '-', 'V_stock_m3': '-',
                        'Embodied_CO2': '-', 'Prep_CO2': '-', 'Trans_CO2': '-', 'Waste_CO2': '-', 'Saw_CO2': '-',
                        'TOTAL_Score': np.inf
                    })

    print(f"Matrix generated! Dimensions: {n_slots} required members x {n_stock} inventory beams.")
    print(f"Physical valid combinations found: {successful_matches}")
    if util_constraint_active:
        print(f"Utilization constraint active with threshold <= {float(max_utilization_threshold):.3f}")

    return cost_matrix, df_stock, pd.DataFrame(detailed_logs)


def analyze_and_export_slot_logs(df_logs, target_slot_for_analysis, all_stock_ids, export_dir, display_fn=None):
    """
    Prepare the detailed analysis for a specific slot ID, display tables, and export CSV.

    Returns:
        (df_logs_slot, df_logs_slot_rs, analysis_export_path)
    """
    print("\n" + "=" * 80)
    print(f"DETAILED ANALYSIS: ALL FACTORS FOR SLOT {target_slot_for_analysis}")
    print("=" * 80)

    analysis_export_path = export_dir / f"diepte_analyse_{target_slot_for_analysis}.csv"

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
        df_logs_slot['Status'] = df_logs_slot['Status'].fillna('⚠️ ontbreekt in log')

    rs_mask = df_logs_slot['Stock_ID'].astype(str).str.contains('RS', case=False, na=False)
    df_logs_slot_rs = df_logs_slot.loc[rs_mask].copy()

    print(f"\nAantal RS-items voor {target_slot_for_analysis}: {len(df_logs_slot_rs)}")
    if not df_logs_slot_rs.empty:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
            if display_fn is not None:
                display_fn(df_logs_slot_rs.fillna('-').reset_index(drop=True))
            else:
                print(df_logs_slot_rs.fillna('-').reset_index(drop=True).to_string(index=False))
    else:
        print("No RS items found in the input stock list.")

    print("\nFull list (NS + RS):")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        if display_fn is not None:
            display_fn(df_logs_slot.fillna('-').reset_index(drop=True))
        else:
            print(df_logs_slot.fillna('-').reset_index(drop=True).to_string(index=False))

    df_logs_slot.to_csv(analysis_export_path, index=False)
    print(f"\nDetailed analysis exported to: {analysis_export_path}")
    print("=" * 80)

    return df_logs_slot, df_logs_slot_rs, analysis_export_path