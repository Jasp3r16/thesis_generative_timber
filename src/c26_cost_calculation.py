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

            all_reasons = _collect_feasibility_reasons(slot, stock_item, utilization_failed)
            feasibility_reason = '+'.join(all_reasons)

            if utilization_failed:
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

    return cost_matrix, df_stock, pd.DataFrame(detailed_logs)


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