import c11_params
import numpy as np
import pandas as pd

# ==========================================
# LCA COST MATRIX PARAMETERS
# ==========================================
PREPARATION_FACTOR = float(c11_params.PREPARATION_EMISSION_FACTOR)
END_OF_LIFE_FACTOR = float(c11_params.END_OF_LIFE_EMISSION_FACTOR)
SAW_CUT_PENALTY = float(c11_params.SAW_CUT_PENALTY)


def prepare_stock_cost_inputs(df_stock_raw):
    """
    Maakt een strikte inputtabel voor kostenberekening.
    Vereiste data moet aanwezig zijn in de stock CSV; er worden geen numerieke fallbacks gebruikt.
    """
    df_stock = df_stock_raw.copy()

    required_columns = ['mean_density', 'Transport_Dist', 'Emmisiefactor', 'Bewerkingsfactor', 'ECC']
    missing_columns = [col for col in required_columns if col not in df_stock.columns]
    if missing_columns:
        raise ValueError(f"Ontbrekende verplichte stock-kolommen: {missing_columns}")

    null_columns = [col for col in required_columns if df_stock[col].isna().any()]
    if null_columns:
        raise ValueError(f"Lege waarden gevonden in verplichte stock-kolommen: {null_columns}")

    df_stock['Density_Resolved'] = df_stock['mean_density'].astype(float)
    df_stock['Distance_Resolved'] = df_stock['Transport_Dist'].astype(float)
    df_stock['TransportFactor_Resolved'] = df_stock['Emmisiefactor'].astype(float)
    df_stock['Bewerkingsfactor_Resolved'] = df_stock['Bewerkingsfactor'].astype(float)
    df_stock['ECC_Resolved'] = df_stock['ECC'].astype(float)
    df_stock['PreparationFactor_Resolved'] = df_stock['Bewerkingsfactor_Resolved'] * PREPARATION_FACTOR

    return df_stock


def calculate_lca_formula(slot, stock_item):
    """
    Stap 1: berekent C_{i,j} volgens de LCA-logica.
    C = E_embodied + E_prep + E_trans + E_waste + E_saw.
    Retourneert (np.inf, None) als de match fysiek niet haalbaar is.
    """
    l_req = slot['Length_Req'] / 1000.0
    a_req = (slot['Width_Req'] / 1000.0) * (slot['Depth_Req'] / 1000.0)

    l_stock = stock_item['Length'] / 1000.0
    a_stock = (stock_item['Width'] / 1000.0) * (stock_item['Depth'] / 1000.0)

    # Hard feasibility rule: onvoldoende lengte of doorsnede-oppervlak -> onmogelijke match.
    if l_stock < l_req or a_stock < a_req:
        return np.inf, None

    density = float(stock_item['Density_Resolved'])
    distance_km = float(stock_item['Distance_Resolved'])
    transport_factor = float(stock_item['TransportFactor_Resolved'])
    embodied_factor = float(stock_item['ECC_Resolved'])
    preparation_factor = float(stock_item['PreparationFactor_Resolved'])

    # Expliciete volumedecompositie:
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

print("✅ Rekenmodules succesvol gedefinieerd.")

def build_cost_matrix(df_design, df_stock_raw, target_stock_ids=None):
    """
    Stap 2: bouwt de kostenmatrix met de assignment-cost functie.
    """
    print("Start generatie van de integrale CO2 Cost Matrix (nieuwe LCA-logica)...")

    df_stock = prepare_stock_cost_inputs(df_stock_raw)

    n_slots = len(df_design)
    n_stock = len(df_stock)

    cost_matrix = np.full((n_slots, n_stock), np.inf)
    succesvolle_matches = 0

    # Hier slaan we de gedetailleerde berekeningen in op!
    detailed_logs = []

    for i in range(n_slots):
        slot = df_design.iloc[i]
        slot_id = slot['edge_id'] # of Element_ID, afhankelijk van je kolomnaam

        for j in range(n_stock):
            stock_item = df_stock.iloc[j]
            stock_id = stock_item['Member_ID']

            total_match_score, components = calculate_lca_formula(slot, stock_item)

            if np.isfinite(total_match_score):
                cost_matrix[i, j] = total_match_score
                succesvolle_matches += 1

                # Uitgebreid loggen voor de diepte-analyse
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
                        'V_req_m3': '-', 'V_over_m3': '-', 'V_waste_m3': '-', 'V_stock_m3': '-',
                        'Embodied_CO2': '-', 'Prep_CO2': '-', 'Trans_CO2': '-', 'Waste_CO2': '-', 'Saw_CO2': '-',
                        'TOTAL_Score': np.inf
                    })

    print(f"✅ Matrix gegenereerd! Dimensies: {n_slots} benodigde staven x {n_stock} inventaris-balken.")
    print(f"📊 Aantal fysiek geldige combinaties gevonden: {succesvolle_matches}")

    return cost_matrix, df_stock, pd.DataFrame(detailed_logs)


def analyze_and_export_slot_logs(df_logs, target_slot_for_analysis, all_stock_ids, export_dir, display_fn=None):
    """
    Bereidt de diepte-analyse voor een specifieke slot-id voor, toont tabellen en exporteert CSV.

    Returns:
        (df_logs_slot, df_logs_slot_rs, analysis_export_path)
    """
    print("\n" + "=" * 80)
    print(f"🔬 DIEPTE-ANALYSE: ALLE FACTOREN VOOR SLOT {target_slot_for_analysis}")
    print("=" * 80)

    analysis_export_path = export_dir / f"diepte_analyse_{target_slot_for_analysis}.csv"

    if df_logs.empty:
        print("Geen logboek data gevonden.")
        print("=" * 80)
        return pd.DataFrame(), pd.DataFrame(), analysis_export_path

    df_logs = df_logs.sort_values(by=['Slot_ID', 'Stock_ID']).reset_index(drop=True)
    slot_mask = df_logs['Slot_ID'].astype(str).str.strip().str.lower() == str(target_slot_for_analysis).lower()
    df_logs_slot = df_logs.loc[slot_mask].copy()

    # Zorg dat alle stock-items zichtbaar zijn voor deze slot, ook als ze ontbreken in logs.
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
        print("Geen RS-items gevonden in de input stock lijst.")

    print("\nVolledige lijst (NS + RS):")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        if display_fn is not None:
            display_fn(df_logs_slot.fillna('-').reset_index(drop=True))
        else:
            print(df_logs_slot.fillna('-').reset_index(drop=True).to_string(index=False))

    df_logs_slot.to_csv(analysis_export_path, index=False)
    print(f"\n💾 Diepte-analyse geëxporteerd naar: {analysis_export_path}")
    print("=" * 80)

    return df_logs_slot, df_logs_slot_rs, analysis_export_path