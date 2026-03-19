import config
import c11_params
import numpy as np
import pandas as pd

# ==========================================
# CEL 2: DE REKENFUNCTIES (MODULES)
# ==========================================
def calculate_pseudo_lca_stock(df_stock):
    """
    Berekent de pseudo-LCA score (E_cost) voor een reeds ingeladen DataFrame
    met de timber stock.
    """
    # Maak een kopie om waarschuwingen (SettingWithCopyWarning) te voorkomen
    df_lca = df_stock.copy()

    print("Start in-memory pseudo-LCA berekeningen...")

    # Stap A: Bereken Volume in m3 (dimensies zijn in mm, dus delen door 1000)
    df_lca['Volume_m3'] = (df_lca['Length'] / 1000) * (df_lca['Width'] / 1000) * (df_lca['Depth'] / 1000)

    df_lca['Impact_Material_kgCO2'] = df_lca['Volume_m3'] * df_lca['ECC']

    # Dichtheid (Density) delen door 1000 om van kg/m3 naar ton/m3 te gaan
    df_lca['Impact_Transport_kgCO2'] = df_lca['Volume_m3'] * (df_lca['Density'] / 1000) * df_lca['Transport_Dist'] * df_lca['Emmisiefactor']

    # Binaire bewerkingsfactor (0 of 1) maal de vaste penalty
    df_lca['Impact_Processing_kgCO2'] = df_lca['Bewerkingsfactor'] * c11_params.PROCESSING_PENALTY_CO2

    # Stap E: Totale E_cost Berekenen
    df_lca['E_cost_Total_kgCO2'] = (
        df_lca['Impact_Material_kgCO2'] +
        df_lca['Impact_Transport_kgCO2'] +
        df_lca['Impact_Processing_kgCO2']
    )

    return df_lca

def calculate_geometric_penalties(slot, stock_item):
    """
    Module 2: Berekent de CO2-penalty's voor zaagverlies en overdimensionering
    voor één specifieke match tussen een Slot (ontwerp) en Stock (voorraad).
    Geeft een tuple terug: (c_waste, c_overdim).
    """
    # 1. Haal afmetingen op en converteer naar meters
    l_slot = slot['Length_Req'] / 1000.0
    w_req = slot['Width_Req'] / 1000.0
    d_req = slot['Depth_Req'] / 1000.0
    a_req = w_req * d_req

    l_stock = stock_item['Length'] / 1000.0
    w_stock = stock_item['Width'] / 1000.0
    d_stock = stock_item['Depth'] / 1000.0
    a_stock = w_stock * d_stock

    rho = stock_item['Density'] # kg/m3

    # Status: 0 = Virgin, 1 = Reclaimed (afhankelijk van hoe je dataset in elkaar zit, pas dit evt. aan)
    gwp_unit = c11_params.GWP_RECLAIMED if stock_item['State'] == 1 else c11_params.GWP_VIRGIN

    # Zaagverlies en Overdimensionering in kg CO2 eq
    c_waste = (l_stock - l_slot) * a_stock * rho * gwp_unit
    c_overdim = (a_stock - a_req) * l_slot * rho * gwp_unit

    return max(0, c_waste), max(0, c_overdim)

print("✅ Rekenmodules succesvol gedefinieerd.")

def build_cost_matrix(df_design, df_stock_raw, target_stock_ids=None):
    print("Start generatie van de integrale CO2 Cost Matrix...")

    # 1. Bereken de basis LCA score (Roep de module uit Cel 2 aan)
    df_stock = calculate_pseudo_lca_stock(df_stock_raw)

    n_slots = len(df_design)
    n_stock = len(df_stock)

    cost_matrix = np.full((n_slots, n_stock), np.inf)
    succesvolle_matches = 0

    # Hier slaan we de gedetailleerde berekeningen in op!
    detailed_logs = []

    for i in range(n_slots):
        slot = df_design.iloc[i]
        slot_id = slot['edge_id'] # of Element_ID, afhankelijk van je kolomnaam

        slot_max_dim = max(slot['Width_Req'], slot['Depth_Req'])
        slot_min_dim = min(slot['Width_Req'], slot['Depth_Req'])

        for j in range(n_stock):
            stock_item = df_stock.iloc[j]
            stock_id = stock_item['Member_ID']

            stock_max_dim = max(stock_item['Width'], stock_item['Depth'])
            stock_min_dim = min(stock_item['Width'], stock_item['Depth'])

            # --- HARD CONSTRAINTS (INCLUSIEF ROTATIE) ---
            fits_physically = (
                stock_item['Length'] >= slot['Length_Req'] and
                stock_max_dim >= slot_max_dim and
                stock_min_dim >= slot_min_dim
            )

            if fits_physically:
                # 2. Haal de individuele LCA componenten op
                i_mat = stock_item['Impact_Material_kgCO2']
                i_trans = stock_item['Impact_Transport_kgCO2']
                i_proc = stock_item['Impact_Processing_kgCO2']
                e_cost_base = stock_item['E_cost_Total_kgCO2']
                c_waste, c_overdim = calculate_geometric_penalties(slot, stock_item)
                total_match_score = e_cost_base + c_waste + c_overdim

                cost_matrix[i, j] = total_match_score
                succesvolle_matches += 1

                # 4. Uitgebreid loggen voor de diepte-analyse
                if target_stock_ids and stock_id in target_stock_ids:
                    detailed_logs.append({
                        'Slot_ID': slot_id,
                        'Stock_ID': stock_id,
                        'Status': '✅',
                        'Mat_CO2': round(i_mat, 3),
                        'Trans_CO2': round(i_trans, 3),
                        'Proc_CO2': round(i_proc, 3),
                        'E_cost_Base': round(e_cost_base, 2),
                        'Waste_CO2': round(c_waste, 3),
                        'Overdim_CO2': round(c_overdim, 3),
                        'TOTAL_Score': round(total_match_score, 2)
                    })
            else:
                if target_stock_ids and stock_id in target_stock_ids:
                    detailed_logs.append({
                        'Slot_ID': slot_id, 'Stock_ID': stock_id, 'Status': '❌',
                        'Mat_CO2': '-', 'Trans_CO2': '-', 'Proc_CO2': '-', 'E_cost_Base': "-",
                        'Waste_CO2': '-', 'Overdim_CO2': '-', 'TOTAL_Score': np.inf
                    })

    print(f"✅ Matrix gegenereerd! Dimensies: {n_slots} benodigde staven x {n_stock} inventaris-balken.")
    print(f"📊 Aantal fysiek geldige combinaties gevonden: {succesvolle_matches}")

    return cost_matrix, df_stock, pd.DataFrame(detailed_logs)