# @title
import pulp
import numpy as np
import pandas as pd
import c11_params

def optimize_cutting_stock(df_slots, verrijkte_stock):

    print("Start Advanced MILP Optimizer (Inclusief 1D Cutting Stock / Nesting)...")
    # ==========================================
    # 1. DATA VOORBEREIDEN (Vanuit df_slots en verrijkte_stock)
    # ==========================================
    # Maak dictionaries zodat de Optimizer razendsnel lengtes en dimensies kan opzoeken
    slot_lengths = {row['edge_id']: row['Length_Req'] for _, row in df_slots.iterrows()}
    slot_max_dim = {row['edge_id']: max(row['Width_Req'], row['Depth_Req']) for _, row in df_slots.iterrows()}
    slot_min_dim = {row['edge_id']: min(row['Width_Req'], row['Depth_Req']) for _, row in df_slots.iterrows()}
    slot_area = {row['edge_id']: (row['Width_Req'] * row['Depth_Req']) / 1e6 for _, row in df_slots.iterrows()} # Area in m2

    stock_lengths = {row['Member_ID']: row['Length'] for _, row in verrijkte_stock.iterrows()}
    stock_max_dim = {row['Member_ID']: max(row['Width'], row['Depth']) for _, row in verrijkte_stock.iterrows()}
    stock_min_dim = {row['Member_ID']: min(row['Width'], row['Depth']) for _, row in verrijkte_stock.iterrows()}
    stock_area = {row['Member_ID']: (row['Width'] * row['Depth']) / 1e6 for _, row in verrijkte_stock.iterrows()} # Area in m2
    stock_rho = {row['Member_ID']: row['Density'] for _, row in verrijkte_stock.iterrows()}

    # Bepaal GWP voor elk hout-item (Afhankelijk of NS of RS in de naam staat)
    stock_gwp = {row['Member_ID']: c11_params.GWP_RECLAIMED if 'RS' in row['Member_ID'] else c11_params.GWP_VIRGIN for _, row in verrijkte_stock.iterrows()}
    stock_base_cost = {row['Member_ID']: row['E_cost_Total_kgCO2'] for _, row in verrijkte_stock.iterrows()}

    stock_items = verrijkte_stock['Member_ID'].tolist()
    construction_slots = df_slots['edge_id'].tolist()
    new_items = [item for item in stock_items if 'NS' in item]
    reclaimed_items = [item for item in stock_items if 'RS' in item]

    # ==========================================
    # 2. FILTEREN VAN GELDIGE COMBINATIES (Alleen Dwarsdoorsnede & Max Lengte)
    # ==========================================
    valid_matches = []
    for slot_id in construction_slots:
        for stock_id in stock_items:
            # HARD CONSTRAINT: Balk moet in ieder geval dik/breed genoeg zijn (inclusief rotatie)
            # EN de stock-balk moet langer of gelijk zijn aan deze ene specifieke staaf.
            if (stock_max_dim[stock_id] >= slot_max_dim[slot_id] and
                stock_min_dim[stock_id] >= slot_min_dim[slot_id] and
                stock_lengths[stock_id] >= slot_lengths[slot_id]):
                valid_matches.append((stock_id, slot_id))

    # ==========================================
    # 3. HET MILP MODEL OPZETTEN
    # ==========================================
    prob = pulp.LpProblem("Timber_Cutting_Stock_Optimization", pulp.LpMinimize)

    # Variabele 1: Wordt deze combinatie (Slot uit Stock) gemaakt? (0 of 1)
    x = pulp.LpVariable.dicts("Match", valid_matches, 0, 1, pulp.LpBinary)

    # Variabele 2: Wordt deze Hout-balk überhaupt AANGESNEDEN? (0 of 1)
    # (Dit is nieuw! Hiermee bepalen we of we het basisafval en base_cost moeten rekenen)
    y = pulp.LpVariable.dicts("UsedStock", stock_items, 0, 1, pulp.LpBinary)

    # ==========================================
    # 4. OBJECTIVE FUNCTION (CO2 Minimalisatie Dynamisch Berekenen)
    # ==========================================
    objective_terms = []

    # A. De Base LCA Cost (alleen als we de balk aansnijden)
    for stock_id in stock_items:
        objective_terms.append(y[stock_id] * stock_base_cost[stock_id])

    # B. De Overdimensionering (Wordt berekend per stukje staaf dat we uit de balk zagen)
    for stock_id, slot_id in valid_matches:
        lengte_m = slot_lengths[slot_id] / 1000.0
        area_verschil = stock_area[stock_id] - slot_area[slot_id]
        overdim_co2 = area_verschil * lengte_m * stock_rho[stock_id] * stock_gwp[stock_id]

        objective_terms.append(x[(stock_id, slot_id)] * max(0, overdim_co2))

    # C. Het Zaagverlies (Restlengte). Dit is de genialiteit van 1D-CSP:
    # (Totale lengte van de balk - Som van alle stukjes die we eruit zagen) * Area * Density * GWP
    for stock_id in stock_items:
        valid_slots_for_this_stock = [s_id for (st_id, s_id) in valid_matches if st_id == stock_id]

        if valid_slots_for_this_stock:
            # Lengte in meters!
            totale_lengte_m = stock_lengths[stock_id] / 1000.0
            gebruikte_lengte_m = pulp.lpSum([x[(stock_id, slot_id)] * (slot_lengths[slot_id] / 1000.0) for slot_id in valid_slots_for_this_stock])

            # De CO2 waarde van een meter 'leeg' hout van deze balk
            co2_per_meter_waste = stock_area[stock_id] * stock_rho[stock_id] * stock_gwp[stock_id]

            # Voeg de afval berekening toe: (Lengte Balk * IsGebruikt - Gebruikte Lengte) * CO2_per_m
            waste_term = (totale_lengte_m * y[stock_id] - gebruikte_lengte_m) * co2_per_meter_waste
            objective_terms.append(waste_term)

    # Voeg alles samen als het doel van de AI
    prob += pulp.lpSum(objective_terms)

    # ==========================================
    # 5. CONSTRAINTS (De Regels)
    # ==========================================
    # Regel 1: Elke staaf in de constructie MOET precies 1 keer gezaagd worden
    for slot_id in construction_slots:
        prob += pulp.lpSum([x[(st_id, slot_id)] for (st_id, sl_id) in valid_matches if sl_id == slot_id]) == 1

    # Regel 2: De stukken die we uit een balk zagen, mogen samen NIET langer zijn dan de balk! (De 1D Bin Packing regel)
    for stock_id in stock_items:
        valid_slots_for_this_stock = [s_id for (st_id, s_id) in valid_matches if st_id == stock_id]
        if valid_slots_for_this_stock:
            # Let op: We vermenigvuldigen de maximale lengte met y[stock_id] (0 of 1).
            # Als y 0 is (niet gebruikt), mag de som dus ook maximaal 0 zijn!
            prob += pulp.lpSum([x[(stock_id, slot_id)] * slot_lengths[slot_id] for slot_id in valid_slots_for_this_stock]) <= stock_lengths[stock_id] * y[stock_id]

    # Regel 3: Reclaimed hout kan maar 1 keer fysiek "aangesneden" worden (y <= 1).
    # Voor New hout geldt dat we er theoretisch oneindig veel van kunnen "kopen" (als ze exact deze lengte hebben),
    # MAAR omdat we hier 'nesting' doen binnen specifieke ID's uit je lijst, is y een binaire variabele voor ELKE balk.
    # De limieten worden dus impliciet beheerd doordat elke balk_id maar 1 object in je voorraad is.

    # ==========================================
    # 6. OPLOSSEN
    # ==========================================
    prob.solve()

    print("\n" + "="*50)
    print(f"STATUS OPLOSSING: {pulp.LpStatus[prob.status]}")
    print("="*50)

    if pulp.LpStatus[prob.status] == 'Optimal':
        total_cost = pulp.value(prob.objective)

        print(f"\n✅ Optimaal 'Nesting' ontwerp gevonden! CO2 Penalty: {total_cost:.2f} kg")

        print("\n📦 HOE DE BALKEN WORDEN OPGEZAAGD:")
        print("-" * 50)
        for stock_id in stock_items:
            if y[stock_id].varValue == 1:
                # Welke staven zijn uit deze balk gehaald?
                gekozen_slots = [s_id for (st_id, s_id) in valid_matches if st_id == stock_id and x[(stock_id, s_id)].varValue == 1]
                if gekozen_slots:
                    totale_gebruikte_lengte = sum([slot_lengths[s_id] for s_id in gekozen_slots])
                    restafval = stock_lengths[stock_id] - totale_gebruikte_lengte

                    print(f"Balk: {stock_id} (Lengte: {stock_lengths[stock_id]:.0f}mm) -> Aangesneden!")
                    print(f"  └─ Bevat staven: {', '.join(gekozen_slots)}")
                    print(f"  └─ Restafval: {restafval:.0f}mm")
    else:
        print("❌ Geen oplossing. De voorraad (zélfs met opdelen) is niet voldoende.")