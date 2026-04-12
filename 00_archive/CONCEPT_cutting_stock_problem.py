# Cutting stock optimization
import pulp
import numpy as np
import pandas as pd
import c11_params

def optimize_cutting_stock(df_slots, verrijkte_stock):

    print("Starting advanced MILP optimizer (including 1D cutting stock / nesting)...")
    # ==========================================
    # 1. PREPARE DATA (from df_slots and verrijkte_stock)
    # ==========================================
    # Build dictionaries so the optimizer can look up lengths and dimensions quickly.
    slot_lengths = {row['edge_id']: row['Length_Req'] for _, row in df_slots.iterrows()}
    slot_max_dim = {row['edge_id']: max(row['Width_Req'], row['Depth_Req']) for _, row in df_slots.iterrows()}
    slot_min_dim = {row['edge_id']: min(row['Width_Req'], row['Depth_Req']) for _, row in df_slots.iterrows()}
    slot_area = {row['edge_id']: (row['Width_Req'] * row['Depth_Req']) / 1e6 for _, row in df_slots.iterrows()} # Area in m2

    stock_lengths = {row['Member_ID']: row['Length'] for _, row in verrijkte_stock.iterrows()}
    stock_max_dim = {row['Member_ID']: max(row['Width'], row['Depth']) for _, row in verrijkte_stock.iterrows()}
    stock_min_dim = {row['Member_ID']: min(row['Width'], row['Depth']) for _, row in verrijkte_stock.iterrows()}
    stock_area = {row['Member_ID']: (row['Width'] * row['Depth']) / 1e6 for _, row in verrijkte_stock.iterrows()} # Area in m2
    stock_rho = {row['Member_ID']: row['Density'] for _, row in verrijkte_stock.iterrows()}

    # Determine GWP for each timber item (based on whether NS or RS is in the ID).
    stock_gwp = {row['Member_ID']: c11_params.GWP_RECLAIMED if 'RS' in row['Member_ID'] else c11_params.GWP_VIRGIN for _, row in verrijkte_stock.iterrows()}
    stock_base_cost = {row['Member_ID']: row['E_cost_Total_kgCO2'] for _, row in verrijkte_stock.iterrows()}

    stock_items = verrijkte_stock['Member_ID'].tolist()
    construction_slots = df_slots['edge_id'].tolist()
    new_items = [item for item in stock_items if 'NS' in item]
    reclaimed_items = [item for item in stock_items if 'RS' in item]

    # ==========================================
    # 2. FILTER VALID COMBINATIONS (cross-section and max length only)
    # ==========================================
    valid_matches = []
    for slot_id in construction_slots:
        for stock_id in stock_items:
            # HARD CONSTRAINT: the beam must be wide/deep enough (including rotation)
            # and the stock beam must be at least as long as the required member.
            if (stock_max_dim[stock_id] >= slot_max_dim[slot_id] and
                stock_min_dim[stock_id] >= slot_min_dim[slot_id] and
                stock_lengths[stock_id] >= slot_lengths[slot_id]):
                valid_matches.append((stock_id, slot_id))

    # ==========================================
    # 3. SET UP THE MILP MODEL
    # ==========================================
    prob = pulp.LpProblem("Timber_Cutting_Stock_Optimization", pulp.LpMinimize)

    # Variable 1: is this combination (slot from stock) selected? (0 or 1)
    x = pulp.LpVariable.dicts("Match", valid_matches, 0, 1, pulp.LpBinary)

    # Variable 2: is this timber beam cut at all? (0 or 1)
    # This determines whether base waste and base cost apply.
    y = pulp.LpVariable.dicts("UsedStock", stock_items, 0, 1, pulp.LpBinary)

    # ==========================================
    # 4. OBJECTIVE FUNCTION (dynamic CO2 minimization)
    # ==========================================
    objective_terms = []

    # A. Base LCA cost (only when the beam is cut)
    for stock_id in stock_items:
        objective_terms.append(y[stock_id] * stock_base_cost[stock_id])

    # B. Oversizing (calculated per piece cut from the beam)
    for stock_id, slot_id in valid_matches:
        lengte_m = slot_lengths[slot_id] / 1000.0
        area_verschil = stock_area[stock_id] - slot_area[slot_id]
        overdim_co2 = area_verschil * lengte_m * stock_rho[stock_id] * stock_gwp[stock_id]

        objective_terms.append(x[(stock_id, slot_id)] * max(0, overdim_co2))

    # C. Saw waste (remaining length). This is the 1D-CSP logic:
    # (Total beam length - Sum of all cut pieces) * area * density * GWP
    for stock_id in stock_items:
        valid_slots_for_this_stock = [s_id for (st_id, s_id) in valid_matches if st_id == stock_id]

        if valid_slots_for_this_stock:
            # Length in meters.
            totale_lengte_m = stock_lengths[stock_id] / 1000.0
            gebruikte_lengte_m = pulp.lpSum([x[(stock_id, slot_id)] * (slot_lengths[slot_id] / 1000.0) for slot_id in valid_slots_for_this_stock])

            # CO2 value of one meter of unused timber from this beam.
            co2_per_meter_waste = stock_area[stock_id] * stock_rho[stock_id] * stock_gwp[stock_id]

            # Add the waste calculation: (beam length * used flag - used length) * CO2_per_m
            waste_term = (totale_lengte_m * y[stock_id] - gebruikte_lengte_m) * co2_per_meter_waste
            objective_terms.append(waste_term)

    # Combine everything as the optimization objective.
    prob += pulp.lpSum(objective_terms)

    # ==========================================
    # 5. CONSTRAINTS
    # ==========================================
    # Rule 1: each construction member must be cut exactly once.
    for slot_id in construction_slots:
        prob += pulp.lpSum([x[(st_id, slot_id)] for (st_id, sl_id) in valid_matches if sl_id == slot_id]) == 1

    # Rule 2: the pieces cut from a beam may not exceed the beam length.
    for stock_id in stock_items:
        valid_slots_for_this_stock = [s_id for (st_id, s_id) in valid_matches if st_id == stock_id]
        if valid_slots_for_this_stock:
            # Multiply the maximum length by y[stock_id] (0 or 1).
            # If y is 0 (not used), the sum must also be <= 0.
            prob += pulp.lpSum([x[(stock_id, slot_id)] * slot_lengths[slot_id] for slot_id in valid_slots_for_this_stock]) <= stock_lengths[stock_id] * y[stock_id]

    # Rule 3: reclaimed timber can only be cut once physically (y <= 1).
    # For new timber, you could theoretically buy unlimited copies if lengths match,
    # but because nesting is done within specific IDs from the list, y is binary per beam.
    # The limits are therefore managed implicitly because each beam_id exists only once in inventory.

    # ==========================================
    # 6. SOLVE
    # ==========================================
    prob.solve()

    print("\n" + "="*50)
    print(f"SOLUTION STATUS: {pulp.LpStatus[prob.status]}")
    print("="*50)

    if pulp.LpStatus[prob.status] == 'Optimal':
        total_cost = pulp.value(prob.objective)

        print(f"\nOptimal nesting design found! CO2 penalty: {total_cost:.2f} kg")

        print("\nHow the beams are cut:")
        print("-" * 50)
        for stock_id in stock_items:
            if y[stock_id].varValue == 1:
                # Which members were cut from this beam?
                chosen_slots = [s_id for (st_id, s_id) in valid_matches if st_id == stock_id and x[(stock_id, s_id)].varValue == 1]
                if chosen_slots:
                    total_used_length = sum([slot_lengths[s_id] for s_id in chosen_slots])
                    leftover_waste = stock_lengths[stock_id] - total_used_length

                    print(f"Beam: {stock_id} (Length: {stock_lengths[stock_id]:.0f}mm) -> Cut")
                    print(f"  └─ Contains members: {', '.join(chosen_slots)}")
                    print(f"  └─ Leftover waste: {leftover_waste:.0f}mm")
    else:
        print("No solution. The inventory is insufficient, even with nesting.")