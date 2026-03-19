# ==========================================
# CEL 1: ALLE INPUT PARAMETERS (NEW & RECLAIMED)
# ==========================================
import itertools
import random
import numpy as np
import pandas as pd

# --- PARAMETERS NEW STOCK (CATALOGUS) - EFFICIËNTE DEFINITIE ---
TUPLE_LENGTHS = (2400, 2700, 3000, 3300, 3600, 4000, 4400, 4800, 5200)

# DEPTH_WIDTH_MAPPING: Voor elke depth, welke widths zijn geldig
DEPTH_WIDTH_MAPPING = {
    100: [38, 50, 63, 75, 100],
    150: [38, 50, 63, 75, 100],
    175: [38, 50, 63, 75],
    200: [38, 50, 63, 75, 100, 150, 200],
    225: [38, 50, 63, 75],
    250: [50, 75, 100, 250],
    300: [50, 75, 100, 150, 300]
}

# Auto-genereer DEPTH_WIDTH_COMBINATIONS uit DEPTH_WIDTH_MAPPING
DEPTH_WIDTH_COMBINATIONS = [
    (depth, width)
    for depth in DEPTH_WIDTH_MAPPING.keys()
    for width in DEPTH_WIDTH_MAPPING[depth]
]

MECH_PROPS_NEW = {
    'E_modulus_eff': 11000.0, # N/mm2
    'f_mk': 24,               # N/mm2
    'Density': 420            # kg/m3
}

# LCA Aannames Nieuw Hout
LCA_NEW = {
    'Embodied Carbon Coëfficiënt': 150.0,  # Fictief hoog (productie + drogen)
    'Transport_Dist': 1500,                 # Vaste grotere afstand (bijv. import)
    'Emmisiefactor_diesel_range': (0.08, 0.15), # Alleen groot diesel transport voor nieuw hout
    'Bewerkingsfactor': 0                  # 0 = Geen ontspijkering nodig
}

# --- PARAMETERS RECLAIMED STOCK ---
DONOR_BATCHES = [
    {"batch_id": "B01", "count": 2, "orig_width": 180, "orig_depth": 600, "orig_length": 12000},
    {"batch_id": "B02", "count": 10, "orig_width": 75, "orig_depth": 225, "orig_length": 5400},
    {"batch_id": "B03", "count": 4, "orig_width": 200, "orig_depth": 200, "orig_length": 4200}
]

MECH_PROPS_RECLAIMED = {
    'C24': {'e_mod': 11000.0, 'f_mk': 24, 'density': 420},
    'C18': {'e_mod': 9000.0,  'f_mk': 18, 'density': 380}
}

# LCA Aannames Reclaimed Hout
LCA_RECLAIMED = {
    'Embodied Carbon Coëfficiënt': 15.0,   # Fictief laag (enkel de-constructie impact)
    'Emmisiefactor_diesel_range': (0.08, 0.15),
    'Emmisiefactor_elektrisch_range': (0.02, 0.05),
    'Kans_op_elektrisch': 0.30, # 30% kans dat het lokaal via een e-truck gaat
    'Bewerkingsfactor': 1                  # 1 = Ontspijkeren en schaven nodig
}
# LCA: Willekeurige afstand vanaf donor site
# Locatie Delft: (lokaal, Groningen)
transport_dist = random.randint(10, 240)

print("Alle parameters succesvol geladen! Ga naar cel 2.")

def generate_new_timber_catalog():
    data = []

    combinaties = list(itertools.product(TUPLE_LENGTHS, DEPTH_WIDTH_COMBINATIONS))
    print(f"📊 Catalogus genereren... {len(combinaties)} balk-typen")

    for index, (length, (depth, width)) in enumerate(combinaties):
        emmisiefactor_new = random.uniform(*LCA_NEW['Emmisiefactor_diesel_range'])

        data.append({
            'Member_ID': f"NS_{index:05d}",
            'State': 0,
            'Length': float(length),
            'Depth': float(depth),
            'Width': float(width),
            'E_modulus_eff': float(MECH_PROPS_NEW['E_modulus_eff']),
            'f_mk': int(MECH_PROPS_NEW['f_mk']),
            'Density': int(MECH_PROPS_NEW['Density']),
            'Embodied Carbon Coëfficiënt': float(LCA_NEW['Embodied Carbon Coëfficiënt']),
            'Transport_Dist': int(LCA_NEW['Transport_Dist']),
            'Emmisiefactor': round(emmisiefactor_new, 4),
            'Bewerkingsfactor': int(LCA_NEW['Bewerkingsfactor'])
        })

    df_new = pd.DataFrame(data)
    print("✅ New stock succesvol gegenereerd!")
    return df_new

def generate_reclaimed_stock():
    inventory_list = []
    current_id_number = 1

    for batch in DONOR_BATCHES:
        for _ in range(batch['count']):
            # Geometrie: Sloop- en schaafverlies
            cut_loss = random.randint(100, 400)
            length = batch['orig_length'] - cut_loss
            depth = batch['orig_depth'] - random.randint(10, 16)
            width = batch['orig_width'] - random.randint(10, 16)

            # Grading (Kwaliteit) bepalen
            grade = np.random.choice(['C24', 'C18'], p=[0.60, 0.40])

            # LCA: Transport afstand en dynamische emissiefactor
            transport_dist = random.randint(20, 150)

            # Bepaal of dit specifieke element met een elektrische of diesel truck gaat
            if random.random() < LCA_RECLAIMED['Kans_op_elektrisch']:
                emmisiefactor_reclaimed = random.uniform(*LCA_RECLAIMED['Emmisiefactor_elektrisch_range'])
            else:
                emmisiefactor_reclaimed = random.uniform(*LCA_RECLAIMED['Emmisiefactor_diesel_range'])

            inventory_list.append({
                "Member_ID": f"RS_{current_id_number:05d}",
                "State": 1, # 1 = Reclaimed

                # Geometrie
                "Length": float(length),
                "Depth": float(depth),
                "Width": float(width),

                # Mechanisch
                "E_modulus_eff": float(MECH_PROPS_RECLAIMED[grade]['e_mod']),
                "f_mk": int(MECH_PROPS_RECLAIMED[grade]['f_mk']),
                "Density": int(MECH_PROPS_RECLAIMED[grade]['density']),

                # LCA
                "Embodied Carbon Coëfficiënt": float(LCA_RECLAIMED['Embodied Carbon Coëfficiënt']),
                "Transport_Dist": int(transport_dist),
                "Emmisiefactor": round(emmisiefactor_reclaimed, 4), # Afgerond op 4 decimalen
                "Bewerkingsfactor": int(LCA_RECLAIMED['Bewerkingsfactor'])
            })
            current_id_number += 1

    df_reused = pd.DataFrame(inventory_list)
    print(f"Reclaimed stock gegenereerd! Totaal elementen: {len(df_reused)}")
    return df_reused
