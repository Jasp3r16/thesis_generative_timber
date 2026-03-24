import random
import json
import config
from generation_timber import generate_length_tuple_from_average

# --- PARAMETERS RECLAIMED STOCK ---
DONOR_BATCHES = [
    {"batch_id": "B01", "count": 4, "orig_width": 63, "orig_depth": 150, "orig_length": 3300},
    {"batch_id": "B02", "count": 10, "orig_width": 75, "orig_depth": 225, "orig_length": 4000},
    {"batch_id": "B03", "count": 4, "orig_width": 100, "orig_depth": 250, "orig_length": 4800}
]

MECH_PROPS_RECLAIMED = {
    'C24': {'e_mod': 11000.0, 'f_mk': 24, 'density': 420},
    'C18': {'e_mod': 9000.0,  'f_mk': 18, 'density': 380}
}

# LCA Aannames Reclaimed Hout
LCA_RECLAIMED = {
    'Embodied Carbon Coëfficiënt': 15.0,                # Fictief laag (enkel de-constructie impact)
    "Transport_distance_range": (5, 240),               # Locatie Delft: (lokaal, Groningen)
    'Emmisiefactor_diesel_range': (0.17, 0.18),
    'Emmisiefactor_elektrisch_range': (0.02, 0.05),
    'Kans_op_elektrisch': 0.30,                         # 30% kans dat het lokaal via een e-truck gaat
    'Bewerkingsfactor': 1                               # 1 = Ontspijkeren en schaven nodig
}

# --- PARAMETERS NEW STOCK (CATALOGUS) ---

DEFAULT_STRUCTURE_AVERAGE_LENGTH_MM = 3600

json_path = 'representative_beam_length.json'
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
value = float(data["representative_length_mm"])
if value == 0:
    print(f"⚠️ Waarschuwing: Nul waarde gevonden in {json_path}: {value}. Fallback naar default.")
    value = DEFAULT_STRUCTURE_AVERAGE_LENGTH_MM

STRUCTURE_AVERAGE_LENGTH_MM = value
LENGTH_INCREMENT_MM = 300
LENGTH_LIBRARY_SIZE = 9
LENGTH_ROUND_TO_MM = 50
MIN_LENGTH_MM = 2400
MAX_LENGTH_MM = 5200

TUPLE_LENGTHS = generate_length_tuple_from_average(
    mean_length_mm=STRUCTURE_AVERAGE_LENGTH_MM,
    increment_mm=LENGTH_INCREMENT_MM,
    n_lengths=LENGTH_LIBRARY_SIZE,
    round_to_mm=LENGTH_ROUND_TO_MM,
    min_length_mm=MIN_LENGTH_MM,
    max_length_mm=MAX_LENGTH_MM
)
print("Gegenereerde lengte tuple (mm):", TUPLE_LENGTHS)

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
    'Emmisiefactor_diesel_range': (0.17, 0.18), # Alleen groot diesel transport voor nieuw hout
    'Bewerkingsfactor': 0                  # 0 = Geen ontspijkering nodig
}