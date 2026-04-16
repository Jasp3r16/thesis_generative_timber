import json
import config
from c22_generation_timber import generate_length_tuple_from_average

# --- RECLAIMED STOCK PARAMETERS ---
# Typology B: strict, discrete cross-section library from historical nominal sizes
# with a fixed planing allowance.
RECLAIMED_HISTORICAL_CROSS_SECTIONS_MM = [
    (60, 160),
    (70, 180),
    (80, 240),
    (90, 220),
]
RECLAIMED_PLANING_ALLOWANCE_MM = 10
RECLAIMED_CROSS_SECTION_LIBRARY_MM = [
    (width_nom - RECLAIMED_PLANING_ALLOWANCE_MM, depth_nom - RECLAIMED_PLANING_ALLOWANCE_MM)
    for width_nom, depth_nom in RECLAIMED_HISTORICAL_CROSS_SECTIONS_MM
]

# Finite reclaimed inventory size and stochastic length model.
RECLAIMED_STOCK_COUNT = 38
RECLAIMED_LENGTH_DISTRIBUTION = "normal"  # supported: "normal"
RECLAIMED_LENGTH_MIN_MM = 1400
RECLAIMED_LENGTH_MAX_MM = 5200
RECLAIMED_LENGTH_MEAN_MM = 3200
RECLAIMED_LENGTH_STD_MM = 450
RECLAIMED_LENGTH_ROUND_TO_MM = 50

# LCA assumptions for reclaimed timber.
RECLAIMED_TIMBER_LCA = {
    "embodied_carbon_coefficient": 0.0,                # Fictitiously low (deconstruction only)
    "transport_distance_range": (5, 240),              # Delft location: (local, Groningen)
    "diesel_emission_factor_range": (0.17, 0.18),
    "electric_emission_factor_range": (0.02, 0.05),
    "electric_transport_probability": 0.30,            # 30% chance that local transport uses an e-truck
    "processing_factor": 1                             # 1 = de-nailing and planing required
}

# --- PARAMETERS NEW STOCK (CATALOGUS) ---
DEFAULT_STRUCTURE_AVERAGE_LENGTH_MM = 3000

json_path = config.DATA_IO_PATH / 'representative_beam_length.json'
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
value = float(data["representative_length_mm"])
if value == 0:
    print(f"Warning: zero value found in {json_path}: {value}. Falling back to default.")
    value = DEFAULT_STRUCTURE_AVERAGE_LENGTH_MM

STRUCTURE_AVERAGE_LENGTH_MM = value
LENGTH_INCREMENT_MM = 300
LENGTH_LIBRARY_SIZE = 13
LENGTH_ROUND_TO_MM = 50
MIN_LENGTH_MM = 1400
MAX_LENGTH_MM = 5200

TUPLE_LENGTHS = generate_length_tuple_from_average(
    mean_length_mm=STRUCTURE_AVERAGE_LENGTH_MM,
    increment_mm=LENGTH_INCREMENT_MM,
    n_lengths=LENGTH_LIBRARY_SIZE,
    round_to_mm=LENGTH_ROUND_TO_MM,
    min_length_mm=MIN_LENGTH_MM,
    max_length_mm=MAX_LENGTH_MM
)

# DEPTH_WIDTH_MAPPING: For each depth, which widths are valid.
DEPTH_WIDTH_MAPPING = {
    100: [38, 50, 63, 75, 100],
    150: [38, 50, 63, 75, 100],
    175: [38, 50, 63, 75],
    200: [38, 50, 63, 75, 100, 150, 200],
    225: [38, 50, 63, 75],
    250: [50, 75, 100, 250],
    300: [50, 75, 100, 150, 300]
}

# Auto-generate DEPTH_WIDTH_COMBINATIONS from DEPTH_WIDTH_MAPPING.
DEPTH_WIDTH_COMBINATIONS = [
    (depth, width)
    for depth in DEPTH_WIDTH_MAPPING.keys()
    for width in DEPTH_WIDTH_MAPPING[depth]
]

# LCA assumptions for new timber.
NEW_TIMBER_LCA = {
    "embodied_carbon_coefficient": 150.0,     # Fictitiously high (production + drying)
    "diesel_emission_factor_range": (0.17, 0.18),  # Diesel transport only for new timber
    "processing_factor": 0                    # 0 = No de-nailing required
}

MECH_PROPS = {
    'C18': {
        'f_mk': 18.0,            # N/mm2
        'f_tk': 11.0,            # N/mm2
        'E_modulus_eff': 9000.0, # N/mm2
        'E_modulus_005': 6000.0, # N/mm2
        'f_vk': 2.0,             # N/mm2
        'f_c0k': 18.0,           # N/mm2
        'k_density': 320,        # kg/m3 (karakteristieke dichtheid)
        'mean_density': 380      # kg/m3
    },
    'C24': {
        'f_mk': 24.0,             # N/mm2
        'f_tk': 14.0,             # N/mm2
        'E_modulus_eff': 11000.0, # N/mm2
        'E_modulus_005': 7400.0,  # N/mm2
        'f_vk': 2.5,              # N/mm2
        'f_c0k': 21.0,            # N/mm2
        'k_density': 350,         # kg/m3 (karakteristieke dichtheid)
        'mean_density': 420       # kg/m3
    }
}