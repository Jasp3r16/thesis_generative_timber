import json
from functools import lru_cache

import config
from c16_generation_timber import generate_length_tuple_from_average

DEFAULT_STRUCTURE_AVERAGE_LENGTH_MM = 3000

@lru_cache(maxsize=1)
def _load_json_file(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_summary_statistics() -> dict:
    json_path = str(config.DATA_IO_PATH / "representative_beam_statistics.json")
    try:
        data = _load_json_file(json_path)
        return dict(data["summary_statistics"])
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(
            f"Warning: could not load representative_beam_statistics.json ({exc}). "
            f"Using default structure length {DEFAULT_STRUCTURE_AVERAGE_LENGTH_MM} mm."
        )

summary_statistics = _load_summary_statistics()

average_length_mm = float(summary_statistics["average_length_mm"])
min_length_mm = float(summary_statistics["min_length_mm"])
max_length_mm = float(summary_statistics["max_length_mm"])
total_length_mm = float(summary_statistics["total_length_mm"])
edge_count = int(summary_statistics["edge_count"])

print(f"Loaded summary statistics: average_length_mm={average_length_mm}, min_length_mm={min_length_mm}, max_length_mm={max_length_mm}, total_length_mm={total_length_mm}, edge_count={edge_count}")

# ================================
# RECLAIMED STOCK — DONOR BUILDING
# ================================
# Parametric donor building: 3-storey Dutch residential, mixed-bay timber floors.
# Member types define structural role, nominal section, count per floor, and bay span.
# Lengths are bay span minus a random cut loss of 0-20 mm with NO rounding.
# Cross-sections are nominal dimensions minus planing allowance per dimension.

RECLAIMED_PLANING_ALLOWANCE_MM = 10  # mm removed per dimension during light planing

DONOR_BUILDING_FLOORS = 3
DONOR_BUILDING_SURVIVAL_RATE = 0.75  # fraction of elements passing visual inspection

# Each entry: (role, nominal_width_mm, nominal_depth_mm, count_per_floor, span_mm)
DONOR_BUILDING_MEMBER_TYPES = [
    ("primary_beam",    120, 240, 12, 4500),
    ("secondary_joist",  80, 200, 15, 3000),
    ("short_joist",      70, 180,  9, 1800),
    ("edge_beam",       100, 200,  8, 2700),
]

# Cut loss applied per element during deconstruction: uniform random int in [0, CUT_LOSS_MAX_MM]
RECLAIMED_CUT_LOSS_MAX_MM = 20

# LCA assumptions for reclaimed timber (unchanged).
RECLAIMED_TIMBER_LCA = {
    "transport_distance_range": (5, 240),
    "diesel_emission_factor_range": (0.17, 0.18),
    "electric_emission_factor_range": (0.02, 0.05),
    "electric_transport_probability": 0.30,
}

# --- PARAMETERS NEW STOCK (CATALOGUS) ---
LENGTH_INCREMENT_MM = 300
LENGTH_LIBRARY_SIZE = 13
LENGTH_ROUND_TO_MM = 50
NEW_STOCK_TAIL_MARGIN_MM = 300  # x in [min_length - x, max_length + x] for tail generation
NEW_STOCK_TAIL_SECTION_COUNT = 6  # small subset of section combinations for tail sizes
STRUCTURE_AVERAGE_LENGTH_MM = average_length_mm
MIN_LENGTH_MM = min_length_mm - LENGTH_INCREMENT_MM
MAX_LENGTH_MM = max_length_mm + LENGTH_INCREMENT_MM

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
    "diesel_emission_factor_range": (0.17, 0.18),  # Diesel transport only for new timber
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