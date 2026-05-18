"""Parameter definitions and defaults for timber generation.

This module loads representative beam statistics (if available) and
exposes constants used by the timber dataset generation code.
"""

from functools import lru_cache
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import config
from c16_generation_timber import generate_length_tuple_from_average

logger = logging.getLogger(__name__)

DEFAULT_STRUCTURE_AVERAGE_LENGTH_MM = 3000


@lru_cache(maxsize=1)
def _load_json_file(json_path: Path) -> Dict:
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_summary_statistics() -> Dict:
    json_path = Path(config.DATA_IO_PATH) / "representative_beam_statistics.json"
    try:
        data = _load_json_file(json_path)
        return dict(data["summary_statistics"])
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as exc:
        logger.warning(
            "Could not load representative_beam_statistics.json (%s). "
            "Using default structure length %d mm.",
            exc,
            DEFAULT_STRUCTURE_AVERAGE_LENGTH_MM,
        )
        return {
            "average_length_mm": DEFAULT_STRUCTURE_AVERAGE_LENGTH_MM,
            "min_length_mm": DEFAULT_STRUCTURE_AVERAGE_LENGTH_MM - 1200,
            "max_length_mm": DEFAULT_STRUCTURE_AVERAGE_LENGTH_MM + 1500,
            "total_length_mm": DEFAULT_STRUCTURE_AVERAGE_LENGTH_MM * 30,
            "edge_count": 30,
        }


summary_statistics = _load_summary_statistics()

average_length_mm = float(summary_statistics["average_length_mm"])
min_length_mm = float(summary_statistics["min_length_mm"])
max_length_mm = float(summary_statistics["max_length_mm"])
total_length_mm = float(summary_statistics["total_length_mm"])
edge_count = int(summary_statistics["edge_count"])

logger.info(
    "Beam statistics: avg=%.0f mm, min=%.0f mm, max=%.0f mm, n=%d",
    average_length_mm,
    min_length_mm,
    max_length_mm,
    edge_count,
)

# ================================
# RECLAIMED STOCK — DONOR BUILDINGS
# ================================
# Two switchable donor building profiles.
# Pass donor_building="A" or "B" to generate_reclaimed_stock().
# Member types: (role, nominal_width_mm, nominal_depth_mm, count_per_floor, span_mm)
# Lengths = span - uniform random cut loss [0, RECLAIMED_CUT_LOSS_MAX_MM], NOT rounded.
# Cross-sections = nominal dimensions minus RECLAIMED_PLANING_ALLOWANCE_MM per dimension.

RECLAIMED_PLANING_ALLOWANCE_MM = 10  # mm removed per dimension during light planing

DONOR_BUILDING_FLOORS = 3
DONOR_BUILDING_SURVIVAL_RATE = 0.75  # fraction of elements passing visual inspection

# Cut loss applied per element during deconstruction: uniform random int in [0, CUT_LOSS_MAX_MM]
RECLAIMED_CUT_LOSS_MAX_MM = 20

# --- Donor Building A: 3-storey Dutch residential, mixed-bay timber floors ---
# Hardcoded residential spans; shorter elements dominate (max 4500 mm).
DONOR_BUILDING_A_MEMBER_TYPES = [
    ("primary_beam",    120, 240, 12, 4500),
    ("secondary_joist",  80, 200, 15, 3000),
    ("short_joist",      70, 180,  9, 1800),
    ("edge_beam",       100, 200,  8, 2700),
]

# --- Donor Building B: commercial/industrial, long-span timber structure ---
# Spans derived from the structure's own length statistics so the RS pool
# covers the same [MIN_LENGTH_MM, MAX_LENGTH_MM] range as new stock.
# Biased toward longer members (upper half of range) to fill slots the GA
# currently can't match with RS from Building A.
_b_long  = int(max_length_mm + 300)                              # at or near structural max (matches MAX_LENGTH_MM = max_length_mm + LENGTH_INCREMENT_MM)
_b_upper = int((average_length_mm + max_length_mm + 300) / 2)   # halfway avg→max
_b_avg   = int(average_length_mm)                                # at structural average
_b_short = int(max(1000, min_length_mm - 300))                   # near structural min (matches MIN_LENGTH_MM)

DONOR_BUILDING_B_MEMBER_TYPES = [
    ("long_rafter",   140, 280, 21, _b_long),   # ×1.5 vs donor A counts → ~150% pool size
    ("mid_beam",      120, 240, 18, _b_upper),
    ("floor_joist",    80, 200, 15, _b_avg),
    ("short_purlin",   70, 150, 12, _b_short),
]

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
DEPTH_WIDTH_COMBINATIONS: List[Tuple[int, int]] = [
    (int(depth), int(width))
    for depth in sorted(DEPTH_WIDTH_MAPPING)
    for width in DEPTH_WIDTH_MAPPING[depth]
]


__all__ = [
    "DEPTH_WIDTH_MAPPING",
    "DEPTH_WIDTH_COMBINATIONS",
    "TUPLE_LENGTHS",
    "generate_length_tuple_from_average",
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