"""
Normalization Constants for Multi-Objective Fitness Function

This module defines the maximum theoretical values (C_max, R_max, W_max) used to normalize
the multi-objective fitness function components to the range [0, 1].

These constants are computed from dataset-driven worst-case scenarios:
  - C_max: Worst-case LCA cost (all virgin timber, maximum waste across all structure beams)
  - R_max: Theoretical maximum reuse (100% reclaimed timber utilization)
  - W_max: Maximum waste scenario (all beams with maximum oversizing + leftover cuts)

Dataset sources:
  - Timber stock: complete_timber.csv (83 items: 55 NS + 28 RS)
  - Geometry: typically 32 members (4×4 grid with cross-bracing)
  - LCA factors: c11_params.py

Extremes identified from complete_timber.csv:
  - Cross-section: Width [38-100] mm, Depth [100-178] mm → Area extremes [3800, 17800] mm²
  - Length: [1500-4000] mm (stored as individual stock lengths)
  - Transport distance: [69.82-1688.82] km
  - ECC (Embodied Carbon Content): [0-150] kg CO2e/m³
  - Emission factor: [0.1724-0.1743] kg CO2e/ton·km
  - Density: 350-420 kg/m³
  - Processing factor: 0-1
"""

# ==========================================
# 1. C_max: WORST-CASE LCA COST (kg CO2e)
# ==========================================
# Scenario: All 32 structure members assigned to virgin timber with maximum waste
#
# Per-beam worst-case cost calculation:
#   E_embodied = V_stock × ECC_max = 0.00076 m³ × 150 = 0.114 kg CO2e
#   E_prep = V_stock × PREP_FACTOR = 0.00076 m³ × 13.3 = 0.0101 kg CO2e
#   E_trans = (V_req + V_over) × ρ × dist × emis_factor
#             Assume: V_req=0.0003 m³, V_over=0.0002 m³, ρ=420, dist=1688.82, emis=0.174
#             = 0.0005 m³ × 420 kg/m³ / 1000 × 1688.82 × 0.174 = 0.062 kg CO2e
#   E_waste = V_waste × EOL_FACTOR
#             V_waste = stock_vol - req_vol ≈ 0.00043 m³ (worst oversizing)
#             = 0.00043 m³ × 12.0 = 0.0052 kg CO2e
#   E_saw = 0.0194 kg CO2e per cut × 1 = 0.0194 kg CO2e
#
# Total per beam: 0.114 + 0.010 + 0.062 + 0.005 + 0.019 ≈ 0.21 kg CO2e
# For 32 beams: 32 × 0.21 ≈ 6.7 kg CO2e (conservative estimate)
#
# Add 20% safety margin for stochastic variations in dimensions/transport
C_MAX_KG_CO2E = 8.0  # kg CO2e total for entire 32-member structure

# ==========================================
# 2. R_max: THEORETICAL MAXIMUM REUSE RATE
# ==========================================
# Scenario: 100% reclaimed timber utilization = all 32 structure members use RS items
#
# Reuse rate definition (count-based):
#   R_max = len(structure) × 100 = 32 × 100 = 3200 (percentage points if all used)
#   Or more intuitively: R_max = 100 (represents 100% reuse)
#
# We'll use normalized definition: R_max = 100 (percentage)
R_MAX_PERCENT = 100.0  # Maximum achievable reuse rate (%)

# ==========================================
# 3. W_max: MAXIMUM WASTE SCENARIO
# ==========================================
# Scenario: All beams cut from smallest available stock (worst efficiency)
#
# Per-beam worst-case waste:
#   Assume typical beam: L_req=2.5 m, b_req=38 mm, d_req=50 mm
#   Worst assignment: stock has L_stock=4.0 m, b_stock=50 mm, d_stock=100 mm
#
#   V_stock = 4.0 × 0.050 × 0.100 = 0.020 m³
#   V_req   = 2.5 × 0.038 × 0.050 = 0.0048 m³
#
#   Oversizing waste: (50-38) × (100-50) × 2.5 mm·mm·m = 0.0015 m³
#   Length waste: 50 × 100 × (4000-2500) mm = 0.0075 m³
#   Total waste per beam: ~0.009 m³
#
# For 32 beams: 32 × 0.009 ≈ 0.29 m³
# Add 30% margin for structural variation: 0.29 × 1.3 ≈ 0.38 m³
W_MAX_M3 = 0.4  # m³ total waste for entire 32-member structure

# ==========================================
# 4. NORMALIZATION HELPER FUNCTIONS
# ==========================================

def get_normalization_constants():
    """
    Return all normalization constants as a dictionary.
    
    Returns:
        dict: Keys are 'C_max', 'R_max', 'W_max' with their respective values
    """
    return {
        'C_max': C_MAX_KG_CO2E,
        'R_max': R_MAX_PERCENT,
        'W_max': W_MAX_M3,
    }


def print_normalization_summary():
    """Print a summary of normalization constants for verification."""
    print("\n" + "=" * 70)
    print("NORMALIZATION CONSTANTS FOR MULTI-OBJECTIVE FITNESS FUNCTION")
    print("=" * 70)
    print(f"\nC_max (worst-case LCA cost):      {C_MAX_KG_CO2E} kg CO2e")
    print(f"  → Represents: 32-member structure with all virgin timber, max waste")
    print(f"\nR_max (theoretical max reuse):    {R_MAX_PERCENT} %")
    print(f"  → Represents: 100% reclaimed timber utilization rate")
    print(f"\nW_max (maximum waste scenario):   {W_MAX_M3} m³")
    print(f"  → Represents: All beams with worst-case oversizing + length waste")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_normalization_summary()
