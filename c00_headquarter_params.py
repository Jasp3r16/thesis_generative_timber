# =====================================
# GEOMETRY CONFIGURATION
# =====================================
GRID_CELLS_X = 5         # Number of cells in X
GRID_CELLS_Y = 3          # Number of cells in Y
EDGE_LENGTH = 3.0         # Size of one cell
LAYER_HEIGHT = 1.5        # Distance between top and bottom layer
DIVISIONS = 8             # Number of steps for discrete shifting
NUM_SAMPLES = 20000       # Number of samples
SCALE_UV = 0.25, 0.75     # Random position in the cell

GRID = f"{GRID_CELLS_X}x{GRID_CELLS_Y}"

print(f"GRID: {GRID}, EDGE_LENGTH: {EDGE_LENGTH}, LAYER_HEIGHT: {LAYER_HEIGHT}, DIVISIONS: {DIVISIONS}, NUM_SAMPLES: {NUM_SAMPLES}\n")

# ====================================
# COST MATRIX VALUES
# ====================================
IMPACT_FACTOR_A1_A3 = 0.25                     #  kg CO2e / kg. This represents the fossil Global Warming Potential of forestry, sawmilling, and kiln-drying for standard softwood
IMPACT_FACTOR_RECOVERED_C1 = 0.0085            # represents the energy and emissions associated with selective deconstruction (Module C1)
ENERGY_PREP_SAW_A5 = 0.02                   # accounts for cleaning, denailing, and structural testing of the salvaged element, represents the energy required for resizing the salvaged beam to fit the new topology slot.
ENERGY_OFFCUT_FACTOR_C3_C4 = 0.276             # represents the environmental penalty for the geometric waste generated during resizing.

SCARCITY_PENALTY = 0                        # allowing the designer to artificially inflate the "cost" of geometric waste so the solver is forced to pay attention to it

print(f"""IMPACT_FACTOR_A1_A3: {IMPACT_FACTOR_A1_A3}, IMPACT_FACTOR_RECOVERED_C1: {IMPACT_FACTOR_RECOVERED_C1}, "ENERGY_PREP_SAW_A5: {ENERGY_PREP_SAW_A5}, ENERGY_OFFCUT_FACTOR_C3_C4: {ENERGY_OFFCUT_FACTOR_C3_C4}, SCARCITY_PENALTY: {SCARCITY_PENALTY}\n""")
'''
- The Lower Bound: ω=0 (Pure LCA) If you set ω=0, you are running a pure Life Cycle Assessment optimization.
- Moderate Scarcity: The Multiplier Approach (1× to 5×M EoL) A highly effective way to set a range is to scale ω relative to your actual End-of-Life disposal penalty (MEoL).
- High Scarcity: The "Do Not Cut" Extreme Range If your primary design goal is strict circularity—meaning local reclaimed wood is 
considered extremely scarce and you want to preserve the stock's original lengths for future uses—you must set ω very high
'''

print(f"parameters loaded from {__file__}")