# --- CONFIGURATION ---
GRID_CELLS_X = 2         # Number of cells in X
GRID_CELLS_Y = 2          # Number of cells in Y
EDGE_LENGTH = 3.0         # Size of one cell
LAYER_HEIGHT = 1.5        # Distance between top and bottom layer
DIVISIONS = 8             # Number of steps for discrete shifting
NUM_SAMPLES = 20000       # Number of samples
SCALE_UV = 0.25, 0.75     # Random position in the cell

GRID = f"{GRID_CELLS_X}x{GRID_CELLS_Y}"

# LCA factors for the assignment cost matrix (kg CO2e)
PREPARATION_EMISSION_FACTOR = 13.3     # per m3 of prepared reclaimed stock volume
END_OF_LIFE_EMISSION_FACTOR = 12.0     # per m3 of offcut waste (C3/C4 processing)
SAW_ENERGY_KWH_PER_CUT = 0.05          # kWh per cross-cut
GRID_INTENSITY_KGCO2_PER_KWH = 0.388   # NL grid factor (2025)
SAW_CUT_PENALTY = SAW_ENERGY_KWH_PER_CUT * GRID_INTENSITY_KGCO2_PER_KWH

print(f"parameters loaded from {__file__}")
print(f"GRID: {GRID}, EDGE_LENGTH: {EDGE_LENGTH}, LAYER_HEIGHT: {LAYER_HEIGHT}, DIVISIONS: {DIVISIONS}, NUM_SAMPLES: {NUM_SAMPLES}\n")