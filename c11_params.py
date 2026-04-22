# --- CONFIGURATION ---
GRID_CELLS_X = 2         # Number of cells in X
GRID_CELLS_Y = 2          # Number of cells in Y
EDGE_LENGTH = 3.0         # Size of one cell
LAYER_HEIGHT = 1.5        # Distance between top and bottom layer
DIVISIONS = 8             # Number of steps for discrete shifting
NUM_SAMPLES = 20000       # Number of samples
SCALE_UV = 0.25, 0.75     # Random position in the cell

GRID = f"{GRID_CELLS_X}x{GRID_CELLS_Y}"

print(f"parameters loaded from {__file__}")
print(f"GRID: {GRID}, EDGE_LENGTH: {EDGE_LENGTH}, LAYER_HEIGHT: {LAYER_HEIGHT}, DIVISIONS: {DIVISIONS}, NUM_SAMPLES: {NUM_SAMPLES}\n")