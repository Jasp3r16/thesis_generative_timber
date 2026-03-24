# --- CONFIGURATIE ---
GRID_CELLS_X = 1          # Aantal cellen in X
GRID_CELLS_Y = 1          # Aantal cellen in Y
EDGE_LENGTH = 3.0       # Afmeting van een cel
LAYER_HEIGHT = 1.5      # Afstand tussen top en bottom layer
DIVISIONS = 8             # Aantal stappen voor de discrete verplaatsing
NUM_SAMPLES = 10000       # Aantal samples
SCALE_UV = 0.25, 0.75        # Random positie in de cel

GRID = f"{GRID_CELLS_X}x{GRID_CELLS_Y}"

# --- PARAMETERS VOOR HOUT EN LCA ---
# GWP Waarden (kg CO2 eq / kg hout)
GWP_VIRGIN = 0.50
GWP_RECLAIMED = 0.08

# LCA Parameters voor E_cost
PREPARATION_EMISSION_FACTOR = 13.3  # kg CO2 boete voor bewerkingen (bijv. ontspijkeren)