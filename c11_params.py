# --- CONFIGURATIE ---
GRID_CELLS_X = 2          # Aantal cellen in X
GRID_CELLS_Y = 2          # Aantal cellen in Y
EDGE_LENGTH = 3.0       # Afmeting van een cel
LAYER_HEIGHT = 1.5      # Afstand tussen top en bottom layer
DIVISIONS = 8             # Aantal stappen voor de discrete verplaatsing
NUM_SAMPLES = 10000       # Aantal samples
SCALE_UV = 0.25, 0.75        # Random positie in de cel

GRID = f"{GRID_CELLS_X}x{GRID_CELLS_Y}"

# LCA factoren voor de assignment cost matrix (kg CO2e)
PREPARATION_EMISSION_FACTOR = 13.3     # per m3 reclaimed stock voorbereid volume
END_OF_LIFE_EMISSION_FACTOR = 12.0     # per m3 offcut waste (C3/C4 verwerking)
SAW_ENERGY_KWH_PER_CUT = 0.05          # kWh per cross-cut
GRID_INTENSITY_KGCO2_PER_KWH = 0.388   # NL netfactor (2025)
SAW_CUT_PENALTY = SAW_ENERGY_KWH_PER_CUT * GRID_INTENSITY_KGCO2_PER_KWH