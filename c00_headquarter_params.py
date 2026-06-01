GRID_CELLS_X = 5          # cells in X
GRID_CELLS_Y = 3          # cells in Y
EDGE_LENGTH  = 3.0        # cell size [m]
LAYER_HEIGHT = 1.5        # top–bottom layer distance [m]
DIVISIONS    = 8          # discrete shift steps per axis
NUM_SAMPLES  = 20000      # training samples to generate
SCALE_UV     = 0.25, 0.75 # vertex random position range within cell

GRID = f"{GRID_CELLS_X}x{GRID_CELLS_Y}"

IMPACT_FACTOR_A1_A3         = 0.25
IMPACT_FACTOR_RECOVERED_C1  = 0.0085
ENERGY_PREP_A5              = 0.010
ENERGY_SAW_A5               = 0.004
ENERGY_OFFCUT_FACTOR_C3_C4  = 0.031
WASTE_TRANSPORT_DIST_KM     = 50
SCARCITY_PENALTY            = 0