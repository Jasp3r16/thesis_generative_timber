# =====================================
# GEOMETRY CONFIGURATION
# =====================================
GRID_CELLS_X = 5          # cells in X
GRID_CELLS_Y = 3          # cells in Y
EDGE_LENGTH  = 3.0        # cell size [m]
LAYER_HEIGHT = 1.5        # top–bottom layer distance [m]
DIVISIONS    = 8          # discrete shift steps per axis
NUM_SAMPLES  = 20000      # training samples to generate
SCALE_UV     = 0.25, 0.75 # vertex random position range within cell

GRID = f"{GRID_CELLS_X}x{GRID_CELLS_Y}"

print(f"Grid: {GRID}, edge={EDGE_LENGTH} m, height={LAYER_HEIGHT} m, divisions={DIVISIONS}, samples={NUM_SAMPLES}")

# =====================================
# COST MATRIX VALUES
# =====================================
# All emission factors in kg CO2e / kg unless noted.
IMPACT_FACTOR_A1_A3      = 0.25    # fossil GWP of forestry, sawmilling, kiln-drying (new timber, modules A1–A3)
IMPACT_FACTOR_RECOVERED_C1 = 0.0085 # selective deconstruction energy penalty (module C1)
ENERGY_PREP_A5           = 0.01    # cleaning, de-nailing, structural testing (always applies to reclaimed, module A5)
ENERGY_SAW_A5            = 0.01    # secondary cross-cut resizing (only when stk_length > req_length, module A5)
ENERGY_OFFCUT_FACTOR_C3_C4 = 0.276 # environmental penalty for geometric offcut waste (modules C3–C4)
WASTE_TRANSPORT_DIST_KM  = 50      # estimated distance from site to waste disposal facility (module C2)

# SCARCITY_PENALTY (ω): artificially inflates the cost of offcut waste so the MILP
# prioritises length-efficient assignments when reclaimed stock is scarce.
# ω=0: pure LCA (no scarcity pressure)
# ω~1–5× EoL penalty: moderate pressure, preserves most original lengths
# ω very high: "do-not-cut" extreme — forces near-zero length waste
SCARCITY_PENALTY = 0

print(f"LCA factors: A1-A3={IMPACT_FACTOR_A1_A3}, C1={IMPACT_FACTOR_RECOVERED_C1}, A5 prep={ENERGY_PREP_A5}, A5 saw={ENERGY_SAW_A5}, C2 dist={WASTE_TRANSPORT_DIST_KM} km, C3-C4={ENERGY_OFFCUT_FACTOR_C3_C4}, ω={SCARCITY_PENALTY}")