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
# Source tags: [EPD] EPD literature; [Bergman2010] Bergman et al. 2010 (USDA FPL, SWST/UNECE); [Ecoinvent] Ecoinvent v3; [EN15978] standard default; [est.] engineering estimate (sensitivity analysis in §9).
IMPACT_FACTOR_A1_A3         = 0.25      # [EPD] fossil GWP, kiln-dried softwood structural timber A1–A3; consistent with 0.24–0.26 range across 81 EPDs; substitute NMD/ÖKOBAUDAT entry for project-specific product
IMPACT_FACTOR_RECOVERED_C1  = 0.0085    # [Bergman2010] derived: Table 2 fossil CO2 = 18.9 kg/m³ total; subtract transport (crude oil ~55%) → 8.5 kg/m³ non-transport; ÷490 kg/m³; C1 share (~50%) ≈ 0.0085 kg CO2e/kg; conservative for EU grid
ENERGY_PREP_A5              = 0.01      # [Bergman2010] derived: A5-preparation share (~50%) of non-transport reclaiming ≈ 0.0085–0.010 kg CO2e/kg; 0.010 is mid-estimate, conservative for EU grid; covers de-nailing, cleaning, structural testing
ENERGY_SAW_A5               = 0.004     # [calc.] 1800 W sliding miter saw, 60 s/cut, ref. element 100×50×1800 mm (4.4 kg) → 0.0068 kWh/kg; ×0.35 kg CO2/kWh (NL grid, CBS StatLine 2023) = 0.0024; doubled to 0.004 as conservative upper bound for dust extraction, overhead, and element size variability
ENERGY_OFFCUT_FACTOR_C3_C4  = 0.031     # [Ecoinvent] fossil GWP, waste wood incineration with energy recovery, Western Europe (C3–C4); replaces prior unsourced value of 0.276 which conflated total GWP with fossil GWP
WASTE_TRANSPORT_DIST_KM     = 50        # [EN15978] standard default transport distance to waste facility (C2)

# SCARCITY_PENALTY (ω): artificially inflates the cost of offcut waste so the MILP
# prioritises length-efficient assignments when reclaimed stock is scarce.
# ω=0: pure LCA (no scarcity pressure)
# ω~1–5× EoL penalty: moderate pressure, preserves most original lengths
# ω very high: "do-not-cut" extreme — forces near-zero length waste
SCARCITY_PENALTY = 0

print(f"LCA factors: A1-A3={IMPACT_FACTOR_A1_A3}, C1={IMPACT_FACTOR_RECOVERED_C1}, A5 prep={ENERGY_PREP_A5}, A5 saw={ENERGY_SAW_A5}, C2 dist={WASTE_TRANSPORT_DIST_KM} km, C3-C4={ENERGY_OFFCUT_FACTOR_C3_C4}, ω={SCARCITY_PENALTY}")