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
# Source tags:
# [EPD]        TDUK 2025 (22 EPDs, sawn softwood UK avg 107 kg CO2e/m³ ÷ 471 kg/m³);
#              Puettmann et al. 2013 (CORRIM, PNW softwood 112 kg CO2e/m³ ÷ 476 kg/m³)
# [Bergman2010] Bergman et al. 2010 (USDA FPL, SWST/UNECE proceedings);
#              full journal version: Bergman & Falk 2013 if accessible
# [Ecoinvent]  Wernet et al. 2016, Ecoinvent v3
# [EN15978]    NEN-EN 15978:2026, section 9.9.3
# [calc.]      engineering estimate from first principles
# [assumption] modelling assumption; sensitivity analysis in §9

IMPACT_FACTOR_A1_A3         = 0.25
# [EPD] fossil GWP, kiln-dried structural softwood A1–A3.
# TDUK 2025: 107 kg CO2e/m³ ÷ 471 kg/m³ = 0.227 kg CO2e/kg.
# Puettmann et al. 2013: 112 kg CO2e/m³ ÷ 476 kg/m³ = 0.235 kg CO2e/kg.
# 0.25 is a marginally conservative upper bound consistent with both sources.
# For project-specific application substitute a verified NMD or ÖKOBAUDAT EPD.

IMPACT_FACTOR_RECOVERED_C1  = 0.0085
# [Bergman2010] Table 2: total fossil CO2 intensity = 18.9 kg CO2/m³ for complete
# reclaiming process; at reference density 490 kg/m³ → 0.039 kg CO2e/kg total.
# Crude oil (transport fuel) = 41% of total reclaiming energy → subtract →
# 0.023 kg CO2e/kg non-transport reclaiming impact.
# Apportioned equally between C1 (on-site deconstruction) and A5-prep (yard
# reprocessing) → 0.0085 kg CO2e/kg per stage.
# EU grid is lower-carbon than US baseline → marginally conservative for NL context.

ENERGY_PREP_A5              = 0.010
# [Bergman2010] A5-preparation share of non-transport reclaiming impact (equal
# apportionment with C1 as above) → 0.0085–0.010 kg CO2e/kg.
# 0.010 is the conservative upper bound; covers de-nailing, cleaning, structural
# inspection at reclamation yard. Applied to full stock element mass regardless
# of whether cutting is required.

ENERGY_SAW_A5               = 0.004
# [calc.] 1800 W sliding compound mitre saw, 60 s/cut, reference softwood element
# 100×50×1800 mm (density 490 kg/m³, mass 4.4 kg) → 0.030 kWh/cut → 0.0068 kWh/kg.
# × 0.483 kg CO2e/kWh (NL grey grid, CO2emissiefactoren.nl 2026) = 0.0033 kg CO2e/kg.
# Rounded up to 0.004 as conservative upper bound for dust extraction, workshop
# overhead, and element size variability.
# Applied only where L_stock > L_req (cutting required).

ENERGY_OFFCUT_FACTOR_C3_C4  = 0.031
# [Ecoinvent] Wernet et al. 2016, Ecoinvent v3.
# Dataset: treatment of waste wood, municipal incineration with energy recovery,
# Western Europe. Covers fossil GWP of chipping, transport to waste-to-energy
# plant, and incineration. Biogenic CO2 excluded (fossil-GWP-only convention).
# Replaces prior value of 0.276 which conflated total GWP (incl. biogenic) with
# fossil-only GWP for an inappropriate disposal pathway.

WASTE_TRANSPORT_DIST_KM     = 50
# [assumption] Default distance from construction site to waste disposal facility.
# Modelling assumption consistent with common Dutch LCA practice.
# EN 15978:2026 section 9.9.3 requires realistic local distances to be specified
# but does not prescribe a fixed default.
# Applied to offcut mass using the element's own transport emission factor.
# Replace with actual site-to-facility distance for project-specific assessment.

# SCARCITY_PENALTY (ω): artificially inflates the cost of offcut waste so the MILP
# prioritises length-efficient assignments when reclaimed stock is scarce.
# ω=0: pure LCA (no scarcity pressure)
# ω~1–5× EoL penalty: moderate pressure, preserves most original lengths
# ω very high: "do-not-cut" extreme — forces near-zero length waste
SCARCITY_PENALTY = 0

print(f"LCA factors: A1-A3={IMPACT_FACTOR_A1_A3}, C1={IMPACT_FACTOR_RECOVERED_C1}, A5 prep={ENERGY_PREP_A5}, A5 saw={ENERGY_SAW_A5}, C2 dist={WASTE_TRANSPORT_DIST_KM} km, C3-C4={ENERGY_OFFCUT_FACTOR_C3_C4}, ω={SCARCITY_PENALTY}")