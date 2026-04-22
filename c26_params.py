# LCA factors for the assignment cost matrix (kg CO2e)
PREPARATION_EMISSION_FACTOR = 13.3     # per m3 of prepared reclaimed stock volume
END_OF_LIFE_EMISSION_FACTOR = 12.0     # per m3 of offcut waste (C3/C4 processing)
SAW_ENERGY_KWH_PER_CUT = 0.05          # kWh per cross-cut
GRID_INTENSITY_KGCO2_PER_KWH = 0.388   # NL grid factor (2025)
SAW_CUT_PENALTY = SAW_ENERGY_KWH_PER_CUT * GRID_INTENSITY_KGCO2_PER_KWH
SCARCITY_PENALTY = 0.5                 # allowing the designer to artificially inflate the "cost" of geometric waste so the solver is forced to pay attention to it
