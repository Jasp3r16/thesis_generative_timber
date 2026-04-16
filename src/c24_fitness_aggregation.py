"""
Multi-Objective Fitness Function Aggregation Module

This module provides functions to:
1. Extract and calculate key metrics from MILP solver results (reuse rate, waste)
2. Normalize multi-objective metrics to [0, 1] range
3. Apply weighted multi-objective optimization formula:
   
   F(x) = ω₁ · (f_inner* / C_max) - ω₂ · (R / R_max) + ω₃ · (W / W_max)

where:
  - f_inner* = MILP objective value (kg CO2e)
  - R = reuse rate (% of reclaimed timber used)
  - W = total waste (m³)
  - ω₁, ω₂, ω₃ = weight coefficients for balancing priorities

Key insight: The formula subtracts reuse rate (negative slope) because:
  - Higher reuse is BETTER, so it should reduce (improve) the fitness value
  - Higher cost is WORSE, so it should increase (worsen) the fitness value
  - Higher waste is WORSE, so it should increase (worsen) the fitness value
"""

import pandas as pd
import numpy as np
from c24_normalization_constants import get_normalization_constants


# ==========================================
# 1. METRIC EXTRACTION FUNCTIONS
# ==========================================

def calculate_reuse_rate(milp_results_df: pd.DataFrame, 
                         enriched_stock_df: pd.DataFrame,
                         mode: str = 'count') -> float:
    """
    Calculate the reuse rate from MILP solver results.
    
    Args:
        milp_results_df: DataFrame with columns [edge_id, assigned_timber, CO2_Penalty]
                         (output from MILP solver)
        enriched_stock_df: DataFrame with columns including Member_ID 
                           (timber inventory metadata)
        mode: 'count' (default) = percentage of RS items used
              'mass' = mass-based ratio of RS items / total structure mass
    
    Returns:
        float: Reuse rate as percentage (0-100)
    
    Examples:
        If structure has 32 members and 20 are assigned RS (reclaimed) timber:
          → count mode returns: 20/32 × 100 = 62.5%
    """
    if milp_results_df.empty:
        return 0.0
    
    # Extract timber IDs from assignment results
    assigned_timber_ids = milp_results_df['assigned_timber'].unique()
    
    # Count RS (reclaimed) items in assignment
    rs_count = sum(1 for tid in assigned_timber_ids if 'RS' in str(tid))
    total_assignments = len(milp_results_df)
    
    if mode == 'count':
        # Count-based reuse rate: percentage of members with reclaimed timber
        reuse_rate = (rs_count / total_assignments) * 100.0 if total_assignments > 0 else 0.0
        return reuse_rate
    
    elif mode == 'mass':
        # Mass-based reuse rate (requires extracting density from enriched_stock_df)
        # For future extension: incorporate timber masses from inventory
        # This mode requires additional data mapping
        raise NotImplementedError("Mass-based reuse rate requires timber mass mapping. "
                                  "Use mode='count' for now.")
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'count' or 'mass'.")


def calculate_total_waste(milp_results_df: pd.DataFrame,
                         df_slots: pd.DataFrame,
                         stock_inventory_df: pd.DataFrame) -> float:
    """
    Calculate total waste volume from MILP assignment results.
    
    This sums waste from two sources:
    1. Oversizing: (cross_section_stock - cross_section_required) × length_required
    2. Length waste: cross_section_stock × (length_stock - length_required)
    
    Args:
        milp_results_df: DataFrame with [edge_id, assigned_timber, CO2_Penalty]
        df_slots: DataFrame with slot requirements [edge_id, Length_Req, Width_Req, Depth_Req]
        stock_inventory_df: Timber inventory [Member_ID, Length, Width, Depth, ...]
    
    Returns:
        float: Total waste volume in m³
    
    Notes:
        - This complements the MILP cost matrix, which already accounts for waste cost
        - Here we compute actual waste volume as a separate objective metric
        - Range is typically 0-0.4 m³ for a 32-member structure
    """
    if milp_results_df.empty:
        return 0.0
    
    total_waste_m3 = 0.0
    
    for _, assignment in milp_results_df.iterrows():
        edge_id = assignment['edge_id']
        timber_id = assignment['assigned_timber']
        
        # Look up slot requirements
        slot_row = df_slots[df_slots['edge_id'] == edge_id]
        if slot_row.empty:
            continue
        
        l_req = slot_row['Length_Req'].values[0] / 1000.0  # mm → m
        w_req = slot_row['Width_Req'].values[0] / 1000.0   # mm → m
        d_req = slot_row['Depth_Req'].values[0] / 1000.0   # mm → m
        
        # Look up stock inventory
        stock_row = stock_inventory_df[stock_inventory_df['Member_ID'] == timber_id]
        if stock_row.empty:
            continue
        
        l_stock = stock_row['Length'].values[0] / 1000.0   # mm → m
        w_stock = stock_row['Width'].values[0] / 1000.0    # mm → m
        d_stock = stock_row['Depth'].values[0] / 1000.0    # mm → m
        
        # Calculate waste components
        # 1. Oversizing: (stock cross-section - required cross-section) × required length
        oversizing_waste = max(0.0, (w_stock * d_stock - w_req * d_req) * l_req)
        
        # 2. Length waste: stock cross-section × (stock length - required length)
        length_waste = max(0.0, w_stock * d_stock * (l_stock - l_req))
        
        # Total waste for this assignment
        waste_m3 = oversizing_waste + length_waste
        total_waste_m3 += waste_m3
    
    return total_waste_m3


def get_inner_cost(milp_result: float) -> float:
    """
    Extract the MILP objective value (inner cost).
    
    Args:
        milp_result: Total cost from MILP solver (kg CO2e)
    
    Returns:
        float: Cost value in kg CO2e
    """
    return float(milp_result)


# ==========================================
# 2. NORMALIZATION FUNCTIONS
# ==========================================

def normalize_metrics(cost: float, 
                     reuse_rate: float, 
                     waste: float,
                     normalization_constants: dict = None) -> tuple:
    """
    Normalize all metrics to [0, 1] range using precomputed maximum values.
    
    Args:
        cost: MILP objective value (kg CO2e)
        reuse_rate: Reuse rate (%)
        waste: Total waste (m³)
        normalization_constants: Dict with keys 'C_max', 'R_max', 'W_max'.
                                If None, loads from normalization_constants module.
    
    Returns:
        tuple: (cost_norm, reuse_norm, waste_norm) all in [0, 1]
    
    Examples:
        >>> cost_n, reuse_n, waste_n = normalize_metrics(
        ...     cost=4.0, reuse_rate=50.0, waste=0.2)
        >>> print(f"Normalized: cost={cost_n:.3f}, reuse={reuse_n:.3f}, waste={waste_n:.3f}")
        Normalized: cost=0.500, reuse=0.500, waste=0.500
    """
    if normalization_constants is None:
        normalization_constants = get_normalization_constants()
    
    C_max = normalization_constants['C_max']
    R_max = normalization_constants['R_max']
    W_max = normalization_constants['W_max']
    
    # Normalize: clamp to [0, 1] to handle edge cases
    cost_norm = np.clip(cost / C_max, 0.0, 1.0)
    reuse_norm = np.clip(reuse_rate / R_max, 0.0, 1.0)
    waste_norm = np.clip(waste / W_max, 0.0, 1.0)
    
    return cost_norm, reuse_norm, waste_norm


# ==========================================
# 3. WEIGHTED MULTI-OBJECTIVE FITNESS FUNCTION
# ==========================================

def fitness_function_multi_objective(cost_norm: float,
                                     reuse_norm: float,
                                     waste_norm: float,
                                     weights: tuple = None) -> float:
    """
    Compute the weighted multi-objective fitness score.
    
    Formula:
        F(x) = ω₁ · cost_norm - ω₂ · reuse_norm + ω₃ · waste_norm
    
    Rationale for sign convention:
      - cost_norm: positive → HIGHER cost is WORSE (penalizes virgin timber + waste in MILP)
      - reuse_norm: negative → HIGHER reuse is BETTER (reward reclaimed material usage)
      - waste_norm: positive → HIGHER waste is WORSE (penalizes inefficiency)
    
    Args:
        cost_norm: Normalized MILP cost (0-1)
        reuse_norm: Normalized reuse rate (0-1)
        waste_norm: Normalized waste volume (0-1)
        weights: Tuple (ω₁, ω₂, ω₃). Defaults to (1.0, 0.5, 0.5) if None.
                 - ω₁: weight on cost (default 1.0 = cost-primary driver)
                 - ω₂: weight on reuse (default 0.5 = moderate priority)
                 - ω₃: weight on waste (default 0.5 = moderate priority)
    
    Returns:
        float: Fitness value (scalar to be minimized by EA)
    
    Examples:
        Low fitness (good design):
          - cost_norm=0.3, reuse_norm=1.0 (100% reclaimed), waste_norm=0.2
          - F = 1.0×0.3 - 0.5×1.0 + 0.5×0.2 = 0.3 - 0.5 + 0.1 = -0.1 (NEGATIVE = good)
        
        High fitness (poor design):
          - cost_norm=0.8, reuse_norm=0.0 (all virgin), waste_norm=0.8
          - F = 1.0×0.8 - 0.5×0.0 + 0.5×0.8 = 0.8 - 0 + 0.4 = 1.2 (POSITIVE = bad)
    
    Notes:
        - Typical range: [-0.5, 1.5] when weights are (1.0, 0.5, 0.5)
        - Negative fitness = design favored reclaimed timber successfully
        - Positive fitness = design relies on virgin timber or waste
    """
    if weights is None:
        weights = (1.0, 0.5, 0.5)  # Default: cost-dominant
    
    omega_1, omega_2, omega_3 = weights
    
    fitness = (omega_1 * cost_norm 
               - omega_2 * reuse_norm 
               + omega_3 * waste_norm)
    
    return fitness


# ==========================================
# 4. CONVENIENCE WRAPPER
# ==========================================

def evaluate_milp_solution(milp_results_df: pd.DataFrame,
                          enriched_stock_df: pd.DataFrame,
                          df_slots: pd.DataFrame,
                          milp_objective_value: float,
                          weights: tuple = None,
                          normalization_constants: dict = None) -> dict:
    """
    All-in-one wrapper: extract metrics, normalize, and compute fitness.
    
    This is the primary interface for obtaining a fitness score from raw MILP output.
    
    Args:
        milp_results_df: Assignment results from MILP [edge_id, assigned_timber, CO2_Penalty]
        enriched_stock_df: Timber inventory with Member_ID
        df_slots: Geometry slots with required dimensions
        milp_objective_value: Total cost from MILP solver (kg CO2e)
        weights: (ω₁, ω₂, ω₃) - see fitness_function_multi_objective()
        normalization_constants: From normalization_constants module if None
    
    Returns:
        dict with keys:
          - 'fitness': scalar fitness value
          - 'cost_raw': raw MILP cost (kg CO2e)
          - 'reuse_rate': reuse percentage (%)
          - 'waste_total': total waste (m³)
          - 'cost_norm': normalized cost (0-1)
          - 'reuse_norm': normalized reuse (0-1)
          - 'waste_norm': normalized waste (0-1)
          - 'weights': tuple of applied weights
    
    Example:
        >>> result = evaluate_milp_solution(
        ...     milp_results_df=df_results,
        ...     enriched_stock_df=enriched_stock,
        ...     df_slots=df_slots,
        ...     milp_objective_value=4.5,
        ...     weights=(1.0, 0.8, 0.6))
        >>> print(f"Fitness: {result['fitness']:.3f}, Reuse: {result['reuse_rate']:.1f}%")
    """
    # Extract metrics
    reuse_rate = calculate_reuse_rate(milp_results_df, enriched_stock_df, mode='count')
    waste_total = calculate_total_waste(milp_results_df, df_slots, enriched_stock_df)
    cost_raw = get_inner_cost(milp_objective_value)
    
    # Normalize
    cost_norm, reuse_norm, waste_norm = normalize_metrics(
        cost_raw, reuse_rate, waste_total, normalization_constants)
    
    # Compute fitness
    fitness = fitness_function_multi_objective(
        cost_norm, reuse_norm, waste_norm, weights)
    
    return {
        'fitness': fitness,
        'cost_raw': cost_raw,
        'reuse_rate': reuse_rate,
        'waste_total': waste_total,
        'cost_norm': cost_norm,
        'reuse_norm': reuse_norm,
        'waste_norm': waste_norm,
        'weights': weights if weights else (1.0, 0.5, 0.5),
    }


# ==========================================
# 5. DEBUGGING & VISUALIZATION HELPERS
# ==========================================

def print_fitness_breakdown(result: dict) -> None:
    """
    Pretty-print a fitness result dictionary for debugging.
    
    Args:
        result: Dict returned from evaluate_milp_solution()
    """
    print("\n" + "=" * 70)
    print("MULTI-OBJECTIVE FITNESS EVALUATION")
    print("=" * 70)
    
    print(f"\nRaw Metrics:")
    print(f"  MILP Cost:        {result['cost_raw']:>8.3f} kg CO2e")
    print(f"  Reuse Rate:       {result['reuse_rate']:>8.1f} %")
    print(f"  Total Waste:      {result['waste_total']:>8.4f} m³")
    
    print(f"\nNormalized (0-1 range):")
    print(f"  Cost (norm):      {result['cost_norm']:>8.3f}")
    print(f"  Reuse (norm):     {result['reuse_norm']:>8.3f}")
    print(f"  Waste (norm):     {result['waste_norm']:>8.3f}")
    
    omega_1, omega_2, omega_3 = result['weights']
    print(f"\nWeights Applied:")
    print(f"  ω₁ (cost):        {omega_1:>8.3f}")
    print(f"  ω₂ (reuse):       {omega_2:>8.3f}")
    print(f"  ω₃ (waste):       {omega_3:>8.3f}")
    
    print(f"\nWeighted Components:")
    term1 = omega_1 * result['cost_norm']
    term2 = omega_2 * result['reuse_norm']
    term3 = omega_3 * result['waste_norm']
    print(f"  ω₁ × cost:        {term1:>8.3f}")
    print(f"  ω₂ × reuse:       {term2:>8.3f} (subtracted)")
    print(f"  ω₃ × waste:       {term3:>8.3f}")
    
    print(f"\nFinal Fitness:")
    print(f"  F(x) = {term1:.3f} - {term2:.3f} + {term3:.3f}")
    print(f"  F(x) = {result['fitness']:>8.3f}")
    print(f"\nInterpretation:")
    if result['fitness'] < -0.5:
        print(f"  ✓ EXCELLENT: Favors reclaimed timber with low waste")
    elif result['fitness'] < 0.0:
        print(f"  ✓ GOOD: Balanced multi-objective design")
    elif result['fitness'] < 0.5:
        print(f"  ⚠ FAIR: Moderate virgin timber usage")
    else:
        print(f"  ✗ POOR: Heavy virgin material dependence")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Multi-Objective Fitness Aggregation Module")
    print("This module is imported by c24_ceo_optimizer.ipynb")
    print("\nKey functions:")
    print("  - calculate_reuse_rate()")
    print("  - calculate_total_waste()")
    print("  - normalize_metrics()")
    print("  - fitness_function_multi_objective()")
    print("  - evaluate_milp_solution()  [main wrapper]")
    print("  - print_fitness_breakdown() [debugging]")
