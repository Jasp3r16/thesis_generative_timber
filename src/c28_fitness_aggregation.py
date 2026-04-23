"""Fitness aggregation utilities for the MILP workflow."""

import numpy as np
import pandas as pd

def derive_normalization_constants_from_solution(
    milp_results_df: pd.DataFrame,
    enriched_stock_df: pd.DataFrame,
    df_slots: pd.DataFrame,
    milp_objective_value: float,
    margin: float = 0.20,
) -> dict:
    """
    Derive normalization constants from the current solution with a margin.

    This avoids external files and keeps normalization inputs reproducible inside
    notebook/function execution.
    """
    if margin < 0:
        raise ValueError("margin must be >= 0")

    cost_raw = get_inner_cost(milp_objective_value)
    reuse_rate = calculate_reuse_rate(milp_results_df, enriched_stock_df, mode='count')
    waste_total = calculate_total_waste(milp_results_df, df_slots, enriched_stock_df)

    scale = 1.0 + float(margin)
    c_max = max(float(cost_raw) * scale, 1e-9)
    r_max = max(float(reuse_rate) * scale, 100.0)
    w_max = max(float(waste_total) * scale, 1e-9)

    return {
        'C_max': c_max,
        'R_max': r_max,
        'W_max': w_max,
    }


def get_weight_config(strategy: str = 'cost-dominant') -> dict:
    """Return a standard weight configuration dictionary."""
    if strategy not in WEIGHT_STRATEGIES:
        available = ", ".join(sorted(WEIGHT_STRATEGIES.keys()))
        raise ValueError(f"Unknown strategy '{strategy}'. Available: {available}")

    omega_1, omega_2, omega_3 = WEIGHT_STRATEGIES[strategy]
    return {
        'omega_1': float(omega_1),
        'omega_2': float(omega_2),
        'omega_3': float(omega_3),
        'strategy': strategy,
    }


def weights_from_config(weight_config: dict) -> tuple:
    """Convert a weight config dictionary to the (omega_1, omega_2, omega_3) tuple."""
    required = ('omega_1', 'omega_2', 'omega_3')
    missing = [key for key in required if key not in weight_config]
    if missing:
        raise ValueError(f"Missing keys in weight_config: {', '.join(missing)}")
    return (
        float(weight_config['omega_1']),
        float(weight_config['omega_2']),
        float(weight_config['omega_3']),
    )


def run_fitness_sanity_checks(normalization_constants: dict = None) -> dict:
    """Run quick checks for normalization and fitness ordering."""
    if normalization_constants is None:
        normalization_constants = get_normalization_constants()

    c_max = float(normalization_constants['C_max'])
    r_max = float(normalization_constants['R_max'])
    w_max = float(normalization_constants['W_max'])

    cost_1, reuse_1, waste_1 = normalize_metrics(
        cost=0.5 * c_max,
        reuse_rate=0.5 * r_max,
        waste=0.5 * w_max,
        normalization_constants=normalization_constants,
    )
    assert 0.45 < cost_1 < 0.55
    assert 0.45 < reuse_1 < 0.55
    assert 0.45 < waste_1 < 0.55

    default_weights = WEIGHT_STRATEGIES['cost-dominant']
    fitness_excellent = fitness_function_multi_objective(
        cost_norm=0.25,
        reuse_norm=1.0,
        waste_norm=0.1,
        weights=default_weights,
    )
    fitness_poor = fitness_function_multi_objective(
        cost_norm=0.8,
        reuse_norm=0.0,
        waste_norm=0.9,
        weights=default_weights,
    )
    assert fitness_excellent < 0.0
    assert fitness_poor > 0.5

    return {
        'normalization_mid_range': {
            'cost_norm': float(cost_1),
            'reuse_norm': float(reuse_1),
            'waste_norm': float(waste_1),
        },
        'fitness_ordering': {
            'excellent': float(fitness_excellent),
            'poor': float(fitness_poor),
            'passes': bool(fitness_excellent < fitness_poor),
        },
    }

def calculate_reuse_rate(milp_results_df: pd.DataFrame,
                         enriched_stock_df: pd.DataFrame,
                         mode: str = 'count') -> float:
    """Calculate reuse rate as percentage of assigned reclaimed members."""
    _ = enriched_stock_df  # kept for API compatibility
    if milp_results_df.empty:
        return 0.0

    assigned_timber_ids = milp_results_df['assigned_timber'].unique()
    rs_count = sum(1 for tid in assigned_timber_ids if 'RS' in str(tid))
    total_assignments = len(milp_results_df)

    if mode == 'count':
        return (rs_count / total_assignments) * 100.0 if total_assignments > 0 else 0.0
    if mode == 'mass':
        raise NotImplementedError("Mass-based reuse rate requires timber mass mapping. "
                                  "Use mode='count' for now.")
    raise ValueError(f"Unknown mode: {mode}. Use 'count' or 'mass'.")


def calculate_total_waste(milp_results_df: pd.DataFrame,
                         df_slots: pd.DataFrame,
                         stock_inventory_df: pd.DataFrame) -> float:
    """Calculate total waste volume (m3) from MILP assignments."""
    if milp_results_df.empty:
        return 0.0

    total_waste_m3 = 0.0

    for _, assignment in milp_results_df.iterrows():
        edge_id = assignment['edge_id']
        timber_id = assignment['assigned_timber']

        slot_row = df_slots[df_slots['edge_id'] == edge_id]
        if slot_row.empty:
            continue

        l_req = slot_row['Length_Req'].values[0] / 1000.0
        w_req = slot_row['Width_Req'].values[0] / 1000.0
        d_req = slot_row['Depth_Req'].values[0] / 1000.0

        stock_row = stock_inventory_df[stock_inventory_df['Member_ID'] == timber_id]
        if stock_row.empty:
            continue

        l_stock = stock_row['Length'].values[0] / 1000.0
        w_stock = stock_row['Width'].values[0] / 1000.0
        d_stock = stock_row['Depth'].values[0] / 1000.0

        oversizing_waste = max(0.0, (w_stock * d_stock - w_req * d_req) * l_req)
        length_waste = max(0.0, w_stock * d_stock * (l_stock - l_req))

        total_waste_m3 += oversizing_waste + length_waste

    return total_waste_m3


def get_inner_cost(milp_result: float) -> float:
    """Return MILP objective value as float."""
    return float(milp_result)

def normalize_metrics(cost: float, 
                     reuse_rate: float, 
                     waste: float,
                     normalization_constants: dict = None) -> tuple:
    """Normalize cost, reuse rate, and waste to [0, 1]."""
    if normalization_constants is None:
        normalization_constants = get_normalization_constants()

    C_max = normalization_constants['C_max']
    R_max = normalization_constants['R_max']
    W_max = normalization_constants['W_max']

    cost_norm = np.clip(cost / C_max, 0.0, 1.0)
    reuse_norm = np.clip(reuse_rate / R_max, 0.0, 1.0)
    waste_norm = np.clip(waste / W_max, 0.0, 1.0)

    return cost_norm, reuse_norm, waste_norm

def fitness_function_multi_objective(cost_norm: float,
                                     reuse_norm: float,
                                     waste_norm: float,
                                     weights: tuple = None) -> float:
    """Compute weighted fitness: omega1*cost - omega2*reuse + omega3*waste."""
    if weights is None:
        weights = WEIGHT_STRATEGIES['cost-dominant']

    omega_1, omega_2, omega_3 = weights

    return (omega_1 * cost_norm - omega_2 * reuse_norm + omega_3 * waste_norm)

def evaluate_milp_solution(milp_results_df: pd.DataFrame,
                          enriched_stock_df: pd.DataFrame,
                          df_slots: pd.DataFrame,
                          milp_objective_value: float,
                          weights: tuple = None,
                          normalization_constants: dict = None) -> dict:
    """Extract metrics, normalize them, and return fitness plus breakdown."""
    reuse_rate = calculate_reuse_rate(milp_results_df, enriched_stock_df, mode='count')
    waste_total = calculate_total_waste(milp_results_df, df_slots, enriched_stock_df)
    cost_raw = get_inner_cost(milp_objective_value)

    cost_norm, reuse_norm, waste_norm = normalize_metrics(
        cost_raw, reuse_rate, waste_total, normalization_constants)

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

def print_fitness_breakdown(result: dict) -> None:
    """Pretty-print a fitness result dictionary for debugging."""
    print("\n" + "=" * 70)
    print("MULTI-OBJECTIVE FITNESS EVALUATION")
    print("=" * 70)
    
    print("\nRaw Metrics:")
    print(f"  MILP Cost:        {result['cost_raw']:>8.3f} kg CO2e")
    print(f"  Reuse Rate:       {result['reuse_rate']:>8.1f} %")
    print(f"  Total Waste:      {result['waste_total']:>8.4f} m³")
    
    print("\nNormalized (0-1 range):")
    print(f"  Cost (norm):      {result['cost_norm']:>8.3f}")
    print(f"  Reuse (norm):     {result['reuse_norm']:>8.3f}")
    print(f"  Waste (norm):     {result['waste_norm']:>8.3f}")
    
    omega_1, omega_2, omega_3 = result['weights']
    print("\nWeights Applied:")
    print(f"  ω₁ (cost):        {omega_1:>8.3f}")
    print(f"  ω₂ (reuse):       {omega_2:>8.3f}")
    print(f"  ω₃ (waste):       {omega_3:>8.3f}")
    
    print("\nWeighted Components:")
    term1 = omega_1 * result['cost_norm']
    term2 = omega_2 * result['reuse_norm']
    term3 = omega_3 * result['waste_norm']
    print(f"  ω₁ × cost:        {term1:>8.3f}")
    print(f"  ω₂ × reuse:       {term2:>8.3f} (subtracted)")
    print(f"  ω₃ × waste:       {term3:>8.3f}")
    
    print("\nFinal Fitness:")
    print(f"  F(x) = {term1:.3f} - {term2:.3f} + {term3:.3f}")
    print(f"  F(x) = {result['fitness']:>8.3f}")
    print("\nInterpretation:")
    if result['fitness'] < -0.5:
        print("  [EXCELLENT] Favors reclaimed timber with low waste")
    elif result['fitness'] < 0.0:
        print("  [GOOD] Balanced multi-objective design")
    elif result['fitness'] < 0.5:
        print("  [FAIR] Moderate virgin timber usage")
    else:
        print("  [POOR] Heavy virgin material dependence")
    
    print("=" * 70 + "\n")
