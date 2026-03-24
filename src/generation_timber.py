"""
Timber Stock Generation Module
Generates new and reclaimed timber inventories from parameters.
"""

import itertools
import random
from typing import Dict, Any
import numpy as np
import pandas as pd


def _get_params_module():
    """Lazy import to avoid circular dependencies with c22_params."""
    import c22_params as params
    return params


def generate_length_tuple_from_average(
    mean_length_mm: float,
    increment_mm: int,
    n_lengths: int,
    round_to_mm: int = 50,
    min_length_mm: int | None = None,
    max_length_mm: int | None = None
) -> tuple[int, ...]:
    """
    Generate stock length tuple deterministically around a target mean.

    Lengths are created in fixed increments around the rounded mean and returned
    as sorted, unique, rounded integers.
    """
    if mean_length_mm <= 0:
        raise ValueError("mean_length_mm must be > 0")
    if increment_mm <= 0:
        raise ValueError("increment_mm must be > 0")
    if n_lengths <= 0:
        raise ValueError("n_lengths must be > 0")
    if round_to_mm <= 0:
        raise ValueError("round_to_mm must be > 0")
    if min_length_mm is not None and max_length_mm is not None and min_length_mm > max_length_mm:
        raise ValueError("min_length_mm must be <= max_length_mm")

    center = int(round(mean_length_mm / round_to_mm) * round_to_mm)

    values = set()
    offset = 0
    max_attempts = 10000

    while len(values) < n_lengths and offset < max_attempts:
        candidates = [center] if offset == 0 else [center - offset * increment_mm, center + offset * increment_mm]
        for candidate in candidates:
            rounded = int(round(candidate / round_to_mm) * round_to_mm)
            if min_length_mm is not None and rounded < min_length_mm:
                continue
            if max_length_mm is not None and rounded > max_length_mm:
                continue
            values.add(rounded)
            if len(values) >= n_lengths:
                break
        offset += 1

    if len(values) < n_lengths:
        raise ValueError(
            "Could not generate enough unique lengths within constraints. "
            "Increase bounds, lower n_lengths, or lower increment_mm."
        )

    return tuple(sorted(values))


def _get_mech_props_new() -> Dict[str, Any]:
    """Cache mechanical properties for new timber."""
    params = _get_params_module()
    return params.MECH_PROPS_NEW


def _get_lca_new() -> Dict[str, Any]:
    """Cache LCA properties for new timber."""
    params = _get_params_module()
    return params.LCA_NEW


def _get_mech_props_reclaimed() -> Dict[str, Any]:
    """Cache mechanical properties for reclaimed timber."""
    params = _get_params_module()
    return params.MECH_PROPS_RECLAIMED


def _get_lca_reclaimed() -> Dict[str, Any]:
    """Cache LCA properties for reclaimed timber."""
    params = _get_params_module()
    return params.LCA_RECLAIMED

def assign_transport_distance():
    """
    Kiest een willekeurige transportafstand op basis van de importpercentages 
    voor bouwhout in Nederland (2021).
    """
    
    # Definitie van de bronnen: (Land, Gewicht/Percentage, Gemiddelde afstand in km)
    # De afstanden zijn schattingen tot centraal Nederland en kunnen voor 
    # de uiteindelijke LCA-berekening in de thesis worden gefinetuned.
    sources = {
        "Duitsland": {"weight": 26, "base_distance": 300},
        "Zweden": {"weight": 18, "base_distance": 1000},
        "België": {"weight": 12, "base_distance": 150},
        "Baltische Staten": {"weight": 9, "base_distance": 1500},
        "Nederland": {"weight": 8, "base_distance": 50},
        "Finland": {"weight": 8, "base_distance": 1800},
        "Polen": {"weight": 5, "base_distance": 900},
        "Frankrijk": {"weight": 4, "base_distance": 500},
        "Spanje & Portugal": {"weight": 3, "base_distance": 1800},
        "Overig": {"weight": 7, "base_distance": 4000} # Resterende 7%
    }

    # Splits de data op in lijsten voor de random.choices functie
    country_names = list(sources.keys())
    weights = [sources[country]["weight"] for country in country_names]

    # 1. Kies een land op basis van de gewogen kansen
    chosen_country = random.choices(country_names, weights=weights, k=1)[0]
    base_dist = sources[chosen_country]["base_distance"]

    # 2. Voeg variatie toe (+/- 15%) voor een realistischere spreiding
    variation = base_dist * 0.15
    final_distance = random.uniform(base_dist - variation, base_dist + variation)

    return chosen_country, round(final_distance, 2)


def generate_new_timber_catalog() -> pd.DataFrame:
    """
    Generate catalog of new timber members with all length/depth/width combinations.
    
    Returns:
        pd.DataFrame: Catalog with columns for geometry, mechanical, and LCA properties
    """
    lca_new = _get_lca_new()
    mech_new = _get_mech_props_new()
    params = _get_params_module()
    
    # Pre-cache values to avoid repeated dict lookups
    e_modulus = float(mech_new['E_modulus_eff'])
    f_mk = int(mech_new['f_mk'])
    density = int(mech_new['Density'])
    embodied_carbon = float(lca_new['Embodied Carbon Coëfficiënt'])
    emission_range = lca_new['Emmisiefactor_diesel_range']
    processing_factor = int(lca_new['Bewerkingsfactor'])
    
    combinations = list(itertools.product(
        params.TUPLE_LENGTHS, 
        params.DEPTH_WIDTH_COMBINATIONS
    ))
    
    print(f"📊 Catalogus genereren... {len(combinations)} balk-typen")
    
    data = []
    for idx, (length, (depth, width)) in enumerate(combinations):
        # Genereer unieke afstandsdata per element
        origin_country, transport_dist = assign_transport_distance()
        
        data.append({
            'Member_ID': f"NS_{idx:05d}",
            'State': 0,
            'Length': float(length),
            'Depth': float(depth),
            'Width': float(width),
            'E_modulus_eff': e_modulus,
            'f_mk': f_mk,
            'Density': density,
            'ECC': embodied_carbon,
            'Origin_Country': origin_country,
            'Transport_Dist': transport_dist,
            'Emmisiefactor': round(random.uniform(*emission_range), 4),
            'Bewerkingsfactor': processing_factor
        })
    
    df_new = pd.DataFrame(data)
    print(f"✅ New stock succesvol gegenereerd! ({len(df_new)} elementen)")
    return df_new


def generate_reclaimed_stock() -> pd.DataFrame:
    """
    Generate reclaimed timber inventory from donor batches with realistic losses.
    
    Returns:
        pd.DataFrame: Inventory with columns for geometry, mechanical, and LCA properties
    """
    mech_reclaimed = _get_mech_props_reclaimed()
    lca_reclaimed = _get_lca_reclaimed()
    params = _get_params_module()
    
    # Pre-cache values
    embodied_carbon = float(lca_reclaimed['Embodied Carbon Coëfficiënt'])
    processing_factor = int(lca_reclaimed['Bewerkingsfactor'])
    prob_electric = lca_reclaimed['Kans_op_elektrisch']
    electric_range = lca_reclaimed['Emmisiefactor_elektrisch_range']
    diesel_range = lca_reclaimed['Emmisiefactor_diesel_range']
    
    inventory_list = []
    current_id = 1
    
    for batch in params.DONOR_BATCHES:
        batch_count = batch['count']
        orig_length = batch['orig_length']
        orig_depth = batch['orig_depth']
        orig_width = batch['orig_width']
        
        for _ in range(batch_count):
            # Geometry with losses
            length = orig_length - random.randint(100, 400)
            depth = orig_depth - random.randint(10, 16)
            width = orig_width - random.randint(10, 16)
            
            # Grade determination
            grade = np.random.choice(['C24', 'C18'], p=[0.60, 0.40])
            grade_props = mech_reclaimed[grade]
            

            transport_dist = random.randint(*lca_reclaimed['Transport_distance_range'])
            if random.random() < prob_electric:
                emission_factor = random.uniform(*electric_range)
            else:
                emission_factor = random.uniform(*diesel_range)
            
            inventory_list.append({
                'Member_ID': f"RS_{current_id:05d}",
                'State': 1,  # 1 = Reclaimed
                'Length': float(length),
                'Depth': float(depth),
                'Width': float(width),
                'E_modulus_eff': float(grade_props['e_mod']),
                'f_mk': int(grade_props['f_mk']),
                'Density': int(grade_props['density']),
                'ECC': embodied_carbon,
                'Origin_Country': "Netherlands",
                'Transport_Dist': transport_dist,
                'Emmisiefactor': round(emission_factor, 4),
                'Bewerkingsfactor': processing_factor
            })
            current_id += 1
    
    df_reclaimed = pd.DataFrame(inventory_list)
    print(f"✅ Reclaimed stock gegenereerd! ({len(df_reclaimed)} elementen)")
    return df_reclaimed


def generate_mixed_stock_subset(
    total_elements: int,
    reused_ratio: float = 0.5,
    random_state: int | None = None,
    allow_replacement: bool = True
) -> pd.DataFrame:
    """
    Generate a smaller mixed set with a target reused/new distribution.

    Args:
        total_elements: Total number of elements in the returned dataset.
        reused_ratio: Share of reused elements in range [0, 1].
        random_state: Seed for reproducible sampling.
        allow_replacement: If False, raises an error when requested count exceeds
            available stock in either source dataset.

    Returns:
        pd.DataFrame: Mixed subset containing both new and reused elements.
    """
    if total_elements <= 0:
        raise ValueError("total_elements must be > 0")
    if not 0 <= reused_ratio <= 1:
        raise ValueError("reused_ratio must be between 0 and 1")

    requested_reused = int(round(total_elements * reused_ratio))
    requested_new = total_elements - requested_reused

    df_new = generate_new_timber_catalog()
    df_reused = generate_reclaimed_stock()

    available_new = len(df_new)
    available_reused = len(df_reused)

    replace_new = requested_new > available_new
    replace_reused = requested_reused > available_reused

    if (replace_new or replace_reused) and not allow_replacement:
        raise ValueError(
            "Requested subset size exceeds available stock. "
            f"Requested new/reused: {requested_new}/{requested_reused}, "
            f"available new/reused: {available_new}/{available_reused}. "
            "Set allow_replacement=True or reduce total_elements/reused_ratio."
        )

    rng = np.random.default_rng(random_state)
    seed_new = int(rng.integers(0, 2**31 - 1))
    seed_reused = int(rng.integers(0, 2**31 - 1))

    sampled_new = df_new.sample(
        n=requested_new,
        replace=replace_new,
        random_state=seed_new
    )
    sampled_reused = df_reused.sample(
        n=requested_reused,
        replace=replace_reused,
        random_state=seed_reused
    )

    df_subset = pd.concat([sampled_new, sampled_reused], ignore_index=True).reset_index(drop=True)

    realized_reused_ratio = (df_subset['State'] == 1).mean()
    print(
        "📦 Mixed subset gegenereerd: "
        f"{len(df_subset)} totaal | "
        f"new={requested_new}, reused={requested_reused} "
        f"(reused_ratio={realized_reused_ratio:.2f})"
    )
    return df_subset
