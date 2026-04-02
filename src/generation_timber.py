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


def _get_mech_props_by_class(strength_class: str) -> Dict[str, Any]:
    """Return mechanical properties for a given strength class (e.g. C18, C24)."""
    params = _get_params_module()
    mech_props = getattr(params, "MECH_PROPS")
    try:
        return mech_props[strength_class]
    except KeyError as exc:
        available = ", ".join(sorted(mech_props.keys()))
        raise KeyError(f"Unknown strength class '{strength_class}'. Available: {available}") from exc


def _mechanical_props_row(mech_props: Dict[str, Any]) -> Dict[str, float]:
    """Return all mechanical properties in a dataset-ready structure."""
    return {
        'f_mk': float(mech_props['f_mk']),
        'f_tk': float(mech_props['f_tk']),
        'E_modulus_eff': float(mech_props['E_modulus_eff']),
        'E_modulus_005': float(mech_props['E_modulus_005']),
        'f_vk': float(mech_props['f_vk']),
        'f_c0k': float(mech_props['f_c0k']),
        'k_density': float(mech_props['k_density']),
        'mean_density': float(mech_props['mean_density']),
    }


def _get_lca_new() -> Dict[str, Any]:
    """Cache LCA properties for new timber."""
    params = _get_params_module()
    return getattr(params, "NEW_TIMBER_LCA")


def _get_lca_reclaimed() -> Dict[str, Any]:
    """Cache LCA properties for reclaimed timber."""
    params = _get_params_module()
    return getattr(params, "RECLAIMED_TIMBER_LCA")

def assign_transport_distance():
    """
    Choose a random transport distance based on timber import shares in the Netherlands (2021).
    """
    
    # Source definitions: (country, weight/share, average distance in km)
    # Distances are estimates to central Netherlands and can be fine-tuned
    # for the final LCA calculations in the thesis.
    sources = {
        "Germany": {"weight": 26, "base_distance": 300},
        "Sweden": {"weight": 18, "base_distance": 1000},
        "Belgium": {"weight": 12, "base_distance": 150},
        "Baltic States": {"weight": 9, "base_distance": 1500},
        "Netherlands": {"weight": 8, "base_distance": 50},
        "Finland": {"weight": 8, "base_distance": 1800},
        "Poland": {"weight": 5, "base_distance": 900},
        "France": {"weight": 4, "base_distance": 500},
        "Spain & Portugal": {"weight": 3, "base_distance": 1800},
        "Other": {"weight": 7, "base_distance": 4000}  # Remaining 7%
    }

    # Split the data into lists for random.choices.
    country_names = list(sources.keys())
    weights = [sources[country]["weight"] for country in country_names]

    # 1. Choose a country based on weighted probabilities.
    chosen_country = random.choices(country_names, weights=weights, k=1)[0]
    base_dist = sources[chosen_country]["base_distance"]

    # 2. Add variation (+/- 15%) for a more realistic spread.
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
    mech_new = _get_mech_props_by_class("C24")
    mech_row = _mechanical_props_row(mech_new)
    params = _get_params_module()
    
    # Pre-cache values to avoid repeated dict lookups.
    embodied_carbon = float(lca_new["embodied_carbon_coefficient"])
    emission_range = lca_new["diesel_emission_factor_range"]
    processing_factor = int(lca_new["processing_factor"])
    
    combinations = list(itertools.product(
        params.TUPLE_LENGTHS, 
        params.DEPTH_WIDTH_COMBINATIONS
    ))
    
    print(f"Generating catalog... {len(combinations)} beam types")
    
    data = []
    for idx, (length, (depth, width)) in enumerate(combinations):
        # Generate unique distance data for each element.
        origin_country, transport_dist = assign_transport_distance()
        emission_factor = round(random.uniform(*emission_range), 4)
        
        data.append({
            'Member_ID': f"NS_{idx:05d}",
            'State': 0,
            'Length': float(length),
            'Depth': float(depth),
            'Width': float(width),
            **mech_row,
            'ECC': embodied_carbon,
            'Origin_Country': origin_country,
            'Transport_Dist': transport_dist,
            'EmissionFactor': emission_factor,
            'ProcessingFactor': processing_factor
        })
    
    df_new = pd.DataFrame(data)
    print(f"New stock generated successfully! ({len(df_new)} elements)")
    return df_new


def generate_reclaimed_stock() -> pd.DataFrame:
    """
    Generate reclaimed timber inventory from a discrete section library and
    stochastic bounded lengths.
    
    Returns:
        pd.DataFrame: Inventory with columns for geometry, mechanical, and LCA properties
    """
    mech_reclaimed = _get_mech_props_by_class("C18")
    mech_row = _mechanical_props_row(mech_reclaimed)
    lca_reclaimed = _get_lca_reclaimed()
    params = _get_params_module()
    
    # Pre-cache values.
    embodied_carbon = float(lca_reclaimed["embodied_carbon_coefficient"])
    processing_factor = int(lca_reclaimed["processing_factor"])
    prob_electric = lca_reclaimed["electric_transport_probability"]
    electric_range = lca_reclaimed["electric_emission_factor_range"]
    diesel_range = lca_reclaimed["diesel_emission_factor_range"]
    
    stock_count = int(params.RECLAIMED_STOCK_COUNT)
    if stock_count <= 0:
        raise ValueError("RECLAIMED_STOCK_COUNT must be > 0")

    section_library = list(params.RECLAIMED_CROSS_SECTION_LIBRARY_MM)
    if not section_library:
        raise ValueError("RECLAIMED_CROSS_SECTION_LIBRARY_MM cannot be empty")

    length_distribution = str(params.RECLAIMED_LENGTH_DISTRIBUTION).lower()
    min_len = int(params.RECLAIMED_LENGTH_MIN_MM)
    max_len = int(params.RECLAIMED_LENGTH_MAX_MM)
    mean_len = float(params.RECLAIMED_LENGTH_MEAN_MM)
    std_len = float(params.RECLAIMED_LENGTH_STD_MM)
    round_to = int(params.RECLAIMED_LENGTH_ROUND_TO_MM)

    if min_len > max_len:
        raise ValueError("RECLAIMED_LENGTH_MIN_MM must be <= RECLAIMED_LENGTH_MAX_MM")
    if std_len <= 0:
        raise ValueError("RECLAIMED_LENGTH_STD_MM must be > 0")
    if round_to <= 0:
        raise ValueError("RECLAIMED_LENGTH_ROUND_TO_MM must be > 0")
    if length_distribution != "normal":
        raise ValueError("Only RECLAIMED_LENGTH_DISTRIBUTION='normal' is supported")

    # Build a balanced ordered sequence so each section appears equally.
    # (difference at most 1 when stock_count is not divisible).
    n_sections = len(section_library)
    full_cycles = stock_count // n_sections
    remainder = stock_count % n_sections
    section_sequence = (section_library * full_cycles) + section_library[:remainder]

    inventory_list = []
    for idx, (width, depth) in enumerate(section_sequence):

        sampled_length = np.random.normal(loc=mean_len, scale=std_len)
        bounded_length = float(np.clip(sampled_length, min_len, max_len))
        length = int(round(bounded_length / round_to) * round_to)

        transport_dist = random.randint(*lca_reclaimed["transport_distance_range"])
        if random.random() < prob_electric:
            emission_factor = random.uniform(*electric_range)
        else:
            emission_factor = random.uniform(*diesel_range)

        inventory_list.append({
            'Member_ID': f"RS_{idx + 1:05d}",
            'State': 1,  # 1 = Reclaimed
            'Length': float(length),
            'Depth': float(depth),
            'Width': float(width),
            **mech_row,
            'ECC': embodied_carbon,
            'Origin_Country': "Netherlands",
            'Transport_Dist': transport_dist,
            'EmissionFactor': round(emission_factor, 4),
            'ProcessingFactor': processing_factor
        })
    
    df_reclaimed = pd.DataFrame(inventory_list)
    print(f"Reclaimed stock generated successfully! ({len(df_reclaimed)} elements)")
    return df_reclaimed


def generate_mixed_stock_subset(
    total_elements: int,
    reclaimed_ratio: float = 0.5,
    random_state: int | None = None,
    allow_replacement: bool = True
) -> pd.DataFrame:
    """
    Generate a smaller mixed set with a target reused/new distribution.

    Args:
        total_elements: Total number of elements in the returned dataset.
        reclaimed_ratio: Share of reclaimed elements in range [0, 1].
        random_state: Seed for reproducible sampling.
        allow_replacement: If False, raises an error when requested count exceeds
            available stock in either source dataset.

    Returns:
        pd.DataFrame: Mixed subset containing both new and reclaimed elements.
    """
    if total_elements <= 0:
        raise ValueError("total_elements must be > 0")
    if not 0 <= reclaimed_ratio <= 1:
        raise ValueError("reclaimed_ratio must be between 0 and 1")

    requested_reclaimed = int(round(total_elements * reclaimed_ratio))
    requested_new = total_elements - requested_reclaimed

    df_new = generate_new_timber_catalog()
    df_reclaimed = generate_reclaimed_stock()

    available_new = len(df_new)
    available_reclaimed = len(df_reclaimed)

    replace_new = requested_new > available_new
    replace_reclaimed = requested_reclaimed > available_reclaimed

    if (replace_new or replace_reclaimed) and not allow_replacement:
        raise ValueError(
            "Requested subset size exceeds available stock. "
            f"Requested new/reclaimed: {requested_new}/{requested_reclaimed}, "
            f"available new/reclaimed: {available_new}/{available_reclaimed}. "
            "Set allow_replacement=True or reduce total_elements/reclaimed_ratio."
        )

    rng = np.random.default_rng(random_state)
    seed_new = int(rng.integers(0, 2**31 - 1))
    seed_reclaimed = int(rng.integers(0, 2**31 - 1))

    sampled_new = df_new.sample(
        n=requested_new,
        replace=replace_new,
        random_state=seed_new
    )
    sampled_reclaimed = df_reclaimed.sample(
        n=requested_reclaimed,
        replace=replace_reclaimed,
        random_state=seed_reclaimed
    )

    df_subset = pd.concat([sampled_new, sampled_reclaimed], ignore_index=True).reset_index(drop=True)

    realized_reclaimed_ratio = (df_subset['State'] == 1).mean()
    print(
        "Mixed subset generated: "
        f"{len(df_subset)} total | "
        f"new={requested_new}, reclaimed={requested_reclaimed} "
        f"(reclaimed_ratio={realized_reclaimed_ratio:.2f})"
    )
    return df_subset
