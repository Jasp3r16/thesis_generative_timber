"""
Timber Stock Generation Module
Generates new and reclaimed timber inventories from parameters.
"""

import itertools
import json
import random
from typing import Dict, Any
import numpy as np
import pandas as pd

# Module-level precomputed source distributions to avoid reallocating dicts on each call
_DEFAULT_SOURCES = {
    "Germany": {"weight": 26, "base_distance": 300},
    "Sweden": {"weight": 18, "base_distance": 1000},
    "Belgium": {"weight": 12, "base_distance": 150},
    "Baltic States": {"weight": 9, "base_distance": 1500},
    "Netherlands": {"weight": 8, "base_distance": 50},
    "Finland": {"weight": 8, "base_distance": 1800},
    "Poland": {"weight": 5, "base_distance": 900},
    "France": {"weight": 4, "base_distance": 500},
    "Spain & Portugal": {"weight": 3, "base_distance": 1800},
    "Other": {"weight": 7, "base_distance": 4000},
}

_EFFICIENT_SOURCES = {
    "Germany": {"weight": 6, "base_distance": 300},
    "Netherlands": {"weight": 4, "base_distance": 50},
}

# Precompute lists for fast sampling
_DEFAULT_COUNTRIES = list(_DEFAULT_SOURCES.keys())
_DEFAULT_WEIGHTS = [_DEFAULT_SOURCES[c]["weight"] for c in _DEFAULT_COUNTRIES]
_DEFAULT_BASE = {c: _DEFAULT_SOURCES[c]["base_distance"] for c in _DEFAULT_COUNTRIES}

_EFFICIENT_COUNTRIES = list(_EFFICIENT_SOURCES.keys())
_EFFICIENT_WEIGHTS = [_EFFICIENT_SOURCES[c]["weight"] for c in _EFFICIENT_COUNTRIES]
_EFFICIENT_BASE = {c: _EFFICIENT_SOURCES[c]["base_distance"] for c in _EFFICIENT_COUNTRIES}


def _get_params_module():
    """Lazy import to avoid circular dependencies with c22_params."""
    import c16_params as params
    return params


def _load_representative_beam_statistics() -> Dict[str, Any]:
    """Load representative beam statistics from disk (if available)."""
    try:
        import config
    except ImportError:
        return {}

    json_path = config.DATA_IO_PATH / "representative_beam_statistics.json"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _get_structural_length_bounds_mm() -> tuple[float, float]:
    """Return structural min/max member lengths in mm from statistics, with safe fallback."""
    stats = _load_representative_beam_statistics()
    summary = stats.get("summary_statistics") if isinstance(stats, dict) else None
    params = _get_params_module()

    if isinstance(summary, dict):
        try:
            min_mm = float(summary.get("min_length_mm", params.MIN_LENGTH_MM))
            max_mm = float(summary.get("max_length_mm", params.MAX_LENGTH_MM))
            if min_mm <= max_mm:
                return min_mm, max_mm
        except (TypeError, ValueError):
            pass

    return float(params.MIN_LENGTH_MM), float(params.MAX_LENGTH_MM)


def _get_pooled_length_percentiles_mm() -> Dict[str, float]:
    """Return pooled length percentiles in mm, with fallback defaults."""
    stats = _load_representative_beam_statistics()
    pooled = stats.get("pooled_length_percentiles_mm") if isinstance(stats, dict) else None
    if isinstance(pooled, dict) and "p5_mm" in pooled and "p95_mm" in pooled:
        return {k: float(v) for k, v in pooled.items()}

    params = _get_params_module()
    summary = stats.get("summary_statistics", {}) if isinstance(stats, dict) else {}
    min_mm = float(summary.get("min_length_mm", params.MIN_LENGTH_MM))
    max_mm = float(summary.get("max_length_mm", params.MAX_LENGTH_MM))
    median_mm = float(summary.get("median_length_mm", params.STRUCTURE_AVERAGE_LENGTH_MM))
    span = max(max_mm - min_mm, 1.0)
    return {
        "p1_mm": min_mm,
        "p5_mm": min_mm + 0.05 * span,
        "p10_mm": min_mm + 0.10 * span,
        "p25_mm": min_mm + 0.25 * span,
        "p50_mm": median_mm,
        "p75_mm": min_mm + 0.75 * span,
        "p95_mm": min_mm + 0.95 * span,
        "p99_mm": max_mm,
    }


def _generate_lengths_within_bounds(
    mean_mm: float,
    increment_mm: int,
    n_lengths: int,
    round_to_mm: int,
    min_length_mm: int,
    max_length_mm: int
) -> tuple[int, ...]:
    """Generate a length tuple within bounds, falling back to bounds if needed."""
    try:
        return generate_length_tuple_from_average(
            mean_length_mm=mean_mm,
            increment_mm=increment_mm,
            n_lengths=n_lengths,
            round_to_mm=round_to_mm,
            min_length_mm=min_length_mm,
            max_length_mm=max_length_mm,
        )
    except ValueError:
        rounded_min = int(round(min_length_mm / round_to_mm) * round_to_mm)
        rounded_max = int(round(max_length_mm / round_to_mm) * round_to_mm)
        values = sorted({rounded_min, rounded_max})
        return tuple(values[:max(1, min(n_lengths, len(values)))])


def _sample_tail_section_combinations(
    section_combinations: list[tuple[int, int]],
    n_tail_sections: int,
    random_state: int | None,
) -> list[tuple[int, int]]:
    """Select a small deterministic subset of section combinations for tails."""
    if n_tail_sections <= 0:
        return []
    if n_tail_sections >= len(section_combinations):
        return list(section_combinations)

    rng = random.Random(random_state)
    sampled = rng.sample(section_combinations, n_tail_sections)
    return sorted(sampled)


def _sample_weighted_tail_lengths(
    min_length_mm: int,
    max_length_mm: int,
    increment_mm: int,
    n_lengths: int,
    random_state: int | None,
    direction: str,
) -> tuple[int, ...]:
    """Sample a small set of tail lengths with probability increasing toward the primary block."""
    if n_lengths <= 0:
        return tuple()
    if min_length_mm > max_length_mm:
        min_length_mm, max_length_mm = max_length_mm, min_length_mm

    # Align candidate lengths to the production increment (e.g., 300 mm)
    start = int(((min_length_mm + increment_mm - 1) // increment_mm) * increment_mm)
    end = int((max_length_mm // increment_mm) * increment_mm)
    candidate_lengths = list(range(start, end + increment_mm, increment_mm))
    if not candidate_lengths:
        return tuple()

    if direction == "short":
        weights = np.linspace(1.0, 2.5, len(candidate_lengths))
    elif direction == "long":
        weights = np.linspace(2.5, 1.0, len(candidate_lengths))
    else:
        raise ValueError("direction must be 'short' or 'long'")

    sample_size = min(n_lengths, len(candidate_lengths))
    rng = random.Random(random_state)
    chosen = rng.choices(candidate_lengths, weights=weights, k=sample_size * 3)

    unique = []
    seen = set()
    for length in chosen:
        if length in seen:
            continue
        unique.append(length)
        seen.add(length)
        if len(unique) >= sample_size:
            break

    if len(unique) < sample_size:
        for length in candidate_lengths:
            if length not in seen:
                unique.append(length)
                seen.add(length)
            if len(unique) >= sample_size:
                break

    return tuple(sorted(unique))


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

    For odd n_lengths, the rounded mean is included as the center value.
    For even n_lengths, values are generated symmetrically around the center
    using half-step offsets (e.g. center +/- 0.5 * increment).
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
    max_attempts = 10000

    for radius in range(max_attempts):
        if n_lengths % 2 == 1:
            multipliers = [0.0] if radius == 0 else [-float(radius), float(radius)]
        else:
            # For even-sized libraries, keep symmetry around the center point.
            step = radius + 0.5
            multipliers = [-step, step]

        for multiplier in multipliers:
            candidate = center + multiplier * increment_mm
            rounded = int(round(candidate / round_to_mm) * round_to_mm)
            if min_length_mm is not None and rounded < min_length_mm:
                continue
            if max_length_mm is not None and rounded > max_length_mm:
                continue
            values.add(rounded)
            if len(values) >= n_lengths:
                break
        if len(values) >= n_lengths:
            break

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
    return _get_lca_properties("NEW_TIMBER_LCA")


def _get_lca_reclaimed() -> Dict[str, Any]:
    """Cache LCA properties for reclaimed timber."""
    return _get_lca_properties("RECLAIMED_TIMBER_LCA")


def _get_lca_properties(attribute_name: str) -> Dict[str, Any]:
    params = _get_params_module()
    return getattr(params, attribute_name)

def assign_transport_distance(efficient: bool = False, random_state: int | None = None) -> tuple[str, float]:
    """
    Choose a random transport distance based on timber import shares in the Netherlands (2021).
    """
    
    # Source definitions: (country, weight/share, average distance in km)
    # Distances are estimates to central Netherlands and can be fine-tuned
    # for the final LCA calculations in the thesis.
    # choose source lists based on mode
    if efficient:
        country_names = _EFFICIENT_COUNTRIES
        weights = _EFFICIENT_WEIGHTS
        base_map = _EFFICIENT_BASE
    else:
        country_names = _DEFAULT_COUNTRIES
        weights = _DEFAULT_WEIGHTS
        base_map = _DEFAULT_BASE

    # deterministic RNG when requested
    if random_state is None:
        rng = random
    else:
        rng = random.Random(random_state)

    chosen_country = rng.choices(country_names, weights=weights, k=1)[0]
    base_dist = base_map[chosen_country]

    # Variation +/- 15%
    variation = base_dist * 0.15
    final_distance = rng.uniform(base_dist - variation, base_dist + variation)

    return chosen_country, round(final_distance, 2)


def generate_new_stock(efficient: bool = False, random_state: int | None = None) -> pd.DataFrame:
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
    emission_range = lca_new["diesel_emission_factor_range"]

    percentiles = _get_pooled_length_percentiles_mm()
    p1_mm = int(round(percentiles["p1_mm"]))
    p5_mm = int(round(percentiles["p5_mm"]))
    p50_mm = float(percentiles["p50_mm"])
    p95_mm = int(round(percentiles["p95_mm"]))
    p99_mm = int(round(percentiles["p99_mm"]))
    structure_min_mm, structure_max_mm = _get_structural_length_bounds_mm()
    structure_min_mm = int(round(structure_min_mm / params.LENGTH_ROUND_TO_MM) * params.LENGTH_ROUND_TO_MM)
    structure_max_mm = int(round(structure_max_mm / params.LENGTH_ROUND_TO_MM) * params.LENGTH_ROUND_TO_MM)

    # Extend tails beyond observed structural bounds by a margin x.
    tail_margin_mm = int(getattr(params, "NEW_STOCK_TAIL_MARGIN_MM", params.LENGTH_INCREMENT_MM))
    tail_margin_mm = max(tail_margin_mm, params.LENGTH_ROUND_TO_MM)

    # Primary catalog keeps the stable catalog lengths and full section coverage.
    # We prefer the bulk interval implied by pooled percentiles, but never shrink
    # the primary library so much that it loses the representative size spread.
    primary_lengths = tuple(
        length for length in params.TUPLE_LENGTHS
        if p5_mm <= length <= p95_mm
    )
    if len(primary_lengths) < max(5, len(params.TUPLE_LENGTHS) // 2):
        primary_lengths = tuple(params.TUPLE_LENGTHS)

    tail_count_each = int(getattr(params, "NEW_STOCK_TAIL_LENGTH_COUNT", 4))
    short_tail_min = max(params.LENGTH_INCREMENT_MM, int((structure_min_mm - tail_margin_mm) // params.LENGTH_INCREMENT_MM) * params.LENGTH_INCREMENT_MM)
    short_tail_max = int(round((primary_lengths[0] - tail_margin_mm) / params.LENGTH_ROUND_TO_MM) * params.LENGTH_ROUND_TO_MM)
    long_tail_min = int(round((primary_lengths[-1] + tail_margin_mm) / params.LENGTH_ROUND_TO_MM) * params.LENGTH_ROUND_TO_MM)
    long_tail_max = int(round((structure_max_mm + tail_margin_mm + 2 * params.LENGTH_INCREMENT_MM) / params.LENGTH_ROUND_TO_MM) * params.LENGTH_ROUND_TO_MM)

    short_tail = _sample_weighted_tail_lengths(
        min_length_mm=short_tail_min,
        max_length_mm=short_tail_max,
        increment_mm=params.LENGTH_INCREMENT_MM,
        n_lengths=tail_count_each,
        random_state=random_state,
        direction="short",
    )
    long_tail = _sample_weighted_tail_lengths(
        min_length_mm=long_tail_min,
        max_length_mm=long_tail_max,
        increment_mm=params.LENGTH_INCREMENT_MM,
        n_lengths=tail_count_each + 2,
        random_state=None if random_state is None else random_state + 1,
        direction="long",
    )

    # Ensure the band touches the structural limits while remaining outside the primary block.
    if short_tail:
        short_tail = tuple(sorted(set(short_tail) | {short_tail_min, short_tail_max}))
    else:
        short_tail = (short_tail_min, short_tail_max)
    if long_tail:
        long_tail = tuple(sorted(set(long_tail) | {long_tail_min, long_tail_max}))
    else:
        long_tail = (long_tail_min, long_tail_max)

    primary_set = set(primary_lengths)
    short_tail = tuple(v for v in short_tail if v not in primary_set)
    long_tail = tuple(v for v in long_tail if v not in primary_set)

    min_tail_sections = int(getattr(params, "NEW_STOCK_TAIL_SECTION_COUNT", 6))
    all_sections = list(params.DEPTH_WIDTH_COMBINATIONS)
    max_tail_sections = max(min_tail_sections, len(all_sections) // 2)

    def _graduated_tail_sections(tail_lengths: tuple, closest_is_last: bool) -> list[tuple[int, list]]:
        """Assign graduated section subsets: more sections for lengths closer to the primary block."""
        n = len(tail_lengths)
        result = []
        for i, length in enumerate(tail_lengths):
            if n <= 1:
                proximity = 1.0
            elif closest_is_last:
                proximity = i / (n - 1)
            else:
                proximity = (n - 1 - i) / (n - 1)
            n_secs = max(min_tail_sections, round(min_tail_sections + proximity * (max_tail_sections - min_tail_sections)))
            seed = None if random_state is None else random_state + i
            secs = _sample_tail_section_combinations(all_sections, n_secs, seed)
            result.append((length, secs))
        return result

    # short_tail sorted ascending: last index (e.g. 1500) is closest to primary (1800)
    short_tail_sections = _graduated_tail_sections(short_tail, closest_is_last=True)
    # long_tail sorted ascending: first index (e.g. 4500) is closest to primary (4200)
    long_tail_sections = _graduated_tail_sections(long_tail, closest_is_last=False)

    primary_combinations = list(itertools.product(primary_lengths, all_sections))
    n_tail = sum(len(secs) for _, secs in short_tail_sections + long_tail_sections)

    print(
        f"\nGenerating catalog...\n"
        f"  primary={len(primary_combinations)} beam types, tail={n_tail} beam types"
    )

    # Prepare RNGs for reproducible sampling when requested
    rng_py = random if random_state is None else random.Random(random_state)

    # Choose countries in batch and sample emission factors
    n_total = len(primary_combinations) + n_tail
    if efficient:
        country_list = rng_py.choices(_EFFICIENT_COUNTRIES, weights=_EFFICIENT_WEIGHTS, k=n_total)
        base_map = _EFFICIENT_BASE
    else:
        country_list = rng_py.choices(_DEFAULT_COUNTRIES, weights=_DEFAULT_WEIGHTS, k=n_total)
        base_map = _DEFAULT_BASE

    emission_factors = [round(rng_py.uniform(*emission_range), 4) for _ in range(n_total)]

    data = []
    idx = 0

    def _append_rows(lengths: list[tuple[int, str, float]], section_combinations: list[tuple[int, int]]) -> None:
        nonlocal idx
        for length, length_category, availability_probability in lengths:
            for depth, width in section_combinations:
                origin_country = country_list[idx]
                emission_factor = emission_factors[idx]
                base_dist = base_map[origin_country]
                variation = base_dist * 0.15
                transport_dist = round(rng_py.uniform(base_dist - variation, base_dist + variation), 2)

                data.append({
                    'Member_ID': f"NS_{idx:05d}",
                    'State': 0,
                    'Length': float(length),
                    'Depth': float(depth),
                    'Width': float(width),
                    'Length_Category': length_category,
                    'Availability_Probability': float(availability_probability),
                    **mech_row,
                    'Origin_Country': origin_country,
                    'Transport_Dist': transport_dist,
                    'EmissionFactor': emission_factor,
                })
                idx += 1

    _append_rows([(l, "primary", 1.0) for l in primary_lengths], all_sections)
    for length, secs in short_tail_sections:
        _append_rows([(length, "short_tail", 0.05)], secs)
    for length, secs in long_tail_sections:
        _append_rows([(length, "long_tail", 0.05)], secs)

    df_new = pd.DataFrame(data)
    short_tail_n_secs = {l: len(s) for l, s in short_tail_sections}
    long_tail_n_secs = {l: len(s) for l, s in long_tail_sections}
    n_primary = (df_new['Length_Category'] == 'primary').sum()
    n_short = (df_new['Length_Category'] == 'short_tail').sum()
    n_long = (df_new['Length_Category'] == 'long_tail').sum()
    length_counts = df_new.groupby('Length').size().sort_index()
    length_summary = "\n".join(f"    {int(l)} mm: {c}" for l, c in length_counts.items())

    print(
        f"\nLength source stats (new stock):"
        f"\n  primary      (p5–p95):  {primary_lengths}"
        f"\n  short tail   (min→p5):  {short_tail}"
        f"\n  long tail    (p95→max): {long_tail}"
    )
    print(
        f"\nGenerated lengths — tail_margin_mm={tail_margin_mm}"
        f"\n  Sections per tail length (graduated {min_tail_sections}..{max_tail_sections}):"
        f"\n    short tail: {short_tail_n_secs}"
        f"\n    long tail:  {long_tail_n_secs}"
    )
    print(
        f"\nNew stock generated: {len(df_new)} elements total"
        f"\n  primary={n_primary}, short_tail={n_short}, long_tail={n_long}"
        f"\n\nElements per length:\n{length_summary}\n"
    )
    return df_new


def generate_reclaimed_stock(
    random_state: int | None = None,
    donor_building: str = "A",
) -> pd.DataFrame:
    """
    Generate reclaimed timber inventory from a parametric donor building.

    Args:
        random_state: Seed for reproducibility.
        donor_building: "A" (residential, hardcoded spans) or
                        "B" (commercial/industrial, spans derived from
                        the structure's own min/max length statistics).

    Returns:
        pd.DataFrame: Inventory with columns for geometry, mechanical,
                      and LCA properties, plus 'Donor_Role' metadata.
    """
    mech_reclaimed = _get_mech_props_by_class("C18")
    mech_row = _mechanical_props_row(mech_reclaimed)
    lca_reclaimed = _get_lca_reclaimed()
    params = _get_params_module()

    planing = int(params.RECLAIMED_PLANING_ALLOWANCE_MM)
    floors = int(params.DONOR_BUILDING_FLOORS)
    survival_rate = float(params.DONOR_BUILDING_SURVIVAL_RATE)
    cut_loss_max = int(params.RECLAIMED_CUT_LOSS_MAX_MM)

    _key = f"DONOR_BUILDING_{donor_building.upper()}_MEMBER_TYPES"
    if not hasattr(params, _key):
        raise ValueError(f"Unknown donor building '{donor_building}'. Expected 'A' or 'B'.")
    member_types = list(getattr(params, _key))

    prob_electric = lca_reclaimed["electric_transport_probability"]
    electric_range = lca_reclaimed["electric_emission_factor_range"]
    diesel_range = lca_reclaimed["diesel_emission_factor_range"]

    rng_py = random if random_state is None else random.Random(random_state)
    rng_np = np.random.default_rng(random_state)

    # --- Step 1: Generate raw pool from donor building ---
    raw_pool = []
    for role, nom_w, nom_d, count_per_floor, span_mm in member_types:
        net_w = nom_w - planing
        net_d = nom_d - planing
        for floor in range(floors):
            for _ in range(count_per_floor):
                cut_loss = int(rng_np.integers(0, cut_loss_max + 1))
                length = span_mm - cut_loss  # NOT rounded
                raw_pool.append({
                    "role": role,
                    "net_w": net_w,
                    "net_d": net_d,
                    "span_mm": span_mm,
                    "cut_loss_mm": cut_loss,
                    "length_mm": length,
                    "floor": floor + 1,
                })

    df_raw = pd.DataFrame(raw_pool)
    print(f"Donor building raw pool: {len(df_raw)} elements across {floors} floors")

    # --- Step 2: Apply per-role survival filter ---
    # Remove bottom (1 - survival_rate) fraction by length within each role.
    # Shortest elements are most likely to be damaged or unusable.
    survived = []
    for role, group in df_raw.groupby("role"):
        threshold = group["length_mm"].quantile(1.0 - survival_rate)
        kept = group[group["length_mm"] >= threshold].copy()
        survived.append(kept)
        print(f"  {role}: {len(group)} raw -> {len(kept)} survived")

    df_survived = pd.concat(survived, ignore_index=True)
    print(f"Total surviving elements: {len(df_survived)}")

    # --- Step 3: Assign LCA properties and build final inventory ---
    inventory_list = []
    for idx, row in enumerate(df_survived.itertuples(index=False)):
        transport_dist = rng_py.randint(*lca_reclaimed["transport_distance_range"])
        if rng_py.random() < prob_electric:
            emission_factor = rng_py.uniform(*electric_range)
        else:
            emission_factor = rng_py.uniform(*diesel_range)

        inventory_list.append({
            "Member_ID": f"RS_{idx + 1:05d}",
            "State": 1,
            "Length": float(row.length_mm),
            "Depth": float(row.net_d),
            "Width": float(row.net_w),
            "Donor_Role": row.role,
            "Cut_Loss_mm": int(row.cut_loss_mm),
            **mech_row,
            "Origin_Country": "Netherlands",
            "Transport_Dist": float(transport_dist),
            "EmissionFactor": round(emission_factor, 4),
        })

    df_reclaimed = pd.DataFrame(inventory_list)

    # --- Step 4: Print summary ---
    print(f"\nReclaimed stock summary:")
    print(f"  Total elements: {len(df_reclaimed)}")
    print(f"  Length mean:    {df_reclaimed['Length'].mean():.1f} mm")
    print(f"  Length std:     {df_reclaimed['Length'].std():.1f} mm")
    print(f"  Length min:     {df_reclaimed['Length'].min():.0f} mm")
    print(f"  Length max:     {df_reclaimed['Length'].max():.0f} mm")
    print(f"  Unique lengths: {df_reclaimed['Length'].nunique()}")
    print(f"\n  By role:")
    for role, grp in df_reclaimed.groupby("Donor_Role"):
        net_w = grp["Width"].iloc[0]
        net_d = grp["Depth"].iloc[0]
        print(f"    {role}: {len(grp)} elements, "
              f"{grp['Length'].min():.0f}–{grp['Length'].max():.0f} mm, "
              f"section {net_w:.0f}x{net_d:.0f} mm")

    return df_reclaimed


def generate_mixed_stock_subset(
    total_elements: int,
    reclaimed_ratio: float = 0.5,
    random_state: int | None = None,
    efficient: bool = False,
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

    df_new = generate_new_stock(efficient, random_state=random_state)
    df_reclaimed = generate_reclaimed_stock(random_state=random_state)

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
