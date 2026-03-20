"""
Timber Stock Generation Module
Generates new and reclaimed timber inventories from parameters.
"""

import itertools
import random
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import c22_params as params


def _get_mech_props_new() -> Dict[str, Any]:
    """Cache mechanical properties for new timber."""
    return params.MECH_PROPS_NEW


def _get_lca_new() -> Dict[str, Any]:
    """Cache LCA properties for new timber."""
    return params.LCA_NEW


def _get_mech_props_reclaimed() -> Dict[str, Any]:
    """Cache mechanical properties for reclaimed timber."""
    return params.MECH_PROPS_RECLAIMED


def _get_lca_reclaimed() -> Dict[str, Any]:
    """Cache LCA properties for reclaimed timber."""
    return params.LCA_RECLAIMED


def generate_new_timber_catalog() -> pd.DataFrame:
    """
    Generate catalog of new timber members with all length/depth/width combinations.
    
    Returns:
        pd.DataFrame: Catalog with columns for geometry, mechanical, and LCA properties
    """
    lca_new = _get_lca_new()
    mech_new = _get_mech_props_new()
    
    # Pre-cache values to avoid repeated dict lookups
    e_modulus = float(mech_new['E_modulus_eff'])
    f_mk = int(mech_new['f_mk'])
    density = int(mech_new['Density'])
    embodied_carbon = float(lca_new['Embodied Carbon Coëfficiënt'])
    transport_dist = int(lca_new['Transport_Dist'])
    emission_range = lca_new['Emmisiefactor_diesel_range']
    processing_factor = int(lca_new['Bewerkingsfactor'])
    
    combinations = list(itertools.product(
        params.TUPLE_LENGTHS, 
        params.DEPTH_WIDTH_COMBINATIONS
    ))
    
    print(f"📊 Catalogus genereren... {len(combinations)} balk-typen")
    
    data = [
        {
            'Member_ID': f"NS_{idx:05d}",
            'State': 0,
            'Length': float(length),
            'Depth': float(depth),
            'Width': float(width),
            'E_modulus_eff': e_modulus,
            'f_mk': f_mk,
            'Density': density,
            'Embodied Carbon Coëfficiënt': embodied_carbon,
            'Transport_Dist': transport_dist,
            'Emmisiefactor': round(random.uniform(*emission_range), 4),
            'Bewerkingsfactor': processing_factor
        }
        for idx, (length, (depth, width)) in enumerate(combinations)
    ]
    
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
            
            # Transport and emissions
            transport_dist = random.randint(20, 150)
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
                'Embodied Carbon Coëfficiënt': embodied_carbon,
                'Transport_Dist': transport_dist,
                'Emmisiefactor': round(emission_factor, 4),
                'Bewerkingsfactor': processing_factor
            })
            current_id += 1
    
    df_reclaimed = pd.DataFrame(inventory_list)
    print(f"✅ Reclaimed stock gegenereerd! ({len(df_reclaimed)} elementen)")
    return df_reclaimed
