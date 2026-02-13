#!/usr/bin/env python3
"""
MNL model fitting script using pylogit with VISIBLE as reference (instead of INVISIBLE).
Estimates coefficients and calculates WTP values.
"""

import argparse
import pandas as pd
import numpy as np
import pylogit as pl
from pathlib import Path
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(description='Fit MNL model with VISIBLE as reference and calculate WTP')
    parser.add_argument('--input', default='data/dce_long_model_visible_base.csv',
                       help='Input processed data CSV')
    parser.add_argument('--mapping', default='output/coding_map_visible_base.csv',
                       help='Coding mapping file')
    parser.add_argument('--coef_output', default='output/coefficients_visible_base.csv',
                       help='Coefficients output file')
    parser.add_argument('--wtp_output', default='output/wtp_levels_visible_base.csv',
                       help='WTP levels output file')
    parser.add_argument('--range_output', default='output/attribute_ranges_visible_base.csv',
                       help='Attribute ranges output file')
    args = parser.parse_args()
    
    print(f"Reading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    # Reset index first to ensure clean DataFrame
    df = df.reset_index(drop=True)
    
    df['alt_num'] = df['alt_num'].astype(int)
    df['choice'] = df['choice'].astype(int)
    
    print(f"Loaded {len(df)} rows, {df['csid'].nunique()} choice situations")
    
    # Sanity check: each csid should have exactly 1 choice==1
    choice_counts = df.groupby('csid')['choice'].sum()
    if not (choice_counts == 1).all():
        bad_csids = choice_counts[choice_counts != 1].index.tolist()
        raise ValueError(f"Found {len(bad_csids)} choice situations without exactly 1 choice. Examples: {bad_csids[:5]}")
    
    # Identify model variables
    required = ['price_eur', 'is_optout', 'csid', 'choice', 'alt_num', 'resp_id']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Collect effects-coded columns (numeric dummy variables only)
    structural_cols = ['resp_id', 'task_id', 'alt_id', 'csid', 'alt_num', 'choice', 'is_optout', 'price_eur']
    original_categorical = ['comfort', 'att', 'speech', 'app', 'group', 'purchase', 'rel_price', 'rel_price_lvl']
    
    # Only include numeric columns that are effects-coded (not original categorical)
    effects_coded = [c for c in df.columns 
                     if c not in structural_cols and c not in original_categorical]
    
    # Ensure proper data types for pylogit compatibility
    # Convert string csid to categorical codes for pylogit (AFTER identifying effects_coded)
    df['csid_cat'] = pd.Categorical(df['csid'])
    df['csid_numeric'] = df['csid_cat'].cat.codes
    
    print(f"\nEffects-coded variables: {effects_coded}")
    
    # Build variable list: is_optout + price_eur + effects-coded
    var_names = ['is_optout', 'price_eur'] + effects_coded
    
    # Build specification OrderedDict (all_same for all variables)
    spec = OrderedDict({var: 'all_same' for var in var_names})
    labels = OrderedDict({var: var for var in var_names})
    
    print(f"\nFitting MNL model using pylogit...")
    print(f"Variables ({len(var_names)}): {var_names}")
    print(f"\n*** NOTE: This model uses VISIBLE as reference (instead of INVISIBLE) ***\n")
    
    # Ensure data is in proper format (already converted above)
    df = df.copy()
    
    # Fit pylogit MNL model
    mnl_model = pl.create_choice_model(
        data=df,
        alt_id_col='alt_num',
        obs_id_col='csid_numeric',
        choice_col='choice',
        specification=spec,
        model_type='MNL',
        names=labels
    )
    
    mnl_model.fit_mle(init_vals=np.zeros(len(var_names)), method='bfgs')
    
    # Extract results
    params = mnl_model.params.values
    se = mnl_model.standard_errors.values
    tvals = mnl_model.tvalues.values
    pvals = mnl_model.pvalues.values
    
    # Create coefficient dataframe
    coef_df = pd.DataFrame({
        'variable': var_names,
        'coefficient': params,
        'std_err': se,
        't_value': tvals,
        'p_value': pvals
    })
    
    output_path = Path(args.coef_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coef_df.to_csv(args.coef_output, index=False)
    print(f"\n✓ Saved coefficients to {args.coef_output}")
    print("\n" + coef_df.to_string(index=False))
    
    # Log-likelihood and model stats
    print(f"\nLog-likelihood: {mnl_model.log_likelihood:.2f}")
    print(f"AIC: {mnl_model.aic:.2f}")
    print(f"BIC: {mnl_model.bic:.2f}")
    
    # ──────────────────────────────────────────────────────────
    # WTP Calculation
    # ──────────────────────────────────────────────────────────
    price_idx = var_names.index('price_eur')
    price_coef = params[price_idx]
    
    if price_coef >= 0:
        print("\nWarning: Price coefficient is non-negative. WTP estimates may be invalid.")
    
    # Read coding map
    print(f"\nReading coding map from {args.mapping}...")
    mapping_df = pd.read_csv(args.mapping)
    
    # Group by attribute and compute WTPs
    wtp_records = []
    
    for attr in mapping_df['attribute'].unique():
        attr_map = mapping_df[mapping_df['attribute'] == attr]
        
        # Get coded levels with their coefficients
        coded_levels = []
        for _, row in attr_map.iterrows():
            if pd.notna(row['column']) and row['column'] != 'REFERENCE':
                col_idx = var_names.index(row['column'])
                coded_levels.append({
                    'level': row['level'],
                    'column': row['column'],
                    'coef': params[col_idx]
                })
        
        # Implied level has utility = -sum of coded levels
        implied_row = attr_map[attr_map['implied_level_utility_rule'].notna()]
        if len(implied_row) > 0:
            implied_level = implied_row.iloc[0]['level']
            implied_util = -sum([lv['coef'] for lv in coded_levels])
        else:
            all_levels = attr_map['level'].tolist()
            implied_level = min(all_levels)
            implied_util = -sum([lv['coef'] for lv in coded_levels])
        
        # Collect all utilities
        all_utils = {lv['level']: lv['coef'] for lv in coded_levels}
        all_utils[implied_level] = implied_util
        
        # Baseline = lowest utility
        baseline_level = min(all_utils, key=all_utils.get)
        baseline_util = all_utils[baseline_level]
        
        # Compute WTP for each level vs baseline
        for level, util in all_utils.items():
            wtp = -(util - baseline_util) / price_coef
            wtp_records.append({
                'attribute': attr,
                'level': level,
                'utility': util,
                'baseline_level': baseline_level,
                'wtp_vs_baseline': wtp
            })
    
    wtp_df = pd.DataFrame(wtp_records)
    wtp_df.to_csv(args.wtp_output, index=False)
    print(f"\n✓ Saved WTP per level to {args.wtp_output}")
    print("\n" + wtp_df.to_string(index=False))
    
    # ──────────────────────────────────────────────────────────
    # Attribute Ranges
    # ──────────────────────────────────────────────────────────
    range_records = []
    for attr in wtp_df['attribute'].unique():
        attr_wtp = wtp_df[wtp_df['attribute'] == attr]
        wtp_min = attr_wtp['wtp_vs_baseline'].min()
        wtp_max = attr_wtp['wtp_vs_baseline'].max()
        wtp_range = wtp_max - wtp_min
        
        range_records.append({
            'attribute': attr,
            'min_wtp': wtp_min,
            'max_wtp': wtp_max,
            'wtp_range': wtp_range
        })
    
    range_df = pd.DataFrame(range_records)
    range_df.to_csv(args.range_output, index=False)
    print(f"\n✓ Saved attribute ranges to {args.range_output}")
    print("\n" + range_df.to_string(index=False))
    
    print("\n=== Model fitting complete (VISIBLE as reference) ===")
    
    return params, var_names


if __name__ == '__main__':
    params, var_names = main()
