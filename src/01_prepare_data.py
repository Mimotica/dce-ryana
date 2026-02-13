#!/usr/bin/env python3
"""
Data preparation script for DCE analysis.
Converts raw long-format DCE data into model-ready format with effects coding.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def effects_code_attribute(df, attr_col, attr_name, force_baseline=None):
    """
    Effects-code a categorical attribute (K levels -> K-1 dummy columns).
    Returns DataFrame with coded columns and mapping information.
    
    Args:
        df: DataFrame
        attr_col: Column name to effects-code
        attr_name: Name for the attribute (used in output column names)
        force_baseline: Optional specific level to use as baseline/reference
    """
    # Get unique levels (excluding NaN/empty)
    levels = df[attr_col].dropna().unique()
    levels = [l for l in levels if l != '' and str(l).strip() != '']
    levels = sorted(levels)
    
    if len(levels) <= 1:
        return df, []
    
    # Reference level: use forced baseline if provided, otherwise last sorted level
    if force_baseline is not None and force_baseline in levels:
        reference_level = force_baseline
    else:
        reference_level = levels[-1]
    mappings = []
    
    # Create K-1 dummy columns for all levels EXCEPT the reference
    coded_levels = [l for l in levels if l != reference_level]
    for level in coded_levels:
        # Convert level to string for column naming
        level_str = str(level).lower().replace(' ', '_')
        col_name = f"{attr_name}_{level_str}"
        df[col_name] = 0
        
        # Effects coding: +1 if level matches, 0 if other non-reference, -1 if reference
        mask_level = df[attr_col] == level
        mask_reference = df[attr_col] == reference_level
        
        df.loc[mask_level, col_name] = 1
        df.loc[mask_reference, col_name] = -1
        
        mappings.append({
            'attribute': attr_name,
            'level': level,
            'column': col_name,
            'implied_level_utility_rule': f"{reference_level} = -sum(all other {attr_name} levels)"
        })
    
    # Add reference level to mapping (no column, implied)
    mappings.append({
        'attribute': attr_name,
        'level': reference_level,
        'column': 'REFERENCE',
        'implied_level_utility_rule': f"Utility = -sum(all other {attr_name} coefficients)"
    })
    
    return df, mappings


def main():
    parser = argparse.ArgumentParser(description='Prepare DCE data for MNL modeling')
    parser.add_argument('--input', default='data/dce_long_raw.csv',
                       help='Input CSV file path')
    parser.add_argument('--output', default='data/dce_long_model.csv',
                       help='Output CSV file path')
    parser.add_argument('--mapping', default='output/coding_map.csv',
                       help='Coding mapping output file path')
    args = parser.parse_args()
    
    print(f"Reading data from {args.input}...")
    df = pd.read_csv(args.input, sep=';')
    
    # Sanity check: Verify required columns exist
    required_cols = ['respondent', 'task', 'alt', 'chosen', 'price_eur']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Loaded {len(df)} rows")
    
    # Rename columns to standard names
    df = df.rename(columns={
        'respondent': 'resp_id',
        'task': 'task_id',
        'alt': 'alt_id'
    })
    
    # Create csid (choice situation ID)
    df['csid'] = df['resp_id'].astype(str) + '_' + df['task_id'].astype(str)
    
    # Create is_optout flag (alt_id == 3 is the opt-out)
    df['is_optout'] = (df['alt_id'] == 3).astype(int)
    
    # Create alt_num (1-indexed alternative number)
    # For each choice situation, number alternatives 1, 2, 3, ...
    df = df.sort_values(['csid', 'alt_id'])
    df['alt_num'] = df.groupby('csid').cumcount() + 1
    
    # Convert price_eur to float
    df['price_eur'] = pd.to_numeric(df['price_eur'], errors='coerce').fillna(0)
    
    # Identify categorical attribute columns (excluding utility columns)
    # Based on the data structure: comfort, att, speech, app
    categorical_attrs = ['comfort', 'att', 'speech', 'app']
    
    # Filter to only existing columns
    categorical_attrs = [col for col in categorical_attrs if col in df.columns]
    
    print(f"\nEffects-coding categorical attributes: {categorical_attrs}")
    
    # Define specific baselines for certain attributes (rest will use last sorted level)
    # Effects coding: the baseline level gets NO dummy variable (utility = -sum of others)
    baseline_levels = {
        'comfort': 1,           # 1 hour as baseline/reference
        'att': 'PARTIAL',       # PARTIAL as baseline/reference (COMPLETE gets a dummy)
        'speech': 'EASY',       # EASY as baseline/reference
        'app': 'INVISIBLE'      # INVISIBLE as baseline/reference
    }
    
    all_mappings = []
    for attr in categorical_attrs:
        baseline = baseline_levels.get(attr, None)
        print(f"  Coding {attr}... (baseline: {baseline if baseline else 'auto'})")
        df, mappings = effects_code_attribute(df, attr, attr, force_baseline=baseline)
        all_mappings.extend(mappings)
    
    # Rename chosen to choice for consistency (do this before sanity checks)
    df = df.rename(columns={'chosen': 'choice'})
    
    # Sanity check: verify exactly one choice per csid
    print("\nPerforming sanity checks...")
    choice_counts = df.groupby('csid')['choice'].sum()
    invalid_csids = choice_counts[choice_counts != 1]
    
    if len(invalid_csids) > 0:
        print(f"WARNING: {len(invalid_csids)} choice situations do not have exactly 1 choice:")
        print(invalid_csids.head(10))
        raise ValueError("Data validation failed: not all choice situations have exactly one chosen alternative")
    else:
        print(f"✓ All {len(choice_counts)} choice situations have exactly 1 choice")
    
    # Check for any NaN in critical columns
    critical_cols = ['csid', 'alt_num', 'choice', 'price_eur', 'is_optout']
    for col in critical_cols:
        if df[col].isna().any():
            print(f"WARNING: Column {col} has {df[col].isna().sum()} NaN values")
    
    # Drop original categorical columns and other non-model columns
    # Keep only: structural columns + effects-coded columns
    structural_keep = ['resp_id', 'task_id', 'alt_id', 'csid', 'alt_num', 'choice', 'is_optout', 'price_eur']
    effects_cols = [m['column'] for m in all_mappings if m['column'] != 'REFERENCE']
    keep_cols = structural_keep + effects_cols
    
    # Drop all other columns
    df = df[keep_cols]
    print(f"\nFiltered to {len(keep_cols)} columns for model data")
    
    # Save processed data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nWriting processed data to {args.output}...")
    df.to_csv(args.output, index=False)
    print(f"✓ Saved {len(df)} rows")
    
    # Save coding mapping
    mapping_path = Path(args.mapping)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    
    if all_mappings:
        mapping_df = pd.DataFrame(all_mappings)
        mapping_df.to_csv(args.mapping, index=False)
        print(f"✓ Saved coding map with {len(mapping_df)} entries to {args.mapping}")
    
    print("\n=== Summary ===")
    print(f"Respondents: {df['resp_id'].nunique()}")
    print(f"Choice situations: {df['csid'].nunique()}")
    print(f"Alternatives: {len(df)}")
    print(f"Opt-out choices: {df[df['is_optout'] == 1]['choice'].sum()}")
    print(f"Purchase choices: {df[(df['is_optout'] == 0) & (df['choice'] == 1)].shape[0]}")
    print(f"\nEffects-coded variables: {len([m for m in all_mappings if m['column'] != 'REFERENCE'])}")


if __name__ == '__main__':
    main()
