#!/usr/bin/env python3
"""
Bootstrap WTP confidence intervals.
Performs respondent-level bootstrap resampling to estimate WTP uncertainty.
"""

import argparse
import pandas as pd
import numpy as np
import pylogit as pl
from pathlib import Path
from collections import OrderedDict


def fit_mnl_model(df, spec_dict, names_dict, verbose=False):
    """Fit MNL model using pylogit."""
    model = pl.create_choice_model(
        data=df,
        alt_id_col='alt_num',
        obs_id_col='csid',
        choice_col='choice',
        specification=spec_dict,
        model_type='MNL',
        names=names_dict
    )
    
    try:
        model.fit_mle(init_vals=None)
        return model
    except Exception as e:
        if verbose:
            print(f"      ERROR: {str(e)[:100]}")
        return None


def calculate_wtp(coefficients, mapping_df, price_coef):
    """Calculate WTP for each level relative to baseline."""
    wtp_results = {}
    
    for attr in mapping_df['attribute'].unique():
        attr_map = mapping_df[mapping_df['attribute'] == attr].copy()
        
        # Get coefficients for this attribute
        attr_coeffs = {}
        for _, row in attr_map.iterrows():
            if row['column'] == 'REFERENCE':
                other_coeffs = [coefficients.get(r['column'], 0) 
                               for _, r in attr_map.iterrows() 
                               if r['column'] != 'REFERENCE']
                attr_coeffs[row['level']] = -sum(other_coeffs)
            else:
                attr_coeffs[row['level']] = coefficients.get(row['column'], 0)
        
        # Find baseline (lowest utility level)
        baseline_level = min(attr_coeffs, key=attr_coeffs.get)
        baseline_util = attr_coeffs[baseline_level]
        
        # Calculate WTP for each level vs baseline
        for level, util in attr_coeffs.items():
            wtp = -(util - baseline_util) / price_coef if price_coef != 0 else np.nan
            key = f"{attr}|{level}"
            wtp_results[key] = wtp
    
    return wtp_results


def calculate_attribute_range(wtp_dict, attr):
    """Calculate WTP range for a specific attribute from WTP dictionary."""
    attr_wtps = [v for k, v in wtp_dict.items() if k.startswith(f"{attr}|") and not np.isnan(v)]
    if len(attr_wtps) == 0:
        return np.nan
    return max(attr_wtps) - min(attr_wtps)


def main():
    parser = argparse.ArgumentParser(description='Bootstrap WTP confidence intervals')
    parser.add_argument('--input', default='data/dce_long_model.csv',
                       help='Input processed data CSV')
    parser.add_argument('--mapping', default='output/coding_map.csv',
                       help='Coding mapping file')
    parser.add_argument('--wtp_ci_output', default='output/wtp_levels_ci.csv',
                       help='WTP levels CI output file')
    parser.add_argument('--range_ci_output', default='output/attribute_ranges_ci.csv',
                       help='Attribute ranges CI output file')
    parser.add_argument('--n_bootstrap', type=int, default=300,
                       help='Number of bootstrap iterations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()
    
    print(f"Reading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    # Ensure numeric columns are float type
    numeric_cols = [c for c in df.columns if c not in ['resp_id', 'task_id', 'alt_id', 'csid']]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Reading coding map from {args.mapping}...")
    mapping_df = pd.read_csv(args.mapping)
    
    # Get unique respondents
    respondents = df['resp_id'].unique()
    print(f"Found {len(respondents)} unique respondents")
    
    # Build model specification (same as in 02_fit_mnl_pylogit.py)
    coded_cols = [col for col in df.columns 
                  if any(attr in col for attr in mapping_df['attribute'].unique())
                  and col not in mapping_df['attribute'].unique()]
    
    spec_dict = OrderedDict()
    spec_dict['price_eur'] = 'all_same'
    spec_dict['is_optout'] = 'all_same'
    
    names_dict = OrderedDict()
    names_dict['price_eur'] = 'price'
    names_dict['is_optout'] = 'asc_optout'
    
    for col in coded_cols:
        spec_dict[col] = 'all_same'
        names_dict[col] = col
    
    print(f"\nBootstrapping with {args.n_bootstrap} iterations...")
    print(f"Resampling at respondent level (N={len(respondents)})")
    
    # Storage for bootstrap results
    bootstrap_wtps = []
    bootstrap_ranges = []
    
    np.random.seed(args.seed)
    
    for b in range(args.n_bootstrap):
        if (b + 1) % 50 == 0:
            print(f"  Bootstrap iteration {b + 1}/{args.n_bootstrap}")
        
        # Resample respondents with replacement
        boot_respondents = np.random.choice(respondents, size=len(respondents), replace=True)
        
        # Create bootstrap sample
        boot_dfs = []
        for idx, resp_id in enumerate(boot_respondents):
            resp_data = df[df['resp_id'] == resp_id].copy()
            # Create unique csid for this bootstrap by adding suffix
            resp_data['csid'] = resp_data['csid'] + f"_b{idx}"
            boot_dfs.append(resp_data)
        
        boot_df = pd.concat(boot_dfs, ignore_index=True)
        
        # Fit model on bootstrap sample
        model = fit_mnl_model(boot_df, spec_dict, names_dict, verbose=(b==0))
        
        if model is None:
            if b == 0:
                print(f"    Warning: Bootstrap {b + 1} failed to converge, skipping")
            continue
        
        # Extract coefficients
        coefficients = dict(zip(model.params.index, model.params.values))
        price_coef = coefficients.get('price', np.nan)
        
        if np.isnan(price_coef) or price_coef == 0:
            print(f"    Warning: Bootstrap {b + 1} has invalid price coefficient, skipping")
            continue
        
        # Calculate WTP
        wtp_dict = calculate_wtp(coefficients, mapping_df, price_coef)
        bootstrap_wtps.append(wtp_dict)
        
        # Calculate attribute ranges
        range_dict = {}
        for attr in mapping_df['attribute'].unique():
            range_val = calculate_attribute_range(wtp_dict, attr)
            range_dict[attr] = range_val
        bootstrap_ranges.append(range_dict)
    
    print(f"\n✓ Completed {len(bootstrap_wtps)} successful bootstrap iterations")
    
    # Calculate confidence intervals for WTP levels
    print("\nCalculating WTP confidence intervals...")
    wtp_ci_results = []
    
    # Get all WTP keys from first bootstrap
    if len(bootstrap_wtps) > 0:
        wtp_keys = bootstrap_wtps[0].keys()
        
        for key in wtp_keys:
            attr, level = key.split('|')
            values = [boot[key] for boot in bootstrap_wtps if key in boot and not np.isnan(boot[key])]
            
            if len(values) > 0:
                wtp_ci_results.append({
                    'attribute': attr,
                    'level': level,
                    'median': np.median(values),
                    'ci_lower': np.percentile(values, 2.5),
                    'ci_upper': np.percentile(values, 97.5),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'n_valid': len(values)
                })
    
    wtp_ci_df = pd.DataFrame(wtp_ci_results)
    
    # Calculate confidence intervals for attribute ranges
    print("Calculating attribute range confidence intervals...")
    range_ci_results = []
    
    if len(bootstrap_ranges) > 0:
        attributes = bootstrap_ranges[0].keys()
        
        for attr in attributes:
            values = [boot[attr] for boot in bootstrap_ranges if attr in boot and not np.isnan(boot[attr])]
            
            if len(values) > 0:
                range_ci_results.append({
                    'attribute': attr,
                    'median': np.median(values),
                    'ci_lower': np.percentile(values, 2.5),
                    'ci_upper': np.percentile(values, 97.5),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'n_valid': len(values)
                })
    
    range_ci_df = pd.DataFrame(range_ci_results)
    
    # Save results
    output_path = Path(args.wtp_ci_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wtp_ci_df.to_csv(args.wtp_ci_output, index=False)
    print(f"✓ Saved WTP CI to {args.wtp_ci_output}")
    
    range_ci_df.to_csv(args.range_ci_output, index=False)
    print(f"✓ Saved attribute range CI to {args.range_ci_output}")
    
    print("\n=== WTP Confidence Intervals Summary ===")
    if len(wtp_ci_df) > 0 and 'median' in wtp_ci_df.columns:
        print(wtp_ci_df[['attribute', 'level', 'median', 'ci_lower', 'ci_upper']].to_string(index=False))
    else:
        print("No successful bootstrap iterations - cannot compute confidence intervals")
    
    print("\n=== Attribute Range Confidence Intervals ===")
    if len(range_ci_df) > 0 and 'median' in range_ci_df.columns:
        print(range_ci_df.to_string(index=False))
    else:
        print("No successful bootstrap iterations - cannot compute confidence intervals")


if __name__ == '__main__':
    main()
