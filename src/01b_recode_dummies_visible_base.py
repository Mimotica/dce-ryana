#!/usr/bin/env python3
"""
Re-code dummy variables with VISIBLE as reference for app attribute (instead of INVISIBLE).
Speech Understanding: EASY (reference)
Comfort: 1 hour (reference) 
Hearing Performance: PARTIAL (reference)
Visibility: VISIBLE (reference) - REVERSED from original
"""

import pandas as pd
import numpy as np

# Read the data
print("Reading data...")
df = pd.read_csv('/Users/ryanabuijsers/Documents/Master/GPS/Data_analysis/Data/dce_long_model.csv')

print("Original columns:", df.columns.tolist())
print("\nCurrent dummy structure detected:")

# Analyze current structure
base_cols = ['resp_id', 'task_id', 'alt_id', 'csid', 'alt_num', 'choice', 'is_optout', 'price_eur']

# COMFORT: Check current dummies
if 'comfort_2.0' in df.columns and 'comfort_4.0' in df.columns:
    print("  - comfort: Has comfort_2.0, comfort_4.0 → reference is currently 1.0 ✓ (correct)")
    # Reconstruct current levels
    df['comfort'] = 1  # reference
    df.loc[df['comfort_2.0'] == 1, 'comfort'] = 2
    df.loc[df['comfort_4.0'] == 1, 'comfort'] = 4
elif 'comfort_1.0' in df.columns and 'comfort_2.0' in df.columns:
    print("  - comfort: Has comfort_1.0, comfort_2.0 → reference is currently 4.0 (fixing...)")
    df['comfort'] = 4  # reference
    df.loc[df['comfort_1.0'] == 1, 'comfort'] = 1
    df.loc[df['comfort_2.0'] == 1, 'comfort'] = 2

# ATT: Check current dummies  
if 'att_partial' in df.columns:
    print("  - att: Has att_partial → reference is currently COMPLETE (fixing...)")
    df['att'] = 'COMPLETE'  # reference
    df.loc[df['att_partial'] == 1, 'att'] = 'PARTIAL'
elif 'att_complete' in df.columns:
    print("  - att: Has att_complete → reference is currently PARTIAL ✓ (correct)")
    df['att'] = 'PARTIAL'  # reference
    df.loc[df['att_complete'] == 1, 'att'] = 'COMPLETE'

# SPEECH: Check current dummies
if 'speech_moderate' in df.columns and 'speech_more_effort' in df.columns:
    print("  - speech: Has speech_moderate, speech_more_effort → reference is currently EASY ✓ (correct)")
    df['speech'] = 'EASY'  # reference
    df.loc[df['speech_moderate'] == 1, 'speech'] = 'MODERATE'
    df.loc[df['speech_more_effort'] == 1, 'speech'] = 'MORE EFFORT'
elif 'speech_easy' in df.columns and 'speech_moderate' in df.columns:
    print("  - speech: Has speech_easy, speech_moderate → reference is currently MORE EFFORT (fixing...)")
    df['speech'] = 'MORE EFFORT'  # reference
    df.loc[df['speech_easy'] == 1, 'speech'] = 'EASY'
    df.loc[df['speech_moderate'] == 1, 'speech'] = 'MODERATE'

# APP: Check current dummies
if 'app_visible' in df.columns:
    print("  - app: Has app_visible → reference is currently INVISIBLE (keeping for new base)")
    df['app'] = 'INVISIBLE'  # reference
    df.loc[df['app_visible'] == 1, 'app'] = 'VISIBLE'
elif 'app_invisible' in df.columns:
    print("  - app: Has app_invisible → reference is currently VISIBLE (reconstructing...)")
    df['app'] = 'VISIBLE'  # reference
    df.loc[df['app_invisible'] == 1, 'app'] = 'INVISIBLE'

# Create dummy coding with CORRECT references
print("\n=== Creating dummy variables with NEW references ===")
print("Target references:")
print("  - comfort: reference = 1")
print("  - att: reference = PARTIAL")
print("  - speech: reference = EASY")
print("  - app: reference = VISIBLE ← CHANGED (was INVISIBLE)") 

# COMFORT: reference = 1 (code 2 and 4)
df['comfort_2.0'] = 0
df['comfort_4.0'] = 0
df.loc[df['comfort'] == 2, 'comfort_2.0'] = 1
df.loc[df['comfort'] == 4, 'comfort_4.0'] = 1
df.loc[df['comfort'] == 1, 'comfort_2.0'] = -1
df.loc[df['comfort'] == 1, 'comfort_4.0'] = -1

# ATT: reference = PARTIAL (code COMPLETE)
df['att_complete'] = 0
df.loc[df['att'] == 'COMPLETE', 'att_complete'] = 1
df.loc[df['att'] == 'PARTIAL', 'att_complete'] = -1

# SPEECH: reference = EASY (code MODERATE and MORE EFFORT)
df['speech_moderate'] = 0
df['speech_more_effort'] = 0
df.loc[df['speech'] == 'MODERATE', 'speech_moderate'] = 1
df.loc[df['speech'] == 'MORE EFFORT', 'speech_more_effort'] = 1
df.loc[df['speech'] == 'EASY', 'speech_moderate'] = -1
df.loc[df['speech'] == 'EASY', 'speech_more_effort'] = -1

# APP: reference = VISIBLE (code INVISIBLE) - REVERSED!
df['app_invisible'] = 0
df.loc[df['app'] == 'INVISIBLE', 'app_invisible'] = 1
df.loc[df['app'] == 'VISIBLE', 'app_invisible'] = -1

# Keep only model-ready columns
final_cols = base_cols + ['comfort_2.0', 'comfort_4.0', 'att_complete', 
                          'speech_moderate', 'speech_more_effort', 'app_invisible']
df_final = df[final_cols]

# Save with new filename to avoid overwriting original
output_path = '/Users/ryanabuijsers/Documents/Master/GPS/Data_analysis/Data/dce_long_model_visible_base.csv'
df_final.to_csv(output_path, index=False)
print(f"\n✓ Saved recoded data to {output_path}")
print(f"  Rows: {len(df_final)}")
print(f"  Columns: {df_final.columns.tolist()}")

print("\n=== Recoding complete ===")
