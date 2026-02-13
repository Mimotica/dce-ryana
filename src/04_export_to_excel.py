#!/usr/bin/env python3
"""
Export all results to a single Excel file with multiple sheets.
"""

import pandas as pd
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Export DCE results to Excel')
    parser.add_argument('--output-dir', default='output', help='Output directory with CSV files')
    parser.add_argument('--excel-file', default='output/dce_results.xlsx', help='Output Excel file')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    excel_file = Path(args.excel_file)
    excel_file.parent.mkdir(parents=True, exist_ok=True)

    # Read all CSV files
    print(f"Reading CSV files from {output_dir}...")
    
    files_to_read = {
        'Coefficients': 'coefficients.csv',
        'WTP_Levels': 'wtp_levels.csv',
        'WTP_Levels_CI': 'wtp_levels_ci.csv',
        'Attribute_Ranges': 'attribute_ranges.csv',
        'Attribute_Ranges_CI': 'attribute_ranges_ci.csv',
        'Coding_Map': 'coding_map.csv'
    }
    
    # Create Excel writer
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        for sheet_name, filename in files_to_read.items():
            filepath = output_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"  ✓ Added sheet: {sheet_name} ({len(df)} rows)")
            else:
                print(f"  ⚠ Skipped {sheet_name}: file not found")
    
    print(f"\n✓ Excel file created: {excel_file}")
    print(f"  Sheets: {', '.join(files_to_read.keys())}")


if __name__ == '__main__':
    main()
