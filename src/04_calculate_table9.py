"""
Calculate Table 9: Willingness to Pay (WTP) estimates
With corrected reference levels:
- Speech Understanding: baseline = EASY
- Comfort: baseline = 1 HOUR
"""

import pandas as pd
import numpy as np

# Read coefficients
coef_df = pd.read_csv('/Users/ryanabuijsers/Documents/Master/GPS/Data_analysis/output/coefficients.csv')

# Extract price coefficient
price_coef = coef_df[coef_df['variable'] == 'price_eur']['coefficient'].values[0]

print(f"Price coefficient: {price_coef}")
print("\n" + "="*80)
print("CALCULATING WTP WITH FORMULA: WTP = -(β_level - β_baseline) / β_price")
print("="*80 + "\n")

# Initialize results list
results = []

# ============================================================================
# 1. COMFORT (baseline = 1 hour, which has lowest utility)
# ============================================================================
comfort_2h = coef_df[coef_df['variable'] == 'comfort_2.0']['coefficient'].values[0]
comfort_4h = coef_df[coef_df['variable'] == 'comfort_4.0']['coefficient'].values[0]
comfort_1h = -(comfort_2h + comfort_4h)  # Reference level

print("COMFORT ATTRIBUTE")
print(f"  β(1 hour) = {comfort_1h:.4f} [BASELINE]")
print(f"  β(2 hours) = {comfort_2h:.4f}")
print(f"  β(4 hours) = {comfort_4h:.4f}")

wtp_comfort_1h = 0  # Baseline
wtp_comfort_2h = -(comfort_2h - comfort_1h) / price_coef
wtp_comfort_4h = -(comfort_4h - comfort_1h) / price_coef

print(f"\n  WTP(1 hour) = €{wtp_comfort_1h:.2f} [BASELINE]")
print(f"  WTP(2 hours) = €{wtp_comfort_2h:.2f}")
print(f"  WTP(4 hours) = €{wtp_comfort_4h:.2f}\n")

results.append({
    'Attribute': 'Draagcomfort',
    'Level': '1 hour',
    'Coefficient': comfort_1h,
    'WTP_EUR': wtp_comfort_1h,
    'Is_Baseline': True
})
results.append({
    'Attribute': 'Draagcomfort',
    'Level': '2 hours',
    'Coefficient': comfort_2h,
    'WTP_EUR': wtp_comfort_2h,
    'Is_Baseline': False
})
results.append({
    'Attribute': 'Draagcomfort',
    'Level': '4 hours',
    'Coefficient': comfort_4h,
    'WTP_EUR': wtp_comfort_4h,
    'Is_Baseline': False
})

# ============================================================================
# 2. SPEECH UNDERSTANDING (baseline = EASY)
# ============================================================================
speech_moderate = coef_df[coef_df['variable'] == 'speech_moderate']['coefficient'].values[0]
speech_more = coef_df[coef_df['variable'] == 'speech_more_effort']['coefficient'].values[0]
speech_easy = -(speech_moderate + speech_more)  # Reference level

print("SPEECH UNDERSTANDING")
print(f"  β(Easy) = {speech_easy:.4f} [BASELINE]")
print(f"  β(Moderate) = {speech_moderate:.4f}")
print(f"  β(More effort) = {speech_more:.4f}")

wtp_speech_easy = 0  # Baseline
wtp_speech_moderate = -(speech_moderate - speech_easy) / price_coef
wtp_speech_more = -(speech_more - speech_easy) / price_coef

print(f"\n  WTP(Easy) = €{wtp_speech_easy:.2f} [BASELINE]")
print(f"  WTP(Moderate) = €{wtp_speech_moderate:.2f}")
print(f"  WTP(More effort) = €{wtp_speech_more:.2f}\n")

results.append({
    'Attribute': 'Spraakbegrip',
    'Level': 'Easy',
    'Coefficient': speech_easy,
    'WTP_EUR': wtp_speech_easy,
    'Is_Baseline': True
})
results.append({
    'Attribute': 'Spraakbegrip',
    'Level': 'Moderate',
    'Coefficient': speech_moderate,
    'WTP_EUR': wtp_speech_moderate,
    'Is_Baseline': False
})
results.append({
    'Attribute': 'Spraakbegrip',
    'Level': 'More effort',
    'Coefficient': speech_more,
    'WTP_EUR': wtp_speech_more,
    'Is_Baseline': False
})

# ============================================================================
# 3. HEARING PERFORMANCE (baseline = PARTIAL)
# ============================================================================
att_complete = coef_df[coef_df['variable'] == 'att_complete']['coefficient'].values[0]
att_partial = -att_complete  # Reference level

print("HEARING PERFORMANCE")
print(f"  β(Partial) = {att_partial:.4f} [BASELINE]")
print(f"  β(Complete) = {att_complete:.4f}")

wtp_att_partial = 0  # Baseline
wtp_att_complete = -(att_complete - att_partial) / price_coef

print(f"\n  WTP(Partial) = €{wtp_att_partial:.2f} [BASELINE]")
print(f"  WTP(Complete) = €{wtp_att_complete:.2f}\n")

results.append({
    'Attribute': 'Hoortoestel prestatie',
    'Level': 'Partial',
    'Coefficient': att_partial,
    'WTP_EUR': wtp_att_partial,
    'Is_Baseline': True
})
results.append({
    'Attribute': 'Hoortoestel prestatie',
    'Level': 'Complete',
    'Coefficient': att_complete,
    'WTP_EUR': wtp_att_complete,
    'Is_Baseline': False
})

# ============================================================================
# 4. VISIBILITY (baseline = INVISIBLE)
# ============================================================================
app_visible = coef_df[coef_df['variable'] == 'app_visible']['coefficient'].values[0]
app_invisible = -app_visible  # Reference level

print("VISIBILITY")
print(f"  β(Invisible) = {app_invisible:.4f} [BASELINE]")
print(f"  β(Visible) = {app_visible:.4f}")

wtp_app_invisible = 0  # Baseline
wtp_app_visible = -(app_visible - app_invisible) / price_coef

print(f"\n  WTP(Invisible) = €{wtp_app_invisible:.2f} [BASELINE]")
print(f"  WTP(Visible) = €{wtp_app_visible:.2f}\n")

results.append({
    'Attribute': 'Zichtbaarheid',
    'Level': 'Invisible',
    'Coefficient': app_invisible,
    'WTP_EUR': wtp_app_invisible,
    'Is_Baseline': True
})
results.append({
    'Attribute': 'Zichtbaarheid',
    'Level': 'Visible',
    'Coefficient': app_visible,
    'WTP_EUR': wtp_app_visible,
    'Is_Baseline': False
})

# ============================================================================
# Create DataFrame and save
# ============================================================================
table9 = pd.DataFrame(results)

# Reorder columns
table9 = table9[['Attribute', 'Level', 'Coefficient', 'WTP_EUR', 'Is_Baseline']]

# Save to CSV
output_path = '/Users/ryanabuijsers/Documents/Master/GPS/Data_analysis/output/table9_wtp.csv'
table9.to_csv(output_path, index=False, float_format='%.2f')

print("="*80)
print("TABLE 9 - WILLINGNESS TO PAY")
print("="*80)
print(table9.to_string(index=False))
print(f"\n✓ Saved to: {output_path}")
