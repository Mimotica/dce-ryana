# DCE Analysis - Hearing Aid Study

This repository contains the data analysis code for a Discrete Choice Experiment (DCE) studying willingness to pay for hearing aid attributes.

## Project Structure

```
├── src/                                      # Python analysis scripts
│   ├── 01_prepare_data.py                   # Data preprocessing and effects coding
│   ├── 01b_recode_dummies.py                # Dummy coding (INVISIBLE as reference)
│   ├── 01b_recode_dummies_visible_base.py   # Dummy coding (VISIBLE as reference)
│   ├── 02_fit_mnl_pylogit.py                # MNL model fitting (INVISIBLE base)
│   ├── 02_fit_mnl_pylogit_visible_base.py   # MNL model fitting (VISIBLE base)
│   ├── 03_bootstrap_wtp.py                  # Bootstrap WTP confidence intervals
│   ├── 04_calculate_table9.py               # WTP calculations
│   ├── 05_neural_network.py                 # Neural network models
│   └── 06_compare_models.py                 # Model comparison
├── Data/                                     # Data files
│   ├── 260209.csv                           # Raw survey data
│   ├── dce_long_raw.csv                     # Preprocessed long format data
│   ├── dce_long_model.csv                   # Model-ready (INVISIBLE base)
│   └── dce_long_model_visible_base.csv      # Model-ready (VISIBLE base)
├── output/                                   # Analysis results
│   ├── coefficients.csv                     # Model coefficients (INVISIBLE base)
│   ├── coefficients_visible_base.csv        # Model coefficients (VISIBLE base)
│   ├── coefficients_formatted.txt           # Formatted results (INVISIBLE base)
│   ├── coefficients_formatted_visible_base.txt  # Formatted results (VISIBLE base)
│   ├── coding_map.csv                       # Effects coding map (INVISIBLE base)
│   ├── coding_map_visible_base.csv          # Effects coding map (VISIBLE base)
│   ├── table9_wtp.csv                       # WTP estimates by attribute level
│   ├── attribute_ranges.csv                 # WTP ranges per attribute (INVISIBLE base)
│   ├── attribute_ranges_visible_base.csv    # WTP ranges per attribute (VISIBLE base)
│   ├── wtp_levels.csv                       # WTP levels (INVISIBLE base)
│   ├── wtp_levels_visible_base.csv          # WTP levels (VISIBLE base)
│   └── wtp_levels_ci.csv                    # Bootstrap confidence intervals
└── README.md

```

## Attributes Studied

- **Draagcomfort (Comfort)**: 1 hour, 2 hours, 4 hours
- **Spraakbegrip (Speech Understanding)**: Easy, Moderate, More effort  
- **Hoortoestel prestatie (Performance)**: Partial, Complete
- **Zichtbaarheid (Visibility)**: Invisible, Visible

## Analysis Pipeline

1. **Data Preparation**: Effects coding of categorical attributes
2. **MNL Model**: Multinomial logit model estimation
3. **Bootstrap**: Confidence interval estimation (300 iterations)
4. **WTP Calculation**: Willingness-to-pay estimates using formula: WTP = -(β_level - β_baseline) / β_price

## Requirements

- Python 3.x
- pandas
- numpy
- pylogit
- sklearn (for neural network models)

## Usage

Run scripts in order:
```bash
python src/01_prepare_data.py
python src/02_fit_mnl_pylogit.py
python src/03_bootstrap_wtp.py
python src/04_calculate_table9.py
```

## Notes

WTP estimates should be interpreted with caution due to relatively weak price coefficient. Focus on relative importance of attributes rather than absolute euro values.
