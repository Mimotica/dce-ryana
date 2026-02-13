# DCE Analysis - Hearing Aid Study

This repository contains the data analysis code for a Discrete Choice Experiment (DCE) studying willingness to pay for hearing aid attributes.

## Project Structure

```
├── src/                          # Python analysis scripts
│   ├── 01_prepare_data.py       # Data preprocessing and effects coding
│   ├── 01b_recode_dummies.py    # Additional dummy coding
│   ├── 02_fit_mnl_pylogit.py    # MNL model fitting
│   ├── 03_bootstrap_wtp.py      # Bootstrap WTP confidence intervals
│   ├── 04_calculate_table9.py   # WTP calculations
│   ├── 05_neural_network.py     # Neural network models
│   └── 06_compare_models.py     # Model comparison
├── Data/                         # Data files (raw data not included in repo)
├── output/                       # Analysis results
│   ├── coefficients.csv         # Model coefficients
│   ├── table9_wtp.csv          # WTP estimates by attribute level
│   ├── attribute_ranges.csv    # WTP ranges per attribute
│   └── wtp_levels_ci.csv       # Bootstrap confidence intervals
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
