# Indian Trade XAI

Reproducibility package for the PSO-, LIME-, and SHAP-based analysis of monthly Indian trade data.

## Study coverage

The workbook contains 60 monthly observations from January 2021 through December 2025 for six commodity sectors:

- International demand (imports): crude oil, cotton, and electronic circuits
- International supply (exports): rice, packed medicine, and spices

The underlying monthly commodity-level trade data are attributed to the Government of India, Department of Commerce, TRADESTAT portal: https://tradestat.commerce.gov.in/

## Repository contents

- `TradeDataUpdated.xlsx`: processed analytical workbook
- `run_analysis.py`: event classification, PSO fitting, monthly LIME analysis, and yearly SHAP aggregation
- `results/`: generated CSV and JSON outputs from the verified reference run
- `figures/`: manuscript figures
- `tables/`: manuscript tables
- `main.tex`: Springer-format manuscript source
- `requirements.txt`: Python dependencies

## Reproduce the analysis

Python 3.9 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 run_analysis.py
```

The script uses fixed random seeds and writes its outputs to `results/`. To use another workbook or output directory:

```bash
python3 run_analysis.py --data /path/to/workbook.xlsx --out /path/to/output
```

## Method summary

Month-on-month target changes are standardized with a sample-standard-deviation z-score and classified using the fixed threshold `|z| = 1.5`. PSO learns a continuous full-vector feature-weight representation on the first 48 months of each sector. LIME produces monthly local explanations, and additive SHAP contributions are aggregated by calendar year.

## Data and privacy

The repository contains no confidential or personally identifiable research data. Users of the underlying TRADESTAT data should follow the source portal's applicable terms and citation requirements.

## Citation

If you use this repository, please cite the accompanying manuscript and the TRADESTAT source listed in `main.tex`.

## License

The analysis code is released under the MIT License. Source data remain subject to the terms of their original provider.
