# Cytokine Plasma Survival Modeling

This repository contains data processing, exploratory analysis, survival modeling, and model interpretation workflows for predicting patient survival from plasma cytokine measurements.

The project combines Python and R workflows:

- Python notebooks and scripts cover preprocessing, feature selection, Random Survival Forest (RSF) modeling, evaluation, and SHAP interpretation.
- R scripts generate formatted datasets and summary statistics tables for demographic/clinical and cytokine variables.

## Project Structure

| Path | Description |
| --- | --- |
| `data/` | Input and intermediate dataset versions (cleaned/formatted CSV files, date-stamped). |
| `data_processed/` | Model artifacts and feature-set evaluation summaries. |
| `figures/` | Generated plots (SHAP, calibration, Kaplan-Meier, time-dependent AUC, feature comparisons). |
| `notebooks/` | End-to-end analysis notebooks for cleaning, EDA, model selection, tuning, and interpretation. |
| `scripts/` | Reusable script entry points for RSF feature-set evaluation (Python) and cohort summaries (R). |
| `summary_stats/` | Generated summary-statistics report documents. |
| `pyproject.toml` | Python project metadata and dependencies. |

## Data Assets

The `data/` folder includes date-stamped versions of combined plasma data in cleaned and formatted forms, for example:

- `CV_plasma_Combined_cleaned_20260731.csv`
- `CV_plasma_Combined_formatted_20260731.csv`

The `data_processed/` folder includes:

- `rsf_bundle.joblib`: serialized RSF bundle artifact.
- RSF feature-set summary CSVs for different training/validation cohort configurations.

Notes:

- Cohort labels used throughout the repository include `LLU`, `UAB`, and pooled variants.
- The exact cohort acronym definitions are not documented in this repository and should be interpreted according to project-specific study context.

## Analysis Workflow (Notebook Pipeline)

Typical sequence:

1. `notebooks/cytokines_data_cleaning.ipynb` - data cleaning and preparation.
2. `notebooks/cytokines_eda.ipynb` - exploratory data analysis.
3. `notebooks/features_selection.ipynb` - feature selection.
4. `notebooks/hyperparameter_tuning_rsf.ipynb` - RSF hyperparameter tuning.
5. `notebooks/cytokines_ml.ipynb` - core modeling workflows.
6. Cohort/model selection notebooks:
   - `notebooks/llu_cytokines_models_selection.ipynb`
   - `notebooks/uab_cytokines_models_selection.ipynb`
   - `notebooks/pooled_cytokines_models_selection.ipynb`
   - `notebooks/llu_uab_cytokines_models_selection.ipynb`
7. `notebooks/cytokines_rsf_shap.ipynb` - SHAP-based model interpretation.
8. `notebooks/survival_stats.ipynb` and `notebooks/mediation.ipynb` - statistical and mediation analyses.

Additional exploration notebooks are available for cohort-specific and pooled model exploration.

## Reusable Scripts

### Python: RSF Feature-Set Evaluation

`scripts/survival_feature_eval.py` provides a reusable pipeline for:

- feature subset filtering,
- stratified train/test split,
- per-cohort standardization,
- RSF training,
- C-index and time-dependent AUC evaluation,
- optional AUC plotting.

### R: Cohort Formatting and Summary Tables

`scripts/cytokine_summary.R`:

- reads a raw combined CSV from `data/`,
- recodes/labels categorical demographic and clinical variables,
- writes a formatted CSV,
- produces summary documents in `summary_stats/`.

## Environment Setup

Python requirement from `pyproject.toml`:

- Python `>=3.12`

Install Python dependencies (from repository root):

```bash
pip install -e .
```

Key Python packages include:

- `scikit-survival`, `lifelines`, `scikit-learn`
- `shap`, `statsmodels`, `hyperopt`, `boruta`
- `pandas`, `numpy`, `polars`, `seaborn`

R workflow dependencies (used in `scripts/cytokine_summary.R`) include:

- `tidyverse`, `gtsummary`, `flextable`

## Outputs

Generated outputs include:

- SHAP plots and feature-importance visualizations in `figures/`.
- Survival/evaluation plots (including time-dependent AUC and calibration) in `figures/`.
- Summary tables in `summary_stats/` as `.docx` files.
- Processed model summaries and artifacts in `data_processed/`.

## Reproducibility Notes

- Several scripts and notebooks use fixed random states for model splitting/training.
- Data files are versioned by date in filenames; choose one date version consistently across a run.
- This repository documents workflow artifacts but does not currently provide a single command orchestrator for the full pipeline.
