# A Monitoring-Oriented Digital Twin Pipeline for Data-Driven Prediction of Residential Prosumer Energy Profiles

This repository contains the reproducible code, processed datasets, derived outputs, and documentation supporting the MSc thesis and related EC3 paper:

**A Monitoring-Oriented Digital Twin Pipeline for Data-Driven Prediction of Residential Prosumer Energy Profiles**

The project investigates how monitored residential prosumer data can be transformed into a data-driven digital twin pipeline for short-term electricity demand prediction, model comparison, and residual-based anomaly detection.

The repository is prepared as a privacy-conscious public version of the research workflow. It includes processed and derived data needed for reproducibility, while excluding raw Home Assistant exports, trained binary model artifacts, cache files, and local-only archives.

## Project Summary

The study uses a monitored detached residential villa as a case study. Operational data from the home monitoring system are processed into an hourly master dataset and then transformed into multiple modelling datasets. These datasets are used to evaluate one-hour-ahead residential electricity demand prediction.

The workflow combines:

- Home Assistant-based operational monitoring data
- Hourly preprocessing and quality checks
- A1-B4 dataset design for different feature and missing-data policies
- Chronological holdout and rolling robustness evaluation
- Random Forest, Multiple Linear Regression, and persistence-based forecasting
- Residual-MAD anomaly detection
- Derived predictions, metrics, figures, and reproducibility documentation

The aim is to make the research workflow transparent and reusable while protecting private household-level raw data.

## Research Context

Residential prosumers both consume and generate electricity through interacting systems such as household demand, photovoltaic production, HVAC-related loads, and other monitored subsystems.

In this project, the digital twin is treated as a monitoring-oriented and data-driven pipeline. The focus is not only on geometric representation, but on the use of operational data to support prediction, evaluation, and anomaly detection in a residential prosumer context.

The main research objective is to test whether monitored operational data can support reliable short-term demand prediction and residual-based anomaly detection.

## Case Study and Data

The case study is a monitored residential villa in the Milan metropolitan area. The original monitoring system included streams such as:

- Whole-building electricity demand
- Photovoltaic power
- Pool-related electricity use
- Outdoor air temperature from the ventilation system

Raw Home Assistant exports are **not included** in this public repository because they may reveal private household behaviour, occupancy patterns, appliance operation, or residential routines.

Instead, this repository includes selected processed and derived datasets needed to support reproducibility of the thesis and EC3 workflow.

## Dataset Design

The project uses an A1-B4 dataset matrix to compare different modelling assumptions.

The datasets vary according to:

- Demand-only versus demand-plus-telemetry features
- Causal or online-like filling versus interpolation-based variants
- Keep-imputed versus clean-only row policies
- Base telemetry versus telemetry-lag feature sets

This design makes it possible to test how data availability, missingness treatment, and telemetry features affect one-hour-ahead demand prediction.

Identically valued files may still represent different experimental conditions when filenames encode different A/B or A1-B4 variants.

## Modelling and Evaluation

The forecasting task is:

**Predict residential electricity demand one hour ahead.**

The main machine-learning model is Random Forest regression. It is compared with simpler baselines such as Multiple Linear Regression and persistence-based forecasting.

Model development includes:

- Hyperparameter search
- Stable-parameter selection
- Grid-search polishing
- Final evaluation across the A1-B4 datasets
- Alternative modelling attempts for comparison

Evaluation includes:

- MAE and RMSE
- Chronological holdout evaluation
- Rolling robustness checks
- Day/night and segmented error analysis
- Residual-MAD anomaly detection
- Event-level and hour-level anomaly evaluation

## Repository Structure

```text
data/
  01 processed_master/
  02 final_datasets_A1_B4/
  03 rolling_datasets_by_split/

docs/
  data_dictionary.md
  methodology_summary.md
  reproducibility_notes.md

manifests/
  DATA_PRIVACY_REVIEW.md

outputs/
  anomaly_detection/
  metrics/
  predictions/

src/
  01 preprocessing/
  02 dataset_design/
  03 splits/
  04 model_training/
  05 evaluation/
  06 anomaly_detection/
  config/

README.md
requirements.txt
CITATION.cff
LICENSE
```

## Reproducibility Workflow

A typical reproduction path is:

1. Review `manifests/DATA_PRIVACY_REVIEW.md`.
2. Install the required Python packages from `requirements.txt`.
3. Review the configuration files under `src/config/`.
4. Rebuild or inspect the processed hourly master dataset.
5. Generate or inspect the A1-B4 modelling datasets.
6. Generate chronological and rolling split tables.
7. Run model training and prediction scripts.
8. Evaluate forecasting performance using MAE and RMSE.
9. Run residual-MAD anomaly detection.
10. Review the derived outputs and documentation.

A more detailed script-order description is provided in:

```text
docs/reproducibility_notes.md
```

## Model Development and Tuning History

The folder `src/04 model_training/experimental_tuning_history/` documents the model-development path used to select and validate the final Random Forest workflow.

It preserves earlier experimental scripts and outputs, including:

- HalvingGridSearchCV experiments
- Stable-parameter selection
- Stage 2 / Stage 3 polish GridSearchCV
- Final all-8-dataset Random Forest evaluation
- Paper-level model-selection scripts
- Alternative modelling attempts, including target transformation, regime switching, and import/export decomposition

These files are kept as research history. They show how the final modelling workflow was developed and validated, but they should not be interpreted as the main current pipeline.

## Included and Excluded Material

Included in this repository:

- Supporting Python scripts
- Configuration files
- Processed hourly/master datasets
- Final A1-B4 modelling datasets
- Rolling split datasets
- Prediction outputs
- Evaluation metrics
- Anomaly-detection outputs
- Documentation and reproducibility notes

Excluded from this repository:

- Raw Home Assistant exports
- Trained `.joblib` model artifacts
- Large binary model outputs
- Cache and checkpoint folders
- Manual-review archives
- Local-only migration or duplicate-review folders

## Data and Privacy Policy

The processed datasets included here are shared for academic reproducibility. However, even processed energy datasets may still encode household operating rhythms and timestamped behaviour patterns.

Users should not attempt to infer occupancy, personal routines, resident identity, appliance usage, or private household behaviour from the data.

Raw Home Assistant exports and trained binary artifacts may be available only upon reasonable request, subject to privacy approval and applicable research/data-sharing constraints.

For more details, see:

```text
manifests/DATA_PRIVACY_REVIEW.md
```

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The project was developed using Python-based data science tools including pandas, NumPy, scikit-learn, pvlib, joblib, and openpyxl.

## Citation

If you use this repository, please cite the thesis or related EC3 paper when available.

Citation metadata is provided in:

```text
CITATION.cff
```

## License

The code is released under the MIT License. See:

```text
LICENSE
```

The included processed datasets and derived outputs are shared for academic reproducibility and remain subject to the privacy limitations described in:

```text
manifests/DATA_PRIVACY_REVIEW.md
```
