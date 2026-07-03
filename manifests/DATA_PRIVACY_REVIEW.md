# Data Privacy Review

## Public-Complete Version

This folder is the public-complete GitHub version prepared for EC3 presentation and research reproducibility. It includes processed datasets, final A1-B4 derived datasets, rolling split datasets, predictions, metrics, anomaly-detection outputs, figures, documentation, and supporting code.

Raw Home Assistant exports are excluded. Private raw data, trained `.joblib` model artifacts, model artifact folders, cache/checkpoint folders, archive/unclassified material, and duplicate-review folders are not included.

## Privacy Limitations

Processed datasets may still encode household energy behavior, timestamped demand patterns, and residential/prosumer operating rhythms. Users should not attempt to infer private occupancy, household behavior, appliance usage, personal routines, or resident identity from these files.

Exact timestamps and energy patterns are provided only to support reproducibility of the research workflow, including dataset construction, split generation, forecast evaluation, and residual-MAD anomaly detection.

## Before Public Upload or Redistribution

- Review all files under `data/` and `outputs/` before publishing.
- Consider anonymization, aggregation, date shifting, or sample/demo extracts if public sharing risk is too high.
- Keep raw Home Assistant exports private unless explicit privacy approval is granted.
- Do not publish trained binary artifacts unless storage, licensing, and privacy constraints are reviewed.
