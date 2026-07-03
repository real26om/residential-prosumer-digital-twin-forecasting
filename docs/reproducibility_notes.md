# Reproducibility Notes

## Recommended Run Order

1. Review configuration under `src/config/`.
2. Install dependencies with `pip install -r requirements.txt`.
3. If raw data is available locally, run preprocessing scripts under `src/preprocessing/` to rebuild the hourly master dataset.
4. Run A1-B4 dataset-design scripts under `src/dataset_design/`.
5. Run split-generation scripts under `src/splits/`.
6. Run model-training scripts under `src/model_training/`.
7. Generate predictions using `src/prediction/` scripts/material.
8. Run evaluation scripts under `src/evaluation/`.
9. Run residual-MAD anomaly-detection scripts under `src/anomaly_detection/`.
10. Regenerate reporting outputs with `src/reporting/`.

## Generated Versus Manual Material

The `data/` and `outputs/` folders in this GitHub-ready export contain selected derived material copied from Candidate C. They should be treated as reproducibility aids, not as a substitute for rebuilding from raw data when a full private local dataset is available.

## Exclusions

Raw Home Assistant exports, large model artifacts, `.joblib` files, caches, duplicate-review folders, and unclassified archive material are excluded from this export.
