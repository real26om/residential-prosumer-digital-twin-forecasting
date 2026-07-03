# Methodology Summary

This project reconstructs a residential prosumer forecasting and anomaly-detection workflow from Home Assistant telemetry to final thesis/paper outputs.

## Pipeline

Raw Home Assistant sensor exports are transformed into an hourly master table. The preprocessing step standardizes time columns, aligns sensor streams, handles missingness, and prepares demand/telemetry features for modelling.

The dataset-design stage creates the A1-B4 modelling matrix. The variants preserve experimental meaning, including demand-only versus demand-plus-telemetry inputs and clean-only versus imputed-row designs.

Split-generation scripts create chronological holdout and rolling robustness splits. Phase 1 is the fixed chronological holdout package; Phase 2 is the rolling robustness/current workflow.

Forecasting methods visible in the organized project include persistence-style baselines, multiple linear regression, and random-forest style modelling/tuning. Forecasts are evaluated with standard error metrics and then used for residual-MAD anomaly detection.

Residual-MAD anomaly detection uses model residuals, median absolute deviation style thresholds, and K sweeps to examine sensitivity. Label evaluation, missingness checks, and robustness checks support interpretation.
