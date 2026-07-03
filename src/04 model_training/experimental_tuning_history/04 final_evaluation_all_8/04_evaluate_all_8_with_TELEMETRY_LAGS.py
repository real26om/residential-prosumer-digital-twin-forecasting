import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

TARGET = "y_demand_t_plus_1"
DROP = ["ts_utc", "is_train", "is_test", TARGET]

OUT_CSV = "rf_results_all_8_datasets_168_TELEMETRY_LAGS.csv"

# Columns that may exist in A3_LAGS but be absent in A4_LAGS/B4_LAGS, etc.
# If absent, treat them as always-false flags (0).
OPTIONAL_ZERO_COLS = {
    "flag_demand_imputed_new",
    "flag_pool_imputed_new",
    "flag_pv_imputed_new",
    "flag_temp_imputed_new",
}

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def safe_mape_percent(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)

def default_feature_cols(df):
    cols = [c for c in df.columns if c not in DROP]
    return sorted(cols)

def load_model_spec(path):
    """
    Accept either:
    - artifact dict: {"params": {...}, "feature_cols": [...]}
    - params-only dict: {...}
    Returns: (params_dict, feature_cols_or_None)
    """
    obj = joblib.load(path)
    if isinstance(obj, dict) and ("params" in obj) and ("feature_cols" in obj):
        return obj["params"], obj["feature_cols"]
    if isinstance(obj, dict):
        return obj, None
    raise ValueError(f"Unsupported joblib content in {path}: {type(obj)}")

def build_rf(params):
    valid = RandomForestRegressor().get_params().keys()
    clean_params = {k: v for k, v in params.items() if k in valid}
    return RandomForestRegressor(**clean_params)

def ensure_feature_cols(df, feature_cols, dataset_code):
    """
    Ensure all columns in feature_cols exist in df.
    - If a missing column is in OPTIONAL_ZERO_COLS -> create it as 0.
    - Otherwise -> raise (it’s a real schema mismatch).
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if not missing:
        return df

    missing_optional = [c for c in missing if c in OPTIONAL_ZERO_COLS]
    missing_hard = [c for c in missing if c not in OPTIONAL_ZERO_COLS]

    if missing_optional:
        for c in missing_optional:
            df[c] = 0

    if missing_hard:
        raise ValueError(f"{dataset_code}: Missing required columns in CSV: {missing_hard}")

    return df

def eval_one(dataset_code, group, csv_path, model_joblib):
    params, feature_cols = load_model_spec(model_joblib)

    df = pd.read_csv(csv_path)
    train = df[df["is_train"] == 1].copy()
    test  = df[df["is_test"] == 1].copy()

    if feature_cols is None:
        feature_cols = default_feature_cols(train)

    # Ensure both train/test have all required features (esp. optional flags)
    train = ensure_feature_cols(train, feature_cols, dataset_code)
    test  = ensure_feature_cols(test, feature_cols, dataset_code)

    X_train = train[feature_cols]
    y_train = train[TARGET].values

    X_test = test[feature_cols]
    y_test = test[TARGET].values

    model = build_rf(params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    row = {
        "dataset": dataset_code,
        "group": group,
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "n_features": int(len(feature_cols)),
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": rmse(y_test, pred),
        "MAPE": safe_mape_percent(y_test, pred),
        "params_used": json.dumps(params),
    }
    return row

# ---------------------------
# CONFIG: set your file paths
# ---------------------------

A1_CSV = "A1_causal_demand_keep-imputed_demand-only_168_READY.csv"
A2_CSV = "A2_causal-fill_clean_demand-only_168_READY.csv"
B1_CSV = "B1_interpolation_keep-imputed_demand-only_168_READY.csv"
B2_CSV = "B2_interpolation_clean_demand-only_168_READY.csv"

BEST_DEMAND_PARAMS = "BEST_PARAMS_DEMAND.joblib"

A3_LAGS_CSV = "A3_causal-fill_keep-imputed_telemetry_168_LAGS_READY.csv"
A4_LAGS_CSV = "A4_causal-fill_clean-only_demand_telemetry_168_LAGS_READY.csv"
B3_LAGS_CSV = "B3_interpolation_keep-imputed_telemetry_168_LAGS_READY.csv"
B4_LAGS_CSV = "B4_interpolation_clean-only_demand_telemetry_168_LAGS_READY.csv"

BEST_TELEMETRY_LAGS_ARTIFACT = "BEST_TELEMETRY_LAGS_ARTIFACT.joblib"

runs = [
    ("A1", "demand", A1_CSV, BEST_DEMAND_PARAMS),
    ("A2", "demand", A2_CSV, BEST_DEMAND_PARAMS),
    ("B1", "demand", B1_CSV, BEST_DEMAND_PARAMS),
    ("B2", "demand", B2_CSV, BEST_DEMAND_PARAMS),

    ("A3_LAGS", "telemetry_lags", A3_LAGS_CSV, BEST_TELEMETRY_LAGS_ARTIFACT),
    ("A4_LAGS", "telemetry_lags", A4_LAGS_CSV, BEST_TELEMETRY_LAGS_ARTIFACT),
    ("B3_LAGS", "telemetry_lags", B3_LAGS_CSV, BEST_TELEMETRY_LAGS_ARTIFACT),
    ("B4_LAGS", "telemetry_lags", B4_LAGS_CSV, BEST_TELEMETRY_LAGS_ARTIFACT),
]

rows = []
for ds, grp, csv_path, model_path in runs:
    print(f"Running {ds} | CSV={csv_path} | model={model_path}")
    rows.append(eval_one(ds, grp, csv_path, model_path))

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)
print("Saved:", OUT_CSV)
print(out[["dataset", "group", "RMSE", "MAE", "MAPE", "rows_test", "n_features"]])
