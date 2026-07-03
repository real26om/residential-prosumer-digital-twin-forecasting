import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error

TARGET = "y_demand_t_plus_1"
DROP_COLS = ["ts_utc", "is_train", "is_test", TARGET]

DATASETS = [
    ("A1", "A1_causal_demand_keep-imputed_demand-only_168_READY.csv", "demand"),
    ("A2", "A2_causal-fill_clean_demand-only_168_READY.csv", "demand"),
    ("B1", "B1_interpolation_keep-imputed_demand-only_168_READY.csv", "demand"),
    ("B2", "B2_interpolation_clean_demand-only_168_READY.csv", "demand"),
    ("A3", "A3_causal-fill_keep-imputed_telemetry_168_READY.csv", "telemetry"),
    ("A4", "A4_causal-fill_clean-only_demand_telemetry_168_READY.csv", "telemetry"),
    ("B3", "B3_interpolation_keep-imputed_telemetry_168_READY.csv", "telemetry"),
    ("B4", "B4_interpolation_clean-only_demand_telemetry_168_READY.csv", "telemetry"),
]

BEST_DEMAND = joblib.load("BEST_PARAMS_DEMAND.joblib")
BEST_TELEM  = joblib.load("BEST_PARAMS_TELEMETRY.joblib")

def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else np.nan

rows = []

for name, path, group in DATASETS:
    df = pd.read_csv(path)
    train = df[df["is_train"] == 1].copy()
    test  = df[df["is_test"] == 1].copy()

    X_train = train.drop(columns=DROP_COLS)
    y_train = train[TARGET]
    X_test  = test.drop(columns=DROP_COLS)
    y_test  = test[TARGET]

    params = (BEST_DEMAND if group == "demand" else BEST_TELEM).copy()
    params["n_jobs"] = -1
    params["random_state"] = 42

    rf = RandomForestRegressor(**params)

    # Transform target to handle negative/zero values
    pt = PowerTransformer(method="yeo-johnson", standardize=True)

    model = TransformedTargetRegressor(
        regressor=rf,
        transformer=pt
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    mape_v = mape(y_test, pred)

    rows.append({
        "dataset": name,
        "group": group,
        "rows_train": len(train),
        "rows_test": len(test),
        "n_features": X_train.shape[1],
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape_v,
        "params_used": str(params),
        "target_transform": "PowerTransformer(yeo-johnson, standardize=True)"
    })

    print(f"{name} ({group}) -> RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape_v:.2f}%")

out = pd.DataFrame(rows)
out.to_csv("rf_results_all_8_datasets_168_TARGET_TRANSFORM.csv", index=False)
print("\nSaved: rf_results_all_8_datasets_168_TARGET_TRANSFORM.csv")
