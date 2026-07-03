import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

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
    y_train = train[TARGET].to_numpy()
    X_test  = test.drop(columns=DROP_COLS)
    y_test  = test[TARGET].to_numpy()

    # sign labels: 1 = export (negative), 0 = import (>=0)
    ysign_train = (y_train < 0).astype(int)
    ysign_test  = (y_test < 0).astype(int)

    # Choose which tuned regressor params to use (demand vs telemetry)
    params = (BEST_DEMAND if group == "demand" else BEST_TELEM).copy()
    params["n_jobs"] = -1
    params["random_state"] = 42

    # 1) Classifier for export/import
    clf = RandomForestClassifier(
        n_estimators=2000,
        random_state=42,
        n_jobs=-1,
        max_depth=20,
        min_samples_leaf=2,
        min_samples_split=2,
        max_features="sqrt",
        bootstrap=True,
    )
    clf.fit(X_train, ysign_train)
    ysign_pred = clf.predict(X_test)
    sign_acc = float(accuracy_score(ysign_test, ysign_pred))

    # 2) Two regressors for magnitude
    # import regressor predicts +magnitude
    reg_import = RandomForestRegressor(**params)
    # export regressor predicts magnitude of export (we model abs(y))
    reg_export = RandomForestRegressor(**params)

    import_mask = (ysign_train == 0)
    export_mask = (ysign_train == 1)

    # Safety: if one class is rare, fall back to single-regressor
    if import_mask.sum() < 200 or export_mask.sum() < 200:
        single = RandomForestRegressor(**params)
        single.fit(X_train, y_train)
        pred = single.predict(X_test)
        sign_acc = np.nan
    else:
        reg_import.fit(X_train[import_mask], y_train[import_mask])
        reg_export.fit(X_train[export_mask], np.abs(y_train[export_mask]))

        pred = np.zeros_like(y_test, dtype=float)
        import_pred_mask = (ysign_pred == 0)
        export_pred_mask = (ysign_pred == 1)

        pred[import_pred_mask] = reg_import.predict(X_test[import_pred_mask])
        pred[export_pred_mask] = -np.abs(reg_export.predict(X_test[export_pred_mask]))

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
        "export_classifier_accuracy": sign_acc,
        "regime_model": "RF classifier(sign) + RF regressors(magnitude)",
        "regressor_params_used": str(params),
    })

    print(f"{name} ({group}) -> RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape_v:.2f}% | sign_acc={sign_acc}")

out = pd.DataFrame(rows)
out.to_csv("rf_results_all_8_datasets_168_REGIME_SWITCH.csv", index=False)
print("\nSaved: rf_results_all_8_datasets_168_REGIME_SWITCH.csv")
