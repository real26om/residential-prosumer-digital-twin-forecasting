import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

EXPORT_WEIGHTS = [1, 3, 5, 10]

# Speed control:
# CV uses fewer trees to make the weight search feasible.
# Final training uses full trees from BEST_PARAMS_*.joblib (e.g., 3000).
N_ESTIMATORS_CV = 600
N_SPLITS = 6

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else np.nan

def fit_predict_decomp(X_tr, y_tr, X_va, params, export_weight, n_estimators_override=None):
    # Build decomposition targets
    y_import = np.maximum(y_tr, 0.0)
    y_export = np.maximum(-y_tr, 0.0)

    # Build per-sample weights (export rows get heavier weight)
    sw = np.where(y_tr < 0, float(export_weight), 1.0)

    p = params.copy()
    p["n_jobs"] = -1
    p["random_state"] = 42
    if n_estimators_override is not None:
        p["n_estimators"] = int(n_estimators_override)

    m_import = RandomForestRegressor(**p)
    m_export = RandomForestRegressor(**p)

    m_import.fit(X_tr, y_import, sample_weight=sw)
    m_export.fit(X_tr, y_export, sample_weight=sw)

    pred = m_import.predict(X_va) - m_export.predict(X_va)
    return pred

results = []

for ds_name, csv_path, group in DATASETS:
    df = pd.read_csv(csv_path)

    train = df[df["is_train"] == 1].copy()
    test  = df[df["is_test"] == 1].copy()

    X_train_full = train.drop(columns=DROP_COLS)
    y_train_full = train[TARGET].to_numpy()

    X_test = test.drop(columns=DROP_COLS)
    y_test = test[TARGET].to_numpy()

    params = (BEST_DEMAND if group == "demand" else BEST_TELEM)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    # ---- CV to choose export_weight (NO test used here) ----
    cv_rows = []
    for ew in EXPORT_WEIGHTS:
        fold_rmses = []
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train_full), start=1):
            X_tr = X_train_full.iloc[tr_idx]
            y_tr = y_train_full[tr_idx]
            X_va = X_train_full.iloc[va_idx]
            y_va = y_train_full[va_idx]

            pred_va = fit_predict_decomp(
                X_tr, y_tr, X_va,
                params=params,
                export_weight=ew,
                n_estimators_override=N_ESTIMATORS_CV
            )
            fold_rmses.append(rmse(y_va, pred_va))

        cv_rows.append({
            "export_weight": ew,
            "cv_mean_rmse": float(np.mean(fold_rmses)),
            "cv_std_rmse": float(np.std(fold_rmses)),
        })

    cv_df = pd.DataFrame(cv_rows).sort_values(["cv_mean_rmse", "cv_std_rmse"], ascending=[True, True])
    best_ew = int(cv_df.iloc[0]["export_weight"])

    # ---- Final fit on full train, evaluate once on test ----
    pred_test = fit_predict_decomp(
        X_train_full, y_train_full, X_test,
        params=params,
        export_weight=best_ew,
        n_estimators_override=None  # use full tuned n_estimators (e.g., 3000)
    )

    out_rmse = rmse(y_test, pred_test)
    out_mae = float(mean_absolute_error(y_test, pred_test))
    out_mape = mape(y_test, pred_test)

    results.append({
        "dataset": ds_name,
        "group": group,
        "rows_train": len(train),
        "rows_test": len(test),
        "n_features": X_train_full.shape[1],
        "best_export_weight_by_train_cv": best_ew,
        "train_cv_table": cv_df.to_dict(orient="records"),
        "MAE": out_mae,
        "RMSE": out_rmse,
        "MAPE": out_mape,
        "n_estimators_cv": N_ESTIMATORS_CV,
        "n_splits": N_SPLITS,
        "model": "RF import/export decomposition + tuned export_weight",
        "regressor_params_used": str(dict(params)),
    })

    print(f"{ds_name} -> best_export_weight={best_ew} | TEST RMSE={out_rmse:.2f}")

final = pd.DataFrame(results)
final.to_csv("rf_results_all_8_datasets_168_IMPORT_EXPORT_DECOMP_WEIGHT_TUNED.csv", index=False)
print("\nSaved: rf_results_all_8_datasets_168_IMPORT_EXPORT_DECOMP_WEIGHT_TUNED.csv")
