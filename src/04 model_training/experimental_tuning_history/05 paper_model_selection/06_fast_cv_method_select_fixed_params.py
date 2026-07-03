import time
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ------------------------
# Config
# ------------------------
TARGET = "y_demand_t_plus_1"
DROP = ["ts_utc", "is_train", "is_test", TARGET]

N_SPLITS = 5          # fast but still solid; set 6 if you want
GAP = 0               # keep 0 unless you intentionally want a gap [web:37]
RANDOM_STATE = 42

OUT_CV = "fast_cv_method_scores.csv"
OUT_BEST = "fast_cv_best_method_per_dataset.csv"
OUT_TEST = "fast_cv_final_test_scores.csv"

OPTIONAL_ZERO_COLS = {
    "flag_demand_imputed_new",
    "flag_pool_imputed_new",
    "flag_pv_imputed_new",
    "flag_temp_imputed_new",
}

# Fixed “strong default” RF params (no halving/grid search)
RF_REG_PARAMS = dict(
    n_estimators=800,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    n_jobs=-1,
    random_state=RANDOM_STATE
)

RF_CLF_PARAMS = dict(
    n_estimators=600,
    max_depth=10,
    min_samples_leaf=2,
    max_features="sqrt",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

def first_existing(*candidates):
    for p in candidates:
        if p and Path(p).exists():
            return p
    raise FileNotFoundError("None exist:\n" + "\n".join([c for c in candidates if c]))

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def safe_mape_percent(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)

def y_transform_signed_log1p(y):
    y = np.asarray(y, dtype=float)
    return np.sign(y) * np.log1p(np.abs(y))

def y_inverse_signed_log1p(z):
    z = np.asarray(z, dtype=float)
    return np.sign(z) * np.expm1(np.abs(z))

def default_feature_cols(df):
    return sorted([c for c in df.columns if c not in DROP])

def ensure_cols(df, feature_cols):
    missing = [c for c in feature_cols if c not in df.columns]
    for c in missing:
        if c in OPTIONAL_ZERO_COLS:
            df[c] = 0
        else:
            raise ValueError(f"Missing required column: {c}")
    return df

def time_order(df):
    if "ts_utc" in df.columns:
        return df.sort_values("ts_utc").reset_index(drop=True)
    return df.reset_index(drop=True)

# ------------------------
# Methods (fixed params)
# ------------------------
def fit_predict_rf(train_df, test_df, feature_cols):
    model = RandomForestRegressor(**RF_REG_PARAMS)
    model.fit(train_df[feature_cols], train_df[TARGET].values)
    pred = model.predict(test_df[feature_cols])
    return model, pred

def fit_predict_target_transform_rf(train_df, test_df, feature_cols):
    model = RandomForestRegressor(**RF_REG_PARAMS)
    ytr = y_transform_signed_log1p(train_df[TARGET].values)
    model.fit(train_df[feature_cols], ytr)
    pred_z = model.predict(test_df[feature_cols])
    pred = y_inverse_signed_log1p(pred_z)
    return model, pred

def fit_predict_regime_switch(train_df, test_df, feature_cols):
    Xtr = train_df[feature_cols]
    ytr = train_df[TARGET].values
    Xte = test_df[feature_cols]

    is_export_tr = (ytr < 0).astype(int)
    clf = RandomForestClassifier(**RF_CLF_PARAMS)
    clf.fit(Xtr, is_export_tr)

    ytr_mag = np.abs(ytr)
    mask_exp = is_export_tr == 1
    mask_imp = is_export_tr == 0

    # Magnitude regressors (fixed params)
    if mask_imp.sum() < 50 or mask_exp.sum() < 50:
        reg = RandomForestRegressor(**RF_REG_PARAMS)
        reg.fit(Xtr, ytr_mag)
        pred_mag = reg.predict(Xte)
        pred_sign = clf.predict(Xte)
        pred = np.where(pred_sign == 1, -pred_mag, pred_mag)
        return {"clf": clf, "reg_all": reg}, pred

    reg_imp = RandomForestRegressor(**RF_REG_PARAMS)
    reg_exp = RandomForestRegressor(**RF_REG_PARAMS)
    reg_imp.fit(Xtr[mask_imp], ytr_mag[mask_imp])
    reg_exp.fit(Xtr[mask_exp], ytr_mag[mask_exp])

    pred_regime = clf.predict(Xte)
    pred_mag = np.where(pred_regime == 1, reg_exp.predict(Xte), reg_imp.predict(Xte))
    pred = np.where(pred_regime == 1, -pred_mag, pred_mag)
    return {"clf": clf, "reg_imp": reg_imp, "reg_exp": reg_exp}, pred

# ------------------------
# CV evaluation
# ------------------------
def cv_score_method(df_train, feature_cols, method_name):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=GAP)  # time-safe splits [web:37]
    rmses = []
    maes = []

    df_train = time_order(df_train)

    X = df_train[feature_cols]
    y = df_train[TARGET].values

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), start=1):
        tr = df_train.iloc[tr_idx].copy()
        va = df_train.iloc[va_idx].copy()

        if method_name == "rf":
            _, pred = fit_predict_rf(tr, va, feature_cols)
        elif method_name == "target_transform_rf":
            _, pred = fit_predict_target_transform_rf(tr, va, feature_cols)
        elif method_name == "regime_switch":
            _, pred = fit_predict_regime_switch(tr, va, feature_cols)
        else:
            raise ValueError(method_name)

        rmses.append(rmse(va[TARGET].values, pred))
        maes.append(float(mean_absolute_error(va[TARGET].values, pred)))

    return float(np.mean(rmses)), float(np.mean(maes))

def run_dataset(dataset_code, base_csv, lags_csv=None):
    df = pd.read_csv(base_csv)
    df_train = df[df["is_train"] == 1].copy()
    df_test = df[df["is_test"] == 1].copy()

    df_train = time_order(df_train)
    df_test = time_order(df_test)

    feature_cols = default_feature_cols(df_train)

    # ensure consistent columns
    df_train = ensure_cols(df_train, feature_cols)
    df_test = ensure_cols(df_test, feature_cols)

    methods = ["rf", "target_transform_rf", "regime_switch"]

    rows = []
    for m in methods:
        cv_rmse, cv_mae = cv_score_method(df_train, feature_cols, m)
        rows.append({
            "dataset": dataset_code,
            "method": m,
            "train_csv": base_csv,
            "cv_RMSE_mean": cv_rmse,
            "cv_MAE_mean": cv_mae,
            "n_features": len(feature_cols),
        })

    # telemetry lags candidate (RF only, fixed params)
    if lags_csv is not None:
        dfl = pd.read_csv(lags_csv)
        trl = dfl[dfl["is_train"] == 1].copy()
        trl = time_order(trl)
        feat_l = default_feature_cols(trl)
        trl = ensure_cols(trl, feat_l)
        cv_rmse, cv_mae = cv_score_method(trl, feat_l, "rf")
        rows.append({
            "dataset": dataset_code,
            "method": "telemetry_lags_rf",
            "train_csv": lags_csv,
            "cv_RMSE_mean": cv_rmse,
            "cv_MAE_mean": cv_mae,
            "n_features": len(feat_l),
        })

    # pick best by CV RMSE
    best = sorted(rows, key=lambda r: r["cv_RMSE_mean"])[0]

    # final fit on full train (chosen csv) -> test
    dfc = pd.read_csv(best["train_csv"])
    tr = dfc[dfc["is_train"] == 1].copy()
    te = dfc[dfc["is_test"] == 1].copy()
    tr = time_order(tr)
    te = time_order(te)
    feat = default_feature_cols(tr)
    tr = ensure_cols(tr, feat)
    te = ensure_cols(te, feat)

    if best["method"] in ("rf", "telemetry_lags_rf"):
        model, pred_te = fit_predict_rf(tr, te, feat)
    elif best["method"] == "target_transform_rf":
        model, pred_te = fit_predict_target_transform_rf(tr, te, feat)
    elif best["method"] == "regime_switch":
        model, pred_te = fit_predict_regime_switch(tr, te, feat)
    else:
        raise ValueError(best["method"])

    yte = te[TARGET].values
    test_row = {
        "dataset": dataset_code,
        "chosen_method": best["method"],
        "chosen_train_csv": best["train_csv"],
        "n_features": len(feat),
        "test_RMSE": rmse(yte, pred_te),
        "test_MAE": float(mean_absolute_error(yte, pred_te)),
        "test_MAPE": safe_mape_percent(yte, pred_te),
    }

    joblib.dump(
        {"dataset": dataset_code, "method": best["method"], "train_csv": best["train_csv"],
         "feature_cols": feat, "model": model},
        f"FAST_BEST_MODEL_{dataset_code}.joblib"
    )

    return rows, best, test_row

# ------------------------
# Map your dataset files
# ------------------------
DATASETS = {
    "A1": {"base": first_existing("A1_causal_demand_keep-imputed_demand-only_168_READY.csv", "A1_causal_demand_keep-imputed_demand-only_168.csv"), "lags": None},
    "A2": {"base": first_existing("A2_causal-fill_clean_demand-only_168_READY.csv", "A2_causal-fill_clean_demand-only_168.csv"), "lags": None},
    "B1": {"base": first_existing("B1_interpolation_keep-imputed_demand-only_168_READY.csv", "B1_interpolation_keep-imputed_demand-only_168.csv"), "lags": None},
    "B2": {"base": first_existing("B2_interpolation_clean_demand-only_168_READY.csv", "B2_interpolation_clean_demand-only_168.csv"), "lags": None},

    "A3": {"base": first_existing("A3_causal-fill_keep-imputed_telemetry_168_READY.csv", "A3_causal-fill_keep-imputed_telemetry_168.csv"),
           "lags": first_existing("A3_causal-fill_keep-imputed_telemetry_168_LAGS_READY.csv")},
    "A4": {"base": first_existing("A4_causal-fill_clean-only_demand_telemetry_168_READY.csv", "A4_causal-fill_clean-only_demand_telemetry_168.csv"),
           "lags": first_existing("A4_causal-fill_clean-only_demand_telemetry_168_LAGS_READY.csv")},
    "B3": {"base": first_existing("B3_interpolation_keep-imputed_telemetry_168_READY.csv", "B3_interpolation_keep-imputed_telemetry_168.csv"),
           "lags": first_existing("B3_interpolation_keep-imputed_telemetry_168_LAGS_READY.csv")},
    "B4": {"base": first_existing("B4_interpolation_clean-only_demand_telemetry_168_READY.csv", "B4_interpolation_clean-only_demand_telemetry_168.csv"),
           "lags": first_existing("B4_interpolation_clean-only_demand_telemetry_168_LAGS_READY.csv")},
}

# ------------------------
# Run
# ------------------------
t0 = time.perf_counter()
all_rows = []
best_rows = []
test_rows = []

for ds, p in DATASETS.items():
    print(f"=== {ds} ===", flush=True)
    rows, best, test_row = run_dataset(ds, p["base"], p["lags"])
    all_rows.extend(rows)
    best_rows.append(best)
    test_rows.append(test_row)

pd.DataFrame(all_rows).to_csv(OUT_CV, index=False)
pd.DataFrame(best_rows).to_csv(OUT_BEST, index=False)
pd.DataFrame(test_rows).to_csv(OUT_TEST, index=False)

dt = time.perf_counter() - t0
print(f"DONE in {dt/60:.1f} minutes", flush=True)
print(pd.DataFrame(test_rows)[["dataset","chosen_method","test_RMSE","test_MAE","test_MAPE","n_features"]])
