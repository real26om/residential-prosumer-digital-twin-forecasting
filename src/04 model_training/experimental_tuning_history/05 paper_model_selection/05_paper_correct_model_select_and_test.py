import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.model_selection import TimeSeriesSplit

# IMPORTANT: enable halving BEFORE importing HalvingGridSearchCV [web:345][web:348]
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# ------------------------
# Global config
# ------------------------
TARGET = "y_demand_t_plus_1"
DROP = ["ts_utc", "is_train", "is_test", TARGET]

N_SPLITS_CV = 6
VAL_FRACTION = 0.15  # last 15% of is_train by time
RANDOM_STATE = 42

OUT_VAL_ALL = "paper_select_val_scores.csv"
OUT_VAL_BEST = "paper_best_method_per_dataset.csv"
OUT_TEST_FINAL = "paper_final_test_scores.csv"

# These flags exist in some LAGS datasets but not all; if absent treat as 0
OPTIONAL_ZERO_COLS = {
    "flag_demand_imputed_new",
    "flag_pool_imputed_new",
    "flag_pv_imputed_new",
    "flag_temp_imputed_new",
}

def first_existing(*candidates):
    for p in candidates:
        if p and Path(p).exists():
            return p
    raise FileNotFoundError("None of these files exist:\n" + "\n".join([c for c in candidates if c]))

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

rmse_scorer = make_scorer(lambda yt, yp: -rmse(yt, yp))  # higher is better (negative RMSE)

def safe_mape_percent(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)

def default_feature_cols(df):
    cols = [c for c in df.columns if c not in DROP]
    return sorted(cols)

def ensure_cols(df, feature_cols):
    missing = [c for c in feature_cols if c not in df.columns]
    missing_optional = [c for c in missing if c in OPTIONAL_ZERO_COLS]
    missing_hard = [c for c in missing if c not in OPTIONAL_ZERO_COLS]

    for c in missing_optional:
        df[c] = 0

    if missing_hard:
        raise ValueError(f"Missing required columns: {missing_hard}")

    return df

def time_order(df):
    if "ts_utc" in df.columns:
        return df.sort_values("ts_utc").reset_index(drop=True)
    return df.reset_index(drop=True)

def split_train_val(train_df, val_fraction=VAL_FRACTION):
    train_df = time_order(train_df)
    n = len(train_df)
    n_val = max(1, int(round(n * val_fraction)))
    cut = n - n_val
    inner_train = train_df.iloc[:cut].copy()
    inner_val = train_df.iloc[cut:].copy()
    return inner_train, inner_val

# ---------
# Target transform helpers
# ---------
def y_transform_signed_log1p(y):
    y = np.asarray(y, dtype=float)
    return np.sign(y) * np.log1p(np.abs(y))

def y_inverse_signed_log1p(z):
    z = np.asarray(z, dtype=float)
    return np.sign(z) * np.expm1(np.abs(z))

# ---------
# Tuning core (HalvingGridSearchCV)
# ---------
def tune_rf_regressor(X, y):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)

    base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    param_grid = {
        "max_depth": [10, 20, 40, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", 0.3, 0.5, 0.8],
        "bootstrap": [True, False],
    }

    search = HalvingGridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring=rmse_scorer,
        cv=tscv,
        factor=3,
        resource="n_estimators",
        min_resources=200,
        max_resources=1800,
        n_jobs=-1,
        verbose=0,
        refit=True
    )
    search.fit(X, y)
    return search.best_estimator_

# ---------
# Candidate methods (all scored on validation only)
# ---------
def method_rf_baseline(train_df, val_df, feature_cols):
    train_df = ensure_cols(train_df, feature_cols)
    val_df = ensure_cols(val_df, feature_cols)

    Xtr = train_df[feature_cols]
    ytr = train_df[TARGET].values
    Xva = val_df[feature_cols]
    yva = val_df[TARGET].values

    model = tune_rf_regressor(Xtr, ytr)
    pred = model.predict(Xva)
    return model, pred, yva

def method_target_transform_rf(train_df, val_df, feature_cols):
    train_df = ensure_cols(train_df, feature_cols)
    val_df = ensure_cols(val_df, feature_cols)

    Xtr = train_df[feature_cols]
    ytr = y_transform_signed_log1p(train_df[TARGET].values)
    Xva = val_df[feature_cols]
    yva = val_df[TARGET].values

    model = tune_rf_regressor(Xtr, ytr)
    pred = y_inverse_signed_log1p(model.predict(Xva))
    return model, pred, yva

def method_regime_switch(train_df, val_df, feature_cols):
    """
    Predict sign with classifier, magnitude with regressors.
    """
    train_df = ensure_cols(train_df, feature_cols)
    val_df = ensure_cols(val_df, feature_cols)

    Xtr = train_df[feature_cols]
    ytr = train_df[TARGET].values
    Xva = val_df[feature_cols]
    yva = val_df[TARGET].values

    is_export_tr = (ytr < 0).astype(int)

    clf = RandomForestClassifier(
        n_estimators=800,
        max_depth=10,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ).fit(Xtr, is_export_tr)

    ytr_mag = np.abs(ytr)
    mask_exp = is_export_tr == 1
    mask_imp = is_export_tr == 0

    # If one regime is too rare, fallback to one magnitude regressor
    if mask_imp.sum() < 50 or mask_exp.sum() < 50:
        reg = tune_rf_regressor(Xtr, ytr_mag)
        pred_mag = reg.predict(Xva)
        pred_sign = clf.predict(Xva)
        pred = np.where(pred_sign == 1, -pred_mag, pred_mag)
        return {"clf": clf, "reg_all": reg}, pred, yva

    reg_imp = tune_rf_regressor(Xtr[mask_imp], ytr_mag[mask_imp])
    reg_exp = tune_rf_regressor(Xtr[mask_exp], ytr_mag[mask_exp])

    pred_regime = clf.predict(Xva)
    pred_mag = np.where(pred_regime == 1, reg_exp.predict(Xva), reg_imp.predict(Xva))
    pred = np.where(pred_regime == 1, -pred_mag, pred_mag)

    return {"clf": clf, "reg_imp": reg_imp, "reg_exp": reg_exp}, pred, yva

BASE_METHODS = {
    "rf": method_rf_baseline,
    "target_transform_rf": method_target_transform_rf,
    "regime_switch": method_regime_switch,
}

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    train = df[df["is_train"] == 1].copy()
    test = df[df["is_test"] == 1].copy()
    return df, train, test

def run_one_dataset(dataset_code, base_csv, lags_csv=None):
    # Load base dataset
    _, train_full, test = load_dataset(base_csv)
    train_inner, val = split_train_val(train_full, VAL_FRACTION)
    feature_cols = default_feature_cols(train_full)

    candidates = []

    # base methods on base_csv
    for method_name, fn in BASE_METHODS.items():
        model, pred_val, y_val = fn(train_inner, val, feature_cols)
        candidates.append({
            "dataset": dataset_code,
            "method": method_name,
            "train_csv": base_csv,
            "val_RMSE": rmse(y_val, pred_val),
            "val_MAE": float(mean_absolute_error(y_val, pred_val)),
            "val_MAPE": safe_mape_percent(y_val, pred_val),
            "n_features": int(len(feature_cols)),
        })

    # telemetry-lags candidate (RF baseline on lags_csv) if provided
    if lags_csv is not None:
        _, train_full_l, _ = load_dataset(lags_csv)
        train_inner_l, val_l = split_train_val(train_full_l, VAL_FRACTION)
        feat_l = default_feature_cols(train_full_l)

        model_l, pred_val_l, y_val_l = method_rf_baseline(train_inner_l, val_l, feat_l)
        candidates.append({
            "dataset": dataset_code,
            "method": "telemetry_lags_rf",
            "train_csv": lags_csv,
            "val_RMSE": rmse(y_val_l, pred_val_l),
            "val_MAE": float(mean_absolute_error(y_val_l, pred_val_l)),
            "val_MAPE": safe_mape_percent(y_val_l, pred_val_l),
            "n_features": int(len(feat_l)),
        })

    # Choose by validation RMSE
    best = sorted(candidates, key=lambda r: r["val_RMSE"])[0]

    # Final refit on FULL train using the chosen train_csv, then evaluate on TEST
    chosen_csv = best["train_csv"]
    df_final = pd.read_csv(chosen_csv)
    train_final = df_final[df_final["is_train"] == 1].copy()
    test_final = df_final[df_final["is_test"] == 1].copy()
    feat_final = default_feature_cols(train_final)

    train_final = ensure_cols(train_final, feat_final)
    test_final = ensure_cols(test_final, feat_final)

    Xtr = train_final[feat_final]
    ytr = train_final[TARGET].values
    Xte = test_final[feat_final]
    yte = test_final[TARGET].values

    if best["method"] == "rf" or best["method"] == "telemetry_lags_rf":
        final_model = tune_rf_regressor(Xtr, ytr)
        pred_te = final_model.predict(Xte)

    elif best["method"] == "target_transform_rf":
        final_model = tune_rf_regressor(Xtr, y_transform_signed_log1p(ytr))
        pred_te = y_inverse_signed_log1p(final_model.predict(Xte))

    elif best["method"] == "regime_switch":
        # Train regime switch on full train and predict test
        is_export_tr = (ytr < 0).astype(int)
        clf = RandomForestClassifier(
            n_estimators=800,
            max_depth=10,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ).fit(Xtr, is_export_tr)

        ytr_mag = np.abs(ytr)
        mask_exp = is_export_tr == 1
        mask_imp = is_export_tr == 0

        if mask_imp.sum() < 50 or mask_exp.sum() < 50:
            reg = tune_rf_regressor(Xtr, ytr_mag)
            pred_mag = reg.predict(Xte)
            pred_sign = clf.predict(Xte)
            pred_te = np.where(pred_sign == 1, -pred_mag, pred_mag)
            final_model = {"clf": clf, "reg_all": reg}
        else:
            reg_imp = tune_rf_regressor(Xtr[mask_imp], ytr_mag[mask_imp])
            reg_exp = tune_rf_regressor(Xtr[mask_exp], ytr_mag[mask_exp])
            pred_regime = clf.predict(Xte)
            pred_mag = np.where(pred_regime == 1, reg_exp.predict(Xte), reg_imp.predict(Xte))
            pred_te = np.where(pred_regime == 1, -pred_mag, pred_mag)
            final_model = {"clf": clf, "reg_imp": reg_imp, "reg_exp": reg_exp}

    else:
        raise ValueError("Unknown method: " + best["method"])

    final_row = {
        "dataset": dataset_code,
        "chosen_method": best["method"],
        "chosen_train_csv": chosen_csv,
        "n_features": int(len(feat_final)),
        "test_RMSE": rmse(yte, pred_te),
        "test_MAE": float(mean_absolute_error(yte, pred_te)),
        "test_MAPE": safe_mape_percent(yte, pred_te),
    }

    # Save per-dataset artifact
    joblib.dump(
        {"dataset": dataset_code, "method": best["method"], "train_csv": chosen_csv,
         "feature_cols": feat_final, "model": final_model},
        f"BEST_MODEL_{dataset_code}.joblib"
    )

    return candidates, best, final_row

# ------------------------
# Dataset file mapping (robust to READY naming)
# ------------------------
DATASETS = {
    "A1": {
        "base": first_existing(
            "A1_causal_demand_keep-imputed_demand-only_168_READY.csv",
            "A1_causal_demand_keep-imputed_demand-only_168.csv",
        ),
        "lags": None
    },
    "A2": {
        "base": first_existing(
            "A2_causal-fill_clean_demand-only_168_READY.csv",
            "A2_causal-fill_clean_demand-only_168.csv",
        ),
        "lags": None
    },
    "B1": {
        "base": first_existing(
            "B1_interpolation_keep-imputed_demand-only_168_READY.csv",
            "B1_interpolation_keep-imputed_demand-only_168.csv",
        ),
        "lags": None
    },
    "B2": {
        "base": first_existing(
            "B2_interpolation_clean_demand-only_168_READY.csv",
            "B2_interpolation_clean_demand-only_168.csv",
        ),
        "lags": None
    },

    "A3": {
        "base": first_existing(
            "A3_causal-fill_keep-imputed_telemetry_168_READY.csv",
            "A3_causal-fill_keep-imputed_telemetry_168.csv",
        ),
        "lags": first_existing(
            "A3_causal-fill_keep-imputed_telemetry_168_LAGS_READY.csv",
        ),
    },
    "A4": {
        "base": first_existing(
            "A4_causal-fill_clean-only_demand_telemetry_168_READY.csv",
            "A4_causal-fill_clean-only_demand_telemetry_168.csv",
        ),
        "lags": first_existing(
            "A4_causal-fill_clean-only_demand_telemetry_168_LAGS_READY.csv",
        ),
    },
    "B3": {
        "base": first_existing(
            "B3_interpolation_keep-imputed_telemetry_168_READY.csv",
            "B3_interpolation_keep-imputed_telemetry_168.csv",
        ),
        "lags": first_existing(
            "B3_interpolation_keep-imputed_telemetry_168_LAGS_READY.csv",
        ),
    },
    "B4": {
        "base": first_existing(
            "B4_interpolation_clean-only_demand_telemetry_168_READY.csv",
            "B4_interpolation_clean-only_demand_telemetry_168.csv",
        ),
        "lags": first_existing(
            "B4_interpolation_clean-only_demand_telemetry_168_LAGS_READY.csv",
        ),
    },
}

# ------------------------
# Run all datasets
# ------------------------
val_rows = []
best_rows = []
test_rows = []

for ds, paths in DATASETS.items():
    print("=== Running dataset:", ds, "===")
    candidates, best, final_row = run_one_dataset(ds, paths["base"], paths["lags"])
    val_rows.extend(candidates)
    best_rows.append(best)
    test_rows.append(final_row)

pd.DataFrame(val_rows).to_csv(OUT_VAL_ALL, index=False)
pd.DataFrame(best_rows).to_csv(OUT_VAL_BEST, index=False)
pd.DataFrame(test_rows).to_csv(OUT_TEST_FINAL, index=False)

print("Saved:", OUT_VAL_ALL)
print("Saved:", OUT_VAL_BEST)
print("Saved:", OUT_TEST_FINAL)
print(pd.DataFrame(test_rows)[["dataset", "chosen_method", "test_RMSE", "test_MAE", "test_MAPE", "n_features"]])
