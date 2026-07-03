import pandas as pd

TELEMETRY_FILES = [
    "A3_causal-fill_keep-imputed_telemetry_168.csv",
    "A4_causal-fill_clean-only_demand_telemetry_168.csv",
    "B3_interpolation_keep-imputed_telemetry_168.csv",
    "B4_interpolation_clean-only_demand_telemetry_168.csv",
]

TARGET = "y_demand_t_plus_1"

BASE_REQUIRED = [
    "ts_utc", "is_train", "is_test", TARGET,
    "month", "is_weekend", "is_holiday_it",
    "demand_w", "flag_grid_exporting",
    "demand_lag_1h", "demand_lag_24h", "demand_lag_168h",
    "pv_w", "pool_w", "outdoor_temp_c",
]

LAGS = [1, 24, 168]
TELE_COLS = ["pv_w", "pool_w", "outdoor_temp_c"]

def add_lags(df):
    df = df.copy()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    df = df.sort_values("ts_utc").reset_index(drop=True)

    for col in TELE_COLS:
        for h in LAGS:
            df[f"{col}_lag_{h}h"] = df[col].shift(h)

    return df

for path in TELEMETRY_FILES:
    df = pd.read_csv(path)

    df_lagged = add_lags(df)

    out_lags = path.replace("_168.csv", "_168_LAGS.csv")
    df_lagged.to_csv(out_lags, index=False)
    print("Saved:", out_lags)

    # READY: drop rows with NaNs in required columns + new lag columns
    required = BASE_REQUIRED + [f"{c}_lag_{h}h" for c in TELE_COLS for h in LAGS]
    df_ready = df_lagged.dropna(subset=required).copy()

    df_ready["is_train"] = df_ready["is_train"].astype(int)
    df_ready["is_test"] = df_ready["is_test"].astype(int)

    out_ready = path.replace("_168.csv", "_168_LAGS_READY.csv")
    df_ready.to_csv(out_ready, index=False)
    print("Saved:", out_ready, "| rows:", len(df_ready))
