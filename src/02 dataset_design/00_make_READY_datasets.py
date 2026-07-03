import pandas as pd

FILES = [
("A1", "A1_causal_demand_keep-imputed_demand-only_168.csv"),
("A2", "A2_causal-fill_clean_demand-only_168.csv"),
("A3", "A3_causal-fill_keep-imputed_telemetry_168.csv"),
("A4", "A4_causal-fill_clean-only_demand_telemetry_168.csv"),
("B1", "B1_interpolation_keep-imputed_demand-only_168.csv"),
("B2", "B2_interpolation_clean_demand-only_168.csv"),
("B3", "B3_interpolation_keep-imputed_telemetry_168.csv"),
("B4", "B4_interpolation_clean-only_demand_telemetry_168.csv"),
]

TARGET = "y_demand_t_plus_1"

# Minimal columns required for training + correct split + your new features
REQUIRED_BASE = [
    "ts_utc", "is_train", "is_test", TARGET,
    "month", "is_weekend", "is_holiday_it",
    "demand_lag_1h", "demand_lag_24h", "demand_lag_168h",
]

TELEMETRY_REQUIRED = ["pv_w", "pool_w", "outdoor_temp_c"]

for name, path in FILES:
    df = pd.read_csv(path)

    required = REQUIRED_BASE.copy()
    if name in ["A3", "A4", "B3", "B4"]:
        required += TELEMETRY_REQUIRED

    # Drop rows with NaNs in required columns
    before = len(df)
    df2 = df.dropna(subset=required).copy()

    # Make sure split flags are 0/1 ints
    df2["is_train"] = df2["is_train"].astype(int)
    df2["is_test"] = df2["is_test"].astype(int)

    out = path.replace("_168.csv", "_168_READY.csv")
    df2.to_csv(out, index=False)
    after = len(df2)

    print(f"{name}: {before} -> {after} rows | saved {out}")
