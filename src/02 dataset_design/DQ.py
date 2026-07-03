import pandas as pd
import numpy as np

INFILE = "A1_causal_demand_keep_imputed_BASE.csv"

OUT_DQ_SUMMARY = "DQ_A1_SUMMARY.csv"
OUT_MISSING_MAIN = "DQ_A1_MISSINGNESS_MAIN_SIGNALS.csv"
OUT_CORR_TOP = "DQ_A1_CORR_WITH_TARGET_TOP.csv"

TRAIN_FRACTION = 0.70
TOP_N = 10
MIN_N_FOR_CORR = 50

def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == "bool":
        return s.fillna(False)
    if s.dtype == "O":
        m = s.astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False})
        return m.fillna(False).astype(bool)
    return s.fillna(0).astype(int).astype(bool)

def spearman_no_scipy(x: pd.Series, y: pd.Series) -> float:
    # Spearman = Pearson correlation of ranks (no SciPy dependency)
    xr = x.rank(method="average")
    yr = y.rank(method="average")
    return float(xr.corr(yr, method="pearson"))

def corr_table(df_sub: pd.DataFrame, subset_name: str, target: str, features: list[str]) -> pd.DataFrame:
    y = pd.to_numeric(df_sub[target], errors="coerce")
    rows = []
    for c in features:
        x = pd.to_numeric(df_sub[c], errors="coerce")
        valid = x.notna() & y.notna()
        n_used = int(valid.sum())
        if n_used < MIN_N_FOR_CORR:
            pearson = np.nan
            spearman = np.nan
        else:
            pearson = float(x[valid].corr(y[valid], method="pearson"))
            spearman = spearman_no_scipy(x[valid], y[valid])
        rows.append({
            "subset": subset_name,
            "feature": c,
            "n_used": n_used,
            "pearson": pearson,
            "spearman": spearman,
            "abs_pearson": np.nan if pd.isna(pearson) else abs(pearson),
        })
    out = pd.DataFrame(rows).sort_values("abs_pearson", ascending=False).drop(columns=["abs_pearson"])
    return out

# -----------------------
# Load + sort
# -----------------------
df = pd.read_csv(INFILE)
df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="raise")
df = df.sort_values("ts_utc").reset_index(drop=True)

n = len(df)
cut = int(round(n * TRAIN_FRACTION))

# -----------------------
# Core series
# -----------------------
demand = pd.to_numeric(df["demand_w"], errors="coerce")
pv = pd.to_numeric(df["pv_w_raw"], errors="coerce")
pool = pd.to_numeric(df["pool_w_raw"], errors="coerce")

# Temperature: sensor-first (meteo counts as imputed)
temp_sensor = pd.to_numeric(df.get("temp_sensor_c"), errors="coerce")
temp_meteo = pd.to_numeric(df.get("temp_meteo_c"), errors="coerce")

flag_temp_sensor_missing = to_bool_series(df["temp_flag_sensor_missing"]) if "temp_flag_sensor_missing" in df.columns else temp_sensor.isna()
flag_temp_used_meteo = to_bool_series(df["temp_flag_used_meteo"]) if "temp_flag_used_meteo" in df.columns else pd.Series(False, index=df.index)

temp_sensor_missing = temp_sensor.isna() | flag_temp_sensor_missing
temp_meteo_missing = temp_meteo.isna()

# “meteo used” = explicitly flagged OR (sensor missing but meteo present)
temp_meteo_used = flag_temp_used_meteo | (temp_sensor_missing & temp_meteo.notna())

# “temp unavailable” = sensor missing AND meteo missing (should be 0 or small)
temp_unavailable = temp_sensor_missing & temp_meteo_missing

# Demand/PV/Pool missing flags from BASE
flag_demand_missing = to_bool_series(df["flag_demand_missing"]) if "flag_demand_missing" in df.columns else pd.Series(False, index=df.index)
flag_demand_imputed = to_bool_series(df["flag_demand_imputed"]) if "flag_demand_imputed" in df.columns else pd.Series(False, index=df.index)
flag_pv_missing = to_bool_series(df["flag_pv_missing"]) if "flag_pv_missing" in df.columns else pd.Series(False, index=df.index)
flag_pool_missing = to_bool_series(df["flag_pool_missing"]) if "flag_pool_missing" in df.columns else pd.Series(False, index=df.index)

orig_demand_missing = demand.isna() | flag_demand_missing | flag_demand_imputed
orig_pv_missing = pv.isna() | flag_pv_missing
orig_pool_missing = pool.isna() | flag_pool_missing

# Clean-only definition (important for “high-quality correlations”)
# Here: treat temp_meteo_used as “imputed”, so it makes the row not-clean.
mask_clean = ~(orig_demand_missing | orig_pv_missing | orig_pool_missing | temp_meteo_used)

# -----------------------
# 1) Concise DQ summary
# -----------------------
ts = df["ts_utc"]
summary = [
    ("rows_total", n),
    ("ts_min", str(ts.min())),
    ("ts_max", str(ts.max())),
    ("ts_duplicates", int(ts.duplicated().sum())),
    ("ts_non_hourly_steps", int((ts.diff().dropna() != pd.Timedelta(hours=1)).sum())),
    ("train_rows_expected_70pct", cut),
    ("test_rows_expected_30pct", n - cut),

    ("clean_rows_count", int(mask_clean.sum())),
    ("clean_rows_pct", float(mask_clean.mean())),

    ("demand_missing_or_imputed_count", int(orig_demand_missing.sum())),
    ("demand_missing_or_imputed_pct", float(orig_demand_missing.mean())),

    ("pv_missing_count", int(orig_pv_missing.sum())),
    ("pv_missing_pct", float(orig_pv_missing.mean())),

    ("pool_missing_count", int(orig_pool_missing.sum())),
    ("pool_missing_pct", float(orig_pool_missing.mean())),

    ("temp_sensor_missing_count", int(temp_sensor_missing.sum())),
    ("temp_sensor_missing_pct", float(temp_sensor_missing.mean())),

    ("temp_meteo_used_count (imputed)", int(temp_meteo_used.sum())),
    ("temp_meteo_used_pct (imputed)", float(temp_meteo_used.mean())),

    ("temp_unavailable_count", int(temp_unavailable.sum())),
    ("temp_unavailable_pct", float(temp_unavailable.mean())),
]
dq_summary = pd.DataFrame(summary, columns=["metric", "value"])
dq_summary.to_csv(OUT_DQ_SUMMARY, index=False)
print("Saved:", OUT_DQ_SUMMARY)

# -----------------------
# 2) Missingness for main signals (compact table)
# -----------------------
missing_main = pd.DataFrame([
    {"signal": "demand_w (missing/imputed)", "missing_count": int(orig_demand_missing.sum()), "missing_pct": float(orig_demand_missing.mean())},
    {"signal": "pv_w_raw (missing)", "missing_count": int(orig_pv_missing.sum()), "missing_pct": float(orig_pv_missing.mean())},
    {"signal": "pool_w_raw (missing)", "missing_count": int(orig_pool_missing.sum()), "missing_pct": float(orig_pool_missing.mean())},
    {"signal": "temp_sensor_c (missing)", "missing_count": int(temp_sensor_missing.sum()), "missing_pct": float(temp_sensor_missing.mean())},
    {"signal": "temp_meteo_used (imputed/substitute)", "missing_count": int(temp_meteo_used.sum()), "missing_pct": float(temp_meteo_used.mean())},
    {"signal": "temp_unavailable (sensor+meteo missing)", "missing_count": int(temp_unavailable.sum()), "missing_pct": float(temp_unavailable.mean())},
    {"signal": "clean_rows (all above OK)", "missing_count": int((~mask_clean).sum()), "missing_pct": float((~mask_clean).mean())},
])
missing_main.to_csv(OUT_MISSING_MAIN, index=False)
print("Saved:", OUT_MISSING_MAIN)

# -----------------------
# 3) Correlation with target (top only)
# -----------------------
target = "y_demand_t_plus_1"
if target not in df.columns:
    raise ValueError(f"Missing target column: {target}")

# Keep features that make sense (avoid pure flags; you can add/remove)
feature_candidates = [
    "demand_w",
    "pv_w_raw",
    "pool_w_raw",
    "temp_sensor_c",
    "temp_meteo_c",
    "outdoor_temp_c",
]

# Keep only features that actually exist in the file
feature_candidates = [c for c in feature_candidates if c in df.columns]

corr_all = corr_table(df, "all_rows", target, feature_candidates)
corr_clean = corr_table(df.loc[mask_clean].copy(), "clean_rows_only", target, feature_candidates)

corr_top = pd.concat([
    corr_all.head(TOP_N),
    corr_clean.head(TOP_N)
], ignore_index=True)

corr_top.to_csv(OUT_CORR_TOP, index=False)
print("Saved:", OUT_CORR_TOP)

print("Done.")
