# validate_prompt1_master.py
# Prompt 1 validator for master_hourly_extended.csv
#
# Usage (PowerShell):
#   python .\validate_prompt1_master.py --csv processed_hourly\master_hourly_extended.csv --config config.yml
#
# Exits with code 0 on PASS, 1 on FAIL.

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# These are from your existing Prompt 0 plumbing
from config_loader import load_frozen_config, cfg_get
from asserts import enforce_frozen_runtime_asserts


# -----------------------------
# Deterministic IT holiday logic
# -----------------------------
def _easter_date_gregorian(year: int) -> pd.Timestamp:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year=year, month=month, day=day)


def italian_public_holidays_utc_dates(year: int) -> set[pd.Timestamp]:
    fixed = [
        (1, 1),
        (1, 6),
        (4, 25),
        (5, 1),
        (6, 2),
        (8, 15),
        (11, 1),
        (12, 8),
        (12, 25),
        (12, 26),
    ]
    easter = _easter_date_gregorian(year)
    easter_monday = easter + pd.Timedelta(days=1)

    out: set[pd.Timestamp] = set()
    for m, d in fixed:
        out.add(pd.Timestamp(year=year, month=m, day=d, tz="UTC"))
    out.add(pd.Timestamp(easter_monday.year, easter_monday.month, easter_monday.day, tz="UTC"))
    return out


def compute_is_holiday_it(ts_utc: pd.DatetimeIndex) -> np.ndarray:
    years = np.unique(ts_utc.year)
    holiday_dates = set()
    for y in years:
        holiday_dates |= italian_public_holidays_utc_dates(int(y))
    dates = ts_utc.normalize()
    return np.array([1 if d in holiday_dates else 0 for d in dates], dtype=int)


# -----------------------------
# Helpers
# -----------------------------
def _as_int01(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy()
    if np.isnan(arr).any():
        raise ValueError("Found NaN in a flag column expected to be 0/1.")
    arr_i = arr.astype(int)
    if not np.all((arr_i == 0) | (arr_i == 1)):
        bad = np.unique(arr_i[~((arr_i == 0) | (arr_i == 1))])
        raise ValueError(f"Flag column contains values outside {{0,1}}: {bad[:10]}")
    return arr_i


def _assert_allclose(a: np.ndarray, b: np.ndarray, name: str, failures: list[str], atol: float = 1e-9) -> None:
    a2 = np.array(a, dtype=float)
    b2 = np.array(b, dtype=float)

    # handle NaNs: must match positions
    nan_mismatch = np.isnan(a2) ^ np.isnan(b2)
    if nan_mismatch.any():
        idx = int(np.where(nan_mismatch)[0][0])
        failures.append(f"{name}: NaN pattern mismatch at row {idx}")
        return

    mask = ~np.isnan(a2)
    if not np.allclose(a2[mask], b2[mask], atol=atol, rtol=0.0):
        idx = int(np.where(np.abs(a2[mask] - b2[mask]) > atol)[0][0])
        failures.append(f"{name}: values differ (first mismatch at masked index {idx})")


# -----------------------------
# Main validation
# -----------------------------
def validate(csv_path: Path, cfg_path: Path) -> None:
    failures: list[str] = []

    # Load frozen config and enforce preregistered asserts
    cfg = load_frozen_config(cfg_path)
    enforce_frozen_runtime_asserts(cfg)

    night_thr = float(cfg_get(cfg, "config.solar.night_altitude_threshold_deg"))

    df = pd.read_csv(csv_path)

    # Required columns (Prompt 1)
    required = [
        "ts_utc",
        "demand_w_obs", "demand_count_samples", "demand_coverage_minutes",
        "pv_w_obs", "pv_count_samples", "pv_coverage_minutes",
        "pool_w_obs", "pool_count_samples", "pool_coverage_minutes",
        "pv_count_samples_clipped_noise", "flag_pv_clipped_noise_hour",
        "pv_count_samples_forced_night_zero", "flag_pv_forced_night_zero_hour",
        "sun_altitude_hour_mid_deg", "is_daylight_hour",
        "temp_sensor_c", "temp_sensor_count_samples", "temp_sensor_coverage_minutes",
        "temp_meteo_c", "outdoor_temp_c_effective", "outdoor_temp_c",
        "flag_demand_missing_raw", "flag_pv_missing_raw", "flag_pool_missing_raw",
        "flag_temp_sensor_missing_raw", "flag_temp_meteo_missing_raw",
        "flag_temp_sensor_missing", "flag_temp_used_meteo",
        "hour", "day_of_week", "month", "is_weekend", "is_holiday_it", "hour_sin", "hour_cos",
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        failures.append(f"Missing required columns: {missing_cols}")

    # Forbidden columns (Prompt 1: do not create imputed series yet)
    forbidden = [c for c in ["demand_w", "pv_w", "pool_w"] if c in df.columns]
    if forbidden:
        failures.append(f"Forbidden columns present (Prompt 1 violation): {forbidden}")

    if failures:
        raise AssertionError("\n".join(failures))

    # Parse ts_utc as tz-aware UTC
    ts = pd.to_datetime(df["ts_utc"], utc=True, errors="raise")
    if ts.dt.tz is None or str(ts.dt.tz) != "UTC":
        failures.append("ts_utc must be tz-aware UTC.")

    # Ensure sorted, unique, continuous hourly
    if ts.duplicated().any():
        failures.append("ts_utc contains duplicate timestamps.")
    ts_sorted = ts.sort_values()
    if not ts_sorted.is_monotonic_increasing:
        failures.append("ts_utc is not monotonic increasing after sorting (unexpected).")
    diffs = ts_sorted.diff().dropna()
    if not (diffs == pd.Timedelta(hours=1)).all():
        failures.append("ts_utc is not a continuous 1-hour backbone (found non-1h step).")

    # Coverage sanity (0..60)
    for cov_col in ["demand_coverage_minutes", "pv_coverage_minutes", "pool_coverage_minutes", "temp_sensor_coverage_minutes"]:
        cov = pd.to_numeric(df[cov_col], errors="coerce")
        if cov.isna().any():
            failures.append(f"{cov_col} contains NaN.")
        if (cov < 0).any() or (cov > 60.0).any():
            failures.append(f"{cov_col} must be within [0,60].")

    # Counts sanity (>=0 int)
    for cnt_col in ["demand_count_samples", "pv_count_samples", "pool_count_samples", "temp_sensor_count_samples",
                    "pv_count_samples_clipped_noise", "pv_count_samples_forced_night_zero"]:
        cnt = pd.to_numeric(df[cnt_col], errors="coerce")
        if cnt.isna().any():
            failures.append(f"{cnt_col} contains NaN.")
        if (cnt < 0).any():
            failures.append(f"{cnt_col} contains negative values.")

    # Raw missing flags must match count==0
    def check_missing(flag_col: str, count_col: str) -> None:
        flag = _as_int01(df[flag_col])
        cnt = pd.to_numeric(df[count_col], errors="coerce").fillna(-999).astype(int).to_numpy()
        expected = (cnt == 0).astype(int)
        if not np.array_equal(flag, expected):
            idx = int(np.where(flag != expected)[0][0])
            failures.append(f"{flag_col} mismatch vs {count_col}==0 at row {idx}")

    check_missing("flag_demand_missing_raw", "demand_count_samples")
    check_missing("flag_pv_missing_raw", "pv_count_samples")
    check_missing("flag_pool_missing_raw", "pool_count_samples")
    check_missing("flag_temp_sensor_missing_raw", "temp_sensor_count_samples")

    # Temp meteo missing should match temp_meteo_c isna (since no meteo count column here)
    meteo_missing = _as_int01(df["flag_temp_meteo_missing_raw"])
    meteo_isna = df["temp_meteo_c"].isna().astype(int).to_numpy()
    if not np.array_equal(meteo_missing, meteo_isna):
        idx = int(np.where(meteo_missing != meteo_isna)[0][0])
        failures.append(f"flag_temp_meteo_missing_raw mismatch vs temp_meteo_c.isna() at row {idx}")

    # Temperature provenance flags
    temp_sensor_missing = _as_int01(df["flag_temp_sensor_missing"])
    temp_sensor_missing_raw = _as_int01(df["flag_temp_sensor_missing_raw"])
    if not np.array_equal(temp_sensor_missing, temp_sensor_missing_raw):
        failures.append("flag_temp_sensor_missing must equal flag_temp_sensor_missing_raw (Prompt 1 lock).")

    temp_used_meteo = _as_int01(df["flag_temp_used_meteo"])
    expected_used_meteo = ((temp_sensor_missing_raw == 1) & (df["temp_meteo_c"].notna().to_numpy())).astype(int)
    if not np.array_equal(temp_used_meteo, expected_used_meteo):
        idx = int(np.where(temp_used_meteo != expected_used_meteo)[0][0])
        failures.append(f"flag_temp_used_meteo mismatch at row {idx}")

    # outdoor_temp_c_effective: sensor preferred else meteo
    eff_expected = df["temp_sensor_c"].copy()
    eff_expected = eff_expected.where(~df["temp_sensor_c"].isna(), df["temp_meteo_c"])
    _assert_allclose(
        df["outdoor_temp_c_effective"].to_numpy(),
        eff_expected.to_numpy(),
        "outdoor_temp_c_effective",
        failures,
        atol=0.0,
    )
    _assert_allclose(
        df["outdoor_temp_c"].to_numpy(),
        df["outdoor_temp_c_effective"].to_numpy(),
        "outdoor_temp_c == outdoor_temp_c_effective",
        failures,
        atol=0.0,
    )

    # PV audit column sanity
    pv_cnt = pd.to_numeric(df["pv_count_samples"], errors="coerce").astype(int).to_numpy()
    pv_clip_cnt = pd.to_numeric(df["pv_count_samples_clipped_noise"], errors="coerce").astype(int).to_numpy()
    pv_night_cnt = pd.to_numeric(df["pv_count_samples_forced_night_zero"], errors="coerce").astype(int).to_numpy()

    if (pv_clip_cnt > pv_cnt).any():
        failures.append("pv_count_samples_clipped_noise exceeds pv_count_samples (impossible).")
    if (pv_night_cnt > pv_cnt).any():
        failures.append("pv_count_samples_forced_night_zero exceeds pv_count_samples (impossible).")

    clip_flag = _as_int01(df["flag_pv_clipped_noise_hour"])
    night_flag = _as_int01(df["flag_pv_forced_night_zero_hour"])
    if not np.array_equal(clip_flag, (pv_clip_cnt > 0).astype(int)):
        failures.append("flag_pv_clipped_noise_hour must equal int(pv_count_samples_clipped_noise > 0).")
    if not np.array_equal(night_flag, (pv_night_cnt > 0).astype(int)):
        failures.append("flag_pv_forced_night_zero_hour must equal int(pv_count_samples_forced_night_zero > 0).")

    # Solar hour classification lock
    sun_alt_mid = pd.to_numeric(df["sun_altitude_hour_mid_deg"], errors="coerce").to_numpy()
    daylight = _as_int01(df["is_daylight_hour"])
    expected_daylight = (sun_alt_mid > night_thr).astype(int)
    if not np.array_equal(daylight, expected_daylight):
        idx = int(np.where(daylight != expected_daylight)[0][0])
        failures.append(f"is_daylight_hour mismatch vs sun_altitude_hour_mid_deg > {night_thr} at row {idx}")

    # Deterministic time features from ts_utc only
    hour_exp = ts.dt.hour.astype(int).to_numpy()
    dow_exp = ts.dt.dayofweek.astype(int).to_numpy()  # Monday=0..Sunday=6 (LOCKED)
    month_exp = ts.dt.month.astype(int).to_numpy()
    weekend_exp = (dow_exp >= 5).astype(int)

    hour = pd.to_numeric(df["hour"], errors="coerce").astype(int).to_numpy()
    dow = pd.to_numeric(df["day_of_week"], errors="coerce").astype(int).to_numpy()
    month = pd.to_numeric(df["month"], errors="coerce").astype(int).to_numpy()
    weekend = _as_int01(df["is_weekend"])

    if not np.array_equal(hour, hour_exp):
        failures.append("hour feature mismatch vs ts_utc.")
    if not np.array_equal(dow, dow_exp):
        failures.append("day_of_week mismatch vs ts_utc (should be Monday=0..Sunday=6).")
    if not np.array_equal(month, month_exp):
        failures.append("month mismatch vs ts_utc.")
    if not np.array_equal(weekend, weekend_exp):
        failures.append("is_weekend mismatch vs ts_utc.")

    # hour_sin/cos
    hour_sin_exp = np.sin(2.0 * math.pi * hour_exp / 24.0)
    hour_cos_exp = np.cos(2.0 * math.pi * hour_exp / 24.0)
    _assert_allclose(df["hour_sin"].to_numpy(), hour_sin_exp, "hour_sin", failures, atol=1e-9)
    _assert_allclose(df["hour_cos"].to_numpy(), hour_cos_exp, "hour_cos", failures, atol=1e-9)

    # is_holiday_it
    hol_exp = compute_is_holiday_it(pd.DatetimeIndex(ts)).astype(int)
    hol = _as_int01(df["is_holiday_it"])
    if not np.array_equal(hol, hol_exp):
        idx = int(np.where(hol != hol_exp)[0][0])
        failures.append(f"is_holiday_it mismatch at row {idx}")

    # Basic observed series sanity
    for col in ["demand_w_obs", "pv_w_obs", "pool_w_obs"]:
        s = pd.to_numeric(df[col], errors="coerce")
        # Values may be NaN if missing, but should not be -inf/inf
        if np.isinf(s.to_numpy()).any():
            failures.append(f"{col} contains inf values.")
    pv_vals = pd.to_numeric(df["pv_w_obs"], errors="coerce")
    if (pv_vals.dropna() < -1e-9).any():
        failures.append("pv_w_obs contains negative values (should be >=0 after cleaning).")

    if failures:
        print("PROMPT 1 VALIDATION: FAIL")
        for i, msg in enumerate(failures[:50], start=1):
            print(f"{i}. {msg}")
        if len(failures) > 50:
            print(f"... and {len(failures) - 50} more")
        sys.exit(1)

    print("PROMPT 1 VALIDATION: PASS")
    sys.exit(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="processed_hourly/master_hourly_extended.csv")
    ap.add_argument("--config", type=str, default="config.yml")
    args = ap.parse_args()

    validate(Path(args.csv), Path(args.config))


if __name__ == "__main__":
    main()