from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import pvlib

from config_loader import load_frozen_config, cfg_get
from asserts import enforce_frozen_runtime_asserts


# -----------------------------
# Deterministic IT holiday logic
# -----------------------------
def _easter_date_gregorian(year: int) -> pd.Timestamp:
    """Anonymous Gregorian algorithm (deterministic). Returns Easter Sunday (date, naive)."""
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
    """Returns a set of UTC-normalized dates (00:00 UTC) for Italy public holidays."""
    fixed = [
        (1, 1),   # New Year
        (1, 6),   # Epiphany
        (4, 25),  # Liberation Day
        (5, 1),   # Labour Day
        (6, 2),   # Republic Day
        (8, 15),  # Ferragosto
        (11, 1),  # All Saints
        (12, 8),  # Immaculate Conception
        (12, 25), # Christmas
        (12, 26), # St. Stephen
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

    # Normalize to date at 00:00 UTC
    dates = ts_utc.normalize()
    return np.array([1 if d in holiday_dates else 0 for d in dates], dtype=int)


# -----------------------------
# tz-aware enforcement
# -----------------------------
def require_tz_aware_utc(dt: Union[pd.Series, pd.DatetimeIndex], name: str) -> None:
    if isinstance(dt, pd.Series):
        tz = dt.dt.tz
    else:
        tz = dt.tz

    if tz is None:
        raise ValueError(f"{name} is tz-naive. Must be tz-aware UTC (fail-fast lock).")
    if str(tz) != "UTC":
        raise ValueError(f"{name} must be UTC. Got tz={tz}.")


# -----------------------------
# HA export reader (fail-fast)
# -----------------------------
@dataclass(frozen=True)
class HaSeries:
    ts_utc: pd.Series
    value: pd.Series


def read_ha_export(path: Path, *, value_name: str, entity_id: Optional[str] = None) -> HaSeries:
    df = pd.read_csv(path)

    expected_cols = {"entity_id", "state", "last_changed"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(
            f"{path.name} must contain columns {sorted(expected_cols)}. Found: {df.columns.tolist()}"
        )

    if entity_id is not None:
        df = df[df["entity_id"] == entity_id].copy()

    unique_entities = df["entity_id"].dropna().unique()
    if len(unique_entities) != 1:
        raise ValueError(
            f"{path.name}: expected exactly 1 entity_id (or pass entity_id=...). Found: {unique_entities}"
        )

    ts = pd.to_datetime(df["last_changed"], errors="raise")  # preserves tz info if present
    # Fail-fast if tz-naive
    if getattr(ts.dt, "tz", None) is None:
        raise ValueError(f"{path.name}: timestamps are tz-naive. Must be tz-aware UTC.")
    ts_utc = ts.dt.tz_convert("UTC")
    require_tz_aware_utc(ts_utc, f"{path.name} ts_utc")

    val = pd.to_numeric(df["state"], errors="coerce")
    ok = val.notna()
    ts_utc = ts_utc[ok].reset_index(drop=True)
    val = val[ok].reset_index(drop=True)

    return HaSeries(ts_utc=ts_utc.rename("ts_utc"), value=val.rename(value_name))


# -----------------------------
# Solarposition wrapper (LOCKED)
# -----------------------------
def solarposition_for_times(times_utc: pd.DatetimeIndex, cfg) -> pd.DataFrame:
    require_tz_aware_utc(times_utc, "solarposition times_utc")

    site_lat = cfg_get(cfg, "config.site.latitude_deg")
    site_lon = cfg_get(cfg, "config.site.longitude_deg")
    site_alt = cfg_get(cfg, "config.site.elevation_m")
    method = cfg_get(cfg, "config.solar.solarposition_method")

    return pvlib.solarposition.get_solarposition(
        time=times_utc,
        latitude=site_lat,
        longitude=site_lon,
        altitude=site_alt,
        method=method,
    )


# -----------------------------
# Aggregation helpers
# -----------------------------
def aggregate_hourly(series: HaSeries, *, prefix: str) -> pd.DataFrame:
    ts = series.ts_utc
    require_tz_aware_utc(ts, f"{prefix} ts_utc")

    hour_start = ts.dt.floor("h")
    df = pd.DataFrame({"hour_start_utc": hour_start, "ts_utc": ts, "value": series.value})

    g = df.groupby("hour_start_utc", sort=True)

    mean_val = g["value"].mean()
    cnt = g["value"].count()

    # coverage minutes = (max_ts - min_ts) if >=2 samples else 0
    min_ts = g["ts_utc"].min()
    max_ts = g["ts_utc"].max()

    coverage = (max_ts - min_ts).dt.total_seconds() / 60.0
    coverage = coverage.where(cnt >= 2, 0.0)

    out = pd.DataFrame(
        {
            f"{prefix}_w_obs": mean_val.astype(float),
            f"{prefix}_count_samples": cnt.astype(int),
            f"{prefix}_coverage_minutes": coverage.astype(float),
        }
    )
    out.index.name = "ts_utc"
    return out


def aggregate_hourly_pv(pv_samples: pd.DataFrame) -> pd.DataFrame:
    # pv_samples columns: ts_utc, pv_sample_w_clean, flag_pv_forced_night_zero_sample, flag_pv_clipped_noise_sample
    require_tz_aware_utc(pv_samples["ts_utc"], "pv sample ts_utc")

    hour_start = pv_samples["ts_utc"].dt.floor("h")
    pv_samples = pv_samples.copy()
    pv_samples["hour_start_utc"] = hour_start

    g = pv_samples.groupby("hour_start_utc", sort=True)

    mean_val = g["pv_sample_w_clean"].mean()
    cnt = g["pv_sample_w_clean"].count()

    min_ts = g["ts_utc"].min()
    max_ts = g["ts_utc"].max()
    coverage = (max_ts - min_ts).dt.total_seconds() / 60.0
    coverage = coverage.where(cnt >= 2, 0.0)

    cnt_clip = g["flag_pv_clipped_noise_sample"].sum().astype(int)
    cnt_night = g["flag_pv_forced_night_zero_sample"].sum().astype(int)

    out = pd.DataFrame(
        {
            "pv_w_obs": mean_val.astype(float),
            "pv_count_samples": cnt.astype(int),
            "pv_coverage_minutes": coverage.astype(float),
            "pv_count_samples_clipped_noise": cnt_clip,
            "flag_pv_clipped_noise_hour": (cnt_clip > 0).astype(int),
            "pv_count_samples_forced_night_zero": cnt_night,
            "flag_pv_forced_night_zero_hour": (cnt_night > 0).astype(int),
        }
    )
    out.index.name = "ts_utc"
    return out


# -----------------------------
# Main builder (PROMPT 1)
# -----------------------------
def build_master_hourly_extended(
    *,
    demand_csv: Path,
    pv_csv: Path,
    pool_csv: Path,
    temp_sensor_csv: Path,
    cfg_path: Path = Path("config.yml"),
    out_path: Path = Path("processed_hourly/master_hourly_extended.csv"),
) -> pd.DataFrame:
    cfg = load_frozen_config(cfg_path)
    enforce_frozen_runtime_asserts(cfg)

    # Load config variables (Prompt 1 Step 0)
    altitude_field = cfg_get(cfg, "config.solar.altitude_field")
    night_thr = cfg_get(cfg, "config.solar.night_altitude_threshold_deg")
    ts_mode = cfg_get(cfg, "config.solar.timestamp_for_hour_classification")
    pv_noise_floor = cfg_get(cfg, "config.frozen_constants.pv_noise_floor_w")

    if ts_mode != "hour_midpoint":
        raise RuntimeError("FROZEN ASSERT FAILED: only 'hour_midpoint' is allowed for hour classification.")

    # Read raw exports
    demand = read_ha_export(demand_csv, value_name="demand_sample_w")
    pv = read_ha_export(pv_csv, value_name="pv_sample_w")
    pool = read_ha_export(pool_csv, value_name="pool_sample_w")
    temp_sensor = read_ha_export(temp_sensor_csv, value_name="temp_sensor_sample_c")

    # Fail-fast tz checks (global lock)
    require_tz_aware_utc(demand.ts_utc, "demand.ts_utc")
    require_tz_aware_utc(pv.ts_utc, "pv.ts_utc")
    require_tz_aware_utc(pool.ts_utc, "pool.ts_utc")
    require_tz_aware_utc(temp_sensor.ts_utc, "temp_sensor.ts_utc")

    # PV raw-sample cleaning (Prompt 1 Step 3)
    pv_ts = pv.ts_utc
    pv_w = pv.value.astype(float).copy()
    pv_w = pv_w.clip(lower=0.0)

    sol = solarposition_for_times(pd.DatetimeIndex(pv_ts), cfg)
    if altitude_field not in sol.columns:
        raise RuntimeError(
            f"FROZEN ASSERT FAILED: altitude_field='{altitude_field}' not in solarposition output columns."
        )
    sun_alt = sol[altitude_field].astype(float).reset_index(drop=True)

    forced_night = (sun_alt <= night_thr).astype(int)
    clipped_noise = ((sun_alt > night_thr) & (pv_w > 0.0) & (pv_w < pv_noise_floor)).astype(int)

    pv_w_clean = pv_w.copy()
    pv_w_clean = pv_w_clean.where(forced_night == 0, 0.0)
    pv_w_clean = pv_w_clean.where(clipped_noise == 0, 0.0)

    pv_samples = pd.DataFrame(
        {
            "ts_utc": pv_ts,
            "pv_sample_w_clean": pv_w_clean.astype(float),
            "flag_pv_forced_night_zero_sample": forced_night.astype(int),
            "flag_pv_clipped_noise_sample": clipped_noise.astype(int),
        }
    )

    # Hourly aggregation (Prompt 1 Step 4) + observed naming lock (Step 5)
    demand_h = aggregate_hourly(demand, prefix="demand")
    pool_h = aggregate_hourly(pool, prefix="pool")

    # Temperature hourly (keep raw sensor; meteo is absent here -> NaNs)
    temp_sensor_h = aggregate_hourly(temp_sensor, prefix="temp_sensor")
    temp_sensor_h = temp_sensor_h.rename(columns={"temp_sensor_w_obs": "temp_sensor_c"})

    pv_h = aggregate_hourly_pv(pv_samples)

    # Build a single continuous hourly backbone from min..max across all series (UTC)
    all_min = min(
        demand.ts_utc.min(),
        pv.ts_utc.min(),
        pool.ts_utc.min(),
        temp_sensor.ts_utc.min(),
    )
    all_max = max(
        demand.ts_utc.max(),
        pv.ts_utc.max(),
        pool.ts_utc.max(),
        temp_sensor.ts_utc.max(),
    )
    start = all_min.floor("h")
    end = all_max.floor("h")

    backbone = pd.date_range(start=start, end=end, freq="h", tz="UTC", name="ts_utc")
    master = pd.DataFrame(index=backbone)

    # Join all hourly tables
    master = master.join(demand_h, how="left")
    master = master.join(pv_h, how="left")
    master = master.join(pool_h, how="left")
    master = master.join(temp_sensor_h, how="left")

    # Fill missing count/coverage with 0 when hour absent
    for col in [
        "demand_count_samples", "demand_coverage_minutes",
        "pv_count_samples", "pv_coverage_minutes",
        "pool_count_samples", "pool_coverage_minutes",
        "temp_sensor_count_samples", "temp_sensor_coverage_minutes",
        "pv_count_samples_clipped_noise", "pv_count_samples_forced_night_zero",
        "flag_pv_clipped_noise_hour", "flag_pv_forced_night_zero_hour",
    ]:
        if col in master.columns:
            if col.endswith("_coverage_minutes"):
                master[col] = master[col].fillna(0.0)
            else:
                master[col] = master[col].fillna(0).astype(int)

    # Solar hour classification at hour midpoint (Prompt 1 Step 7)
    mid_times = master.index + pd.Timedelta(minutes=30)
    sol_mid = solarposition_for_times(mid_times, cfg)
    sun_alt_mid = sol_mid[altitude_field].astype(float).to_numpy()
    master["sun_altitude_hour_mid_deg"] = sun_alt_mid
    master["is_daylight_hour"] = (sun_alt_mid > night_thr).astype(int)

    # Temperature auditability + modeling compatibility (Prompt 1 Step 8)
    master["temp_meteo_c"] = np.nan  # raw meteo absent in current inputs
    master["outdoor_temp_c_effective"] = master["temp_sensor_c"]
    master["outdoor_temp_c"] = master["outdoor_temp_c_effective"]

    # Raw missing flags (Prompt 1 Step 9)
    master["flag_demand_missing_raw"] = (master["demand_count_samples"] == 0).astype(int)
    master["flag_pv_missing_raw"] = (master["pv_count_samples"] == 0).astype(int)
    master["flag_pool_missing_raw"] = (master["pool_count_samples"] == 0).astype(int)
    master["flag_temp_sensor_missing_raw"] = (master["temp_sensor_count_samples"] == 0).astype(int)
    master["flag_temp_meteo_missing_raw"] = 1  # because meteo series is absent here (all NaN)

    # Temperature provenance flags (Prompt 1 Step 10)
    master["flag_temp_sensor_missing"] = master["flag_temp_sensor_missing_raw"]
    master["flag_temp_used_meteo"] = (
        (master["flag_temp_sensor_missing_raw"] == 1) & (master["temp_meteo_c"].notna())
    ).astype(int)

    # Deterministic time features from ts_utc only (Prompt 1 Step 11)
    ts = master.index
    master["hour"] = ts.hour.astype(int)
    master["day_of_week"] = ts.dayofweek.astype(int)  # Mon=0..Sun=6
    master["month"] = ts.month.astype(int)
    master["is_weekend"] = (master["day_of_week"] >= 5).astype(int)

    master["is_holiday_it"] = compute_is_holiday_it(ts).astype(int)

    master["hour_sin"] = np.sin(2.0 * np.pi * master["hour"].to_numpy() / 24.0)
    master["hour_cos"] = np.cos(2.0 * np.pi * master["hour"].to_numpy() / 24.0)

    # Output
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master.reset_index().to_csv(out_path, index=False)

    return master


if __name__ == "__main__":
    build_master_hourly_extended(
        demand_csv=Path("Villa Demand from 08-12-2025 to 11-02-2026.csv"),
        pv_csv=Path("PV Power from 08-12-2025 to 11-02-2026.csv"),
        pool_csv=Path("Pool Power from 08-12-2025 to 11-02-2026.csv"),
        temp_sensor_csv=Path("Outside Temperature from 08-12-2025 to 11-02-2026.csv"),
        cfg_path=Path("config.yml"),
        out_path=Path("processed_hourly/master_hourly_extended.csv"),
    )
    print("Wrote processed_hourly/master_hourly_extended.csv")