# prompt2_build_datasets.py
# PROMPT 2 - BUILD DATASETS A1-B4 PER SPLIT (FULLY LOCKED)
#
# Inputs:
#  - processed_hourly/master_hourly_extended.csv
#  - config.yml (Prompt 0 frozen YAML)
#  - metrics/split_table.csv (Prompt 3)
#
# Outputs:
#  - datasets_by_split/{split_id}/{dataset_id}.csv
#  - datasets_by_split/{split_id}/dataset_manifest.csv
#  - datasets_by_split/manifest_all_splits.csv (optional, always written)
#
# Usage (PowerShell):
#   python .\prompt2_build_datasets.py ^
#       --master processed_hourly\master_hourly_extended.csv ^
#       --splits metrics\split_table.csv ^
#       --config config.yml ^
#       --out_dir datasets_by_split
#
# Notes:
#  - All constants come from config.yml (single source of truth).
#  - Split membership is keyed to ts_target_utc = ts_utc + 1h.
#  - Gap lengths computed ONCE globally on master series using RAW missing flags only.
#  - Pipeline A: causal fill only (ffill + fallback)
#  - Pipeline B: interpolation (short gaps only) within train/test segments, then ffill + fallback
#  - PV rules are PV-safe and twilight-safe by design.
#
# This script intentionally avoids interactive PowerShell execution of Python statements.
# Run it via: python .\prompt2_build_datasets.py ...

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Prompt 0 plumbing you already have
from config_loader import load_frozen_config, cfg_get

# You likely already created asserts.py for Prompt 1 validation.
# If not present, we still enforce the required checks locally below.
try:
    from asserts import enforce_frozen_runtime_asserts
except Exception:
    enforce_frozen_runtime_asserts = None


# -----------------------------
# Runtime asserts (Prompt 2)
# -----------------------------
def _enforce_prompt2_runtime_asserts(cfg) -> None:
    # pvlib version assert
    import pvlib  # noqa

    expected = str(cfg_get(cfg, "config.solar.library_version"))
    installed = str(pvlib.__version__)
    if installed != expected:
        raise RuntimeError(
            f"pvlib version mismatch: installed={installed} expected={expected}. "
            f"Fix by installing the exact version pinned in config.yml."
        )

    # Canonical PV regime edges and q33/q66
    edges = cfg_get(cfg, "config.frozen_constants.pv_regime_edges_w")

    # Accept list OR tuple (deep-freeze turns YAML lists into tuples)
    if not isinstance(edges, (list, tuple)) or len(edges) < 3:
        raise RuntimeError("config.frozen_constants.pv_regime_edges_w must be a list/tuple with >=3 elements")

    pv_q33 = float(cfg_get(cfg, "config.frozen_constants.pv_q33_w"))
    pv_q66 = float(cfg_get(cfg, "config.frozen_constants.pv_q66_w"))

    if pv_q33 != float(edges[1]) or pv_q66 != float(edges[2]):
        raise RuntimeError(
            "Canonical PV regime edges mismatch: "
            "pv_q33_w must equal pv_regime_edges_w[1] and pv_q66_w must equal pv_regime_edges_w[2]"
        )


# -----------------------------
# Global gap length computation
# -----------------------------
def gaplen_from_raw_missing(flag_missing_raw: np.ndarray) -> np.ndarray:
    """
    Given a boolean/int array where True/1 means raw missing at that hour,
    return gaplen_h where each missing element gets the total length (hours)
    of its contiguous missing run, and non-missing gets 0.
    """
    miss = np.asarray(flag_missing_raw).astype(bool)
    n = miss.size
    out = np.zeros(n, dtype=int)
    if n == 0:
        return out

    i = 0
    while i < n:
        if not miss[i]:
            i += 1
            continue
        j = i
        while j < n and miss[j]:
            j += 1
        run_len = j - i
        out[i:j] = run_len
        i = j
    return out


# -----------------------------
# Short-gap interpolation (J0)
# -----------------------------
def _interpolate_short_gaps_time(series: pd.Series, max_gap_h: int) -> tuple[pd.Series, np.ndarray]:
    """
    Linear interpolation in time, filling ONLY interior NaNs,
    and ONLY for NaN runs of length <= max_gap_h.

    Returns:
      filled_series, interp_mask (bool array aligned to series.index)
    """
    s = series.copy()

    # If nothing missing, quick exit
    na = s.isna().to_numpy()
    if not na.any():
        return s, np.zeros(len(s), dtype=bool)

    # Full interior interpolation (no extrapolation) to get candidate fills
    cand = s.interpolate(method="time", limit_area="inside")

    # Identify NaN runs in original s
    n = len(s)
    interp_mask = np.zeros(n, dtype=bool)

    i = 0
    while i < n:
        if not na[i]:
            i += 1
            continue
        j = i
        while j < n and na[j]:
            j += 1
        run_len = j - i

        # Interior means: has a valid point before and after inside this segment
        left_ok = (i - 1) >= 0 and not na[i - 1]
        right_ok = j < n and not na[j] if j < n else False

        if run_len <= max_gap_h and left_ok and right_ok:
            interp_mask[i:j] = True

        i = j

    # Apply only eligible interpolations
    filled = s.copy()
    filled.iloc[interp_mask] = cand.iloc[interp_mask]

    # interp_mask should only mark positions that were NaN and now non-NaN
    interp_mask = interp_mask & s.isna().to_numpy() & filled.notna().to_numpy()
    return filled, interp_mask


def interpolate_by_segments(
    s: pd.Series,
    ts_utc: pd.DatetimeIndex,
    is_train_split: np.ndarray,
    is_test_split: np.ndarray,
    max_gap_h: int,
) -> tuple[pd.Series, np.ndarray]:
    """
    Apply J0 interpolation separately within train segment and test segment.
    Never crosses test_start boundary because segments are isolated.
    """
    filled = s.copy()
    mask_all = np.zeros(len(s), dtype=bool)

    # Train segment
    if is_train_split.any():
        idx_train = np.where(is_train_split)[0]
        s_train = pd.Series(filled.iloc[idx_train].to_numpy(), index=ts_utc[idx_train])
        s_train_f, m_train = _interpolate_short_gaps_time(s_train, max_gap_h)
        filled.iloc[idx_train] = s_train_f.to_numpy()
        mask_all[idx_train] = m_train

    # Test segment
    if is_test_split.any():
        idx_test = np.where(is_test_split)[0]
        s_test = pd.Series(filled.iloc[idx_test].to_numpy(), index=ts_utc[idx_test])
        s_test_f, m_test = _interpolate_short_gaps_time(s_test, max_gap_h)
        filled.iloc[idx_test] = s_test_f.to_numpy()
        mask_all[idx_test] = mask_all[idx_test] | m_test

    return filled, mask_all


def limited_ffill(s: pd.Series, max_h: int) -> tuple[pd.Series, np.ndarray]:
    before = s.copy()
    after = s.ffill(limit=int(max_h))
    mask = before.isna().to_numpy() & after.notna().to_numpy()
    return after, mask


def fallback_fill_scalar(s: pd.Series, scalar: float) -> tuple[pd.Series, np.ndarray]:
    before = s.copy()
    after = s.fillna(float(scalar))
    mask = before.isna().to_numpy() & after.notna().to_numpy()
    return after, mask


def fallback_fill_zero(s: pd.Series) -> tuple[pd.Series, np.ndarray]:
    before = s.copy()
    after = s.fillna(0.0)
    mask = before.isna().to_numpy() & after.notna().to_numpy()
    return after, mask


def nanmedian_or_fail(values: np.ndarray, name: str) -> float:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        raise RuntimeError(f"Train-only nanmedian fallback failed: no observed values for {name} in train.")
    return float(np.nanmedian(v))


# -----------------------------
# Binning helpers
# -----------------------------
def digitize_bin(x: np.ndarray, edges: list[float]) -> np.ndarray:
    """
    temp_bin rule:
      bin = digitize(x, edges, right=False) - 1
      clamp to [0, n_bins-1]
      missing -> -1
    """
    arr = np.asarray(x, dtype=float)
    out = np.full(arr.shape, -1, dtype=int)

    ok = ~np.isnan(arr)
    if not ok.any():
        return out

    e = np.asarray(edges, dtype=float)
    b = np.digitize(arr[ok], e, right=False) - 1
    n_bins = len(e) - 1
    b = np.clip(b, 0, max(n_bins - 1, 0))
    out[ok] = b.astype(int)
    return out


# -----------------------------
# PV regime
# -----------------------------
def compute_pv_regime(is_daylight: np.ndarray, pv_w: np.ndarray, pv_daylight_thr: float, pv_q33: float, pv_q66: float) -> np.ndarray:
    is_day = (np.asarray(is_daylight).astype(int) == 1)
    pv = np.asarray(pv_w, dtype=float)

    regime = np.array(["night"] * len(pv), dtype=object)

    # only classify day hours further
    day_idx = np.where(is_day)[0]
    if day_idx.size == 0:
        return regime

    pv_day = pv[day_idx]

    low = (pv_daylight_thr < pv_day) & (pv_day <= pv_q33)
    med = (pv_q33 < pv_day) & (pv_day <= pv_q66)
    high = (pv_day > pv_q66)

    regime_day = np.array(["night"] * len(day_idx), dtype=object)
    regime_day[low] = "low"
    regime_day[med] = "med"
    regime_day[high] = "high"

    regime[day_idx] = regime_day
    return regime


# -----------------------------
# Main dataset builder
# -----------------------------
def build_all(cfg, master_csv: Path, splits_csv: Path, out_dir: Path) -> None:
    # Enforce Prompt 0 asserts if available + Prompt 2 asserts
    if enforce_frozen_runtime_asserts is not None:
        enforce_frozen_runtime_asserts(cfg)
    _enforce_prompt2_runtime_asserts(cfg)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load master
    df = pd.read_csv(master_csv)
    if "ts_utc" not in df.columns:
        raise RuntimeError("master_hourly_extended.csv must contain ts_utc")

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="raise")
    df = df.sort_values("ts_utc").reset_index(drop=True)

    # Global gap lengths (Prompt 2.D)
    for sig, flag_col in [
        ("demand", "flag_demand_missing_raw"),
        ("pv", "flag_pv_missing_raw"),
        ("pool", "flag_pool_missing_raw"),
        # temperature: raw missing means outdoor_temp_c missing (effective)
        # your Prompt 1 provides both sensor/meteo raw missing flags; we define temp raw missing as outdoor_temp_c isna
    ]:
        if flag_col not in df.columns:
            raise RuntimeError(f"Missing required raw missing flag column: {flag_col}")
        df[f"gaplen_{sig}_h"] = gaplen_from_raw_missing(df[flag_col].to_numpy())

    df["flag_temp_missing_raw"] = df["outdoor_temp_c"].isna().astype(int)
    df["gaplen_temp_h"] = gaplen_from_raw_missing(df["flag_temp_missing_raw"].to_numpy())

    # Load split table (Prompt 3)
    sp = pd.read_csv(splits_csv)
    required_sp = ["split_id", "train_start_utc", "train_end_utc", "test_start_utc", "test_end_utc", "week_index_utc"]
    for c in required_sp:
        if c not in sp.columns:
            raise RuntimeError(f"split_table.csv missing column: {c}")

    sp["train_start_utc"] = pd.to_datetime(sp["train_start_utc"], utc=True, errors="raise")
    sp["train_end_utc"] = pd.to_datetime(sp["train_end_utc"], utc=True, errors="raise")
    sp["test_start_utc"] = pd.to_datetime(sp["test_start_utc"], utc=True, errors="raise")
    sp["test_end_utc"] = pd.to_datetime(sp["test_end_utc"], utc=True, errors="raise")

    # Config constants
    SHORTGAPHOURS = int(cfg_get(cfg, "config.frozen_constants.SHORTGAPHOURS"))
    demand_ffill_max_h = int(cfg_get(cfg, "config.frozen_constants.demand_ffill_max_h"))
    pool_ffill_max_h = int(cfg_get(cfg, "config.frozen_constants.pool_ffill_max_h"))
    temp_ffill_max_h = int(cfg_get(cfg, "config.frozen_constants.temp_ffill_max_h"))
    pv_ffill_max_h = int(cfg_get(cfg, "config.frozen_constants.pv_ffill_max_h"))

    pv_daylight_thr = float(cfg_get(cfg, "config.frozen_constants.pv_daylight_threshold_w"))
    edges = cfg_get(cfg, "config.frozen_constants.pv_regime_edges_w")
    pv_q33 = float(edges[1])
    pv_q66 = float(edges[2])

    pool_on_thr = float(cfg_get(cfg, "config.frozen_constants.pool_on_threshold_w"))

    temp_bin_edges_c = list(cfg_get(cfg, "config.frozen_constants.temp_bin_edges_c"))
    gaplen_bin_edges_h = list(cfg_get(cfg, "config.frozen_constants.gaplen_bin_edges_h"))

    # Feature schema (from config)
    feat_map = cfg_get(cfg, "config.features.feature_cols_by_dataset")
    # deep-freeze gives MappingProxyType, values may be tuple
    feat_map = dict(feat_map)

    dataset_ids = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]

    all_manifests = []

    # Process each split
    for _, row in sp.iterrows():
        split_id = str(row["split_id"])
        train_start = row["train_start_utc"]
        train_end = row["train_end_utc"]
        test_start = row["test_start_utc"]
        test_end = row["test_end_utc"]
        week_idx = int(row["week_index_utc"])

        # Build per-split frame with membership keyed to ts_target_utc
        df_s = df.copy()
        df_s["ts_target_utc"] = df_s["ts_utc"] + pd.Timedelta(hours=1)

        is_train = ((df_s["ts_target_utc"] >= train_start) & (df_s["ts_target_utc"] < train_end)).astype(int).to_numpy()
        is_test = ((df_s["ts_target_utc"] >= test_start) & (df_s["ts_target_utc"] < test_end)).astype(int).to_numpy()

        keep = (is_train == 1) | (is_test == 1)
        df_s = df_s.loc[keep].copy().reset_index(drop=True)
        is_train = is_train[keep]
        is_test = is_test[keep]

        df_s["split_id"] = split_id
        df_s["train_start_utc"] = train_start.isoformat()
        df_s["train_end_utc"] = train_end.isoformat()
        df_s["test_start_utc"] = test_start.isoformat()
        df_s["test_end_utc"] = test_end.isoformat()
        df_s["week_index_utc"] = week_idx
        df_s["is_train_split"] = is_train.astype(int)
        df_s["is_test_split"] = is_test.astype(int)

        # -----------------------------
        # Pipeline builder (A or B)
        # -----------------------------
        def apply_pipeline(pipeline_name: str) -> pd.DataFrame:
            dfx = df_s.copy()
            ts = pd.DatetimeIndex(pd.to_datetime(dfx["ts_utc"], utc=True, errors="raise"))

            is_tr = dfx["is_train_split"].to_numpy().astype(bool)
            is_te = dfx["is_test_split"].to_numpy().astype(bool)

            # Observed series
            demand_obs = pd.Series(pd.to_numeric(dfx["demand_w_obs"], errors="coerce").to_numpy(), index=ts)
            pool_obs = pd.Series(pd.to_numeric(dfx["pool_w_obs"], errors="coerce").to_numpy(), index=ts)
            pv_obs = pd.Series(pd.to_numeric(dfx["pv_w_obs"], errors="coerce").to_numpy(), index=ts)
            temp_obs = pd.Series(pd.to_numeric(dfx["outdoor_temp_c"], errors="coerce").to_numpy(), index=ts)

            # Raw missing flags (use existing, locked)
            flag_demand_missing_raw = pd.to_numeric(dfx["flag_demand_missing_raw"], errors="coerce").fillna(0).astype(int).to_numpy()
            flag_pool_missing_raw = pd.to_numeric(dfx["flag_pool_missing_raw"], errors="coerce").fillna(0).astype(int).to_numpy()
            flag_pv_missing_raw = pd.to_numeric(dfx["flag_pv_missing_raw"], errors="coerce").fillna(0).astype(int).to_numpy()

            # temp raw missing is outdoor_temp_c missing
            flag_temp_missing_raw = pd.to_numeric(dfx["flag_temp_missing_raw"], errors="coerce").fillna(0).astype(int).to_numpy()

            # is_daylight (solar-only)
            is_daylight = pd.to_numeric(dfx["is_daylight_hour"], errors="coerce").fillna(0).astype(int).to_numpy()

            # Fallback scalars (train-only, observed train rows only)
            demand_fallback = nanmedian_or_fail(demand_obs.to_numpy()[is_tr], "demand_w_obs")
            temp_fallback = nanmedian_or_fail(temp_obs.to_numpy()[is_tr], "outdoor_temp_c")

            # -----------------------------
            # Demand imputation
            # -----------------------------
            demand_w = demand_obs.copy()
            demand_imputed_interp = np.zeros(len(demand_w), dtype=int)

            if pipeline_name == "B":
                demand_w, m_interp = interpolate_by_segments(
                    demand_w, ts, is_tr, is_te, SHORTGAPHOURS
                )
                demand_imputed_interp = m_interp.astype(int)

            demand_w, m_ffill = limited_ffill(demand_w, demand_ffill_max_h)
            demand_w, m_fallback = fallback_fill_scalar(demand_w, demand_fallback)

            demand_imputed_ffill = m_ffill.astype(int)
            demand_imputed_fallback = m_fallback.astype(int)

            demand_imputed_any = ((demand_imputed_interp + demand_imputed_ffill + demand_imputed_fallback) > 0).astype(int)
            flag_demand_imputed = demand_imputed_any.astype(int)

            # -----------------------------
            # Pool imputation
            # -----------------------------
            pool_w = pool_obs.copy()
            pool_imputed_interp = np.zeros(len(pool_w), dtype=int)

            if pipeline_name == "B":
                pool_w, m_interp = interpolate_by_segments(
                    pool_w, ts, is_tr, is_te, SHORTGAPHOURS
                )
                pool_imputed_interp = m_interp.astype(int)

            pool_w, m_ffill = limited_ffill(pool_w, pool_ffill_max_h)
            pool_w, m_fallback = fallback_fill_zero(pool_w)

            pool_imputed_ffill = m_ffill.astype(int)
            pool_imputed_fallback = m_fallback.astype(int)

            pool_imputed_any = ((pool_imputed_interp + pool_imputed_ffill + pool_imputed_fallback) > 0).astype(int)
            flag_pool_imputed = pool_imputed_any.astype(int)

            # -----------------------------
            # Temperature imputation
            # -----------------------------
            outdoor_temp_c = temp_obs.copy()
            temp_imputed_interp = np.zeros(len(outdoor_temp_c), dtype=int)

            if pipeline_name == "B":
                outdoor_temp_c, m_interp = interpolate_by_segments(
                    outdoor_temp_c, ts, is_tr, is_te, SHORTGAPHOURS
                )
                temp_imputed_interp = m_interp.astype(int)

            outdoor_temp_c, m_ffill = limited_ffill(outdoor_temp_c, temp_ffill_max_h)
            outdoor_temp_c, m_fallback = fallback_fill_scalar(outdoor_temp_c, temp_fallback)

            temp_imputed_ffill = m_ffill.astype(int)
            temp_imputed_fallback = m_fallback.astype(int)

            temp_imputed_any = ((temp_imputed_interp + temp_imputed_ffill + temp_imputed_fallback) > 0).astype(int)
            flag_temp_imputed = temp_imputed_any.astype(int)

            # -----------------------------
            # PV imputation (PV-safe + twilight-safe)
            # -----------------------------
            pv_w = pv_obs.copy()
            pv_imputed_interp = np.zeros(len(pv_w), dtype=int)

            if pipeline_name == "B":
                pv_w, m_interp = interpolate_by_segments(
                    pv_w, ts, is_tr, is_te, SHORTGAPHOURS
                )
                pv_imputed_interp = m_interp.astype(int)

            # PV-safe night-ok ffill to zero (Pipeline A rule), only on remaining NaNs
            pv_imputed_ffill_night_ok = np.zeros(len(pv_w), dtype=int)
            pv_imputed_fallback_zero = np.zeros(len(pv_w), dtype=int)

            # Determine last observed pv_w_obs (raw missing == 0)
            pv_obs_arr = pv_obs.to_numpy()
            obs_ok = (np.asarray(flag_pv_missing_raw).astype(int) == 0) & (~np.isnan(pv_obs_arr))

            last_obs_val = pd.Series(np.where(obs_ok, pv_obs_arr, np.nan), index=ts).ffill().to_numpy()
            last_obs_ts = pd.Series(np.where(obs_ok, ts.view("i8"), np.nan), index=ts).ffill().to_numpy()

            # Distance in hours since last observed
            # last_obs_ts holds int64 ns; convert
            dist_h = np.full(len(ts), 10**9, dtype=int)
            ok_last = ~np.isnan(last_obs_ts)
            if ok_last.any():
                last_ns = last_obs_ts[ok_last].astype(np.int64)
                cur_ns = ts.view("i8")[ok_last].astype(np.int64)
                dist = (cur_ns - last_ns) // (3600 * 10**9)
                dist_h[ok_last] = dist.astype(int)

            # Apply night-ok fill only where pv_w is still NaN
            cur_nan = np.isnan(pv_w.to_numpy())
            eligible = (
                (np.asarray(flag_pv_missing_raw).astype(int) == 1)
                & (np.asarray(is_daylight).astype(int) == 0)
                & cur_nan
                & (np.asarray(last_obs_val) == 0.0)
                & (dist_h <= pv_ffill_max_h)
            )

            if eligible.any():
                pv_w.iloc[np.where(eligible)[0]] = 0.0
                pv_imputed_ffill_night_ok[eligible] = 1

            # Remaining missing -> zero
            still_nan = np.isnan(pv_w.to_numpy())
            if still_nan.any():
                pv_w.iloc[np.where(still_nan)[0]] = 0.0
                pv_imputed_fallback_zero[still_nan] = 1

            # Twilight-safe post-imputation enforcement:
            # if raw-missing AND is_daylight==0 then pv_w=0
            raw_missing_night = (np.asarray(flag_pv_missing_raw).astype(int) == 1) & (np.asarray(is_daylight).astype(int) == 0)
            if raw_missing_night.any():
                pv_w.iloc[np.where(raw_missing_night)[0]] = 0.0

            # Never override observed PV hours
            observed_pv_hours = (np.asarray(flag_pv_missing_raw).astype(int) == 0) & (~np.isnan(pv_obs_arr))
            if observed_pv_hours.any():
                pv_w.iloc[np.where(observed_pv_hours)[0]] = pv_obs.iloc[np.where(observed_pv_hours)[0]].to_numpy()

            pv_imputed_any = ((pv_imputed_interp + pv_imputed_ffill_night_ok + pv_imputed_fallback_zero) > 0).astype(int)
            flag_pv_imputed = pv_imputed_any.astype(int)

            # -----------------------------
            # Attach to dfx (post-imputation)
            # -----------------------------
            dfx["demand_w"] = demand_w.to_numpy()
            dfx["pool_w"] = pool_w.to_numpy()
            dfx["pv_w"] = pv_w.to_numpy()
            dfx["outdoor_temp_c"] = outdoor_temp_c.to_numpy()

            # Method flags (int 0/1)
            dfx["demand_imputed_interp"] = demand_imputed_interp.astype(int)
            dfx["demand_imputed_ffill"] = demand_imputed_ffill.astype(int)
            dfx["demand_imputed_fallback"] = demand_imputed_fallback.astype(int)
            dfx["demand_imputed_any"] = demand_imputed_any.astype(int)

            dfx["pool_imputed_interp"] = pool_imputed_interp.astype(int)
            dfx["pool_imputed_ffill"] = pool_imputed_ffill.astype(int)
            dfx["pool_imputed_fallback"] = pool_imputed_fallback.astype(int)
            dfx["pool_imputed_any"] = pool_imputed_any.astype(int)

            dfx["temp_imputed_interp"] = temp_imputed_interp.astype(int)
            dfx["temp_imputed_ffill"] = temp_imputed_ffill.astype(int)
            dfx["temp_imputed_fallback"] = temp_imputed_fallback.astype(int)
            dfx["temp_imputed_any"] = temp_imputed_any.astype(int)

            dfx["pv_imputed_interp"] = pv_imputed_interp.astype(int)
            dfx["pv_imputed_ffill_night_ok"] = pv_imputed_ffill_night_ok.astype(int)
            dfx["pv_imputed_fallback_zero"] = pv_imputed_fallback_zero.astype(int)
            dfx["pv_imputed_any"] = pv_imputed_any.astype(int)

            # Combined flags (int 0/1, no NaNs)
            dfx["flag_demand_imputed"] = flag_demand_imputed.astype(int)
            dfx["flag_pool_imputed"] = flag_pool_imputed.astype(int)
            dfx["flag_temp_imputed"] = flag_temp_imputed.astype(int)
            dfx["flag_pv_imputed"] = flag_pv_imputed.astype(int)

            # Schema aliasing (materialize *_new identical)
            dfx["flag_demand_imputed_new"] = dfx["flag_demand_imputed"].astype(int)
            dfx["flag_pool_imputed_new"] = dfx["flag_pool_imputed"].astype(int)
            dfx["flag_temp_imputed_new"] = dfx["flag_temp_imputed"].astype(int)
            dfx["flag_pv_imputed_new"] = dfx["flag_pv_imputed"].astype(int)

            # Fail hard if mismatch (locked)
            for base, new in [
                ("flag_demand_imputed", "flag_demand_imputed_new"),
                ("flag_pool_imputed", "flag_pool_imputed_new"),
                ("flag_temp_imputed", "flag_temp_imputed_new"),
                ("flag_pv_imputed", "flag_pv_imputed_new"),
            ]:
                if not np.array_equal(dfx[base].to_numpy().astype(int), dfx[new].to_numpy().astype(int)):
                    raise RuntimeError(f"Schema alias mismatch: {base} != {new}")

            # Cross-signal summaries (K3)
            dfx["imputed_any"] = ((dfx["demand_imputed_any"] + dfx["pv_imputed_any"] + dfx["pool_imputed_any"] + dfx["temp_imputed_any"]) > 0).astype(int)
            dfx["imputed_count"] = (
                dfx["demand_imputed_any"].astype(int)
                + dfx["pv_imputed_any"].astype(int)
                + dfx["pool_imputed_any"].astype(int)
                + dfx["temp_imputed_any"].astype(int)
            ).astype(int)

            # Grid exporting flag (K4)
            dfx["flag_grid_exporting"] = (pd.to_numeric(dfx["demand_w"], errors="coerce") < 0).astype(int)

            # Target (E2)
            dfx = dfx.sort_values("ts_utc").reset_index(drop=True)
            dfx["y_demand_t_plus_1"] = pd.to_numeric(dfx["demand_w"], errors="coerce").shift(-1)

            # Drop rows where target missing (locked)
            dfx = dfx.loc[~dfx["y_demand_t_plus_1"].isna()].copy().reset_index(drop=True)

            # Regimes and bins (G + Pool regime + binning lock)
            dfx["is_daylight"] = pd.to_numeric(dfx["is_daylight_hour"], errors="coerce").fillna(0).astype(int)

            dfx["pv_regime"] = compute_pv_regime(
                dfx["is_daylight"].to_numpy(),
                pd.to_numeric(dfx["pv_w"], errors="coerce").to_numpy(),
                pv_daylight_thr,
                pv_q33,
                pv_q66,
            )

            dfx["pool_on"] = (pd.to_numeric(dfx["pool_w"], errors="coerce") > pool_on_thr).astype(int)
            dfx["pool_on_lag_1h"] = dfx["pool_on"].shift(1)
            dfx["pool_switch"] = (dfx["pool_on"] != dfx["pool_on_lag_1h"]).astype(int)

            # Bins
            dfx["temp_bin"] = digitize_bin(pd.to_numeric(dfx["outdoor_temp_c"], errors="coerce").to_numpy(), temp_bin_edges_c)

            dfx["gaplen_bin_demand"] = digitize_bin(pd.to_numeric(dfx["gaplen_demand_h"], errors="coerce").to_numpy(), gaplen_bin_edges_h)
            dfx["gaplen_bin_pv"] = digitize_bin(pd.to_numeric(dfx["gaplen_pv_h"], errors="coerce").to_numpy(), gaplen_bin_edges_h)
            dfx["gaplen_bin_pool"] = digitize_bin(pd.to_numeric(dfx["gaplen_pool_h"], errors="coerce").to_numpy(), gaplen_bin_edges_h)
            dfx["gaplen_bin_temp"] = digitize_bin(pd.to_numeric(dfx["gaplen_temp_h"], errors="coerce").to_numpy(), gaplen_bin_edges_h)

            # Lags (M4)
            for lag_h in [1, 24, 168]:
                dfx[f"demand_lag_{lag_h}h"] = pd.to_numeric(dfx["demand_w"], errors="coerce").shift(lag_h)
                dfx[f"pv_w_lag_{lag_h}h"] = pd.to_numeric(dfx["pv_w"], errors="coerce").shift(lag_h)
                dfx[f"pool_w_lag_{lag_h}h"] = pd.to_numeric(dfx["pool_w"], errors="coerce").shift(lag_h)
                dfx[f"outdoor_temp_c_lag_{lag_h}h"] = pd.to_numeric(dfx["outdoor_temp_c"], errors="coerce").shift(lag_h)

                # Lag provenance flags: was the source hour imputed?
                dfx[f"prov_demand_lag_{lag_h}h_imputed"] = pd.to_numeric(dfx["flag_demand_imputed"], errors="coerce").fillna(0).astype(int).shift(lag_h)
                dfx[f"prov_pv_lag_{lag_h}h_imputed"] = pd.to_numeric(dfx["flag_pv_imputed"], errors="coerce").fillna(0).astype(int).shift(lag_h)
                dfx[f"prov_pool_lag_{lag_h}h_imputed"] = pd.to_numeric(dfx["flag_pool_imputed"], errors="coerce").fillna(0).astype(int).shift(lag_h)
                dfx[f"prov_temp_lag_{lag_h}h_imputed"] = pd.to_numeric(dfx["flag_temp_imputed"], errors="coerce").fillna(0).astype(int).shift(lag_h)

            # Recompute ts_target_utc after drops (still consistent)
            dfx["ts_utc"] = pd.to_datetime(dfx["ts_utc"], utc=True, errors="raise")
            dfx["ts_target_utc"] = dfx["ts_utc"] + pd.Timedelta(hours=1)

            # Recompute split membership (must stay keyed to ts_target_utc)
            tr2 = ((dfx["ts_target_utc"] >= train_start) & (dfx["ts_target_utc"] < train_end)).astype(int)
            te2 = ((dfx["ts_target_utc"] >= test_start) & (dfx["ts_target_utc"] < test_end)).astype(int)
            dfx["is_train_split"] = tr2.astype(int)
            dfx["is_test_split"] = te2.astype(int)

            # Ensure we keep only rows in train OR test (locked)
            dfx = dfx.loc[(dfx["is_train_split"] == 1) | (dfx["is_test_split"] == 1)].copy().reset_index(drop=True)

            return dfx

        # Build base frames for pipelines
        df_A = apply_pipeline("A")
        df_B = apply_pipeline("B")

        # Output per split
        split_dir = out_dir / split_id
        split_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows = []

        # Helper: apply clean-only drops for a dataset_id
        def apply_clean_only(df_in: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
            dfo = df_in.copy()

            clean_only = dataset_id in ["A2", "A4", "B2", "B4"]
            if not clean_only:
                return dfo

            demand_now = pd.to_numeric(dfo["flag_demand_imputed"], errors="coerce").fillna(0).astype(int).to_numpy()

            # Demand lags used by all demand-only datasets
            prov_d1 = pd.to_numeric(dfo["prov_demand_lag_1h_imputed"], errors="coerce").fillna(0).astype(int).to_numpy()
            prov_d24 = pd.to_numeric(dfo["prov_demand_lag_24h_imputed"], errors="coerce").fillna(0).astype(int).to_numpy()
            prov_d168 = pd.to_numeric(dfo["prov_demand_lag_168h_imputed"], errors="coerce").fillna(0).astype(int).to_numpy()

            if dataset_id in ["A2", "B2"]:
                drop = (demand_now == 1) | (prov_d1 == 1) | (prov_d24 == 1) | (prov_d168 == 1)
                return dfo.loc[~drop].copy().reset_index(drop=True)

            # Telemetry clean-only (A4,B4): any signal now or any lag source imputed
            pv_now = pd.to_numeric(dfo["flag_pv_imputed"], errors="coerce").fillna(0).astype(int).to_numpy()
            pool_now = pd.to_numeric(dfo["flag_pool_imputed"], errors="coerce").fillna(0).astype(int).to_numpy()
            temp_now = pd.to_numeric(dfo["flag_temp_imputed"], errors="coerce").fillna(0).astype(int).to_numpy()

            provs = []
            for lag_h in [1, 24, 168]:
                provs.append(pd.to_numeric(dfo[f"prov_demand_lag_{lag_h}h_imputed"], errors="coerce").fillna(0).astype(int).to_numpy())
                provs.append(pd.to_numeric(dfo[f"prov_pv_lag_{lag_h}h_imputed"], errors="coerce").fillna(0).astype(int).to_numpy())
                provs.append(pd.to_numeric(dfo[f"prov_pool_lag_{lag_h}h_imputed"], errors="coerce").fillna(0).astype(int).to_numpy())
                provs.append(pd.to_numeric(dfo[f"prov_temp_lag_{lag_h}h_imputed"], errors="coerce").fillna(0).astype(int).to_numpy())

            prov_any = np.zeros(len(dfo), dtype=bool)
            for p in provs:
                prov_any = prov_any | (p == 1)

            drop = (demand_now == 1) | (pv_now == 1) | (pool_now == 1) | (temp_now == 1) | prov_any
            return dfo.loc[~drop].copy().reset_index(drop=True)

        # Helper: final required feature NaN drop
        def drop_required_feature_nans(df_in: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
            req = feat_map.get(dataset_id)
            if req is None:
                raise RuntimeError(f"Missing feature list for dataset {dataset_id} in config.yml")

            # deep-freeze may make this tuple
            req_cols = list(req)

            # Drop rows where any required feature is NaN
            dfo = df_in.copy()
            missing_any = np.zeros(len(dfo), dtype=bool)
            for c in req_cols:
                if c not in dfo.columns:
                    raise RuntimeError(f"Dataset {dataset_id} requires feature column missing from frame: {c}")
                missing_any = missing_any | pd.to_numeric(dfo[c], errors="coerce").isna().to_numpy()

            dfo = dfo.loc[~missing_any].copy().reset_index(drop=True)
            return dfo

        # Mandatory columns (Prompt 2.P minimum set plus locked additions)
        def select_output_columns(df_in: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
            req_cols = list(feat_map[dataset_id])

            base_cols = [
                "split_id", "train_start_utc", "train_end_utc", "test_start_utc", "test_end_utc",
                "is_train_split", "is_test_split",
                "ts_utc", "ts_target_utc",
                "y_demand_t_plus_1",
            ]

            regime_cols = [
                "is_daylight", "pv_regime", "pool_on", "pool_on_lag_1h", "pool_switch",
                "temp_bin", "flag_grid_exporting",
            ]

            imputation_cols = [
                "demand_imputed_interp", "demand_imputed_ffill", "demand_imputed_fallback", "demand_imputed_any",
                "pool_imputed_interp", "pool_imputed_ffill", "pool_imputed_fallback", "pool_imputed_any",
                "temp_imputed_interp", "temp_imputed_ffill", "temp_imputed_fallback", "temp_imputed_any",
                "pv_imputed_interp", "pv_imputed_ffill_night_ok", "pv_imputed_fallback_zero", "pv_imputed_any",
                "flag_demand_imputed", "flag_demand_imputed_new",
                "flag_pv_imputed", "flag_pv_imputed_new",
                "flag_pool_imputed", "flag_pool_imputed_new",
                "flag_temp_imputed", "flag_temp_imputed_new",
                "imputed_any", "imputed_count",
            ]

            gap_cols = [
                "gaplen_demand_h", "gaplen_pv_h", "gaplen_pool_h", "gaplen_temp_h",
                "gaplen_bin_demand", "gaplen_bin_pv", "gaplen_bin_pool", "gaplen_bin_temp",
            ]

            # Keep observed columns too (audit-friendly)
            obs_cols = ["demand_w_obs", "pv_w_obs", "pool_w_obs", "outdoor_temp_c"]

            # Keep post-imputation series
            post_cols = ["demand_w", "pv_w", "pool_w"]

            # Lag provenance columns used by clean-only logic
            prov_cols = []
            for lag_h in [1, 24, 168]:
                prov_cols += [
                    f"prov_demand_lag_{lag_h}h_imputed",
                    f"prov_pv_lag_{lag_h}h_imputed",
                    f"prov_pool_lag_{lag_h}h_imputed",
                    f"prov_temp_lag_{lag_h}h_imputed",
                ]

            cols = base_cols + req_cols + regime_cols + post_cols + obs_cols + imputation_cols + gap_cols + prov_cols

            # Deduplicate while preserving order
            seen = set()
            final_cols = []
            for c in cols:
                if c in seen:
                    continue
                if c not in df_in.columns:
                    # Only allow missing provenance columns in demand-only datasets for pv/pool/temp lags,
                    # but we still create them, so missing is an error.
                    raise RuntimeError(f"Output column missing for dataset {dataset_id}: {c}")
                seen.add(c)
                final_cols.append(c)

            return df_in[final_cols].copy()

        # Build and write each dataset
        for dataset_id in dataset_ids:
            base_frame = df_A if dataset_id.startswith("A") else df_B

            dfo = base_frame.copy()

            # Clean-only drops (N)
            dfo = apply_clean_only(dfo, dataset_id)

            # Drop rows with NaNs in required features (O)
            dfo = drop_required_feature_nans(dfo, dataset_id)

            # Ensure rows only belong to train or test (B)
            dfo = dfo.loc[(dfo["is_train_split"] == 1) | (dfo["is_test_split"] == 1)].copy().reset_index(drop=True)

            # Select final columns
            dfo_out = select_output_columns(dfo, dataset_id)

            # Write
            out_path = split_dir / f"{dataset_id}.csv"
            dfo_out.to_csv(out_path, index=False)

            n_total = len(dfo_out)
            n_train = int(pd.to_numeric(dfo_out["is_train_split"], errors="coerce").fillna(0).astype(int).sum())
            n_test = int(pd.to_numeric(dfo_out["is_test_split"], errors="coerce").fillna(0).astype(int).sum())

            manifest_rows.append(
                {
                    "split_id": split_id,
                    "dataset_id": dataset_id,
                    "pipeline": "A" if dataset_id.startswith("A") else "B",
                    "mode": "clean-only" if dataset_id in ["A2", "A4", "B2", "B4"] else "keep-imputed",
                    "n_rows_total": n_total,
                    "n_rows_train": n_train,
                    "n_rows_test": n_test,
                }
            )

            all_manifests.append(
                {
                    "split_id": split_id,
                    "dataset_id": dataset_id,
                    "pipeline": "A" if dataset_id.startswith("A") else "B",
                    "mode": "clean-only" if dataset_id in ["A2", "A4", "B2", "B4"] else "keep-imputed",
                    "n_rows_total": n_total,
                    "n_rows_train": n_train,
                    "n_rows_test": n_test,
                    "train_start_utc": train_start.isoformat(),
                    "train_end_utc": train_end.isoformat(),
                    "test_start_utc": test_start.isoformat(),
                    "test_end_utc": test_end.isoformat(),
                    "week_index_utc": week_idx,
                }
            )

        # Per-split manifest
        df_manifest = pd.DataFrame(manifest_rows)
        df_manifest.to_csv(split_dir / "dataset_manifest.csv", index=False)

    # Global manifest
    df_all = pd.DataFrame(all_manifests)
    df_all.to_csv(out_dir / "manifest_all_splits.csv", index=False)

    print(f"Wrote datasets to: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=str, default="processed_hourly/master_hourly_extended.csv")
    ap.add_argument("--splits", type=str, default="metrics/split_table.csv")
    ap.add_argument("--config", type=str, default="config.yml")
    ap.add_argument("--out_dir", type=str, default="datasets_by_split")
    args = ap.parse_args()

    cfg = load_frozen_config(args.config)
    build_all(cfg, Path(args.master), Path(args.splits), Path(args.out_dir))


if __name__ == "__main__":
    main()