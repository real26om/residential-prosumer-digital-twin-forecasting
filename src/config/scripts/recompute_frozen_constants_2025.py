import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

# --- USER SETTINGS ---
PATH = "A1_causal_demand_keep_imputed_BASE.csv"  # your hourly BASE file (has ts_utc, pv_w_raw, pool_w_raw, flags)
LAT = 45.48   # <-- FILL ME (Segrate latitude, decimal degrees)
LON = 9.29    # <-- FILL ME (Segrate longitude, decimal degrees)

# Frozen "SET NOW" constants (from your prereg)
pv_noise_floor_w = 10.0
pv_daylight_margin_w = 10.0
night_altitude_threshold_deg = 0.0  # sun altitude <= 0 => night

pool_fallback_threshold_w = 100.0
pool_min_separation_w = 200.0
kmeans_random_state = 0

# Solar hour classification timestamp: hour midpoint
HOUR_MIDPOINT_MINUTES = 30
# ---------------------

def percentile(x, q):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        raise ValueError(f"Cannot compute percentile {q}: empty array.")
    return float(np.nanpercentile(x, q))

def main():
    # Load
    df = pd.read_csv(PATH, parse_dates=["ts_utc"])

    # 2025 training only
    train_2025 = df[(df["is_train"] == 1) & (df["ts_utc"].dt.year == 2025)].copy()

    # ---------- SOLAR DAY/NIGHT (hour midpoint) ----------
    try:
        import pvlib
    except ImportError as e:
        raise SystemExit("Missing pvlib. Install with: pip install pvlib") from e

    ts_mid = train_2025["ts_utc"] + pd.to_timedelta(HOUR_MIDPOINT_MINUTES, unit="m")
    # pvlib wants tz-aware timestamps; your ts_utc has +00:00 so this should already be tz-aware.
    solpos = pvlib.solarposition.get_solarposition(time=ts_mid, latitude=LAT, longitude=LON)
    # Use apparent elevation (deg). Night if <= threshold.
    sun_alt_deg = solpos["apparent_elevation"].to_numpy(dtype=float)
    is_night = sun_alt_deg <= night_altitude_threshold_deg
    is_daylight = ~is_night

    # ---------- PV: clean hourly using new rules ----------
    # Observed PV only (pre-imputation): use raw hourly pv plus flag
    pv_raw = train_2025["pv_w_raw"].to_numpy(dtype=float)
    pv_missing = train_2025["flag_pv_missing"].to_numpy(dtype=int) == 1

    # Start from raw hourly values; clamp negatives to 0
    pv = np.where(np.isnan(pv_raw), np.nan, np.maximum(pv_raw, 0.0))

    # If raw PV is missing, keep NaN (we are computing constants on observed only)
    pv = np.where(pv_missing, np.nan, pv)

    # Solar night forcing: night hours -> 0 (even if small positive noise)
    pv = np.where(is_night & ~np.isnan(pv), 0.0, pv)

    # Day-only noise floor: if 0 < pv < pv_noise_floor_w, clip to 0
    pv = np.where(is_daylight & (pv > 0.0) & (pv < pv_noise_floor_w), 0.0, pv)

    # ---------- F) pv_daylight_threshold_w from solar-night hours ----------
    pv_night = pv[is_night]  # already forced to 0 when observed
    pv_night_p95_w = percentile(pv_night, 95)

    pv_daylight_threshold_w = max(pv_noise_floor_w, pv_night_p95_w + pv_daylight_margin_w)

    # ---------- G) pv_q33_w / pv_q66_w from daylight_reference ----------
    daylight_reference = pv[is_daylight & (pv > pv_daylight_threshold_w)]
    if np.asarray(daylight_reference).size == 0:
        raise ValueError(
            "daylight_reference is empty. This can happen if pv_daylight_threshold_w is too high "
            "or PV values are mostly small. Consider reviewing pv_noise_floor_w/pv_daylight_margin_w."
        )

    pv_q33_w = percentile(daylight_reference, 33)
    pv_q66_w = percentile(daylight_reference, 66)
    pv_regime_edges_w = [pv_daylight_threshold_w, pv_q33_w, pv_q66_w]

    # ---------- H) pool_on_threshold_w via k-means midpoint + fallback ----------
    pool_raw = train_2025["pool_w_raw"].to_numpy(dtype=float)
    pool_missing = train_2025["flag_pool_missing"].to_numpy(dtype=int) == 1

    pool_obs = pool_raw[~pool_missing]
    pool_obs = pool_obs[~np.isnan(pool_obs)]
    # optional physical clamp
    pool_obs = np.maximum(pool_obs, 0.0)

    if pool_obs.size < 20:
        pool_on_threshold_w = pool_fallback_threshold_w
        pool_debug = {"reason": "too_few_samples", "n": int(pool_obs.size)}
    else:
        try:
            km = KMeans(n_clusters=2, random_state=kmeans_random_state, n_init="auto").fit(pool_obs.reshape(-1, 1))
            c_low, c_high = np.sort(km.cluster_centers_.ravel())
            sep = float(c_high - c_low)
            candidate = float((c_low + c_high) / 2.0)

            if not np.isfinite(candidate) or sep < pool_min_separation_w:
                pool_on_threshold_w = pool_fallback_threshold_w
                pool_debug = {"reason": "fallback_sep_or_nan", "c_low": float(c_low), "c_high": float(c_high), "sep": sep}
            else:
                pool_on_threshold_w = candidate
                pool_debug = {"reason": "kmeans_ok", "c_low": float(c_low), "c_high": float(c_high), "sep": sep}
        except Exception as e:
            pool_on_threshold_w = pool_fallback_threshold_w
            pool_debug = {"reason": "kmeans_failed", "error": str(e)}

    # ---------- PRINT (copy/paste into YAML) ----------
    print("### FROZEN CONSTANTS (computed once on 2025 training; paste into config) ###")
    print(f"pv_daylight_threshold_w: {pv_daylight_threshold_w}")
    print(f"pv_q33_w: {pv_q33_w}")
    print(f"pv_q66_w: {pv_q66_w}")
    print(f"pv_regime_edges_w: [{pv_regime_edges_w[0]}, {pv_regime_edges_w[1]}, {pv_regime_edges_w[2]}]")
    print(f"pool_on_threshold_w: {pool_on_threshold_w}")
    print("")
    print("### DEBUG ###")
    print(f"pv_night_p95_w: {pv_night_p95_w}")
    print(f"pool_debug: {pool_debug}")

if __name__ == "__main__":
    main()
