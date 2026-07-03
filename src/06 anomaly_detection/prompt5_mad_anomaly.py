# prompt5_mad_anomaly.py
# PROMPT 5 — MAD ANOMALY SCORING (RESIDUAL-BASED + RAW CONTROL)
#
# Usage (PowerShell):
#   python .\prompt5_mad_anomaly.py --config .\config.yml --datasets_root .\datasets_by_split --preds_root .\predictions --out .\metrics
#
# Outputs:
#   metrics/anomaly_scores_{dataset_id}_{split_id}.csv
#   metrics/k_sweep_summary.csv

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


# -----------------------------
# Errors / config helpers
# -----------------------------
def _fail(msg: str) -> None:
    raise RuntimeError(msg)


def load_cfg(path: Path) -> Dict[str, Any]:
    if not path.exists():
        _fail(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "config" not in cfg or not isinstance(cfg["config"], dict):
        _fail("config.yml must parse to a dict with top-level key: 'config'")
    return cfg


def cfg_get(cfg: Dict[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            _fail(f"Missing config key: {dotted} (stuck at '{part}')")
    return cur


def enforce_runtime_asserts(cfg: Dict[str, Any]) -> None:
    # pvlib version lock
    expected = str(cfg_get(cfg, "config.solar.library_version"))
    try:
        import pvlib  # type: ignore
    except Exception as e:
        _fail(f"pvlib import failed (Prompt lock requires pvlib installed): {e}")

    installed = getattr(pvlib, "__version__", None)
    if installed is None:
        _fail("pvlib.__version__ not found.")
    if str(installed) != expected:
        _fail(f"pvlib version mismatch: installed {installed} != expected {expected}")

    # PV regime edge consistency lock (also enforced earlier, keep here too)
    edges = cfg_get(cfg, "config.frozen_constants.pv_regime_edges_w")
    if not isinstance(edges, (list, tuple)) or len(edges) < 3:
        _fail("config.frozen_constants.pv_regime_edges_w must be a list/tuple with >=3 elements")

    pv_q33 = float(cfg_get(cfg, "config.frozen_constants.pv_q33_w"))
    pv_q66 = float(cfg_get(cfg, "config.frozen_constants.pv_q66_w"))
    if float(edges[1]) != pv_q33 or float(edges[2]) != pv_q66:
        _fail("pv_q33_w / pv_q66_w must match pv_regime_edges_w[1] / [2] exactly")


def get_feature_cols(cfg: Dict[str, Any], dataset_id: str) -> List[str]:
    cols = cfg_get(cfg, f"config.features.feature_cols_by_dataset.{dataset_id}")
    if not isinstance(cols, list) or not cols or not all(isinstance(x, str) for x in cols):
        _fail(
            f"Frozen feature list missing/invalid for dataset {dataset_id} "
            f"at config.features.feature_cols_by_dataset.{dataset_id}"
        )
    return cols


def get_rf_params(cfg: Dict[str, Any], dataset_id: str) -> Dict[str, Any]:
    params = cfg_get(cfg, f"config.model.rf.hyperparams_by_dataset.{dataset_id}")
    if not isinstance(params, dict) or not params:
        _fail(
            f"RF hyperparams missing/invalid for dataset {dataset_id} "
            f"at config.model.rf.hyperparams_by_dataset.{dataset_id}"
        )
    return dict(params)


# -----------------------------
# Time helpers (UTC weeks keyed to ts_target_utc)
# -----------------------------
def _to_utc(ts: Any) -> pd.DatetimeIndex:
    return pd.to_datetime(ts, utc=True, errors="raise")


def week_start_utc(ts_utc: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Monday 00:00 UTC
    day = ts_utc.floor("D")
    return day - pd.to_timedelta(day.dayofweek, unit="D")


def build_full_week_starts(train_start: pd.Timestamp, train_end: pd.Timestamp) -> List[pd.Timestamp]:
    """
    Return Monday week_start_utc values such that the full week [ws, ws+7d) is inside [train_start, train_end).
    """
    if train_start.tzinfo is None or train_end.tzinfo is None:
        _fail("train_start/train_end must be tz-aware UTC.")
    if train_end <= train_start:
        _fail("train_end must be after train_start.")

    start_ws = week_start_utc(pd.DatetimeIndex([train_start]))[0]
    end_ws = week_start_utc(pd.DatetimeIndex([train_end - pd.Timedelta(seconds=1)]))[0]

    candidates = pd.date_range(start=start_ws, end=end_ws, freq="7D", tz="UTC").to_list()

    full: List[pd.Timestamp] = []
    for ws in candidates:
        we = ws + pd.Timedelta(days=7)
        if ws >= train_start and we <= train_end:
            full.append(pd.Timestamp(ws))
    return full


# -----------------------------
# MAD + calibration
# -----------------------------
@dataclass(frozen=True)
class CalStats:
    median_r: float
    mad_r: float
    median_y: float
    mad_y: float
    n_resid: int
    n_y: int


def mad_no_scale(x: np.ndarray, eps: float) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        _fail("MAD requested on empty / all-nonfinite array.")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    mad = max(mad, float(eps))
    return med, mad


def _require_cols(df: pd.DataFrame, cols: Iterable[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        _fail(f"Missing required columns ({ctx}): {missing}")


def _fit_predict_model(
    model_name: str,
    dataset_id: str,
    cfg: Dict[str, Any],
    fit_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> np.ndarray:
    """
    Fit (if needed) on fit_df, predict pred_df.
    Residual definition everywhere: y_true - yhat.
    """
    y_col = "y_demand_t_plus_1"

    if model_name == "persistence":
        # yhat_{t+1} = y_t, where y_t = demand_w at current hour
        _require_cols(pred_df, ["demand_w"], "persistence requires demand_w")
        return pred_df["demand_w"].to_numpy(dtype=float)

    feat_cols = get_feature_cols(cfg, dataset_id)
    _require_cols(fit_df, feat_cols + [y_col], f"fit_df for {model_name}")
    _require_cols(pred_df, feat_cols, f"pred_df for {model_name}")

    X_fit = fit_df[feat_cols].to_numpy(dtype=float)
    y_fit = fit_df[y_col].to_numpy(dtype=float)
    X_pred = pred_df[feat_cols].to_numpy(dtype=float)

    # Prompt 2 should have removed any NaNs from required feature cols
    if np.isnan(X_fit).any() or np.isnan(X_pred).any() or np.isnan(y_fit).any():
        _fail(f"NaNs in required columns for model={model_name} dataset={dataset_id}. Prompt2 should have dropped them.")

    if model_name == "mlr":
        model = LinearRegression()
        model.fit(X_fit, y_fit)
        return model.predict(X_pred)

    if model_name == "rf":
        rf_params = get_rf_params(cfg, dataset_id)
        model = RandomForestRegressor(**rf_params)
        model.fit(X_fit, y_fit)
        return model.predict(X_pred)

    _fail(f"Unknown model_name: {model_name}")
    raise AssertionError


def build_calibration_stats(
    dataset_df: pd.DataFrame,
    dataset_id: str,
    cfg: Dict[str, Any],
    model_name: str,
    eps: float,
    n_cal_weeks: int,
    block_size_weeks: int,
) -> CalStats:
    """
    Calibration residual set (blocked):
      - use training rows only (is_train_split==1)
      - define weeks using ts_target_utc and Monday 00:00 UTC boundaries
      - take last N_calibration_weeks of training as week blocks (size=calibration_block_size_weeks)
      - for each block: fit on data strictly before block start, predict the block, collect residuals
    """
    _require_cols(dataset_df, ["ts_target_utc", "is_train_split", "y_demand_t_plus_1", "demand_w"], "dataset base")

    df = dataset_df.copy()
    df["ts_target_utc"] = _to_utc(df["ts_target_utc"])
    df_train = df[df["is_train_split"].astype(int) == 1].copy()
    if df_train.empty:
        _fail(f"No train rows (is_train_split==1) in dataset {dataset_id}. Cannot calibrate.")

    # Use split interval if present (it should be)
    if "train_start_utc" in df_train.columns and "train_end_utc" in df_train.columns:
        train_start = pd.to_datetime(df_train["train_start_utc"].iloc[0], utc=True, errors="raise")
        train_end = pd.to_datetime(df_train["train_end_utc"].iloc[0], utc=True, errors="raise")
    else:
        train_start = df_train["ts_target_utc"].min()
        train_end = df_train["ts_target_utc"].max() + pd.Timedelta(hours=1)

    # full-week starts completely inside training interval
    full_week_starts = build_full_week_starts(train_start, train_end)
    if not full_week_starts:
        _fail(f"Training window too short for full-week calibration: train_start={train_start} train_end={train_end}")

    # select last N calibration weeks
    n_take = min(int(n_cal_weeks), len(full_week_starts))
    cal_week_starts = full_week_starts[-n_take:]

    block_delta = pd.Timedelta(days=7 * int(block_size_weeks))

    resid_all: List[float] = []
    y_all: List[float] = []
    used_blocks = 0

    for ws in cal_week_starts:
        we = ws + block_delta

        block_df = df_train[(df_train["ts_target_utc"] >= ws) & (df_train["ts_target_utc"] < we)].copy()
        if block_df.empty:
            continue

        fit_df = df_train[df_train["ts_target_utc"] < ws].copy()
        if model_name in ("rf", "mlr") and fit_df.empty:
            # no past data to fit -> skip this block
            continue

        yhat = _fit_predict_model(model_name, dataset_id, cfg, fit_df, block_df)
        y_true = block_df["y_demand_t_plus_1"].to_numpy(dtype=float)

        resid = y_true - yhat
        resid_all.extend(resid.tolist())
        y_all.extend(y_true.tolist())
        used_blocks += 1

    if used_blocks == 0 or len(resid_all) == 0:
        _fail(
            f"Calibration produced no residuals for model={model_name} dataset={dataset_id}. "
            f"(Possibly too little history before the earliest calibration week.)"
        )

    median_r, mad_r = mad_no_scale(np.array(resid_all, dtype=float), eps=eps)
    median_y, mad_y = mad_no_scale(np.array(y_all, dtype=float), eps=eps)

    return CalStats(
        median_r=median_r,
        mad_r=mad_r,
        median_y=median_y,
        mad_y=mad_y,
        n_resid=int(len(resid_all)),
        n_y=int(len(y_all)),
    )


def score_one_dataset_split(
    cfg: Dict[str, Any],
    dataset_csv: Path,
    preds_csv: Path,
    out_dir: Path,
    eps: float,
    n_cal_weeks: int,
    block_size_weeks: int,
) -> pd.DataFrame:
    dataset_id = dataset_csv.stem
    split_id = dataset_csv.parent.name

    df_d = pd.read_csv(dataset_csv)
    df_p = pd.read_csv(preds_csv)

    _require_cols(
        df_p,
        ["model_name", "ts_utc", "ts_target_utc", "is_test_split", "y_true_t_plus_1", "yhat_t_plus_1", "residual_t_plus_1"],
        f"preds: {preds_csv.name}",
    )

    # scoring is on test rows
    df_p = df_p[df_p["is_test_split"].astype(int) == 1].copy()
    if df_p.empty:
        _fail(f"No test rows found in preds file: {preds_csv}")

    df_p["ts_utc"] = _to_utc(df_p["ts_utc"])
    df_p["ts_target_utc"] = _to_utc(df_p["ts_target_utc"])

    parts: List[pd.DataFrame] = []
    for model_name in sorted(df_p["model_name"].unique()):
        g = df_p[df_p["model_name"] == model_name].copy()

        cal = build_calibration_stats(
            dataset_df=df_d,
            dataset_id=dataset_id,
            cfg=cfg,
            model_name=model_name,
            eps=eps,
            n_cal_weeks=n_cal_weeks,
            block_size_weeks=block_size_weeks,
        )

        resid = pd.to_numeric(g["residual_t_plus_1"], errors="coerce").to_numpy(dtype=float)
        y_true = pd.to_numeric(g["y_true_t_plus_1"], errors="coerce").to_numpy(dtype=float)
        if np.isnan(resid).any() or np.isnan(y_true).any():
            _fail(f"NaNs in residual/y_true in preds for {dataset_id}/{split_id}/{model_name}")

        g["median_r_cal"] = cal.median_r
        g["mad_r_cal"] = cal.mad_r
        g["median_y_cal"] = cal.median_y
        g["mad_y_cal"] = cal.mad_y
        g["n_resid_cal"] = cal.n_resid
        g["n_y_cal"] = cal.n_y

        # anomaly_score = |residual_test − median_r_cal| / MAD_r_cal
        g["anomaly_score"] = np.abs(resid - cal.median_r) / cal.mad_r

        # raw control: z_raw = |y_true_test − median_y_cal| / MAD_y_cal
        g["z_raw"] = np.abs(y_true - cal.median_y) / cal.mad_y

        g["dataset_id"] = dataset_id
        g["split_id"] = split_id

        parts.append(g)

    scored = pd.concat(parts, axis=0, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"anomaly_scores_{dataset_id}_{split_id}.csv"
    scored.to_csv(out_path, index=False)
    return scored


def build_k_sweep_summary(all_scored: List[pd.DataFrame], out_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    Ks = list(range(3, 16))

    for df in all_scored:
        for (dataset_id, split_id, model_name), g in df.groupby(["dataset_id", "split_id", "model_name"], dropna=False):
            n = int(len(g))
            if n == 0:
                continue

            a = pd.to_numeric(g["anomaly_score"], errors="coerce").to_numpy(dtype=float)
            z = pd.to_numeric(g["z_raw"], errors="coerce").to_numpy(dtype=float)

            for K in Ks:
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "split_id": split_id,
                        "model_name": model_name,
                        "K": int(K),
                        "n_test": n,
                        "n_anom_resid": int(np.sum(a > K)),
                        "rate_anom_resid": float(np.mean(a > K)),
                        "n_anom_raw": int(np.sum(z > K)),
                        "rate_anom_raw": float(np.mean(z > K)),
                    }
                )

    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_df


# -----------------------------
# File discovery
# -----------------------------
DATASET_IDS = {"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"}


def find_dataset_files(root: Path) -> List[Path]:
    if not root.exists():
        _fail(f"datasets_root not found: {root}")
    files = sorted(
        [
            p
            for p in root.rglob("*.csv")
            if p.parent.name.startswith("roll_") and p.stem in DATASET_IDS
        ]
    )
    return files


def expected_preds_path(preds_root: Path, dataset_id: str, split_id: str) -> Path:
    return preds_root / f"preds_{dataset_id}_{split_id}.csv"


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yml")
    ap.add_argument("--datasets_root", type=str, default="datasets_by_split")
    ap.add_argument("--preds_root", type=str, default="predictions")
    ap.add_argument("--out", type=str, default="metrics")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    enforce_runtime_asserts(cfg)

    eps = float(cfg_get(cfg, "config.frozen_constants.eps"))
    n_cal_weeks = int(cfg_get(cfg, "config.frozen_constants.N_calibration_weeks"))
    block_size_weeks = int(cfg_get(cfg, "config.frozen_constants.calibration_block_size_weeks"))

    if n_cal_weeks <= 0:
        _fail("N_calibration_weeks must be > 0")
    if block_size_weeks <= 0:
        _fail("calibration_block_size_weeks must be > 0")

    datasets_root = Path(args.datasets_root)
    preds_root = Path(args.preds_root)
    out_dir = Path(args.out)

    dataset_files = find_dataset_files(datasets_root)
    if not dataset_files:
        _fail(f"No dataset CSVs found under {datasets_root}. Expected datasets_by_split/roll_xx/A1.csv etc.")

    all_scored: List[pd.DataFrame] = []
    skipped = 0

    for dataset_csv in dataset_files:
        dataset_id = dataset_csv.stem
        split_id = dataset_csv.parent.name
        preds_csv = expected_preds_path(preds_root, dataset_id, split_id)

        if not preds_csv.exists():
            print(f"[WARN] Missing preds file, skipping: {preds_csv}")
            skipped += 1
            continue

        scored = score_one_dataset_split(
            cfg=cfg,
            dataset_csv=dataset_csv,
            preds_csv=preds_csv,
            out_dir=out_dir,
            eps=eps,
            n_cal_weeks=n_cal_weeks,
            block_size_weeks=block_size_weeks,
        )
        all_scored.append(scored)
        print(f"Wrote {out_dir / f'anomaly_scores_{dataset_id}_{split_id}.csv'} ({len(scored)} rows)")

    if not all_scored:
        _fail("No anomaly score files were produced (all datasets skipped?).")

    sweep_path = out_dir / "k_sweep_summary.csv"
    sweep_df = build_k_sweep_summary(all_scored, sweep_path)
    print(f"Wrote {sweep_path} ({len(sweep_df)} rows). Skipped datasets: {skipped}")


if __name__ == "__main__":
    main()