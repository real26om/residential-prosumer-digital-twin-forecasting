# metrics_prompt4_mae_rmse.py
#
# Computes MAE + RMSE for every Prompt 4 prediction file:
#   predictions/preds_{dataset_id}_{split_id}.csv
#
# Outputs one CSV with metrics for:
#   - each dataset_id x split_id (32 combos)
#   - each model_name (rf / persistence / mlr)
#   - each segment: all / night / day
#
# Uses ONLY test rows (is_test_split == 1).
#
# PowerShell example:
#   python .\metrics_prompt4_mae_rmse.py --preds_dir .\predictions --out .\metrics\mae_rmse_by_segment.csv

from __future__ import annotations

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd


def _fail(msg: str) -> None:
    raise RuntimeError(msg)


def _to_int01(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy()
    if np.isnan(arr).any():
        # allow NaN -> treat as missing; caller can decide what to do
        return arr
    arr_i = arr.astype(int)
    return arr_i


def _infer_ids_from_filename(path: Path) -> tuple[str | None, str | None]:
    # expects preds_{dataset_id}_{split_id}.csv
    m = re.match(r"^preds_([A-Z]\d)_(roll_\d\d)\.csv$", path.name)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _pick_col(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    _fail(f"Missing required column for {what}. Tried: {candidates}. Present cols (first 50): {list(df.columns)[:50]}")


def _mae_rmse(y_true: np.ndarray, y_hat: np.ndarray) -> tuple[float, float, int]:
    mask = np.isfinite(y_true) & np.isfinite(y_hat)
    n = int(mask.sum())
    if n == 0:
        return (np.nan, np.nan, 0)
    err = y_true[mask] - y_hat[mask]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return (mae, rmse, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_dir", type=str, default="predictions", help="Folder containing preds_*.csv")
    ap.add_argument("--out", type=str, default="metrics/mae_rmse_by_segment.csv", help="Output CSV path")
    args = ap.parse_args()

    preds_dir = Path(args.preds_dir)
    out_path = Path(args.out)

    if not preds_dir.exists():
        _fail(f"preds_dir does not exist: {preds_dir.resolve()}")

    files = sorted(preds_dir.glob("preds_*.csv"))
    if not files:
        _fail(f"No files found like preds_*.csv in: {preds_dir.resolve()}")

    rows: list[dict] = []

    for fp in files:
        df = pd.read_csv(fp)

        # dataset_id / split_id: prefer columns; fallback to filename
        ds_from_name, roll_from_name = _infer_ids_from_filename(fp)

        dataset_id = df["dataset_id"].iloc[0] if "dataset_id" in df.columns else ds_from_name
        split_id = df["split_id"].iloc[0] if "split_id" in df.columns else roll_from_name
        if dataset_id is None or split_id is None:
            _fail(f"Could not infer dataset_id/split_id for file: {fp.name}. "
                  f"Either include columns dataset_id & split_id, or use name preds_A1_roll_01.csv")

        # Required columns (robust)
        model_col = _pick_col(df, ["model_name"], "model_name")
        y_true_col = _pick_col(df, ["y_true_t_plus_1", "y_true", "y_demand_t_plus_1"], "y_true")
        y_hat_col = _pick_col(df, ["yhat_t_plus_1", "y_hat", "yhat"], "y_hat")
        is_test_col = _pick_col(df, ["is_test_split"], "is_test_split")

        # Day/night column: prefer is_daylight; fallback to is_daylight_hour
        daylight_col = None
        if "is_daylight" in df.columns:
            daylight_col = "is_daylight"
        elif "is_daylight_hour" in df.columns:
            daylight_col = "is_daylight_hour"
        elif "is_daylight_hour_mid" in df.columns:
            daylight_col = "is_daylight_hour_mid"

        # Filter to test rows only (LOCKED by your Prompt 4)
        is_test = pd.to_numeric(df[is_test_col], errors="coerce").fillna(0).astype(int)
        df_test = df.loc[is_test == 1].copy()

        if len(df_test) == 0:
            # still write a row so you notice it
            rows.append({
                "dataset_id": dataset_id,
                "split_id": split_id,
                "model_name": "ALL",
                "segment": "all",
                "n": 0,
                "mae": np.nan,
                "rmse": np.nan,
                "source_file": fp.name,
                "note": "NO TEST ROWS (is_test_split==1)",
            })
            continue

        # Coerce y columns
        y_true_all = pd.to_numeric(df_test[y_true_col], errors="coerce").to_numpy(dtype=float)
        y_hat_all = pd.to_numeric(df_test[y_hat_col], errors="coerce").to_numpy(dtype=float)

        # Segment masks
        if daylight_col is not None:
            is_daylight = pd.to_numeric(df_test[daylight_col], errors="coerce").fillna(0).astype(int).to_numpy()
            mask_day = (is_daylight == 1)
            mask_night = (is_daylight == 0)
        else:
            # If missing, we still compute "all"; day/night become NaN with n=0
            mask_day = None
            mask_night = None

        # Compute per model
        for model_name, g in df_test.groupby(model_col):
            y_true = pd.to_numeric(g[y_true_col], errors="coerce").to_numpy(dtype=float)
            y_hat = pd.to_numeric(g[y_hat_col], errors="coerce").to_numpy(dtype=float)

            mae, rmse, n = _mae_rmse(y_true, y_hat)
            rows.append({
                "dataset_id": dataset_id,
                "split_id": split_id,
                "model_name": str(model_name),
                "segment": "all",
                "n": n,
                "mae": mae,
                "rmse": rmse,
                "source_file": fp.name,
            })

            if daylight_col is not None:
                is_day = pd.to_numeric(g[daylight_col], errors="coerce").fillna(0).astype(int).to_numpy()
                # day
                mae_d, rmse_d, n_d = _mae_rmse(y_true[is_day == 1], y_hat[is_day == 1])
                rows.append({
                    "dataset_id": dataset_id,
                    "split_id": split_id,
                    "model_name": str(model_name),
                    "segment": "day",
                    "n": n_d,
                    "mae": mae_d,
                    "rmse": rmse_d,
                    "source_file": fp.name,
                })
                # night
                mae_n, rmse_n, n_n = _mae_rmse(y_true[is_day == 0], y_hat[is_day == 0])
                rows.append({
                    "dataset_id": dataset_id,
                    "split_id": split_id,
                    "model_name": str(model_name),
                    "segment": "night",
                    "n": n_n,
                    "mae": mae_n,
                    "rmse": rmse_n,
                    "source_file": fp.name,
                })
            else:
                rows.append({
                    "dataset_id": dataset_id,
                    "split_id": split_id,
                    "model_name": str(model_name),
                    "segment": "day",
                    "n": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "source_file": fp.name,
                    "note": "NO is_daylight column in preds file",
                })
                rows.append({
                    "dataset_id": dataset_id,
                    "split_id": split_id,
                    "model_name": str(model_name),
                    "segment": "night",
                    "n": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "source_file": fp.name,
                    "note": "NO is_daylight column in preds file",
                })

    out_df = pd.DataFrame(rows)

    # Sort nicely
    seg_order = pd.Categorical(out_df["segment"], categories=["all", "night", "day"], ordered=True)
    out_df["segment"] = seg_order
    out_df = out_df.sort_values(["dataset_id", "split_id", "model_name", "segment"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Wrote: {out_path.as_posix()}")
    # quick console preview
    preview = out_df.head(18)
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()