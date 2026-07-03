# prompt6_error_slicing.py
# PROMPT 6 — REGIME-BASED ERROR SLICING (NATIVE + INTERSECTION FAIRNESS; PV + POOL + TEMPERATURE)
#
# Usage (PowerShell):
#   python .\prompt6_error_slicing.py --config .\config.yml --preds_root .\predictions --out .\metrics
#
# Outputs:
#   metrics/error_slices_native.csv
#   metrics/error_slices_intersection.csv

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Prompt 0 plumbing (your existing modules)
from config_loader import load_frozen_config, cfg_get
from asserts import enforce_frozen_runtime_asserts


DATASET_IDS = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
SPLIT_IDS = ["roll_01", "roll_02", "roll_03", "roll_04"]

REQUIRED_COLS = [
    # Identity + membership
    "dataset_id", "split_id", "model_name",
    "ts_utc", "ts_target_utc",
    "is_train_split", "is_test_split",
    # Truth/prediction
    "y_true_t_plus_1", "yhat_t_plus_1",
    # Regimes / context
    "is_daylight", "pv_regime",
    "pool_on", "pool_switch",
    "temp_bin", "flag_grid_exporting",
    # Missingness / provenance
    "imputed_any", "imputed_count",
    "flag_demand_imputed", "flag_pv_imputed", "flag_pool_imputed", "flag_temp_imputed",
    "flag_demand_imputed_new", "flag_pv_imputed_new", "flag_pool_imputed_new", "flag_temp_imputed_new",
    "gaplen_demand_h", "gaplen_pv_h", "gaplen_pool_h", "gaplen_temp_h",
]


def _fail(msg: str) -> None:
    raise RuntimeError(msg)


def _require_cols(df: pd.DataFrame, cols: List[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        _fail(f"Missing required columns ({context}): {missing}")


def _as_int01_no_nan(series: pd.Series, name: str) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy()
    if np.isnan(arr).any():
        _fail(f"Column {name} contains NaN; expected int 0/1 with no NaNs.")
    arr_i = arr.astype(int)
    if not np.all((arr_i == 0) | (arr_i == 1)):
        bad = np.unique(arr_i[~((arr_i == 0) | (arr_i == 1))])
        _fail(f"Column {name} has values outside {{0,1}} (examples: {bad[:10]})")
    return arr_i


def _enforce_aliasing(df: pd.DataFrame) -> None:
    # Enforce schema aliasing consistency:
    # flag_*_imputed == flag_*_imputed_new for every row; int 0/1; no NaNs.
    pairs = [
        ("flag_demand_imputed", "flag_demand_imputed_new"),
        ("flag_pv_imputed", "flag_pv_imputed_new"),
        ("flag_pool_imputed", "flag_pool_imputed_new"),
        ("flag_temp_imputed", "flag_temp_imputed_new"),
    ]
    for a, b in pairs:
        aa = _as_int01_no_nan(df[a], a)
        bb = _as_int01_no_nan(df[b], b)
        if not np.array_equal(aa, bb):
            idx = int(np.where(aa != bb)[0][0])
            _fail(f"Aliasing mismatch: {a} != {b} at row index {idx}")


def _ensure_no_nan_truth_pred(df: pd.DataFrame) -> None:
    y_true = pd.to_numeric(df["y_true_t_plus_1"], errors="coerce")
    y_hat = pd.to_numeric(df["yhat_t_plus_1"], errors="coerce")
    if y_true.isna().any():
        idx = int(np.where(y_true.isna().to_numpy())[0][0])
        _fail(f"NaN in y_true_t_plus_1 at row index {idx} (Prompt 6 forbids altering values; fix upstream).")
    if y_hat.isna().any():
        idx = int(np.where(y_hat.isna().to_numpy())[0][0])
        _fail(f"NaN in yhat_t_plus_1 at row index {idx} (Prompt 6 forbids altering values; fix upstream).")


def _compute_gap_bin(x: pd.Series, edges: List[float]) -> np.ndarray:
    # digitize lock: right=False, then -1, clamp to [0,n_bins-1], missing -> -1
    x_num = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(x_num), -1, dtype=int)
    mask = ~np.isnan(x_num)
    if mask.any():
        bins = np.digitize(x_num[mask], edges, right=False) - 1
        n_bins = max(len(edges) - 1, 0)
        if n_bins <= 0:
            _fail("config.frozen_constants.gaplen_bin_edges_h must define at least 2 edges (>=1 bin).")
        bins = np.clip(bins, 0, n_bins - 1)
        out[mask] = bins.astype(int)
    return out


def _metrics_for_slice(df: pd.DataFrame, eps: float) -> Tuple[int, float, float, float]:
    # Metric definitions:
    # nrows = count
    # MAE  = mean(abs_err)
    # RMSE = sqrt(mean(sq_err))
    # MAPE = mean(abs_err / max(abs(y_true), eps))*100 (elementwise max)
    n = int(len(df))
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan")

    y_true = pd.to_numeric(df["y_true_t_plus_1"], errors="coerce").to_numpy(dtype=float)
    y_hat = pd.to_numeric(df["yhat_t_plus_1"], errors="coerce").to_numpy(dtype=float)

    err = y_hat - y_true
    abs_err = np.abs(err)
    sq_err = err * err

    mae = float(np.mean(abs_err))
    rmse = float(math.sqrt(np.mean(sq_err)))

    denom = np.maximum(np.abs(y_true), eps)
    mape = float(np.mean(abs_err / denom) * 100.0)

    return n, mae, rmse, mape


def _compute_all_slices(
    df_test: pd.DataFrame,
    dataset_id: str,
    split_id: str,
    model_name: str,
    eps: float,
    gap_edges: List[float],
) -> List[Dict]:
    # Derived columns (deterministic)
    d = df_test.copy()

    # Enforce required columns again on this subset (safe)
    _require_cols(d, REQUIRED_COLS, f"{dataset_id}/{split_id}/{model_name}")

    # Derived gap bins (tokens MUST match)
    d["gaplendemandbin"] = _compute_gap_bin(d["gaplen_demand_h"], gap_edges)
    d["gaplenpvbin"] = _compute_gap_bin(d["gaplen_pv_h"], gap_edges)
    d["gaplenpoolbin"] = _compute_gap_bin(d["gaplen_pool_h"], gap_edges)
    d["gaplentempbin"] = _compute_gap_bin(d["gaplen_temp_h"], gap_edges)

    # Slice dimensions (1D)
    slice_specs = [
        ("overall", None),  # special: ALL
        ("is_daylight", "is_daylight"),
        ("pv_regime", "pv_regime"),
        ("pool_on", "pool_on"),
        ("pool_switch", "pool_switch"),
        ("temp_bin", "temp_bin"),
        ("imputed_any", "imputed_any"),
        ("imputed_count", "imputed_count"),
        ("flag_demand_imputed", "flag_demand_imputed"),
        ("flag_pv_imputed", "flag_pv_imputed"),
        ("flag_pool_imputed", "flag_pool_imputed"),
        ("flag_temp_imputed", "flag_temp_imputed"),
        ("gaplendemandbin", "gaplendemandbin"),
        ("gaplenpvbin", "gaplenpvbin"),
        ("gaplenpoolbin", "gaplenpoolbin"),
        ("gaplentempbin", "gaplentempbin"),
    ]

    rows: List[Dict] = []

    # Overall slice
    n, mae, rmse, mape = _metrics_for_slice(d, eps)
    rows.append({
        "datasetid": dataset_id,
        "splitid": split_id,
        "modelname": model_name,
        "slicename": "overall",
        "slicevalue": "ALL",
        "nrows": n,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    })

    # Other slices
    for slicename, col in slice_specs:
        if col is None:
            continue
        # values present in filtered data (including -1 if present)
        vals = d[col].dropna().unique()
        # Deterministic sorting (lexicographic by string)
        vals_sorted = sorted([str(v) for v in vals])

        for sv in vals_sorted:
            sub = d[d[col].astype(str) == sv]
            n, mae, rmse, mape = _metrics_for_slice(sub, eps)
            rows.append({
                "datasetid": dataset_id,
                "splitid": split_id,
                "modelname": model_name,
                "slicename": slicename,
                "slicevalue": sv,
                "nrows": n,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
            })

    return rows


def _load_preds_file(path: Path, cfg_eps: float, cfg_gap_edges: List[float]) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_cols(df, REQUIRED_COLS, f"preds file: {path.name}")

    # Types + basic normalization
    df["ts_target_utc"] = pd.to_datetime(df["ts_target_utc"], utc=True, errors="raise")
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="raise")

    # Enforce aliasing and no NaNs in truth/pred (on the full file; stricter)
    _enforce_aliasing(df)
    _ensure_no_nan_truth_pred(df)

    # NOTE: derived metrics computed later per subset.
    # Quick sanity: eps must be >0
    if not (isinstance(cfg_eps, (int, float)) and cfg_eps > 0):
        _fail("config.frozen_constants.eps must be a positive number.")
    if not isinstance(cfg_gap_edges, list) or len(cfg_gap_edges) < 2:
        _fail("config.frozen_constants.gaplen_bin_edges_h must be a list with >= 2 edges.")

    return df


def run_native(cfg, preds_root: Path, out_dir: Path) -> pd.DataFrame:
    eps = float(cfg_get(cfg, "config.frozen_constants.eps"))
    gap_edges = list(cfg_get(cfg, "config.frozen_constants.gaplen_bin_edges_h"))

    all_rows: List[Dict] = []

    for split_id in SPLIT_IDS:
        for dataset_id in DATASET_IDS:
            p = preds_root / f"preds_{dataset_id}_{split_id}.csv"
            if not p.exists():
                _fail(f"Missing predictions file: {p}")

            df = _load_preds_file(p, eps, gap_edges)

            # Evaluate errors ONLY on is_test_split == 1
            test_mask = pd.to_numeric(df["is_test_split"], errors="coerce").fillna(0).astype(int) == 1
            df_test = df.loc[test_mask].copy()
            if len(df_test) == 0:
                _fail(f"No test rows found in {p.name} after filtering is_test_split==1")

            # For each model_name in this dataset/split
            for model_name in sorted(df_test["model_name"].astype(str).unique()):
                sub = df_test[df_test["model_name"].astype(str) == model_name].copy()
                rows = _compute_all_slices(sub, dataset_id, split_id, model_name, eps, gap_edges)
                all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows)

    # Sort lexicographically by (datasetid, splitid, modelname, slicename, slicevalue)
    out_df["__sv"] = out_df["slicevalue"].astype(str)
    out_df = out_df.sort_values(["datasetid", "splitid", "modelname", "slicename", "__sv"], kind="mergesort")
    out_df = out_df.drop(columns=["__sv"])

    # Ensure types
    out_df["nrows"] = out_df["nrows"].astype(int)
    for c in ["mae", "rmse", "mape"]:
        out_df[c] = out_df[c].astype(float)

    out_path = out_dir / "error_slices_native.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path.as_posix()}")

    return out_df


def run_intersection(cfg, preds_root: Path, out_dir: Path) -> pd.DataFrame:
    eps = float(cfg_get(cfg, "config.frozen_constants.eps"))
    gap_edges = list(cfg_get(cfg, "config.frozen_constants.gaplen_bin_edges_h"))

    intersection_label = "A1..B4_by_ts_target_utc"
    all_rows: List[Dict] = []

    for split_id in SPLIT_IDS:
        # Load all files for this split once
        loaded: Dict[str, pd.DataFrame] = {}
        for dataset_id in DATASET_IDS:
            p = preds_root / f"preds_{dataset_id}_{split_id}.csv"
            if not p.exists():
                _fail(f"Missing predictions file: {p}")
            loaded[dataset_id] = _load_preds_file(p, eps, gap_edges)

        # Determine model_names available (must exist for all datasets for fair comparison)
        model_sets = []
        for dataset_id in DATASET_IDS:
            df = loaded[dataset_id]
            test_mask = pd.to_numeric(df["is_test_split"], errors="coerce").fillna(0).astype(int) == 1
            model_sets.append(set(df.loc[test_mask, "model_name"].astype(str).unique()))
        common_models = set.intersection(*model_sets) if model_sets else set()
        if not common_models:
            _fail(f"No common model_name across datasets for split {split_id} after is_test_split==1 filter.")

        for model_name in sorted(common_models):
            # Build alignment set I = intersection over ALL dataset_ids of S(dataset_id)
            sets = []
            for dataset_id in DATASET_IDS:
                df = loaded[dataset_id]
                test_mask = (pd.to_numeric(df["is_test_split"], errors="coerce").fillna(0).astype(int) == 1)
                df_tm = df.loc[test_mask & (df["model_name"].astype(str) == model_name)].copy()
                if len(df_tm) == 0:
                    _fail(f"Empty test rows for {dataset_id}/{split_id}/{model_name}; cannot form intersection.")
                # use int64 ns for stable set ops
                sets.append(set(df_tm["ts_target_utc"].view("int64").to_numpy()))

            I = set.intersection(*sets)
            if len(I) == 0:
                _fail(f"Intersection set is empty for split {split_id}, model {model_name}. This indicates upstream membership/schema bug.")
            intersectionnrows = int(len(I))

            # Now compute slices per dataset on aligned rows
            for dataset_id in DATASET_IDS:
                df = loaded[dataset_id]
                test_mask = (pd.to_numeric(df["is_test_split"], errors="coerce").fillna(0).astype(int) == 1)
                sub = df.loc[test_mask & (df["model_name"].astype(str) == model_name)].copy()
                sub = sub[sub["ts_target_utc"].view("int64").isin(I)].copy()

                if len(sub) == 0:
                    _fail(f"After intersection filtering, got 0 rows for {dataset_id}/{split_id}/{model_name} (should never happen).")

                rows = _compute_all_slices(sub, dataset_id, split_id, model_name, eps, gap_edges)
                for r in rows:
                    r["intersection"] = intersection_label
                    r["intersectionnrows"] = intersectionnrows
                all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows)

    # Sort lexicographically by (datasetid, splitid, modelname, slicename, slicevalue)
    out_df["__sv"] = out_df["slicevalue"].astype(str)
    out_df = out_df.sort_values(["datasetid", "splitid", "modelname", "slicename", "__sv"], kind="mergesort")
    out_df = out_df.drop(columns=["__sv"])

    # Ensure types
    out_df["nrows"] = out_df["nrows"].astype(int)
    out_df["intersectionnrows"] = out_df["intersectionnrows"].astype(int)
    for c in ["mae", "rmse", "mape"]:
        out_df[c] = out_df[c].astype(float)

    out_path = out_dir / "error_slices_intersection.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path.as_posix()}")

    return out_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yml")
    ap.add_argument("--preds_root", type=str, default="predictions")
    ap.add_argument("--out", type=str, default="metrics")
    args = ap.parse_args()

    cfg = load_frozen_config(args.config)
    # Keep the global lock style consistent with your pipeline
    enforce_frozen_runtime_asserts(cfg)

    preds_root = Path(args.preds_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Part 1: native
    _ = run_native(cfg, preds_root, out_dir)
    # Part 2: intersection
    _ = run_intersection(cfg, preds_root, out_dir)

    print("PROMPT 6 COMPLETE: wrote native + intersection slicing outputs.")


if __name__ == "__main__":
    main()
