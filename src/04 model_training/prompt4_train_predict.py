# prompt4_train_predict.py
# PROMPT 4 — Train RF + baselines; export per-hour predictions (schema locked)
#
# Usage (PowerShell):
#   python .\prompt4_train_predict.py --config .\config.yml --datasets_root .\datasets_by_split --out .\predictions --models_out .\models

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence, Any
from collections.abc import Mapping  # <-- IMPORTANT FIX

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from config_loader import load_frozen_config, cfg_get
from asserts import enforce_frozen_runtime_asserts


DATASET_IDS = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
SPLIT_IDS = ["roll_01", "roll_02", "roll_03", "roll_04"]


def _fail(msg: str) -> None:
    raise RuntimeError(msg)


def _as_utc_datetime(series: pd.Series, colname: str) -> pd.Series:
    dt = pd.to_datetime(series, utc=True, errors="raise")
    if dt.dt.tz is None or str(dt.dt.tz) != "UTC":
        _fail(f"{colname} must be tz-aware UTC.")
    return dt


def _enforce_prompt4_runtime_asserts(cfg: Any) -> None:
    # Includes: pvlib.__version__ == config.solar.library_version
    enforce_frozen_runtime_asserts(cfg)

    # PV edges consistency (cheap + catches config drift)
    edges = cfg_get(cfg, "config.frozen_constants.pv_regime_edges_w")
    if not isinstance(edges, (list, tuple)) or len(edges) < 3:
        _fail("config.frozen_constants.pv_regime_edges_w must be a list/tuple with >=3 elements")
    q33 = float(cfg_get(cfg, "config.frozen_constants.pv_q33_w"))
    q66 = float(cfg_get(cfg, "config.frozen_constants.pv_q66_w"))
    if q33 != float(edges[1]) or q66 != float(edges[2]):
        _fail("Frozen PV regime edges mismatch: pv_q33_w must equal pv_regime_edges_w[1] and pv_q66_w must equal pv_regime_edges_w[2]")


def get_feature_cols(cfg: Any, dataset_id: str) -> list[str]:
    key = f"config.features.feature_cols_by_dataset.{dataset_id}"
    cols = cfg_get(cfg, key)

    # config_loader deep-freezes YAML lists into tuples => accept list OR tuple
    if not isinstance(cols, (list, tuple)) or len(cols) == 0:
        _fail(f"Frozen feature list missing/invalid for dataset {dataset_id} at {key}")

    out: list[str] = []
    for c in cols:
        if not isinstance(c, str):
            _fail(f"Non-string feature name in {key}: {c!r}")
        out.append(c)
    return out


def get_rf_params(cfg: Any, dataset_id: str) -> dict:
    # <-- IMPORTANT FIX: accept MappingProxyType (Mapping), not only dict
    key = f"config.model.rf.hyperparams_by_dataset.{dataset_id}"
    params = cfg_get(cfg, key)
    if not isinstance(params, Mapping) or len(params) == 0:
        _fail(f"RF hyperparams missing/invalid for dataset {dataset_id} at {key}")
    return dict(params)


def _require_cols(df: pd.DataFrame, cols: Sequence[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        _fail(f"Missing required columns ({context}): {missing}")


def _assert_no_nans(df: pd.DataFrame, cols: Sequence[str], context: str) -> None:
    bad = [c for c in cols if df[c].isna().any()]
    if bad:
        c0 = bad[0]
        idx = int(df[df[c0].isna()].index[0])
        _fail(f"NaNs found in required columns ({context}): {bad}. First NaN at row index {idx} col {c0}")


def train_and_predict_one(
    cfg: Any,
    dataset_csv: Path,
    dataset_id: str,
    split_id: str,
    out_dir: Path,
    models_out: Path | None,
) -> Path:
    df = pd.read_csv(dataset_csv, low_memory=False)

    # If dataset_id/split_id columns are missing (allowed), create them deterministically
    if "dataset_id" not in df.columns:
        df["dataset_id"] = dataset_id
    if "split_id" not in df.columns:
        df["split_id"] = split_id

    # Validate ids (after creation)
    if not (df["dataset_id"] == dataset_id).all():
        bad = df.loc[df["dataset_id"] != dataset_id, "dataset_id"].unique()[:10]
        _fail(f"{dataset_csv}: dataset_id values not equal to {dataset_id}. Examples: {bad}")
    if not (df["split_id"] == split_id).all():
        bad = df.loc[df["split_id"] != split_id, "split_id"].unique()[:10]
        _fail(f"{dataset_csv}: split_id values not equal to {split_id}. Examples: {bad}")

    # Required input schema (Prompt 4)
    base_required = [
        "ts_utc", "ts_target_utc",
        "is_train_split", "is_test_split",
        "y_demand_t_plus_1",
        "train_start_utc", "train_end_utc", "test_start_utc", "test_end_utc",
        # regimes
        "is_daylight", "pv_regime", "pool_on", "pool_on_lag_1h", "pool_switch", "temp_bin", "flag_grid_exporting",
        # missingness summaries
        "imputed_any", "imputed_count",
        "gaplen_demand_h", "gaplen_pv_h", "gaplen_pool_h", "gaplen_temp_h",
        # persistence needs demand_w
        "demand_w",
    ]
    _require_cols(df, base_required, f"{dataset_id}/{split_id} base schema")

    # Parse timestamps as UTC
    df["ts_utc"] = _as_utc_datetime(df["ts_utc"], "ts_utc")
    df["ts_target_utc"] = _as_utc_datetime(df["ts_target_utc"], "ts_target_utc")

    # Train/test split
    train_df = df[df["is_train_split"] == 1].copy()
    test_df = df[df["is_test_split"] == 1].copy()

    if len(train_df) == 0:
        _fail(f"{dataset_id}/{split_id}: empty train set (is_train_split==1)")
    if len(test_df) == 0:
        _fail(f"{dataset_id}/{split_id}: empty test set (is_test_split==1)")

    # Frozen features
    feature_cols = get_feature_cols(cfg, dataset_id)
    _require_cols(df, feature_cols, f"{dataset_id}/{split_id} features")
    _assert_no_nans(train_df, feature_cols, f"{dataset_id}/{split_id} train features")
    _assert_no_nans(test_df, feature_cols, f"{dataset_id}/{split_id} test features")

    # y must be present and non-missing (Prompt 2 should have dropped missing y rows)
    if train_df["y_demand_t_plus_1"].isna().any():
        _fail(f"{dataset_id}/{split_id}: y_demand_t_plus_1 contains NaN in train (Prompt 2 violation)")
    if test_df["y_demand_t_plus_1"].isna().any():
        _fail(f"{dataset_id}/{split_id}: y_demand_t_plus_1 contains NaN in test (Prompt 2 violation)")

    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df["y_demand_t_plus_1"].to_numpy(dtype=float)

    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df["y_demand_t_plus_1"].to_numpy(dtype=float)

    # ---------- Models ----------
    # RF (frozen hyperparams)
    rf_params = get_rf_params(cfg, dataset_id)
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train, y_train)
    yhat_rf = rf.predict(X_test)

    # Persistence: yhat_{t+1} = y_t
    yhat_persist = test_df["demand_w"].to_numpy(dtype=float)

    # MLR: same feature set
    mlr = LinearRegression()
    mlr.fit(X_train, y_train)
    yhat_mlr = mlr.predict(X_test)

    # Save models if requested
    if models_out is not None:
        models_out.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf, models_out / f"{dataset_id}_{split_id}_rf.joblib")
        joblib.dump(mlr, models_out / f"{dataset_id}_{split_id}_mlr.joblib")

    # ---------- Output ----------
    def pack(model_name: str, yhat: np.ndarray) -> pd.DataFrame:
        out = test_df.copy()
        out.insert(0, "model_name", model_name)
        out["y_true_t_plus_1"] = out["y_demand_t_plus_1"].astype(float)
        out["yhat_t_plus_1"] = np.asarray(yhat, dtype=float)
        out["residual_t_plus_1"] = out["y_true_t_plus_1"] - out["yhat_t_plus_1"]

        keep_cols = [
            "dataset_id", "split_id", "model_name",
            "ts_utc", "ts_target_utc",
            "is_train_split", "is_test_split",
            "train_start_utc", "train_end_utc", "test_start_utc", "test_end_utc",
            "y_true_t_plus_1", "yhat_t_plus_1", "residual_t_plus_1",
            # regimes
            "is_daylight", "pv_regime", "pool_on", "pool_on_lag_1h", "pool_switch", "temp_bin", "flag_grid_exporting",
            # missingness
            "imputed_any", "imputed_count",
            "gaplen_demand_h", "gaplen_pv_h", "gaplen_pool_h", "gaplen_temp_h",
        ]

        # PV provenance flags if present
        pv_prov = [c for c in out.columns if c.startswith("pv_imputed_")]
        for c in pv_prov:
            if c not in keep_cols:
                keep_cols.append(c)

        # core imputation flags if present
        for c in [
            "flag_demand_imputed", "flag_pv_imputed", "flag_pool_imputed", "flag_temp_imputed",
            "flag_demand_imputed_new", "flag_pv_imputed_new", "flag_pool_imputed_new", "flag_temp_imputed_new",
            "demand_imputed_any", "pool_imputed_any", "temp_imputed_any", "pv_imputed_any",
        ]:
            if c in out.columns and c not in keep_cols:
                keep_cols.append(c)

        # Keep frozen features for debugging/repro
        for c in feature_cols:
            if c not in keep_cols:
                keep_cols.append(c)

        return out[keep_cols]

    out_dir.mkdir(parents=True, exist_ok=True)
    preds = pd.concat(
        [
            pack("rf", yhat_rf),
            pack("persistence", yhat_persist),
            pack("mlr", yhat_mlr),
        ],
        axis=0,
        ignore_index=True,
    )

    out_path = out_dir / f"preds_{dataset_id}_{split_id}.csv"
    preds.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yml")
    ap.add_argument("--datasets_root", type=str, default="datasets_by_split")
    ap.add_argument("--out", type=str, default="predictions")
    ap.add_argument("--models_out", type=str, default=None)
    args = ap.parse_args()

    cfg = load_frozen_config(Path(args.config))
    _enforce_prompt4_runtime_asserts(cfg)

    datasets_root = Path(args.datasets_root)
    out_dir = Path(args.out)
    models_out = Path(args.models_out) if args.models_out else None

    wrote = 0
    for split_id in SPLIT_IDS:
        for dataset_id in DATASET_IDS:
            csv_path = datasets_root / split_id / f"{dataset_id}.csv"
            if not csv_path.exists():
                _fail(f"Missing dataset file: {csv_path}")

            out_path = train_and_predict_one(
                cfg=cfg,
                dataset_csv=csv_path,
                dataset_id=dataset_id,
                split_id=split_id,
                out_dir=out_dir,
                models_out=models_out,
            )
            wrote += 1
            print(f"Wrote {out_path}")

    print(f"DONE: wrote {wrote} prediction files.")
    sys.exit(0)


if __name__ == "__main__":
    main()