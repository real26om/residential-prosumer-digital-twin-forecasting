# prompt8_missingness_sanity.py
# PROMPT 8 — MISSINGNESS SANITY CHECKS
#
# Usage (PowerShell):
#   python .\prompt8_missingness_sanity.py --config .\config.yml --scores_root .\metrics --out .\metrics
#
# Outputs:
#   metrics/missingness_anomaly_sanity.csv
#   metrics/missingness_filtered_comparison.csv

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from config_loader import load_frozen_config, cfg_get
from asserts import enforce_frozen_runtime_asserts


def _fail(msg: str) -> None:
    raise RuntimeError(msg)


def _as_list_edges(x: Any, key: str) -> List[float]:
    """
    Accept list/tuple/np array. Reject scalar/None/str.
    """
    if x is None:
        _fail(f"{key} is missing in config")
    if isinstance(x, (list, tuple, np.ndarray)):
        edges = list(x)
    else:
        _fail(f"{key} must be a list/tuple/array, got {type(x)}")
    if len(edges) < 2:
        _fail(f"{key} must be a list with >=2 elements")
    try:
        return [float(v) for v in edges]
    except Exception as e:
        _fail(f"{key} contains non-numeric values: {e}")
    return []


def _digitize_gaplen(values: pd.Series, edges: List[float]) -> np.ndarray:
    """
    gaplen_bin = digitize(x, edges, right=False) - 1, then clamp to [0, n_bins-1]
    missing -> -1
    """
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.full(arr.shape, -1, dtype=int)
    mask = ~np.isnan(arr)
    if mask.any():
        bins = np.digitize(arr[mask], edges, right=False) - 1
        nb = len(edges) - 1
        bins = np.clip(bins, 0, nb - 1)
        out[mask] = bins.astype(int)
    return out


def _require_cols(df: pd.DataFrame, cols: List[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        _fail(f"Missing required columns ({context}): {missing}")


def _as_int01_no_nan(s: pd.Series, name: str) -> np.ndarray:
    arr = pd.to_numeric(s, errors="coerce").to_numpy()
    if np.isnan(arr).any():
        _fail(f"{name} contains NaN (must be int 0/1, no NaNs)")
    arr_i = arr.astype(int)
    if not np.all((arr_i == 0) | (arr_i == 1)):
        bad = np.unique(arr_i[~((arr_i == 0) | (arr_i == 1))])
        _fail(f"{name} has values outside {{0,1}}: {bad[:10]}")
    return arr_i


def load_all_score_files(scores_root: Path) -> pd.DataFrame:
    files = sorted(scores_root.glob("anomaly_scores_*_*.csv"))
    if not files:
        _fail(f"No anomaly_scores_*.csv files found in {scores_root}")

    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def compute_outputs(cfg: Dict[str, Any], scores_root: Path, out_dir: Path) -> Tuple[Path, Path, int, int]:
    eps = float(cfg_get(cfg, "config.frozen_constants.eps"))

    # Robust: accept the key exactly as Prompt 8 states
    edges_raw = cfg_get(cfg, "config.frozen_constants.gaplen_bin_edges_h")
    edges = _as_list_edges(edges_raw, "config.frozen_constants.gaplen_bin_edges_h")

    df = load_all_score_files(scores_root)

    required = [
        "dataset_id", "split_id", "model_name",
        "ts_target_utc", "is_test_split",
        "anomaly_score", "z_raw",
        "imputed_any",
        "gaplen_demand_h", "gaplen_pv_h", "gaplen_pool_h", "gaplen_temp_h",
        "flag_demand_imputed", "flag_pv_imputed", "flag_pool_imputed", "flag_temp_imputed",
        "flag_demand_imputed_new", "flag_pv_imputed_new", "flag_pool_imputed_new", "flag_temp_imputed_new",
    ]
    _require_cols(df, required, "anomaly score files")

    # Only test rows (LOCK)
    df = df[pd.to_numeric(df["is_test_split"], errors="coerce").fillna(0).astype(int) == 1].copy()
    if df.empty:
        _fail("After filtering is_test_split==1, no rows remain in anomaly score files.")

    # Enforce schema aliasing (LOCK)
    for base, new in [
        ("flag_demand_imputed", "flag_demand_imputed_new"),
        ("flag_pv_imputed", "flag_pv_imputed_new"),
        ("flag_pool_imputed", "flag_pool_imputed_new"),
        ("flag_temp_imputed", "flag_temp_imputed_new"),
    ]:
        b = _as_int01_no_nan(df[base], base)
        n = _as_int01_no_nan(df[new], new)
        if not np.array_equal(b, n):
            idx = int(np.where(b != n)[0][0])
            _fail(f"Schema alias mismatch: {base} != {new} at row {idx}")

    # Derived bins (LOCK names)
    df["gaplendemandbin"] = _digitize_gaplen(df["gaplen_demand_h"], edges)
    df["gaplenpvbin"] = _digitize_gaplen(df["gaplen_pv_h"], edges)
    df["gaplenpoolbin"] = _digitize_gaplen(df["gaplen_pool_h"], edges)
    df["gaplentempbin"] = _digitize_gaplen(df["gaplen_temp_h"], edges)

    # Normalize imputed_any
    df["imputed_any"] = _as_int01_no_nan(df["imputed_any"], "imputed_any")

    Ks = list(range(3, 16))

    # -------------------------
    # File 1: sanity slices
    # -------------------------
    slicers = [
        ("imputed_any", "imputed_any"),
        ("gaplendemandbin", "gaplendemandbin"),
        ("gaplenpvbin", "gaplenpvbin"),
        ("gaplenpoolbin", "gaplenpoolbin"),
        ("gaplentempbin", "gaplentempbin"),
    ]

    sanity_rows = []
    group_cols = ["dataset_id", "split_id", "model_name"]

    for (dataset_id, split_id, model_name), g in df.groupby(group_cols, sort=False):
        g = g.copy()
        # protect dtype
        g["anomaly_score"] = pd.to_numeric(g["anomaly_score"], errors="coerce")
        g["z_raw"] = pd.to_numeric(g["z_raw"], errors="coerce")
        if g["anomaly_score"].isna().any() or g["z_raw"].isna().any():
            _fail(f"Found NaNs in anomaly_score or z_raw for {dataset_id}/{split_id}/{model_name}")

        for k in Ks:
            resid_alert = (g["anomaly_score"].to_numpy(dtype=float) >= float(k))
            raw_alert = (g["z_raw"].to_numpy(dtype=float) >= float(k))

            for detector, alert_mask in [("residual", resid_alert), ("raw", raw_alert)]:
                # overall ALL
                sanity_rows.append({
                    "datasetid": dataset_id,
                    "splitid": split_id,
                    "modelname": model_name,
                    "detector": detector,
                    "k": int(k),
                    "slicename": "overall",
                    "slicevalue": "ALL",
                    "nrows": int(len(g)),
                    "nalerts": int(alert_mask.sum()),
                    "alert_rate": float(alert_mask.mean()) if len(g) > 0 else float("nan"),
                })

                # slicers
                for slicename, col in slicers:
                    vals = g[col].to_numpy()
                    # unique slice values present
                    uniq = pd.unique(vals)
                    for sv in sorted(uniq, key=lambda x: (str(type(x)), str(x))):
                        mask = (vals == sv)
                        nrows = int(mask.sum())
                        if nrows == 0:
                            continue
                        nalerts = int(alert_mask[mask].sum())
                        sanity_rows.append({
                            "datasetid": dataset_id,
                            "splitid": split_id,
                            "modelname": model_name,
                            "detector": detector,
                            "k": int(k),
                            "slicename": slicename,
                            "slicevalue": int(sv) if isinstance(sv, (np.integer, int)) else sv,
                            "nrows": nrows,
                            "nalerts": nalerts,
                            "alert_rate": float(nalerts / nrows),
                        })

    df_sanity = pd.DataFrame(sanity_rows)
    df_sanity = df_sanity.sort_values(
        ["datasetid", "splitid", "modelname", "detector", "k", "slicename", "slicevalue"],
        kind="mergesort",
    ).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    sanity_path = out_dir / "missingness_anomaly_sanity.csv"
    df_sanity.to_csv(sanity_path, index=False)

    # -------------------------
    # File 2: filtered comparison
    # -------------------------
    comp_rows = []

    for (dataset_id, split_id, model_name), g in df.groupby(group_cols, sort=False):
        g = g.copy()
        g["anomaly_score"] = pd.to_numeric(g["anomaly_score"], errors="coerce")
        g["z_raw"] = pd.to_numeric(g["z_raw"], errors="coerce")
        if g["anomaly_score"].isna().any() or g["z_raw"].isna().any():
            _fail(f"Found NaNs in anomaly_score or z_raw for {dataset_id}/{split_id}/{model_name}")

        imputed_any = g["imputed_any"].to_numpy(dtype=int)
        mask_clean = (imputed_any == 0)
        mask_imp = (imputed_any == 1)

        n_test = int(len(g))
        n_clean = int(mask_clean.sum())
        n_imputed = int(mask_imp.sum())

        for k in Ks:
            resid_alert = (g["anomaly_score"].to_numpy(dtype=float) >= float(k))
            raw_alert = (g["z_raw"].to_numpy(dtype=float) >= float(k))

            def rate(cnt: int, denom: int) -> float:
                return float(cnt / denom) if denom > 0 else float("nan")

            ra_total = int(resid_alert.sum())
            ra_clean = int(resid_alert[mask_clean].sum()) if n_clean > 0 else 0
            ra_imp = int(resid_alert[mask_imp].sum()) if n_imputed > 0 else 0

            za_total = int(raw_alert.sum())
            za_clean = int(raw_alert[mask_clean].sum()) if n_clean > 0 else 0
            za_imp = int(raw_alert[mask_imp].sum()) if n_imputed > 0 else 0

            resid_rate_clean = rate(ra_clean, n_clean)
            raw_rate_clean = rate(za_clean, n_clean)

            comp_rows.append({
                "datasetid": dataset_id,
                "splitid": split_id,
                "modelname": model_name,
                "k": int(k),
                "n_test": n_test,
                "n_clean": n_clean,
                "n_imputed": n_imputed,

                "residual_alerts_total": ra_total,
                "residual_alerts_clean": ra_clean,
                "residual_alerts_imputed": ra_imp,
                "residual_alert_rate_total": rate(ra_total, n_test),
                "residual_alert_rate_clean": resid_rate_clean,
                "residual_alert_rate_imputed": rate(ra_imp, n_imputed),
                "residual_ratio_alerts_imputed_to_clean": float(rate(ra_imp, n_imputed) / resid_rate_clean) if (n_imputed > 0 and n_clean > 0 and resid_rate_clean > 0) else float("inf") if (n_imputed > 0 and n_clean > 0 and resid_rate_clean == 0) else float("nan"),

                "raw_alerts_total": za_total,
                "raw_alerts_clean": za_clean,
                "raw_alerts_imputed": za_imp,
                "raw_alert_rate_total": rate(za_total, n_test),
                "raw_alert_rate_clean": raw_rate_clean,
                "raw_alert_rate_imputed": rate(za_imp, n_imputed),
                "raw_ratio_alerts_imputed_to_clean": float(rate(za_imp, n_imputed) / raw_rate_clean) if (n_imputed > 0 and n_clean > 0 and raw_rate_clean > 0) else float("inf") if (n_imputed > 0 and n_clean > 0 and raw_rate_clean == 0) else float("nan"),

                "residual_over_raw_rate_clean": float(resid_rate_clean / raw_rate_clean) if (n_clean > 0 and raw_rate_clean > 0) else float("inf") if (n_clean > 0 and raw_rate_clean == 0) else float("nan"),
            })

    df_comp = pd.DataFrame(comp_rows)
    df_comp = df_comp.sort_values(["datasetid", "splitid", "modelname", "k"], kind="mergesort").reset_index(drop=True)

    comp_path = out_dir / "missingness_filtered_comparison.csv"
    df_comp.to_csv(comp_path, index=False)

    return sanity_path, comp_path, len(df_sanity), len(df_comp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yml")
    ap.add_argument("--scores_root", type=str, default="metrics")
    ap.add_argument("--out", type=str, default="metrics")
    args = ap.parse_args()

    cfg = load_frozen_config(Path(args.config))
    # keep your global runtime asserts consistent with other prompts
    enforce_frozen_runtime_asserts(cfg)

    try:
        sanity_path, comp_path, n1, n2 = compute_outputs(cfg, Path(args.scores_root), Path(args.out))
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Wrote {sanity_path}")
    print(f"Wrote {comp_path}")
    print(f"Rows: sanity={n1}, compare={n2}")
    sys.exit(0)


if __name__ == "__main__":
    main()
