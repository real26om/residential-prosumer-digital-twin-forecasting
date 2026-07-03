# write_sanity_checks_report.py
#
# Creates sanity-check outputs as files (CSV + TXT) from Prompt 2 datasets.
#
# What it checks (per split_id × dataset_id CSV):
# - Row count
# - Sums of flag_*_imputed and imputed_any / imputed_count
# - Consistency: imputed_any == OR(flags)  (row-wise)
# - Consistency: imputed_count == SUM(flags) (row-wise)
# - Clean-only expectations:
#     A2,B2: flag_demand_imputed must be 0 for all rows
#     A4,B4: all four flag_*_imputed must be 0 for all rows
# - Optional: compares row counts against datasets_by_split/manifest_all_splits.csv if present
#
# Usage (PowerShell, from your project root):
#   python .\write_sanity_checks_report.py
#
# Outputs:
#   metrics/sanity_checks_prompt2.csv
#   metrics/sanity_checks_prompt2.txt

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DATASET_IDS = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
CLEAN_ONLY_DEMAND_ONLY = {"A2", "B2"}
CLEAN_ONLY_TELEMETRY = {"A4", "B4"}

FLAG_COLS = ["flag_demand_imputed", "flag_pv_imputed", "flag_pool_imputed", "flag_temp_imputed"]


@dataclass
class CheckResult:
    split_id: str
    dataset_id: str
    path: str
    n_rows: int

    sum_flag_demand_imputed: int
    sum_flag_pv_imputed: int
    sum_flag_pool_imputed: int
    sum_flag_temp_imputed: int

    sum_imputed_any: int
    sum_imputed_count: int

    # row-wise consistency checks
    imputed_any_or_ok: bool
    imputed_count_sum_ok: bool

    # clean-only checks
    clean_only_expected: str
    clean_only_ok: bool

    # manifest check
    manifest_expected_rows: Optional[int]
    manifest_rows_match: Optional[bool]


def _as_int01(series: pd.Series, colname: str) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").to_numpy()
    if np.isnan(x).any():
        raise ValueError(f"{colname}: contains NaN but should be int 0/1.")
    xi = x.astype(int)
    if not np.all((xi == 0) | (xi == 1)):
        bad = np.unique(xi[~((xi == 0) | (xi == 1))])
        raise ValueError(f"{colname}: has values outside {{0,1}}: {bad[:10]}")
    return xi


def _as_int_nonneg(series: pd.Series, colname: str) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").to_numpy()
    if np.isnan(x).any():
        raise ValueError(f"{colname}: contains NaN but should be integer >=0.")
    xi = x.astype(int)
    if (xi < 0).any():
        raise ValueError(f"{colname}: contains negative values (should be >=0).")
    return xi


def _find_manifest(datasets_root: Path) -> Optional[Path]:
    # Your earlier file name: datasets_by_split/manifest_all_splits.csv
    cand = datasets_root / "manifest_all_splits.csv"
    if cand.exists():
        return cand
    # fallback: sometimes people place it next to datasets_root
    cand2 = datasets_root.parent / "manifest_all_splits.csv"
    if cand2.exists():
        return cand2
    return None


def _load_manifest(manifest_path: Path) -> pd.DataFrame:
    mf = pd.read_csv(manifest_path)
    # expected minimal columns
    for c in ["split_id", "dataset_id"]:
        if c not in mf.columns:
            raise ValueError(f"manifest is missing required column: {c}")
    # try to infer rowcount column
    rowcount_col = None
    for c in ["n_rows", "rows", "row_count", "nrows"]:
        if c in mf.columns:
            rowcount_col = c
            break
    if rowcount_col is None:
        # if your manifest doesn’t store it, we just won’t compare
        mf["__expected_rows__"] = np.nan
        return mf
    mf["__expected_rows__"] = pd.to_numeric(mf[rowcount_col], errors="coerce")
    return mf


def _expected_rows_from_manifest(mf: pd.DataFrame, split_id: str, dataset_id: str) -> Optional[int]:
    if "__expected_rows__" not in mf.columns:
        return None
    sub = mf[(mf["split_id"] == split_id) & (mf["dataset_id"] == dataset_id)]
    if len(sub) == 0:
        return None
    val = sub["__expected_rows__"].iloc[0]
    if pd.isna(val):
        return None
    return int(val)


def check_one(csv_path: Path, split_id: str, dataset_id: str, mf: Optional[pd.DataFrame]) -> CheckResult:
    df = pd.read_csv(csv_path)

    # required columns for these sanity checks
    missing = [c for c in (FLAG_COLS + ["imputed_any", "imputed_count"]) if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing required columns for sanity checks: {missing}")

    flags = {c: _as_int01(df[c], c) for c in FLAG_COLS}
    imputed_any = _as_int01(df["imputed_any"], "imputed_any")
    imputed_count = _as_int_nonneg(df["imputed_count"], "imputed_count")

    # row-wise expectations
    or_expected = (flags["flag_demand_imputed"] |
                   flags["flag_pv_imputed"] |
                   flags["flag_pool_imputed"] |
                   flags["flag_temp_imputed"]).astype(int)

    sum_expected = (flags["flag_demand_imputed"] +
                    flags["flag_pv_imputed"] +
                    flags["flag_pool_imputed"] +
                    flags["flag_temp_imputed"]).astype(int)

    imputed_any_or_ok = bool(np.array_equal(imputed_any, or_expected))
    imputed_count_sum_ok = bool(np.array_equal(imputed_count, sum_expected))

    # clean-only expectations
    if dataset_id in CLEAN_ONLY_DEMAND_ONLY:
        clean_only_expected = "demand-only clean-only: flag_demand_imputed all-zero"
        clean_only_ok = bool(flags["flag_demand_imputed"].sum() == 0)
    elif dataset_id in CLEAN_ONLY_TELEMETRY:
        clean_only_expected = "telemetry clean-only: all flag_*_imputed all-zero"
        clean_only_ok = bool(
            flags["flag_demand_imputed"].sum() == 0 and
            flags["flag_pv_imputed"].sum() == 0 and
            flags["flag_pool_imputed"].sum() == 0 and
            flags["flag_temp_imputed"].sum() == 0
        )
    else:
        clean_only_expected = "keep-imputed dataset: no all-zero requirement"
        clean_only_ok = True

    exp_rows = _expected_rows_from_manifest(mf, split_id, dataset_id) if mf is not None else None
    match = (exp_rows == len(df)) if exp_rows is not None else None

    return CheckResult(
        split_id=split_id,
        dataset_id=dataset_id,
        path=str(csv_path),
        n_rows=len(df),
        sum_flag_demand_imputed=int(flags["flag_demand_imputed"].sum()),
        sum_flag_pv_imputed=int(flags["flag_pv_imputed"].sum()),
        sum_flag_pool_imputed=int(flags["flag_pool_imputed"].sum()),
        sum_flag_temp_imputed=int(flags["flag_temp_imputed"].sum()),
        sum_imputed_any=int(imputed_any.sum()),
        sum_imputed_count=int(imputed_count.sum()),
        imputed_any_or_ok=imputed_any_or_ok,
        imputed_count_sum_ok=imputed_count_sum_ok,
        clean_only_expected=clean_only_expected,
        clean_only_ok=clean_only_ok,
        manifest_expected_rows=exp_rows,
        manifest_rows_match=match,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_root", type=str, default="datasets_by_split")
    ap.add_argument("--outdir", type=str, default="metrics")
    args = ap.parse_args()

    datasets_root = Path(args.datasets_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_path = _find_manifest(datasets_root)
    mf = None
    if manifest_path is not None:
        mf = _load_manifest(manifest_path)

    results: list[CheckResult] = []
    missing_files: list[str] = []
    errors: list[str] = []

    # discover split folders roll_01..roll_04 automatically
    split_dirs = sorted([p for p in datasets_root.iterdir() if p.is_dir() and p.name.startswith("roll_")])

    for split_dir in split_dirs:
        split_id = split_dir.name
        for dataset_id in DATASET_IDS:
            csv_path = split_dir / f"{dataset_id}.csv"
            if not csv_path.exists():
                missing_files.append(str(csv_path))
                continue
            try:
                results.append(check_one(csv_path, split_id, dataset_id, mf))
            except Exception as e:
                errors.append(f"{csv_path}: {type(e).__name__}: {e}")

    # write CSV summary
    rows = []
    for r in results:
        rows.append({
            "split_id": r.split_id,
            "dataset_id": r.dataset_id,
            "path": r.path,
            "n_rows": r.n_rows,
            "sum_flag_demand_imputed": r.sum_flag_demand_imputed,
            "sum_flag_pv_imputed": r.sum_flag_pv_imputed,
            "sum_flag_pool_imputed": r.sum_flag_pool_imputed,
            "sum_flag_temp_imputed": r.sum_flag_temp_imputed,
            "sum_imputed_any": r.sum_imputed_any,
            "sum_imputed_count": r.sum_imputed_count,
            "imputed_any_or_ok": int(r.imputed_any_or_ok),
            "imputed_count_sum_ok": int(r.imputed_count_sum_ok),
            "clean_only_expected": r.clean_only_expected,
            "clean_only_ok": int(r.clean_only_ok),
            "manifest_expected_rows": r.manifest_expected_rows if r.manifest_expected_rows is not None else "",
            "manifest_rows_match": (int(r.manifest_rows_match) if r.manifest_rows_match is not None else ""),
        })
    summary_df = pd.DataFrame(rows).sort_values(["split_id", "dataset_id"])
    csv_out = outdir / "sanity_checks_prompt2.csv"
    summary_df.to_csv(csv_out, index=False)

    # write TXT report
    txt_out = outdir / "sanity_checks_prompt2.txt"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # failures
    bad_any_or = summary_df[summary_df["imputed_any_or_ok"] == 0]
    bad_count = summary_df[summary_df["imputed_count_sum_ok"] == 0]
    bad_clean = summary_df[summary_df["clean_only_ok"] == 0]
    bad_manifest = summary_df[summary_df["manifest_rows_match"] == 0] if "manifest_rows_match" in summary_df.columns else pd.DataFrame()

    with txt_out.open("w", encoding="utf-8") as f:
        f.write(f"Prompt 2 sanity checks report (UTC): {now}\n")
        f.write(f"datasets_root: {datasets_root.resolve()}\n")
        f.write(f"manifest: {manifest_path.resolve() if manifest_path is not None else 'NOT FOUND'}\n")
        f.write(f"checked_files: {len(results)}\n")
        f.write(f"missing_files: {len(missing_files)}\n")
        f.write(f"errors: {len(errors)}\n\n")

        if missing_files:
            f.write("MISSING FILES:\n")
            for p in missing_files[:200]:
                f.write(f"  - {p}\n")
            if len(missing_files) > 200:
                f.write(f"  ... and {len(missing_files)-200} more\n")
            f.write("\n")

        if errors:
            f.write("ERRORS (during checking):\n")
            for e in errors[:200]:
                f.write(f"  - {e}\n")
            if len(errors) > 200:
                f.write(f"  ... and {len(errors)-200} more\n")
            f.write("\n")

        f.write("FAILURE COUNTS:\n")
        f.write(f"  imputed_any_or_ok failures: {len(bad_any_or)}\n")
        f.write(f"  imputed_count_sum_ok failures: {len(bad_count)}\n")
        f.write(f"  clean_only_ok failures: {len(bad_clean)}\n")
        f.write(f"  manifest_rows_match failures: {len(bad_manifest)}\n\n")

        if len(bad_any_or):
            f.write("DETAIL: imputed_any != OR(flags)\n")
            f.write(bad_any_or[["split_id","dataset_id","n_rows","sum_imputed_any","sum_flag_demand_imputed",
                               "sum_flag_pv_imputed","sum_flag_pool_imputed","sum_flag_temp_imputed","path"]].to_string(index=False))
            f.write("\n\n")

        if len(bad_count):
            f.write("DETAIL: imputed_count != SUM(flags)\n")
            f.write(bad_count[["split_id","dataset_id","n_rows","sum_imputed_count","sum_flag_demand_imputed",
                               "sum_flag_pv_imputed","sum_flag_pool_imputed","sum_flag_temp_imputed","path"]].to_string(index=False))
            f.write("\n\n")

        if len(bad_clean):
            f.write("DETAIL: clean-only expectations violated\n")
            f.write(bad_clean[["split_id","dataset_id","n_rows","clean_only_expected",
                               "sum_flag_demand_imputed","sum_flag_pv_imputed","sum_flag_pool_imputed","sum_flag_temp_imputed","path"]].to_string(index=False))
            f.write("\n\n")

        if len(bad_manifest):
            f.write("DETAIL: manifest row count mismatch\n")
            f.write(bad_manifest[["split_id","dataset_id","n_rows","manifest_expected_rows","path"]].to_string(index=False))
            f.write("\n\n")

        # Example “robustness” block for roll_04/A2 if present
        ex = summary_df[(summary_df["split_id"] == "roll_04") & (summary_df["dataset_id"] == "A2")]
        if len(ex):
            f.write("EXAMPLE ROBUSTNESS CHECK (roll_04 / A2):\n")
            f.write(ex[["n_rows","sum_flag_demand_imputed","sum_flag_pv_imputed","sum_flag_pool_imputed",
                        "sum_flag_temp_imputed","sum_imputed_any"]].to_string(index=False))
            f.write("\n")
            f.write("Interpretation:\n")
            f.write("- A2 is demand-only clean-only, so flag_demand_imputed must be 0.\n")
            f.write("- imputed_any can be >0 due to PV/pool/temp imputation (telemetry is not a feature in A2).\n")

    print(f"Wrote: {csv_out}")
    print(f"Wrote: {txt_out}")

    # exit non-zero if any critical failures
    critical_failures = len(errors) + len(bad_any_or) + len(bad_count) + len(bad_clean)
    raise SystemExit(1 if critical_failures else 0)


if __name__ == "__main__":
    main()