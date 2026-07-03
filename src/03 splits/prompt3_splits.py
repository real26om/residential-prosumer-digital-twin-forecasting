# prompt3_splits.py
#
# PROMPT 3 — ROLLING-WEEK SPLIT PROTOCOL (EXPANDING; KEYED TO ts_target_utc)
#
# Usage (PowerShell):
#   python .\prompt3_splits.py --master processed_hourly\master_hourly_extended.csv --config config.yml
#
# Output:
#   metrics/split_table.csv

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config_loader import load_frozen_config
from asserts import enforce_frozen_runtime_asserts


N_TEST_WEEKS_LOCKED = 4


def _monday_00_utc(d: pd.Timestamp) -> pd.Timestamp:
    """Return Monday 00:00 UTC for the week containing timestamp d (UTC)."""
    d0 = d.floor("D")
    return d0 - pd.Timedelta(days=int(d0.dayofweek))


def _get_last_n_complete_week_starts(
    ts_target_min: pd.Timestamp,
    ts_target_max: pd.Timestamp,
    n_weeks: int,
) -> list[pd.Timestamp]:
    """
    Compute all Monday 00:00 UTC week starts spanning [min,max], then keep only complete weeks:
      complete week if (week_end - 1h) <= ts_target_max
    Return the last n week starts (chronological).
    """
    if ts_target_min.tz is None or str(ts_target_min.tz) != "UTC":
        raise ValueError("ts_target_min must be tz-aware UTC.")
    if ts_target_max.tz is None or str(ts_target_max.tz) != "UTC":
        raise ValueError("ts_target_max must be tz-aware UTC.")

    start_monday = _monday_00_utc(ts_target_min)
    end_monday = _monday_00_utc(ts_target_max)

    # Mondays at 00:00 UTC
    week_starts = pd.date_range(start=start_monday, end=end_monday, freq="W-MON", tz="UTC").to_pydatetime()
    week_starts = [pd.Timestamp(x).tz_convert("UTC") for x in week_starts]

    complete = []
    for ws in week_starts:
        we = ws + pd.Timedelta(days=7)
        # Last hourly target timestamp inside the week is (we - 1 hour)
        if (we - pd.Timedelta(hours=1)) <= ts_target_max:
            complete.append(ws)

    if len(complete) < n_weeks:
        raise RuntimeError(
            f"Not enough complete weeks in ts_target_utc range. "
            f"Need {n_weeks}, found {len(complete)} complete weeks.\n"
            f"ts_target_min={ts_target_min}, ts_target_max={ts_target_max}"
        )

    return complete[-n_weeks:]


def build_split_table(
    *,
    master_path: Path,
    cfg_path: Path,
    out_path: Path = Path("metrics/split_table.csv"),
) -> pd.DataFrame:
    # Load frozen config + enforce preregistered runtime asserts
    cfg = load_frozen_config(cfg_path)
    enforce_frozen_runtime_asserts(cfg)

    df = pd.read_csv(master_path)

    if "ts_utc" not in df.columns:
        raise ValueError("master_hourly_extended.csv must contain column 'ts_utc'.")

    # Parse tz-aware UTC
    ts_utc = pd.to_datetime(df["ts_utc"], utc=True, errors="raise")
    if ts_utc.dt.tz is None or str(ts_utc.dt.tz) != "UTC":
        raise ValueError("ts_utc must be tz-aware UTC.")

    # Membership key: ts_target_utc
    ts_target = ts_utc + pd.Timedelta(hours=1)

    ts_target_min = ts_target.min()
    ts_target_max = ts_target.max()

    # Global training start (LOCKED): earliest available ts_target_utc in master
    train_start_utc = ts_target_min

    # Determine last 4 complete UTC weeks available (Monday 00:00 UTC boundaries)
    test_week_starts = _get_last_n_complete_week_starts(
        ts_target_min=ts_target_min,
        ts_target_max=ts_target_max,
        n_weeks=N_TEST_WEEKS_LOCKED,
    )

    rows = []
    for i, test_start_utc in enumerate(test_week_starts, start=1):
        test_end_utc = test_start_utc + pd.Timedelta(days=7)

        # Expanding window:
        # train = all history strictly before test_start_utc
        train_end_utc = test_start_utc

        rows.append(
            {
                "split_id": f"roll_{i:02d}",
                "train_start_utc": train_start_utc.isoformat(),
                "train_end_utc": train_end_utc.isoformat(),
                "test_start_utc": test_start_utc.isoformat(),
                "test_end_utc": test_end_utc.isoformat(),
                "week_index_utc": i,
            }
        )

    split_table = pd.DataFrame(rows)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    split_table.to_csv(out_path, index=False)

    return split_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=str, default="processed_hourly/master_hourly_extended.csv")
    ap.add_argument("--config", type=str, default="config.yml")
    ap.add_argument("--out", type=str, default="metrics/split_table.csv")
    args = ap.parse_args()

    st = build_split_table(
        master_path=Path(args.master),
        cfg_path=Path(args.config),
        out_path=Path(args.out),
    )

    print(f"Wrote {args.out}")
    print(st)


if __name__ == "__main__":
    main()