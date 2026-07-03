# prompt7_label_eval.py
# PROMPT 7 — LABELING + EVALUATION (KEYED TO ts_target_utc)
#
# Builds labels.csv keyed to ts_target_utc using z_raw (raw-demand control),
# then evaluates anomaly_score detection at hour-level and event-level for K=3..15.
#
# Usage (PowerShell):
#   python .\prompt7_label_eval.py --config .\config.yml --scores_root .\metrics --out .\metrics --ref_dataset A1
#
# Outputs:
#   metrics/labels.csv
#   metrics/anomaly_eval_hour_level.csv
#   metrics/anomaly_eval_event_level.csv
#   metrics/k_sweep_eval_summary.csv

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from config_loader import load_frozen_config, cfg_get
from asserts import enforce_frozen_runtime_asserts


# -----------------------------
# Helpers
# -----------------------------
def _fail(msg: str) -> None:
    raise RuntimeError(msg)


def _require_cols(df: pd.DataFrame, cols: List[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        _fail(f"Missing required columns ({context}): {missing}")


def _as_int01(s: pd.Series, context: str) -> np.ndarray:
    arr = pd.to_numeric(s, errors="coerce").to_numpy()
    if np.isnan(arr).any():
        _fail(f"Found NaN in 0/1 column ({context}).")
    arr = arr.astype(int)
    bad = arr[~((arr == 0) | (arr == 1))]
    if bad.size:
        _fail(f"Non 0/1 values in ({context}): {np.unique(bad)[:10]}")
    return arr


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


def _parse_ts_utc(series: pd.Series, name: str) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="raise")
    if ts.dt.tz is None or str(ts.dt.tz) != "UTC":
        _fail(f"{name} must be tz-aware UTC.")
    return ts


# -----------------------------
# Events (contiguous 1h positives)
# -----------------------------
@dataclass(frozen=True)
class Event:
    start: pd.Timestamp  # inclusive
    end: pd.Timestamp    # inclusive
    n_hours: int


def _to_events(ts: pd.Series, is_pos: np.ndarray) -> List[Event]:
    # assumes ts sorted ascending, hourly-ish
    if len(ts) == 0:
        return []

    events: List[Event] = []
    in_evt = False
    start = None
    prev_ts = None
    count = 0

    for t, pos in zip(ts, is_pos):
        if pos == 1:
            if not in_evt:
                in_evt = True
                start = t
                prev_ts = t
                count = 1
            else:
                # continue if exactly 1 hour apart, else new event
                if (t - prev_ts) == pd.Timedelta(hours=1):
                    prev_ts = t
                    count += 1
                else:
                    events.append(Event(start=start, end=prev_ts, n_hours=count))
                    start = t
                    prev_ts = t
                    count = 1
        else:
            if in_evt:
                events.append(Event(start=start, end=prev_ts, n_hours=count))
                in_evt = False
                start = None
                prev_ts = None
                count = 0

    if in_evt:
        events.append(Event(start=start, end=prev_ts, n_hours=count))
    return events


def _events_overlap(a: Event, b: Event) -> bool:
    return (a.start <= b.end) and (b.start <= a.end)


def _event_metrics(ts: pd.Series, y_true_pos: np.ndarray, y_pred_pos: np.ndarray) -> Dict[str, float]:
    true_events = _to_events(ts, y_true_pos)
    pred_events = _to_events(ts, y_pred_pos)

    n_true = len(true_events)
    n_pred = len(pred_events)

    hit_true = 0
    for te in true_events:
        if any(_events_overlap(te, pe) for pe in pred_events):
            hit_true += 1

    hit_pred = 0
    for pe in pred_events:
        if any(_events_overlap(pe, te) for te in true_events):
            hit_pred += 1

    prec_e = _safe_div(hit_pred, n_pred)
    rec_e = _safe_div(hit_true, n_true)
    f1_e = _safe_div(2 * prec_e * rec_e, (prec_e + rec_e)) if (prec_e + rec_e) > 0 else 0.0

    mean_true_len = float(np.mean([e.n_hours for e in true_events])) if n_true else 0.0
    mean_pred_len = float(np.mean([e.n_hours for e in pred_events])) if n_pred else 0.0

    return {
        "n_true_events": int(n_true),
        "n_pred_events": int(n_pred),
        "n_hit_true_events": int(hit_true),
        "n_hit_pred_events": int(hit_pred),
        "precision_event": float(prec_e),
        "recall_event": float(rec_e),
        "f1_event": float(f1_e),
        "mean_true_event_len_h": float(mean_true_len),
        "mean_pred_event_len_h": float(mean_pred_len),
    }


# -----------------------------
# File discovery
# -----------------------------
_SCORE_RE = re.compile(r"^anomaly_scores_(?P<dataset_id>[A-Z]\d)_(?P<split_id>roll_\d{2})\.csv$", re.IGNORECASE)


def discover_score_files(scores_root: Path) -> Dict[Tuple[str, str], Path]:
    out: Dict[Tuple[str, str], Path] = {}
    for p in scores_root.glob("anomaly_scores_*_roll_*.csv"):
        m = _SCORE_RE.match(p.name)
        if not m:
            continue
        dataset_id = m.group("dataset_id").upper()
        split_id = m.group("split_id").lower()
        out[(dataset_id, split_id)] = p
    if not out:
        _fail(f"No anomaly score files found under {scores_root} matching anomaly_scores_*_roll_*.csv")
    return out


# -----------------------------
# Labels
# -----------------------------
def build_labels(scores_map: Dict[Tuple[str, str], Path], ref_dataset: str) -> pd.DataFrame:
    ref_dataset = ref_dataset.upper()
    rows: List[pd.DataFrame] = []

    for (ds, split), path in scores_map.items():
        if ds != ref_dataset:
            continue
        df = pd.read_csv(path)
        _require_cols(df, ["split_id", "ts_target_utc", "y_true_t_plus_1", "z_raw"], f"labels ref {path.name}")
        df["ts_target_utc"] = _parse_ts_utc(df["ts_target_utc"], "ts_target_utc")
        # If multiple model_name rows exist, z_raw should be identical; keep first per timestamp
        df = df.sort_values(["split_id", "ts_target_utc"])
        df = df.drop_duplicates(subset=["split_id", "ts_target_utc"], keep="first")
        df = df[["split_id", "ts_target_utc", "y_true_t_plus_1", "z_raw"]].copy()
        rows.append(df)

    if not rows:
        _fail(f"Could not find any score files for ref_dataset={ref_dataset} to build labels.")

    labels = pd.concat(rows, ignore_index=True)
    # enforce uniqueness
    if labels.duplicated(subset=["split_id", "ts_target_utc"]).any():
        _fail("labels.csv would contain duplicate (split_id, ts_target_utc) keys.")
    labels = labels.sort_values(["split_id", "ts_target_utc"]).reset_index(drop=True)
    return labels


# -----------------------------
# Evaluation
# -----------------------------
def hour_level_metrics(y_true_pos: np.ndarray, y_pred_pos: np.ndarray) -> Dict[str, float]:
    # confusion
    tp = int(np.sum((y_true_pos == 1) & (y_pred_pos == 1)))
    fp = int(np.sum((y_true_pos == 0) & (y_pred_pos == 1)))
    fn = int(np.sum((y_true_pos == 1) & (y_pred_pos == 0)))
    tn = int(np.sum((y_true_pos == 0) & (y_pred_pos == 0)))

    precision = _safe_div(tp, (tp + fp))
    recall = _safe_div(tp, (tp + fn))
    f1 = _safe_div(2 * precision * recall, (precision + recall)) if (precision + recall) > 0 else 0.0

    tpr = recall
    fpr = _safe_div(fp, (fp + tn))

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "n_true_pos": int(np.sum(y_true_pos == 1)),
        "n_pred_pos": int(np.sum(y_pred_pos == 1)),
    }


def evaluate_all(
    cfg: dict,
    scores_map: Dict[Tuple[str, str], Path],
    labels: pd.DataFrame,
    k_values: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Use eps from config for any safety checks (not required in these metrics, but kept for consistency)
    _ = float(cfg_get(cfg, "config.frozen_constants.eps"))

    # index labels for fast join
    lab = labels.copy()
    lab["split_id"] = lab["split_id"].astype(str).str.lower()
    lab = lab.set_index(["split_id", "ts_target_utc"])

    hour_rows: List[dict] = []
    event_rows: List[dict] = []

    for (dataset_id, split_id), path in sorted(scores_map.items(), key=lambda x: (x[0][0], x[0][1])):
        df = pd.read_csv(path)
        required = [
            "dataset_id", "split_id", "model_name",
            "ts_target_utc", "is_test_split",
            "anomaly_score", "z_raw",
        ]
        _require_cols(df, required, f"score file {path.name}")

        df["split_id"] = df["split_id"].astype(str).str.lower()
        df["dataset_id"] = df["dataset_id"].astype(str).str.upper()
        df["model_name"] = df["model_name"].astype(str).str.lower()
        df["ts_target_utc"] = _parse_ts_utc(df["ts_target_utc"], "ts_target_utc")

        # Evaluate only test rows
        is_test = _as_int01(df["is_test_split"], f"is_test_split in {path.name}")
        df = df.loc[is_test == 1].copy()

        # Join labels (ground truth control) by (split_id, ts_target_utc)
        key = pd.MultiIndex.from_frame(df[["split_id", "ts_target_utc"]])
        if not key.isin(lab.index).all():
            missing = key[~key.isin(lab.index)]
            ex = missing[0]
            _fail(f"Labels missing for some rows in {path.name}. Example missing key: {ex}")
        df = df.join(lab, on=["split_id", "ts_target_utc"], rsuffix="_label")

        # Sort for event extraction
        df = df.sort_values("ts_target_utc").reset_index(drop=True)
        ts = df["ts_target_utc"]

        for K in k_values:
            y_true_pos = (pd.to_numeric(df["z_raw_label"], errors="coerce").to_numpy() >= K).astype(int)
            y_pred_pos = (pd.to_numeric(df["anomaly_score"], errors="coerce").to_numpy() >= K).astype(int)

            hm = hour_level_metrics(y_true_pos, y_pred_pos)
            em = _event_metrics(ts, y_true_pos, y_pred_pos)

            hour_rows.append({
                "datasetid": dataset_id,
                "splitid": split_id,
                "modelname": df["model_name"].iloc[0] if df["model_name"].nunique() == 1 else "mixed",
                "K": int(K),
                "nrows": int(len(df)),
                **hm,
            })
            event_rows.append({
                "datasetid": dataset_id,
                "splitid": split_id,
                "modelname": df["model_name"].iloc[0] if df["model_name"].nunique() == 1 else "mixed",
                "K": int(K),
                "nrows": int(len(df)),
                **em,
            })

        # Important: if a file contains multiple model_names, the above would mix.
        # In your Prompt 5 outputs, model_name is present; typically each file contains all models.
        # So we need to split by model_name properly.
        if df["model_name"].nunique() > 1:
            # Remove the incorrect mixed rows we just added and redo correctly
            hour_rows = hour_rows[:-len(k_values)]
            event_rows = event_rows[:-len(k_values)]

            for model_name, g in df.groupby("model_name", sort=True):
                g = g.sort_values("ts_target_utc").reset_index(drop=True)
                ts_g = g["ts_target_utc"]
                for K in k_values:
                    y_true_pos = (pd.to_numeric(g["z_raw_label"], errors="coerce").to_numpy() >= K).astype(int)
                    y_pred_pos = (pd.to_numeric(g["anomaly_score"], errors="coerce").to_numpy() >= K).astype(int)

                    hm = hour_level_metrics(y_true_pos, y_pred_pos)
                    em = _event_metrics(ts_g, y_true_pos, y_pred_pos)

                    hour_rows.append({
                        "datasetid": dataset_id,
                        "splitid": split_id,
                        "modelname": str(model_name),
                        "K": int(K),
                        "nrows": int(len(g)),
                        **hm,
                    })
                    event_rows.append({
                        "datasetid": dataset_id,
                        "splitid": split_id,
                        "modelname": str(model_name),
                        "K": int(K),
                        "nrows": int(len(g)),
                        **em,
                    })

    hour_df = pd.DataFrame(hour_rows)
    event_df = pd.DataFrame(event_rows)

    # Summary across splits (k sweep eval summary)
    # Aggregate by datasetid, modelname, K
    def agg_hour(g: pd.DataFrame) -> pd.Series:
        tp = int(g["tp"].sum()); fp = int(g["fp"].sum()); fn = int(g["fn"].sum()); tn = int(g["tn"].sum())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
        return pd.Series({
            "splits_covered": int(g["splitid"].nunique()),
            "nrows_total": int(g["nrows"].sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "n_true_pos": int(g["n_true_pos"].sum()),
            "n_pred_pos": int(g["n_pred_pos"].sum()),
        })

    def agg_event(g: pd.DataFrame) -> pd.Series:
        n_true = int(g["n_true_events"].sum())
        n_pred = int(g["n_pred_events"].sum())
        hit_true = int(g["n_hit_true_events"].sum())
        hit_pred = int(g["n_hit_pred_events"].sum())
        prec = _safe_div(hit_pred, n_pred)
        rec = _safe_div(hit_true, n_true)
        f1 = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) > 0 else 0.0
        return pd.Series({
            "splits_covered": int(g["splitid"].nunique()),
            "nrows_total": int(g["nrows"].sum()),
            "n_true_events": n_true,
            "n_pred_events": n_pred,
            "n_hit_true_events": hit_true,
            "n_hit_pred_events": hit_pred,
            "precision_event": float(prec),
            "recall_event": float(rec),
            "f1_event": float(f1),
        })

    hour_sum = (
        hour_df.groupby(["datasetid", "modelname", "K"], sort=True)
        .apply(agg_hour)
        .reset_index()
    )
    hour_sum.insert(0, "level", "hour")

    event_sum = (
        event_df.groupby(["datasetid", "modelname", "K"], sort=True)
        .apply(agg_event)
        .reset_index()
    )
    event_sum.insert(0, "level", "event")

    summary = pd.concat([hour_sum, event_sum], ignore_index=True)

    # deterministic sorting
    hour_df = hour_df.sort_values(["datasetid", "splitid", "modelname", "K"]).reset_index(drop=True)
    event_df = event_df.sort_values(["datasetid", "splitid", "modelname", "K"]).reset_index(drop=True)
    summary = summary.sort_values(["level", "datasetid", "modelname", "K"]).reset_index(drop=True)

    return hour_df, event_df, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yml")
    ap.add_argument("--scores_root", type=str, default="metrics")
    ap.add_argument("--out", type=str, default="metrics")
    ap.add_argument("--ref_dataset", type=str, default="A1", help="Dataset used to generate labels.csv from z_raw (default A1)")
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=15)
    args = ap.parse_args()

    cfg = load_frozen_config(Path(args.config))
    enforce_frozen_runtime_asserts(cfg)

    scores_root = Path(args.scores_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_map = discover_score_files(scores_root)

    labels = build_labels(scores_map, ref_dataset=args.ref_dataset)
    labels_out = out_dir / "labels.csv"
    labels.to_csv(labels_out, index=False)

    k_values = list(range(int(args.kmin), int(args.kmax) + 1))

    hour_df, event_df, summary_df = evaluate_all(cfg, scores_map, labels, k_values)

    hour_path = out_dir / "anomaly_eval_hour_level.csv"
    event_path = out_dir / "anomaly_eval_event_level.csv"
    sum_path = out_dir / "k_sweep_eval_summary.csv"

    hour_df.to_csv(hour_path, index=False)
    event_df.to_csv(event_path, index=False)
    summary_df.to_csv(sum_path, index=False)

    print(f"Wrote {labels_out}")
    print(f"Wrote {hour_path}")
    print(f"Wrote {event_path}")
    print(f"Wrote {sum_path}")


if __name__ == "__main__":
    main()
