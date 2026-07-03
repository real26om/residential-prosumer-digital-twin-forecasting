import argparse
from pathlib import Path
import pandas as pd

def _need_cols(df: pd.DataFrame, cols, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name}: missing columns: {missing}")

def build_summary(df: pd.DataFrame, source: str) -> pd.DataFrame:
    # We only need overall + daylight slices
    keep = df[
        ((df["slicename"] == "overall") & (df["slicevalue"] == "ALL")) |
        (df["slicename"] == "is_daylight")
    ].copy()

    keep["source"] = source

    # Normalize slicevalue types a bit (keep as string for safety)
    keep["slicevalue"] = keep["slicevalue"].astype(str)

    # Pivot to make a compact table per dataset/split/model:
    # overall MAE/RMSE/MAPE + day MAE/RMSE/MAPE + night MAE/RMSE/MAPE
    overall = keep[(keep["slicename"] == "overall") & (keep["slicevalue"] == "ALL")].copy()
    overall = overall.rename(columns={
        "nrows": "nrows_overall",
        "mae": "mae_overall",
        "rmse": "rmse_overall",
        "mape": "mape_overall",
    })
    overall = overall[["datasetid","splitid","modelname","source","nrows_overall","mae_overall","rmse_overall","mape_overall"]]

    dn = keep[keep["slicename"] == "is_daylight"].copy()
    # slicevalue: "0" night, "1" day
    dn["dn_tag"] = dn["slicevalue"].map({"0": "night", "1": "day"}).fillna("unknown")
    dn = dn.rename(columns={"nrows":"nrows_dn","mae":"mae_dn","rmse":"rmse_dn","mape":"mape_dn"})
    dn = dn[["datasetid","splitid","modelname","source","dn_tag","nrows_dn","mae_dn","rmse_dn","mape_dn"]]

    # pivot day/night into columns
    dn_wide = dn.pivot_table(
        index=["datasetid","splitid","modelname","source"],
        columns="dn_tag",
        values=["nrows_dn","mae_dn","rmse_dn","mape_dn"],
        aggfunc="first"
    )
    dn_wide.columns = [f"{a}_{b}" for a,b in dn_wide.columns]
    dn_wide = dn_wide.reset_index()

    out = overall.merge(dn_wide, on=["datasetid","splitid","modelname","source"], how="left")
    out = out.sort_values(["source","datasetid","splitid","modelname"]).reset_index(drop=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", type=str, required=True)
    ap.add_argument("--intersection", type=str, required=True)
    ap.add_argument("--out", type=str, default="metrics/prompt6_summary_overall_daynight.csv")
    args = ap.parse_args()

    native = pd.read_csv(args.native)
    inter = pd.read_csv(args.intersection)

    base_cols = ["datasetid","splitid","modelname","slicename","slicevalue","nrows","mae","rmse","mape"]
    _need_cols(native, base_cols, "native")
    _need_cols(inter, base_cols + ["intersection","intersectionnrows"], "intersection")

    s1 = build_summary(native, "native")
    s2 = build_summary(inter, "intersection")

    out = pd.concat([s1, s2], ignore_index=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Wrote {out_path} ({len(out)} rows)")

if __name__ == "__main__":
    main()
