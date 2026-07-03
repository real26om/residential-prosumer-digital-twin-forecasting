import pandas as pd
import joblib

TARGET = "y_demand_t_plus_1"
DROP_COLS = ["ts_utc", "is_train", "is_test", TARGET]

DEMAND_TRAIN_CSV = "A1_causal_demand_keep-imputed_demand-only_168_READY.csv"
TELEM_TRAIN_CSV  = "A3_causal-fill_keep-imputed_telemetry_168_READY.csv"

DEMAND_PARAMS_JOBLIB = "BEST_PARAMS_DEMAND.joblib"
TELEM_PARAMS_JOBLIB  = "BEST_PARAMS_TELEMETRY.joblib"

OUT_DEMAND_ARTIFACT = "BEST_DEMAND_ARTIFACT.joblib"
OUT_TELEM_ARTIFACT  = "BEST_TELEMETRY_ARTIFACT.joblib"

def build_feature_cols(df, drop_cols):
    # stable order independent of CSV column order
    cols = [c for c in df.columns if c not in drop_cols]
    return sorted(cols)

def make_artifact(train_csv, params_joblib, out_joblib):
    df = pd.read_csv(train_csv)
    train = df[df["is_train"] == 1].copy()

    feature_cols = build_feature_cols(train, DROP_COLS)

    params = joblib.load(params_joblib)
    # store only what you need + fix reproducibility
    params["n_jobs"] = -1
    params["random_state"] = 42

    artifact = {
        "params": params,
        "feature_cols": feature_cols,
        "target_col": TARGET,
        "drop_cols": DROP_COLS,
        "train_csv_used_to_define_features": train_csv,
    }
    joblib.dump(artifact, out_joblib)
    print("Saved:", out_joblib, "| n_features =", len(feature_cols))

make_artifact(DEMAND_TRAIN_CSV, DEMAND_PARAMS_JOBLIB, OUT_DEMAND_ARTIFACT)
make_artifact(TELEM_TRAIN_CSV,  TELEM_PARAMS_JOBLIB,  OUT_TELEM_ARTIFACT)
