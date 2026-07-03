import pandas as pd
import ast
import joblib

IN_CSV = "A3_LAGS_stage1_halving_cv_results.csv"
FEATURES_JOBLIB = "A3_LAGS_feature_cols.joblib"

OUT_STABLE_PARAMS = "A3_LAGS_stage1_stable_choice.joblib"
OUT_ARTIFACT = "A3_LAGS_stage1_ARTIFACT.joblib"

df = pd.read_csv(IN_CSV)

# Higher is better because mean_test_score is negative RMSE
# Penalize instability a bit
df["adj_score"] = df["mean_test_score"] - 0.10 * df["std_test_score"]

top = df.sort_values("adj_score", ascending=False).head(30).copy()
best = top.sort_values("adj_score", ascending=False).iloc[0]

best_params = ast.literal_eval(best["params"])
print("Chosen stable params:", best_params)
print("Chosen mean CV RMSE:", -best["mean_test_score"])
print("Chosen std:", best["std_test_score"])

joblib.dump(best_params, OUT_STABLE_PARAMS)
print("Saved:", OUT_STABLE_PARAMS)

# Save an artifact that includes feature order + params (paper-safe reproducibility)
feature_cols = joblib.load(FEATURES_JOBLIB)

artifact = {
    "params": best_params,
    "feature_cols": feature_cols
}
joblib.dump(artifact, OUT_ARTIFACT)
print("Saved:", OUT_ARTIFACT, "| n_features =", len(feature_cols))
