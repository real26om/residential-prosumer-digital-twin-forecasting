import pandas as pd
import ast
import joblib

IN_CSV = "A1_stage1_halving_cv_results.csv"
OUT_JOBLIB = "A1_stage1_stable_choice.joblib"

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

joblib.dump(best_params, OUT_JOBLIB)
print("Saved:", OUT_JOBLIB)
