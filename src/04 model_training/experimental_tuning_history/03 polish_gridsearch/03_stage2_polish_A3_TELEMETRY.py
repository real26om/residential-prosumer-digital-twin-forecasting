import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer

CSV_PATH = "A3_causal-fill_keep-imputed_telemetry_168_READY.csv"
TARGET = "y_demand_t_plus_1"
DROP = ["ts_utc", "is_train", "is_test", TARGET]

BASE_PARAMS_PATH = "A3_stage1_stable_choice.joblib"

OUT_CV_RESULTS = "A3_stage2_polish_cv_results.csv"
OUT_BEST_PARAMS = "BEST_PARAMS_TELEMETRY.joblib"

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

rmse_scorer = make_scorer(rmse, greater_is_better=False)  # negative RMSE

def uniq_keep_order(seq):
    # Remove duplicates while preserving order (works for mixed types)
    return list(dict.fromkeys(seq))

def to_float_if_possible(x):
    try:
        return float(x)
    except Exception:
        return None

# Load base params selected from Stage 1
base_params = joblib.load(BASE_PARAMS_PATH)

# Load data
df = pd.read_csv(CSV_PATH)
train = df[df["is_train"] == 1].copy()

X = train.drop(columns=DROP)
y = train[TARGET]

tscv = TimeSeriesSplit(n_splits=6)

# Fixed strong tree count for stable ranking in Stage 2
rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1,
    n_estimators=3000,
    bootstrap=bool(base_params.get("bootstrap", True)),
)

# --- Build a small "neighborhood" grid around the Stage-1 choice ---

# max_depth can be None
best_depth = base_params.get("max_depth", None)
if pd.isna(best_depth):
    best_depth = None
if best_depth is not None:
    best_depth = int(best_depth)

if best_depth is None:
    max_depth_list = [None, 20, 50, 80]  # sensible neighborhood when Stage-1 picked None
else:
    max_depth_list = uniq_keep_order([
        max(5, best_depth - 10),
        best_depth,
        best_depth + 10,
        best_depth + 30
    ])

# min_samples_leaf / split
best_leaf = int(base_params.get("min_samples_leaf", 1))
best_split = int(base_params.get("min_samples_split", 2))

min_samples_leaf_list = sorted(list(set([
    max(1, best_leaf // 2),
    best_leaf,
    best_leaf * 2
])))

min_samples_split_list = sorted(list(set([
    max(2, best_split // 2),
    best_split,
    best_split * 2
])))

# max_features can be "sqrt" or float
mf = base_params.get("max_features", "sqrt")
mf_num = to_float_if_possible(mf)

if str(mf) == "sqrt" or mf_num is None:
    max_features_list = ["sqrt", 0.3, 0.5, 0.8]
else:
    max_features_list = [max(0.1, mf_num - 0.2), mf_num, min(1.0, mf_num + 0.2), "sqrt"]

max_features_list = uniq_keep_order(max_features_list)

param_grid = {
    "max_depth": max_depth_list,
    "min_samples_leaf": min_samples_leaf_list,
    "min_samples_split": min_samples_split_list,
    "max_features": max_features_list,
}

print("Stage-2 grid sizes:")
for k, v in param_grid.items():
    print(k, "=", v)

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring=rmse_scorer,
    cv=tscv,
    n_jobs=-1,
    verbose=2,
    refit=True
)

grid.fit(X, y)

print("\nBest params A1 stage2:", grid.best_params_)
print("Best CV RMSE A1 stage2:", -grid.best_score_)

pd.DataFrame(grid.cv_results_).to_csv(OUT_CV_RESULTS, index=False)
print("Saved:", OUT_CV_RESULTS)

# Save full estimator params so you can reuse directly (includes n_estimators=3000, random_state, etc.)
best_final = grid.best_estimator_.get_params()
joblib.dump(best_final, OUT_BEST_PARAMS)
print("Saved:", OUT_BEST_PARAMS)
