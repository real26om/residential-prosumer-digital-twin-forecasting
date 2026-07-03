import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, make_scorer

from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV

CSV_PATH = "A3_causal-fill_keep-imputed_telemetry_168_READY.csv"
TARGET = "y_demand_t_plus_1"
DROP = ["ts_utc", "is_train", "is_test", TARGET]

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

rmse_scorer = make_scorer(rmse, greater_is_better=False)  # negative RMSE

df = pd.read_csv(CSV_PATH)
train = df[df["is_train"] == 1].copy()

X = train.drop(columns=DROP)
y = train[TARGET]

tscv = TimeSeriesSplit(n_splits=6)

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

param_grid = {
    "max_depth": [10, 20, 35, 50, 80, None],
    "min_samples_split": [2, 5, 10, 20, 40, 80],
    "min_samples_leaf": [1, 2, 4, 8, 16, 32],
    "max_features": ["sqrt", 0.3, 0.5, 0.8],
    "bootstrap": [True, False],
}

search = HalvingGridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring=rmse_scorer,
    cv=tscv,
    factor=3,
    resource="n_estimators",
    min_resources=200,
    max_resources=5400,   # 200 -> 600 -> 1800 -> 5400
    n_jobs=-1,
    verbose=2,
    refit=True
)

search.fit(X, y)

print("Best params A1 stage1:", search.best_params_)
print("Best CV RMSE A1 stage1:", -search.best_score_)

pd.DataFrame(search.cv_results_).to_csv("A3_stage1_halving_cv_results.csv", index=False)
