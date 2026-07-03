import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error

TARGET = "y_demand_t_plus_1"
DROP_COLS = ["ts_utc", "is_train", "is_test", TARGET]

CSV_PATH = "A3_causal-fill_keep-imputed_telemetry_168_READY.csv"  # change as needed
PARAMS_PATH = "BEST_PARAMS_TELEMETRY.joblib"                      # or BEST_PARAMS_DEMAND.joblib

def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else np.nan

df = pd.read_csv(CSV_PATH)
train = df[df["is_train"] == 1].copy()
test  = df[df["is_test"] == 1].copy()

X_train = train.drop(columns=DROP_COLS)
y_train = train[TARGET]
X_test  = test.drop(columns=DROP_COLS)
y_test  = test[TARGET]

params = joblib.load(PARAMS_PATH)
params["n_jobs"] = -1
params["random_state"] = 42

rf = RandomForestRegressor(**params)

# Yeo-Johnson supports negative values
pt = PowerTransformer(method="yeo-johnson", standardize=True)

model = TransformedTargetRegressor(
    regressor=rf,
    transformer=pt
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
mae = float(mean_absolute_error(y_test, pred))
mape_v = mape(y_test, pred)

print("RMSE:", round(rmse, 2))
print("MAE:", round(mae, 2))
print("MAPE:", round(mape_v, 2), "%")
