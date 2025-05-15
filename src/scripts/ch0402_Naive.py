import pandas as pd
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.evaluation import evaluate_performance
from src.models.statsforecast_models import Naive
from utilsforecast.losses import mase, mae, mse, rmse, smape
#from statsforecast.losses import mase, mae, mse, rmse, smape
from src.models import statsforecast_models
from src.utils.ts_utils import forecast_bias

preprocessed = Path("./data")

#df = pd.read_csv("./data/B0005_discharge_adjusted.csv", index_col = "datetime_", parse_dates = True)
#print(df.head())

train_df = pd.read_parquet(preprocessed / "B0005_train.parquet")
val_df = pd.read_parquet(preprocessed / "B0005_val.parquet")
test_df = pd.read_parquet(preprocessed / "B0005_test.parquet")

ts_train = train_df[["Voltage_measured"]]
ts_val = val_df[["Voltage_measured"]]
ts_test = test_df[["Voltage_measured"]]

ts_train = ts_train.reset_index()
ts_val = ts_val.reset_index()
ts_test = ts_test.reset_index()

ts_train['datetime_'] = pd.to_datetime(ts_train['datetime_'])
ts_val['datetime_'] = pd.to_datetime(ts_val['datetime_'])
ts_test['datetime_'] = pd.to_datetime(ts_test['datetime_'])

for df in [ts_train, ts_val, ts_test]:
    df['unique_id'] = 'B0005'

print("Columns in ts_train after resetting index:", ts_train.columns)

#print(f"Column names in the dataset: {df.columns.tolist()}")
#print(df.head())
print(ts_train.head())

# Baseline Forecasts
pred_df = pd.concat([ts_train, ts_val])

# NAIVE FORECAST
freq = "1min"
metrics_df = pd.DataFrame()

metrics = [mase, mae, mse, rmse, smape, forecast_bias]

results, metrics_df = evaluate_performance(
    ts_train=ts_train, 
    ts_test=ts_val, 
    models=[Naive()],
    freq=freq,
    level=[],  # Ensure this is correct or adjust as necessary
    id_col='unique_id',
    time_col= 'datetime_',
    target_col='Voltage_measured',
    h=len(ts_val),
    metric_df=metrics_df  # Pass None or an existing DataFrame if you want to append results
)


print(metrics_df)
