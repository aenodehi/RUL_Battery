import pandas as pd
import sys
from pathlib import Path
import time
import os
from typing import List, Optional
from functools import partial
#from src.utils.evaluation import evaluate_performance
from src.models.statsforecast_models import Naive, SeasonalNaive
from utilsforecast.losses import mase, mae, mse, rmse, smape
from src.models import statsforecast_models
from statsforecast.core import StatsForecast
from src.utils.ts_utils import forecast_bias
from src.utils.ts_utils import forecast_bias_aggregate, forecast_bias_NIXTLA
from utilsforecast.evaluation import evaluate
from src.utils.plotting_utils import plot_forecast, format_plot

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))

# === Paths ===
preprocessed = Path("./data")
output_dir = Path("imgs")
output_dir.mkdir(exist_ok=True)

# === Load Data ===
try:
    train_df = pd.read_parquet(preprocessed/"B0005_train.parquet")
    val_df = pd.read_parquet(preprocessed/"B0005_val.parquet")
    test_df = pd.read_parquet(preprocessed/"B0005_test.parquet")

    #print("Train Min and Max Date",train_df.index.min(), train_df.index.max())
    #print("Val Min and Max Date",val_df.index.min(), val_df.index.max())
    #print("Test Min and Max Date",test_df.index.min(), test_df.index.max())
except FileNotFoundError:
    display(HTML("""
    <div class="alert alert-block alert-warning">
    <b>Warning!</b> File not found. Please make sure you have run 01-Feature Engineering.ipynb in Chapter06
    </div>
    """))

# === Prepare Data for Forecasting ===
ts_train = train_df[["Voltage_measured"]].reset_index()
ts_val = val_df[["Voltage_measured"]].reset_index()
ts_test = test_df[["Voltage_measured"]].reset_index()

for df in [ts_train, ts_val, ts_test]:
    df.rename(columns={"datetime_": "ds", "Voltage_measured": "y"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"])
    df["unique_id"] = "B0005"

tr = ts_train.copy()
vl = ts_val.copy()
ts = ts_test.copy()

tr_vl = pd.concat([tr, vl], ignore_index=True)
tr_vl_ts = pd.concat([tr, vl, ts], ignore_index=True)

models = [
    Naive(),
    SeasonalNaive(season_length=5760)  # 5760 = 1 week @ 15s intervals
]
model_names = [model.__class__.__name__ for model in models]

sf = StatsForecast(
    models=models,
    freq='15s',
    n_jobs=-1
)

# === Cross-validation on Validation Set ===
crossval_val_df = sf.cross_validation(
    df=tr_vl,
    h=1,
    step_size=1,
    n_windows=len(vl["ds"].unique()),
    id_col="unique_id",
    time_col="ds",
    target_col="y",
    level=[],
)

# === Cross-validation on Test Set ===
crossval_test_df = sf.cross_validation(
    df=tr_vl_ts,
    h=1,
    step_size=1,
    n_windows=len(ts["ds"].unique()),
    id_col="unique_id",
    time_col="ds",
    target_col="y",
    level=[],
)

# === Print Result Sample ===
#print("📈 Validation Forecast Sample:")
#print(crossval_val_df.head())

#print("📈 Test Forecast Sample:")
#print(crossval_test_df.head())

# === 1) Define wrapped metrics ===
fcst_mase = partial(mase, seasonality=1)
fcst_mase.__name__ = "mase"
forecast_bias_NIXTLA.__name__ = "forecast_bias"

baseline_val_metrics_df = evaluate(
                        df   = crossval_val_df.drop(['cutoff'], axis =1 ), 
                        metrics  = [mse, mae, rmse, fcst_mase, forecast_bias_NIXTLA],
                        models=model_names,
                        train_df  = tr_vl[['ds', 'unique_id', 'y']],
                        id_col = 'unique_id',
                        time_col = 'ds',
                        target_col = 'y',
                        )
print(baseline_val_metrics_df.head())
