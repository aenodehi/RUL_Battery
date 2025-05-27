import pandas as pd
import sys
from pathlib import Path
import time
import os

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.evaluation import evaluate_performance
from src.models.statsforecast_models import Naive, ARIMA, Theta
from utilsforecast.losses import mase, mae, mse, rmse, smape
from src.models import statsforecast_models
from src.utils.ts_utils import forecast_bias
from src.utils.plotting_utils import plot_forecast, format_plot

# === Paths ===
preprocessed = Path("./data")
output_dir = Path("imgs")
output_dir.mkdir(exist_ok=True)

# === Load Data ===
train_df = pd.read_parquet(preprocessed / "B0005_train.parquet")
val_df = pd.read_parquet(preprocessed / "B0005_val.parquet")
test_df = pd.read_parquet(preprocessed / "B0005_test.parquet")

ts_train = train_df[["Voltage_measured"]].reset_index()
ts_val = val_df[["Voltage_measured"]].reset_index()
ts_test = test_df[["Voltage_measured"]].reset_index()

for df in [ts_train, ts_val, ts_test]:
    df['datetime_'] = pd.to_datetime(df['datetime_'])
    df['unique_id'] = 'B0005'

#print("Columns in ts_train after resetting index:", ts_train.columns)
#print(ts_train.head())
#print(ts_train.tail())

# === Forecast ===
freq = "15s"
metrics_df = pd.DataFrame()

metrics = [mase, mae, mse, rmse, smape, forecast_bias]

results, metrics_df = evaluate_performance(
    ts_train=ts_train,
    ts_target=ts_val,
    #ts_target=ts_test,
    models=[Theta(season_length =48, decomposition_type = 'additive' )],
    freq=freq,
    level=[],
    id_col='unique_id',
    time_col='datetime_',
    target_col='Voltage_measured',
    h=len(ts_val),
    metric_df=metrics_df
)

print(metrics_df)

# === Plot ===
model_name = ['Theta']
model_display_name = ['Theta']

fig = plot_forecast(
    results,
    forecast_columns=model_name,
    forecast_display_names=model_display_name,
    timestamp_col='datetime_',
    target_col='Voltage_measured'
)

fig = format_plot(
    fig,
    title=f"{model_name[0]}: "
          f"MAE: {metrics_df.loc[metrics_df.Model == model_name[0], 'MAE'].iloc[0]:.4f} | "
          f"MASE: {metrics_df.loc[metrics_df.Model == model_name[0], 'MASE'].iloc[0]:.4f} | "
          f"BIAS: {metrics_df.loc[metrics_df.Model == model_name[0], 'Bias'].iloc[0]:.4f}"
)

fig.update_xaxes(
    type="date",
    range=[
        pd.to_datetime(results['datetime_'].min()),
        pd.to_datetime(results['datetime_'].max())
    ]
)

fig.write_image(str(output_dir / "theta.png"))
fig.write_html(str(output_dir / "theta_forecast.html"))

print("✅ Forecast plots saved:")
print(f" - Static PNG: {output_dir / 'theta.png'}")
print(f" - Interactive HTML: {output_dir / 'theta_forecast.html'}")

fig.show()

