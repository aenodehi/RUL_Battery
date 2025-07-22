import pandas as pd
import sys
from pathlib import Path
import time
import os
import plotly.express as px
import plotly.graph_objects as go

from typing import List, Optional
from functools import partial
from src.utils import ts_utils
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
output = Path("data/B0005")
output.mkdir(exist_ok=True)

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

# === 2) Evaluate on Validation ===
baseline_val_metrics_df = evaluate(
                        df   = crossval_val_df.drop(['cutoff'], axis =1 ), 
                        metrics  = [mse, mae, rmse, fcst_mase, forecast_bias_NIXTLA],
                        models=model_names,
                        train_df  = tr_vl[['ds', 'unique_id', 'y']],
                        id_col = 'unique_id',
                        time_col = 'ds',
                        target_col = 'y',
                        )
#print(baseline_val_metrics_df.head())


baseline_val_metrics_df_pivot = (baseline_val_metrics_df
    .melt(id_vars = ['unique_id','metric'], value_vars = model_names, var_name ='Algorithm', value_name='score')
    .pivot_table(index = ['unique_id','Algorithm'], columns = 'metric', values = 'score', observed = 'True')
).reset_index()
#print(baseline_val_metrics_df_pivot.head())

# === 3) Evaluate on Test ===

baseline_test_metrics_df = evaluate(
                        df   = crossval_test_df.drop(['cutoff'], axis =1 ), 
                        metrics  = [mse, mae, rmse, fcst_mase, forecast_bias_NIXTLA],
                        models=model_names,
                        train_df  = tr_vl[['ds', 'unique_id', 'y']],
                        id_col = 'unique_id',
                        time_col = 'ds',
                        target_col = 'y',
                        )
#print(baseline_test_metrics_df.head())


baseline_test_metrics_df_pivot = (baseline_test_metrics_df
    .melt(id_vars = ['unique_id','metric'], value_vars = model_names, var_name ='Algorithm', value_name='score')
    .pivot_table(index = ['unique_id','Algorithm'], columns = 'metric', values = 'score', observed = 'True')
).reset_index()
#print(baseline_test_metrics_df_pivot.head())


# === 4) Prepare Prediction DataFrames ===
# Naive

naive_pred_val_df = (crossval_val_df
                     .rename(columns = {'y':'voltage_measured','Naive':'naive_predictions'})
                     .drop(['cutoff','SeasonalNaive'],axis=1).set_index('ds')
)

naive_pred_test_df = (crossval_test_df
                     .rename(columns = {'y':'voltage_measured','Naive':'naive_predictions'})
                     .drop(['cutoff','SeasonalNaive'],axis=1).set_index('ds')
)

# Seasonal-naive

snaive_pred_val_df = (crossval_val_df
                     .rename(columns = {'y':'voltage_measured','SeasonalNaive':'snaive_predictions'})
                     .drop(['cutoff','Naive'],axis=1).set_index('ds')
)

snaive_pred_test_df = (crossval_test_df
                     .rename(columns = {'y':'voltage_measured','SeasonalNaive':'snaive_predictions'})
                     .drop(['cutoff','Naive'],axis=1).set_index('ds')
)

baseline_metrics_val_df = baseline_val_metrics_df_pivot.copy()
baseline_metrics_test_df = baseline_test_metrics_df_pivot.copy()

#print(naive_pred_val_df.head())

# === Overall Metrics ===
overall_metrics_naive_val = {
    "MAE": ts_utils.mae(crossval_val_df["y"], crossval_val_df["Naive"]),
    "MSE": ts_utils.mse(crossval_val_df["y"], crossval_val_df["Naive"]),
    "meanMASE": baseline_val_metrics_df[baseline_val_metrics_df.metric =='mase']["Naive"].mean(),
    "Forecast Bias": ts_utils.forecast_bias_aggregate(crossval_val_df["y"], crossval_val_df["Naive"])
}
#print(overall_metrics_naive_val)

overall_metrics_snaive_val = {
    "MAE": ts_utils.mae(crossval_val_df["y"], crossval_val_df["SeasonalNaive"]),
    "MSE": ts_utils.mse(crossval_val_df["y"], crossval_val_df["SeasonalNaive"]),
    "meanMASE": baseline_val_metrics_df[baseline_val_metrics_df.metric =='mase']["SeasonalNaive"].mean(),
    "Forecast Bias": ts_utils.forecast_bias_aggregate(crossval_val_df["y"], crossval_val_df["SeasonalNaive"])
}

overall_metrics_naive_test = {
    "MAE": ts_utils.mae(crossval_test_df["y"], crossval_test_df["Naive"]),
    "MSE": ts_utils.mse(crossval_test_df["y"], crossval_test_df["Naive"]),
    "meanMASE": baseline_test_metrics_df[baseline_test_metrics_df.metric =='mase']["Naive"].mean(),
    "Forecast Bias": ts_utils.forecast_bias_aggregate(crossval_test_df["y"], crossval_test_df["Naive"])
}


overall_metrics_snaive_test = {
    "MAE": ts_utils.mae(crossval_test_df["y"], crossval_test_df["SeasonalNaive"]),
    "MSE": ts_utils.mse(crossval_test_df["y"], crossval_test_df["SeasonalNaive"]),
    "meanMASE": baseline_test_metrics_df[baseline_test_metrics_df.metric =='mase']["SeasonalNaive"].mean(),
    "Forecast Bias": ts_utils.forecast_bias_aggregate(crossval_test_df["y"], crossval_test_df["SeasonalNaive"])
}


# === Evaluation of Baseline Forecast ===

agg_metric_val_df = pd.DataFrame([overall_metrics_naive_val, overall_metrics_snaive_val], index=["Naive","Seasonal Naive"])

agg_metric_val_df.style.format({"MAE": "{:.3f}", 
                          "MSE": "{:.3f}", 
                          "meanMASE": "{:.3f}", 
                          "Forecast Bias": "{:.2f}%"}).highlight_min(color='lightgreen')

#print(agg_metric_val_df)


agg_metric_test_df = pd.DataFrame([overall_metrics_naive_test, overall_metrics_snaive_test], index=["Naive","Seasonal Naive"])

agg_metric_test_df.style.format({"MAE": "{:.3f}", 
                          "MSE": "{:.3f}", 
                          "meanMASE": "{:.3f}", 
                          "Forecast Bias": "{:.2f}%"}).highlight_min(color='lightgreen')

#print(agg_metric_val_df)

# === Fig ===
fig_mase = px.histogram(baseline_val_metrics_df_pivot, 
                   x="mase", 
                   color="Algorithm",
                   pattern_shape="Algorithm", 
                   marginal="box", 
                   nbins=500, 
                   barmode="overlay",
                   histnorm="probability density")
fig_mase = format_plot(fig_mase, xlabel="MASE", ylabel="Probability Density", title="Distribution of MASE in the dataset")
fig_mase.update_layout(xaxis_range=[0,3.2])
# fig_mase.write_image("imgs/chapter_8/mase_dist.png")
#fig_mase.show()

fig_mae = px.histogram(baseline_val_metrics_df_pivot, 
                   x="mae", 
                   color="Algorithm",
                   pattern_shape="Algorithm", 
                   marginal="box", 
                   nbins=100, 
                   barmode="overlay",
                   histnorm="probability density")
fig_mae = format_plot(fig_mae, xlabel="MAE", ylabel="Probability Density", title="Distribution of MAE in the dataset")
# fig_mae.write_image("imgs/chapter_8/mae_dist.png")
#fig_mae.show()

fig_mse = px.histogram(baseline_val_metrics_df_pivot, 
                   x="mse", 
                   color="Algorithm",
                   pattern_shape="Algorithm", 
                   marginal="box", 
                   nbins=500, 
                   barmode="overlay",
                   histnorm="probability density")
fig_mse = format_plot(fig_mse, xlabel="MSE", ylabel="Probability Density", title="Distribution of MSE in the dataset")
fig_mse.update_layout(xaxis_range=[0,0.6])
# fig_mse.write_image("imgs/chapter_8/mse_dist.png")
#fig_mse.show()

fig_bias = px.histogram(baseline_val_metrics_df_pivot, 
                   x="forecast_bias", 
                   color="Algorithm",
                   pattern_shape="Algorithm", 
                   marginal="box", 
                   nbins=250,
                   barmode="overlay",
                   histnorm="probability density")
fig_bias = format_plot(fig_bias, xlabel="Forecast Bias", ylabel="Probability Density", title="Distribution of Forecast Bias in the dataset")
fig_bias.update_layout(xaxis_range=[-40,40])
# fig_bias.write_image("imgs/chapter_8/bias_dist.png")
#fig_bias.show()


# ===  ===
baseline_pred_val_df = naive_pred_val_df.reset_index().merge(snaive_pred_val_df.reset_index().drop(columns='voltage_measured'), on=['ds','unique_id'], how='outer')
baseline_pred_test_df = naive_pred_test_df.reset_index().merge(snaive_pred_test_df.reset_index().drop(columns='voltage_measured'), on=['ds','unique_id'], how='outer')
print(baseline_pred_val_df.head())
print(baseline_pred_test_df.head())

# === Saving the Baseline Forecasts and Metrics ===
baseline_pred_val_df.to_pickle(output/"single_step_backtesting_baseline_prediction_val_df.pkl")
baseline_metrics_val_df.to_pickle(output/"single_step_backtesting_baseline_metrics_val_df.pkl")
agg_metric_val_df.to_pickle(output/"single_step_backtesting_baseline_aggregate_metrics_val.pkl")
baseline_pred_test_df.to_pickle(output/"single_step_backtesting_baseline_prediction_test_df.pkl")
baseline_metrics_test_df.to_pickle(output/"single_step_backtesting_baseline_metrics_test_df.pkl")
agg_metric_test_df.to_pickle(output/"single_step_backtesting_baseline_aggregate_metrics_test.pkl")
