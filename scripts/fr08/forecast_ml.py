# This file is Forecasting with ML

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"

import warnings
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRFRegressor
from lightgbm import LGBMRegressor

from sklearn.exceptions import DataConversionWarning
from src.forecasting.ml_forecasting import (
    FeatureConfig,
    MissingValueConfig,
    MLForecast,
    ModelConfig,
    calculate_metrics,
)

from src.utils import ts_utils
from src.utils import plotting_utils
from src.utils.general import LogTime
from src.utils.ts_utils import  forecast_bias, metrics_adapter, mae, mse, mase
from tqdm.autonotebook import tqdm
from IPython.display import display, HTML
from src.utils.plotting_utils import plot_forecast, format_plot

np.random.seed(42)
tqdm.pandas()


# === Paths ===
preprocessed = Path("./data")
output = Path("data/B0005")
output.mkdir(exist_ok=True)
output_img = Path("imgs/ch08")
output_img.mkdir(exist_ok=True)


# === Load Data ===
try:
    train_df = pd.read_parquet(preprocessed/"selected_blocks_train_missing_imputed_feature_engg.parquet")
    # Read in the Validation dataset as test_df so that we predict on it
    test_df = pd.read_parquet(preprocessed/"selected_blocks_val_missing_imputed_feature_engg.parquet")
except FileNotFoundError:
    display(HTML("""
    <div class="alert alert-block alert-warning">
    <b>Warning!</b> File not found. Please make sure you have run in Chapter06
    </div>
    """))

#Loading the single step backtesting baselines for validation
try:
    baseline_metrics_df = pd.read_pickle(preprocessed/"B0005/single_step_backtesting_baseline_metrics_val_df.pkl")
    baseline_aggregate_metrics_df = pd.read_pickle(preprocessed/"B0005/single_step_backtesting_baseline_aggregate_metrics_val.pkl")
except FileNotFoundError:
    display(HTML("""
    <div class="alert alert-block alert-warning">
    <b>Warning!</b> File not found. Please make sure you have run Single Step Backtesting Baselines in Chapter08
    </div>
    """))

# === Feature Definition ===
feat_config = FeatureConfig(
    date="ds",
    target="y",
    continuous_features=[
        "y_lag_1", "y_lag_2", "y_lag_3", "y_lag_4", "y_lag_5",
        "y_lag_240", "y_lag_241", "y_lag_242", "y_lag_243", "y_lag_244",
        "y_lag_5760", "y_lag_5761", "y_lag_5762", "y_lag_5763", "y_lag_5764",
        "y_rolling_3_mean", "y_rolling_3_std",
        "y_rolling_6_mean", "y_rolling_6_std",
        "y_rolling_12_mean", "y_rolling_12_std",
        "y_rolling_48_mean", "y_rolling_48_std",
        "y_5760_seasonal_rolling_3_mean", "y_5760_seasonal_rolling_3_std",
        "y_40320_seasonal_rolling_3_mean", "y_40320_seasonal_rolling_3_std",
        "y_ewma_span_40320", "y_ewma_span_5760", "y_ewma_span_240",
        "ds_Elapsed",
        "ds_Month_sin_1", "ds_Month_sin_2", "ds_Month_sin_3", "ds_Month_sin_4", "ds_Month_sin_5",
        "ds_Month_cos_1", "ds_Month_cos_2", "ds_Month_cos_3", "ds_Month_cos_4", "ds_Month_cos_5",
        "ds_Hour_sin_1", "ds_Hour_sin_2", "ds_Hour_sin_3", "ds_Hour_sin_4", "ds_Hour_sin_5",
        "ds_Hour_cos_1", "ds_Hour_cos_2", "ds_Hour_cos_3", "ds_Hour_cos_4", "ds_Hour_cos_5",
        "ds_Minute_sin_1", "ds_Minute_sin_2"
    ],
    categorical_features=[
        "ds_Month", "ds_Quarter", "ds_Day", "ds_Dayofweek", "ds_Dayofyear",
        "ds_Hour", "ds_Minute", "ds_Second"
    ],
    boolean_features=[
        "ds_Is_quarter_end", "ds_Is_quarter_start", "ds_Is_year_end",
        "ds_Is_year_start", "ds_Is_month_start"
    ],
    index_cols=["ds"],
    exogenous_features=[
        "ds_Month", "ds_Quarter", "ds_Dayofweek", "ds_Hour", "ds_Minute",
        "ds_Elapsed"
    ]
)

# === Sample ===
selected_id = "B0005"
sample_train_df = train_df.loc[train_df.unique_id == selected_id, :]
sample_test_df = test_df.loc[test_df.unique_id == selected_id, :]

train_features, train_target, train_original_target = feat_config.get_X_y(
    sample_train_df, categorical=False, exogenous=False
)
# Loading the Validation as test
test_features, test_target, test_original_target = feat_config.get_X_y(
    sample_test_df, categorical=False, exogenous=False
)
del sample_train_df, sample_test_df

missing_value_config = MissingValueConfig(
    bfill_columns=[
        "y_lag_1", "y_lag_2", "y_lag_3", "y_lag_4", "y_lag_5",
        "y_lag_240", "y_lag_241", "y_lag_242", "y_lag_243", "y_lag_244",
        "y_lag_5760", "y_lag_5761", "y_lag_5762", "y_lag_5763", "y_lag_5764",
        "y_rolling_3_mean", "y_rolling_3_std",
        "y_rolling_6_mean", "y_rolling_6_std",
        "y_rolling_12_mean", "y_rolling_12_std",
        "y_rolling_48_mean", "y_rolling_48_std",
        "y_5760_seasonal_rolling_3_mean", "y_5760_seasonal_rolling_3_std",
        "y_40320_seasonal_rolling_3_mean", "y_40320_seasonal_rolling_3_std",
        "y_ewma_span_240", "y_ewma_span_5760", "y_ewma_span_40320"
    ],
    ffill_columns=[  
        "y_40320_seasonal_rolling_3_mean", "y_40320_seasonal_rolling_3_std",
    ],
    zero_fill_columns=[  
        "y_40320_seasonal_rolling_3_mean", "y_40320_seasonal_rolling_3_std",
    ],
)


# === Running ML models on a Sample ===
pred_df = pd.concat([train_target, test_target])
metric_record = []

metric_record += (
    baseline_metrics_df.loc[baseline_metrics_df.unique_id == "B0005"]
    .drop(columns="unique_id")
    .to_dict(orient="records")
)

def evaluate_model(
    model_config,
    feature_config,
    missing_config,
    train_features,
    train_target,
    test_features,
    test_target,
):
    ml_model = MLForecast(
        model_config=model_config,
        feature_config=feature_config,
        missing_config=missing_config,
    )
    ml_model.fit(train_features, train_target)
    y_pred = ml_model.predict(test_features)
    feat_df = ml_model.feature_importance()
    metrics = calculate_metrics(test_target, y_pred, model_config.name, train_target)
    return y_pred,  metrics, feat_df 

def highlight_abs_min(s, props=''):
    return np.where(s == np.nanmin(np.abs(s.values)), props, '')

# === Linear Regression ===
model_config = ModelConfig(
    model=LinearRegression(),
    name="Linear Regression",
    normalize=True,
    fill_missing=True,
)
with LogTime() as timer:
    y_pred, metrics, feat_df, = evaluate_model(
        model_config,
        feat_config,
        missing_value_config,
        train_features,
        train_target,
        test_features,
        test_target,
    )
metrics["Time Elapsed"] = timer.elapsed
metric_record.append(metrics)
pred_df = pred_df.join(y_pred)

pred_df = pred_df.reset_index()

# === Ridge Regression (L2) ===
model_config = ModelConfig(
    model=RidgeCV(), 
    name="Ridge Regression", 
    normalize=True, 
    fill_missing=True
)
with LogTime() as timer:
    y_pred, metrics, feat_df,= evaluate_model(
        model_config,
        feat_config,
        missing_value_config,
        train_features,
        train_target,
        test_features,
        test_target,
    )
metrics["Time Elapsed"] = timer.elapsed
metric_record.append(metrics)
pred_df = pred_df.join(y_pred)

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# === Lasso Regression (L1) ===
model_config = ModelConfig(
    model=LassoCV(), 
    name="Lasso Regression", 
    normalize=True, 
    fill_missing=True
)
with LogTime() as timer:
    y_pred, metrics, feat_df, = evaluate_model(
        model_config,
        feat_config,
        missing_value_config,
        train_features,
        train_target,
        test_features,
        test_target,
    )
metrics["Time Elapsed"] = timer.elapsed
metric_record.append(metrics)
pred_df = pred_df.join(y_pred)

# === Decision Tree ===
model_config = ModelConfig(
    model=DecisionTreeRegressor(max_depth=4, random_state=42),
    name="Decision Tree",
    normalize=False,
    fill_missing=True,
)
with LogTime() as timer:
    y_pred, metrics, feat_df = evaluate_model(
        model_config,
        feat_config,
        missing_value_config,
        train_features,
        train_target,
        test_features,
        test_target,
    )
metrics["Time Elapsed"] = timer.elapsed
metric_record.append(metrics)
pred_df = pred_df.join(y_pred)

# === Bagging and Boosting Trees ===

# === Random Forest ===
model_config = ModelConfig(
    model=RandomForestRegressor(random_state=42, max_depth=4),
    name="Random Forest",
    normalize=False,
    fill_missing=True,
)
with LogTime() as timer:
    y_pred, metrics, feat_df = evaluate_model(
        model_config,
        feat_config,
        missing_value_config,
        train_features,
        train_target,
        test_features,
        test_target,
    )
metrics["Time Elapsed"] = timer.elapsed
metric_record.append(metrics)
pred_df = pred_df.join(y_pred)


# === XGBoost Random Forest ===

model_config = ModelConfig(
    model=XGBRFRegressor(random_state=42, max_depth=4),
    name="XGB Random Forest",
    normalize=False,
    fill_missing=False,
)
with LogTime() as timer:
    y_pred, metrics, feat_df = evaluate_model(
        model_config,
        feat_config,
        missing_value_config,
        train_features,
        train_target,
        test_features,
        test_target,
    )
metrics["Time Elapsed"] = timer.elapsed
metric_record.append(metrics)
pred_df = pred_df.join(y_pred)


# === LightGBM ===

model_config = ModelConfig(
    model=LGBMRegressor(random_state=42),
    name="LightGBM",
    normalize=False,
    fill_missing=False,
)
with LogTime() as timer:
    y_pred, metrics, feat_df = evaluate_model(
        model_config,
        feat_config,
        missing_value_config,
        train_features,
        train_target,
        test_features,
        test_target,
    )
metrics["Time Elapsed"] = timer.elapsed
metric_record.append(metrics)
pred_df = pred_df.join(y_pred)


# ===Summary ===
formatted = pd.DataFrame(metric_record).style.format({"MAE": "{:.4f}", 
                          "MSE": "{:.4f}", 
                          "MASE": "{:.4f}", 
                          "Forecast Bias": "{:.2f}%"})
formatted.highlight_min(color='lightgreen', subset=["MAE","MSE","MASE"]).apply(highlight_abs_min, props='color:black;background-color:lightgreen', axis=0, subset=['Forecast Bias'])


# === Running ML Forecast for all consumers ===

# Running Lasso Regression, XGB Random Forest, and LightGBM

lcl_ids = sorted(train_df.unique_id.unique())
models_to_run = [
    ModelConfig(
        model=LassoCV(), name="Lasso Regression", normalize=True, fill_missing=True
    ),
    ModelConfig(
        model=XGBRFRegressor(random_state=42, max_depth=4),
        name="XGB Random Forest",
        normalize=False,
        fill_missing=False,
    ),
    ModelConfig(
        model=LGBMRegressor(random_state=42),
        name="LightGBM",
        normalize=False,
        fill_missing=False,
    ),
]

all_preds = []
all_metrics = []

with LogTime() as timer:
    for lcl_id in tqdm(lcl_ids):
        for model_config in models_to_run:
            model_config = model_config.clone()
            X_train, y_train, _ = feat_config.get_X_y(
                train_df.loc[train_df.unique_id == lcl_id, :],
                categorical=False,
                exogenous=False,
            )
            X_test, y_test, _ = feat_config.get_X_y(
                test_df.loc[test_df.unique_id == lcl_id, :], categorical=False, exogenous=False
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred, metrics, feat_df = evaluate_model(
                    model_config,
                    feat_config,
                    missing_value_config,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                )
            y_pred.name = "predictions"
            y_pred = y_pred.to_frame()
            y_pred["unique_id"] = lcl_id
            y_pred["Algorithm"] = model_config.name
            metrics["unique_id"] = lcl_id
            metrics["Algorithm"] = model_config.name
            y_pred["voltage_measured"] = y_test.values
            all_preds.append(y_pred)
            all_metrics.append(metrics)


pred_df = pd.concat(all_preds)

metrics_df = pd.DataFrame(all_metrics)


metrics = baseline_aggregate_metrics_df.reset_index().rename(columns={"index":"Algorithm"}).to_dict(orient="records")

for model_config in models_to_run:
    pred_mask = pred_df.Algorithm==model_config.name
    metric_mask = metrics_df.Algorithm==model_config.name
    metrics.append({
    "Algorithm": model_config.name,
    "MAE": ts_utils.mae(pred_df.loc[pred_mask,"voltage_measured"], pred_df.loc[pred_mask,"predictions"]),
    "MSE": ts_utils.mse(pred_df.loc[pred_mask,"voltage_measured"], pred_df.loc[pred_mask,"predictions"]),
    "meanMASE": metrics_df.loc[metric_mask, "MASE"].mean(),
    "Forecast Bias": ts_utils.forecast_bias_aggregate(pred_df.loc[pred_mask,"voltage_measured"], pred_df.loc[pred_mask,"predictions"])
})

agg_metrics_df = pd.DataFrame(metrics)
agg_metrics_df.style.format({"MAE": "{:.4f}", 
                          "MSE": "{:.4f}", 
                          "meanMASE": "{:.4f}", 
                          "Forecast Bias": "{:.2f}%"}).highlight_min(color='lightgreen', subset=["MAE","MSE","meanMASE"]).apply(highlight_abs_min, props='color:black;background-color:lightgreen', axis=0, subset=['Forecast Bias'])


html = agg_metrics_df.to_html()

# === Saving the Baseline Forecasts and Metrics ===
pred_df.to_pickle(output/"ml_single_step_prediction_val_df.pkl")
metrics_df.to_pickle(output/"ml_single_step_metrics_val_df.pkl")
agg_metrics_df.to_pickle(output/"ml_single_step_aggregate_metrics_val.pkl")

# === Using Exogenous Variables ===
# run LightGBM, which was our best performing algorithm with exogenous variables

lcl_ids = sorted(train_df.unique_id.unique())
models_to_run = [
    ModelConfig(model = LGBMRegressor(random_state=42), name="LightGBM", normalize=False, fill_missing=False)
]

all_preds = []
all_metrics = []
for lcl_id in tqdm(lcl_ids):
    for model_config in models_to_run:
        model_config = model_config.clone()
        X_train, y_train, _ = feat_config.get_X_y(train_df.loc[train_df.unique_id==lcl_id,:], categorical=False, exogenous=True)
        X_test, y_test, _ = feat_config.get_X_y(test_df.loc[test_df.unique_id==lcl_id,:], categorical=False, exogenous=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred, metrics, feat_df = evaluate_model(model_config, feat_config, missing_value_config, X_train, y_train, X_test, y_test)
        y_pred.name = "predictions"
        y_pred = y_pred.to_frame()
        y_pred['unique_id'] = lcl_id
        y_pred['Algorithm'] = model_config.name+"_w_exog"
        metrics["unique_id"] = lcl_id
        metrics["Algorithm"] = model_config.name+"_w_exog"
        y_pred['voltage_measured'] = y_test.values
        all_preds.append(y_pred)
        all_metrics.append(metrics)

pred_w_ex_df = pd.concat(all_preds)

metrics_w_ex_df = pd.DataFrame(all_metrics)


# === Evaluation of ML Forecast with Exogenous ===

metrics = baseline_aggregate_metrics_df.reset_index().rename(columns={"index":"Algorithm"}).to_dict(orient="records")


metrics.append(agg_metrics_df.iloc[4].to_dict())

for model_config in models_to_run:
    pred_mask = pred_w_ex_df.Algorithm==model_config.name+"_w_exog"
    metric_mask = metrics_w_ex_df.Algorithm==model_config.name+"_w_exog"
    metrics.append({
    "Algorithm": model_config.name+"_w_exog",
    "MAE": ts_utils.mae(pred_w_ex_df.loc[pred_mask,"voltage_measured"], pred_w_ex_df.loc[pred_mask,"predictions"]),
    "MSE": ts_utils.mse(pred_w_ex_df.loc[pred_mask,"voltage_measured"], pred_w_ex_df.loc[pred_mask,"predictions"]),
    "meanMASE": metrics_w_ex_df.loc[metric_mask, "MASE"].mean(),
    "Forecast Bias": ts_utils.forecast_bias_aggregate(pred_w_ex_df.loc[pred_mask,"voltage_measured"], pred_w_ex_df.loc[pred_mask,"predictions"])
})

agg_metrics_w_ex_df = pd.DataFrame(metrics)
agg_metrics_w_ex_df.style.format({"MAE": "{:.3f}", 
                          "MSE": "{:.3f}", 
                          "meanMASE": "{:.3f}", 
                          "Forecast Bias": "{:.2f}%"}).highlight_min(color='lightgreen', subset=["MAE","MSE","meanMASE"]).apply(highlight_abs_min, props='color:black;background-color:lightgreen', axis=0, subset=['Forecast Bias'])


html = agg_metrics_w_ex_df.to_html()

