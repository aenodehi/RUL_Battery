# import os
# import time

import numpy as np
import pandas as pd
# import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"

import warnings
from pathlib import Path

# import humanize

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from src.forecasting.ml_forecasting import (
    FeatureConfig,
    MissingValueConfig,
    MLForecast,
    ModelConfig,
    calculate_metrics,
)
from src.utils import plotting_utils
from src.utils.general import LogTime
from src.utils.ts_utils import  forecast_bias, metrics_adapter, mae, mse, mase
from tqdm.autonotebook import tqdm
from IPython.display import display, HTML
from src.utils.plotting_utils import plot_forecast, format_plot

# from itertools import cycle

np.random.seed(42)
tqdm.pandas()


# === Paths ===
preprocessed = Path("./data")
output = Path("data/B0005")
output.mkdir(exist_ok=True)
output_img = Path("imgs")
output_img.mkdir(exist_ok=True)


# === Load Data ===
#Readin the missing value imputed and train test split data
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
    # baseline_metrics_test_df = pd.read_pickle(output/"single_step_backtesting_baseline_metrics_test_df.pkl")
    # baseline_aggregate_metrics_test_df = pd.read_pickle(output/"single_step_backtesting_baseline_aggregate_metrics_test.pkl")
except FileNotFoundError:
    display(HTML("""
    <div class="alert alert-block alert-warning">
    <b>Warning!</b> File not found. Please make sure you have run Single Step Backtesting Baselines in Chapter08
    </div>
    """))

#print(train_df.columns)

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
        # You could list continuous + categorical exogenous features here
        # (i.e., anything not lagged or derived from `y`)
        "ds_Month", "ds_Quarter", "ds_Dayofweek", "ds_Hour", "ds_Minute",
        "ds_Elapsed"
    ]
)

# === Sample ===
selected_id = "B0005"
# sample_train_df = train_df.loc[train_df.unique_id == "B0005", :]
sample_train_df = train_df.loc[train_df.unique_id == selected_id]
# sample_test_df = test_df.loc[test_df.unique_id == "B0005", :]
sample_test_df = test_df.loc[test_df.unique_id == selected_id]

train_features, train_target, train_original_target = feat_config.get_X_y(
    sample_train_df, categorical=False, exogenous=False
)
# Loading the Validation as test
test_features, test_target, test_original_target = feat_config.get_X_y(
    sample_test_df, categorical=False, exogenous=False
)
del sample_train_df, sample_test_df

#nc = train_features.isnull().sum()
#print(nc[nc>0])

#nc = test_features.isnull().sum()
#print(nc[nc>0])

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
    ffill_columns=[],  # e.g., weather or calendar indicators if needed
    zero_fill_columns=[]  # For categorical one-hot or flags, if any
)


# === Running ML models on a Sample ===
pred_df = pd.concat([train_target, test_target])
metric_record = []

metric_record += (
    baseline_metrics_df.loc[baseline_metrics_df.unique_id == "B0005"]
    .drop(columns="unique_id")
    .to_dict(orient="records")
)
#print(metric_record)

# Manually impute missing values before model training
train_features = missing_value_config.impute_missing_values(train_features)
test_features = missing_value_config.impute_missing_values(test_features)


# Drop columns that are still entirely NaN after all imputation
all_nan_cols = df.columns[df.isnull().all()]
if len(all_nan_cols) > 0:
    print(f"[INFO] Dropping columns with all NaNs: {list(all_nan_cols)}")
    df = df.drop(columns=all_nan_cols)

train_features = train_features.loc[:, train_features.columns.isin(train_df.columns)]
test_features = test_features.loc[:, test_features.columns.isin(test_df.columns)]


def drop_zero_variance_features(df: pd.DataFrame, name: str) -> pd.DataFrame:
    zero_var_cols = df.columns[df.std() == 0]
    if not zero_var_cols.empty:
        print(f"[INFO] Dropping zero-variance columns from {name}: {list(zero_var_cols)}")
        df = df.drop(columns=zero_var_cols)
    return df

train_features = drop_zero_variance_features(train_features, "train_features")
test_features = test_features[train_features.columns]  # Align test columns to train

print("Train features shape:", train_features.shape)
print("Test features shape:", test_features.shape)

# Debug remaining NaNs
nan_summary = train_features.isnull().sum()
nan_cols = nan_summary[nan_summary > 0]
if not nan_cols.empty:
    print("[ERROR] NaNs remaining in train_features:")
    print(nan_cols)

# check there are no NaNs left
assert not train_features.isnull().any().any(), "train_features still contains NaNs"
assert not test_features.isnull().any().any(), "test_features still contains NaNs"
assert not (train_features.std() == 0).any(), "Some train features have zero variance"

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
    model=make_pipeline(StandardScaler(), LinearRegression()),
    name="Linear Regression",
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
# pred_df = pred_df.join(y_pred)
# pred_df["y_pred"] = y_pred
pred_df.loc[test_target.index, "y_pred"] = y_pred
print(metrics)




