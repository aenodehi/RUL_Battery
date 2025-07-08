import os
import time
import shutil

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import joblib
pio.templates.default = "plotly_white"

import warnings
from pathlib import Path

import torch
import typing
import collections
import omegaconf
from types import SimpleNamespace
torch.serialization.add_safe_globals([
    dict, list, tuple, int, float, bool,
    typing.Any,
    SimpleNamespace,
    collections.defaultdict,
    omegaconf.dictconfig.DictConfig,
    omegaconf.base.ContainerMetadata,
    omegaconf.base.Metadata,   
    omegaconf.listconfig.ListConfig,
    omegaconf.nodes.AnyNode,
])

import humanize

from sklearn.preprocessing import StandardScaler
from src.forecasting.ml_forecasting import (
    FeatureConfig,
    MissingValueConfig,
    MLForecast,
    ModelConfig,
    calculate_metrics,
)
from src.utils import plotting_utils
from src.utils.general import LogTime
from src.utils.ts_utils import metrics_adapter, forecast_bias, mae, mase, mse
from tqdm.autonotebook import tqdm
from src.forecasting.ml_forecasting import calculate_metrics
from src.utils import ts_utils
from IPython.display import display, HTML

from pytorch_tabular import TabularModel
from pytorch_tabular.models import FTTransformerConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig

np.random.seed(42)
tqdm.pandas()

def evaluate_forecast(y_pred, test_target, train_target, model_name):
    metric_l = []
    for _id in tqdm(test_target.index.get_level_values(0).unique(), desc="Calculating metrics..."):
        target = test_target.xs(_id)
        _y_pred = y_pred.xs(_id)
        history = train_target.xs(_id)
        metric_l.append(
            calculate_metrics(target, _y_pred, name=model_name, y_train=history)
        )
    eval_metrics_df = pd.DataFrame(metric_l)
    agg_metrics = {
            "Algorithm": model_name,
            "MAE": ts_utils.mae(
                test_target, y_pred
            ),
            "MSE": ts_utils.mse(
                test_target, y_pred
            ),
            "meanMASE": eval_metrics_df.loc[:, "MASE"].mean(),
            "Forecast Bias": ts_utils.forecast_bias_aggregate(
                test_target, y_pred
            )
    }
    return agg_metrics, eval_metrics_df

def highlight_abs_min(s, props=''):
    return np.where(s == np.nanmin(np.abs(s.values)), props, '')

# === Paths ===
preprocessed = Path("./data")
output = Path("data/B0005")
output.mkdir(exist_ok=True)
output_img = Path("imgs/ch13")
output_img.mkdir(exist_ok=True)

# === Load Data ===

try:
    train_df = pd.read_parquet(preprocessed/"selected_blocks_train_missing_imputed_feature_engg.parquet")
    test_df = pd.read_parquet(preprocessed / "selected_blocks_val_missing_imputed_feature_engg.parquet")
except FileNotFoundError:
    display(HTML("""
    <div class="alert alert-block alert-warning">
    <b>Warning!</b> File not found. Please make sure you have run in Chapter06 
    </div>
    """))
#print(train_df.columns)
#print(test_df.columns)

sel_lclids = train_df.unique_id.unique().tolist()
target = "y"
index_cols = ["unique_id", "ds"]

train_df.set_index(index_cols, inplace=True, drop=False)
test_df.set_index(index_cols, inplace=True, drop=False)
pred_df = pd.concat([train_df[[target]], test_df[[target]]])

# Loading the GFM Forecast and calculating the aggregate metrics on selected unique_ids
try:
    global_ml_fc_df = pd.read_pickle(output/"gfm_predictions_val_df.pkl")
    global_ml_fc_df = global_ml_fc_df.loc[global_ml_fc_df.index.get_level_values(0).isin(sel_lclids)]

    baseline_agg_metrics, baseline_metrics_df = evaluate_forecast(y_pred=global_ml_fc_df["GFM+Meta  (NativeLGBM)"],
                                                                  test_target = global_ml_fc_df["y"], train_target = train_df["y"], model_name="GFM+Meta  (NativeLGBM)")
except FileNotFoundError:
    display(HTML("""
    <div class="alert alert-block alert-warning">
    <b>Warning!</b> File not found. Please make sure you have run Global Forecasting Models-ML in Chapter10
    </div>
    """))


# Missing Value Handling

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
    ffill_columns=[  # e.g., weather or calendar indicators if needed
        "y_40320_seasonal_rolling_3_mean", "y_40320_seasonal_rolling_3_std",
    ],
    zero_fill_columns=[  # For categorical one-hot or flags, if any
        "y_40320_seasonal_rolling_3_mean", "y_40320_seasonal_rolling_3_std",
    ],
)

train_df = missing_value_config.impute_missing_values(train_df)
test_df = missing_value_config.impute_missing_values(test_df)

#nc = train_df.isnull().sum()
#print(nc[nc>0])

metric_record = [baseline_agg_metrics]

data_config = DataConfig(
    target=[target],
    continuous_cols=[
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
    categorical_cols=[
        "ds_Month", "ds_Quarter", "ds_Day", "ds_Dayofweek", "ds_Dayofyear",
        "ds_Hour", "ds_Minute", "ds_Second"
    ],
    normalize_continuous_features=True
)
trainer_config = TrainerConfig(
    auto_lr_find=True, 
    batch_size=1024,
    max_epochs=1000,
    accelerator="auto",
)
optimizer_config = OptimizerConfig()

train_df[data_config.categorical_cols] = train_df[data_config.categorical_cols].astype(str)
test_df[data_config.categorical_cols] = test_df[data_config.categorical_cols].astype(str)

# FT Transformer Model
linear_head_config = LinearHeadConfig(
    layers="32",
    activation="ReLU"
)
model_config = FTTransformerConfig(
    task="regression",
    num_attn_blocks=3,
    num_heads=4,
    transformer_head_dim=64,
    attn_dropout=0.2,
    ff_dropout=0.1,
    learning_rate = 1e-3,
    head_config=linear_head_config.__dict__,
    metrics=["mean_squared_error"]
)
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)

tabular_model.fit(train=train_df)
# Saving 
tabular_model.save_model(output_img/"ft_transformer_global_1")
# Deleting automatically saved checkpoints
shutil.rmtree("saved_models")




