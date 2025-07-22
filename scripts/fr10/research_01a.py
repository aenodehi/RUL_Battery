# Global Forecasting Models ML for test dataset

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import copy
pio.templates.default = "plotly_white"
import pickle
import warnings
from pathlib import Path
import joblib
from functools import partial

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, HuberRegressor
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
from src.utils.plotting_utils import plot_forecast, format_plot, plot_correlation_plot
from src.forecasting.ensembling import calculate_performance, greedy_optimization, stochastic_hillclimbing, simulated_annealing, find_optimal_combination, calculate_diversity
from category_encoders import CountEncoder
from category_encoders import TargetEncoder
from lightgbm import LGBMRegressor
import random


np.random.seed(42)
tqdm.pandas()


# === Paths ===
preprocessed = Path("./data")
output = Path("data/B0005")
output.mkdir(exist_ok=True)
output_img = Path("imgs/ch10")
output_img.mkdir(exist_ok=True)

# === Load Data ===

# === Reading the Test Predictions & Metrics ===

try:
    train_df = pd.read_parquet(
        preprocessed / "selected_blocks_train_missing_imputed_feature_engg.parquet"
    )
    val_df = pd.read_parquet(
        preprocessed / "selected_blocks_val_missing_imputed_feature_engg.parquet"
    )
    train_df = pd.concat([train_df, val_df])
    del val_df
    test_df = pd.read_parquet(preprocessed/"selected_blocks_test_missing_imputed_feature_engg.parquet")
except FileNotFoundError:
    display(HTML("""
    <div class="alert alert-block alert-warning">
    <b>Warning!</b> File not found. Please make sure you have run in Chapter06 
    </div>
    """))

# Loading the single step backtesting baselines for validation

try:
    baseline_aggregate_metrics_df = pd.read_pickle(
        output / "ml_single_step_aggregate_metrics_auto_stationary_test.pkl"
    )
except FileNotFoundError:
    display(HTML("""
    <div class="alert alert-block alert-warning">
    <b>Warning!</b> File not found. Please make sure you have run Forecasting with Target Transformation in Chapter08
    </div>
    """))

#print(train_df.columns)
#print(test_df.columns)
# print(baseline_aggregate_metrics_df.columns)

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
    index_cols=["unique_id", "ds"],
    exogenous_features=[
        # You could list continuous + categorical exogenous features here
        # (i.e., anything not lagged or derived from `y`)
        "ds_Month", "ds_Quarter", "ds_Dayofweek", "ds_Hour", "ds_Minute",
        "ds_Elapsed"
    ]
)

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

# Training Global ML Model
def train_model(
    model_config,
    feature_config,
    missing_config,
    train_features,
    train_target,
    test_features,
    fit_kwargs={}
):
    ml_model = MLForecast(
        model_config=model_config,
        feature_config=feature_config,
        missing_config=missing_config,
    )
    ml_model.fit(train_features, train_target, fit_kwargs=fit_kwargs)
    y_pred = ml_model.predict(test_features)
    feat_df = ml_model.feature_importance()
    return y_pred, feat_df

def evaluate_forecast(y_pred, test_target, train_target, model_config):
    metric_l = []
    for _id in tqdm(test_target.index.get_level_values(0).unique(), desc="Calculating metrics..."):
        target = test_target.xs(_id)
        _y_pred = y_pred.xs(_id)
        history = train_target.xs(_id)
        metric_l.append(
            calculate_metrics(target, _y_pred, name=model_config.name, y_train=history)
        )
    eval_metrics_df = pd.DataFrame(metric_l)
    agg_metrics = {
            "Algorithm": model_config.name,
            "MAE": ts_utils.mae(
                test_target['y'], y_pred
            ),
            "MSE": ts_utils.mse(
                test_target['y'], y_pred
            ),
            "meanMASE": eval_metrics_df.loc[:, "MASE"].mean(),
            "Forecast Bias": ts_utils.forecast_bias_aggregate(
                test_target['y'], y_pred
            )
    }
    return agg_metrics, eval_metrics_df

metric_record = []
individual_metrics = dict()

metric_record = (
    baseline_aggregate_metrics_df.iloc[[4]]
    .to_dict(orient="records")
)

def highlight_abs_min(s, props=""):
    return np.where(s == np.nanmin(np.abs(s.values)), props, "")

def display_metrics(agg_metrics_l, save_path=None, print_console=True):
    _agg_metrics_df = pd.DataFrame(agg_metrics_l)

    styled = (
        _agg_metrics_df.style.format({
            "MAE": "{:.4f}",
            "MSE": "{:.4f}",
            "meanMASE": "{:.4f}",
            "Forecast Bias": "{:.2f}%"
        })
        .highlight_min(color="lightgreen", subset=["MAE", "MSE", "meanMASE"])
        .apply(
            highlight_abs_min,
            props="color:black;background-color:lightgreen",
            axis=0,
            subset=["Forecast Bias"],
        )
    )

    if save_path:
        styled.to_html(save_path)
        print(f"Saved styled metrics to {save_path}")

    if print_console:
        print(_agg_metrics_df.to_string(index=False))

# Baseline
_feat_config = copy.deepcopy(feat_config)

train_features, train_target, train_original_target = _feat_config.get_X_y(
    train_df, categorical=True, exogenous=False
)

test_features, test_target, test_original_target = _feat_config.get_X_y(
    test_df, categorical=True, exogenous=False
)

pred_df = test_target.copy()

#print(test_target.index.names)
#print(test_target.index)

model_config = ModelConfig(
    model=LGBMRegressor(random_state=42),
    name="GFM Baseline",
    # LGBM is not sensitive to normalized data
    normalize=False,
    # LGBM can handle missing values
    fill_missing=False,
)

with LogTime() as timer:
    y_pred, feat_df = train_model(
        model_config,
        _feat_config,
        missing_value_config,
        train_features,
        train_target,
        test_features,
    )
agg_metrics, eval_metrics_df = evaluate_forecast(
    y_pred, test_target, train_target, model_config
)
agg_metrics["Time Elapsed"] = timer.elapsed
metric_record.append(agg_metrics)
individual_metrics[model_config.name]=eval_metrics_df
pred_df = pred_df.join(y_pred)


# --- With Metadata ---
feat_conf_dict = copy.deepcopy(feat_config.__dict__)
feat_conf_dict.pop("feature_list")
#feat_conf_dict['categorical_features']+=["stdorToU", "Acorn", "Acorn_grouped", "unique_id"]
feat_conf_dict['categorical_features']+=["unique_id"]
_feat_config = FeatureConfig(**feat_conf_dict)

train_features, train_target, train_original_target = _feat_config.get_X_y(
    train_df, categorical=True, exogenous=False
)
# Loading the Validation as test
test_features, test_target, test_original_target = _feat_config.get_X_y(
    test_df, categorical=True, exogenous=False
)

cat_features = set(train_features.columns).intersection(_feat_config.categorical_features)

# CountEncoder
cat_encoder = CountEncoder(cols=cat_features)

model_config = ModelConfig(
    model=LGBMRegressor(random_state=42),
    name="GFM+Meta (CountEncoder)",
    # LGBM is not sensitive to normalized data
    normalize=False,
    # LGBM can handle missing values
    fill_missing=False,
    encode_categorical=True,
    categorical_encoder=cat_encoder
)

with LogTime() as timer:
    y_pred, feat_df = train_model(
        model_config,
        _feat_config,
        missing_value_config,
        train_features,
        train_target,
        test_features,
    )
agg_metrics, eval_metrics_df = evaluate_forecast(y_pred, test_target, train_target, model_config)
agg_metrics["Time Elapsed"] = timer.elapsed
metric_record.append(agg_metrics)
individual_metrics[model_config.name]=eval_metrics_df
pred_df = pred_df.join(y_pred)

# Target Encoding
cat_encoder = TargetEncoder(cols=cat_features)

model_config = ModelConfig(
    model=LGBMRegressor(random_state=42),
    name="GFM+Meta  (TargetEncoder)",
    normalize=False,
    fill_missing=False,
    encode_categorical=True,
    categorical_encoder=cat_encoder
)
with LogTime() as timer:
    y_pred, feat_df = train_model(
        model_config,
        _feat_config,
        missing_value_config,
        train_features,
        train_target,
        test_features,
    )
agg_metrics, eval_metrics_df = evaluate_forecast(y_pred, test_target, train_target, model_config)
agg_metrics["Time Elapsed"] = timer.elapsed
metric_record.append(agg_metrics)
individual_metrics[model_config.name]=eval_metrics_df
pred_df = pred_df.join(y_pred)

# Native LightGBM Encoding
model_config = ModelConfig(
    model=LGBMRegressor(random_state=42),
    name="GFM+Meta  (NativeLGBM)",
    normalize=False,
    fill_missing=False,
    # We are using inbuilt categorical feature handling
    encode_categorical=False,
)
for col in cat_features:
    if train_features[col].dtype == "object":
        train_features[col] = train_features[col].astype("category")
        test_features[col] = test_features[col].astype("category")

with LogTime() as timer:
    y_pred, feat_df = train_model(
        model_config,
        _feat_config,
        missing_value_config,
        train_features,
        train_target,
        test_features,
        fit_kwargs=dict(categorical_feature=cat_features),
    )
agg_metrics, eval_metrics_df = evaluate_forecast(y_pred, test_target, train_target, model_config)
agg_metrics["Time Elapsed"] = timer.elapsed
metric_record.append(agg_metrics)
individual_metrics[model_config.name]=eval_metrics_df
pred_df = pred_df.join(y_pred)

# Hyperparameter Tuning


# Grid Search


# Random Search


# Bayesian Optimization


# Hyperparameter Tuning Techniques (Comparison)


# Using the tuned parameters
best_params = {
    "num_leaves": 99,
    "objective": "regression_l1",
    "colsample_bytree": 0.9786759775515064,
    "lambda_l1": 8.160098582954642,
    "lambda_l2": 0.17840888757497253,
    "random_state": 42,
}

model_config = ModelConfig(
    model=LGBMRegressor(**best_params),
    name="Tuned GFM+Meta",
    normalize=False,
    fill_missing=False,
)
with LogTime() as timer:
    y_pred, feat_df = train_model(
        model_config,
        _feat_config,
        missing_value_config,
        train_features,
        train_target,
        test_features,
        fit_kwargs=dict(categorical_feature=cat_features)
    )
agg_metrics, eval_metrics_df = evaluate_forecast(y_pred, test_target, train_target, model_config)
agg_metrics["Time Elapsed"] = timer.elapsed
metric_record.append(agg_metrics)
individual_metrics[model_config.name]=eval_metrics_df
pred_df = pred_df.join(y_pred)

# --- Partitioning ---
#best_params = {
#    "num_leaves": 99,
#    "objective": "regression_l1",
#    "colsample_bytree": 0.9786759775515064,
#    "lambda_l1": 8.160098582954642,
#    "lambda_l2": 0.17840888757497253,
#    "random_state": 42,
#}
#
## Random partition
#model_config = ModelConfig(
#    model=LGBMRegressor(**best_params, verbose=-1),
#    name="Tuned GFM+Meta+Random Part",
#    # LGBM is not sensitive to normalized data
#    normalize=False,
#    # LGBM can handle missing values
#    fill_missing=False,
#)
#def partition (list_in, n):
#    random.shuffle(list_in)
#    return [list_in[i::n] for i in range(n)]
#
#
#
#partitions = partition(train_df.unique_id.cat.categories.tolist(), 3)
#
#y_pred_l = []
#feat_df_l = []
#time_elapsed_l = []
#for lclids in tqdm(partitions, desc="Training groups..."):
#    _train_df = train_df.loc[train_df.unique_id.isin(lclids)]
#    _test_df = test_df.loc[test_df.unique_id.isin(lclids)]
#    train_features, train_target, train_original_target = _feat_config.get_X_y(
#        _train_df, categorical=True, exogenous=False
#    )
#    test_features, test_target, test_original_target = _feat_config.get_X_y(
#        _test_df, categorical=True, exogenous=False
#    )
#    cat_features = set(train_features.columns).intersection(
#        _feat_config.categorical_features
#    )
#
#    _model_config = model_config.clone()
#    with LogTime() as timer:
#        y_pred, feat_df = train_model(
#            _model_config,
#            _feat_config,
#            missing_value_config,
#            train_features,
#            train_target,
#            test_features,
#            fit_kwargs=dict(categorical_feature=cat_features),
#        )
#    y_pred_l.append(y_pred)
#    feat_df_l.append(feat_df)
#    time_elapsed_l.append(timer.elapsed)
#
#y_pred = pd.concat(y_pred_l)
#
#test_features, test_target, test_original_target = _feat_config.get_X_y(
#    test_df, categorical=True, exogenous=False
#)
#train_features, train_target, train_original_target = _feat_config.get_X_y(
#    train_df, categorical=True, exogenous=False
#)
#
#agg_metrics, eval_metrics_df = evaluate_forecast(y_pred, test_target, train_target, model_config)
#agg_metrics["Time Elapsed"] = np.sum(time_elapsed_l)
#metric_record.append(agg_metrics)
#individual_metrics[model_config.name]=eval_metrics_df
#pred_df = pred_df.join(y_pred)

# Judgmental partitioning
# Algorithmic partitioning



#html = metric_record.to_html()
#with open(output_img/"metric_record.html", "w") as f:
#    f.write(html)

output_path = output_img/"metric_record.html"
display_metrics(metric_record, save_path=output_path)


# Saving the GFM Forecasts & Metrics

pred_df.to_pickle(output/"gfm_predictions_test_df.pkl")
joblib.dump(individual_metrics, output/"gfm_metrics_test_df.pkl")
pd.DataFrame([agg_metrics]).to_pickle(output/"gfm_aggregate_metrics_test.pkl")




