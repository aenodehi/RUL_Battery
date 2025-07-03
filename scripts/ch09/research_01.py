# Forecast Combinations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"

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


np.random.seed(42)
tqdm.pandas()


# === Paths ===
preprocessed = Path("./data")
output = Path("data/B0005")
output.mkdir(exist_ok=True)
output_img = Path("imgs/ch08")
output_img.mkdir(exist_ok=True)

# === Load Data ===

# === Reading the Test Predictions & Metrics ===

try:
    train_df = pd.read_parquet(
        preprocessed / "selected_blocks_train_missing_imputed_feature_engg.parquet"
    )
    train_df = train_df.loc[:, ["ds", "unique_id", "y"]].set_index(
        ["ds", "unique_id"]
    )
    val_df = pd.read_parquet(
        preprocessed / "selected_blocks_val_missing_imputed_feature_engg.parquet"
    )
    val_df = val_df.loc[:, ["ds", "unique_id", "y"]].set_index(
        ["ds", "unique_id"]
    )

    train_target = train_df.reset_index().set_index("ds")
    # Combine train and val into new train
    train_val_target = pd.concat([train_df, val_df]).reset_index().set_index("ds")

    del val_df, train_df

except FileNotFoundError:
    display(HTML("""
    <div class="alert alert-block alert-warning">
    <b>Warning!</b> File not found. Please make sure you have run in Chapter06 
    </div>
    """))

try:
    pred_test_df = pd.read_pickle(output / "ml_single_step_prediction_test_df.pkl")
    metrics_test_df = pd.read_pickle(output / "ml_single_step_metrics_test_df.pkl")
    pred_auto_stat_test_df = pd.read_pickle(
        output / "ml_single_step_prediction_auto_stationary_test_df.pkl"
    )
    metrics_auto_stat_test_df = pd.read_pickle(
        output / "ml_single_step_metrics_auto_stationary_test_df.pkl"
    )
    agg_metrics_auto_stat_test_df = pd.read_pickle(
        output / "ml_single_step_aggregate_metrics_auto_stationary_test.pkl"
    )
    pred_baselines_test_df = pd.read_pickle(output / "single_step_backtesting_baseline_prediction_test_df.pkl")
    metrics_baselines_test_df = pd.read_pickle(output / "single_step_backtesting_baseline_metrics_test_df.pkl")
    agg_metrics_baselines_test_df = pd.read_pickle(
        output / "single_step_backtesting_baseline_aggregate_metrics_test.pkl"
    )

    pred_val_df = pd.read_pickle(output / "ml_single_step_prediction_val_df.pkl")
    metrics_val_df = pd.read_pickle(output / "ml_single_step_metrics_val_df.pkl")
    pred_auto_stat_val_df = pd.read_pickle(
        output / "ml_single_step_prediction_auto_stationary_val_df.pkl"
    )
    metrics_auto_stat_val_df = pd.read_pickle(
        output / "ml_single_step_metrics_auto_stationary_val_df.pkl"
    )
    agg_metrics_auto_stat_val_df = pd.read_pickle(
        output / "ml_single_step_aggregate_metrics_auto_stationary_val.pkl"
    )
    pred_baselines_val_df = pd.read_pickle(output / "single_step_backtesting_baseline_prediction_val_df.pkl")
    metrics_baselines_val_df = pd.read_pickle(output / "single_step_backtesting_baseline_metrics_val_df.pkl")
    agg_metrics_baselines_val_df = pd.read_pickle(
        output / "single_step_backtesting_baseline_aggregate_metrics_val.pkl"
    )
except FileNotFoundError:
    display(HTML("""
    <div class="alert alert-block alert-warning">
    <b>Warning!</b> File not found. Please make sure you have run Chapter08 and Baseline Forecasts using NIXTLA in Chapter04
    </div>
    """))


#print(pred_val_df.head())
#print(pred_auto_stat_val_df.head())

#print(pred_baselines_val_df.columns)

pred_baselines_val_df = pred_baselines_val_df.set_index('ds').melt(id_vars = ['unique_id','voltage_measured'],value_vars=['naive_predictions', 'snaive_predictions'], var_name='Algorithm', value_name='predictions', ignore_index=False)

pred_baselines_test_df = pred_baselines_test_df.set_index('ds').melt(id_vars = ['unique_id','voltage_measured'],value_vars=['naive_predictions', 'snaive_predictions'], var_name='Algorithm', value_name='predictions', ignore_index=False)


#print(pred_baselines_test_df)

pred_val_df = pd.concat([pred_val_df, pred_auto_stat_val_df, pred_baselines_val_df])
pred_val_df.index.name = "ds"

pred_wide_val = pd.pivot(
    pred_val_df.reset_index(),
    index=["unique_id", "ds"],
    columns="Algorithm",
    values="predictions",
)
pred_wide_val = pred_wide_val.join(
    pred_val_df.loc[
        pred_val_df.Algorithm == "Lasso Regression", ["unique_id", "voltage_measured"]
    ]
    .reset_index()
    .set_index(["unique_id", "ds"])
)
#print(pred_wide_val.head())


pred_test_df = pd.concat([pred_test_df, pred_auto_stat_test_df, pred_baselines_test_df])
pred_test_df.index.name = "ds"

pred_wide_test = pd.pivot(
    pred_test_df.reset_index(),
    index=["unique_id", "ds"],
    columns="Algorithm",
    values="predictions",
)
pred_wide_test = pred_wide_test.join(
    pred_test_df.loc[
        pred_test_df.Algorithm == "Lasso Regression", ["unique_id", "voltage_measured"]
    ]
    .reset_index()
    .set_index(["unique_id", "ds"])
)
#print(pred_wide_test.columns)

# -----------------------------------------------------------
metrics_combined_df = pd.concat([metrics_val_df, metrics_auto_stat_val_df])
metrics_combined_df = pd.pivot(
    metrics_combined_df, index="unique_id", columns="Algorithm", values="MAE"
)
# print(metrics_combined_df.head())




# ---- Combining Forecasts ----
def evaluate_ensemble(pred_wide, target_history, model, target, unique_id):
    metric_l = []
    for _id in tqdm(pred_wide.reset_index()[unique_id].unique()):
        # unique_mask = pred_wide[unique_id]==_id
        wide_df = pred_wide.xs(_id)
        test_target = wide_df.loc[:, target]
        y_pred = wide_df.loc[:, model]
        history = target_history.loc[target_history[unique_id] == _id, target]
        metric_l.append(
            calculate_metrics(test_target, y_pred, name=model, y_train=history)
        )
    eval_metrics_df = pd.DataFrame(metric_l)
    return {
        "Algorithm": model,
        "MAE": ts_utils.mae(
            pred_wide.loc[:, "voltage_measured"], pred_wide.loc[:, model]
        ),
        "MSE": ts_utils.mse(
            pred_wide.loc[:, "voltage_measured"], pred_wide.loc[:, model]
        ),
        "meanMASE": eval_metrics_df.loc[:, "MASE"].mean(),
        "Forecast Bias": ts_utils.forecast_bias_aggregate(
            pred_wide.loc[:, "voltage_measured"], pred_wide.loc[:, model]
        ),
    }


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


ensemble_forecasts = [
    "naive_predictions",
    "Lasso Regression",
    "Lasso Regression_auto_stat",
    "LightGBM",
    "LightGBM_auto_stat",
    "snaive_predictions",
    "XGB Random Forest",
    "XGB Random Forest_auto_stat",
]


agg_metrics_l = agg_metrics_auto_stat_test_df.iloc[[4]].to_dict(orient="records")

best_alg = metrics_combined_df.idxmin(axis=1)
#print(best_alg.head())

pred_wide_test["best_fit"] = np.nan
pred_wide_test["best_fit_alg"] = ""

common_ids = pred_wide_test.index.get_level_values(0).unique().intersection(best_alg.index)

for lcl_id in tqdm(common_ids):
    alg = best_alg[lcl_id]
    pred_wide_test.loc[lcl_id, "best_fit"] = pred_wide_test.loc[lcl_id, alg].values
    pred_wide_test.loc[lcl_id, "best_fit_alg"] = alg


pred_wide_test["y"] = pred_wide_test["voltage_measured"]

#print(pred_wide_test.columns)
#print(pred_wide_test.head(2))

#print(train_val_target.columns)
agg_metric_ = evaluate_ensemble(
    pred_wide_test, train_val_target, "best_fit", "y", "unique_id"
)
#print(agg_metric_)
agg_metrics_l.append(agg_metric_)



# --- Average and Median Ensemble ---
 
pred_wide_test["average_ensemble"] = pred_wide_test[ensemble_forecasts].mean(axis=1)
pred_wide_test["median_ensemble"] = pred_wide_test[ensemble_forecasts].median(axis=1)
 
agg_metric_ = evaluate_ensemble(
     pred_wide_test, train_val_target, "median_ensemble", "y", "unique_id"
)
#print(agg_metric_)
agg_metrics_l.append(agg_metric_)
agg_metric_ = evaluate_ensemble(
    pred_wide_test, train_val_target, "average_ensemble", "y", "unique_id"
)
#print(agg_metric_)
agg_metrics_l.append(agg_metric_)


#output_path = output_img / "agg_metrics.html"
#display_metrics(agg_metrics_l, save_path=output_path)


# --- Greedy Optimization ---

objective = partial(
    calculate_performance, pred_wide=pred_wide_val, target="voltage_measured"
)

solution, best_score = greedy_optimization(objective, ensemble_forecasts)

pred_wide_test["greedy_ensemble"] = pred_wide_test[solution].mean(axis=1)

agg_metric_ = evaluate_ensemble(
    pred_wide_test, train_val_target, "greedy_ensemble", "y", "unique_id"
)
#print(agg_metric_)
agg_metrics_l.append(agg_metric_)



# --- Stochastic Hill-climbing with Validation Forecasts ---
objective = partial(
    calculate_performance, pred_wide=pred_wide_val, target="voltage_measured"
)

solution, best_score = stochastic_hillclimbing(
    objective, ensemble_forecasts, n_iterations=10, init="best", random_state=42
)

pred_wide_test["stochastic_hillclimb__ensemble"] = pred_wide_test[solution].mean(axis=1)
 
agg_metric_ = evaluate_ensemble(
    pred_wide_test,
    train_val_target,
    "stochastic_hillclimb__ensemble",
    "y",
    "unique_id",
)
#print(agg_metric_)
agg_metrics_l.append(agg_metric_)


# output_path = output_img / "agg_metrics.html"
# display_metrics(agg_metrics_l, save_path=output_path)

# --- Simulated Annealing with Validation Forecasts ---

objective = partial(
    calculate_performance, pred_wide=pred_wide_val, target="voltage_measured"
)

solution, best_score = simulated_annealing(
    objective,
    ensemble_forecasts,
    p_range=(0.5, 0.0001),
    n_iterations=50,
    init="best",
    temperature_decay="geometric",
    random_state=42,
)


pred_wide_test["simulated_annealing_ensemble"] = pred_wide_test[solution].mean(axis=1)

agg_metric_ = evaluate_ensemble(
    pred_wide_test,
    train_val_target,
    "simulated_annealing_ensemble",
    "y",
    "unique_id",
)
#print(agg_metric_)
agg_metrics_l.append(agg_metric_)

# --- Optimal Weighted Ensemble ---
optimal_weights = find_optimal_combination(
    ensemble_forecasts, pred_wide_val, target="voltage_measured"
)

pred_wide_test["optimal_combination_ensemble"] = np.sum(
    pred_wide_test[ensemble_forecasts].values * np.array(optimal_weights), axis=1
)

agg_metric_ = evaluate_ensemble(
    pred_wide_test,
    train_val_target,
    "optimal_combination_ensemble",
    "y",
    "unique_id",
)
#print(agg_metric_)
agg_metrics_l.append(agg_metric_)

# --- Stacking/Blending Model ---
# Linear Regression
stacking_model = LinearRegression(positive=True, fit_intercept=False)
stacking_model.fit(
    pred_wide_val[ensemble_forecasts], pred_wide_val["voltage_measured"]
)

pred_wide_test["linear_reg_blending"] = stacking_model.predict(
    pred_wide_test[ensemble_forecasts]
)

agg_metric_ = evaluate_ensemble(
    pred_wide_test,
    train_val_target,
    "linear_reg_blending",
    "y",
    "unique_id",
)
#print(agg_metric_)
agg_metrics_l.append(agg_metric_)

# Ridge Regression

stacking_model = RidgeCV()
stacking_model.fit(
    pred_wide_val[ensemble_forecasts], pred_wide_val["voltage_measured"]
)
pred_wide_test["ridge_reg_blending"] = stacking_model.predict(
    pred_wide_test[ensemble_forecasts]
)
agg_metric_ = evaluate_ensemble(
    pred_wide_test,
    train_val_target,
    "ridge_reg_blending",
    "y",
    "unique_id",
)
#print(agg_metric_)
agg_metrics_l.append(agg_metric_)

# Lasso Regression
stacking_model = LassoCV()
stacking_model.fit(
    pred_wide_val[ensemble_forecasts], pred_wide_val["voltage_measured"]
)
pred_wide_test["lasso_reg_blending"] = stacking_model.predict(
    pred_wide_test[ensemble_forecasts]
)
agg_metric_ = evaluate_ensemble(
    pred_wide_test,
    train_val_target,
    "lasso_reg_blending",
    "y",
    "unique_id",
)
#print(agg_metric_)
agg_metrics_l.append(agg_metric_)

# Huber Regression
stacking_model = HuberRegressor()
stacking_model.fit(
    pred_wide_val[ensemble_forecasts], pred_wide_val["voltage_measured"]
)
pred_wide_test["huber_reg_blending"] = stacking_model.predict(
    pred_wide_test[ensemble_forecasts]
)

agg_metric_ = evaluate_ensemble(
    pred_wide_test,
    train_val_target,
    "huber_reg_blending",
    "y",
    "unique_id",
)
#print(agg_metric_)
agg_metrics_l.append(agg_metric_)

# --- Using Variety as Regularization ---

def calculate_diverse_objective(ens, pred_wide, target, diversity_matrix, alpha):
    perf = calculate_performance(ens, pred_wide, target)
    div = calculate_diversity(ens, diversity_matrix)
    return perf + alpha * div

objective = partial(
    calculate_diverse_objective,
    pred_wide=pred_wide_val,
    target="voltage_measured",
    diversity_matrix=pred_wide_val[ensemble_forecasts].corr(),
    alpha=0.05,
)

solution, best_score = stochastic_hillclimbing(
    objective, ensemble_forecasts, n_iterations=10, random_state=42
)


pred_wide_test["hillclimbing_w_reg_ensemble"] = pred_wide_test[solution].mean(axis=1)

agg_metric_ = evaluate_ensemble(
    pred_wide_test,
    train_val_target,
    "hillclimbing_w_reg_ensemble",
    "y",
    "unique_id",
)
#print(agg_metric_)
agg_metrics_l.append(agg_metric_)


output_path = output_img / "agg_metrics.html"
display_metrics(agg_metrics_l, save_path=output_path)


