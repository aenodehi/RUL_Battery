import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsforecast import StatsForecast
from src.models.statsforecast_models import Naive
from src.utils.ts_utils import forecast_bias
import time

# Define array-based metrics
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def mase(y_true, y_pred, y_train, m=48):
    # Mean Absolute Scaled Error with seasonal period m
    n = len(y_train)
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return np.mean(np.abs(y_true - y_pred)) / scale

# Main evaluation function
def evaluate_performance(
    ts_train: pd.DataFrame,
    ts_test: pd.DataFrame,
    models: list,
    freq: str,
    level: list,
    id_col: str,
    time_col: str,
    target_col: str,
    h: int,
    metric_df: pd.DataFrame = None,
    return_y_pred: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate forecasting performance using StatsForecast and array-based metrics.

    Returns:
        results: DataFrame with merged forecasts
        metric_df: DataFrame of computed metrics
    """
    if metric_df is None:
        metric_df = pd.DataFrame()

    # Forecast and timings
    results = ts_test.copy()
    timing = {}

    # Define metrics list as (name, function)
    metrics = [
        ("MAE", mean_absolute_error),
        ("MSE", mean_squared_error),
        ("RMSE", rmse),
        ("SMAPE", smape),
        ("MASE", mase),
        ("Bias", forecast_bias)
    ]

    for model in models:
        model_name = model.__class__.__name__
        evaluation = {}

        # Fit and forecast
        start_time = time.time()
        sf = StatsForecast(
            models=[model],
            freq=freq,
            n_jobs=-1,
            fallback_model=Naive()
        )
        y_pred = sf.forecast(
            h=h,
            df=ts_train,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            level=level,
        )
        timing[model_name] = time.time() - start_time

        # Merge forecasts
        results = results.merge(y_pred, how='left', on=[id_col, time_col])

        # Evaluate per series
        for uid in ts_train[id_col].unique():
            temp_res = results[results[id_col] == uid]
            y_train = ts_train[ts_train[id_col] == uid][target_col].values
            y_true = temp_res[target_col].values
            y_hat = temp_res[model_name].values

            # Drop any NaNs before metric calculations
            mask = (~pd.isna(y_true)) & (~pd.isna(y_hat))
            y_true = y_true[mask]
            y_hat  = y_hat[mask]

            for name, fn in metrics:
                if name == "MASE":
                    val = fn(y_true, y_hat, y_train, m=48)
                else:
                    val = fn(y_true, y_hat)
                evaluation[name] = val

            evaluation[id_col] = uid
            evaluation['Time Elapsed'] = timing[model_name]
            evaluation['Model'] = model_name

            metric_df = pd.concat([metric_df, pd.DataFrame(evaluation, index=[0])], ignore_index=True)

    if return_y_pred:
        return results, metric_df, y_pred
    return results, metric_df

