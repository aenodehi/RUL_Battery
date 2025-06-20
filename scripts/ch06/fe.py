import math
import os
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
from src.utils.general import LogTime
from tqdm.autonotebook import tqdm
from IPython.display import display, HTML

from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    RollingMean,
    RollingStd,
    RollingMin,
    RollingMax,
    SeasonalRollingMean,
    SeasonalRollingMin,
    SeasonalRollingMax,
    SeasonalRollingStd,
    ExponentiallyWeightedMean,
)
from sklearn.ensemble import RandomForestRegressor

from src.feature_engineering.temporal_features import (
    add_fourier_features,
    bulk_add_fourier_features,
)
from src.feature_engineering.autoregressive_features import add_lags

np.random.seed(42)
tqdm.pandas()

# === Paths ===
preprocessed = Path("./data")
output_dir = Path("imgs")
output_dir.mkdir(exist_ok=True)
 
# === Load Data ===
try:
    train_df = pd.read_parquet(preprocessed / "B0005_train.parquet")[["Voltage_measured"]].reset_index()
    val_df = pd.read_parquet(preprocessed / "B0005_val.parquet")[["Voltage_measured"]].reset_index()
    test_df = pd.read_parquet(preprocessed / "B0005_test.parquet")[["Voltage_measured"]].reset_index()
except FileNotFoundError as e:
    display(HTML("""
                 <div class="alert alert-block alert-warning">
                 <b>Warning!</b> File not found.
                 </div>
                 """))
    print(f"Error: {e}")
    exit(1)

# === Preprocess ===
for df in [train_df, val_df, test_df]:
    df['datetime_'] = pd.to_datetime(df['datetime_'])
    df['unique_id'] = 'B0005'

train_df["type"] = "train"
val_df["type"] = "val"
test_df["type"] = "test"

full_df = pd.concat([train_df, val_df, test_df]).sort_values(["unique_id", "datetime_"])
full_df = full_df.rename(columns={"datetime_": "ds", "Voltage_measured": "y"})

full_df_type_map = full_df
full_df = full_df.drop(columns=['type'])

# === Lag Features ===
lags = (
        (np.arange(5) + 1).tolist() +
        (np.arange(5) + 240).tolist() +
        (np.arange(5) + 5760).tolist() 
        #(np.arange(5) + 40320).tolist()
        )

with LogTime():
    full_df, added_features = add_lags(
            full_df, lags=lads, column="y", ts_id="unique_id", use_32_bit=True
            )
print(f"Features Created: {','.join(added_features)}")

# === Rolling ===
with LogTime():
    full_df, added_features = add_rolling_features(
        full_df,
        rolls=[3, 6, 12, 48],
        column="y",
        agg_funcs=["mean", "std"],
        ts_id="unique_id",
        use_32_bit=True,
    )
print(f"Features Created: {','.join(added_features)}")

# === Seasonal Rolling ===

with LogTime():
    full_df, added_features = add_seasonal_rolling_features(
        full_df,
        rolls=[3],
        seasonal_periods=[5760, 40320],
        column="y",
        agg_funcs=["mean", "std"],
        ts_id="unique_id",
        use_32_bit=True,
    )
print(f"Features Created: {','.join(added_features)}")

# === EWMA ===
with LogTime():
    # full_df, added_features = add_ewma(full_df, alphas=[0.2, 0.5, 0.9], column="energy_consumption", ts_id="LCLid", use_32_bit=True)
    full_df, added_features = add_ewma(
        full_df,
        spans=[40320, 5760, 240],
        column="y",
        ts_id="unique_id",
        use_32_bit=True,
    )
print(f"Features Created: {','.join(added_features)}")

# === Temporal Features ===

with LogTime():
    full_df, added_features = add_temporal_features(
        full_df,
        field_name="ds",
        frequency="15s",
        add_elapsed=True,
        drop=False,
        use_32_bit=True,
    )
print(f"Features Created: {','.join(added_features)}")

# === Fourier Terms ===
with LogTime():
    full_df, added_features = bulk_add_fourier_features(
            full_df,
            ["month", "hour", "minute"],
            max_values=[12, 24, 60],
            n_fourier_terms=5,
            use_32_bit=True,
            )
print(f"Features Created: {','.join(added_features)}")


# === Plotting Fourier Terms ===
plot_df = (
    full_df[["month", "month_sin_1"]]
    .drop_duplicates()
    .sort_values("month")
)

plot_df.columns = ["calendar", "fourier"]

plot_df = pd.concat([plot_df, plot_df, plot_df]).reset_index(drop=True)
# plot_df.reset_index(drop=True, inplace=True)

plot_df.reset_index(inplace=True)
plot_df["index"] += 1
plot_df = pd.melt(
    plot_df, id_vars="index", var_name="month", value_name="Representation"
)

fig = px.line(plot_df, x="index", y="Representation", facet_row="month")
fig.update_layout(
    autosize=False,
    width=900,
    height=800,
    title={
        "text": "Step Function vs Continuous Function",
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
        "font": {"size": 20},
    },
    legend_title=None,
    # yaxis=dict(
    #     # title_text=ylabel,
    #     # titlefont=dict(size=12),
    # ),
    xaxis=dict(
        title_text="Time",
        # titlefont=dict(size=12),
    ),
)
fig.update_yaxes(matches=None)
fig.update_xaxes(
    ticktext=np.arange(1, 13).tolist() * 3,
    tickvals=np.arange(len(plot_df)) + 1,
)
fig.write_image(f"imgs/ch06/fourier.png")
fig.show()


# === Saving the feature engineered file ===
# print(transformed_df.info(memory_usage="deep", verbose=False))

full_df = pd.merge(full_df, full_df_type_map, on=["ds", "unique_id"], how="left")

full_df[full_df["type"] == "train"].drop(columns="type").to_parquet(
        preprocessed / "selected_blocks_train_missing_imputed_feature_engg.parquet"
)
full_df[full_df["type"] == "val"].drop(columns="type").to_parquet(
        preprocessed / "selected_blocks_val_missing_imputed_feature_engg.parquet"
)
full_df[full_df["type"] == "test"].drop(columns="type").to_parquet(
        preprocessed / "selected_blocks_test_missing_imputed_feature_engg.parquet"
)
# print(full_df.head())
# print(transformed_df.columns)
