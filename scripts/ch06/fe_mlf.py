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
lag_transforms = defaultdict(list)

lags = (
        (np.arange(5) + 1).tolist() +
        (np.arange(5) + 240).tolist() +
        (np.arange(5) + 5760).tolist() 
        #(np.arange(5) + 40320).tolist()
        )

# === Rolling ===
lag_transforms[1] += [RollingMean(window_size=n) for n in [3, 6, 12, 48]] + [
        RollingStd(window_size=n) for n in [3, 6, 12, 48]
]

# === Seasonal Rolling ===
lag_transforms[5760] += [
        SeasonalRollingMean(season_length=5760, window_size=3),
        SeasonalRollingStd(season_length=5760, window_size=3)
        ]
lag_transforms[40320] += [
        SeasonalRollingMean(season_length=40320, window_size=3),
        SeasonalRollingStd(season_length=40320, window_size=3)
        ]

# === EWMA ===
lag_transforms[1] += [ExponentiallyWeightedMean(alpha=alpha) for alpha in [0.2, 0.5, 0.9]]

# === Temporal Features ===
temporal_features = [
    "month",
    "quarter",
    "is_quarter_end",
    "is_quarter_start",
    "is_year_end",
    "is_year_start",
    "is_month_start",
    "is_month_end",
    "week",
    "day",
    "dayofweek",
    "dayofyear",
    "hour",
    "minute",
]

# === Calculating the Features ===
fcst = MLForecast(
        models=[],
        freq='15s',
        lags=lags,
        lag_transforms=lag_transforms,
        date_features=temporal_features,
        )

with LogTime():
    transformed_df = fcst.preprocess(
            full_df,
            time_col="ds",
            id_col="unique_id",
            target_col="y",
            static_features=[],
            dropna=False,
            )

# === Fourier Terms ===
with LogTime():
    transformed_df, added_features = bulk_add_fourier_features(
            transformed_df,
            ["month", "hour", "minute"],
            max_values=[12, 24, 60],
            n_fourier_terms=5,
            use_32_bit=True,
            )
print(f"Features Created: {','.join(added_features)}")


# === Plotting Fourier Terms ===
plot_df = (
    transformed_df[["month", "month_sin_1"]]
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
# fig.write_image(f"imgs/ch06/fourier.png")
fig.show()


# === Saving the feature engineered file ===
full_df = pd.merge(transformed_df, full_df_type_map[["ds", "unique_id", "type"]], on=["ds", "unique_id"], how="left")

full_df[full_df["type"] == "train"].drop(columns="type").to_parquet(
        preprocessed / "selected_blocks_train_missing_imputed_feature_engg_mlforecast.parquet"
)
full_df[full_df["type"] == "val"].drop(columns="type").to_parquet(
        preprocessed / "selected_blocks_val_missing_imputed_feature_engg_mlforecast.parquet"
)
full_df[full_df["type"] == "test"].drop(columns="type").to_parquet(
        preprocessed / "selected_blocks_test_missing_imputed_feature_engg_mlforecast.parquet"
)
