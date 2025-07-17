# this is the file of TFT NeuralForecast

import os
import shutil
import json
import sys
import ray

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

from pathlib import Path

from tqdm.autonotebook import tqdm
from IPython.display import display, HTML
# %load_ext autoreload
# %autoreload 2
np.random.seed(42)
tqdm.pandas()

from statsforecast import StatsForecast
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.auto import AutoTFT
from neuralforecast.losses.pytorch import MAE
from neuralforecast.losses.pytorch import MQLoss
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmse, mae, mse, mase
from functools import partial

from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch


import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

import hyperopt


TRAIN_SUBSAMPLE = True
RETUNE = False

def highlight_abs_min(s, props=''):
    return np.where(s == np.nanmin(np.abs(s.values)), props, '')

def display_metrics(agg_metrics_l, save_path=None, print_console=True):
    _agg_metrics_df = pd.DataFrame(agg_metrics_l)

    styled = (
        _agg_metrics_df.style.format({
            "MAE": "{:.4f}",
            "MSE": "{:.4f}",
            "MASE": "{:.4f}",
            "Forecast Bias": "{:.2f}%"
        })
        .highlight_min(color="lightgreen", subset=["MAE", "MSE", "MASE"])
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


# sys.path.append(str(Path(__file__).resolve().parents[1]))
# sys.path.append(str(Path(__file__).resolve().parents[2]))

# === Paths ===
preprocessed = Path("./data")
output = Path("data/B0005")
output.mkdir(exist_ok=True)
output_img = Path("imgs/ch16")
output_img.mkdir(exist_ok=True)

# === Load Data ===
try:
    train_df = pd.read_parquet(preprocessed/"selected_blocks_train_missing_imputed_feature_engg.parquet")
    test_df = pd.read_parquet(preprocessed /"selected_blocks_val_missing_imputed_feature_engg.parquet")
except FileNotFoundError:
    display(HTML("""
    <div class="alert alert-block alert-warning">
    <b>Warning!</b> File not found. Please make sure you have run in Chapter06 
    </div>
    """))

rename_cols = {
        "ds": "timestamp",
        "y": "voltage_measured",
        "unique_id": "BTid"
        }

train_df.rename(columns=rename_cols, inplace=True)
test_df.rename(columns=rename_cols, inplace=True)
# print(train_df.head(2))
# print(test_df.columns)

# plot_df = train_df[train_df['unique_id'] == 'B0005'].copy()
# 
# fig = StatsForecast.plot(
#         plot_df,
#         engine='matplotlib',
#         id_col='unique_id',
#         time_col='timestamp',
#         target_col='voltage_measured'
#         )
# 
# plot_path = output_img/"statsforecast_plot_B0005.png"
# fig.savefig(plot_path, dpi=300, bbox_inches="tight")
# plt.show()
# plt.close(fig)


if TRAIN_SUBSAMPLE:
    # print("sub sampling")
    # SAMPLE = 10
    SAMPLE = 1
    sampled_BTids = pd.Series(train_df.BTid.unique()).sample(SAMPLE, random_state=99).tolist()
    train_df = train_df.loc[train_df.BTid.isin(sampled_BTids)]
    test_df = test_df.loc[test_df.BTid.isin(sampled_BTids)]


# print("Total # of IDs Post Sampling: ", len(train_df.BTid.unique()))

# Train, Validation, Test Set
# print("Training Min Date: ", train_df.timestamp.min(), 
#      "\nTraining Max Date: ", train_df.timestamp.max(), 
#      "\nTesting Min Date: ", test_df.timestamp.min(),
#      "\nTesting Max Date: ", test_df.timestamp.max()
# )

# cutoff = train_df.timestamp.max() - pd.Timedelta(1, "D")
cutoff = train_df.timestamp.max() - pd.Timedelta(minutes=12)

validation_df = train_df[(train_df.timestamp>cutoff)].reset_index(drop=True)
training_df = train_df[(train_df.timestamp<=cutoff)].reset_index(drop=True)

# print(f"Train Max: {training_df.timestamp.max()} \nValidation Min: {validation_df.timestamp.min()} \nValidation Max: {validation_df.timestamp.max()}")
# print(f"Validation Horizon: {len(validation_df.timestamp.unique())}")

# Define Validation model parameters
# h = 1127 
# max_steps = 200
h = 48 
max_steps = 10

# Training TFT Model
model_untuned = [TFT(h=h, input_size= 48*2,
                     max_steps=max_steps)]

model_untuned = NeuralForecast(models=model_untuned, freq='15s')
model_untuned.fit(training_df[['BTid', 'timestamp', 'voltage_measured']], 
                  id_col= 'BTid', 
                  time_col = 'timestamp',
                  target_col='voltage_measured')

future_df = model_untuned.make_future_dataframe(df=training_df[['BTid', 'timestamp', 'voltage_measured']])

pred_df = model_untuned.predict(futr_df=future_df).reset_index()

pred_df = pred_df.merge(
        validation_df[['BTid', 'timestamp', 'voltage_measured']], 
        on=["BTid", "timestamp"], 
        how="left",)

# print(pred_df.head())

fig = StatsForecast.plot(
    validation_df[['BTid', 'timestamp', 'voltage_measured']],
    pred_df,
    engine='matplotlib',
    id_col='BTid',
    time_col='timestamp',
    target_col='voltage_measured',
    models=['TFT']
)

# fig.savefig(output_img/"tft_prediction_vs_actual.png", dpi=300, bbox_inches="tight")
# plt.close(fig)

# Evaluate TFT forecast
fcst_mase = partial(mase, seasonality=48)

# === Evaluate individual BTid series ===
TFT_metrics = evaluate(
    pred_df,
    metrics=[rmse, mae, mse, fcst_mase],
    train_df=training_df[['timestamp', 'BTid', 'voltage_measured']],
    id_col='BTid',
    time_col='timestamp',
    target_col='voltage_measured'
)

# === Evaluate aggregated metrics across all BTids ===
TFT_metrics_agg = evaluate(
    pred_df,
    metrics=[rmse, mae, mse, fcst_mase],
    train_df=training_df[['timestamp', 'BTid', 'voltage_measured']],
    id_col='BTid',
    time_col='timestamp',
    target_col='voltage_measured',
    agg_fn='mean'
)

# display_metrics(TFT_metrics.to_dict(orient='records'))
# display_metrics(TFT_metrics_agg.to_dict(orient='records'))

# display_metrics(TFT_metrics.to_dict(orient='records'), save_path=output / "TFT_metrics_individual.html")
# display_metrics(TFT_metrics_agg.to_dict(orient='records'), save_path=output / "TFT_metrics_agg.html")

# print(TFT_metrics_agg)

# TFT Tuned
config_file_path = "scripts/ch16/TFT_best_config.json"

try:
    with open(config_file_path, 'r') as config_file:
        loaded_config = json.load(config_file)
        print(loaded_config)
except FileNotFoundError:
    display(HTML("""
    <div class="alert alert-block alert-warning">
    <b>Warning!</b> File not found. 
    </div>
    """))
    
TFT_config = {
    "max_steps": max_steps,
    #"input_size": tune.choice([h, h*7, h*7*2, h*7*3]),
    "input_size": tune.choice([h, h*7]),
    "learning_rate": tune.loguniform(1e-2, 1e-1),
    "scaler_type": tune.choice(["minmax", "standard"]),
    "batch_size": tune.choice([32, 64]),
    "valid_batch_size": 16,

}

# Model setup
if RETUNE:
    models = [AutoTFT(h=h, 
                      config=TFT_config,
                      loss=MAE(),
                      search_alg=HyperOptSearch(),
                      backend='ray',
                      num_samples=100)]  # Reduced num_samples for faster tuning
else:
    models = [AutoTFT(h=h, 
                      config=loaded_config,
                      search_alg=None,
                      backend='ray')]

# Fit the model
model_tuned = NeuralForecast(models=models, freq='15s')

model_tuned.fit(training_df[['BTid', 'timestamp', 'voltage_measured']], 
                id_col='BTid', 
                time_col='timestamp', 
                target_col='voltage_measured', 
                )

if RETUNE == True:
    TFT_best_config = model_tuned.models[0].results.get_best_result().config

    # Remove specific keys using the pop method and then saving so we can extract parameters later
    # TFT_best_config.pop("loss", None)
    # TFT_best_config.pop("valid_loss", None)
    # TFT_best_config.pop("h", None)
    TFT_best_config = {k: v for k, v in TFT_best_config.items() if k not in ['loss', 'valid_loss', 'h', 'verbose']}

    # Save the filtered configuration to a JSON file
    with open(config_file_path, 'w') as config_file:
        json.dump(TFT_best_config, config_file, indent=4)

    print("Best configuration as string:")
    print(TFT_best_config)

# results = model_tuned.models[0].results.get_dataframe()
# print(results[['loss', 'train_loss', 'timestamp', 
#        'training_iteration', 
#         'config/max_steps', 'config/input_size',
#        'config/learning_rate', 'config/h', 'config/loss',
#        ]].head(2))

future_df_autoTFT = model_tuned.make_future_dataframe(df=training_df[['BTid', 'timestamp', 'voltage_measured']])

pred_df_autoTFT = model_tuned.predict(futr_df=future_df_autoTFT).reset_index()

pred_df_autoTFT = pred_df_autoTFT.merge(
        validation_df[['BTid', 'timestamp', 'voltage_measured']], 
        on=["BTid", "timestamp"], 
        how="left",)

# print(pred_df_autoTFT.head(2))


fcst_mase = partial(mase, seasonality=48)

# === Evaluate individual BTid series ===
autoTFT_metrics = evaluate(
    pred_df_autoTFT,
    metrics=[rmse, mae, mse, fcst_mase],
    train_df=training_df[['timestamp', 'BTid', 'voltage_measured']],
    id_col='BTid',
    time_col='timestamp',
    target_col='voltage_measured'
)

# === Evaluate aggregated metrics across all BTids ===
autoTFT_metrics_agg = evaluate(
    pred_df_autoTFT,
    metrics=[rmse, mae, mse, fcst_mase],
    train_df=training_df[['timestamp', 'BTid', 'voltage_measured']],
    id_col='BTid',
    time_col='timestamp',
    target_col='voltage_measured',
    agg_fn='mean'
)

print(autoTFT_metrics_agg)

if hasattr(ray, "is_initialized") and ray.is_initialized():
    ray.shutdown()

# TEST SET Predictions
h_test = len(test_df.timestamp.unique()) # horizon of the test set

if RETUNE == True:
    config = TFT_best_config
else:
    config = loaded_config

# models_test = [AutoTFT(h=h_test, 
#                     config = config,
#                     search_alg = None,
#                     backend = 'ray')]
# 
# models_test = NeuralForecast(models=models_test, freq='15s')
# models_test.fit(train_df[['BTid','timestamp','voltage_measured']], id_col = 'BTid',time_col = 'timestamp',target_col='voltage_measured')

final_test_model = TFT(
    h=h_test,
    input_size=h_test,        
    max_steps=5,                
    learning_rate=0.01,
    scaler_type='standard',
    batch_size=2,
    valid_batch_size=2,
)

models_test = NeuralForecast(models=[final_test_model], freq='15s')
models_test.fit(
    train_df[['BTid', 'timestamp', 'voltage_measured']],
    id_col='BTid',
    time_col='timestamp',
    target_col='voltage_measured'
)

futr_df_test = test_df[['BTid', 'timestamp', 'voltage_measured']]
pred_df_test = models_test.predict(futr_df=futr_df_test).reset_index()
pred_df_test = pred_df_test.merge(
        test_df[['BTid', 'timestamp', 'voltage_measured']], 
        on=["BTid", "timestamp"], 
        how="left",)
print(pred_df_test.head(2))


# fcst_mase = partial(mase, seasonality=48)
# 
# # === Evaluate individual BTid series ===
# TFT_metrics_test = evaluate(
#     pred_df_test,
#     metrics=[rmse, mae, mse, fcst_mase],
#     train_df=training_df[['timestamp', 'BTid', 'voltage_measured']],
#     id_col='BTid',
#     time_col='timestamp',
#     target_col='voltage_measured'
# )
# 
# # === Evaluate aggregated metrics across all BTids ===
# TFT_metrics_agg_test = evaluate(
#     pred_df_test,
#     metrics=[rmse, mae, mse, fcst_mase],
#     train_df=training_df[['timestamp', 'BTid', 'voltage_measured']],
#     id_col='BTid',
#     time_col='timestamp',
#     target_col='voltage_measured',
#     agg_fn='mean'
# )
# 
# print(TFT_metrics_agg_test.head())
# 
# TFT_metrics_agg_test.to_pickle(output/'TFT_metrics_agg_test.pkl')
# TFT_metrics_test.to_pickle(output/'TFT_metrics_test.pkl')
# 
# 
