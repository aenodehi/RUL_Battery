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
from src.utils import ts_utils
from IPython.display import display, HTML

from src.dl.dataloaders import TimeSeriesDataModule
from src.dl.models import SingleStepRNNConfig, SingleStepRNNModel
import pytorch_lightning as pl
import torch
# For reproduceability set a random seed
pl.seed_everything(42)

np.random.seed(42)
tqdm.pandas()


def evaluate_forecast(y_pred, test_target, train_target, model_name):
    metric_l = []
    for uid in tqdm(test_target.index.get_level_values(0).unique(), desc="Calculating metrics..."):
        y_true_id = test_target.xs(uid)[target]
        y_pred_id = y_pred.xs(uid)[model_name]
        history_id = train_target.xs(uid)[target]
        metric_l.append(
            calculate_metrics(y_true_id, y_pred_id, name=model_name, y_train=history_id)
        )

    eval_metrics_df = pd.DataFrame(metric_l)

    true = test_target[target]
    pred = y_pred[model_name]

    agg_metrics = {
        "Algorithm": model_name,
        "MAE": ts_utils.mae(true, pred),
        "MSE": ts_utils.mse(true, pred),
        "meanMASE": eval_metrics_df["MASE"].mean(),
        "Forecast Bias": ts_utils.forecast_bias_aggregate(true, pred),
    }
    return agg_metrics, eval_metrics_df

def highlight_abs_min(s, props=''):
    return np.where(s == np.nanmin(np.abs(s.values)), props, '')

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

# Loading the Single Step ML Forecast
try:
    single_step_ahead_ml_fc_df = pd.read_pickle(output/"ml_single_step_prediction_val_df.pkl")
    single_step_ahead_ml_metrics_df = pd.read_pickle(output/"ml_single_step_metrics_val_df.pkl")
    single_step_ahead_ml_agg_metrics_df = pd.read_pickle(output/"ml_single_step_aggregate_metrics_val.pkl")
except FileNotFoundError:
    display(HTML("""
    <div class="alert alert-block alert-warning">
    <b>Warning!</b> File not found. Please make sure you have run Forecasting with ML in Chapter08
    </div>
    """))


# Running the RNN on a Sample 
# Selecting the sample data and metrics

# print(train_df.index.get_level_values("ds").min)
# print(train_df.index.get_level_values("ds").max)


sample_train_dfs = []
sample_val_dfs = []

for uid in train_df.index.get_level_values("unique_id").unique():
    df = train_df.xs(uid, level="unique_id")
    sample_train = df.loc[:"2008-05-12"].copy()
    sample_val = df.loc["2008-05-12":].copy()

    sample_train["unique_id"] = uid
    sample_val["unique_id"] = uid

    sample_train_dfs.append(sample_train)
    sample_val_dfs.append(sample_val)

sample_train_df = pd.concat(sample_train_dfs)
sample_val_df = pd.concat(sample_val_dfs)

sample_train_df.set_index(["unique_id", "ds"], inplace=True)
sample_val_df.set_index(["unique_id", "ds"], inplace=True)
sample_test_df = test_df.copy()

sample_train_df['type'] = "train"
sample_val_df['type'] = "val"
sample_test_df['type'] = "test"
sample_df = pd.concat([sample_train_df[[target, "type"]], sample_val_df[[target, "type"]], sample_test_df[[target, "type"]]])

# print(sample_train_df.head())


metric_record = []
metric_record += (
    single_step_ahead_ml_metrics_df.loc[single_step_ahead_ml_metrics_df.unique_id == "B0005"]
    .drop(columns="unique_id")
    .to_dict(orient="records")
)
#print(metric_record)

#print(sample_df.head())


# Creating the datamodule which splits and formats the data into windows
datamodule = TimeSeriesDataModule(data = sample_df[[target]],
        n_val = sample_val_df.shape[0],
        n_test = sample_test_df.shape[0],
        window = 48, # giving enough memory to capture daily seasonality
        horizon = 1, # single step
        normalize = "global", # normalizing the data
        batch_size = 32,
        num_workers = 0)
datamodule.setup()

# Setting the config for the RNN and initializing the model
rnn_config = SingleStepRNNConfig(
    rnn_type="RNN",
    input_size=1,
    hidden_size=128,
    num_layers=3,
    bidirectional=True,
    learning_rate=1e-3,
)

model = SingleStepRNNModel(rnn_config)

# Manual Inspection
for batch in datamodule.train_dataloader():
    x, y = batch
    break
x = x.float()
y = y.float()

# print("Shape of x: ",x.shape)
# print("Shape of y: ",y.shape)

y_hat, y = model((x, y))
# print("Shape of y_hat: ",y_hat.shape)
# print("Shape of y: ",y.shape)

l = model.loss(y_hat, y)
# print(l)

# Full Training

# # Load the TensorBoard notebook extension
# %load_ext tensorboard
# os.makedirs(lightning_logs, exist_ok=True)
# %tensorboard --logdir lightning_logs/

trainer = pl.Trainer(
    accelerator="auto",
    min_epochs=5,
    max_epochs=100,
    callbacks=[pl.callbacks.EarlyStopping(monitor="valid_loss", patience=3)],
)
trainer.fit(model, datamodule)

shutil.rmtree("lightning_logs")

pred = trainer.predict(model, datamodule.test_dataloader())
# pred is a list of outputs, one for each batch
pred = torch.cat(pred).squeeze().detach().numpy()
# Apply reverse transformation because we applied global normalization
pred = pred * datamodule.train.std + datamodule.train.mean
pred_df_ = pd.DataFrame({rnn_config.rnn_type: pred}, index=sample_test_df.index)
pred_df = pred_df.join(pred_df_)



#metrics = calculate_metrics(
#        sample_test_df[target], 
#        pred_df_[rnn_config.rnn_type], 
#        rnn_config.rnn_type, 
#        pd.concat([sample_train_df[target],sample_val_df[target]]))
#metric_record.append(metrics)
#formatted = pd.DataFrame(metric_record)
#output_path = output_img/"metric_record_one_step.html
#display_metrics(formatted, save_path=output_path)


agg_metrics, eval_metrics_df = evaluate_forecast(
    y_pred=pred_df_[[rnn_config.rnn_type]], 
    test_target=sample_test_df[[target]], 
    train_target=pd.concat([
        sample_train_df[[target]],
        sample_val_df[[target]]
    ]),
    model_name=rnn_config.rnn_type
)

metric_record.append(agg_metrics)

# Running LSTMs and GRUs on a Sample
# LSTM
rnn_config = SingleStepRNNConfig(
    rnn_type="LSTM",
    input_size=1,
    hidden_size=128,
    num_layers=3,
    bidirectional=True,
    learning_rate=1e-3,
)

model = SingleStepRNNModel(rnn_config)

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=100,
    callbacks=[pl.callbacks.EarlyStopping(monitor="valid_loss", patience=3)],
)
trainer.fit(model, datamodule)
# Removing artifacts created during training
shutil.rmtree("lightning_logs")

pred = trainer.predict(model, datamodule.test_dataloader())
# pred is a list of outputs, one for each batch
pred = torch.cat(pred).squeeze().detach().numpy()
# Apply reverse transformation because we applied global normalization
pred = pred * datamodule.train.std + datamodule.train.mean
pred_df_ = pd.DataFrame({rnn_config.rnn_type: pred}, index=sample_test_df.index)
pred_df = pred_df.join(pred_df_)

agg_metrics, eval_metrics_df = evaluate_forecast(
    y_pred=pred_df_[[rnn_config.rnn_type]], 
    test_target=sample_test_df[[target]], 
    train_target=pd.concat([
        sample_train_df[[target]],
        sample_val_df[[target]]
    ]),
    model_name=rnn_config.rnn_type
)

metric_record.append(agg_metrics)

# GRU
rnn_config = SingleStepRNNConfig(
    rnn_type="GRU",
    input_size=1,
    hidden_size=128,
    num_layers=3,
    bidirectional=True,
    learning_rate=1e-3,
)

model = SingleStepRNNModel(rnn_config)

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=100,
    callbacks=[pl.callbacks.EarlyStopping(monitor="valid_loss", patience=3)],
)
trainer.fit(model, datamodule)
# Removing artifacts created during training
shutil.rmtree("lightning_logs")

pred = trainer.predict(model, datamodule.test_dataloader())
# pred is a list of outputs, one for each batch
pred = torch.cat(pred).squeeze().detach().numpy()
# Apply reverse transformation because we applied global normalization
pred = pred * datamodule.train.std + datamodule.train.mean
pred_df_ = pd.DataFrame({rnn_config.rnn_type: pred}, index=sample_test_df.index)
pred_df = pred_df.join(pred_df_)

agg_metrics, eval_metrics_df = evaluate_forecast(
    y_pred=pred_df_[[rnn_config.rnn_type]], 
    test_target=sample_test_df[[target]], 
    train_target=pd.concat([
        sample_train_df[[target]],
        sample_val_df[[target]]
    ]),
    model_name=rnn_config.rnn_type
)

metric_record.append(agg_metrics)


output_path = output_img / "metric_record_one_step.html"
display_metrics(metric_record, save_path=output_path)

pred_df.to_pickle(output/"dl_single_step_prediction_val_df_B0005.pkl")
joblib.dump(metric_record, output/"dl_single_step_metrics_val_df_B0005.pkl")

