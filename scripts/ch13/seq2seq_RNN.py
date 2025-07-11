# This is the main code for Seq to Seq RNN 

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
from src.dl.models import RNNConfig, Seq2SeqConfig, Seq2SeqModel
import pytorch_lightning as pl
import torch
# For reproduceability set a random seed
pl.seed_everything(42)

np.random.seed(42)
tqdm.pandas()

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
# print(train_df.columns)
# print(test_df.columns)

# sel_lclids = train_df.unique_id.unique().tolist()
target = "y"
index_cols = ["unique_id", "ds"]

train_df.set_index(index_cols, inplace=True, drop=False)
test_df.set_index(index_cols, inplace=True, drop=False)
# pred_df = pd.concat([train_df[[target]], test_df[[target]]])

# Running the RNN on a Sample 
# Selecting the sample data and metrics

sample_train_df = train_df.xs("B0005")
sample_test_df = test_df.xs("B0005")
# Creating a pred_df with actuals
pred_df = pd.concat([sample_train_df[[target]], sample_test_df[[target]]])

sample_val_df = sample_train_df.loc["2008-05-12":]
sample_train_df = sample_train_df.loc[:"2008-05-12"]

sample_train_df['type'] = "train"
sample_val_df['type'] = "val"
sample_test_df['type'] = "test"
sample_df = pd.concat([sample_train_df[[target, "type"]], sample_val_df[[target, "type"]], sample_test_df[[target, "type"]]])
sample_df.head()

# print(sample_train_df.head())

try:
    pred_df = pd.read_pickle(output/"dl_single_step_prediction_val_df_B0005_01.pkl")
    metric_record = joblib.load(output/"dl_single_step_metrics_val_df_B0005_01.pkl")
except FileNotFoundError:
    display(HTML("""
    <div class="alert alert-block alert-warning">
    <b>Warning!</b> File not found. Please make sure you have run One-Step RNN in Chapter13
    </div>
    """))


# Creating the datamodule which splits and formats the data into windows
HORIZON = 1
WINDOW = 48

# Creating the datamodule which splits and formats the data into windows
datamodule = TimeSeriesDataModule(data = sample_df[[target]],
        n_val = sample_val_df.shape[0],
        n_test = sample_test_df.shape[0],
        window = WINDOW, # giving enough memory to capture daily seasonality
        horizon = HORIZON, # single step
        normalize = "global", # normalizing the data
        batch_size = 32,
        num_workers = 0)
datamodule.setup()

# 1.LSTM-FC Seq2Seq
encoder_config = RNNConfig(
        input_size=1,
        hidden_size=128,
        num_layers=3,
        bidirectional=True,
        ).__dict__
rnn2fc_config = Seq2SeqConfig(
        encoder_type="LSTM",
        decoder_type="FC",
        encoder_params=encoder_config,
        decoder_params={"window_size":WINDOW, "horizon":HORIZON},
        decoder_use_all_hidden=False,
        learning_rate=1e-3,
        )
model = Seq2SeqModel(rnn2fc_config)

trainer = pl.Trainer(
        accelerator="auto",
        min_epochs=5,
        max_epochs=100,
        callbacks=[pl.callbacks.EarlyStopping(monitor="valid_loss", patience=3)],
        )
trainer.fit(model, datamodule)
shutil.rmtree("lightning_logs")

tag = f"{rnn2fc_config.encoder_type}_{rnn2fc_config.decoder_type}_{'all_hidden' if rnn2fc_config.decoder_use_all_hidden else 'last_hidden'}"
pred = trainer.predict(model, datamodule.test_dataloader())
pred = torch.cat(pred).squeeze().detach().numpy()
pred = pred * datamodule.train.std + datamodule.train.mean
pred_df_ = pd.DataFrame({tag: pred}, index=sample_test_df.index)
pred_df = pred_df.join(pred_df_)
metrics = calculate_metrics(
        sample_test_df[target],
        pred_df_[tag],
        tag,
        pd.concat([sample_train_df[target], sample_val_df[target]])
        )
metric_record.append(metrics)

# 2.LSTM-FC Seq2Seq use all hidden
encoder_config = RNNConfig(
        input_size=1,
        hidden_size=128,
        num_layers=3,
        bidirectional=True,
        ).__dict__
rnn2fc_config = Seq2SeqConfig(
        encoder_type="LSTM",
        decoder_type="FC",
        encoder_params=encoder_config,
        decoder_params={"window_size":WINDOW, "horizon":HORIZON},
        decoder_use_all_hidden=True,
        learning_rate=1e-3,
        )

model = Seq2SeqModel(rnn2fc_config)

trainer = pl.Trainer(
        accelerator="auto",
        min_epochs=5,
        max_epochs=100,
        callbacks=[pl.callbacks.EarlyStopping(monitor="valid_loss", patience=3)],
        )
trainer.fit(model, datamodule)
shutil.rmtree("lightning_logs")

tag = f"{rnn2fc_config.encoder_type}_{rnn2fc_config.decoder_type}_{'all_hidden' if rnn2fc_config.decoder_use_all_hidden else 'last_hidden'}"
pred = trainer.predict(model, datamodule.test_dataloader())
pred = torch.cat(pred).squeeze().detach().numpy()
pred = pred * datamodule.train.std + datamodule.train.mean
pred_df_ = pd.DataFrame({tag:pred}, index=sample_test_df.index)
pred_df = pred_df.join(pred_df_)

metrics = calculate_metrics(
        sample_test_df[target],
        pred_df_[tag],
        tag,
        pd.concat([sample_train_df[target], sample_val_df[target]])
        )
metric_record.append(metrics)



# 3. LSTM-LSTM Seq2Seq
encoder_config = RNNConfig(
        input_size=1,
        hidden_size=128,
        num_layers=3,
        bidirectional=True,
        ).__dict__
rnn2rnn_config = Seq2SeqConfig(
        encoder_type="LSTM",
        decoder_type="LSTM",
        encoder_params=encoder_config,
        decoder_params=encoder_config,
        learning_rate=1e-3,
        )

model = Seq2SeqModel(rnn2rnn_config)

trainer = pl.Trainer(
        accelerator="auto",
        min_epochs=5,
        max_epochs=100,
        callbacks=[pl.callbacks.EarlyStopping(monitor="valid_loss", patience=3)],
        )
trainer.fit(model, datamodule)
shutil.rmtree("lightning_logs")

tag = f"{rnn2rnn_config.encoder_type}_{rnn2rnn_config.decoder_type}"
pred = trainer.predict(model, datamodule.test_dataloader())
pred = torch.cat(pred).squeeze().detach().numpy()
pred = pred * datamodule.train.std + datamodule.train.mean
pred_df_ = pd.DataFrame({tag:pred}, index=sample_test_df.index)
pred_df = pred_df.join(pred_df_)
metrics = calculate_metrics(
        sample_test_df[target],
        pred_df_[tag],
        tag,
        pd.concat([sample_train_df[target], sample_val_df[target]])
        )
metric_record.append(metrics)

# Multi-Step Prediction
HORIZON = 48
WINDOW = 48*2

datamodule = TimeSeriesDataModule(data=sample_df[[target]],
                                  n_val=sample_val_df.shape[0],
                                  n_test=sample_test_df.shape[0],
                                  window=WINDOW,
                                  horizon=HORIZON,
                                  normalize="global",
                                  batch_size=32,
                                  num_workers=0)
datamodule.setup()

# 4. LSTM-FC Seq2Seq Last Hidden
encoder_config = RNNConfig(
        input_size=1,
        hidden_size=128,
        num_layers=3,
        bidirectional=True,
        ).__dict__
rnn2fc_config = Seq2SeqConfig(
        encoder_type="LSTM",
        decoder_type="FC",
        encoder_params=encoder_config,
        decoder_params={"window_size":WINDOW, "horizon":HORIZON},
        decoder_use_all_hidden=False,
        learning_rate=1e-3,
        )
model = Seq2SeqModel(rnn2fc_config)

trainer = pl.Trainer(
        accelerator="auto",
        min_epochs=5,
        max_epochs=100,
        callbacks=[pl.callbacks.EarlyStopping(monitor="valid_loss", patience=3)],
        )
trainer.fit(model, datamodule)
shutil.rmtree("lightning_logs")

tag = f"Multi-Step {rnn2fc_config.encoder_type}_{rnn2fc_config.decoder_type}_{'all_hidden' if rnn2fc_config.decoder_use_all_hidden else 'last_hidden'}"
pred = trainer.predict(model, datamodule.test_dataloader())
pred = torch.cat(pred).squeeze().detach().numpy()
pred = pred[0::48].ravel()
pred = pred * datamodule.train.std + datamodule.train.mean
pred_index = sample_test_df.index[:len(pred)]
pred_df_ = pd.DataFrame({tag:pred}, index=pred_index)
pred_df = pred_df.join(pred_df_)
metrics = calculate_metrics(
        sample_test_df[target],
        pred_df_[tag],
        tag,
        pd.concat([sample_train_df[target], sample_val_df[target]]),
        )
metric_record.append(metrics)

# 5. LSTM-FC Seq2Seq All Hidden
encoder_config = RNNConfig(
        input_size=1,
        hidden_size=128,
        num_layers=3,
        bidirectional=True,
        ).__dict__

rnn2fc_config = Seq2SeqConfig(
        encoder_type="LSTM",
        decoder_type="FC",
        encoder_params=encoder_config,
        decoder_params={"window_size": WINDOW, "horizon":HORIZON},
        decoder_use_all_hidden=True,
        learning_rate=1e-3,
        )
model = Seq2SeqModel(rnn2fc_config)

trainer = pl.Trainer(
        accelerator="auto",
        min_epochs=5,
        max_epochs=100,
        callbacks=[pl.callbacks.EarlyStopping(monitor="valid_loss", patience=3)],
        )
trainer.fit(model, datamodule)
shutil.rmtree("lightning_logs")

tag = f"MultiStep {rnn2fc_config.encoder_type}_{rnn2fc_config.decoder_type}_{'all_hidden' if rnn2fc_config.decoder_use_all_hidden else 'last_hidden'}"
pred = trainer.predict(model, datamodule.test_dataloader())
pred = torch.cat(pred).squeeze().detach().numpy()
pred = pred[0::48].ravel()
pred = pred * datamodule.train.std + datamodule.train.mean
pred_index = sample_test_df.index[:len(pred)]
pred_df_ = pd.DataFrame({tag: pred}, index=pred_index)
pred_df = pred_df.join(pred_df_)
metrics = calculate_metrics(
        sample_test_df[target],
        pred_df_[tag],
        tag,
        pd.concat([sample_train_df[target], sample_val_df[target]]),
        )
metric_record.append(metrics)

# 6. LSTM-RNN Seq2Seq No Teacher Forcing
encoder_config = RNNConfig(
        input_size=1,
        hidden_size=128,
        num_layers=3,
        bidirectional=True,
        ).__dict__
rnn2rnn_config = Seq2SeqConfig(
        encoder_type="LSTM",
        decoder_type="LSTM",
        encoder_params=encoder_config,
        decoder_params=encoder_config,
        teacher_forcing_ratio=0.0,
        learning_rate=1e-3,
        )
model = Seq2SeqModel(rnn2rnn_config)

trainer = pl.Trainer(
        accelerator="auto",
        min_epochs=5,
        max_epochs=100,
        callbacks=[pl.callbacks.EarlyStopping(monitor="valid_loss", patience=3)],
        )
trainer.fit(model, datamodule)
shutil.rmtree("lightning_logs")

tag = f"MultiStep {rnn2rnn_config.encoder_type}_{rnn2rnn_config.decoder_type}_teacher_forcing_{rnn2rnn_config.teacher_forcing_ratio}"
pred = trainer.predict(model, datamodule.test_dataloader())
pred = torch.cat(pred).squeeze().detach().numpy()
pred = pred[0::48].ravel()
pred = pred * datamodule.train.std + datamodule.train.mean
pred_index = sample_test_df.index[:len(pred)]
pred_df_ = pd.DataFrame({tag: pred}, index=pred_index)
pred_df = pred_df.join(pred_df_)
metrics = calculate_metrics(
        sample_test_df[target],
        pred_df_[tag],
        tag,
        pd.concat([sample_train_df[target], sample_val_df[target]])
        )
metric_record.append(metrics)


# 7. LSTM-RNN Seq2Seq With Stochastic Teacher Forcing
# warning - next block can take a few hours.

encoder_config = RNNConfig(
        input_size=1,
        hidden_size=128,
        num_layers=3,
        bidirectional=True,
        ).__dict__
rnn2rnn_config = Seq2SeqConfig(
        encoder_type="LSTM",
        decoder_type="LSTM",
        encoder_params=encoder_config,
        decoder_params=encoder_config,
        teacher_forcing_ratio=0.5,
        learning_rate=1e-3,
        )
model = Seq2SeqModel(rnn2rnn_config)
trainer = pl.Trainer(
        accelerator="auto",
        min_epochs=5,
        max_epochs=100,
        callbacks=[pl.callbacks.EarlyStopping(monitor="valid_loss", patience=3)],
        )
trainer.fit(model, datamodule)
shutil.rmtree("lightning_logs")

tag = f"MultiStep {rnn2rnn_config.encoder_type}_{rnn2rnn_config.decoder_type}_teacher_forcing_{rnn2rnn_config.teacher_forcing_ratio}"
pred = trainer.predict(model, datamodule.test_dataloader())
pred = torch.cat(pred).squeeze().detach().numpy()
pred = pred[0::48].ravel()
pred = pred * datamodule.train.std + datamodule.train.mean
pred_index = sample_test_df.index[:len(pred)]
pred_df_ = pd.DataFrame({tag: pred}, index=pred_index)
pred_df = pred_df.join(pred_df_)
metrics = calculate_metrics(
        sample_test_df[target],
        pred_df_[tag],
        tag,
        pd.concat([sample_train_df[target], sample_val_df[target]])
        )
metric_record.append(metrics)

# 8. LSTM-RNN Seq2Seq With complete Teacher Forcing
# warning - next block can take a few hours
encoder_config = RNNConfig(
        input_size=1,
        hidden_size=128,
        num_layers=3,
        bidirectional=True,
        ).__dict__
rnn2rnn_config = Seq2SeqConfig(
        encoder_type="LSTM",
        decoder_type="LSTM",
        encoder_params=encoder_config,
        decoder_params=encoder_config,
        teacher_forcing_ratio=1,
        learning_rate=1e-3,
        )
model = Seq2SeqModel(rnn2rnn_config)

trainer = pl.Trainer(
        accelerator="auto",
        min_epochs=5,
        max_epochs=100,
        callbacks=[pl.callbacks.EarlyStopping(monitor="valid_loss", patience=3)],
        )
trainer.fit(model, datamodule)
shutil.rmtree("lightning_logs")

tag = f"MultiStep {rnn2rnn_config.encoder_type}_{rnn2rnn_config.decoder_type}_teacher_forcing_{rnn2rnn_config.teacher_forcing_ratio}"
pred = trainer.predict(model, datamodule.test_dataloader())
pred = torch.cat(pred).squeeze().detach().numpy()
pred = pred[0::48].ravel()
pred = pred * datamodule.train.std + datamodule.train.mean
pred_index = sample_test_df.index[:len(pred)]
pred_df_ = pd.DataFrame({tag:pred}, index=pred_index)
pred_df = pred_df.join(pred_df_)
metrics = calculate_metrics(
        sample_test_df[target],
        pred_df_[tag],
        tag,
        pd.concat([sample_train_df[target], sample_val_df[target]])
        )
metric_record.append(metrics)

shutil.rmtree("lightning_logs")



output_path = output_img / "metric_record_seq2seq_RNN.html"
display_metrics(metric_record, save_path=output_path)

pred_df.to_pickle(output/"dl_seq_2_seq_prediction_val_df_B0005.pkl")
joblib.dump(metric_record, output/"dl_seq_2_seq_metrics_val_df_B0005.pkl")






