import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.autonotebook import tqdm
import warnings
import joblib

from src.transforms.target_transformations import AutoStationaryTransformer

# === Setup ===
np.random.seed(42)
tqdm.pandas()
warnings.filterwarnings("ignore")

# === Paths ===
preprocessed = Path("./data")
output_dir = Path("imgs")
output_dir.mkdir(exist_ok=True)

# === Load Train + Validation Sets ===
try:
    train_df = pd.read_parquet(preprocessed / "selected_blocks_train_missing_imputed_feature_engg.parquet")
    val_df = pd.read_parquet(preprocessed / "selected_blocks_val_missing_imputed_feature_engg.parquet")
    # print(f"📂 Train rows: {train_df.shape[0]}")
    # print(f"📂 Val rows:   {val_df.shape[0]}")   
    train_df["type"] = "train"
    val_df["type"] = "val"
    df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"🧩 Combined total rows: {df.shape[0]}")
    del train_df, val_df
except FileNotFoundError as e:
    raise FileNotFoundError(f"❌ Required file missing: {e}")

df["ds"] = pd.to_datetime(df["ds"])

# === Parameters ===
FREQ = "15s"
SEASONAL_PERIOD = 5760 * 7    # one week of 15s steps
MIN_ROWS = 100

# === Containers ===
transformer_pipelines = {}
all_results = []

# === Streaming transformation per series ===
for uid in tqdm(df["unique_id"].unique()):
    # isolate the series
    ser = (
        df[df["unique_id"] == uid]
        .set_index("ds")
        .sort_index()["y"]
    )

    if len(ser) < MIN_ROWS:
        continue

    # 1) Reindex & fill to uniform 15s
    print(f"🔎 Original row count for {uid}: {len(ser)}")
    full_idx = pd.date_range(ser.index.min(), ser.index.max(), freq=FREQ)
    print(f"🧱 New full index row count: {len(full_idx)}")
    # ser = ser.reindex(full_idx).interpolate("time").ffill().bfill()
    ser.index.name = "ds"

    # 2) Instantiate and fit on first point
    transformer = AutoStationaryTransformer(seasonal_period=SEASONAL_PERIOD)
    first_ts = ser.index[:1]
    init = ser.loc[first_ts]
    transformer.fit(init, freq=FREQ)

    # 3) Transform first point
    y0 = transformer.transform(init)
    records = [(uid, first_ts[0], y0.iloc[0])]

    # 4) Stream-transform remaining timestamps
    for t in ser.index[1:]:
        window = ser.loc[:t]
        y_t = transformer.transform(window)
        records.append((uid, t, y_t.iloc[-1]))

    # collect pipeline and results
    transformer_pipelines[uid] = transformer
    df_out = pd.DataFrame(records, columns=["unique_id","ds","y_auto_stat"])
    all_results.append(df_out)

# === Combine all results ===
out_df = pd.concat(all_results, ignore_index=True)

# === Reindex into desired final DataFrame ===
train_out = (
    out_df
    .set_index(["unique_id","ds"])
)

# === Preview ===
print(train_out.head())

# === Save ===
train_out.to_parquet(preprocessed / "selected_blocks_train_auto_stat_target.parquet")
joblib.dump(transformer_pipelines, preprocessed / "auto_transformer_pipelines_train.pkl")

print("✅ Done. Stationary target saved and pipelines dumped.")
print(f"📊 Total rows in transformed output: {train_out.shape[0]}")
