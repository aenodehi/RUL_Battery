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

# === Load and prepare data ===
df = pd.read_parquet(preprocessed / "selected_blocks_train_missing_imputed_feature_engg.parquet")
df["ds"] = pd.to_datetime(df["ds"])

# === Parameters ===
FREQ = "15s"
SEASONAL_PERIOD = 5760 * 7  # one week of 15s steps

# === Containers ===
pipelines = {}
all_results = []

for uid in tqdm(df["unique_id"].unique()):
    sub = (
        df[df["unique_id"] == uid]
        .set_index("ds")
        .sort_index()["y"]
    )

    # Reindex & fill to ensure uniform frequency
    full_idx = pd.date_range(sub.index.min(), sub.index.max(), freq=FREQ)
    sub = sub.reindex(full_idx).interpolate("time").ffill().bfill()
    sub.index.name = "ds"

    # 1) Instantiate and fit on the first timestamp
    transformer = AutoStationaryTransformer(seasonal_period=SEASONAL_PERIOD)
    first_ts = sub.index[0:1]
    init_series = sub.loc[first_ts]
    transformer.fit(init_series, freq=FREQ)

    # 2) Transform first point
    y0 = transformer.transform(init_series)
    results = [ (first_ts[0], y0.iloc[0]) ]

    # 3) Now stream each subsequent timestamp
    for t in sub.index[1:]:
        window = sub.loc[:t]
        y_stat = transformer.transform(window)
        # grab only the last value for this timestamp
        results.append( (t, y_stat.iloc[-1]) )

    # 4) Save pipeline and results for this UID
    pipelines[uid] = transformer
    tmp = pd.DataFrame(results, columns=["ds", "y_stat"])
    tmp["unique_id"] = uid
    all_results.append(tmp)

# === Combine & Save ===
out = pd.concat(all_results, ignore_index=True)
out.to_parquet(output_dir / "train_streaming_auto_stat.parquet", index=False)
joblib.dump(pipelines, output_dir / "streaming_pipelines.pkl")

print("✅ Streaming transformation complete; saved to train_streaming_auto_stat.parquet")

