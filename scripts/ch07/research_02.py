import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.autonotebook import tqdm
import warnings
import joblib

# === Setup ===
np.random.seed(42)
tqdm.pandas()
warnings.filterwarnings("ignore")

# === Paths ===
preprocessed = Path("./data")
output_dir = Path("imgs")
output_dir.mkdir(exist_ok=True)

# === Load the feature-engineered data ===
try:
    train_df = pd.read_parquet(preprocessed / "selected_blocks_train_missing_imputed_feature_engg.parquet")
except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
    exit(1)

# === Parameters ===
min_required_rows = 100
seasonal_lag = 5760     # one day of 15s samples (96 per hour × 24)
diffs = [1, seasonal_lag]

# === Prepare to collect transformed series ===
transformed_dfs = []

for _id in tqdm(train_df["unique_id"].unique()):
    sub = train_df[train_df["unique_id"] == _id][["ds", "y"]].copy()
    sub["ds"] = pd.to_datetime(sub["ds"])
    sub.set_index("ds", inplace=True)
    sub = sub.sort_index()

    if len(sub) < min_required_rows:
        print(f"[SKIP] {_id}: too few rows ({len(sub)})")
        continue

    # 1. reindex to uniform 15s
    idx = pd.date_range(sub.index.min(), sub.index.max(), freq="15s")
    sub = sub.reindex(idx)
    sub.index.name = "ds"
    # interpolate missing
    sub["y"] = sub["y"].interpolate("time").ffill().bfill()

    # 2. apply differencing in sequence
    y = sub["y"]
    for lag in diffs:
        y = y.diff(lag)
    # any remaining NaNs (first few rows), fill with zero
    y.fillna(0, inplace=True)

    # 3. assemble back into DataFrame
    out = y.to_frame(name="y_stat").reset_index()
    out["unique_id"] = _id
    transformed_dfs.append(out)

# === Combine and save ===
if not transformed_dfs:
    raise ValueError("No series transformed – check your data.")

all_y_stat_df = pd.concat(transformed_dfs, ignore_index=True)
all_y_stat_df.to_parquet(output_dir / "train_manual_diff.parquet", index=False)

print("✅ Manual differencing complete. File saved to", output_dir / "train_manual_diff.parquet")
