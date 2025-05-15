import pandas as pd

df = pd.read_csv("./data/B0005_discharge_adjusted.csv", index_col = "datetime_", parse_dates = True)
#print(df.head())
df['datetime'] = pd.to_datetime(df['datetime'], yearfirst=True)
#print(df.datetime.min(), df.datetime.max())
print(f"Number of rows: {df.shape[0]}")

df = df[~df.index.duplicated(keep='first')]  # Removing duplicate timestamps (keeping the first)

# Train, Validation, Test Split (same as before)
train_size = int(0.8 * len(df))
val_size = int(0.1 * len(df))
test_size = len(df) - train_size - val_size

train = df[:train_size]
val = df[train_size:train_size+val_size]
test = df[train_size+val_size:]

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
print(f"Max Date in Train: {train.index.max()} | Min Date in Validation: {val.index.min()} | Min Date in Test: {test.index.min()}")

train.to_parquet("./data/B0005_train.parquet")
val.to_parquet("./data/B0005_val.parquet")
test.to_parquet("./data/B0005_test.parquet")
print("Saved train/val/test datasets to Parquet format in ./data/")

