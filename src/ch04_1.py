import pandas as pd

df = pd.read_csv("./data/B0005_discharge_adjusted.csv", index_col="datetime_", parse_dates=True)
df['datetime'] = pd.to_datetime(df['datetime'], yearfirst=True)

total = len(df)
train_end = int(0.8 * total)
val_end = train_end + int(0.1 * total)

# Train set (up to train_end)
train = df.iloc[:train_end]

# Validation set (start from train_end + 1 to avoid overlap)
val = df.iloc[train_end + 1:val_end]

# Test set (start after validation)
test = df.iloc[val_end:]

print(f"Number of rows: {total}")
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

print("Boundary check:")
print(f"Train max: {train.datetime.max()}")
print(f"Val min:   {val.datetime.min()}")
print(f"Val max:   {val.datetime.max()}")
print(f"Test min:  {test.datetime.min()}")


print("\nDatetime before and after the last row in Train:")
print(f"Last row in Train datetime: {train.datetime.max()}")
print(f"First row in Val datetime:  {val.datetime.min()}")
print(f"Datetime before last row in Train: {train.iloc[-2].datetime}")
print(f"Datetime after first row in Val: {val.iloc[0].datetime}")
