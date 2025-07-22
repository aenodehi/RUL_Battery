import pandas as pd

df = pd.read_csv("./data/B0005_discharge_adjusted.csv", index_col = "datetime_", parse_dates = True)
#print(df.head())
df['datetime'] = pd.to_datetime(df['datetime'], yearfirst=True)
print(df.datetime.min(), df.datetime.max())
