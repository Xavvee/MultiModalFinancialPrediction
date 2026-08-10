import pandas as pd

# We load only the first row to check the headers instantly
df_test = pd.read_csv("data/new_dataset/raw/bitcoin_tweets_2025_26.csv", nrows=1, on_bad_lines='skip', low_memory=False)
print("--- DETECTED COLUMNS ---")
print(df_test.columns.tolist())