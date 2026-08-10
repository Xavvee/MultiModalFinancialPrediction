import pandas as pd

# We load only the first row to check the headers instantly
df_test = pd.read_csv("data/old_dataset/raw/tweets.csv", nrows=1, on_bad_lines='skip', low_memory=False)
print("--- DETECTED COLUMNS ---")
print(df_test.columns.tolist())