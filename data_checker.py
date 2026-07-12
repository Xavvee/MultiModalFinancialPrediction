import pandas as pd

df_test = pd.read_csv("tweets.csv", nrows=10, on_bad_lines='skip', low_memory=False)
print("--- DETECTED COLUMNS ---")
print(df_test.columns.tolist())
print("\n--- SAMPLE DATA ---")
print(df_test.head(10))