
import pandas as pd

df = pd.read_csv("parking_data.csv")

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nFIRST 5 ROWS:")
print(df.head().to_string())

print("\nUNIQUE VALUES:")
for col in ["Day", "Parking_Zone", "Working_Day", "Weather"]:
    if col in df.columns:
        print(col, ":", df[col].dropna().unique()[:20])
