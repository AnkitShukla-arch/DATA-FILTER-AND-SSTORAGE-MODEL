# explore_data.py
import pandas as pd
import os
import sys

DATA = "data/incoming/mydata.csv"
na_values = ["null", "NULL", "NaN", "nan", "", " "]

if not os.path.exists(DATA):
    print(f"[ERROR] File not found: {DATA}")
    sys.exit(1)

df = pd.read_csv(DATA, na_values=na_values, keep_default_na=True)
df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nDtypes:\n", df.dtypes)
print("\nMissing counts:\n", df.isna().sum())
print("\nTop unique values for small categorical columns:")
for c in df.columns:
    if df[c].dtype == object and df[c].nunique() <= 10:
        print(f"  {c}: {df[c].unique().tolist()}")
