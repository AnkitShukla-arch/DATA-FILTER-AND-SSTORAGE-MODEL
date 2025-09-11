# filter.py
import os
import pandas as pd
from utils import safe_makedirs

RAW_PATH = "data/incoming/mydata.csv"
FILTERED_PATH = "data/filtered/filtered_data.csv"

# Ensure dirs
safe_makedirs("data/incoming")
safe_makedirs("data/filtered")

print("[INFO] Starting filter.py...")

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"[ERROR] Raw CSV not found at {RAW_PATH}")

df = pd.read_csv(RAW_PATH)
print(f"[INFO] Loaded raw CSV with shape {df.shape}")

# Basic cleaning
df = df.drop_duplicates()
df = df.dropna(axis=1, how="all")  # drop empty cols
df = df.dropna(axis=0, how="all")  # drop empty rows

# Save
df.to_csv(FILTERED_PATH, index=False)
print(f"[INFO] Filtered data saved to {FILTERED_PATH} with shape {df.shape}")
