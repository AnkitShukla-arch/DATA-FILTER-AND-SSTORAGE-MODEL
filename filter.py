import os
import pandas as pd
import numpy as np
from utils import safe_makedirs

INPUT_DIR = "data/incoming/"
OUTPUT_DIR = "data/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "filtered_output.csv")

safe_makedirs(OUTPUT_DIR)

def detect_target_column(df):
    # Heuristic: numeric column with less than 10 unique values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() < 50:
            return col
    return df.columns[-1]  # fallback to last column

def clean_data(df):
    # Convert numeric-like strings to numbers
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('"', ''), errors='ignore')

    # Fill NaNs
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].median())

    return df

def detect_pattern(df):
    rows, cols = df.shape
    pattern = "Unknown"
    if cols == 5:
        pattern = "5D"
    elif cols == 3:
        pattern = "3D"

    # Star vs snowflake detection
    unique_rows = df.drop_duplicates().shape[0]
    pattern_type = "Star" if unique_rows < rows else "Snowflake"
    return pattern, pattern_type

def main():
    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

    # Read all files and concatenate
    df_list = [pd.read_csv(os.path.join(INPUT_DIR, f)) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)

    df = clean_data(df)
    target_column = detect_target_column(df)
    pattern, pattern_type = detect_pattern(df)

    print(f"Detected target column: {target_column}")
    print(f"Detected pattern: {pattern}, type: {pattern_type}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Filtered CSV saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
