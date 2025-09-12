# filter.py (sunshot version - handles messy datasets)
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utils import safe_makedirs

# Define input/output paths
RAW_DATA_PATH = "data/incoming/mydata.csv"
FILTERED_DATA_PATH = "data/processed/filtered_output.csv"
LOG_PATH = "data/logs/cleaning_log.txt"

# Ensure directories exist
for path in [os.path.dirname(FILTERED_DATA_PATH), os.path.dirname(LOG_PATH)]:
    safe_makedirs(path, force=True)


def clean_and_filter_data(input_path=RAW_DATA_PATH, output_path=FILTERED_DATA_PATH, log_path=LOG_PATH):
    """
    Cleans and filters a raw CSV dataset:
    - Handles NaN values (numeric → median, categorical → mode)
    - Removes duplicates
    - Encodes categorical variables (LabelEncoder)
    - Ensures final dataset is not empty
    - Logs all cleaning steps
    """

    log_messages = []

    try:
        df = pd.read_csv(input_path)
        log_messages.append(f"Loaded dataset from {input_path} with shape {df.shape}.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Drop duplicate rows
    before_dupes = df.shape[0]
    df = df.drop_duplicates()
    after_dupes = df.shape[0]
    log_messages.append(f"Removed {before_dupes - after_dupes} duplicate rows.")

    # Handle NaN values
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            log_messages.append(f"Filled NaNs in numeric column '{col}' with median ({median_val}).")
        else:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col].fillna(mode_val, inplace=True)
            log_messages.append(f"Filled NaNs in categorical column '{col}' with mode ({mode_val}).")

    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        log_messages.append(f"Encoded categorical column '{col}' with LabelEncoder.")

    # Ensure dataset not empty
    if df.empty:
        raise ValueError("Filtered dataset is empty after cleaning.")

    # Save cleaned dataset
    df.to_csv(output_path, index=False)
    log_messages.append(f"Cleaned dataset saved to {output_path} with shape {df.shape}.")

    # Save logs
    with open(log_path, "w") as log_file:
        log_file.write("\n".join(log_messages))

    return df


if __name__ == "__main__":
    final_df = clean_and_filter_data()
    print(f"✅ Cleaning complete. Final dataset shape: {final_df.shape}")
