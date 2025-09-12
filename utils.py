import pandas as pd
import os

def load_and_preprocess_data(file_path: str, target_col: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    df = pd.read_csv(file_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Available: {df.columns.tolist()}")

    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Fill missing values in features
    X = X.fillna(-999)  # or use mean/median imputation later in pipeline

    return X, y
