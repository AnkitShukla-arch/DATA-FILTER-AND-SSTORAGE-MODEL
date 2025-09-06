import os
import json
import pandas as pd
from datetime import datetime


def get_latest_csv(input_dir: str = "data/incoming") -> str:
    """
    Return the path of the latest CSV file in the given directory.
    Raises FileNotFoundError if no CSV files exist.
    """
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}/")

    latest_file = sorted(
        csv_files, key=lambda f: os.path.getmtime(os.path.join(input_dir, f))
    )[-1]
    return os.path.join(input_dir, latest_file)


def save_schema_metadata(df: pd.DataFrame, output_path: str = "data/curated/schema_metadata.json") -> str:
    """
    Save column names and dtypes of a DataFrame to JSON.
    """
    schema = {
        "generated_at": datetime.now().isoformat(),
        "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "rows": len(df),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(schema, f, indent=4)

    return output_path


def ensure_directories():
    """
    Make sure all required directories exist.
    """
    for folder in ["data/incoming", "data/curated", "data/visualizations", "models"]:
        os.makedirs(folder, exist_ok=True)


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file safely into a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    return pd.read_csv(file_path)


def save_dataframe(df: pd.DataFrame, output_path: str = "data/curated/filtered_output.csv") -> str:
    """
    Save a DataFrame to CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path

