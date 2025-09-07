import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
from visualizations import plot_target_distribution, plot_feature_importance, plot_correlation_heatmap



def safe_makedirs(path, force=False):
    """
    Create directory at `path` if it doesn't exist.
    If a non-directory (file/symlink) exists at `path`:
      - if force=False: raise FileExistsError with a clear message
      - if force=True: remove the file and create the directory
    """
    if os.path.exists(path):
        if os.path.isdir(path):
            return
        # path exists but is not a dir
        if force:
            # remove the file/symlink then create directory
            os.remove(path)
            os.makedirs(path, exist_ok=True)
            return
        raise FileExistsError(
            f"Path already exists and is not a directory: {path!r}. "
            "Rename or remove it, or call safe_makedirs(path, force=True) to overwrite."
        )
    # path doesn't exist -> create
    os.makedirs(path, exist_ok=True)




# Local imports
from utils import (
    get_latest_csv,
    load_csv,
    save_schema_metadata,
    save_dataframe,
    ensure_directories,
)
from visualizations import (
    plot_correlation_heatmap,
    plot_target_distribution,
    plot_feature_importance,
)


def run_pipeline():
    """Main pipeline for data filtration, training, and visualization."""

    print("🔹 Ensuring directories exist...")
    ensure_directories()

    print("🔹 Fetching latest CSV file...")
    csv_path = get_latest_csv("data/incoming")
    print(f"   Using file: {csv_path}")

    print("🔹 Loading dataset...")
    df = load_csv(csv_path)
    print(f"   Shape: {df.shape}")

    print("🔹 Saving schema metadata...")
    save_schema_metadata(df)

    # Check if dataset has a target column
    if "target" not in df.columns:
        print("⚠️ No 'target' column found. Skipping training & visualizations.")
        return

    print("🔹 Splitting data...")
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🔹 Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("🔹 Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))


# === Safe directory creation ===
for folder in ["data", "data/visualizations", "models"]:
    os.makedirs(folder, exist_ok=True)


    print("🔹 Generating visualizations...")
    plot_correlation_heatmap(df)
    plot_class_distribution(df, "target")
    plot_feature_importance(model, X.columns)

    print("✅ Pipeline complete!")


if __name__ == "__main__":
    run_pipeline()
