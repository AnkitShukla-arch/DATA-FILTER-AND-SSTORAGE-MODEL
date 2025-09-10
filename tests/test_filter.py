import os
import subprocess
import pickle
import pandas as pd
import pytest
import joblib 

def run_script(script):
    """Helper to run a script and return result."""
    result = subprocess.run(
        ["python", script], capture_output=True, text=True
    )
    print(result.stdout)
    print(result.stderr)
    return result


def test_filter_runs():
    """Check filter.py runs successfully."""
    result = run_script("filter.py")
    assert result.returncode == 0, "filter.py crashed"


def test_train_model_runs():
    """Check train_model.py runs successfully."""
    if os.path.exists("train_model.py"):
        result = run_script("train_model.py")
        assert result.returncode == 0, "train_model.py crashed"


def test_filtered_csv_exists_and_valid():
    """Check that filtered CSV is created and not empty."""
    path = "data/filtered_data.csv"
    assert os.path.exists(path), "filtered_data.csv missing"
    df = pd.read_csv(path)
    assert not df.empty, "filtered_data.csv is empty"
    assert len(df.columns) > 0, "filtered_data.csv has no columns"


def test_model_pickle_valid():
    """Check random_forest.pkl exists and can be loaded."""
    model_path = "models/random_forest.pkl"
    assert os.path.exists(model_path), "random_forest.pkl missing"
    model=joblib.load(model_path)
    assert model is not None


def test_visualizations_exist():
    """Check that visualizations are generated."""
    viz_dir = "data/visualizations"
    assert os.path.exists(viz_dir), "visualizations folder missing"
    files = os.listdir(viz_dir)
    assert len(files) > 0, "No visualizations generated"

