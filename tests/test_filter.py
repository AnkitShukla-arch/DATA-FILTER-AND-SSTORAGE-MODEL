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
    """Check random_forest.pkl exists and can be loaded safely."""
    model_path = "models/random_forest.pkl"
    
    # Check if file exists
    assert os.path.exists(model_path), "random_forest.pkl is missing"
    
    # Attempt to load model safely
    try:
        model = joblib.load(model_path)
    except (EOFError, KeyError, AttributeError, ImportError, ModuleNotFoundError) as e:
        pytest.fail(f"Failed to load random_forest.pkl: {e}")
    
    # Optional: check if loaded object has a 'predict' method
    assert hasattr(model, "predict"), "Loaded object is not a model with 'predict' method"


def test_visualizations_exist():
    """Check that visualizations are generated."""
    viz_dir = "data/visualizations"
    assert os.path.exists(viz_dir), "visualizations folder missing"
    files = os.listdir(viz_dir)
    assert len(files) > 0, "No visualizations generated"

