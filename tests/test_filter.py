# tests/test_filter.py
import os
import subprocess
import joblib

def run_script(script):
    return subprocess.run(["python", script], capture_output=True, text=True)

def test_filter_runs():
    result = run_script("filter.py")
    assert result.returncode == 0, f"filter.py crashed:\n{result.stderr}"

def test_train_model_runs():
    result = run_script("train_model.py")
    assert result.returncode == 0, f"train_model.py crashed:\n{result.stderr}"

def test_filtered_csv_exists_and_valid():
    path = "data/incoming/mydata.csv"
    assert os.path.exists(path), "filtered_data.csv missing"
    import pandas as pd
    df = pd.read_csv(path)
    assert not df.empty, "filtered_data.csv is empty"

def test_model_pickle_valid():
    model_path = "models/random_forest.pkl"
    assert os.path.exists(model_path), "random_forest.pkl missing"
    model = joblib.load(model_path)
    assert model is not None, "Failed to load trained model"

def test_visualizations_exist():
    viz_dir = "data/visuals"
    assert os.path.exists(viz_dir), "visualizations dir missing"
    files = os.listdir(viz_dir)
    assert len(files) > 0, "No visualizations generated"
