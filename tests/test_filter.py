# tests/test_filter.py
import os
import subprocess
import joblib
import importlib.util

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
    assert os.path.exists(path), "mydata.csv missing"
    import pandas as pd
    df = pd.read_csv(path)
    assert not df.empty, "mydata.csv is empty"

def test_model_pickle_valid():
    model_path = "models/random_forest.pkl"
    assert os.path.exists(model_path), "random_forest.pkl missing"
    model = joblib.load(model_path)
    assert model is not None, "Failed to load trained model"

def test_visualizations_exist():
    # 1️⃣ Ensure visualizations.py exists
    viz_file = "visualizations.py"
    assert os.path.exists(viz_file), "visualizations.py script is missing"

    # 2️⃣ Ensure the script is importable
    spec = importlib.util.spec_from_file_location("visualizations", viz_file)
    assert spec is not None, "Failed to load visualizations.py"

    # 3️⃣ Ensure the output folder exists
    viz_dir = "data/visuals"
    assert os.path.isdir(viz_dir), f"{viz_dir} folder missing. Run visualizations.py to generate it."

    # 4️⃣ Ensure plots exist inside folder
    files = os.listdir(viz_dir)
    plot_files = [f for f in files if f.endswith(".png")]
    assert len(plot_files) > 0, "No plots generated in data/visuals"
