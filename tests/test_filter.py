import os
import pytest
import pandas as pd
import joblib

from filter import main as filter_main

FILTERED_CSV = "data/filtered_output.csv"
MODEL_PATH = "models/random_forest.pkl"

def test_filtered_csv_created():
    """Check if filtered_output.csv is created successfully."""
    filter_main()
    assert os.path.exists(FILTERED_CSV), f"{FILTERED_CSV} not found"

def test_filtered_csv_non_empty():
    """Check if filtered CSV is not empty and has rows."""
    df = pd.read_csv(FILTERED_CSV)
    assert not df.empty, "Filtered CSV is empty"

def test_model_created_and_loadable():
    """Check if the RandomForest model is trained and loadable."""
    from train_model import pipeline, MODEL_PATH as TRAINED_MODEL_PATH
    assert os.path.exists(TRAINED_MODEL_PATH), f"{TRAINED_MODEL_PATH} not found"
    model = joblib.load(TRAINED_MODEL_PATH)
    # Check if pipeline has classifier
    assert hasattr(model.named_steps['classifier'], 'predict'), "Classifier not found in pipeline"

def test_detect_target_column_generic():
    """Ensure target column detection works generically."""
    from filter import detect_target_column
    df = pd.read_csv(FILTERED_CSV)
    target = detect_target_column(df)
    assert target in df.columns, "Target column detection failed"

def test_pattern_detection():
    """Check pattern detection (3D/5D, Star/Snowflake)."""
    from filter import detect_pattern
    df = pd.read_csv(FILTERED_CSV)
    pattern, pattern_type = detect_pattern(df)
    assert pattern in ['3D', '5D', 'Unknown'], f"Invalid pattern detected: {pattern}"
    assert pattern_type in ['Star', 'Snowflake'], f"Invalid pattern type: {pattern_type}"
