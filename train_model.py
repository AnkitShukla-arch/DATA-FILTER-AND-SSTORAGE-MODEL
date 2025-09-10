# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from utils import safe_makedirs

# CONFIG
CSV_PATH = "data/filtered_output.csv"  # your CSV path
TARGET_COLUMN = "your_target_column"   # replace with your label column

# Ensure models directory exists
safe_makedirs("models")

# Load dataset
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

if TARGET_COLUMN not in df.columns:
    raise KeyError(f"Target column '{TARGET_COLUMN}' not found in CSV. Available columns: {df.columns.tolist()}")

# Split features and target
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Full pipeline with RandomForest
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train-test split (optional, can skip if training on full data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model_pipeline.fit(X_train, y_train)

# Save trained model
joblib.dump(model_pipeline, "models/random_forest.pkl")
print("Model trained and saved successfully at 'models/random_forest.pkl'")
