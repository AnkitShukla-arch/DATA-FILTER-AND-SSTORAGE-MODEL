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
import os

# CONFIG
CSV_PATH = "data/incoming/mydata.csv"  # replace with your CSV path
TARGET_COLUMN = "Completed"  # replace with your label column
MODEL_PATH = "models/random_forest.pkl"

# --- Ensure models directory exists ---
safe_makedirs("models")

# --- Load CSV safely ---
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# Strip quotes/spaces from headers
df.columns = df.columns.str.strip().str.replace('"', '')

# Check CSV is not empty
if df.empty:
    raise ValueError(f"The CSV at {CSV_PATH} is empty. Please provide valid data.")

# Check target column exists
if TARGET_COLUMN not in df.columns:
    raise KeyError(f"Target column '{TARGET_COLUMN}' not found. Available columns: {df.columns.tolist()}")

# --- Drop ID-like columns (high cardinality, mostly strings) ---
candidate_columns = df.drop(columns=[TARGET_COLUMN]).columns.tolist()
for col in candidate_columns:
    if df[col].dtype == object and df[col].nunique() > 1000:
        df.drop(columns=[col], inplace=True)

# --- Convert numeric-looking strings to float ---
for col in df.columns:
    if col != TARGET_COLUMN:
        df[col] = pd.to_numeric(df[col], errors='ignore')

# --- Split features and target ---
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# --- Detect 3D vs 5D ---
num_features = X.shape[1]
if num_features == 3:
    print("Dataset is 3D (3 features).")
elif num_features == 5:
    print("Dataset is 5D (5 features).")
else:
    print(f"Dataset has {num_features} features (neither 3D nor 5D).")

# --- Identify numeric and categorical features ---
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

# --- Detect Star vs Snowflake pattern ---
pattern_type = "Unknown pattern"
high_cardinality_cats = [c for c in categorical_features if X[c].nunique() > 50]

if len(numeric_features) > 1:
    corr_matrix = X[numeric_features].corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = any(upper_triangle.stack() > 0.9)
else:
    high_corr = False

if high_cardinality_cats and not high_corr:
    pattern_type = "Star-like pattern"
elif high_corr:
    pattern_type = "Snowflake-like pattern"

print(f"Detected data pattern: {pattern_type}")

# --- Preprocessing pipelines ---
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

# --- Full pipeline ---
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model ---
model_pipeline.fit(X_train, y_train)

# --- Save model ---
joblib.dump(model_pipeline, MODEL_PATH)
print(f"Model trained and saved successfully at '{MODEL_PATH}'")
