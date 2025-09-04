# moonshot_pipeline_interactive.py
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.express as px

# ===== CONFIG =====
INPUT_CSV = "data/raw_data.csv"              # Your raw CSV
FILTERED_CSV = "data/filtered_data.csv"      # Where filtered data will go
VIS_DIR = "data/visualizations"              # Directory for Plotly visuals
MODEL_FILE = "models/random_forest.pkl"     # Saved trained model
TARGET_COL = "target"                        # Column to predict/filter

os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ===== 1️⃣ Load CSV =====
print("[INFO] Loading CSV...")
df = pd.read_csv(INPUT_CSV)

# Simple placeholder if TARGET_COL missing
if TARGET_COL not in df.columns:
    df[TARGET_COL] = np.random.randint(0, 2, size=len(df))

# ===== 2️⃣ Train RandomForest (if model not exists) =====
if not os.path.exists(MODEL_FILE):
    print("[INFO] Training RandomForest...")
    X = df.drop(TARGET_COL, axis=1).select_dtypes(include=np.number)  # numeric features
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print("[INFO] Classification Report:\n", classification_report(y_test, y_pred))

    # Save model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(rf, f)
else:
    print("[INFO] Loading existing RandomForest model...")
    with open(MODEL_FILE, "rb") as f:
        rf = pickle.load(f)

# ===== 3️⃣ Filter Data =====
print("[INFO] Filtering data...")
X_all = df.drop(TARGET_COL, axis=1).select_dtypes(include=np.number)
df["predicted_target"] = rf.predict(X_all)
filtered_df = df[df["predicted_target"] == 1]  # Keep only relevant rows

filtered_df.to_csv(FILTERED_CSV, index=False)
print(f"[INFO] Filtered CSV saved to {FILTERED_CSV}")

# ===== 4️⃣ Generate placeholder schemas =====
print("[INFO] Generating schema placeholders...")

def generate_star_schema(df):
    # Placeholder: create a fact table + dimension tables
    fact = df.copy()
    dimensions = {col: df[[col]].drop_duplicates() for col in df.columns if col != TARGET_COL}
    return fact, dimensions

def generate_snowflake_schema(df):
    # Placeholder: similar to star but normalized dimensions
    fact, dimensions = generate_star_schema(df)
    # Here you could normalize dimension tables further
    return fact, dimensions

fact_star, dims_star = generate_star_schema(filtered_df)
fact_snow, dims_snow = generate_snowflake_schema(filtered_df)

# ===== 5️⃣ Plotly Visualizations =====
print("[INFO] Creating visualizations...")
numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
for col in numeric_cols:
    fig = px.histogram(filtered_df, x=col, title=f"Distribution of {col}")
    fig.write_html(os.path.join(VIS_DIR, f"{col}_hist.html"))

print(f"[INFO] Visualizations saved in {VIS_DIR}")
print("[INFO] Pipeline completed successfully!")

