# train_model.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils import safe_makedirs

# === Setup required folders ===
safe_makedirs("data")
safe_makedirs("data/filtered")
safe_makedirs("models")

FILTERED_CSV = "data/filtered_data.csv"
MODEL_PATH = "models/random_forest.pkl"

print("[INFO] Starting train_model.py...")

# === 1️⃣ Load filtered data or create dummy ===
if os.path.exists(FILTERED_CSV):
    df = pd.read_csv(FILTERED_CSV)
    print(f"[INFO] Loaded {FILTERED_CSV} with shape {df.shape}")
else:
    print("[WARN] Filtered CSV not found, creating dummy dataset...")
    df = pd.DataFrame({
        "feature1": np.random.randn(50),
        "feature2": np.random.randint(0, 100, 50),
        "target": np.random.choice([0, 1], 50)
    })
    df.to_csv(FILTERED_CSV, index=False)
    print(f"[INFO] Dummy filtered CSV created → {FILTERED_CSV}")

# Ensure target column exists
if "target" not in df.columns:
    print("[WARN] Target column missing. Creating dummy target...")
    df["target"] = np.random.choice([0, 1], len(df))

# === 2️⃣ Prepare features and labels ===
X = df.drop("target", axis=1).select_dtypes(include=np.number)
y = df["target"]

if X.empty:
    print("[WARN] No numeric features found, creating dummy feature.")
    df["dummy_feature"] = np.random.randn(len(df))
    X = df[["dummy_feature"]]

# === 3️⃣ Train RandomForest ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
modes.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("[INFO] Classification Report:")
print(classification_report(y_test, y_pred))

# === 4️⃣ Save trained model ===
joblib.dump(model, MODEL_PATH)
print(f"[INFO] Model saved → {MODEL_PATH}")

print("[INFO] train_model.py completed successfully ✅")
