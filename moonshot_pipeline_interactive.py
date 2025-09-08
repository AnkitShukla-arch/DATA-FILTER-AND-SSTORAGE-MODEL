# moonshot_pipeline_interactive.py
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import utilities and visualization functions
from utils import get_latest_csv, load_csv, save_dataframe, ensure_directories
from visualizations import plot_correlation_heatmap, plot_target_distribution, plot_feature_importance

# ===== CONFIG =====
INPUT_CSV = "data/incoming/mydata.csv"       # Raw CSV
FILTERED_CSV = "data/filtered_data.csv"      # Filtered CSV
VIS_DIR = "data/visualizations"              # Directory for plots
MODEL_FILE = "models/random_forest.pkl"      # Saved RandomForest model
TARGET_COL = "target"                         # Target column to predict/filter

safe_makedirs("data")          # parent data folder
safe_makedirs("data/filtered") # filtered data folder
safe_makedirs(VIS_DIR)         # visualizations folder
safe_makedirs("models")        # models folder


# ===== ENSURE DIRECTORIES =====
ensure_directories([VIS_DIR, "models", "data/incoming"])

# ===== 1️⃣ LOAD CSV =====
print("[INFO] Loading CSV...")
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"{INPUT_CSV} not found. Please add your dataset.")

df = pd.read_csv(INPUT_CSV)

# Placeholder target if missing
if TARGET_COL not in df.columns:
    df[TARGET_COL] = np.random.randint(0, 2, size=len(df))

# ===== 2️⃣ PREPARE FEATURES =====
# Convert categorical variables using one-hot encoding
X = pd.get_dummies(df.drop(TARGET_COL, axis=1))
y = df[TARGET_COL]

# ===== 3️⃣ TRAIN OR LOAD RANDOM FOREST =====
if not os.path.exists(MODEL_FILE):
    print("[INFO] Training RandomForest model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print("[INFO] Classification Report:\n", classification_report(y_test, y_pred))

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(rf, f)
else:
    print("[INFO] Loading existing RandomForest model...")
    with open(MODEL_FILE, "rb") as f:
        rf = pickle.load(f)

# ===== 4️⃣ FILTER DATA =====
print("[INFO] Filtering data...")
df["predicted_target"] = rf.predict(X)
filtered_df = df[df["predicted_target"] == 1]  # Keep relevant rows
save_dataframe(filtered_df, FILTERED_CSV)
print(f"[INFO] Filtered CSV saved to {FILTERED_CSV}")

# ===== 5️⃣ GENERATE VISUALIZATIONS =====
print("[INFO] Generating visualizations...")
# Correlation heatmap
plot_correlation_heatmap(filtered_df, save_path=VIS_DIR)

# Class distribution plot
plot_target_distribution(filtered_df, TARGET_COL, save_path=VIS_DIR)

# Feature importance
plot_feature_importance(rf, X.columns, save_path=VIS_DIR)

print(f"[INFO] Visualizations saved in {VIS_DIR}")
print("[INFO] Pipeline completed successfully!")
