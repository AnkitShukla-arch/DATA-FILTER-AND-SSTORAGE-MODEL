# train_model.py
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml
from utils import safe_makedirs

# Paths
DATA_PATH = "data/filtered/filtered_data.csv"
MODEL_PATH = "models/random_forest.pkl"
VIZ_DIR = "data/visualizations"
CONFIG_PATH = "config.yml"

safe_makedirs("models")
safe_makedirs(VIZ_DIR)

print("[INFO] Starting train_model.py...")

# Load config
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {}

config = load_config()
DATA_PATH = config.get("data_path", DATA_PATH)
TARGET_COLUMN= config.get("target_col", None)

# Load data
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"[ERROR] Missing dataset at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"[INFO] Loaded dataset with shape {df.shape}")

target_column = df.columns[-1]
print(f"[INFO] Using target column: {target_column}")

 
# === 2️⃣ Prepare features and target ===

# Auto-detect target column if not already set
if "target_column" not in locals() or target_column is None:
    target_column = df.columns[-1]  # fallback: use last column

TARGET_COLUMN = target_column
print(f"[INFO] Using target column: {TARGET_COLUMN}")

# Handle missing values in target column
if df[TARGET_COLUMN].dtype == "object" or df[TARGET_COLUMN].dtype.name == "category":
    df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna("Unknown")
else:
    df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(-1)

# Split features and target
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]



# Detect target col
def detect_target(df, provided):
    if provided and provided in df.columns:
        return provided
    for cand in ["target", "label", "class", "outcome", "status"]:
        if cand in df.columns:
            return cand
    return df.columns[-1]

target_col = detect_target(df, TARGET_COL)
if target_col not in df.columns:
    raise ValueError(f"[ERROR] Target column '{target_col}' not found.")

print(f"[INFO] Using target column: {target_col}")

X = df.drop(columns=[target_col])
y = df[target_col]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(exclude=["int64", "float64"]).columns

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train
model.fit(X_train, y_train)
print("[INFO] Training complete.")

# Eval
y_pred = model.predict(X_test)
print("[INFO] Classification Report:")
print(classification_report(y_test, y_pred))

# Save
joblib.dump(model, MODEL_PATH)
print(f"[INFO] Model saved at {MODEL_PATH}")

# Visualization
plt.figure(figsize=(6, 4))
pd.Series(y).value_counts().plot(kind="bar")
plt.title("Target Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
viz_path = os.path.join(VIZ_DIR, "target_distribution.png")
plt.savefig(viz_path)
plt.close()
print(f"[INFO] Visualization saved at {viz_path}")
