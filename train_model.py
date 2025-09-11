import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
import logging

# =======================
# Setup Logging
# =======================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# =======================
# Paths
# =======================
DATA_PATH = "data/incoming/mydata.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_importances.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

logging.info("🚀 Starting advanced training pipeline...")

# =======================
# Load Data
# =======================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
logging.info(f"✅ Loaded dataset with shape {df.shape}")

# Assume last column is target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

logging.info(f"🔢 Numeric features: {numeric_features}")
logging.info(f"🔤 Categorical features: {categorical_features}")

# =======================
# Preprocessing Pipelines
# =======================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())   # Robust to outliers
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)

# =======================
# Model Pipeline
# =======================
rf_clf = RandomForestClassifier(random_state=42, n_jobs=-1)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("feature_selection", SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42), threshold="median")),
    ("classifier", rf_clf)
])

# =======================
# Hyperparameter Tuning
# =======================
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring="f1_macro"
)

# =======================
# Train/Test Split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =======================
# Train Model
# =======================
logging.info("⚡ Training model with GridSearchCV...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
logging.info(f"✅ Best Params: {grid_search.best_params_}")

# =======================
# Evaluation
# =======================
y_pred = best_model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=False)
logging.info("📊 Classification Report:\n" + classification_report(y_test, y_pred))

# =======================
# Save Model
# =======================
joblib.dump(best_model, MODEL_PATH)
logging.info(f"💾 Model saved at {MODEL_PATH}")

# =======================
# Feature Importances
# =======================
# Extract feature names post-preprocessing
feature_names_num = numeric_features
feature_names_cat = best_model.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"].get_feature_names_out(categorical_features)
feature_names = np.concatenate([feature_names_num, feature_names_cat])

# Extract feature importances from RF inside feature_selection
clf = best_model.named_steps["classifier"]
importances = clf.feature_importances_

feat_imp = pd.DataFrame({
    "feature": feature_names[:len(importances)],  # align sizes
    "importance": importances
}).sort_values(by="importance", ascending=False)

feat_imp.to_csv(FEATURES_PATH, index=False)
logging.info(f"📈 Feature importances saved at {FEATURES_PATH}")
