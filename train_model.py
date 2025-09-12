#!/usr/bin/env python3
"""
Robust train_model.py
- auto-detects target (or reads from config.yml)
- converts common null tokens to NaN
- fills dummy values for features and target
- builds pipeline and trains RandomForest
- saves pipeline to models/random_forest.pkl
- prints diagnostics to help CI logs
"""
import os
import sys
import joblib
import traceback
import numpy as np
import pandas as pd

# sklearn imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Optional config
import yaml

# -----------------------
# CONFIG / PATHS
# -----------------------
DEFAULT_DATA_PATH = "data/incoming/mydata.csv"   # change if you put csv elsewhere
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
VIZ_DIR = "data/visuals"
CONFIG_PATH = "config.yml"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

def load_config(path=CONFIG_PATH):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            print("[WARN] Could not parse config.yml, ignoring.")
    return {}

def detect_target_column(df, provided=None):
    # If provided and exists, use it
    if provided:
        if provided in df.columns:
            return provided
        else:
            print(f"[WARN] Provided target '{provided}' not in columns.")
    # common names
    common = ["target", "label", "class", "outcome", "status"]
    for c in common:
        if c in df.columns:
            print(f"[INFO] Auto-detected target column by common name: {c}")
            return c
    # choose binary-looking column (exactly 2 unique non-null values)
    for col in df.columns:
        if df[col].nunique(dropna=True) == 2:
            print(f"[INFO] Auto-detected target by binary-uniques: {col}")
            return col
    # fallback: last column
    print(f"[WARN] Falling back to last column as target: {df.columns[-1]}")
    return df.columns[-1]

def main():
    print("[INFO] Starting train_model.py...")

    config = load_config()
    data_path = config.get("data_path", DEFAULT_DATA_PATH)
    provided_target = config.get("target_col", None)

    if not os.path.exists(data_path):
        print(f"[ERROR] Data file not found at {data_path}")
        sys.exit(1)

    # read CSV and treat common null-strings as NaN
    na_values = ["null", "NULL", "NaN", "nan", "", " "]
    df = pd.read_csv(data_path, na_values=na_values, keep_default_na=True)
    print(f"[INFO] Loaded dataset {data_path} shape={df.shape}")

    # If CSV has weird duplicate column names or leading/trailing spaces - clean them
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # detect target column
    TARGET_COLUMN = detect_target_column(df, provided_target)
    if TARGET_COLUMN not in df.columns:
        print(f"[ERROR] Detected target '{TARGET_COLUMN}' not found in columns: {df.columns.tolist()}")
        sys.exit(1)
    print(f"[INFO] Using target column: {TARGET_COLUMN!r}")

    # --- Fill target missing values with dummy ---
    y = df[TARGET_COLUMN]
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        missing_before = y.isna().sum()
        if missing_before > 0:
            print(f"[INFO] Filling {missing_before} missing values in categorical target with 'Unknown'")
        y = y.fillna("Unknown")
    else:
        missing_before = y.isna().sum()
        if missing_before > 0:
            print(f"[INFO] Filling {missing_before} missing values in numeric target with -1")
        y = y.fillna(-1)

    # assign cleaned target back to dataframe (so we keep consistent X later)
    df[TARGET_COLUMN] = y

    # --- Build X and fill feature missing values robustly ---
    X = df.drop(columns=[TARGET_COLUMN])
    # convert obvious numeric-like columns that are strings? try to coerce numeric columns where possible
    # (this is optional and safe)
    for col in X.columns:
        # a quick attempt to convert numeric-looking strings to numeric
        if X[col].dtype == object:
            # attempt conversion but don't force errors
            coerced = pd.to_numeric(X[col], errors="coerce")
            num_nonnull = coerced.notna().sum()
            if num_nonnull > len(X) * 0.5:  # if most values convert, keep as numeric
                X[col] = coerced

    # Now fill missing values: numeric -> median, categorical -> "Unknown"
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            if X[col].isna().any():
                med = X[col].median()
                # if median is nan (all null), set to -1
                if pd.isna(med):
                    med = -1
                X[col] = X[col].fillna(med)
        else:
            # categorical or object
            X[col] = X[col].fillna("Unknown")

    # Quick diagnostics: ensure no NaNs remain in X or y
    total_X_nans = int(X.isna().sum().sum())
    total_y_nans = int(pd.isna(y).sum())
    print(f"[DEBUG] NaNs after feature filling -> X: {total_X_nans}, y: {total_y_nans}")
    if total_y_nans > 0:
        print("[WARN] There are still NaNs in target after filling. Filling with -1 for safety.")
        if y.dtype == object:
            y = y.fillna("Unknown")
        else:
            y = y.fillna(-1)
        df[TARGET_COLUMN] = y
        total_y_nans = int(pd.isna(y).sum())

    # after this step, there should be no NaNs left
    print(f"[INFO] Final NaNs (X, y): ({int(X.isna().sum().sum())}, {int(pd.isna(y).sum())})")

    # --- Identify feature types for preprocessing ---
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"[INFO] Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"[INFO] Categorical features ({len(categorical_features)}): {categorical_features}")

    # --- Build preprocessing pipeline (imputers present but X has been pre-filled as extra safety) ---
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # handle OneHotEncoder param compatibility
    ohe_kwargs = {"handle_unknown": "ignore"}
    try:
        import sklearn
        ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
        if ver >= (1,2):
            ohe_kwargs["sparse_output"] = False
        else:
            ohe_kwargs["sparse"] = False
    except Exception:
        ohe_kwargs["sparse_output"] = False

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(**ohe_kwargs))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )

    # --- Final pipeline ---
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    # --- Train/test split ---
    stratify_arg = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )

    # Final check BEFORE fitting
    print(f"[INFO] Pre-fit check: X_train shape={X_train.shape}, y_train shape={y_train.shape}")
    print(f"[INFO] Any NaNs in X_train (raw)? {int(X_train.isna().sum().sum())}")
    print(f"[INFO] Any NaNs in y_train? {int(pd.isna(y_train).sum())}")

    try:
        # Fit pipeline (preprocessor will run on X_train)
        pipeline.fit(X_train, y_train)

        # after fit, verify processed array has no NaNs (just in case)
        try:
            X_train_trans = preprocessor.transform(X_train)
            # handle sparse matrix check
            from scipy.sparse import issparse
            if issparse(X_train_trans):
                X_arr = X_train_trans.todense()
            else:
                X_arr = np.asarray(X_train_trans)
            nans_after_trans = int(np.isnan(X_arr).sum())
            print(f"[DEBUG] NaNs after preprocessor transform: {nans_after_trans}")
        except Exception as e_inner:
            print(f"[WARN] Could not compute NaNs after transform: {e_inner}")

        # Evaluate
        y_pred = pipeline.predict(X_test)
        print("[INFO] Classification Report:")
        print(classification_report(y_test, y_pred))

        # Save pipeline
        joblib.dump(pipeline, MODEL_PATH)
        print(f"[INFO] Saved trained pipeline to {MODEL_PATH}")

    except Exception as e:
        print("[ERROR] Training failed with exception:")
        traceback.print_exc()
        # dump debug artifact for inspection (optionally)
        debug_path = os.path.join(MODEL_DIR, "train_error_debug.pkl")
        try:
            joblib.dump({"X_head": X.head(20), "y_head": y.head(20)}, debug_path)
            print(f"[INFO] Wrote debug sample to {debug_path}")
        except Exception:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()
