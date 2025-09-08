# train_model.py
"""
Moonshot RandomForest training script.

Produces:
- models/random_forest_pipeline.pkl  (Pipeline: preprocessor + classifier)
- models/training_report.json        (metrics, params)
- models/feature_importances.csv
- data/visualizations/ (confusion matrix, ROC (if binary), feature importance plot)

Usage:
    python train_model.py --input data/curated/filtered_output.csv --target target
"""

import os
import json
import argparse
import joblib
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    RocCurveDisplay,
)

# ---------------------------
# Helpers
# ---------------------------

from utils import safe_makedirs

# make sure model directory exists
safe_makedirs("models")

# when saving model
joblib.dump(models, "models/random_forest.pkl")

def ensure_dirs():
    safe_makedirs("models")
    safe_makedirs("data/visualizations")
    safe_makedirs("data/curated")

def get_latest_csv(input_dir="data/incoming"):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    csvs = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".csv")]
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    csvs.sort(key=os.path.getmtime)
    return csvs[-1]

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def plot_and_save_confusion(y_true, y_pred, outpath):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_feature_importances(names, importances, outpath):
    df = pd.DataFrame({"feature": names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(40)  # top 40
    plt.figure(figsize=(10, max(4, 0.25 * len(df))))
    sns.barplot(x="importance", y="feature", data=df)
    plt.title("Feature Importances (top features)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return df

# ---------------------------
# Main training function
# ---------------------------
def main(args):
    ensure_dirs()

    # 1) Resolve input file
    input_path = args.input
    if input_path is None:
        input_path = get_latest_csv("data/incoming")
        print(f"[INFO] No --input provided. Using latest file: {input_path}")
    else:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Provided input file not found: {input_path}")
        print(f"[INFO] Using input file: {input_path}")

    # 2) Load data
    df = pd.read_csv(input_path)
    print(f"[INFO] Loaded data shape: {df.shape}")

    # 3) Check target
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in input data. Columns: {list(df.columns)}")

    # 4) Basic cleaning: keep rows with at least one non-null? For safety do row-dropna if requested
    if args.dropna:
        df = df.dropna().reset_index(drop=True)
        print(f"[INFO] After dropna: {df.shape}")

    # 5) Features / target
    y = df[args.target]
    X = df.drop(columns=[args.target])

    # 6) Label-encode y if non-numeric
    y_is_numeric = pd.api.types.is_numeric_dtype(y)
    label_encoder = None
    if not y_is_numeric:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print("[INFO] Encoded non-numeric target with LabelEncoder.")

    # 7) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=(y if args.stratify else None)
    )
    print(f"[INFO] Train/Test splits: {X_train.shape} / {X_test.shape}")

    # 8) Build preprocessing pipelines
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    print(f"[INFO] Numeric cols: {len(numeric_cols)}, Categorical cols: {len(categorical_cols)}")

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ], remainder="drop", sparse_threshold=0)

    # 9) Build pipeline
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight="balanced" if args.class_weight_balanced else None
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])

    # 10) Optionally run randomized search
    if args.tune:
        print("[INFO] Running RandomizedSearchCV for quick hyperparameter tuning...")
        param_dist = {
            "clf__n_estimators": [100, 200, 400],
            "clf__max_depth": [None, 10, 20, 40],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2", None]
        }
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=min(20, args.n_iter),
            scoring=args.scoring,
            cv=min(5, max(2, args.cv)),
            random_state=args.random_state,
            n_jobs=-1,
            verbose=1
        )
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        best_params = search.best_params_
        print(f"[INFO] Best params: {best_params}")
    else:
        print("[INFO] Training pipeline without hyperparameter search...")
        pipeline.fit(X_train, y_train)
        best_pipeline = pipeline
        best_params = {}

    # 11) Evaluate
    y_pred = best_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"[RESULT] Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # 12) Save trained pipeline
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_out = f"models/random_forest_pipeline_{timestamp}.pkl"
    joblib.dump(best_pipeline, model_out)
    # also symlink/generic name
    joblib.dump(best_pipeline, "models/random_forest_pipeline.pkl")
    print(f"[INFO] Saved pipeline to {model_out} and models/random_forest_pipeline.pkl")

    # 13) Save metrics & params
    report_obj = {
        "timestamp": timestamp,
        "input_file": os.path.abspath(input_path),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "accuracy": float(acc),
        "classification_report": report,
        "best_params": best_params,
    }
    save_json_path = f"models/training_report_{timestamp}.json"
    save_json(save_json_path, report_obj)
    save_json("models/training_report.json", report_obj)
    print(f"[INFO] Saved training report to {save_json_path}")

    # 14) Feature importances mapping (get feature names)
    try:
        preproc = best_pipeline.named_steps["preprocessor"]
        clf_step = best_pipeline.named_steps["clf"]
        # get feature names after transformation
        feature_names = preproc.get_feature_names_out()
        importances = clf_step.feature_importances_
        feat_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        feat_df = feat_df.sort_values("importance", ascending=False)
        feat_csv = f"models/feature_importances_{timestamp}.csv"
        feat_df.to_csv(feat_csv, index=False)
        feat_df.to_csv("models/feature_importances.csv", index=False)
        print(f"[INFO] Wrote feature importances to {feat_csv}")
        # Also plot top features
        plot_feature_importances(feature_names, importances, f"data/visualizations/feature_importance_{timestamp}.png")
    except Exception as e:
        print(f"[WARN] Could not extract feature importances/feature names: {e}")

    # 15) Confusion matrix plot
    plot_and_save_confusion(y_test, y_pred, f"data/visualizations/confusion_matrix_{timestamp}.png")

    # 16) If binary classification, plot ROC curve
    n_classes = len(np.unique(y))
    if n_classes == 2:
        try:
            y_score = best_pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(6,5))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            plt.plot([0,1],[0,1],"--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.tight_layout()
            roc_path = f"data/visualizations/roc_curve_{timestamp}.png"
            plt.savefig(roc_path)
            plt.close()
            print(f"[INFO] Saved ROC curve to {roc_path}")
            report_obj["roc_auc"] = float(roc_auc)
            save_json("models/training_report.json", report_obj)
        except Exception as e:
            print(f"[WARN] ROC plot failed: {e}")

    print("[DONE] Training script finished successfully.")

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RandomForest pipeline (moonshot).")
    parser.add_argument("--input", type=str, default=None, help="Input CSV path (default: latest in data/incoming/)")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--tune", action="store_true", help="Run randomized hyperparameter search")
    parser.add_argument("--n_iter", type=int, default=20, help="n_iter for RandomizedSearchCV")
    parser.add_argument("--cv", type=int, default=5, help="CV folds for RandomizedSearchCV")
    parser.add_argument("--scoring", type=str, default="accuracy", help="Scoring for RandomizedSearchCV")
    parser.add_argument("--dropna", action="store_true", help="Drop NA rows before training")
    parser.add_argument("--stratify", action="store_true", help="Use stratified split if possible")
    parser.add_argument("--class_weight_balanced", action="store_true", help="Use class_weight='balanced' for RF")
    args = parser.parse_args()
    main(args)
