import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Paths
incoming_path = "data/incoming/*.csv"
curated_path = "data/curated/filtered_data.csv"
model_path = "random_forest.pkl"
viz_dir = "data/visualizations/"

os.makedirs("data/curated", exist_ok=True)
os.makedirs(viz_dir, exist_ok=True)

def load_data():
    files = glob.glob(incoming_path)
    if not files:
        print("❌ No CSV found in data/incoming/")
        sys.exit(1)
    print(f"✅ Found input file: {files[0]}")
    return pd.read_csv(files[0])

def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    print(f"📊 Raw data shape: {df.shape}")
    df = df.dropna()
    print(f"✅ After dropna: {df.shape}")
    return df

def save_filtered(df: pd.DataFrame):
    df.to_csv(curated_path, index=False)
    print(f"✅ Filtered data saved → {curated_path}")

def train_and_validate(df: pd.DataFrame):
    if "target" not in df.columns:
        print("⚠️ No 'target' column found → skipping model training")
        return None

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Model trained → Accuracy: {acc:.2f}")

    print("\nClassification Report:\n", classification_report(y_test, preds))

    joblib.dump(model, model_path)
    print(f"✅ Model saved → {model_path}")

    return model, X, y

def create_visualizations(df: pd.DataFrame, model=None, X=None, y=None):
    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(viz_dir, "correlation_heatmap.png"))
    plt.close()
    print("📊 Saved correlation heatmap")

    # Class distribution
    if "target" in df.columns:
        plt.figure(figsize=(6, 4))
        df["target"].value_counts().plot(kind="bar", color="skyblue")
        plt.title("Target Class Distribution")
        plt.savefig(os.path.join(viz_dir, "class_distribution.png"))
        plt.close()
        print("📊 Saved class distribution plot")

    # Feature importance
    if model and X is not None:
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

        plt.figure(figsize=(8, 5))
        feat_imp.plot(kind="bar")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "feature_importance.png"))
        plt.close()
        print("📊 Saved feature importance plot")

def prepare_schema_hooks(df: pd.DataFrame):
    # Placeholder for star/snowflake schema prep
    print("🛠️ Preparing schema transformations (stub)")
    schema_info = {
        "columns": list(df.columns),
        "row_count": df.shape[0],
        "schema_type": "star/snowflake (to be implemented)"
    }
    pd.DataFrame([schema_info]).to_json("data/curated/schema_metadata.json", orient="records")
    print("✅ Schema metadata saved → data/curated/schema_metadata.json")

def main():
    try:
        df = load_data()
        df = filter_data(df)
        save_filtered(df)

        model_info = train_and_validate(df)
        if model_info:
            model, X, y = model_info
            create_visualizations(df, model, X, y)
        else:
            create_visualizations(df)

        prepare_schema_hooks(df)

        print("🚀 Moonshot pipeline completed successfully")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
