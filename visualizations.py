# visualizations.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Directory where plots will be stored
VIZ_DIR = "data/visualization"
os.makedirs(VIZ_DIR, exist_ok=True)

def generate_visualizations(csv_path="data/incoming/mydata.csv"):
    """Generate basic visualizations and save them into data/visualizations/"""
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} not found, skipping visualizations.")
        return

    df = pd.read_csv(csv_path)

    # === 1️⃣ Histogram for numeric columns ===
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols[:3]:  # limit to first 3 to keep it fast
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Histogram of {col}")
        plt.savefig(os.path.join(VIZ_DIR, f"hist_{col}.png"))
        plt.close()

    # === 2️⃣ Count plot for categorical columns ===
    categorical_cols = df.select_dtypes(include="object").columns
    for col in categorical_cols[:2]:  # limit to first 2 for speed
        plt.figure()
        sns.countplot(y=col, data=df, order=df[col].value_counts().index)
        plt.title(f"Count Plot of {col}")
        plt.savefig(os.path.join(VIZ_DIR, f"count_{col}.png"))
        plt.close()

    # === 3️⃣ Correlation heatmap ===
    if len(numeric_cols) > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(VIZ_DIR, "heatmap.png"))
        plt.close()

    print(f"[INFO] Visualizations saved in {VIZ_DIR}")

if __name__ == "__main__":
    generate_visualizations()
