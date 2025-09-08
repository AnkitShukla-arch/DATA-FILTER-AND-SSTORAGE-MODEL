# filter.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import safe_makedirs

# === Setup required folders ===
safe_makedirs("data")
safe_makedirs("data/filtered")
safe_makedirs("data/visualizations")
safe_makedirs("models")

INPUT_CSV = "data/incoming/mydata.csv"
FILTERED_CSV = "data/filtered_data.csv"
VIZ_DIR = "data/visualizations"

print("[INFO] Starting filter.py pipeline...")

# === 1️⃣ Load CSV or create dummy ===
if os.path.exists(INPUT_CSV):
    df = pd.read_csv(INPUT_CSV)
    print(f"[INFO] Loaded {INPUT_CSV} with shape {df.shape}")
else:
    print("[WARN] No incoming CSV found, creating dummy dataset...")
    df = pd.DataFrame({
        "feature1": np.random.randn(50),
        "feature2": np.random.randint(0, 100, 50),
        "target": np.random.choice([0, 1], 50)
    })

# Ensure target column exists
if "target" not in df.columns:
    print("[WARN] Target column missing. Creating dummy target...")
    df["target"] = np.random.choice([0, 1], len(df))

# === 2️⃣ Filter step (basic logic: keep target==1) ===
filtered_df = df[df["target"] == 1].copy()
if filtered_df.empty:
    print("[WARN] Filtered data is empty. Keeping 5 random rows instead.")
    filtered_df = df.sample(min(5, len(df)))

filtered_df.to_csv(FILTERED_CSV, index=False)
print(f"[INFO] Filtered data saved → {FILTERED_CSV} ({filtered_df.shape[0]} rows)")

# === 3️⃣ Schema placeholders ===
def generate_star_schema(df):
    fact = df.copy()
    dimensions = {col: df[[col]].drop_duplicates() for col in df.columns if col != "target"}
    return fact, dimensions

def generate_snowflake_schema(df):
    fact, dimensions = generate_star_schema(df)
    return fact, dimensions

fact_star, dims_star = generate_star_schema(filtered_df)
fact_snow, dims_snow = generate_snowflake_schema(filtered_df)
print("[INFO] Schema placeholders generated.")

# === 4️⃣ Visualizations ===
def save_plot(fig, name):
    path = os.path.join(VIZ_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

# Target distribution
fig, ax = plt.subplots()
sns.countplot(x="target", data=df, ax=ax)
ax.set_title("Target Distribution")
save_plot(fig, "target_distribution.png")

# Numeric histograms
for col in df.select_dtypes(include=np.number).columns:
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")
    save_plot(fig, f"{col}_hist.png")

print(f"[INFO] Visualizations saved → {VIZ_DIR}")

print("[INFO] filter.py completed successfully ✅")
