import pandas as pd

df = pd.read_csv("data/filtered_data.csv")

# Example split (same as your train_model.py)
from sklearn.model_selection import train_test_split

X = df.drop(columns=["target_column"])  # replace with your target col
y = df["target_column"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("[DEBUG] Missing values in X_train:")
print(X_train.isnull().sum())

print("[DEBUG] Missing values in X_test:")
print(X_test.isnull().sum())

