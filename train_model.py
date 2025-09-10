import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from utils import safe_makedirs

CSV_PATH = "data/incoming/mydata.csv"
MODEL_PATH = "models/random_forest.pkl"

# Load CSV
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
except pd.errors.EmptyDataError:
    raise ValueError(f"CSV at {CSV_PATH} is empty")

# Automatically pick last column as target
TARGET_COLUMN = df.columns[-1]
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Separate numeric & categorical
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Pipelines
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)
print("Model training completed.")

# Ensure models folder exists
safe_makedirs("models")

# Save model
joblib.dump(pipeline, MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")
