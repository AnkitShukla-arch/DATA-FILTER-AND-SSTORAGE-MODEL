# moonshot_pipeline_interactive.py

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.figure_factory as ff

# ----------------------------
# Feature Extraction for Columns
# ----------------------------
def extract_column_features(df):
    features = []
    for col in df.columns:
        col_data = df[col]
        features.append({
            'column': col,
            'unique_ratio': col_data.nunique() / len(col_data),
            'missing_ratio': col_data.isna().mean(),
            'is_numeric': int(pd.api.types.is_numeric_dtype(col_data)),
            'mean': col_data.mean() if pd.api.types.is_numeric_dtype(col_data) else 0,
            'std': col_data.std() if pd.api.types.is_numeric_dtype(col_data) else 0,
            'skew': col_data.skew() if pd.api.types.is_numeric_dtype(col_data) else 0
        })
    return pd.DataFrame(features)

# ----------------------------
# Train RandomForest Model
# ----------------------------
def train_filter_model(df, labels, save_path='column_filter_model.pkl'):
    features_df = extract_column_features(df)
    features_df['keep'] = labels

    X = features_df[['unique_ratio', 'missing_ratio', 'is_numeric', 'mean', 'std', 'skew']]
    y = features_df['keep']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    print("Train Accuracy:", clf.score(X_train, y_train))
    print("Test Accuracy:", clf.score(X_test, y_test))

    joblib.dump(clf, save_path)
    print(f"Model saved to {save_path}")
    return clf

# ----------------------------
# Filter Columns Using RandomForest
# ----------------------------
def filter_columns_with_rf(df, clf):
    features_df = extract_column_features(df)
    X_new = features_df[['unique_ratio', 'missing_ratio', 'is_numeric', 'mean', 'std', 'skew']]
    predictions = clf.predict(X_new)
    filtered_columns = features_df['column'][predictions == 1].tolist()
    return df[filtered_columns]

# ----------------------------
# Detect Schema Type
# ----------------------------
def detect_schema_type(df):
    num_columns = len(df.columns)
    if num_columns <= 5:
        return '3D'
    elif num_columns <= 10:
        return '5D'
    elif num_columns <= 20:
        return 'Star'
    else:
        return 'Snowflake'

# ----------------------------
# Generate Schema
# ----------------------------
def generate_schema(df, schema_type):
    schema = {'type': schema_type, 'fact_table': None, 'dimension_tables': []}
    if schema_type in ['3D', '5D']:
        schema['fact_table'] = df.iloc[:, :1]
        schema['dimension_tables'] = [df.iloc[:, 1:]]
    elif schema_type == 'Star':
        schema['fact_table'] = df.iloc[:, :1]
        schema['dimension_tables'] = [df.iloc[:, 1:]]
    elif schema_type == 'Snowflake':
        schema['fact_table'] = df.iloc[:, :1]
        schema['dimension_tables'] = [df.iloc[:, i:i+5] for i in range(1, len(df.columns), 5)]
    return schema

# ----------------------------
# Interactive Visualization
# ----------------------------
def visualize_interactive(filtered_df, schema_type):
    # Column Data Type Distribution
    col_types = filtered_df.dtypes.value_counts().reset_index()
    col_types.columns = ['dtype', 'count']
    fig1 = px.bar(col_types, x='dtype', y='count', color='dtype', title='Column Data Types in Filtered Dataset')
    fig1.show()

    # Missing Values Heatmap
    z = filtered_df.isna().astype(int).values
    fig2 = ff.create_annotated_heatmap(
        z=z,
        x=list(filtered_df.columns),
        y=list(range(len(filtered_df))),
        colorscale='Viridis',
        showscale=True
    )
    fig2.update_layout(title_text='Missing Values Heatmap (Interactive)')
    fig2.show()

    # Basic Stats Boxplots for Numeric Columns
    numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        fig3 = px.box(filtered_df, y=numeric_cols, title='Boxplot of Numeric Columns (Filtered Dataset)')
        fig3.show()

    print(f"Detected Schema Type for Visualization: {schema_type}")

# ----------------------------
# Main Pipeline
# ----------------------------
if __name__ == "__main__":
    # Load CSV Dataset
    df = pd.read_csv('data/incoming/mydata.csv')  # replace with your CSV path

    # Labels for RandomForest training (1=keep, 0=drop)
    labels = [1 if col not in ['unnecessary_col1','unnecessary_col2'] else 0 for col in df.columns]

    # Train RandomForest Model
    clf = train_filter_model(df, labels)

    # Filter Columns
    filtered_df = filter_columns_with_rf(df, clf)
    print("Filtered Columns:", filtered_df.columns.tolist())

    # Detect Schema Type
    schema_type = detect_schema_type(filtered_df)
    print("Detected Schema Type:", schema_type)

    # Generate Schema
    schema = generate_schema(filtered_df, schema_type)
    print("Schema Generated:", schema_type)

    # Save Filtered Data
    filtered_df.to_csv('data/filtered_data.csv', index=False)
    print("Filtered data saved to data/filtered_data.csv")

    # Interactive Visualization
    visualize_interactive(filtered_df, schema_type)

    print("Moonshot pipeline complete with interactive visualization ✅")
