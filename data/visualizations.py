import os
import matplotlib.pyplot as plt
import seaborn as sns

def correlation_heatmap(df, output_dir="data/visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(path)
    plt.close()
    return path

def class_distribution(df, target, output_dir="data/visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target, data=df)
    plt.title("Class Distribution")
    path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(path)
    plt.close()
    return path

def feature_importance(model, feature_names, output_dir="data/visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    importances = model.feature_importances_
    indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=[importances[i] for i in indices],
                y=[feature_names[i] for i in indices])
    plt.title("Feature Importance")
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path)
    plt.close()
    return path

