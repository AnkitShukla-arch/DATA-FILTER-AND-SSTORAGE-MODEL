import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_target_distribution(y, title="Target Distribution"):
    """
    Plots the distribution of the target variable.
    """
    if y is None:
        print("[WARNING] Target data is None. Cannot plot distribution.")
        return

    plt.figure(figsize=(6,4))
    sns.countplot(x=y)
    plt.title(title)
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plots top N feature importances from a tree-based model.
    """
    if model is None:
        print("[WARNING] Model is None. Cannot plot feature importances.")
        return

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).head(top_n)

    plt.figure(figsize=(10,6))
    sns.barplot(x="importance", y="feature", data=feature_importance_df)
    plt.title(f"Top {top_n} Feature Importances")
    plt.show()


def plot_correlation_heatmap(df):
    """
    Plots a correlation heatmap of the dataframe.
    """
    if df is None or df.empty:
        print("[WARNING] Dataframe is empty. Cannot plot heatmap.")
        return

    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()


if __name__ == "__main__":
    # Optional demo/test code
    print("[INFO] visualization.py loaded successfully")
