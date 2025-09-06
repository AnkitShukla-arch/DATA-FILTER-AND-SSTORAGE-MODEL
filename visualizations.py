# visualizations.py  (CI-friendly)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def _save_fig(save_path, fig_name):
    os.makedirs(save_path, exist_ok=True)
    full = os.path.join(save_path, fig_name)
    plt.savefig(full, bbox_inches="tight")
    plt.close()

def plot_target_distribution(y, save_path=None, name="target_dist.png"):
    if y is None:
        print("[WARN] no y")
        return
    fig = plt.figure(figsize=(6,4))
    sns.countplot(x=y)
    plt.title("Target Distribution")
    if save_path:
        _save_fig(save_path, name)
    else:
        plt.show()
        plt.close()

def plot_feature_importance(model, feature_names, save_path=None, name="feature_imp.png", top_n=20):
    if model is None:
        print("[WARN] no model")
        return
    import numpy as np
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        print("[WARN] model has no feature_importances_")
        return
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n)
    fig = plt.figure(figsize=(10,6))
    sns.barplot(x="importance", y="feature", data=df)
    plt.title("Feature Importance")
    if save_path:
        _save_fig(save_path, name)
    else:
        plt.show()
        plt.close()

def plot_correlation_heatmap(df, save_path=None, name="corr_heatmap.png"):
    if df is None or df.empty:
        print("[WARN] empty df")
        return
    fig = plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    if save_path:
        _save_fig(save_path, name)
    else:
        plt.show()
        plt.close()
