import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.tree import plot_tree

FIGURES_DIR = "reports/figures"
BAR_COLOR = "#4C72B0"
HIGHLIGHT_COLOR = "#DD8452"
MUTED_COLOR = "#B0B0B0"
MODEL_COLORS = {"rf": "#4C72B0", "xgb": "#DD8452"}


def _load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def _savefig(fig, filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def correlation_matrix(df, filename="correlation_matrix.png"):
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(corr)), max(8, 0.5 * len(corr))))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Correlation Matrix")
    return _savefig(fig, filename)


def class_balance(df, target="y", filename="class_balance.png"):
    counts = df[target].value_counts()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(counts.index.astype(str), counts.values, color=BAR_COLOR)
    ax.set_xlabel(target)
    ax.set_ylabel("Count")
    ax.set_title(f"Class Balance ({target})")
    return _savefig(fig, filename)


def subscription_rate_by_category(df, column, target="y", filename=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(df, x=column, y=target, color=BAR_COLOR, err_kws={"color": "#333333"}, ax=ax)
    ax.set_xlabel(column)
    ax.set_ylabel(f"Subscription Rate ({target})")
    ax.set_title(f"Subscription Rate by {column.title()}")
    ax.tick_params(axis="x", rotation=45)
    return _savefig(fig, filename or f"subscription_rate_by_{column}.png")


def feature_importance_plot(model_path, X, model_name, filename=None):
    model = _load_model(model_path)
    classifier = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
    color = MODEL_COLORS.get(model_name, BAR_COLOR)

    importances = pd.Series(classifier.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importances.index, importances.values, color=color)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"{model_name} Feature Importance")
    return _savefig(fig, filename or f"{model_name}_feature_importance.png")


def confusion_matrix_plot(model_path, X_test, y_test, model_name, filename=None):
    model = _load_model(model_path)

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"{model_name} Confusion Matrix")
    return _savefig(fig, filename or f"{model_name}_confusion_matrix.png")


def curve_comparison(model_paths, X_test, y_test, filename="model_comparison_curves.png"):
    """ROC and precision-recall curves for multiple models, side by side."""
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(12, 5))

    for name, path in model_paths.items():
        model = _load_model(path)
        curve_kwargs = {"color": MODEL_COLORS.get(name, BAR_COLOR)}
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, curve_kwargs=curve_kwargs, ax=roc_ax)
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, name=name, curve_kwargs=curve_kwargs, ax=pr_ax)

    roc_ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED_COLOR, linewidth=1)
    roc_ax.set_title("ROC Curve")
    pr_ax.set_title("Precision-Recall Curve")
    fig.suptitle(" vs ".join(model_paths.keys()))
    return _savefig(fig, filename)


def f1_comparison_bar(results_df, highlight=("rf", "xgb"), filename="f1_comparison.png"):
    """Bar chart of F1 score per model (as returned by predict.evaluate_models), with `highlight` models called out."""
    results_df = results_df.sort_values("f1_score", ascending=False)
    colors = [HIGHLIGHT_COLOR if m in highlight else MUTED_COLOR for m in results_df["model"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(results_df["model"], results_df["f1_score"], color=colors)
    ax.set_ylabel("F1 Score")
    ax.set_title("Model Comparison (F1 Score)")
    return _savefig(fig, filename)

