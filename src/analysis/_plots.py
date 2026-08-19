"""
_plots.py

Matplotlib / seaborn plotting functions replicating key figures from
the R analysis notebooks.

Each function saves a figure to a file and returns the figure object.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --- Common style ---
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150


def plot_predicted_vs_observed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Predicted vs. Observed",
    subtitle: str = None,
    xlabel: str = "Predicted Score (Out-of-Sample)",
    ylabel: str = "Observed Score",
) -> plt.Figure:
    """
    Scatterplot of predicted vs. observed values with a 1:1 reference line
    and a linear fit. Replicates the R ggplot predicted-vs-observed plots.
    """
    # compute R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    axis_min = min(np.min(y_true), np.min(y_pred))
    axis_max = max(np.max(y_true), np.max(y_pred))

    fig, ax = plt.subplots(figsize=(6, 6))

    # 1:1 reference line
    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        linestyle="--",
        color="gray50",
        linewidth=1,
    )

    # scatter
    ax.scatter(y_pred, y_true, alpha=0.6, color="#440154", s=30, edgecolors="none")

    # linear fit
    z = np.polyfit(y_pred, y_true, 1)
    p = np.poly1d(z)
    x_line = np.linspace(axis_min, axis_max, 200)
    ax.plot(x_line, p(x_line), color="#FDE725", linewidth=2)

    # R² annotation
    ax.text(
        axis_min + (axis_max - axis_min) * 0.05,
        axis_max,
        f"R² = {r2:.3f}",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    full_title = title
    if subtitle:
        full_title += f"\n{subtitle}"
    ax.set_title(full_title, fontsize=13, fontweight="bold")

    sns.despine(ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")
    return fig


def plot_feature_importance(
    coef_df: pd.DataFrame,
    output_path: Path,
    title: str = "Top Predictors",
    subtitle: str = "Standardized Coefficients from Elastic Net Model",
    n_top: int = 15,
) -> plt.Figure:
    """
    Horizontal bar chart of the top N non-zero coefficients.

    Replicates the R feature-importance plots with color coding for
    positive (protective) vs negative (risk) coefficients.
    """
    # filter non-zero and take top N by absolute value
    plot_df = (
        coef_df[coef_df["estimate"] != 0]
        .copy()
        .assign(abs_est=lambda df: df["estimate"].abs())
        .nlargest(n_top, "abs_est")
        .sort_values("abs_est", ascending=True)
    )

    colors = plot_df["estimate"].apply(lambda x: "#21908C" if x > 0 else "#FDE725")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(
        plot_df["term"],
        plot_df["estimate"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel("Standardized Coefficient", fontsize=11)
    ax.set_ylabel("")
    full_title = title
    if subtitle:
        full_title += f"\n{subtitle}"
    ax.set_title(full_title, fontsize=13, fontweight="bold")

    # legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#21908C", edgecolor="black", label="Protective"),
        Patch(facecolor="#FDE725", edgecolor="black", label="Risk Factor"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    sns.despine(ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")
    return fig


def plot_scree(
    pca,
    output_path: Path,
    title: str = "Scree Plot of Principal Components",
    n_components: int = 10,
) -> plt.Figure:
    """
    Scree plot showing eigenvalues (variance explained) per PC.

    Replicates the R scree plots with the Kaiser criterion line at y=1.
    """
    eigenvalues = pca.explained_variance_[:n_components]
    components = np.arange(1, len(eigenvalues) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(components, eigenvalues, color="gray60", linewidth=1.5, marker="o", markersize=6)
    ax.axhline(y=1, color="red", linestyle="--", alpha=0.6, linewidth=1)
    ax.set_xticks(components)
    ax.set_xlabel("Principal Component Number", fontsize=11)
    ax.set_ylabel("Eigenvalue (Variance Explained)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    sns.despine(ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")
    return fig


def plot_incremental_r2(
    r2_values: dict,
    output_path: Path,
    title: str = "Incremental Variance Explained in Internalizing Problems",
) -> plt.Figure:
    """
    Horizontal stacked bar chart showing incremental R² contributions.

    Replicates the LIWC incremental R² figure from the R notebook.
    """
    labels = list(r2_values.keys())
    values = list(r2_values.values())

    fig, ax = plt.subplots(figsize=(8, 2))
    left = 0
    colors = ["#440154", "#21908C", "#FDE725"]
    for i, (label, value) in enumerate(zip(labels, values)):
        ax.barh(
            "Model",
            value,
            left=left,
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.5,
            height=0.5,
        )
        # label in the middle of the segment
        ax.text(
            left + value / 2,
            0,
            f"{value * 100:.1f}%",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )
        left += value

    ax.set_xlim(0, left + 0.02)
    ax.set_xlabel(r"Cross-validated $R^2$", fontsize=11)
    ax.set_yticks([])
    ax.set_title(title, fontsize=13, fontweight="bold")

    # total annotation
    ax.text(
        left + 0.005,
        0,
        f"Total R² = {left * 100:.1f}%",
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")
    return fig


def plot_umap_scatter(
    umap_df: pd.DataFrame,
    output_path: Path,
    title: str = "Semantic Map of Linguistic Styles",
    subtitle: str = "UMAP projection revealing distinct clusters",
) -> plt.Figure:
    """
    2D UMAP scatter plot colored by cluster assignment.

    Replicates the RoBERTa UMAP figure from the embeddings notebook.
    """
    fig, ax = plt.subplots(figsize=(6.1, 5))

    cluster_colors = {
        0: "gray70",
        1: "#440154",
        2: "#21908C",
    }

    for cluster_id in sorted(umap_df["Cluster"].unique()):
        subset = umap_df[umap_df["Cluster"] == cluster_id]
        color = cluster_colors.get(cluster_id, "black")
        label = f"Cluster {cluster_id}"
        ax.scatter(
            subset["UMAP1"],
            subset["UMAP2"],
            c=color,
            alpha=0.8,
            s=40,
            edgecolors="none",
            label=label,
        )

    ax.set_xlabel("UMAP Dimension 1", fontsize=11)
    ax.set_ylabel("UMAP Dimension 2", fontsize=11)
    full_title = title
    if subtitle:
        full_title += f"\n{subtitle}"
    ax.set_title(full_title, fontsize=13, fontweight="bold")
    ax.legend(title="Cluster", loc="best")

    sns.despine(ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")
    return fig


def plot_cluster_violin(
    df: pd.DataFrame,
    outcome_col: str,
    cluster_col: str,
    output_path: Path,
    title: str = "Internalizing Problems Differ by Linguistic Style",
    subtitle: str = "YSR scores for the two primary semantic clusters",
) -> plt.Figure:
    """
    Violin + boxplot + jitter for cluster-wise outcome comparison.

    Replicates the RoBERTa cluster violin figure.
    """
    plot_df = df[df[cluster_col] != 0].copy()

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.violinplot(
        data=plot_df,
        x=cluster_col,
        y=outcome_col,
        inner=None,
        palette={1: "#440154", 2: "#21908C"},
        alpha=0.4,
        ax=ax,
    )

    sns.boxplot(
        data=plot_df,
        x=cluster_col,
        y=outcome_col,
        width=0.15,
        showcaps=True,
        showfliers=False,
        boxprops={"alpha": 0.8},
        ax=ax,
    )

    # jitter points
    sns.stripplot(
        data=plot_df,
        x=cluster_col,
        y=outcome_col,
        color="black",
        alpha=0.3,
        size=4,
        jitter=True,
        ax=ax,
    )

    ax.set_xlabel("Linguistic Style Cluster", fontsize=11)
    ax.set_ylabel("YSR Internalizing Score", fontsize=11)
    full_title = title
    if subtitle:
        full_title += f"\n{subtitle}"
    ax.set_title(full_title, fontsize=13, fontweight="bold")

    sns.despine(ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")
    return fig
